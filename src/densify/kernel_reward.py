"""Verifiable shaped reward for CUDA-kernel RFT (RLVR).

Bakes the robust-kbench-style signal into a single scalar:
    parse → compile → numerically-correct vs eager → speedup vs eager.
Dense shaping (partial credit for compile) so GRPO has gradient even from the
0/2 SFT floor. Used by the GRPO loop and the KernelBench eval.
"""
from __future__ import annotations

import os
import re
import time

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

import torch  # noqa: E402

_CODE_RE = re.compile(r"```(?:cpp|cuda|c\+\+)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


def compile_kernel(code: str, name: str):
    from torch.utils.cpp_extension import load_inline
    kw = dict(name=f"{name}_{int(time.time()*1000)%1000000}", cuda_sources=code,
              with_cuda=True, verbose=False)
    if "PYBIND11_MODULE" in code:           # model supplied its own binding
        kw["cpp_sources"] = ""
    else:
        kw["cpp_sources"] = "torch::Tensor forward(torch::Tensor input);"
        kw["functions"] = ["forward"]
    return load_inline(**kw)


class _Timeout(Exception):
    pass


def _timed(seconds):
    import signal

    class _ctx:
        def __enter__(self):
            signal.signal(signal.SIGALRM, self._h)
            signal.alarm(seconds)

        def __exit__(self, *a):
            signal.alarm(0)

        def _h(self, *a):
            raise _Timeout()
    return _ctx()


def evaluate_kernel(code: str, ref_fn, shape=(4096, 4096), name="k",
                    atol=1e-3, rtol=1e-3, timeout=60) -> dict:
    """Return dict(parsed, compiled, correct, speedup, max_diff, error). Guarded by SIGALRM timeout."""
    r = {"parsed": bool(code and "forward" in code), "compiled": False,
         "correct": False, "speedup": None, "max_diff": None, "error": None}
    if not r["parsed"]:
        return r
    try:
        with _timed(timeout):
            mod = compile_kernel(code, name)
        r["compiled"] = True
    except _Timeout:
        r["error"] = "compile timeout"
        return r
    except Exception as e:
        r["error"] = str(e)[-200:]
        return r
    try:
        with _timed(timeout):
            x = torch.randn(*shape, device="cuda")
            y = mod.forward(x)
            ref = ref_fn(x)
            r["max_diff"] = float((y - ref).abs().max())
            r["correct"] = bool(torch.allclose(y, ref, atol=atol, rtol=rtol))
    except _Timeout:
        r["error"] = "run timeout"
        return r
    except Exception as e:
        r["error"] = "run: " + str(e)[-160:]
        return r
    if r["correct"]:
        def t(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            s = time.time()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            return (time.time() - s) / 50
        r["speedup"] = t(lambda: ref_fn(x)) / max(t(lambda: mod.forward(x)), 1e-9)
    return r


def shaped_reward(r: dict, speedup_cap: float = 3.0) -> float:
    """Dense, verifiable reward in ~[-0.2, 1.0]."""
    rew = 0.0
    rew += 0.10 if r.get("parsed") else -0.10
    rew += 0.20 if r.get("compiled") else 0.0
    rew += 0.40 if r.get("correct") else 0.0
    if r.get("correct") and r.get("speedup"):
        rew += 0.30 * min(max(r["speedup"], 0.0), speedup_cap) / speedup_cap
    return rew


def reward_for_text(text: str, ref_fn, shape=(4096, 4096), name="k") -> tuple[float, dict]:
    r = evaluate_kernel(extract_code(text), ref_fn, shape=shape, name=name)
    return shaped_reward(r), r
