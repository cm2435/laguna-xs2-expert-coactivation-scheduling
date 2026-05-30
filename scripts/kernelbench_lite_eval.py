"""Tiny KernelBench-Lite eval: model generates a CUDA kernel for a PyTorch op,
we compile (load_inline) -> check correctness vs eager -> measure speedup.
Reports fast_0 (compiles+correct) and fast_1 (correct & >=1x faster)."""
import os, re, sys, time, argparse
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.cpp_extension import load_inline

SYS = ("You are an expert GPU kernel engineer. Convert PyTorch modules into correct, "
       "optimized CUDA kernels. Define `torch::Tensor forward(torch::Tensor input)`.")

# (name, pytorch source string, reference fn) — 20 unary elementwise ops
def _m(body):
    return f"class Model(nn.Module):\n    def forward(self, x):\n        return {body}"


TASKS = [
    ("ReLU", _m("torch.relu(x)"), torch.relu),
    ("Square", _m("x * x"), lambda x: x * x),
    ("Sigmoid", _m("torch.sigmoid(x)"), torch.sigmoid),
    ("Tanh", _m("torch.tanh(x)"), torch.tanh),
    ("GeLU", _m("torch.nn.functional.gelu(x)"), lambda x: torch.nn.functional.gelu(x)),
    ("Abs", _m("torch.abs(x)"), torch.abs),
    ("Negate", _m("-x"), lambda x: -x),
    ("Exp", _m("torch.exp(x)"), torch.exp),
    ("Softplus", _m("torch.nn.functional.softplus(x)"), torch.nn.functional.softplus),
    ("ELU", _m("torch.nn.functional.elu(x)"), torch.nn.functional.elu),
    ("LeakyReLU", _m("torch.nn.functional.leaky_relu(x, 0.01)"), lambda x: torch.nn.functional.leaky_relu(x, 0.01)),
    ("SiLU", _m("torch.nn.functional.silu(x)"), torch.nn.functional.silu),
    ("Sign", _m("torch.sign(x)"), torch.sign),
    ("Floor", _m("torch.floor(x)"), torch.floor),
    ("Ceil", _m("torch.ceil(x)"), torch.ceil),
    ("Round", _m("torch.round(x)"), torch.round),
    ("Cos", _m("torch.cos(x)"), torch.cos),
    ("Sin", _m("torch.sin(x)"), torch.sin),
    ("Mish", _m("torch.nn.functional.mish(x)"), torch.nn.functional.mish),
    ("HardSwish", _m("torch.nn.functional.hardswish(x)"), torch.nn.functional.hardswish),
]


def gen(model, tok, py, max_new=1024):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\n{py}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def extract_code(text):
    m = re.search(r"```(?:cpp|cuda|c\+\+)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from densify.kernel_reward import evaluate_kernel  # noqa: E402


def evaluate(name, code, ref_fn):
    r = evaluate_kernel(code, ref_fn, name=f"kb_{name.lower()}", timeout=50)
    r["task"] = name
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    ap.add_argument("--trace-dir", default="runs/eval/kernelbench_lite")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    import json as _json
    os.makedirs(a.trace_dir, exist_ok=True)
    trace = open(os.path.join(a.trace_dir, "traces.jsonl"), "w")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[loaded] {a.model}\n")
    K = a.k
    f0 = f1 = 0
    for name, py, ref in TASKS:
        best = {"compiled": False, "correct": False, "speedup": None}
        comp = 0
        for k in range(K):
            code = extract_code(gen(model, tok, py))
            r = evaluate(name, code, ref)
            comp += r["compiled"]
            # save every attempt for review
            with open(os.path.join(a.trace_dir, f"{name}_attempt{k}.cu"), "w") as fh:
                fh.write(code)
            trace.write(_json.dumps({"task": name, "attempt": k, "compiled": r["compiled"],
                                     "correct": r["correct"], "speedup": r.get("speedup"),
                                     "max_diff": r.get("max_diff"), "error": r.get("error"),
                                     "code": code}) + "\n"); trace.flush()
            if r["correct"] and (not best["correct"] or (r.get("speedup") or 0) > (best.get("speedup") or 0)):
                best = r
        f0 += best["correct"]
        f1 += bool(best["correct"] and best.get("speedup") and best["speedup"] >= 1.0)
        sp = f"{best['speedup']:.2f}x" if best.get("speedup") else "-"
        print(f"{name:8s} (best of {K}): compiled {comp}/{K} | correct={best['correct']} | speedup={sp}")
    n = len(TASKS)
    summary = {"model": a.model, "k": K, "tasks": n, "fast_0": f0, "fast_1": f1}
    _json.dump(summary, open(os.path.join(a.trace_dir, "summary.json"), "w"), indent=2)
    print(f"\nfast_0 (compile+correct, best-of-{K}): {f0}/{n}   fast_1 (correct & >=1x): {f1}/{n}")
    print(f"[traces saved] {a.trace_dir}/ (traces.jsonl, *.cu, summary.json)")


if __name__ == "__main__":
    main()
