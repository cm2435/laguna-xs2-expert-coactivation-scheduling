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

# (name, pytorch source string, reference fn) — unary elementwise ops
TASKS = [
    ("ReLU", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)", torch.relu),
    ("Square", "class Model(nn.Module):\n    def forward(self, x):\n        return x * x", lambda x: x * x),
]


def gen(model, tok, py):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\n{py}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def extract_code(text):
    m = re.search(r"```(?:cpp|cuda|c\+\+)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def evaluate(name, code, ref_fn):
    res = {"task": name, "compiled": False, "correct": False, "speedup": None}
    try:
        has_pybind = "PYBIND11_MODULE" in code
        kw = dict(name=f"kb_{name.lower()}_{int(time.time()*1000)%100000}",
                  cuda_sources=code, with_cuda=True, verbose=False)
        if has_pybind:
            # model supplied its own module binding -> don't let load_inline add a duplicate
            kw["cpp_sources"] = ""
        else:
            kw["cpp_sources"] = "torch::Tensor forward(torch::Tensor input);"
            kw["functions"] = ["forward"]
        mod = load_inline(**kw)
        res["compiled"] = True
    except Exception as e:
        res["error"] = str(e)[:160]
        return res
    x = torch.randn(4096, 4096, device="cuda")
    try:
        y = mod.forward(x); ref = ref_fn(x)
        res["correct"] = bool(torch.allclose(y, ref, atol=1e-3, rtol=1e-3))
        res["max_diff"] = float((y - ref).abs().max())
    except Exception as e:
        res["error"] = "run: " + str(e)[:140]; return res
    if res["correct"]:
        def t(fn):
            for _ in range(5): fn()
            torch.cuda.synchronize(); s = time.time()
            for _ in range(50): fn()
            torch.cuda.synchronize(); return (time.time() - s) / 50
        res["speedup"] = t(lambda: ref_fn(x)) / t(lambda: mod.forward(x))
    return res


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
