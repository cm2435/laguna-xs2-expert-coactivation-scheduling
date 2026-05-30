"""3-way KernelBench-Lite (L1, 10 elementwise ops, K=4, subprocess-isolated):
SFT vs SFT-extended vs SFT-extended-RFT. Teacher numbers reused from earlier."""
import os, sys, json, subprocess, tempfile, hashlib, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

HERE = os.path.dirname(__file__)
SYS = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
       "- Use `input.scalar_type()` (NOT `input.type()`); dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
       "- Bounds guard `if (idx < size)`; output = `torch::empty_like(input)`.\n"
       "- For vectorization use `reinterpret_cast<float4*>(ptr)` (NOT `float4* v = float4* ptr;`).\n"
       "- Define `torch::Tensor forward(torch::Tensor input)` and end with a PYBIND11_MODULE binding.")
OPS = ["relu", "tanh", "sigmoid", "gelu", "abs", "silu", "softplus", "elu", "leakyrelu", "mish"]
BODY = {"relu": "torch.relu(x)", "tanh": "torch.tanh(x)", "sigmoid": "torch.sigmoid(x)",
        "gelu": "torch.nn.functional.gelu(x)", "abs": "torch.abs(x)", "silu": "torch.nn.functional.silu(x)",
        "softplus": "torch.nn.functional.softplus(x)", "elu": "torch.nn.functional.elu(x)",
        "leakyrelu": "torch.nn.functional.leaky_relu(x, 0.01)", "mish": "torch.nn.functional.mish(x)"}
K = 4
MODELS = [
    ("SFT", "runs/sft/kernel_cuda_sft/checkpoint-final"),
    ("SFT-ext", "runs/sft/kernel_cuda_sft_v2/checkpoint-final"),
    ("SFT-ext-RFT", "runs/rft/kernel_cuda_rft/checkpoint-final"),
]


def gen(model, tok, op):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {BODY[op]}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=1024, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return extract_code(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))


def iso(code, op, tag):
    name = f"{op}_{tag}_{hashlib.md5(code.encode()).hexdigest()[:6]}"
    with tempfile.TemporaryDirectory() as d:
        i, o = f"{d}/i.json", f"{d}/o.json"
        json.dump({"code": code, "dsl": "CUDA", "op": op, "name": name}, open(i, "w"))
        env = dict(os.environ, CUDA_HOME="/usr/local/cuda", PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
                   PYTHONPATH=os.path.join(HERE, "..", "src"), TORCH_CUDA_ARCH_LIST="9.0",
                   TORCH_EXTENSIONS_DIR=f"{d}/ext")
        try:
            subprocess.run([sys.executable, f"{HERE}/eval_worker.py", "--in", i, "--out", o], env=env, timeout=160, capture_output=True)
            return json.load(open(o)) if os.path.exists(o) else {"compiled": False, "correct": False}
        except subprocess.TimeoutExpired:
            return {"compiled": False, "correct": False, "error": "timeout"}


def main():
    summary, traces = {}, []
    for tag, path in MODELS:
        if not os.path.exists(os.path.join(path, "model.safetensors")):
            print(f"[skip {tag}] missing {path}", flush=True); continue
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
        comp = corr = 0; speeds = []
        for op in OPS:
            cc = co = 0; best_sp = 0
            for k in range(K):
                r = iso(gen(model, tok, op), op, f"{tag}{k}")
                cc = max(cc, int(r.get("compiled", False))); co = max(co, int(r.get("correct", False)))
                if r.get("correct") and r.get("speedup"): best_sp = max(best_sp, r["speedup"])
                traces.append({"model": tag, "op": op, "k": k, "compiled": r.get("compiled"), "correct": r.get("correct"), "speedup": r.get("speedup")})
            comp += cc; corr += co
            if best_sp: speeds.append(best_sp)
            print(f"[{tag}] {op:10} compile@{K}={cc} correct@{K}={co} best_sp={round(best_sp,2) if best_sp else '-'}", flush=True)
        summary[tag] = {"compile_passK": comp, "correct_passK": corr, "n": len(OPS), "K": K,
                        "mean_best_speedup": round(sum(speeds)/len(speeds), 3) if speeds else None}
        print(f"[{tag}] compile {comp}/{len(OPS)}  correct {corr}/{len(OPS)}  meanSpeedup {summary[tag]['mean_best_speedup']}\n", flush=True)
        del model; torch.cuda.empty_cache()
    summary["TEACHER (reused, k=1)"] = {"correct_passK": 4, "n": 6, "note": "head-to-head ReLU/Tanh/Abs/SiLU all <1x eager"}
    os.makedirs("runs/eval", exist_ok=True)
    json.dump({"summary": summary, "traces": traces}, open("runs/eval/eval_3way.json", "w"), indent=2)
    print("=" * 60)
    for tag, _ in MODELS:
        if tag in summary:
            s = summary[tag]; print(f"{tag:14} compile {s['compile_passK']}/{s['n']}  correct {s['correct_passK']}/{s['n']}  speedup {s['mean_best_speedup']}")


if __name__ == "__main__":
    main()
