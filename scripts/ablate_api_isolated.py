"""API-hint A/B on the ISOLATED harness: baseline vs API-hinted CUDA prompt,
pass@K, subprocess-isolated eval. Measures whether the prompt fix lifts fast_0."""
import os, sys, json, subprocess, tempfile, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

HERE = os.path.dirname(__file__)
BASE = ("You are an expert GPU kernel engineer. Write correct, optimized CUDA kernels. "
        "Define `torch::Tensor forward(torch::Tensor input)`.")
HINT = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
        "Rules:\n- Use `input.scalar_type()` (NOT the deprecated `input.type()`).\n"
        "- Dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
        "- Bounds guard: `if (idx < size) { ... }` (process valid indices only).\n"
        "- Allocate output with `torch::empty_like(input)`; define `torch::Tensor forward(torch::Tensor input)`.\n"
        "- End with `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }`.")
OPS = ["relu", "tanh", "sigmoid", "gelu", "abs"]
BODY = {"relu": "torch.relu(x)", "tanh": "torch.tanh(x)", "sigmoid": "torch.sigmoid(x)",
        "gelu": "torch.nn.functional.gelu(x)", "abs": "torch.abs(x)"}
K = 3


def gen(model, tok, sysp, op):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {BODY[op]}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=1024, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return extract_code(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))


def iso(code, op):
    with tempfile.TemporaryDirectory() as d:
        i, o = f"{d}/i.json", f"{d}/o.json"
        json.dump({"code": code, "dsl": "CUDA", "op": op}, open(i, "w"))
        env = dict(os.environ, CUDA_HOME="/usr/local/cuda", PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
                   PYTHONPATH=os.path.join(HERE, "..", "src"), TORCH_CUDA_ARCH_LIST="9.0")
        try:
            subprocess.run([sys.executable, f"{HERE}/eval_worker.py", "--in", i, "--out", o], env=env, timeout=140, capture_output=True)
            return json.load(open(o)) if os.path.exists(o) else {"compiled": False, "correct": False}
        except subprocess.TimeoutExpired:
            return {"compiled": False, "correct": False, "error": "timeout"}


def main():
    ck = "runs/sft/kernel_cuda_sft/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    print("[loaded]", flush=True)
    out = {}
    for label, sysp in [("BASELINE", BASE), ("API-HINTED", HINT)]:
        ok = 0; per = {}
        for op in OPS:
            c = 0
            for k in range(K):
                r = iso(gen(model, tok, sysp, op), op)
                c += int(r.get("correct", False))
            per[op] = c; ok += (c > 0)
            print(f"[{label}] {op:9} correct {c}/{K}", flush=True)
        out[label] = {"fast0": ok, "n": len(OPS), "per": per}
        print(f"[{label}] fast_0 pass@{K}: {ok}/{len(OPS)}\n", flush=True)
    json.dump(out, open("runs/eval/api_hint_isolated.json", "w"), indent=2)
    print("="*40)
    print(f"BASELINE   fast_0 pass@{K}: {out['BASELINE']['fast0']}/{len(OPS)}")
    print(f"API-HINTED fast_0 pass@{K}: {out['API-HINTED']['fast0']}/{len(OPS)}")


if __name__ == "__main__":
    main()
