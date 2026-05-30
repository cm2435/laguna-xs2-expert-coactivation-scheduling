"""Ablation: does putting the API/version in the prompt help? A/B two system
prompts (baseline vs API-hinted) on the same ops, K samples each, report fast_0."""
import os, sys, time, json
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code, evaluate_kernel

BASE_SYS = ("You are an expert GPU kernel engineer. Write correct, optimized CUDA kernels. "
            "Define `torch::Tensor forward(torch::Tensor input)`.")
HINT_SYS = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
            "Rules:\n"
            "- Use `input.scalar_type()` (NOT the deprecated `input.type()`).\n"
            "- Dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
            "- Bounds guard: `if (idx < size) { ... }` (process valid indices only).\n"
            "- Allocate output with `torch::empty_like(input)`; define `torch::Tensor forward(torch::Tensor input)`.\n"
            "- End with `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }`.")
OPS = [
    ("ReLU", "torch.relu(x)", torch.relu),
    ("Tanh", "torch.tanh(x)", torch.tanh),
    ("Sigmoid", "torch.sigmoid(x)", torch.sigmoid),
    ("GeLU", "torch.nn.functional.gelu(x)", lambda x: torch.nn.functional.gelu(x)),
    ("Abs", "torch.abs(x)", torch.abs),
]
K = 4
MAXNEW = 1024


def gen(model, tok, sysp, body):
    py = f"class Model(nn.Module):\n    def forward(self, x):\n        return {body}"
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\n{py}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    ck = "runs/sft/kernel_cuda_sft/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    print("[loaded]", flush=True)
    out = {}
    for label, sysp in [("BASELINE", BASE_SYS), ("API-HINTED", HINT_SYS)]:
        comp = corr = 0
        per = []
        for name, body, ref in OPS:
            c0 = c1 = 0
            for k in range(K):
                r = evaluate_kernel(extract_code(gen(model, tok, sysp, body)), ref, name=name.lower(), timeout=45)
                c0 += r["compiled"]; c1 += r["correct"]
            ok = c1 > 0
            comp += (c0 > 0); corr += ok
            per.append({"op": name, "compiled_k": c0, "correct_k": c1, "passK": ok})
            print(f"[{label}] {name:8} compiled {c0}/{K} correct {c1}/{K}", flush=True)
        out[label] = {"fast0_passK": corr, "any_compile": comp, "per": per}
        print(f"[{label}] fast_0 (pass@{K}): {corr}/{len(OPS)}\n", flush=True)
    json.dump(out, open("runs/eval/api_hint_ablation.json", "w"), indent=2)
    print("="*50); print(f"BASELINE   fast_0 pass@{K}: {out['BASELINE']['fast0_passK']}/{len(OPS)}")
    print(f"API-HINTED fast_0 pass@{K}: {out['API-HINTED']['fast0_passK']}/{len(OPS)}")


if __name__ == "__main__":
    main()
