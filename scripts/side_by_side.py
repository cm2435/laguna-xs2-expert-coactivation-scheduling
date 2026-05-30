"""Generate + SAVE every kernel for each question, both models, side by side.
Outputs one .md per question with OURS vs TEACHER code + saves raw .cu files."""
import os, sys, time, json, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

SYS = ("You are an expert GPU kernel engineer. Write correct, optimized CUDA kernels. "
       "Define `torch::Tensor forward(torch::Tensor input)`.")
OPS = [("ReLU", "torch.relu(x)"), ("Tanh", "torch.tanh(x)"), ("Sigmoid", "torch.sigmoid(x)"),
       ("GeLU", "torch.nn.functional.gelu(x)"), ("Abs", "torch.abs(x)"), ("SiLU", "torch.nn.functional.silu(x)")]
OUT = "runs/eval/side_by_side"


def run(model_id, label):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    res = {}
    for name, body in OPS:
        user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {body}\n```"
        s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=1024, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
        dt = time.time() - t0
        code = extract_code(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
        res[name] = {"code": code, "gen_s": round(dt, 1), "tokens": int(out.shape[1] - ids.shape[1])}
        with open(f"{OUT}/{name}_{label}.cu", "w") as f:
            f.write(code)
        print(f"[{label}] {name}: {res[name]['tokens']} tok, {dt:.1f}s -> saved", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    ours = run("runs/sft/kernel_cuda_sft/checkpoint-final", "OURS")
    teacher = run("poolside/Laguna-XS.2", "TEACHER")
    # write side-by-side markdown per question
    for name, _ in OPS:
        o, t = ours[name], teacher[name]
        with open(f"{OUT}/{name}.md", "w") as f:
            f.write(f"# {name} — side by side\n\n")
            f.write(f"## OURS (3.0B dense) — {o['tokens']} tok, {o['gen_s']}s\n```cpp\n{o['code']}\n```\n\n")
            f.write(f"## TEACHER (33.4B) — {t['tokens']} tok, {t['gen_s']}s\n```cpp\n{t['code']}\n```\n")
    json.dump({"ours": ours, "teacher": teacher}, open(f"{OUT}/all.json", "w"), indent=2)
    print(f"\n[done] saved per-question .md + .cu files + all.json in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
