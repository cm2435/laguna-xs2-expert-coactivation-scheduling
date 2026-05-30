"""Triton ablation — a PROPER Triton test: Triton system prompt + real Triton
execution. A/B baseline-Triton-prompt vs hinted-Triton-prompt; report pass@K."""
import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code, evaluate_triton

BASE_SYS = ("You are an expert GPU kernel engineer specializing in Triton. Write a correct, optimized "
            "Triton kernel using triton and triton.language as tl, plus a Python `forward(x)` that "
            "launches it and returns the output tensor.")
HINT_SYS = ("You are an expert Triton kernel engineer (triton 3.3). Write CORRECT Triton.\n"
            "Interface — define exactly:\n"
            "  @triton.jit\n  def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
            "      pid = tl.program_id(0); offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
            "      mask = offs < n\n      x = tl.load(x_ptr + offs, mask=mask)\n      # compute out\n"
            "      tl.store(out_ptr + offs, out, mask=mask)\n"
            "  def forward(x):\n      out = torch.empty_like(x); n = x.numel()\n"
            "      grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "      kernel[grid](x, out, n, BLOCK_SIZE=1024); return out\n"
            "Rules: use mask=mask on load/store; no C/C++ syntax (no braces, no semicolons); pure Python.")
OPS = [
    ("ReLU", "torch.relu(x)", torch.relu),
    ("Tanh", "torch.tanh(x)", torch.tanh),
    ("Sigmoid", "torch.sigmoid(x)", torch.sigmoid),
    ("Abs", "torch.abs(x)", torch.abs),
]
K = 4


def gen(model, tok, sysp, body):
    user = (f"Write a Triton kernel that applies this operation elementwise to a tensor x and returns "
            f"the result, with a `forward(x)` launcher:\n\n```python\ndef op(x):\n    return {body}\n```")
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=900, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    ck = "runs/sft/kernel_cuda_sft/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    print("[loaded]", flush=True)
    out = {}
    for label, sysp in [("TRITON-BASELINE", BASE_SYS), ("TRITON-HINTED", HINT_SYS)]:
        ran_ops = corr_ops = 0
        per = []
        for name, body, ref in OPS:
            ran = corr = 0
            errs = []
            for k in range(K):
                r = evaluate_triton(extract_code(gen(model, tok, sysp, body)), ref, timeout=35)
                ran += r["ran"]; corr += r["correct"]
                if r.get("error"):
                    errs.append(r["error"][:40])
            ran_ops += (ran > 0); corr_ops += (corr > 0)
            per.append({"op": name, "ran_k": ran, "correct_k": corr, "errs": errs[:2]})
            print(f"[{label}] {name:8} ran {ran}/{K} correct {corr}/{K} | {errs[:1]}", flush=True)
        out[label] = {"ran_ops": ran_ops, "correct_ops": corr_ops, "per": per}
        print(f"[{label}] ran {ran_ops}/{len(OPS)} | correct(pass@{K}) {corr_ops}/{len(OPS)}\n", flush=True)
    json.dump(out, open("runs/eval/triton_ablation.json", "w"), indent=2)
    print("="*50)
    print(f"TRITON-BASELINE correct pass@{K}: {out['TRITON-BASELINE']['correct_ops']}/{len(OPS)}")
    print(f"TRITON-HINTED   correct pass@{K}: {out['TRITON-HINTED']['correct_ops']}/{len(OPS)}")


if __name__ == "__main__":
    main()
