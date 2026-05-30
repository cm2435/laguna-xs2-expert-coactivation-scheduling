"""10 ops × {CUDA, Triton} pass@K eval, with per-attempt COMPILE-TIME monitoring.
Flags slow compiles, logs timings, uses the hinted prompts (best per DSL)."""
import os, sys, time, json
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code, evaluate_kernel, evaluate_triton

CUDA_SYS = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
            "- Use `input.scalar_type()` (NOT `input.type()`); dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
            "- Bounds guard `if (idx < size)`; output = `torch::empty_like(input)`.\n"
            "- Define `torch::Tensor forward(torch::Tensor input)` and end with a PYBIND11_MODULE binding.")
TRITON_SYS = ("You are an expert Triton kernel engineer (triton 3.3). Write CORRECT pure-Python Triton.\n"
              "Define `@triton.jit def kernel(x_ptr,out_ptr,n,BLOCK_SIZE: tl.constexpr)` using "
              "`offs=tl.program_id(0)*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE); mask=offs<n; "
              "tl.load(x_ptr+offs,mask=mask)` ... `tl.store(out_ptr+offs,out,mask=mask)`, and "
              "`def forward(x): out=torch.empty_like(x); n=x.numel(); "
              "kernel[(triton.cdiv(n,1024),)](x,out,n,BLOCK_SIZE=1024); return out`. No C syntax.")
OPS = [
    ("ReLU", "torch.relu(x)", torch.relu),
    ("Tanh", "torch.tanh(x)", torch.tanh),
    ("Sigmoid", "torch.sigmoid(x)", torch.sigmoid),
    ("GeLU", "torch.nn.functional.gelu(x)", lambda x: torch.nn.functional.gelu(x)),
    ("Abs", "torch.abs(x)", torch.abs),
    ("SiLU", "torch.nn.functional.silu(x)", torch.nn.functional.silu),
    ("Softplus", "torch.nn.functional.softplus(x)", torch.nn.functional.softplus),
    ("ELU", "torch.nn.functional.elu(x)", torch.nn.functional.elu),
    ("LeakyReLU", "torch.nn.functional.leaky_relu(x, 0.01)", lambda x: torch.nn.functional.leaky_relu(x, 0.01)),
    ("Mish", "torch.nn.functional.mish(x)", torch.nn.functional.mish),
]
K = 3
SLOW = 25  # seconds -> flag


def gen(model, tok, sysp, user, max_new=1024):
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    ck = "runs/sft/kernel_cuda_sft/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    print("[loaded]", flush=True)
    summary = {}
    for dsl in ["CUDA", "Triton"]:
        sysp = CUDA_SYS if dsl == "CUDA" else TRITON_SYS
        ok_ops = 0
        ctimes = []
        for name, body, ref in OPS:
            if dsl == "CUDA":
                user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {body}\n```"
            else:
                user = f"Write a Triton kernel for this op with a forward(x) launcher:\n\n```python\ndef op(x):\n    return {body}\n```"
            best = 0
            for k in range(K):
                code = extract_code(gen(model, tok, sysp, body if False else user))
                t0 = time.time()
                if dsl == "CUDA":
                    r = evaluate_kernel(code, ref, name=name.lower(), timeout=40)
                else:
                    r = evaluate_triton(code, ref, timeout=35)
                ct = time.time() - t0
                ctimes.append(ct)
                flag = "  <SLOW>" if ct > SLOW else ""
                best = max(best, int(r["correct"]))
                print(f"[{dsl}] {name:9} k{k} correct={r['correct']} eval_time={ct:.1f}s{flag} {str(r.get('error'))[:45] if r.get('error') else ''}", flush=True)
            ok_ops += best
        summary[dsl] = {"correct_passK": ok_ops, "n": len(OPS),
                        "avg_eval_s": round(sum(ctimes)/len(ctimes), 1),
                        "max_eval_s": round(max(ctimes), 1), "slow_count": sum(1 for c in ctimes if c > SLOW)}
        print(f"[{dsl}] correct pass@{K}: {ok_ops}/{len(OPS)} | avg eval {summary[dsl]['avg_eval_s']}s "
              f"max {summary[dsl]['max_eval_s']}s slow>{SLOW}s: {summary[dsl]['slow_count']}\n", flush=True)
    json.dump(summary, open("runs/eval/eval_10ops.json", "w"), indent=2)
    print("="*55)
    for dsl in ["CUDA", "Triton"]:
        s = summary[dsl]
        print(f"{dsl:8} pass@{K} {s['correct_passK']}/{s['n']} | eval avg {s['avg_eval_s']}s max {s['max_eval_s']}s slow {s['slow_count']}")


if __name__ == "__main__":
    main()
