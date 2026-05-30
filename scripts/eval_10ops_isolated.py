"""10 ops × {CUDA, Triton} pass@K, with SUBPROCESS-ISOLATED kernel evaluation
(faulty kernels can't crash the model process) + compile-time monitoring."""
import os, sys, time, json, subprocess, tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

HERE = os.path.dirname(__file__)
CUDA_SYS = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
            "- Use `input.scalar_type()` (NOT `input.type()`); dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
            "- Bounds guard `if (idx < size)`; output = `torch::empty_like(input)`.\n"
            "- Define `torch::Tensor forward(torch::Tensor input)` and end with a PYBIND11_MODULE binding.")
TRITON_SYS = ("You are an expert Triton kernel engineer (triton 3.3). Write CORRECT pure-Python Triton.\n"
              "Define `@triton.jit def kernel(x_ptr,out_ptr,n,BLOCK_SIZE: tl.constexpr)` using "
              "`offs=tl.program_id(0)*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE); mask=offs<n; "
              "tl.load(x_ptr+offs,mask=mask)` ... `tl.store(out_ptr+offs,out,mask=mask)`, and "
              "`def forward(x): out=torch.empty_like(x); n=x.numel(); "
              "kernel[(triton.cdiv(n,1024),)](x,out,n,BLOCK_SIZE=1024); return out`. No C/C++ syntax.")
OPS = ["relu", "tanh", "sigmoid", "gelu", "abs", "silu", "softplus", "elu", "leakyrelu", "mish"]
BODY = {"relu": "torch.relu(x)", "tanh": "torch.tanh(x)", "sigmoid": "torch.sigmoid(x)",
        "gelu": "torch.nn.functional.gelu(x)", "abs": "torch.abs(x)", "silu": "torch.nn.functional.silu(x)",
        "softplus": "torch.nn.functional.softplus(x)", "elu": "torch.nn.functional.elu(x)",
        "leakyrelu": "torch.nn.functional.leaky_relu(x, 0.01)", "mish": "torch.nn.functional.mish(x)"}
K = 3
SLOW = 25


def gen(model, tok, sysp, user, max_new=1024):
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def isolated_eval(code, dsl, op):
    """Run the kernel in a fresh subprocess; on crash/timeout return failure."""
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        json.dump({"code": code, "dsl": dsl, "op": op}, open(inp, "w"))
        env = dict(os.environ, CUDA_HOME="/usr/local/cuda",
                   PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
                   PYTHONPATH=os.path.join(HERE, "..", "src"),
                   TORCH_CUDA_ARCH_LIST="9.0")
        t0 = time.time()
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "eval_worker.py"), "--in", inp, "--out", outp],
                           env=env, timeout=140, capture_output=True)
            dt = time.time() - t0
            if os.path.exists(outp):
                r = json.load(open(outp)); r["eval_s"] = dt; return r
            return {"compiled": False, "correct": False, "error": "worker crashed (isolated)", "eval_s": dt}
        except subprocess.TimeoutExpired:
            return {"compiled": False, "correct": False, "error": "subprocess timeout", "eval_s": time.time() - t0}


def main():
    ck = "runs/sft/kernel_cuda_sft/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    print("[loaded — kernels run in isolated subprocesses]", flush=True)
    summary, traces = {}, []
    for dsl in ["CUDA", "Triton"]:
        sysp = CUDA_SYS if dsl == "CUDA" else TRITON_SYS
        ok = 0; ctimes = []
        for op in OPS:
            if dsl == "CUDA":
                user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {BODY[op]}\n```"
            else:
                user = f"Write a Triton kernel for this op with a forward(x) launcher:\n\n```python\ndef op(x):\n    return {BODY[op]}\n```"
            best = 0
            for k in range(K):
                code = extract_code(gen(model, tok, sysp, user))
                r = isolated_eval(code, dsl, op)
                ctimes.append(r.get("eval_s", 0))
                best = max(best, int(r.get("correct", False)))
                traces.append({"dsl": dsl, "op": op, "k": k, **{x: r.get(x) for x in ("compiled", "correct", "speedup", "eval_s", "error")}})
                flag = "  <SLOW>" if r.get("eval_s", 0) > SLOW else ""
                print(f"[{dsl}] {op:10} k{k} compiled={r.get('compiled')} correct={r.get('correct')} {r.get('eval_s',0):.0f}s{flag} {str(r.get('error'))[:42] if r.get('error') else ''}", flush=True)
            ok += best
        summary[dsl] = {"correct_passK": ok, "n": len(OPS), "K": K,
                        "avg_eval_s": round(sum(ctimes)/len(ctimes), 1), "max_eval_s": round(max(ctimes), 1),
                        "slow_count": sum(1 for c in ctimes if c > SLOW)}
        print(f"[{dsl}] correct pass@{K}: {ok}/{len(OPS)} | eval avg {summary[dsl]['avg_eval_s']}s max {summary[dsl]['max_eval_s']}s slow>{SLOW}s {summary[dsl]['slow_count']}\n", flush=True)
    os.makedirs("runs/eval", exist_ok=True)
    json.dump({"summary": summary, "traces": traces}, open("runs/eval/eval_10ops.json", "w"), indent=2)
    print("="*55)
    for dsl in ["CUDA", "Triton"]:
        s = summary[dsl]
        print(f"{dsl:8} pass@{K}: {s['correct_passK']}/{s['n']} correct | eval avg {s['avg_eval_s']}s max {s['max_eval_s']}s slow {s['slow_count']}")


if __name__ == "__main__":
    main()
