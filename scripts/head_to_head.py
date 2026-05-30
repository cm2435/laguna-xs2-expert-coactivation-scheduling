"""Side-by-side: teacher Laguna-XS.2 (33B/3B-active MoE) vs our dense SFT model,
on the same KernelBench-Lite CUDA questions. Measures decode tok/s + compile/correct,
sequentially (memory) with identical fair settings."""
import os, sys, time, json, gc, argparse
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code, evaluate_kernel

SYS = ("You are an expert GPU kernel engineer. Write correct, optimized CUDA kernels. "
       "Define `torch::Tensor forward(torch::Tensor input)`.")
OPS = [
    ("ReLU", "torch.relu(x)", torch.relu),
    ("Tanh", "torch.tanh(x)", torch.tanh),
    ("Sigmoid", "torch.sigmoid(x)", torch.sigmoid),
    ("GeLU", "torch.nn.functional.gelu(x)", lambda x: torch.nn.functional.gelu(x)),
    ("Abs", "torch.abs(x)", torch.abs),
    ("SiLU", "torch.nn.functional.silu(x)", torch.nn.functional.silu),
]
MAXNEW = 1024


def run(model_id, label, compile_test=True):
    print(f"\n{'#'*60}\n# {label}: {model_id}\n{'#'*60}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    load_t = time.time() - t0
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[load] {load_t:.0f}s | params {nparams/1e9:.1f}B | VRAM {torch.cuda.memory_allocated()/1e9:.0f}GB", flush=True)

    # warmup
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": "Write a CUDA kernel."}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    wids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        model.generate(wids, max_new_tokens=16, do_sample=False, pad_token_id=9)

    rows = []
    for name, body, ref in OPS:
        py = f"class Model(nn.Module):\n    def forward(self, x):\n        return {body}"
        user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\n{py}\n```"
        s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.6, top_k=20, pad_token_id=9)
        torch.cuda.synchronize(); dt = time.time() - t
        n = out.shape[1] - ids.shape[1]
        txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        row = {"op": name, "gen_tokens": int(n), "gen_sec": round(dt, 2), "tok_s": round(n / dt, 1)}
        if compile_test:
            r = evaluate_kernel(extract_code(txt), ref, name=name.lower(), timeout=50)
            row.update(compiled=r["compiled"], correct=r["correct"], speedup=r.get("speedup"))
        rows.append(row)
        print(json.dumps(row), flush=True)

    avg = sum(r["tok_s"] for r in rows) / len(rows)
    f0 = sum(1 for r in rows if r.get("correct"))
    print(f"[{label}] avg {avg:.1f} tok/s | fast_0 {f0}/{len(rows)} | load {load_t:.0f}s", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"label": label, "model": model_id, "params_b": round(nparams/1e9, 1),
            "avg_tok_s": round(avg, 1), "fast_0": f0, "load_s": round(load_t), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="runs/sft/kernel_cuda_sft/checkpoint-final")
    ap.add_argument("--teacher", default="poolside/Laguna-XS.2")
    ap.add_argument("--out", default="runs/eval/head_to_head.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {}
    res["ours"] = run(a.ours, "OURS (dense SFT)")
    res["teacher"] = run(a.teacher, "TEACHER Laguna-XS.2", compile_test=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n" + "=" * 60 + "\nSIDE-BY-SIDE\n" + "=" * 60)
    o, t = res["ours"], res["teacher"]
    print(f"{'':14}{'OURS':>16}{'TEACHER':>16}")
    print(f"{'params':14}{o['params_b']:>15}B{t['params_b']:>15}B")
    print(f"{'avg tok/s':14}{o['avg_tok_s']:>16}{t['avg_tok_s']:>16}")
    print(f"{'fast_0/'+str(len(OPS)):14}{o['fast_0']:>16}{t['fast_0']:>16}")
    print(f"{'load (s)':14}{o['load_s']:>16}{t['load_s']:>16}")
    print(f"\nper-op tok/s:")
    for ro, rt in zip(o["rows"], t["rows"]):
        print(f"  {ro['op']:10} ours {ro['tok_s']:>6} tok/s ({ro['gen_tokens']}tok)  |  teacher {rt['tok_s']:>6} tok/s ({rt['gen_tokens']}tok)")


if __name__ == "__main__":
    main()
