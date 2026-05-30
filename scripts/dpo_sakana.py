"""DPO on the SakanaAI CUDA Engineer Archive traces (offline RFT).

Exploits the archive's evolutionary refinement trajectory: per task, PREFER a
correct+fast kernel over an incorrect/slow one, using the recorded verified labels
(`Correct`, `CUDA_Speedup_Native`). No live compilation.

DPO loss (Rafailov et al.):
  Δ = β[(logπ(chosen)−logπ_ref(chosen)) − (logπ(rejected)−logπ_ref(rejected))]
  L = −log σ(Δ)
Reference = the frozen SFT-extended model (implicit KL anchor). Train routed_dense + lm_head.
"""
import argparse, json, sys, time, random
from pathlib import Path
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SYS = ("You are an expert GPU kernel engineer. Convert PyTorch modules into correct, "
       "optimized CUDA kernels. Define `torch::Tensor forward(torch::Tensor input)`.")


def prompt_ids(tok, pm, dev):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\n{pm}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)


def comp_logp(model, prompt_ids, cuda, tok, dev, max_len):
    cids = tok("```cpp\n" + cuda + "\n```", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    ids = torch.cat([prompt_ids, cids], dim=1)[:, :max_len]
    plen = prompt_ids.shape[1]
    out = model(ids)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    return tok_lp[plen - 1:].sum()


def build_pairs(splits, max_tasks, pairs_per_task):
    bytask = defaultdict(list)
    for sp in splits:
        ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=sp, streaming=True)
        for r in ds:
            if not r["CUDA_Code"] or not r["PyTorch_Code_Module"]:
                continue
            sp_v = float(r["CUDA_Speedup_Native"] or 0.0)
            bytask[(sp, r["Task_ID"])].append((r["PyTorch_Code_Module"], r["CUDA_Code"], bool(r["Correct"]), sp_v))
        if len(bytask) >= max_tasks:
            break
    pairs = []
    for cands in bytask.values():
        pm = cands[0][0]
        correct = sorted([c for c in cands if c[2] and len(c[1]) < 6000], key=lambda c: -c[3])
        wrong = [c for c in cands if not c[2] and len(c[1]) < 6000]
        if not correct:
            continue
        chosen = correct[0]                                  # correct + fastest
        # rejected pool: incorrect kernels, else much-slower correct ones
        rej_pool = wrong if wrong else [c for c in correct[1:] if c[3] < chosen[3] * 0.5]
        random.shuffle(rej_pool)
        for rej in rej_pool[:pairs_per_task]:
            pairs.append((pm, chosen[1], rej[1]))
    random.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--splits", default="level_1,level_2")
    ap.add_argument("--max-tasks", type=int, default=200)
    ap.add_argument("--pairs-per-task", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--max-len", type=int, default=1536)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (a.output_dir / "metrics.jsonl").open("a")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    refm = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    refm.eval(); [p.requires_grad_(False) for p in refm.parameters()]
    for n, p in policy.named_parameters():
        p.requires_grad_(("routed_dense" in n) or ("lm_head" in n))
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=a.lr)
    dev = next(policy.parameters()).device

    pairs = build_pairs(a.splits.split(","), a.max_tasks, a.pairs_per_task)
    print(f"[data] {len(pairs)} preference pairs (correct+fast > incorrect/slow) from Sakana traces", flush=True)
    start = time.time(); wins = 0; seen = 0
    for step in range(1, a.steps + 1):
        pm, chosen, rejected = pairs[(step - 1) % len(pairs)]
        pids = prompt_ids(tok, pm, dev)
        policy.train(); opt.zero_grad(set_to_none=True)
        lc = comp_logp(policy, pids, chosen, tok, dev, a.max_len)
        lr = comp_logp(policy, pids, rejected, tok, dev, a.max_len)
        with torch.no_grad():
            rc = comp_logp(refm, pids, chosen, tok, dev, a.max_len)
            rr = comp_logp(refm, pids, rejected, tok, dev, a.max_len)
        delta = a.beta * ((lc - rc) - (lr - rr))
        loss = -F.logsigmoid(delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
        opt.step()
        seen += 1; wins += int(delta.item() > 0)
        if step % 10 == 0 or step == 1:
            row = {"step": step, "loss": round(float(loss.detach().cpu()), 4),
                   "margin": round(float(delta.detach().cpu()), 3),
                   "pref_acc": round(wins / seen, 3), "elapsed": round(time.time() - start, 1)}
            metrics.write(json.dumps(row) + "\n"); metrics.flush(); print(json.dumps(row), flush=True)
    policy.save_pretrained(a.output_dir / "checkpoint-final", safe_serialization=True)
    print(f"[done] saved {a.output_dir}/checkpoint-final  (pref_acc {wins/max(seen,1):.2f})", flush=True)


if __name__ == "__main__":
    main()
