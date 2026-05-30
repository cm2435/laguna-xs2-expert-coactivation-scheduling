"""Offline GRPO/RLVR on the SakanaAI CUDA Engineer Archive TRACES.

No live compilation — the dataset already carries the verified reward per kernel
(`Correct` + `CUDA_Speedup_Native`). Group candidates by Task_ID, reward each from
the recorded trace, compute group-relative (Dr.GRPO) advantages, and advantage-weight
the log-prob of each kernel under the policy (KL-anchored to the SFT reference).

reward r = CUDA_Speedup_Native if Correct else 0.0   (verifiable, from the table)
advantage A_i = r_i - mean(r over the task group)     [Dr.GRPO: no std/length norm]
loss = mean_i [ -A_i * logp(kernel_i | prompt)/n_i ] + beta * KL(policy||ref)
"""
import argparse, json, sys, time, random
from pathlib import Path
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SYS = ("You are an expert GPU kernel engineer. Convert PyTorch modules into correct, "
       "optimized CUDA kernels. Define `torch::Tensor forward(torch::Tensor input)`.")


def build_prompt(tok, pytorch_module, dev):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\n{pytorch_module}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)


def seq_logprob(model, ids, prompt_len):
    out = model(ids)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    gen = tok_lp[prompt_len - 1:]
    return gen.sum(), gen.numel()


def load_groups(splits, group_size, max_tasks):
    """Return list of groups; each = (pytorch_module, [(cuda_code, reward), ...])."""
    bytask = defaultdict(list)
    for sp in splits:
        ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=sp, streaming=True)
        for r in ds:
            if not r["CUDA_Code"] or not r["PyTorch_Code_Module"]:
                continue
            rew = float(r["CUDA_Speedup_Native"] or 0.0) if r["Correct"] else 0.0
            bytask[(sp, r["Task_ID"])].append((r["PyTorch_Code_Module"], r["CUDA_Code"], rew))
            if len(bytask) >= max_tasks and all(len(v) >= group_size for v in bytask.values()):
                pass
        # stop early once we have enough tasks
        if len(bytask) >= max_tasks:
            break
    groups = []
    for k, cands in bytask.items():
        if len(cands) < 2:
            continue
        pm = cands[0][0]
        # pick a high-variance subset: top + bottom + random middles
        cands = sorted(cands, key=lambda c: c[2])
        picks = [cands[-1], cands[-2], cands[0], cands[1]]                      # 2 best, 2 worst
        mids = cands[2:-2]
        picks += random.sample(mids, min(group_size - 4, len(mids))) if mids else []
        groups.append((pm, [(c[1], c[2]) for c in picks[:group_size]]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--splits", default="level_1,level_2")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--max-tasks", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-beta", type=float, default=0.02)
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

    groups = load_groups(a.splits.split(","), a.group_size, a.max_tasks)
    print(f"[data] {len(groups)} task groups from Sakana traces (recorded rewards)", flush=True)
    start = time.time()
    for step in range(1, a.steps + 1):
        pm, cands = groups[(step - 1) % len(groups)]
        pids = build_prompt(tok, pm, dev); plen = pids.shape[1]
        rewards = torch.tensor([c[1] for c in cands])
        if rewards.std() < 1e-6:                                  # DAPO: skip no-variance groups
            row = {"step": step, "skipped": "no-variance", "mean_reward": float(rewards.mean())}
            metrics.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True); continue
        adv = (rewards - rewards.mean()).to(dev)                  # Dr.GRPO
        policy.train(); opt.zero_grad(set_to_none=True)
        terms = []
        for i, (cuda, _) in enumerate(cands):
            cids = tok("```cpp\n" + cuda + "\n```", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            ids = torch.cat([pids, cids], dim=1)[:, :a.max_len]
            lp, ntok = seq_logprob(policy, ids, plen)
            with torch.no_grad():
                rlp, _ = seq_logprob(refm, ids, plen)
            kl = (lp - rlp) / max(ntok, 1)
            terms.append(-(adv[i] * lp / max(ntok, 1)) + a.kl_beta * kl)
        loss = torch.stack(terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
        opt.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), "mean_reward": float(rewards.mean()),
               "max_reward": float(rewards.max()), "best_correct_speedup": float(rewards.max()),
               "elapsed": round(time.time() - start, 1)}
        metrics.write(json.dumps(row) + "\n"); metrics.flush()
        if step % 5 == 0 or step == 1:
            print(json.dumps(row), flush=True)
    policy.save_pretrained(a.output_dir / "checkpoint-final", safe_serialization=True)
    print(f"[done] saved {a.output_dir}/checkpoint-final", flush=True)


if __name__ == "__main__":
    main()
