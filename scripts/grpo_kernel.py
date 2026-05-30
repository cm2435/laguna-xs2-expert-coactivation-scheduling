"""GRPO/RLVR for CUDA-kernel generation (Dr.GRPO + DAPO dynamic sampling).

Reward = verifiable compile→correct→speedup (densify.kernel_reward). Per prompt,
sample G kernels, reward each, advantage = r − mean(r)  [Dr.GRPO: no std/length
normalization], skip groups with zero reward variance [DAPO dynamic sampling],
policy-gradient on routed_dense (+lm_head) with KL anchor to the SFT reference.

Trainable: routed_dense + lm_head. No teacher; SFT model is also the KL ref.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from densify.kernel_reward import reward_for_text_isolated  # noqa: E402

SYS = ("You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.\n"
       "- Use `input.scalar_type()` (NOT `input.type()`); dispatch with AT_DISPATCH_FLOATING_TYPES.\n"
       "- Bounds guard `if (idx < size)`; output = `torch::empty_like(input)`.\n"
       "- For vectorization use `reinterpret_cast<float4*>(ptr)` (NOT `float4* v = float4* ptr;`).\n"
       "- Define `torch::Tensor forward(torch::Tensor input)` and end with a PYBIND11_MODULE binding.")
# Verifiable elementwise tasks (op name must be a key in eval_worker.REFS) — reward
# is numerical: generated kernel's forward(x) vs PyTorch eager, isolated subprocess.
TASKS = [
    ("relu", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)"),
    ("sigmoid", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.sigmoid(x)"),
    ("tanh", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.tanh(x)"),
    ("gelu", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.nn.functional.gelu(x)"),
    ("silu", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.nn.functional.silu(x)"),
    ("softplus", "class Model(nn.Module):\n    def forward(self, x):\n        return torch.nn.functional.softplus(x)"),
]


def prompt_ids(tok, py, dev):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\n{py}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)


def seq_logprob(model, ids, prompt_len):
    """Sum log p(generated tokens) under model for one sequence [1, L]."""
    out = model(ids)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]   # [L-1]
    gen = tok_lp[prompt_len - 1:]                                # generated region
    return gen.sum(), gen.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.9)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (a.output_dir / "metrics.jsonl").open("a")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    ref = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    ref.eval(); [p.requires_grad_(False) for p in ref.parameters()]
    policy.config.use_cache = True
    for n, p in policy.named_parameters():
        p.requires_grad_(("routed_dense" in n) or ("lm_head" in n))
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=a.lr)
    dev = next(policy.parameters()).device

    start = time.time()
    for step in range(1, a.steps + 1):
        name, py = TASKS[(step - 1) % len(TASKS)]
        pids = prompt_ids(tok, py, dev)
        plen = pids.shape[1]
        policy.eval()
        with torch.inference_mode():
            gen = policy.generate(pids, do_sample=True, temperature=a.temperature, top_k=20,
                                  num_return_sequences=a.group_size, max_new_tokens=a.max_new_tokens, pad_token_id=9)
        # reward each sample — ISOLATED subprocess per kernel, run in PARALLEL
        # (a bad kernel can't corrupt the trainer's CUDA context; cold compiles overlap)
        texts = [tok.decode(gen[g][plen:], skip_special_tokens=True) for g in range(a.group_size)]
        with ThreadPoolExecutor(max_workers=a.group_size) as ex:
            out = list(ex.map(lambda gt: reward_for_text_isolated(gt[1], op=name, uniq=f"s{step}g{gt[0]}"),
                              list(enumerate(texts))))
        rewards = [o[0] for o in out]; results = [o[1] for o in out]
        rt = torch.tensor(rewards)
        # DAPO dynamic sampling: skip zero-variance groups (no learning signal)
        if rt.std() < 1e-6:
            row = {"step": step, "task": name, "mean_reward": float(rt.mean()),
                   "compiled": sum(x["compiled"] for x in results), "correct": sum(x["correct"] for x in results),
                   "skipped": "no-variance", "elapsed": time.time() - start}
            metrics.write(json.dumps(row) + "\n"); metrics.flush(); print(json.dumps(row), flush=True)
            continue
        adv = (rt - rt.mean()).to(dev)                 # Dr.GRPO: no std/length normalization

        policy.train()
        opt.zero_grad(set_to_none=True)
        loss_terms = []
        for g in range(a.group_size):
            seq = gen[g:g+1]
            lp, ntok = seq_logprob(policy, seq, plen)
            with torch.no_grad():
                rlp, _ = seq_logprob(ref, seq, plen)
            kl = (lp - rlp) / max(ntok, 1)
            loss_terms.append(-(adv[g] * lp / max(ntok, 1)) + a.kl_beta * kl)
        loss = torch.stack(loss_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
        opt.step()

        row = {"step": step, "task": name, "loss": float(loss.detach().cpu()),
               "mean_reward": float(rt.mean()), "max_reward": float(rt.max()),
               "compiled": sum(x["compiled"] for x in results), "correct": sum(x["correct"] for x in results),
               "best_speedup": max([x["speedup"] for x in results if x.get("speedup")] or [0]),
               "elapsed": time.time() - start}
        metrics.write(json.dumps(row) + "\n"); metrics.flush(); print(json.dumps(row), flush=True)
        if step % 10 == 0:
            policy.save_pretrained(a.output_dir / f"checkpoint-step-{step}", safe_serialization=True)
    policy.save_pretrained(a.output_dir / "checkpoint-final", safe_serialization=True)


if __name__ == "__main__":
    main()
