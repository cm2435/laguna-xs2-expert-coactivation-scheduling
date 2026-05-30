# Densifying Laguna XS.2 (33B MoE → ~3B dense): recovering code-gen with SFT

I took a dense ~3B model densified from **Laguna XS.2** (poolside's 33B Mixture-of-Experts coder) and ran supervised fine-tuning on code instructions to **recover its single-shot code-generation ability**, which the MoE→dense conversion destroys. Before SFT the model emitted repetitive non-code text; after SFT it writes clean, documented Python. I then tested whether **scaling SFT** fixes correctness — it doesn't, which is the interesting part. This tracks single-shot code generation (measured on HumanEval) — a separate lane from the team's agentic tool-calling model, which is evaluated through an agent harness instead.

This is the **recovery + measurement end** of the team's densification pipeline: I take the shared dense checkpoint and answer "how much code-gen survives, and what recovers it?" — the single-shot proxy for a model whose agentic eval needs a harness nobody's built yet.

## Background

Laguna XS.2 is a **33B MoE, 3B active/token** (256 routed experts + 1 shared). The team is converting it into a **dense ~3B** model (cheaper / faster to serve) by selecting a diverse subset of experts (DO-ACP, per the KRAFTON MoE→dense paper, arXiv 2605.28207) and reconstructing the dense FFNs against the teacher. Densification costs quality → it needs **recovery training**. This experiment covers one recovery path: **single-shot SFT**.

## Setup

| | |
|---|---|
| Student | `EvanOLeary/laguna-xs2-dense-k8-recon` (~3B dense, post-reconstruction) |
| Data | `nvidia/OpenCodeInstruct` (instruction → code) |
| Objective | causal-LM next-token cross-entropy, chat-template formatted |
| Hyperparams | batch 1 · seq 2048 · lr 2e-5 · **10,000 steps** (~10M tokens) · bf16 · AdamW |
| Hardware | 1× A100 80GB |

## Loss curve

Loss drops sharply **(~2.1 → 0.5)** in the first few hundred steps, then converges around ~0.3 (batch-1 noise). (See `results/sft_25k_metrics.jsonl` for the continued-run loss data.)

## Before / after

Greedy decode, prompt: *"Write a Python function that adds two numbers."*

**Before SFT (dense recon model)** — repetitive, no code:

```text
The function should return a tuple containing the first number and the second number.
The function should also handle the following cases:
- The input is a list of integers.
- The input is a list of integers.   ← loops, never produces code
...
```

**After SFT** — correct, documented:

```python
def add_numbers(a, b):
    """
    Adds two numbers and returns their result.

    Parameters:
    a (int or float): The first number.
    b (int or float): The second number.

    Returns:
    int or float: The sum of a and b.
    """
    return a + b
```

## Evaluation — HumanEval (10-problem subset)

| Model | pass@1 |
|---|---|
| Dense + **10k** single-shot SFT | 1/10 (10%) |
| Dense + **25k** single-shot SFT (10k → +15k, warmup+cosine) | 1/10 (10%) |

The model writes **clean, well-formed code** — correct signatures, type hints, docstrings, valid Python — but its **problem-solving logic is weak**. The failures are *reasoning* bugs, not *format* bugs:

- `below_zero` sums all operations and checks the final total, instead of the **running** balance.
- `sum_product` initializes the product to **0** instead of 1.
- `mean_absolute_deviation` computes **variance** (squared deviation) instead of absolute deviation.

Every function is syntactically perfect and wrong on the logic. (Full per-problem dumps in `results/humaneval_outputs.md` and `results/humaneval_after.md`.)

## Does more SFT fix the logic? (no)

The obvious hypothesis is "it just needs more tokens." I tested it: **continued SFT from 10k → 25k steps** (with a proper warmup + cosine LR schedule). pass@1 stayed **exactly 1/10**, and the failures are the same class of reasoning bug.

So the bottleneck **isn't SFT volume** — single-shot SFT recovers the *form* of code generation and **plateaus on correctness around ~10%**. Fixing the logic needs a method that transfers the teacher's *reasoning*, not more next-token imitation: **logit-KD** (match the teacher's distribution, Stage 2) or **RFT** (reward correct solutions against unit tests).

## Findings

- **SFT cleanly recovers code generation.** The dense model went from *no code* → *clean, well-documented Python*, including the chat / `</think>` format.
- **But it plateaus on logic.** Scaling SFT 10k → 25k steps did **not** move pass@1 (10% → 10%); the failures are reasoning bugs, not formatting. The bottleneck is the **recovery method**, not token volume.
- **The next lever isn't more SFT** — it's reasoning transfer: **logit-KD** from the teacher, or **RFT** with test-based rewards.
- This is a **single-shot baseline**, distinct from the agentic tool-calling model the broader project targets (which needs an agent harness + failure-recovery trajectories, not plain code SFT).

## Deployment — does the dense model actually run small?

The whole point of densifying is a smaller, deployable model, so I checked it actually serves:

- **On-device:** exported the dense student to **ExecuTorch** (`torch.export` → `.pte`); it loads and runs a forward pass (logits `[1, 16, 100352]`) — the custom Laguna-dense arch is exportable for mobile/edge runtimes.
- **Cloud:** served it via FastAPI + a tunnel for a live chat demo (see `serve.py`).

Note: GGUF/llama.cpp (PocketPal-style) **doesn't** support the custom arch (dual FFN, per-head gating, QK-norm), so on-device has to go through ExecuTorch, not GGUF.

## Reproduce

```bash
# the recon checkpoint ships without its modeling code on HF — copy it from the copied-shell repo
hf download EvanOLeary/laguna-xs2-dense-k8-recon --local-dir ./recon_model
hf download cm2435-new/laguna-xs2-dense-k8-copied-shell --include "*.py" --local-dir ./code
cp ./code/*.py ./recon_model/

# Stage A — 10k SFT baseline
python scripts/train_dense_sft.py \
  --model ./recon_model --dataset nvidia/OpenCodeInstruct --split train \
  --output-dir runs/sft --max-steps 10000 --batch-size 1 --seq-len 2048 \
  --learning-rate 2e-5 --log-every 100 --save-every 2000

# Stage B — continue to 25k with warmup+cosine (the scaling test)
# starts from the published 10k model — no separate 25k checkpoint needed
hf download Jessicacat0305/laguna-xs2-dense-k8-sft-opencode --local-dir ./sft_model
python scripts/train_dense_sft.py \
  --model ./sft_model --dataset nvidia/OpenCodeInstruct --split train \
  --output-dir runs/sft_more2 --max-steps 15000 --batch-size 1 --seq-len 2048 \
  --learning-rate 1e-5 --warmup-steps 300 --log-every 100 --save-every 2500

# Eval — HumanEval pass@1 with per-problem output dump
python eval_humaneval.py --model ./sft_model --num 10 --out results/humaneval_outputs.md
```

Training script: [`scripts/train_dense_sft.py`](../scripts/train_dense_sft.py) · Eval harness: [`eval_humaneval.py`](../eval_humaneval.py) · Model: [`Jessicacat0305/laguna-xs2-dense-k8-sft-opencode`](https://huggingface.co/Jessicacat0305/laguna-xs2-dense-k8-sft-opencode)
