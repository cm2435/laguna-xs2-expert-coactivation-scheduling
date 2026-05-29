# Laguna-XS.2 → Dense — Mega Summary, Recipe & Plan

Densifying the **`poolside/Laguna-XS.2`** MoE into a small **dense coding LLM** by replacing the
routed Mixture-of-Experts with dense SwiGLU FFNs, then recovering quality by distillation.
Aligned to **RADLADS** (arXiv:2505.03005) and **Pruning & Distilling MoE into Dense**
(arXiv:2605.28207). Built during the Poolside Laguna XS.2 research hackathon.

---

## TL;DR
- **Teacher:** `poolside/Laguna-XS.2` — 33 B total / 3 B active MoE (256 experts top-8 + 1 shared).
- **Student (output):** **`laguna_dense`, 2,996,678,656 params (~3.00 B), fully dense, 5.99 GB bf16.**
- **Method:** teacher-forced, **all-39-layer-parallel** reconstruction of each MoE block by a dense
  SwiGLU FFN (`routed_dense`), then logit-KD + student-forced SFT (paper-aligned stages).
- **Live run:** all-layer reconstruction, **Adafactor**, on **1× H100 80 GB** — loss **0.25 → 0.021**,
  **~6.2 M tokens in ~26 min**, 6 checkpoints pushed to HF.

## Hardware (this session)
| | |
|---|---|
| GPU | **1× NVIDIA H100 PCIe, 80 GB** (driver 570.148, CUDA 12.8) |
| Peak VRAM in training | **77.2 / 81.5 GB** (teacher 66 GB frozen + student ~6 GB + Adafactor) |
| Util / temp | ~88 % / ~75 °C |
| CPU / RAM / disk | 26 vCPU / 221 GB / 968 GB free |
| Throughput | **~1.04 s/optimizer-step** (eff-batch 2, seq 2048) ≈ ~4k tok/s |
| Note | All-39-layer fits **only** with Adafactor (AdamW state overflows 80 GB); trivial on the GB300 (192 GB). |

## Output model
| | Teacher `Laguna-XS.2` | **Student `laguna-xs2-dense-k8`** |
|---|---|---|
| Type / params | MoE, 33 B total / 3 B active | **dense, ~3.00 B (all active)** |
| FFN | 256 experts top-8 + shared | **1 SwiGLU, width K8×512=4096 + kept shared** |
| Param split | — | attn **1.43 B** · routed_dense **0.98 B** · embed+lm_head **0.41 B** · shared **0.12 B** |
| Attention | 48 h / 8 KV GQA; 30 SWA(512)+10 global | identical (copied) |
| Hidden/layers · ctx · vocab · act | 2048/40 · 262 k · 100 352 · SiLU | same |
| Disk (bf16) | ~66 GB | **5.99 GB** |

---

## What we've done (journey)
1. **Teacher up:** downloaded Laguna-XS.2 (63 GB), verified load+generation (66.9 GB, ~22 s).
2. **OpenAI shim:** transformers-backed `/v1/chat/completions` with Laguna tool-call/`<think>` parsing.
3. **pool harness** installed + standalone wiring.
4. **Expert-activation analysis** (161,932 C4 tokens): all 256 experts used but **~158 effective/layer**, Gini ≈ 0.53, coactivation ~3–3.6× chance → *report merged in PR #3*.
5. **Densification scoring MVP** (`src/densify/densify_layer.py`): frequency / ACP / **DO-ACP** + dense-FFN build + 4 CPU tests → *PR #3 merged*.
6. **Dense reconstruction pipeline** (`scripts/train_dense_reconstruction.py` + `reconstruction*.py`, commit `b8e2aaf`).
7. **Smoke run** (8 layers, 20 steps): loss 0.0486→0.0332, cosine 0.95→0.58 → loop validated.
8. **All-39-layer run** (this branch): Adafactor, eff-batch 2, seq 2048 → loss 0.25→0.021, ~6.2 M tokens, checkpoints on HF.
9. **Kernel data wired in** + **paper-aligned recipe (eng-plan 007)** + this README.

---

## Training recipe (updated, staged, paper-aligned)

| Stage | Source | What | Loss | Tokens | Trainable |
|---|---|---|---|---|---|
| **0 Warm-start** | KRAFTON | DO-ACP select K experts → concat into `routed_dense` + α·2.5 down-scaling | — | 0 | — |
| **1 Reconstruction** *(running)* | RADLADS-1 | all-39-layer **teacher-forced**; hooks capture `(x_ℓ,y_ℓ)`; `routed_dense_ℓ(x_ℓ)` predicts | `mean_ℓ(MSE + 0.05·(1−cos))`, masked, **÷mean(y²)** | 0.3–0.7 B | routed_dense |
| **2 Logit-KD** | RADLADS-2 | full student fwd, KL to teacher logits (+CE) | `KL + CE` | 0.5–4 B | routed_dense (+norms) |
| **3 Post-train** | RADLADS-3 | **student-forced** SFT on coding + kernel data; optional DPO | `CE (+pref)` | 0.2–1 B | + norms, lm_head |

**Optimizer:** Adafactor lr 2e-4 (memory) · **only `routed_dense` trainable** (attn/embed/norms/shared/lm_head frozen) · seq 2048 · eff-batch 2 (batch 1 × grad-accum 2) · bf16. **Eval:** per-layer recon MSE/cosine → **perplexity** → **SWE-bench-Verified via `pool`** (the meaningful end-to-end metrics; teacher-forced MSE is only a proxy).

---

## Training data contents

| Dataset | Role | Schema → format | Status |
|---|---|---|---|
| **`nvidia/OpenCodeInstruct`** | code instruction SFT (read→reason→code) | `messages` / `instruction`+`output` → `<user>/<assistant>` | **in use (stage 1)** |
| **`GPUMODE/KernelBook`** | **GPU kernels** — PyTorch→Triton pairs | `python_code` → `triton_code` → conversion prompt | **wired in** (`--dataset GPUMODE/KernelBook`) |
| `*/KernelBook-messages`, multiturn Triton reasoning traces | kernel reasoning, varied internal states | JSON-string `messages`/`full_messages` | supported |
| (earlier) `allenai/c4` excerpt | expert-activation profiling only | text | analysis only |

- Tokenized to seq 2048, streamed, padded with attention-mask (pad tokens excluded from loss).
- **Tokens consumed so far: ~6.23 M** (= steps × grad_accum 2 × batch 1 × seq 2048); extending toward the **~0.5–0.7 B RADLADS budget**.
- *Why SFT/kernel traces over raw code:* they give varied internal states (prose, code, tests, patches, Triton) → better coverage for layer reconstruction than next-token raw-file continuation.

---

## Live run results
- Total loss **0.25 → 0.0214** by step 1520 (~12×), **~26 min**, ~6.2 M tokens, on 1× H100.
- Per-depth: shallow layers (1–10) reach MSE ~1e-3; **deep layers (28–39) lag** (~0.1–0.3) — larger MoE-output magnitudes.
- Diagram + loss curve: `EvanOLeary/laguna-xs2-dense-k8-recon` (`pipeline.png`, `loss_curve.png`).

## The plan + additions (trick matrix)
| # | Trick | Lever | Basis | Gain | Status |
|---|---|---|---|---|---|
| 1 | **DO-ACP warm-start** init | init | KRAFTON | high | next |
| 2 | ACP magnitude scaling into down-proj | init | KRAFTON | med | planned |
| 3 | Per-layer loss norm (÷mean(y²)) + depth weighting | loss | ours | med | planned |
| 4 | K sweep 8→16→32 | arch | ours (158 eff) | med–high | planned |
| 5 | Logit-KD (KL) tail | objective | RADLADS-2 | high | stage 2 |
| 6 | Student-forced fine-tune | objective | RADLADS | high | stage 3 |
| 7 | **Kernel-data mix (KernelBook)** | data | ours | med | **done** |
| 8 | Perplexity + SWE-bench (`pool`) eval | eval | ours | — (real metric) | required |
| 9 | NVFP4 / expert-mixed-precision quant | downstream | brief | mem/speed | future |
| 10 | Speculative decode (kernel-grammar / observation draft) | downstream | brief | throughput | future |

Full 14-row matrix + concept summary: `docs/eng_plans/007-reconstruction-recipe-paper-aligned.md`.

## Artifacts
- **HF:** `EvanOLeary/laguna-xs2-dense-k8-recon` (checkpoints + model card + diagram + curve) · `EvanOLeary/laguna-xs2-densify-smoke`.
- **Gists:** notes/compute `ba18a3e` · run/comparison `08f51e7` · smoke `5e53fb8`.
- **Code:** `src/densify/{densify_layer,reconstruction,reconstruction_data}.py`, `scripts/train_dense_reconstruction.py`, eng-plans 005/007.
- **Refs:** RADLADS arXiv:2505.03005 · MoE→Dense arXiv:2605.28207.
