# Model Changes — Laguna-XS.2 MoE → Dense

What we changed about the model (and the recipe that trains it), and why. Teacher is
`poolside/Laguna-XS.2`; the student we are building is
`cm2435-new/laguna-xs2-dense-k8-copied-shell`.

> Companion diagnostics: C4 expert-activation gist —
> https://gist.github.com/Tyronita/cdcb80969d208b83e3f48cddfbbb1422

---

## 0. The one-line change

Replace every routed **MoE block** (256 experts, top-8) with a **single dense SwiGLU
FFN**, collapsing a **33.4B-total / 3.0B-active** sparse model into a **~3.3B
fully-dense** coding LLM, then recover quality by distillation. Everything else below
is in service of making that swap lossless.

---

## 1. Architecture changes (teacher → student)

| | Teacher `Laguna-XS.2` (MoE) | Student `…-dense-k8-copied-shell` | Why |
|---|---|---|---|
| `model_type` | `laguna` | **`laguna_dense`** | new FFN class; attention/embeddings unchanged |
| Params | 33.4B total / **3.0B active** | **≈3.3B, all active** | dense = active; removes routing/serving complexity |
| FFN per sparse layer | 256 routed experts (top-8) + 1 shared | **1 dense SwiGLU, width 4096 (K8×512) + shared kept** | one matmul instead of a gather over 256 experts |
| Router | sigmoid + `e_score_correction_bias`, top-8 renorm, ×2.5 scale | **removed (collapsed)** | no routing in a dense model |
| Routed scaling ×2.5 | applied to expert outputs | **folded into dense down-projection init** | preserve output magnitude after collapse |
| Shared expert | always-on, width 512 | **kept verbatim** | already dense; it carries the always-on pathway |
| Hidden / layers | 2048 / 40 (1 dense + 39 sparse) | **2048 / 40 (identical)** | only the FFN type changes |
| Attention | 48 heads / 8 KV (GQA), 30 SWA(512) + 10 global | **identical (weights copied)** | attention is not the target of densification |
| Activation | SwiGLU/SiLU | **SwiGLU/SiLU** | match teacher |
| Disk (bf16) | ~66 GB (14 shards) | **~6.6 GB (2 shards)** | 10× smaller to serve |

**Why "K8" dense width.** The dense FFN width is set to `K × moe_intermediate (512)`.
We start at **K=8** (= top-k, width 4096) so the collapsed FFN has the same per-token
expert capacity the router used. The C4 diagnostics show ~**158 effective experts/layer**,
so **K is a planned sweep (8 → 16 → 32)** to trade size against reconstruction fidelity.

**Why these adaptations.** The two source methods (RADLADS arXiv:2505.03005; KRAFTON
MoE→Dense arXiv:2605.28207) assume a **softmax router + ReLU-family FFN**. Laguna uses
**sigmoid routing**, **SwiGLU**, an **always-on shared expert**, and a **2.5× routed
scaling** — so the collapse keeps the shared expert as-is and folds the 2.5× into the
dense down-proj rather than dropping it.

---

## 2. How the dense FFN is built (DO-ACP)

`src/densify/densify_layer.py` — score experts → select top-K → merge → concatenate into
one dense SwiGLU (with down-proj magnitude scaling).

- **Scoring = DO-ACP** (KRAFTON's winner): D-optimal greedy `arg max log det` of the
  importance-weighted expert-output Gram kernel — maximizes both **importance** and
  **diversity** (effective rank). Frequency scoring alone picks redundant experts; the
  paper shows scoring dominates grouping (~5.7pp vs ~1pp).
- Output of this stage is used as a **warm-start** for the dense FFN (vs random init),
  which is the single biggest convergence lever.

---

## 3. Training recipe = **reconstruction pretraining** (not SFT, not logit-KD)

`scripts/train_dense_reconstruction.py` + `src/densify/reconstruction*.py`.

```
teacher = Laguna-XS.2  (frozen, eval, no_grad, use_cache=False)
student = dense-k8-copied-shell  (only *.routed_dense.* trainable; attn/embed/norms/shared/lm_head frozen)

per step (all 39 layers in parallel, TEACHER-FORCED):
  one teacher forward; hooks capture (x_l, y_l) at each MoE block
  pred_l = student.layer[l].routed_dense(x_l)      # teacher's input, not student's
  loss   = mean_l( mse(pred_l, y_l) + 0.05*cos(pred_l, y_l) )   # attention-masked
  AdamW(lr=2e-4).step()
```

**Why teacher-forced + all-layer-parallel:** each dense FFN learns `teacher_x_l →
teacher_y_l` independently, so untrained early layers don't corrupt later layers' inputs
(no error compounding). Maps to RADLADS step-1 / KRAFTON feature-reconstruction.

**Recipe mods already landed** (commits `3ff0c5c`, `8a687e2`):
- **DO-ACP warm-start** of `routed_dense` instead of random init.
- **Per-layer loss normalization** (divide MSE by `mean(y_l²)`) so deep/shallow layers
  contribute evenly.
- **Memory-fit optimizer**: restricting the loss to a layer subset keeps AdamW state in
  budget on an 80GB H100 (AdamW allocates state lazily only for params that get grads);
  the full 39-layer run fits on the GB300.

---

## 4. Data-mixture changes (this branch: `recipe/paper-aligned-densification-kernel-data`)

The reconstruction signal only depends on *what text drives the teacher forward pass*, so
the data mix is a model-quality lever. Changes:

- **Multi-dataset interleave** (`--datasets name:weight,…` in
  `train_dense_reconstruction.py`): probabilistically interleaves several **streaming**
  datasets by weight with a **seeded RNG** (deterministic/reproducible), draining each
  source as it exhausts. Replaces the single-`--dataset` path. **Why:** C4 coverage is
  domain-dependent — densifying on one domain overfits the surrogate to that domain's
  active experts; mixing code/Triton/CUDA broadens the activated-expert footprint.
- **Wider SFT field handling** (`reconstruction_data.py`): the row formatter now also
  recognizes `query` (instruction) and `kernel`/`code` (raw content) keys, so kernel/code
  datasets (KernelBook, cuda_kernels, CodeFeedback) format correctly without per-dataset
  glue. **Why:** these datasets don't use the OpenCodeInstruct `instruction`/`output`
  schema; without this they raised "could not format row".

The mixture deliberately anchors on **Triton** (KernelBook) and adds **CUDA C++** (Sakana
source, commit `b8e50c6`) since the downstream target is low-level kernel optimization.

---

## 5. What is NOT changed

- Attention (GQA, sliding-window/global pattern, RoPE/YaRN), embeddings, norms, vocab,
  and context length are **copied verbatim** from the teacher.
- The shared expert is **kept**, not retrained.
- This is **representation alignment only** — logit-KD (KL) and SFT/DPO are later phases,
  not part of these changes.

---

## 6. Open levers (next)

K-sweep 8→16→32 · down-proj scaling variant (marginal-routing vs uniform) · logit-KD
phase after reconstruction · student-forced fine-tune tail · perplexity + SWE-bench eval
as the score (not just MSE/cosine) · full 39-layer run on GB300.
