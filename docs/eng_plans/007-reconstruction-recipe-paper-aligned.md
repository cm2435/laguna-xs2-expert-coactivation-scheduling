# Engineering Plan 007: Paper-Aligned Densification Recipe + Trick Matrix

**Status:** Active. Supersedes the SFT-first compromise in [005](005-dense-midtraining-posttraining.md)
with a recipe aligned to the two reference papers, plus the modifications from our paper-reading
sessions and the live reconstruction run.

---

## 1. Output model

| | Value |
|---|---|
| Class | `LagunaDense` (`laguna_dense`), dense SwiGLU FFN per layer |
| **Total params** | **2,996,678,656 (~3.00 B)** — all active |
| Disk (bf16) | 5.99 GB (1 shard) — vs teacher ~66 GB |
| Breakdown | attention **1.43 B** · routed_dense (trainable) **0.98 B** · embed+lm_head **0.41 B** · shared expert **0.12 B** |
| Hidden / layers | 2048 / 40 (1 dense + 39 densified-routed) |
| Routed FFN width | K8 × 512 = **4096** (SwiGLU gate/up/down) + kept shared expert |
| Attention | 48 heads / 8 KV (GQA); 30 sliding-window(512) + 10 global — copied verbatim |
| Context / vocab | 262 144 / 100 352 |

The teacher (`poolside/Laguna-XS.2`) is a 33 B-total / 3 B-active MoE; we collapse the 256-way
routed experts (top-8) into one dense SwiGLU per layer → **~10× smaller on disk, fully dense**.

---

## 2. Concept summary (what our approach rests on)

### 2.1 The teacher's MoE, precisely
- **256 routed experts, top-8 per token**, plus **1 always-on shared expert** (a dense MLP).
- **Sigmoid router** (not softmax) + `e_score_correction_bias`; top-8 weights renormalized to sum 1; routed output scaled by **2.5** then added to the shared output.
- FFN activation **SiLU → SwiGLU** (`down(silu(gate(x))·up(x))`), not ReLU.
- These three (sigmoid / shared / SwiGLU) are where Laguna diverges from the papers' softmax-MoE assumptions and drive every adaptation below.

### 2.2 Densification = MoE→Dense (KRAFTON, arXiv:2605.28207)
- Pipeline: **score → select top-K → group → merge → concatenate into a dense FFN (with down-proj magnitude scaling) → distill**.
- **Scoring dominates** (5.7 pp spread vs ~1 pp for grouping). Winner: **DO-ACP** —
  - **ACP** = activation-weighted conditional prob = `CP_e · √E_t‖f_e(t)‖²` (routing confidence × output magnitude).
  - **DO** (D-optimal) = greedy `arg max log det(K_S + λI)` on the importance-weighted **expert-output Gram** `K_ij = √(I_iI_j)·G_ij`, `G_ij = E_t⟨f_i,f_j⟩` — picks experts that are important **and** mutually non-redundant.
  - Diversity measured by **effective rank** of the selected set; frequency-based scoring picks redundant experts.
- **Pure pruning (no merge)** + DO-ACP is best; magnitude scaling on down-proj preserves activation magnitude (static α can't reproduce token-dependent routing → why distillation is needed).
- Result: MoE→dense beats dense-to-dense pruning by **+6.3 pp** after ~4 B-token KD, 1.6× faster.

### 2.3 Cheap transfer (RADLADS, arXiv:2505.03005)
- 3-stage architecture transfer: **(1) align intermediate representations → (2) logit KD → (3) SFT/preference**, on **350–700 M tokens (<0.005 %** of pretraining).
- Lesson: align reps *first*; small token budget suffices; dataset choice matters.

### 2.4 Our empirical anchors (C4, 161,932 tokens)
- All 256 experts fire, but only **~158 effective experts/layer** (load Gini ≈ 0.53) → K must exceed top-8 to capture capacity; K=8 under-fits, especially deep layers.
- Coactivation pairs fire ~3–3.6× above chance → grouping structure exists.

### 2.5 Training mechanics
- **Teacher-forced** per-layer reconstruction: student learns `teacher_x_ℓ → teacher_y_ℓ`, so untrained early layers don't corrupt later inputs (no compounding). All 39 layers trained **in parallel** in one backward.
- **Compounding caveat:** low teacher-forced MSE ≠ low end-to-end perplexity → must finish with student-forced + measure perplexity.
- **Optimizer memory:** AdamW state (m+v fp32) for 0.98 B trainable = 7.9 GB → teacher+student+AdamW > 80 GB. **Adafactor** (≈0 extra state) fits all-layer on one H100; trivial on GB300.
- **Meaningful metrics:** reconstruction MSE/cosine are *proxies*; **perplexity** and **SWE-bench via `pool`** are the end-to-end scores.

---

## 3. Updated recipe (staged, paper-aligned)

| Stage | Source | What | Loss | Tokens | Trainable |
|---|---|---|---|---|---|
| **0 · Warm-start init** | KRAFTON score+concat | DO-ACP select K experts → concat into `routed_dense` (gate/up/down) + α·2.5 magnitude scaling | — (init) | 0 | — |
| **1 · Reconstruction** | RADLADS step-1 | all-39-layer teacher-forced; **current run** | `mean_ℓ(MSE + 0.05·(1−cos))`, masked, **÷mean(y²)** norm | 0.3–0.7 B | routed_dense |
| **2 · Logit KD** | RADLADS step-2 / KRAFTON | full student forward, KL to teacher logits | `KL(student‖teacher)` (+CE) | 0.5–4 B | routed_dense (+norms) |
| **3 · Post-train** | RADLADS step-3 | student-forced SFT on coding + **kernel data**; optional DPO | CE (+ pref) | 0.2–1 B | routed_dense (+norms, lm_head) |

**Data mix (stages 1–3):** `nvidia/OpenCodeInstruct` + **`GPUMODE/KernelBook` (PyTorch→Triton kernels, newly wired in)** + repo/patch traces. Eval after each stage: per-layer recon MSE/cosine → **perplexity** → **SWE-bench-Verified subset via `pool`**.

**Defaults:** Adafactor lr 2e-4; seq 2048; eff-batch 2 (grad-accum); K=8 now, K=16 sweep; bf16.

---

## 4. Trick matrix (what we could try)

| # | Trick | Lever | Paper basis | Expected gain | Cost | Risk | Status |
|---|---|---|---|---|---|---|---|
| 1 | **DO-ACP warm-start** init of routed_dense | init | KRAFTON | **High** — start near teacher | low (have `densify_layer.build_dense_ffn`) | low | planned next |
| 2 | ACP **magnitude scaling** (α·2.5) into down-proj | init | KRAFTON §3.2 | med | low | low | planned |
| 3 | **Per-layer loss norm** (÷mean(y²)) + depth weighting | loss | ours | med — balances deep layers | low | low | planned |
| 4 | **K sweep 8→16→32** | arch | ours (158 eff-experts) | med–high | mem/size | med | planned |
| 5 | **Logit-KD (KL) tail** | objective | RADLADS-2 | high — fixes function not just features | med (full fwd) | med | stage 2 |
| 6 | **Student-forced fine-tune** tail | objective | RADLADS | high — kills compounding | med | med | stage 3 |
| 7 | **Kernel-data mix** (KernelBook) | data | ours/brief | med — domain coverage | low (done) | low | **wired in** |
| 8 | Cosine-weight / LR-warmup tuning | opt | — | low–med | low | low | open |
| 9 | Expert **grouping+merge** vs pure-prune at init | init | KRAFTON | low (~1 pp) | med | low | optional |
| 10 | Gradient checkpointing / bigger batch | systems | — | throughput | low | low | as needed |
| 11 | **Perplexity + SWE-bench (`pool`) eval** | eval | ours | — (the real metric) | med | low | required |
| 12 | NVFP4 / expert-mixed-precision quant of dense student | downstream | brief | mem/speed | med | med | future |
| 13 | Speculative decode (observation/kernel-grammar draft) | downstream | brief | throughput | med | low | future |
| 14 | RADLADS-style **attention** linearization | arch | RADLADS | long-ctx speed | high | high | out of scope |

---

## 5. References
- RADLADS — arXiv:2505.03005 · MoE→Dense — arXiv:2605.28207
- Companion: notes gist `ba18a3e`, run/comparison gist `08f51e7`, smoke gist `5e53fb8`, HF `EvanOLeary/laguna-xs2-dense-k8-recon`.
