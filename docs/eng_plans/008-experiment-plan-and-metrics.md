# Engineering Plan 008: Densification Experiment Plan & Metrics Dashboard

Roadmap for the untried ideas (data distribution · architecture · bigger arch · quantization)
and a **teammate-ready set of graphs** to measure. Builds on the reconstruction recipe
([007](007-reconstruction-recipe-paper-aligned.md)), RADLADS (arXiv:2505.03005), and
KRAFTON MoE→Dense (arXiv:2605.28207).

Baseline so far: ~3.0 B dense student, all-39-layer teacher-forced reconstruction, Adafactor,
OpenCodeInstruct. Random-init run: loss 0.25→0.022. **Warm-start run (live):** by step 510 deep-MSE
0.018 / shallow 0.00008 — ≈ the random run's *final* quality, ~4× sooner.

---

## Part A — Experiment plan

### Tracks & ordering
```
T1 Architecture ──► T3 Objective ──► T4 Quantization
T2 Data         ──┘  (each track feeds eval in Part B)
```

| ID | Hypothesis | Change vs baseline | Success metric | Effort | Risk | Depends |
|----|------------|--------------------|----------------|--------|------|---------|
| **E0** | Warm-start beats random | DO-ACP init (done) | deep-MSE ↓ faster | done | — | — |
| **E1** | **Wider dense FFN** absorbs the 256-expert function better | routed_dense intermediate 4096 → **8192/12288** | lower recon MSE + perplexity at matched tokens | low | size↑ | placeholder rebuild |
| **E2** | **Coverage-max data mix** supervises all experts | weight OpenCode + KernelBook + repo/patches to max expert-activation entropy | higher per-expert coverage; lower tail-layer MSE | low–med | mix tuning | profiler |
| **E3** | **K sweep / grouping** | K=8→16/32; pure-prune vs weight/router/output clustering | recon MSE vs K knee | med | mem | E1 |
| **E4** | **Hybrid partial densify** | densify concentrated mid-stack only; keep diverse layers MoE | quality/size Pareto point | med | plumbing | profiler |
| **E5** | **Logit-KD tail** | + KL(student‖teacher) on full forward | perplexity ↓ toward teacher | med | compute | E1/E2 |
| **E6** | **Student-forced fine-tune** | switch teacher-forced → student-forced | perplexity (fixes compounding) | med | instability | E5 |
| **E7** | **On-policy rollout distill** | teacher rollouts via `pool` as data | SWE-bench pass ↑ | high | harness | E5 |
| **E8** | **NVFP4 QAT reconstruction** | train routed_dense quant-aware (sim FP4) | quality at FP4 vs post-quant | med–high | kernels | E1 |

**Defaults:** Adafactor 2e-4, seq 2048, eff-batch 2, bf16, all-39-layer, `--normalize-loss`.
**Budgets:** smoke 0.2–1 M → recovery 10–50 M → RADLADS MVP 0.5–0.7 B tokens.

---

## Part B — Graphs a teammate can measure

All reconstruction graphs read `runs/<run>/metrics.jsonl` (one JSON/step with `step`, `loss`,
`elapsed_sec`, `per_layer[ℓ].{mse,cosine_loss,token_count}`). Eval graphs need a held-out set + a
short eval script.

### B1 · Reconstruction (have data now)
| # | Graph | x | y | Source | "Good" |
|---|-------|---|---|--------|--------|
| 1 | **Total loss vs step** | step | loss | metrics.jsonl | monotone ↓ |
| 2 | **Per-layer MSE heatmap** | step | layer 1–39 (color=MSE, log) | per_layer.mse | deep rows cool over time |
| 3 | **Deep vs shallow MSE** | step | mean MSE (L28–39 vs L1–10) | per_layer.mse | deep gap closes |
| 4 | **Cosine-loss vs step** | step | 1−cos | per_layer.cosine_loss | → 0 |
| 5 | **Warm-start vs random overlay** | step | deep-MSE | two runs | warm-start lower/faster |
| 6 | **Effective rank per layer** | layer | eff-rank of selected K | warm_start log | ↑ = diverse (avoid L30-type collapse) |

### B2 · Model quality (needs held-out eval)
| # | Graph | x | y | Notes |
|---|-------|---|---|-------|
| 7 | **Perplexity vs tokens** | tokens | ppl (held-out code + C4) | the real metric; vs teacher line |
| 8 | **Perplexity gap to teacher** | run/stage | ppl_student − ppl_teacher | target → 0 |
| 9 | **Per-domain perplexity** | domain (py/triton/cuda/prose) | ppl | shows kernel-data effect |

### B3 · Efficiency / Pareto (the hackathon money-shot)
| # | Graph | x | y | Notes |
|---|-------|---|---|-------|
| 10 | **Quality vs size Pareto** | params / GB | SWE-bench pass or ppl | dense student vs teacher vs K-variants |
| 11 | **Tokens/sec vs config** | config | tok/s | bf16 vs FP4; batch; seq |
| 12 | **VRAM vs config** | config | peak GB | fits-on-1-GPU story |
| 13 | **Quality vs bits** | bf16/INT4/NVFP4/2-bit | ppl or pass | FP4 = speed floor |

### B4 · Data / coverage
| # | Graph | x | y | Notes |
|---|-------|---|---|-------|
| 14 | **Expert-activation coverage by dataset** | expert id (sorted) | activation freq | C4 vs OpenCode vs KernelBook overlay |
| 15 | **Coverage vs mixture weights** | mixture | effective-experts/layer | tune the "perfect distribution" |

### B5 · Agentic (via `pool`)
| # | Graph | x | y | Notes |
|---|-------|---|---|-------|
| 16 | **SWE-bench pass-rate vs tokens** | training tokens | % solved | student vs teacher |
| 17 | **Tokens / trajectory** | task | tokens | efficiency on real coding |

### Quick recipes
- Loss/heatmap: `metrics.jsonl` → matplotlib (`loss_curve.png`, `pipeline.png` already produced).
- Perplexity: load checkpoint (`trust_remote_code`), CE over held-out, `ppl=exp(loss)`.
- Coverage (#14): reuse `scripts/analyze_expert_activation.py` per dataset.
- SWE-bench: `pool exec --api-url <served student>` on a Verified subset.

---

## Owner split (suggested)
- **A — Architecture/recon:** E1, E3, E4 + graphs 1–6, 10.
- **B — Data:** E2, E7 + graphs 14, 15, 9.
- **C — Objective/eval:** E5, E6 + graphs 7, 8, 16, 17.
- **D — Quant/serving:** E8 + graphs 11, 12, 13.

*Refs: RADLADS arXiv:2505.03005 · MoE→Dense arXiv:2605.28207.*
