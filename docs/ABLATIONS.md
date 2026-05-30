# Ablations — Laguna-Dense CUDA kernel generation

Living log of experiments. **Inference-time knobs (no retraining)** are the primary sweep axes here;
training-side variants (K experts, FFN width) noted separately at bottom.

## Inference-controlled variables (our ablation axes — no retraining)
| Axis | Current | Hypothesis |
|---|---|---|
| **pass@k / best-of-k** | 1 | ↑k → ↑fast_0 for free (kernel-gen is high-variance) |
| `max_new_tokens` | 1024 | too low truncates; too high → over-engineering/rambling failures |
| `repetition_penalty` / stop-on-```` ``` ```` | off | kills loops + post-kernel rambling |
| `temperature` | 0.6 | ↓ reliability/sample, ↑ diversity for pass@k |
| `top_k`/`top_p`/`min_p` | top_k 20 | distribution truncation |
| `enable_thinking` | False | reason-then-kernel may ↑ complex-op correctness (ConCuR) |
| prompt format | SFT format | exact `\`\`\`python nn.Module\`\`\`` > plain text |
| system prompt | CUDA-only | must match target language (CUDA vs Triton); "keep simple" vs "optimized" |
| few-shot | 0 | 1–2 PyTorch→correct-CUDA examples |
| eval: atol/rtol, shape, dtype, reps | 1e-3, 4096², fp32, 50 | affects measured score, not model |

---

## Baseline A — Head-to-head: OURS vs TEACHER (k=1, 6 CUDA ops)
Settings: temp 0.6, top_k 20, max_new 1024, enable_thinking=False, same prompt/harness, sequential load.

| | OURS (dense SFT) | TEACHER Laguna-XS.2 |
|---|---|---|
| Params | **3.0 B** | 33.4 B |
| Load / VRAM | **3 s / 6 GB** | 35 s / 67 GB |
| **Avg tok/s** | **32.1** | 25.4 |
| **fast_0** | 0/6 (this run) | 4/6 |
| fast_1 (≥1× eager) | 0/6 | **0/6** |

Per-op tok/s (ours faster on all): ReLU 31.7/24.4 · Tanh 30.7/25.9 · Sigmoid 32.9/25.3 · GeLU 33.4/24.6 · Abs 31.8/25.6 · SiLU 32.3/26.4.
Teacher correct: ReLU 0.70× · Tanh 0.65× · Abs 0.71× · SiLU 0.87× (all <1× = slower than eager); Sigmoid/GeLU wrong.

**Findings:**
1. Speed: **+26% tok/s, 11× fewer params, 12× less VRAM** — stable across all ops (densification win).
2. Correctness k=1 is **high-variance**: ours 0/6 here but **2/5** in the variety test (Tanh 0.92×, ReLU). True rate ≈ 1–2/6 at k=1.
3. **Neither beats PyTorch eager** — teacher's best is 0.87×. `fast_1` ≈ 0 for both on these memory-bound ops.
4. **Over-generation ⇒ failure**: our Sigmoid used full 1024 tok and failed; teacher's correct ops were shorter. → justify repetition-penalty / stop-on-``` / Dr.GRPO.

## Baseline B — Variety test (ours, k=1, CUDA + Triton)
CUDA (run-tested): Tanh ✅0.92×, ReLU ✅(earlier), GeLU compiled-but-wrong, Sigmoid/Softmax fail.
Triton (clean prompt): ReLU/GeLU/Softmax all emit valid `@triton.jit` (structurally right, buggy idioms).
**Finding:** both languages work; CUDA stronger than Triton (SFT was CUDA-only). Triton needs SFT data or RFT.

## Baseline C — SFT KernelBench-Lite (ReLU, best-of-4)
ReLU compile+correct ✅ 0.93×. fast_0 ≥1/op achievable with k=4.

---

## Planned inference sweeps (no retraining) — TODO
- [ ] pass@k: k ∈ {1,2,4,8} over 20 ops → fast_0(k) curve
- [ ] repetition_penalty {1.0,1.1,1.2} + stop-on-``` → failure-rate vs length
- [ ] enable_thinking True vs False → complex-op correctness
- [ ] temperature {0.2,0.4,0.6,0.8,1.0} → correctness vs diversity
- [ ] prompt: SFT-format vs plain; CUDA-only vs Triton-only system

## Training-side variants (retraining — separate, later)
- K experts (8 → 16/32), dense FFN width, hybrid partial densification, +reasoning SFT data, RFT.

*Refs: RADLADS 2505.03005 · MoE→Dense 2605.28207 · Dr.GRPO · DAPO · robust-kbench 2509.14279 · KernelBench.*
