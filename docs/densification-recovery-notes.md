# Densification Recovery — Working Notes (Jessica)

Running log of the team's evolving plan + my piece. Updated from the team Discord.

## Big picture
Laguna **33B MoE → dense ~3B**, then recover quality by training.
Stages (gist): warm-start (DO-ACP) → reconstruction (MSE) → SFT → online KL.

## 🔑 KEY DISTINCTION: agentic vs single-shot model
cm2435's target is an **agentic coding model** that only operates in **tool-call format**:
- Its "coding" lives *inside* the agentic policy — there's no standalone code-gen number.
- **HumanEval ≈ 0 for it** = *distribution mismatch* (it expects to act as an agent), **not** a coding deficit.
- So an agentically-trained model can't be measured by single-shot benchmarks.

**My variant is different:** I SFT on **single-shot** code (plain instruction→code), so my model
*can* be measured on HumanEval. Same pipeline, different target distribution.

## The team's (cm2435's) agentic path
1. **Agent harness** — prompt → tool call → compile → test → feedback loop
2. Run **teacher (MoE Laguna)** through harness → collect **tool-calling trajectories** = SFT data
3. **SFT** student on those trajectories
4. **Sanity-check** — run SFT'd student through harness: valid (non-gibberish) tool calls?
5. → **online KL** (on-policy distillation)

The harness (step 1) is the gating piece — needed for **both** the data (step 2) and the eval.

## 🔑 KEY LEARNING: agentic data needs failure→recovery
cm2435's agent is underbaked because his gold data (from Claude) is **too clean**:
- Claude never e.g. uses a wrong file path, so the model never learned to **recover** from mistakes.
- Need trajectories that **start wrong, then recover** (e.g. bad path → `ls -a | grep <thing>`), not
  just negative samples. (Evan/cm2435 musing: "metacognition" — mask the recovery-thinking turn.)

## 🔑 KEY LEARNING: KernelBench/SWE-bench need a harness
You **can't** score KernelBench or agentic SWE-bench without an agent harness — same blocker as
the SFT data. Single-shot benchmarks (HumanEval) don't need one.

## My piece (Jessica) — single-shot recovery variant
- **SFT on OpenCodeInstruct** (general Python, single-shot) → a single-shot code model.
- **Eval on HumanEval** (single-shot, ~164 problems + unit tests, **no harness**).
- Story: "densified Laguna 33B→3B, recovered single-shot code-gen X%→Y% on HumanEval."
- Tractable solo on limited credits. Distinct from cm2435's agentic model (don't conflate).

### Status / open bug
- Pipeline **trains** (run1: loss 1.6→0.5 over 425 steps, then OOM at batch 2-4 → fix = batch 1).
- Current chat-template script (`scripts/train_dense_sft.py`) yields **ZERO batches** → no training
  (loop never runs, just re-saves the model). Diagnosing: dataset field names vs `row_to_messages`,
  or `apply_chat_template` failing silently (the `try/except` hides it). **← next thing to fix.**

## Model loading gotcha (solved)
`EvanOLeary/laguna-xs2-dense-k8-recon` was uploaded **without its modeling `.py` files** → won't
load via `trust_remote_code`. Fix: copied `configuration_laguna_dense.py` + `modeling_laguna_dense.py`
from `cm2435-new/laguna-xs2-dense-k8-copied-shell` into `./recon_model`. Now loads clean (~3B).
Laguna chat template = `<system>` (Poolside prompt) + `<user>` + `<assistant>` with a `</think>` tag.

## Datasets seen
- `nvidia/OpenCodeInstruct` — general Python instruction→code. **My data (single-shot, HumanEval-aligned).**
- `GPUMODE/KernelBook` — 18k pytorch→triton pairs (`python_code`/`triton_code`). For KernelBench (needs harness).
- `SakanaAI/AI-CUDA-Engineer-Archive` — CUDA, non-agentic (Evan).
- Reference model: `dhaya98/gpt-oss-20b-cuda-sft`.

## Open decisions
- Is my single-shot variant useful to the team, or should I plug into the agentic pipeline (own step 3 SFT)?
- My eval = HumanEval (single-shot). Team's = KernelBench/SWE-bench (agentic, harness-gated).

## Infra
- My GPU: A100 80GB on Prime Intellect (`ubuntu@`). Limited credits — finish ONE clean run, don't flail.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--batch-size 1` to avoid OOM. Terminate when idle.
