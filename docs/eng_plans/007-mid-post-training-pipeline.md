# Mid/Post-Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two separate training stages: SFT mid-training from Laguna teacher rollouts, then v1 post-training with one-step online/on-policy KD from dense-student-generated actions.

**Architecture:** vLLM produces teacher rollouts for SFT. The SFT converter emits one row per assistant action and masks loss to the final assistant message. Online KD later uses prompt-only rollout states: the dense student generates one assistant action, Laguna scores that exact generated action, and the dense student trains on teacher/student divergence over generated action tokens.

**Tech Stack:** Python, PyTorch, Transformers, vLLM OpenAI-compatible endpoints, TRL implementation as reference, existing custom coding harness, SWE-bench Verified registries, JSONL artifacts.

---

## Current State

Already available:

```text
scripts/run_coding_swebench_batch.py
scripts/run_coding_swebench_rollout.py
scripts/train_dense_reconstruction.py
scripts/summarize_coding_rollouts.py
scripts/build_sft_from_rollouts.py
scripts/train_dense_sft.py
src/densify/coding_harness/
src/densify/tasks/
src/densify/rollout_sft/
tasks/registry_balanced_100_train80.jsonl
tasks/registry_balanced_100_val20.jsonl
repos/trl/
```

Not yet available:

```text
scripts/build_on_policy_prompts.py
scripts/train_dense_online_kd.py
scripts/nightly_online_kd.sh
validated GPU SFT run
validated online-KD smoke run
validated held-out eval report
```

## Target Now

While rollouts or reconstruction jobs are running, implement CPU-testable data plumbing for the actual plan:

```text
SFT dataset validation
prompt-only on-policy dataset builder
online-KD collator/loss utilities
training script skeleton with dry-run / fake-model tests
```

Do not implement extra distillation variants in this plan. The next coding work should target the SFT lane or the online/on-policy KD lane.

## Version Vocabulary

```text
mid-training:
  SFT behavior cloning from teacher Laguna assistant actions

post-training v1:
  one-step online/on-policy KD
  dense student generates the action
  Laguna teacher scores that exact generated action
```

## Task 1: Validate SFT Dataset

**Files:**

- Create: `scripts/validate_sft_dataset.py`
- Test: `tests/test_validate_sft_dataset.py`

Checks:

```text
row count
all rows end with assistant
no rows end with tool
active Laguna chat-template rendering
target token length histogram
tool-call target count
empty assistant target count
quality distribution
max rendered length
first trainable token preview
```

Done when:

```text
uv run --no-sync pytest tests/test_validate_sft_dataset.py -q
python scripts/validate_sft_dataset.py --input data/sft/<file>.jsonl --output runs/validation/<run_id>.json
```

## Task 2: Build Prompt-Only On-Policy Rows

**Files:**

- Create: `src/densify/on_policy/prompts.py`
- Create: `scripts/build_on_policy_prompts.py`
- Test: `tests/test_on_policy_prompts.py`

Behavior:

```text
input: reconstructed rollout messages
output: one prompt-only row for each assistant-action boundary
row messages end before the assistant action
tool observations may appear in context
no final assistant target is included
```

Example:

```text
system, user, assistant, tool, assistant, tool
```

emits:

```text
prompt 1: system, user
prompt 2: system, user, assistant, tool
```

Done when tests prove prompt rows never include the action they are meant to elicit.

## Task 3: Online-KD Collator

**Files:**

- Create: `src/densify/on_policy/collate.py`
- Test: `tests/test_online_kd_collator.py`

Responsibilities:

```text
tokenize prompt-only messages
apply chat template with assistant generation prompt if available
track prompt token lengths
left/right pad consistently
produce prompt input_ids and prompt_attention_mask
```

This should mirror TRL's `_DistillationCollator` idea, but keep our implementation small.

## Task 4: Online-KD Generation And Labels

**Files:**

- Create: `src/densify/on_policy/generation.py`
- Test: `tests/test_online_kd_generation.py`

Responsibilities:

```text
student.generate(prompt_ids)
build full input_ids = prompt_ids + completion_ids
build attention_mask
build labels = -100 on prompt tokens, completion ids on generated tokens
record completion lengths
record truncated completion fraction
```

This mirrors TRL's `_build_sequence_batch`.

## Task 5: Online-KD Loss

**Files:**

- Create: `src/densify/on_policy/loss.py`
- Test: `tests/test_online_kd_loss.py`

Responsibilities:

```text
student forward on prompt + student completion
teacher forward on same prompt + student completion
slice logits to completion positions
compute generalized JSD / KL over labels != -100
```

Start with full-vocab local teacher loss. Add sparse/server scoring only if VRAM forces it.

Initial defaults:

```text
beta: 1.0
temperature: 1.0
max_completion_tokens: 128
batch_size: 1
```

## Task 6: Online-KD Trainer Script

**Files:**

- Create: `scripts/train_dense_online_kd.py`
- Test: `tests/test_train_dense_online_kd_utils.py`

Script shape:

```bash
python scripts/train_dense_online_kd.py \
  --student-model /path/to/dense-sft-checkpoint \
  --teacher-model poolside/Laguna-XS.2 \
  --prompts data/on_policy_prompts/<run_id>.jsonl \
  --output-dir runs/online_kd/<run_id> \
  --max-steps 500 \
  --max-prompt-length 8192 \
  --max-completion-tokens 128 \
  --lr 1e-6 \
  --beta 1.0 \
  --temperature 1.0
```

Metrics per log step:

```text
loss
completion_tokens
completion_mean_length
truncated_completion_fraction
tokens_per_second
peak_cuda_memory_gb
```

## Task 7: Overnight Online-KD Script

**Files:**

- Create: `scripts/nightly_online_kd.sh`

Stages:

```text
build prompt-only rows
run a 1-5 batch online-KD smoke
run online-KD training
run held-out coding smoke eval
write artifacts.json and status.json
```

## First Real Commands

SFT:

```bash
bash scripts/nightly_rollout_to_sft.sh \
  --registry tasks/registry_balanced_100_train80.jsonl \
  --validation-registry tasks/registry_balanced_100_val20.jsonl \
  --limit 80 \
  --max-turns 15 \
  --teacher-api-url http://127.0.0.1:8791/v1 \
  --student-model /path/to/pretrained-dense-checkpoint \
  --seq-len 8192 \
  --disable-thinking \
  --run-id $(date -u +%Y%m%dT%H%M%SZ)
```

Online KD:

```bash
bash scripts/nightly_online_kd.sh \
  --runs-dir runs/coding_harness_<sft_run_id> \
  --student-model runs/sft/<sft_run_id>/checkpoint-final \
  --teacher-model poolside/Laguna-XS.2 \
  --max-steps 500 \
  --run-id $(date -u +%Y%m%dT%H%M%SZ)
```

## Validation Before Claiming Done

Run:

```bash
uv run --no-sync pytest \
  tests/test_rollout_summary.py \
  tests/test_build_sft_from_rollouts.py \
  tests/test_sft_tokenize.py \
  tests/test_train_dense_sft_utils.py \
  tests/test_on_policy_prompts.py \
  tests/test_online_kd_collator.py \
  tests/test_online_kd_generation.py \
  tests/test_online_kd_loss.py \
  tests/test_train_dense_online_kd_utils.py \
  -q
```
