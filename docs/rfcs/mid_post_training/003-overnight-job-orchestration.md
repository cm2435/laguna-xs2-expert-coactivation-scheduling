# RFC 003: Overnight Job Orchestration

**Status:** Draft.

## Purpose

Define orchestration for the two actual training jobs:

```text
mid-training: teacher rollouts -> SFT
post-training v1: prompt states -> online/on-policy KD
```

This RFC describes only those two jobs.

## Script 1: Rollout To SFT

Path:

```text
scripts/nightly_rollout_to_sft.sh
```

Responsibilities:

```text
start or verify teacher vLLM
prepare repo templates
run teacher rollouts
summarize rollouts
convert rollouts to assistant-action SFT JSONL
train dense SFT
run smoke eval
```

Log layout:

```text
runs/nightly/rollout_to_sft_<run_id>/
  command.sh
  env.txt
  logs/
    01_prepare_templates.log
    02_rollouts.log
    03_summarize.log
    04_build_sft.log
    05_train_sft.log
    06_eval.log
  artifacts.json
```

The script should write `artifacts.json`:

```json
{
  "run_id": "20260530T020000Z",
  "rollout_dir": "runs/coding_harness_...",
  "sandbox_dir": "sandboxes/coding_harness_...",
  "sft_jsonl": "data/sft/rollout_sft_....jsonl",
  "sft_run_dir": "runs/sft/...",
  "checkpoint_final": "runs/sft/.../checkpoint-final"
}
```

## Script 2: Online KD

Path:

```text
scripts/nightly_online_kd.sh
```

Responsibilities:

```text
build prompt-only state rows from rollout traces
run dense student one-step action generation
score generated actions with Laguna teacher
train dense student with online/on-policy KD
evaluate before/after on held-out tasks
```

Log layout:

```text
runs/nightly/online_kd_<run_id>/
  command.sh
  env.txt
  logs/
    01_build_prompts.log
    02_online_kd_smoke.log
    03_online_kd_train.log
    04_eval.log
  artifacts.json
```

## Required Python Scripts

For SFT:

```text
scripts/summarize_coding_rollouts.py
scripts/build_sft_from_rollouts.py
scripts/train_dense_sft.py
scripts/eval_dense_coding_smoke.py
```

For online KD:

```text
scripts/build_on_policy_prompts.py
scripts/train_dense_online_kd.py
scripts/eval_dense_coding_smoke.py
```

## Failure Behavior

The scripts should be resume-friendly.

Rules:

```text
never overwrite an existing run directory
write every stage output before moving on
if training fails, keep rollouts / prompts / checkpoints
if rollout generation fails, keep partial traces but mark run incomplete
write a final status.json with success=false
```

`status.json`:

```json
{
  "success": false,
  "failed_stage": "online_kd_train",
  "message": "CUDA out of memory",
  "completed_stages": ["build_prompts", "online_kd_smoke"]
}
```

## Resource Modes

### Mode A: One GPU Sequential

Use when VRAM is scarce.

```text
teacher vLLM -> SFT rollouts -> stop teacher -> HF SFT training -> online KD with local/sequential teacher scoring
```

### Mode B: Two GPU Parallel

Use if available.

```text
GPU 0: Laguna teacher scoring server or local teacher
GPU 1: dense student online KD training
```

### Mode C: Two VMs

Useful for teacher scoring.

```text
VM A: Laguna teacher scoring server
VM B: dense student training
```

## Initial Defaults

Rollout-to-SFT:

```text
limit: 80
validation_limit: 20
max_turns: 15
temperature: 0.0
seq_len: 8192
sft_max_steps: 500-1000
lr: 5e-5
```

Online KD:

```text
prompt_rows: 200-1000, depending on rollout supply
max_prompt_length: 4096-8192
max_completion_tokens: 128
batch_size: 1
kd_max_steps: 200-1000
lr: 1e-6 to 2e-5
beta: 1.0
temperature: 1.0
```

## Done Definition

This orchestration layer is ready when:

```text
one command creates rollouts, SFT JSONL, and an SFT checkpoint
one command creates prompt rows and an online-KD checkpoint
both scripts leave clear logs and artifact manifests
failed runs are resumable from saved artifacts
```
