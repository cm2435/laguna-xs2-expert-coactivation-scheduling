# RFC 003: Overnight Job Orchestration

**Status:** Draft.

## Purpose

Define bash-level orchestration for overnight jobs so rollout collection, SFT, and on-policy distillation can run without manual babysitting.

The scripts should be thin wrappers around Python modules. Bash owns process order and logging; Python owns data validation and training logic.

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
convert rollouts to SFT JSONL
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
  "run_id": "20260529T230000Z",
  "rollout_dir": "runs/coding_harness_...",
  "sandbox_dir": "sandboxes/coding_harness_...",
  "sft_jsonl": "data/sft/rollout_sft_....jsonl",
  "sft_run_dir": "runs/sft/...",
  "checkpoint_final": "runs/sft/.../checkpoint-final"
}
```

## Script 2: On-Policy Distillation

Path:

```text
scripts/nightly_on_policy_distill.sh
```

Responsibilities:

```text
run student rollouts
select correction states
query teacher for corrections
build correction SFT JSONL
train dense student
evaluate before/after
```

Log layout:

```text
runs/nightly/on_policy_distill_<run_id>/
  command.sh
  env.txt
  logs/
    01_student_rollouts.log
    02_select_states.log
    03_teacher_corrections.log
    04_train.log
    05_eval.log
  artifacts.json
```

## Required Python Scripts

For rollout-to-SFT:

```text
scripts/summarize_coding_rollouts.py
scripts/build_sft_from_rollouts.py
scripts/train_dense_sft.py
scripts/eval_dense_coding_smoke.py
```

For on-policy:

```text
scripts/run_student_rollouts.py
scripts/select_on_policy_states.py
scripts/query_teacher_corrections.py
scripts/train_dense_sft.py
scripts/eval_dense_coding_smoke.py
```

Reuse `train_dense_sft.py` for both stages.

## Failure Behavior

The scripts should be resume-friendly.

Rules:

```text
never overwrite an existing run directory
write every stage output before moving on
if training fails, keep rollouts and SFT JSONL
if rollout generation fails, keep partial traces but mark run incomplete
write a final status.json with success=false
```

`status.json`:

```json
{
  "success": false,
  "failed_stage": "train_sft",
  "message": "CUDA out of memory",
  "completed_stages": ["prepare_templates", "rollouts", "build_sft"]
}
```

## Resource Modes

### Mode A: One GPU Sequential

Use when VRAM is scarce.

```text
teacher vLLM -> rollouts -> stop teacher -> HF SFT training
```

### Mode B: Two GPU Parallel

Use if we have enough hardware.

```text
GPU 0: teacher vLLM
GPU 1: student training or student serving
```

### Mode C: Two VMs

Useful for on-policy.

```text
VM A: teacher correction generation
VM B: student training
```

## Initial Defaults

Rollout-to-SFT:

```text
limit: 20
max_turns: 100
temperature: 0.0
seq_len: 4096
sft_max_steps: 500
lr: 5e-5
```

On-policy:

```text
limit: 20
student_max_turns: 60
teacher_corrections_per_task: 5
distill_max_steps: 500
lr: 2e-5
```

## Done Definition

This orchestration layer is ready when:

```text
one command creates rollouts, SFT JSONL, and a checkpoint
one command creates student rollouts, teacher corrections, and a distilled checkpoint
both scripts leave clear logs and artifact manifests
failed runs are resumable from saved artifacts
```

