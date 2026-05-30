# Mid/Post-Training RFCs

**Status:** Active planning.

This folder owns the training stages after dense-layer reconstruction.

## Version Scope

Use this vocabulary consistently:

```text
mid-training:
  teacher Laguna rollouts -> assistant/tool-call masked SFT
  goal: recover chat/tool/action format after MoE-to-dense reconstruction

post-training v1:
  one-step online/on-policy KD
  saved rollout states -> dense student generates one assistant action
  Laguna teacher scores that exact student-generated action
  dense student trains on teacher/student divergence over generated action tokens
```

These are the only two training lanes described in this folder.

## Pipeline Stages

The stages are:

```text
Stage A: rollout generation with teacher Laguna via vLLM
Stage B: teacher rollout -> assistant-action SFT dataset
Stage C: dense student mid-training with SFT
Stage D: rollout states -> one-step online/on-policy KD prompts
Stage E: dense student post-training with online/on-policy KD
```

The key design rule is to keep expensive generation and expensive training separable where possible, while preserving the on-policy property for KD:

```text
SFT rollouts can be generated ahead of time
SFT can train from frozen JSONL
online/on-policy KD must generate student actions from the current or checkpointed student
teacher must score the exact student-generated action, not regenerate its own answer
```

## RFC Index

- [001: Rollout To SFT Pipeline](001-rollout-to-sft-pipeline.md)
- [002: Online On-Policy KD Pipeline](002-on-policy-distillation-pipeline.md)
- [003: Overnight Job Orchestration](003-overnight-job-orchestration.md)
- [004: Metacognitive Recovery Data For Dense Student Policy Repair](004-metacognitive-recovery-data.md)

## Current Readiness

```text
reconstruction pretraining: running / partially implemented
rollout generation harness: implemented, needs clean post-fix pilot
rollout -> assistant-action SFT converter: implemented as scaffold, needs real-run validation
SFT trainer for dense model: implemented as scaffold, needs GPU validation
online/on-policy KD trainer: not implemented
```

## Near-Term Target

Mid-training SFT:

```bash
bash scripts/nightly_rollout_to_sft.sh
```

should produce:

```text
data/rollouts/<run_id>/
data/sft/rollout_sft_<run_id>.jsonl
runs/sft/<run_id>/
checkpoints/<dense_student_sft_run>/
```

Post-training v1:

```bash
bash scripts/nightly_online_kd.sh
```

should produce:

```text
data/on_policy_prompts/<run_id>.jsonl
runs/online_kd/<run_id>/
checkpoints/<dense_student_online_kd_run>/
```
