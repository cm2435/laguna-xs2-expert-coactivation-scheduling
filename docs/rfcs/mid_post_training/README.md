# Mid/Post-Training RFCs

**Status:** Active planning.

This folder owns the training stages after dense-layer reconstruction:

```text
Stage A: rollout generation with teacher Laguna via vLLM
Stage B: rollout -> SFT dataset conversion
Stage C: dense student mid-training / behavior recovery SFT
Stage D: on-policy teacher/student distillation
```

The key design rule is to keep expensive generation and expensive training separable:

```text
rollout collection can run overnight with vLLM
SFT can run overnight from frozen JSONL data
on-policy distillation can run overnight as a separate longest job
```

## RFC Index

- [001: Rollout To SFT Pipeline](001-rollout-to-sft-pipeline.md)
- [002: On-Policy Distillation Pipeline](002-on-policy-distillation-pipeline.md)
- [003: Overnight Job Orchestration](003-overnight-job-orchestration.md)

## Current Readiness

```text
reconstruction pretraining: partially implemented
rollout generation harness: implemented, needs clean post-fix pilot
rollout -> SFT converter: not implemented
SFT trainer for dense model: not implemented
on-policy distillation: not implemented
```

## Near-Term Target

Within one night, we want to be able to run:

```bash
bash scripts/nightly_rollout_to_sft.sh
```

and produce:

```text
data/rollouts/<run_id>/
data/sft/rollout_sft_<run_id>.jsonl
runs/sft/<run_id>/
checkpoints/<dense_student_sft_run>/
```

Separately, once the student checkpoint is usable:

```bash
bash scripts/nightly_on_policy_distill.sh
```

and produce:

```text
data/on_policy/<run_id>/
runs/distill/<run_id>/
checkpoints/<dense_student_distilled_run>/
```

