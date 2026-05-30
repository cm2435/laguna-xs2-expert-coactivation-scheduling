# Engineering Plan 006: Larger SWE-bench Rollout Collection

**Status:** Active planning.

**Goal:** Increase rollout collection from the first 20-task / 20-turn smoke batch to a larger, more useful dataset with up to 100 turns per task, while controlling GPU spend and filtering low-value capped traces.

**Architecture:** Use the existing minimal coding harness against the local vLLM Laguna endpoint. Keep the model-serving path fixed, improve the rollout orchestration and summaries, then run staged batches rather than immediately launching a large expensive run.

**Tech Stack:** vLLM OpenAI-compatible endpoint, Laguna XS.2, `scripts/run_coding_swebench_batch.py`, task manifests in `tasks/registry.jsonl`, sandboxed repos in `sandboxes/`, rollout artifacts in `runs/`.

---

## Executive Summary

The first batch proved the harness works, but the model mostly hit the 20-turn cap:

```text
20 tasks completed
1 self-declared success
8 non-empty patches
19 capped at max_turns=20
0 tasks actually graded by SWE-bench
```

The next run should increase `max_turns` to 100, but not blindly. More turns are only valuable if the model uses them to inspect, patch, verify, and exit. If it loops for 100 turns, we get expensive low-signal traces.

The plan is:

1. Add stronger rollout summaries before scaling.
2. Run a 5-task pilot at 100 turns.
3. Inspect whether extra turns produce more patches / exits / verification.
4. If useful, run 20-50 tasks at 100 turns.
5. Keep all traces, but label them by outcome quality for training.

The immediate metric is not SWE-bench pass rate yet. The immediate metric is trace quality:

```text
non-empty patch rate
self-exit rate
focused test / verification rate
turns-to-first-edit
turns-to-final-exit
number of capped loops
```

Once public/hidden grading is wired, replace self-success with real grade success.

## Why Increase Max Turns?

The 20-turn cap is probably too low for realistic coding rollouts. In the successful scikit-learn trace, the model only reached `exit(success=true)` on turn 20. That means the cap may be truncating otherwise usable solves.

However, the failure mode is also clear: many traces spend all 20 turns without producing a patch. Raising to 100 helps only if we can distinguish:

```text
productive long rollouts: read -> locate -> patch -> test -> exit
unproductive long rollouts: repeated search / failed tests / no final patch
```

This plan adds enough instrumentation to make that distinction.

## Current Command Shape

The runner already supports the knobs we need:

```bash
uv run --no-sync python scripts/run_coding_swebench_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8791/v1 \
  --model laguna \
  --output-dir runs/coding_harness_rollouts_100t_pilot \
  --sandbox-root sandboxes/coding_harness_100t_pilot \
  --limit 5 \
  --max-turns 100 \
  --temperature 0.0
```

For the full run:

```bash
uv run --no-sync python scripts/run_coding_swebench_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8791/v1 \
  --model laguna \
  --output-dir runs/coding_harness_rollouts_100t_50 \
  --sandbox-root sandboxes/coding_harness_100t_50 \
  --limit 50 \
  --max-turns 100 \
  --temperature 0.0
```

If the registry still only contains 20 tasks, first regenerate a larger registry using the existing SWE-bench task selection path.

## Success Criteria

The 100-turn rollout plan is worth continuing if the 5-task pilot shows at least two of:

```text
self-exit rate improves over 20-turn batch
non-empty patch rate improves over 40%
at least one task exits before turn 100 with a plausible patch
model performs focused verification after editing
turns 21-100 contain meaningful new actions rather than loops
```

Abort or revise prompts/tools if:

```text
most tasks reach 100 turns with no patch
the model repeatedly runs blocked package installs
the model repeatedly reads/searches without editing
the traces are dominated by environment/setup failures
```

## Task 1: Add Rollout Quality Summary

**Files:**

- Modify: `src/densify/coding_harness/runner.py`
- Modify: `src/densify/tasks/coding_runner.py`
- Test: `tests/test_coding_harness.py`

**Why:** The existing summaries only tell us `success` and `turns`. For a 100-turn run, we need to know whether the extra turns were useful.

Add a post-run summary with these fields:

```json
{
  "success": false,
  "turns": 100,
  "finish_reason": "max_turns",
  "tool_call_count": 100,
  "tool_counts": {
    "shell": 52,
    "read_file": 35,
    "apply_patch": 1,
    "write_file": 0,
    "exit": 0
  },
  "first_edit_turn": 37,
  "first_test_turn": 43,
  "blocked_install_attempts": 0
}
```

Implementation notes:

- Count `apply_patch` and `write_file` as edits.
- Also count `shell` commands containing `python`, `pytest`, `tox`, `unittest`, or `./test` as verification attempts.
- Set `finish_reason` to one of:

```text
exit_success
exit_failure
max_turns
api_error
```

Acceptance test:

```bash
uv run pytest tests/test_coding_harness.py -q
```

Expected:

```text
all coding harness tests pass
summary.json includes finish_reason and tool_counts
```

## Task 2: Add Batch-Level Aggregation

**Files:**

- Modify: `scripts/run_coding_swebench_batch.py`
- Create: `scripts/summarize_coding_rollouts.py`
- Test: `tests/test_coding_harness.py` or a new focused summary test

**Why:** After a 50-task run, we need one command that answers “how did it do?” without manually opening traces.

Create a summarizer command:

```bash
uv run python scripts/summarize_coding_rollouts.py \
  --runs-dir runs/coding_harness_rollouts_100t_pilot \
  --sandboxes-dir sandboxes/coding_harness_100t_pilot \
  --output runs/coding_harness_rollouts_100t_pilot/aggregate_summary.json
```

The output should include:

```json
{
  "tasks": 5,
  "self_success": 1,
  "nonempty_patch": 3,
  "max_turn_capped": 2,
  "mean_turns": 78.2,
  "median_turns": 100,
  "mean_patch_bytes": 1240,
  "tasks_with_verification": 4,
  "tasks_with_first_edit_after_turn_20": 2
}
```

Also write a CSV/JSONL table with one row per task:

```text
task_id,success,turns,finish_reason,patch_bytes,first_edit_turn,first_test_turn,tool_call_count
```

This table is the artifact we inspect before deciding whether to scale.

## Task 3: Regenerate Or Extend The Task Registry

**Files:**

- Use: `configs/swebench_verified_20.yaml`
- Use: `scripts/build_swebench_task_manifests.py`
- Output: `tasks/registry.jsonl`
- Output: `tasks/swebench_verified/*.yaml`

**Why:** The current local tree may only have the 20-task registry. For a larger run, we need 50-100 tasks, preferably still Python-heavy and repo-diverse.

Recommended new config:

```yaml
dataset: princeton-nlp/SWE-bench_Verified
split: test
output_dir: tasks/swebench_verified
registry_path: tasks/registry.jsonl
target_total: 50
repos:
  astropy/astropy: 8
  django/django: 8
  pytest-dev/pytest: 8
  sympy/sympy: 8
  matplotlib/matplotlib: 8
  scikit-learn/scikit-learn: 10
```

Keep the first large run repo-diverse, but bias slightly toward scikit-learn because the first plausible solve came from there and the test/setup burden may be lower.

Run:

```bash
uv run python scripts/build_swebench_task_manifests.py \
  --config configs/swebench_verified_50.yaml
```

Then prepare templates:

```bash
uv run python scripts/prepare_repo_templates.py \
  --registry tasks/registry.jsonl
```

Acceptance:

```bash
wc -l tasks/registry.jsonl
```

Expected:

```text
50 tasks
```

## Task 4: Run A 5-Task / 100-Turn Pilot

**VM:** `prime-gpu-31`

**vLLM endpoint:**

```text
http://127.0.0.1:8791/v1
```

Before running, confirm the endpoint:

```bash
curl -s http://127.0.0.1:8791/v1/models | jq .
```

Run:

```bash
uv run --no-sync python scripts/run_coding_swebench_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8791/v1 \
  --model laguna \
  --output-dir runs/coding_harness_rollouts_100t_pilot \
  --sandbox-root sandboxes/coding_harness_100t_pilot \
  --limit 5 \
  --max-turns 100 \
  --temperature 0.0
```

Then summarize:

```bash
uv run python scripts/summarize_coding_rollouts.py \
  --runs-dir runs/coding_harness_rollouts_100t_pilot \
  --sandboxes-dir sandboxes/coding_harness_100t_pilot \
  --output runs/coding_harness_rollouts_100t_pilot/aggregate_summary.json
```

Manual review:

```bash
find sandboxes/coding_harness_100t_pilot -name patch.diff -size +0 -print
```

Open the non-empty patches first. For each, decide:

```text
plausible solve
partial but useful trace
bad/noisy trace
```

## Task 5: Decide Whether To Scale

Proceed to 20-50 tasks at 100 turns if the pilot has useful traces.

Use this decision table:

| Pilot result | Next action |
| --- | --- |
| 0/5 patches, mostly capped | Stop. Improve prompt/tooling. |
| 1-2 plausible patches | Run 20 tasks at 100 turns. |
| 3+ plausible patches or exits | Run 50 tasks at 100 turns. |
| Repeated loops after turn 40 | Add loop detection or lower cap to 60. |
| Repeated test/env failures | Add repo-specific smoke-test guidance. |

Recommended next command if pilot is healthy:

```bash
uv run --no-sync python scripts/run_coding_swebench_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8791/v1 \
  --model laguna \
  --output-dir runs/coding_harness_rollouts_100t_50 \
  --sandbox-root sandboxes/coding_harness_100t_50 \
  --limit 50 \
  --max-turns 100 \
  --temperature 0.0
```

## Task 6: Label The Dataset For Training Use

Do not treat all traces equally.

Label each rollout as:

```text
gold: real graded pass, or manually verified plausible patch
silver: non-empty patch with coherent reasoning but incomplete verification
bronze: useful exploration with no final patch
reject: loops, package-install attempts, malformed tool use, no useful inspection
```

The immediate batch will mostly produce silver/bronze. That is still useful for:

```text
tool-use format preservation
patch-attempt behavior
teacher trace mining
failure-mode analysis
```

It is not yet clean enough for final “successful solve” SFT without filtering.

## Risks

### Cost / GPU Time

A 100-turn rollout can be roughly 5x the first batch. If 20 tasks took N minutes, 50 tasks at 100 turns can be about 12.5x that wall-clock unless many tasks exit early.

Mitigation:

```text
run 5-task pilot first
inspect summaries
scale only if extra turns are productive
```

### Low-Signal Long Traces

The model may loop or keep reading without editing.

Mitigation:

```text
track first_edit_turn
track repeated shell/read calls
cap at 60 if 100 mostly loops
add prompt pressure to patch and exit
```

### No Real Grading Yet

Self-declared success is not SWE-bench success.

Mitigation:

```text
keep grade_status explicit
do not report pass rate until grader runs
use patch plausibility / verification as interim metrics
```

### Dirty Repo Noise

The first run had `repo_dirty=true` on every task because generated files polluted sandbox state.

Mitigation:

```text
use patch.diff size and git diff over tracked source files as the main patch signal
exclude harness metadata from grade dirty checks
```

## Recommended Immediate Sequence

Run these in order:

```text
1. Add rollout quality summaries.
2. Add aggregate summarizer.
3. Run 5 tasks at 100 turns.
4. Read the 5 traces manually.
5. If healthy, launch 20 or 50 tasks at 100 turns.
```

Do not start with 50-100 tasks at 100 turns until the pilot shows the model uses the extra turns productively.

