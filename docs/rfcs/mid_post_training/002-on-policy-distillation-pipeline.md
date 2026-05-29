# RFC 002: On-Policy Distillation Pipeline

**Status:** Draft.

## Purpose

Train the dense student on its own action distribution using the true Laguna MoE model as teacher.

This is the post-training stage after reconstruction and rollout-SFT:

```text
student acts
teacher corrects / scores / demonstrates
student trains on teacher feedback for states the student actually visits
```

This matters because sparse-to-dense errors compound. Offline SFT only teaches the student the teacher trajectory distribution; on-policy distillation teaches recovery from the dense model's own mistakes.

## Minimum Viable Version

Use teacher demonstrations on student-reached states.

For each task:

```text
1. run student in the coding harness for N turns
2. at selected states, ask teacher for the next assistant/tool action
3. save (student context, teacher next action)
4. SFT student on those teacher actions
```

This is not full RL. It is on-policy DAgger-style imitation, which is much easier to land overnight.

## Optional KL Version

If logits are available cheaply:

```text
student context -> teacher logits
student context -> student logits
loss = KL(teacher || student) on target assistant tokens
```

Likely implementation choices:

```text
HF teacher logits: expensive, simple, exact
vLLM top-logprobs: cheaper, approximate, only top-k
teacher text only: easiest, no KL
```

Recommendation:

```text
start with teacher text actions
add KL only after the data loop works
```

## Data Artifacts

Student rollouts:

```text
data/on_policy/<run_id>/student_rollouts/
```

Teacher corrections:

```text
data/on_policy/<run_id>/teacher_corrections.jsonl
```

Rows:

```json
{
  "id": "sympy__sympy-11618:student_turn_0012:teacher",
  "task_id": "sympy__sympy-11618",
  "student_checkpoint": "cm2435/laguna-xs2-dense-k8-sft-preview",
  "context_messages": [...],
  "student_action": {...},
  "teacher_action": {...},
  "selection_reason": "after_failed_tool_or_before_edit",
  "quality": "teacher_correction"
}
```

## State Selection

Do not ask the teacher at every turn in the first version. It is too expensive.

Select states where teacher signal is most valuable:

```text
first tool action
after failed shell command
before first edit
after first edit before verification
when student repeats a similar command
final turn if no exit
```

Budget:

```text
3-8 teacher corrections per task
20 tasks -> 60-160 teacher calls
```

## Training

Use the same SFT trainer as RFC 001, with a different dataset:

```bash
python scripts/train_dense_sft.py \
  --model cm2435/laguna-xs2-dense-k8-sft-preview \
  --dataset data/on_policy/<run_id>/teacher_corrections.jsonl \
  --output-dir runs/distill/<run_id> \
  --max-steps 500 \
  --lr 2e-5
```

Freeze policy:

```text
start with dense routed FFNs + norms
consider lm_head if tool-call syntax is poor
keep attention frozen
```

## Overnight Command Shape

Proposed script:

```text
scripts/nightly_on_policy_distill.sh
```

Command:

```bash
bash scripts/nightly_on_policy_distill.sh \
  --registry tasks/registry_balanced_100.jsonl \
  --limit 20 \
  --max-turns 60 \
  --student-model cm2435/laguna-xs2-dense-k8-sft-preview \
  --teacher-api-url http://127.0.0.1:8791/v1 \
  --student-api-url http://127.0.0.1:8793/v1 \
  --run-id $(date -u +%Y%m%dT%H%M%SZ)
```

Stages:

```text
1. ensure teacher endpoint is reachable
2. ensure student endpoint is reachable
3. run student rollouts
4. select correction states
5. query teacher on correction states
6. build teacher-correction SFT JSONL
7. train student
8. evaluate before/after on held-out tasks
```

## Metrics

Minimum:

```text
student self-exit rate before/after
bounded patch rate before/after
blocked-command rate before/after
turns to first edit before/after
tool-call JSON validity before/after
```

Better:

```text
teacher/student action agreement
teacher correction acceptance rate by state type
small held-out SWE-bench-lite pass/fail
```

## Risks

### Serving Two Large Models

Teacher MoE plus dense student may not fit simultaneously.

Fallback:

```text
run student rollouts first
stop student server
start teacher server
generate corrections offline
stop teacher
train student in HF/PyTorch
```

### Bad Student States

If the student context becomes nonsense, teacher corrections may be low value.

Mitigation:

```text
stop student rollouts after repeated invalid tool calls
sample early states more heavily
mix offline teacher SFT data with on-policy corrections
```

### Tool Format Drift

Student may produce invalid tool calls.

Mitigation:

```text
include tool-schema formatting examples in SFT
mask loss to assistant/tool-call JSON
evaluate JSON validity separately from task success
```

