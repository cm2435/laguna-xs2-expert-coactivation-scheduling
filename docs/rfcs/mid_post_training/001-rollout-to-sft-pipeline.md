# RFC 001: Rollout To SFT Pipeline

**Status:** Draft.

## Purpose

Turn teacher-generated coding-agent rollouts into a supervised fine-tuning dataset for the dense Laguna student.

The goal is not to create perfect SWE-bench solutions immediately. The goal is to recover the behavior shape that reconstruction training does not teach:

```text
follow the chat/tool-call format
inspect files
run focused commands
apply small patches
stop with an exit tool
avoid package installs and branch switching
```

This RFC is SFT only. Post-training online/on-policy KD is owned by RFC 002.

## Inputs

Teacher rollouts from the custom coding harness:

```text
runs/coding_harness_*/{run_id}/
  requests/turn_0001.json
  responses/turn_0001.json
  model_turns.jsonl
  tool_calls.jsonl
  summary.json
  rollout_summary.json

sandboxes/coding_harness_*/{run_id}/
  repo/
  patch.diff
  grade_result.json
```

Recommended source sequence:

```text
runs/coding_harness_balanced_100t_clean_pilot5_rerun
runs/coding_harness_balanced_100t_clean_20
runs/coding_harness_balanced_100t_clean_100
```

## Outputs

Write one JSONL row per assistant action:

```text
data/sft/rollout_sft_<source_run_id>.jsonl
```

Each row must end with an assistant message. Tool observations may appear in the context, but must never be the target:

```json
{
  "id": "django__django-10097:assistant_0027",
  "task_id": "django__django-10097",
  "source_rollout": "runs/coding_harness_balanced_100t_clean_20/...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Repository root: ...\n\nTask:\n..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...]}
  ],
  "quality": "silver",
  "weight": 1.0,
  "metadata": {
    "success": true,
    "turns": 66,
    "patch_bytes": 1412,
    "assistant_index": 27,
    "target_role": "assistant"
  }
}
```

For a rollout:

```text
system, user, assistant, tool, assistant, tool
```

emit:

```text
row 1: system, user, assistant
row 2: system, user, assistant, tool, assistant
```

## Filtering

Do not train equally on every trace.

Labels:

```text
gold: real graded pass or manually verified strong patch
silver: self-success + bounded patch + coherent verification
bronze: useful exploration but no final patch
reject: no useful inspection, giant patch, dirty-template artifact, repeated blocked commands
```

First-pass automatic filters:

```text
reject if patch_bytes > 200_000
reject if patch contains many /dev/null deletions
reject if no tool_calls
reject if repeated blocked command count > 5
silver if success=true and 0 < patch_bytes < 200_000
bronze if patch_bytes == 0 but read_file/shell exploration happened
```

Manual review can promote/demote.

## Loss Masking

Train on assistant outputs only.

For each row:

```text
context = all messages before final assistant
target = final assistant message, including tool call JSON
loss mask = target assistant tokens only
```

Tokenization must use the Laguna tokenizer's chat template as the source of truth:

```python
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=...,
    tokenize=False,
    enable_thinking=...,
)
```

Do not hand-write Laguna tags for training unless the tokenizer has no chat
template. The manual renderer is only a fallback/debug comparison. After
rendering with `apply_chat_template(..., tokenize=False)`, tokenize with
`add_special_tokens=False`; the chat template has already placed the control
tokens.

For rollout SFT rows that do not contain real thinking traces, run with
`--disable-thinking`. With Laguna's template, `enable_thinking=True` makes the
prompt end at `<think>` and the trainable target begins with `</think>`. That is
valid if we intentionally want to teach "close thinking immediately", but it is
not the cleanest behavior-cloning setup for no-thinking gold rollouts.

Do not compute loss on:

```text
repository observations
tool outputs
problem statement
system prompt
previous messages
```

This matters because observations can be huge and deterministic; training on them wastes budget and teaches the model to imitate the environment rather than act in it.

## SFT Trainer

The trainer should be separate from rollout collection.

Proposed script:

```text
scripts/train_dense_sft.py
```

Inputs:

```bash
python scripts/train_dense_sft.py \
  --model cm2435/laguna-xs2-dense-k8-reconstruction \
  --dataset data/sft/rollout_sft_<run_id>.jsonl \
  --output-dir runs/sft/<run_id> \
  --max-steps 1000 \
  --seq-len 8192 \
  --lr 5e-5 \
  --disable-thinking
```

Freeze policy for first run:

```text
train dense routed FFNs
optionally train norms
freeze attention
freeze embeddings
freeze lm_head initially
```

If the model cannot follow the tool schema:

```text
unfreeze lm_head and final norm
lower lr
mix in simple chat/tool formatting examples
```

## Overnight Command Shape

The first full pipeline should be:

```bash
bash scripts/nightly_rollout_to_sft.sh \
  --registry tasks/registry_balanced_100_train80.jsonl \
  --validation-registry tasks/registry_balanced_100_val20.jsonl \
  --limit 80 \
  --max-turns 15 \
  --teacher-api-url http://127.0.0.1:8791/v1 \
  --student-model cm2435/laguna-xs2-dense-k8-reconstruction \
  --run-id $(date -u +%Y%m%dT%H%M%SZ)
```

Stages inside the script:

```text
1. ensure vLLM teacher endpoint is reachable
2. prepare repo templates for selected tasks
3. run teacher rollouts
4. summarize and filter rollouts
5. build assistant-action SFT JSONL
6. run dense SFT
7. run tiny held-out harness eval if time remains
```

## Success Criteria

A first useful run produces:

```text
80 teacher rollouts
>= 5 bounded non-empty patches
SFT JSONL with assistant/tool-call targets
one dense SFT checkpoint
before/after qualitative comparison on 20 held-out prompts/tasks
```
