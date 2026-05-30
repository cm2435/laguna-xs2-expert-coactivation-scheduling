# RFC 002: Online On-Policy KD Pipeline

**Status:** Draft.

## Purpose

Define the post-training stage after SFT: one-step online/on-policy knowledge distillation from the true Laguna MoE teacher into the dense student.

The key property is:

```text
student generates the action being trained on
teacher scores that exact student-generated action
student updates against the teacher distribution on those generated tokens
```

This follows the TRL GKD / DistillationTrainer control flow:

```text
prompt/state x
student samples y_student ~ p_student(. | x)
teacher scores p_teacher(. | x, y_student_<t)
student optimizes KL/JSD on y_student tokens
```

This is not RL: no scalar reward model, no advantage estimates, no PPO/GRPO loop.

## Inputs

Use saved rollout states as prompts. Each row should contain context only, with no final assistant target:

```json
{
  "id": "django__django-10097:state_0004",
  "task_id": "django__django-10097",
  "source_rollout": "runs/coding_harness_...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "content": "..."}
  ],
  "metadata": {
    "state_reason": "after_first_observation",
    "turn": 4
  }
}
```

The prompt builder may draw states from teacher rollouts or from student rollouts. For the first v1 implementation, use saved teacher-rollout states as prompts but generate the next action with the dense student during training. This preserves the on-policy action distribution without requiring a full interactive harness inside the trainer.

## Boundary

Avoid these patterns in v1:

```text
teacher rollout action -> teacher scores teacher action -> student trains on that action
teacher regenerates its own answer from the context
student trains only on teacher text corrections
```

Those are not the v1 post-training method.

## v1 Training Loop

For each training batch:

```text
1. collate prompt-only message rows
2. tokenize prompts with an assistant generation prompt
3. dense student generates one assistant action
4. build input_ids = prompt_tokens + student_generated_tokens
5. labels = -100 on prompt tokens, generated token ids on student action tokens
6. run dense student forward on input_ids
7. run Laguna teacher scoring on the same input_ids
8. compute KL/JSD on generated action tokens only
9. optimizer step on the dense student
```

This is one-step on-policy KD. The student action is on-policy; the environment is not stepped inside the trainer.

## Teacher Scoring

Preferred scoring path:

```text
HF/PyTorch teacher forward pass if teacher + student fit sequentially or with CPU/offload
```

Fallback scoring path:

```text
teacher vLLM server endpoint that scores existing token sequences
```

The server endpoint must score supplied token sequences. It must not use chat completion replay. The correct API shape mirrors TRL's `get_sequence_logprobs`:

```json
{
  "sequences": [[... prompt + student_completion token ids ...]],
  "prompt_lengths": [1234],
  "top_logprobs": 1,
  "temperature": 1.0
}
```

Required response fields:

```text
actual_logprobs: teacher log p(actual generated token) per completion position
topk_logprobs: teacher top-k logprobs per completion position
topk_token_ids: token ids for teacher top-k entries
```

For the first implementation, local HF scoring is simpler to validate. Server scoring is useful only if VRAM makes local teacher scoring impossible.

## Loss

Start with TRL-style generalized JSD or reverse-KL over generated tokens:

```text
student_logits = student(prompt + student_completion)
teacher_logits = teacher(prompt + student_completion)
labels mask = generated completion positions only
loss = generalized_jsd(student_logits, teacher_logits, labels, beta)
```

Initial defaults:

```text
beta: 1.0       # reverse-KL style, matching common on-policy setting
temperature: 1.0
max_completion_tokens: 128
batch_size: 1
learning_rate: 1e-6 to 2e-5 depending on stability
```

If full-vocabulary teacher logits are too expensive, use sparse top-1 or top-k support following TRL's `DistillationTrainer`:

```text
forward KL: teacher top-k support
reverse KL: actual sampled token support, optionally teacher top-1
JSD: union of selected supports
```

## Scripts

Create:

```text
scripts/build_on_policy_prompts.py
scripts/train_dense_online_kd.py
scripts/nightly_online_kd.sh
```

`build_on_policy_prompts.py`:

```text
input: runs/coding_harness_<run_id>/
output: data/on_policy_prompts/<run_id>.jsonl
behavior: emit prompt-only states ending before an assistant action
```

`train_dense_online_kd.py`:

```text
input: prompt-only JSONL
student: dense SFT checkpoint
teacher: Laguna XS.2 local HF model or scoring server
output: runs/online_kd/<run_id>/checkpoint-final
```

`nightly_online_kd.sh`:

```text
1. validate prompt dataset
2. run online KD smoke for 1-5 batches
3. run online KD training
4. run held-out coding smoke eval
```

## Metrics

Minimum training metrics:

```text
loss
jsd_or_kl_loss
completion_tokens_per_step
student_completion_mean_length
student_completion_max_length
truncated_completion_fraction
peak_cuda_memory_gb
tokens_per_second
```

Minimum behavioral metrics:

```text
tool-call JSON validity
empty-action rate
exit-only rate
blocked-command rate in held-out harness eval
bounded patch rate in held-out harness eval
```

Debug artifacts:

```text
sample prompt/completion pairs every N steps
raw generated token ids for failed examples
teacher/student top-token disagreement examples
```

## Risks

### Student Generates Junk

If the dense student is too weak after SFT, generated actions may be invalid and teacher feedback may be low-value.

Mitigation:

```text
start with short prompt states
cap max_completion_tokens
log completions every few steps
filter empty completions from loss only if they dominate
fall back to more SFT before online KD
```

### Teacher And Student Do Not Fit Together

Laguna teacher plus dense student may exceed VRAM.

Mitigation:

```text
sequential local scoring with teacher loaded only for scoring windows
teacher scoring server on a second GPU/VM
smaller max_prompt_length and max_completion_tokens
gradient checkpointing on student
```

### Tool Action Boundaries Are Ambiguous

The trainer must know where the generated assistant action ends.

Mitigation:

```text
use EOS / assistant-turn stop tokens from Laguna tokenizer
set max_completion_tokens conservatively
log truncated_completion_fraction
prefer one assistant action per prompt, not multi-turn generation
```
