# RFC 008: ToolSpec-Style SWE-Bench Rollouts and Speculative Tool Decoding

## Purpose

Pivot the project from full RL or full MoE-to-dense training toward a lower-risk inference/agent-systems experiment:

```text
run pool on a small SWE-bench subset
serve Laguna XS.2 through our local OpenAI-compatible HF backend
record full agent/model/tool rollouts
capture optional activation traces
use those traces to build a ToolSpec-style schema-aware and retrieval-augmented speculative decoder
```

The intended outcome is a concrete speedup story for agentic coding:

```text
tool traces are structured
pool tool calls obey predictable schemas
SWE-bench trajectories contain repeated tool invocation patterns
we can draft deterministic schema tokens and likely argument spans
the verifier model accepts long chunks, reducing decode latency
```

## Current Status

The critical pool-to-local-model path is now working on the B300 VM.

Verified:

```text
pool exec --api-url http://127.0.0.1:<port>/v1 ... talks to our local backend
dummy backend returns READY and exits cleanly
HF backend loads poolside/Laguna-XS.2
HF backend generates via Transformers
backend streams completions back to pool
backend writes request/model artifacts to disk
```

Known command shape:

```bash
uv run python scripts/run_openai_probe_server.py \
  --mode hf \
  --host 127.0.0.1 \
  --port 8790 \
  --max-new-tokens 8 \
  --no-sample \
  --disable-thinking \
  --output-dir runs/pool_hf_backend_smoke_clean
```

Pool call:

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8790/v1 \
  -a default \
  -p "Reply with the single word READY." \
  -o json
```

Artifacts currently written per model call:

```text
request.json
input_tokens.pt
generated_tokens.pt
generated_text.txt
served_text.txt
metadata.json
```

This means the backend boundary is real. The next work is data collection, trace schema design, and speculative decoding instrumentation.

## Why ToolSpec Has Legs Here

The ToolSpec paper argues that tool-calling traces are:

```text
highly structured
schema-constrained
repetitive across turns and tasks
amenable to training-free speculative drafts
```

That maps well to pool/SWE-bench because agentic coding loops repeatedly emit tool-like actions:

```text
read file
search
run shell command
edit file
inspect test output
rerun tests
```

Even if pool's exact wire format is not the same as OpenAI function calling, the output patterns are still constrained by pool's action syntax and the tool schemas available to the agent. That gives us deterministic or near-deterministic spans:

```text
tool/action delimiters
JSON keys or XML-ish tags
command field names
file path field names
repeated shell commands
repeated test invocations
edit scaffolding
```

ToolSpec's key components are useful:

1. **Schema-aware drafting**
   - Use a finite-state machine or grammar to draft deterministic schema tokens.

2. **Variable-field speculation**
   - Use the base model or retrieval only for high-entropy fields such as paths, commands, edits, or line numbers.

3. **Retrieval-augmented drafting**
   - Retrieve similar historical tool calls and reuse their token spans as drafts.

4. **Training-free integration**
   - Does not require RL or fine-tuning.

References:

- Paper: https://arxiv.org/abs/2604.13519
- Code: https://github.com/hemingkx/ToolSpec

## Non-Goals

This RFC does not propose:

- a full RL training run,
- immediate full MoE-to-dense training,
- replacing pool,
- building a full SWE-bench leaderboard run,
- modifying the model weights.

The goal is a trace-driven speculative decoding experiment with real agentic coding trajectories.

## Data Collection Target

Start with:

```text
10-20 SWE-bench Verified / Lite tasks
spread across several Python repositories
pool as the harness
local HF Laguna backend as the model endpoint
full request/response/tool traces
optional one-layer activation capture
```

Do not start with all SWE-bench. The value comes from high-quality full traces and latency instrumentation, not broad benchmark coverage.

Recommended first repository spread:

```text
astropy/astropy
django/django
pytest-dev/pytest
sympy/sympy
scikit-learn/scikit-learn
matplotlib/matplotlib
```

Pick tasks that:

- install reasonably,
- have known reproducible tests,
- are Python-only,
- are not massive dependency traps,
- exercise file reads/search/shell/edit loops.

## Trace Artifacts

Each rollout should write:

```text
runs/swebench_pool_rollouts/{run_id}/
  task.yaml
  pool_command.sh
  pool_stdout.json
  pool_stderr.txt
  pool_events.jsonl
  patch.diff
  grade_result.json
  model_calls/
    call_000001/
      request.json
      messages.json
      input_tokens.pt
      generated_tokens.pt
      generated_text.txt
      served_text.txt
      metadata.json
      decoded_segments.json
      activations/
        optional_layer_inputs_outputs.pt
  tool_calls.jsonl
  speculation_dataset.jsonl
```

### Model Call Metadata

Extend `metadata.json` with:

```json
{
  "call_id": "call_000001",
  "task_id": "django__django-xxxxx",
  "repo_id": "django/django",
  "phase": "decode",
  "input_token_count": 12345,
  "generated_token_count": 512,
  "latency_s": 8.2,
  "tokens_per_second": 62.4,
  "served_text_was_cleaned": true,
  "activation_capture": "none"
}
```

### Tool Call Row

`tool_calls.jsonl` should normalize pool outputs into a schema useful for ToolSpec:

```json
{
  "task_id": "pytest-dev__pytest-xxxxx",
  "call_id": "call_000004",
  "turn_index": 4,
  "tool_name": "shell",
  "arguments": {
    "cmd": "pytest tests/test_foo.py -q"
  },
  "raw_text": "...",
  "start_token": 84,
  "end_token": 129,
  "accepted_by_pool": true,
  "observation_summary": "2 failed, 1 passed"
}
```

If pool does not expose explicit tool-call JSON, build this parser from served/raw text and pool event logs.

### Speculation Dataset Row

For ToolSpec-style experiments:

```json
{
  "task_id": "astropy__astropy-12907",
  "call_id": "call_000003",
  "prefix_tokens_path": "model_calls/call_000003/input_tokens.pt",
  "target_tokens_path": "model_calls/call_000003/generated_tokens.pt",
  "target_text_path": "model_calls/call_000003/generated_text.txt",
  "tool_schema": {
    "shell": ["cmd"],
    "read_file": ["path"],
    "edit": ["path", "old", "new"]
  },
  "segments": [
    {
      "kind": "schema",
      "start": 0,
      "end": 12
    },
    {
      "kind": "variable",
      "field": "cmd",
      "start": 12,
      "end": 41
    }
  ]
}
```

This is the bridge from raw rollouts to speculative decoding.

## Speculative Decoding Plan

### Stage 1: Offline Acceptance Analysis

Before implementing online acceleration, compute an offline upper bound:

```text
given prefix and generated target
construct ToolSpec drafts from schemas and retrieval
measure how many draft tokens match target exactly
estimate accepted-token lengths
```

Metrics:

```text
schema token match rate
variable field match rate
mean accepted tokens per verify step
p50/p95 accepted tokens per tool call
fraction of generation covered by tool-call spans
estimated decode step reduction
```

This tells us whether ToolSpec has legs on pool/SWE-bench before we touch generation internals.

### Stage 2: Schema FSM for Pool Actions

Implement a finite-state machine for the actual pool action format.

Draft deterministic spans:

```text
tool/action prefix
tool name
JSON/object keys
quotes, commas, braces, separators
known field names
closing markers
```

Leave variable spans to:

```text
normal model decode
retrieval drafts
prompt lookup
```

Use ToolSpec's `SchemaFSM` as conceptual reference, but adapt to pool's real action syntax and Laguna tokenizer.

### Stage 3: Retrieval-Augmented Drafting

Build retrieval over prior tool calls:

```text
key: task/repo/turn context + tool name + recent observation summary
value: tokenized previous tool invocation
```

Retrieval candidates:

```text
same task earlier calls
same repo calls
same tool name across tasks
same command family, e.g. pytest
same file path suffix
```

Expected high-value cases:

```text
pytest reruns
repeated grep/rg searches
reading the same file after edits
edit scaffolding
shell command boilerplate
```

### Stage 4: Online Prototype

Integrate into the local HF backend:

```text
model generates or is about to generate a tool/action span
ToolSpec drafter proposes candidate token sequence
teacher/verifier model verifies draft tokens
accepted tokens are appended
rejected token falls back to normal decode
```

This may require moving beyond `model.generate(...)` into a manual decode loop for tool spans. That is a later step. First prove offline acceptance and trace coverage.

## Activation Capture Role

Activation capture is still useful, but it is no longer the primary goal.

Use activations for:

```text
analysis of tool-call vs reasoning token representations
future MoE densification data
possible router/expert behavior during tool spans
```

Initial capture should be narrow:

```text
one or two MoE layers
tool-call-heavy turns only
limited token budget
```

Do not let activation capture block rollout collection.

## Backend Choice: HF vs vLLM

### HF Backend

Use first because:

- already works with pool,
- gives direct token artifacts,
- gives direct PyTorch hooks,
- easiest for manual decode-loop experiments.

### vLLM Backend

Consider later if:

- we need higher-throughput rollout generation,
- we want realistic batched serving metrics,
- we can still get enough token/request logs,
- speculative decoding hooks are easier or faster there.

For ToolSpec, HF is probably better for the prototype because the hard part is not raw throughput. The hard part is controlling and instrumenting draft/verify behavior.

## Evaluation Metrics

### Rollout Dataset Metrics

```text
number of tasks
number of model calls
number of generated tokens
number of parsed tool calls
tool-call token fraction
unique tool schemas observed
repeated tool-call rate
```

### Speculation Metrics

```text
draft tokens proposed
draft tokens accepted
mean accepted length
acceptance rate by tool
acceptance rate by field
estimated verifier calls saved
estimated decode latency reduction
```

### Agent Metrics

```text
task completed
patch applies
public tests pass
hidden tests pass
turn count
tool call count
wall-clock runtime
```

### Serving Metrics

```text
p50/p95 model call latency
p50/p95 time-to-first-token if streaming is measured
tokens/sec baseline vs ToolSpec prototype
tool-span decode speedup
end-to-end rollout wall-clock
```

## Implementation Order

### Phase 1: Real Tiny Coding Rollout

Use the working OpenAI-compatible HF endpoint.

Tasks:

```text
create one tiny repo sandbox
run pool against local HF backend
capture full request/model artifacts
export patch
run public/hidden grader
```

Success:

```text
one real pool coding rollout exists on disk
tool/action text can be parsed or at least segmented
```

### Phase 2: SWE-Bench Mini Subset

Tasks:

```text
select 10-20 SWE-bench tasks
create task manifests
prepare repo templates/sandboxes
run pool on each task with local HF backend
store full traces and grader outputs
```

Success:

```text
>= 10 complete traces
tool_calls.jsonl extracted for most turns
model call artifacts exist for every turn
```

### Phase 3: Offline ToolSpec Analyzer

Tasks:

```text
derive pool tool schema
parse tool-call spans
build schema FSM
build retrieval index over historical tool calls
compute offline draft acceptance against recorded generated tokens
```

Success:

```text
report estimated accepted tokens and speedup by tool type
identify whether schema-only, retrieval-only, or combined drafting is strongest
```

### Phase 4: Online ToolSpec Prototype

Tasks:

```text
replace model.generate with a manual decode loop for tool spans
verify schema/retrieval drafts against Laguna logits
stream accepted text back to pool
measure actual latency
```

Success:

```text
same task output quality
lower tool-call decode latency
artifact logs show accepted/rejected draft spans
```

## Go / No-Go Criteria

Continue ToolSpec if:

```text
tool-call spans are parseable from pool/Laguna outputs
tool-call tokens are a meaningful fraction of generated tokens
schema tokens match recorded generations with high acceptance
retrieval improves variable-field acceptance
offline estimated decode-step reduction is substantial
```

Pivot if:

```text
Laguna rarely emits structured tool/action spans through pool
pool hides tool calls in a format we cannot reconstruct
tool-call spans are too small a fraction of total decode tokens
retrieval does not repeat across SWE-bench tasks
manual decode integration is too invasive for the hackathon
```

## Current Recommendation

This has legs, but only if we validate trace structure before building the online decoder.

The next engineering milestone should be:

```text
one tiny coding rollout
then 10-20 SWE-bench rollouts
then offline ToolSpec acceptance analysis
then online speculative decoding only if the offline curves are good
```

The most important near-term artifact is not a faster decoder. It is a high-quality trace dataset:

```text
full pool events
full OpenAI requests
raw and served model text
token tensors
tool-call segmentation
grader outcomes
optional activations
```

That dataset will tell us whether ToolSpec is the right hackathon bet.
