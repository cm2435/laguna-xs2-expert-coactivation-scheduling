# RFC 003: Evals

**Status:** Partially superseded. Reconstruction and densification evals are archived background; repo-task sandboxing guidance now lives in [RFC 007](007-real-repo-task-environments.md), and the active inference study lives in [RFC 008](008-toolspec-swebench-rollout-speculation.md).

## Purpose

Measure whether MoE densification preserves useful coding behavior and whether it improves inference properties.

This workstream owns quality evals, reconstruction diagnostics, and systems metrics.

## Evaluation Ladder

Use a staged ladder so we do not burn time on expensive evals before the model can pass basic checks.

### Stage 0: Reconstruction Evals

Input:

```text
held-out activation shards
```

Metrics:

```text
MSE
relative error
cosine similarity
variance explained
worst-token error
worst-layer error
```

Pass condition:

```text
trained surrogate beats random init by a large margin
validation error tracks training error
```

### Stage 1: Smoke Generation

Prompts:

- "Write a Python function that sorts a list of integers."
- "Implement binary search in Python."
- "Parse a unified diff and return changed file paths."
- "Fix this off-by-one function given failing tests."
- "Given this traceback, identify the bug."

Metrics:

```text
can generate non-degenerate code
syntax validity
basic unit tests pass
teacher/student qualitative comparison
```

Pass condition:

```text
student outputs coherent code and passes at least some simple tests
```

### Stage 2: Small Coding Benchmarks

Use:

```text
HumanEval subset
MBPP subset
CruxEval subset
```

Metrics:

```text
pass@1
exact output for CruxEval-style tasks
syntax error rate
timeout rate
tokens generated
```

CruxEval is especially useful because it may catch representational damage even when function-generation prompts look okay.

### Stage 3: Agentic / Repo Tasks

Use:

```text
small SWE-bench Lite or Verified subset
captured pool turns
repo bug-fix microtasks
```

Repo tasks must run in disposable sandboxes created from clean repo templates, with hidden graders kept outside the model-visible workspace. See [RFC 007: Real Repo Task Environments](007-real-repo-task-environments.md).

Metrics:

```text
task solved
patch applies
tests pass
tool-call / action quality
teacher agreement on next action
```

This is expensive. Only run after Stage 1 and Stage 2 look viable.

## Systems Metrics

Measure both model-size and runtime behavior.

### Memory

Report:

```text
checkpoint size
GPU memory at load
GPU memory during generation
KV cache memory
activation memory during training
```

Compare:

```text
teacher MoE
partial densified model
full densified model
different surrogate widths
```

### Throughput

Report:

```text
decode tokens/sec
prefill tokens/sec
requests/sec under fixed concurrency
latency p50/p95
```

Important: dense surrogates may reduce memory and dispatch overhead but increase per-token dense compute. Throughput is empirical.

### Device Feasibility

Inventory possible deployment targets:

```text
single H100/A100
consumer GPU
Apple Silicon
mobile/edge runtimes
```

For each:

```text
model size after densification
quantized size estimate
RAM/VRAM requirement
runtime support
expected tokens/sec
```

Mobile serving is a stretch goal, but the densified architecture is more mobile-friendly than routed MoE because it removes expert dispatch.

## Eval Data Suggestions

### Tiny Handwritten Set

Create `data/evals/tiny_coding.jsonl` with 20-50 examples:

```json
{
  "id": "sort_001",
  "prompt": "Write a Python function sort_numbers(xs) that returns xs sorted ascending.",
  "tests": "assert sort_numbers([3,1,2]) == [1,2,3]"
}
```

### HumanEval / MBPP

Use small subsets first:

```text
HumanEval first 20
MBPP sanitized first 50
```

### CruxEval

Use for code reasoning:

```text
input prediction
output prediction
```

### Teacher Agreement Set

For captured coding-agent turns:

```text
teacher continuation
student continuation
token-level KL if logits available
embedding similarity
action parse validity
```

## Reporting

Produce:

```text
reconstruction_by_layer.csv
eval_scores_by_checkpoint.csv
systems_metrics_by_checkpoint.csv
```

Plots:

- Surrogate width vs reconstruction error.
- Reconstruction error by layer.
- Reconstruction error vs coding eval score.
- Model memory vs pass@1.
- Tokens/sec vs pass@1.

## Deliverables

- Tiny coding eval file.
- Smoke generation runner.
- HumanEval/MBPP subset runner.
- CruxEval runner or wrapper.
- Systems benchmark script.
- Summary report template.

## Open Questions

- Which eval catches failure earliest?
- What amount of reconstruction error is acceptable?
- Do later layers matter more for coding quality?
- Does a partial densification strategy outperform full replacement?
- Does lower memory translate into useful local/device serving?
