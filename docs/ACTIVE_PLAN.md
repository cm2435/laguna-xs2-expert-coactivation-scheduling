# Active Plan: Pool/SWE-Bench ToolSpec Speculation

## One-Line Goal

Run realistic Pool/SWE-bench rollouts with Laguna XS.2, record the model/tool traces, and evaluate whether ToolSpec-style schema-aware plus retrieval-augmented drafting can reduce single-request agent latency beyond the DFlash baseline.

## Why This Is The Active Track

Full MoE-to-dense distillation is too large for the hackathon timeline. Laguna XS.2 already ships a neural speculator, DFlash, so the interesting lower-risk question is not “can we do speculative decoding at all?” It is:

```text
Can agent-harness structure make speculative decoding better for tool-call spans?
```

Pool/SWE-bench is a good target because coding agents repeatedly produce structured actions:

- shell commands,
- file reads,
- search commands,
- edit/patch calls,
- test reruns,
- tool-call JSON/schema scaffolding,
- file paths copied from repo state,
- failure strings copied from observations.

## Baselines

Measure or estimate against:

1. **Vanilla Laguna XS.2**
   - no speculative decoding.

2. **Laguna XS.2 + DFlash**
   - official neural draft baseline.
   - DFlash proposes up to 7 tokens per step.

3. **ToolSpec-style drafting**
   - schema/FSM drafting for deterministic tool-call structure.
   - repo/observation/history retrieval for ambiguous argument values.

4. **DFlash + ToolSpec routing**
   - DFlash for free-form reasoning/code.
   - ToolSpec for tool-call skeletons and copied argument spans.

## Workstreams

### 1. Rollout Infrastructure

Use [Engineering Plan 003](eng_plans/003-vllm-swebench-rollout-data-infra.md).

Target stack:

```text
pool
  -> recording OpenAI proxy
  -> vLLM Laguna XS.2 endpoint
  -> SWE-bench sandbox/grader
```

Deliverable:

```text
20+ SWE-bench Verified rollouts with complete request/response/tool/timing logs
```

### 2. Real Repo Environments

Use [RFC 007](rfcs/007-real-repo-task-environments.md).

Rules:

- task manifests and graders are central,
- repo templates are clean and reusable,
- pool runs only in disposable sandboxes,
- hidden graders are outside the model-visible repo.

### 3. ToolSpec Dataset Extraction

From each rollout, extract:

```text
assistant output spans
tool-call spans
tool names
argument keys
argument values
shell commands
file paths
patch/diff spans
recent observations
repo context snippets
token ids
timing
```

Deliverable:

```text
runs/.../speculation_dataset.jsonl
```

### 4. Offline Speculation Profiler

Simulate the drafting policy without yet changing vLLM:

```text
FSM draft:
  deterministic schema tokens, tool names, argument keys

context retrieval draft:
  file paths, test names, traceback strings, opened code spans

historical retrieval draft:
  similar prior pool/SWE-bench tool calls

DFlash reference:
  official baseline for general speculation
```

Metrics:

```text
accepted tokens per verify step
tool-call token fraction
schema-token fraction
argument-value fraction
copy-from-observation fraction
copy-from-repo fraction
predicted model forward passes saved
predicted latency speedup by span type
```

### 5. Optional Live Prototype

If the offline profiler shows enough structure, build a live prototype:

```text
pool
  -> recording/speculation proxy
  -> vLLM Laguna
```

The proxy should route:

```text
tool-call skeleton -> grammar/FSM draft
argument values -> retrieval draft
free text -> DFlash/vLLM default path
```

## Success Criteria

Minimum useful result:

```text
20 real rollouts
clean trace schema
token decomposition report
offline ToolSpec acceptance estimates
clear comparison against DFlash as the baseline
```

Strong result:

```text
tool-call spans show long accepted draft lengths
retrieval handles file paths / shell commands / patch scaffolds
predicted speedup is meaningful for whole agent turns, not only tiny JSON spans
```

Stop/pivot if:

```text
pool traces contain little explicit tool-call structure
tool-call tokens are too small a fraction of total decode
argument values are rarely copied/repeated
DFlash already captures almost all available acceptance
```

## Current Source Of Truth

Current design:

- [RFC 008](rfcs/008-toolspec-swebench-rollout-speculation.md)
- [RFC 007](rfcs/007-real-repo-task-environments.md)

Current execution:

- [Engineering Plan 003](eng_plans/003-vllm-swebench-rollout-data-infra.md)

Historical background:

- [MASTER_PLAN.md](MASTER_PLAN.md#archived-background)
