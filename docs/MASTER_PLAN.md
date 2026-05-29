# Laguna XS.2 Hackathon Master Plan

## Status

**Active direction:** ToolSpec-style speculative decoding for agentic coding tool calls.

**Archived direction:** MoE-to-dense Laguna XS.2 densification. The densification RFCs remain useful background, but they are not the primary hackathon implementation path.

For the current plan, start with [ACTIVE_PLAN.md](ACTIVE_PLAN.md).

## Current Thesis

Laguna XS.2 already has a strong neural speculative decoding baseline via DFlash. Our possible contribution is narrower and more agent-specific:

```text
Pool/SWE-bench rollouts contain structured, repetitive tool-call spans.
Those spans include deterministic schema tokens and repo-derived argument values.
Schema-aware and retrieval-augmented drafting may reduce single-request decode latency,
especially when fused with DFlash as the general fallback drafter.
```

The project is therefore a rollout-data and inference-systems study, not a full model-training project.

## Current Implementation Stack

```text
pool
  -> recording OpenAI-compatible proxy
  -> vLLM OpenAI-compatible Laguna XS.2 server
  -> rollout artifacts
  -> ToolSpec-style offline profiler
  -> optional live drafter prototype
```

HF/PyTorch remains useful for smoke tests, token inspection, and later activation replay. vLLM is the main serving path because it supports Laguna, Poolside tool parsing, Poolside reasoning parsing, and DFlash.

## Active Documents

- [RFC 008: ToolSpec-Style SWE-Bench Rollouts and Speculative Tool Decoding](rfcs/008-toolspec-swebench-rollout-speculation.md)
- [RFC 007: Real Repo Task Environments](rfcs/007-real-repo-task-environments.md)
- [Engineering Plan 003: vLLM SWE-Bench Rollout Data Infrastructure](eng_plans/003-vllm-swebench-rollout-data-infra.md)
- [Runbook: vLLM Laguna + Pool](runbooks/vllm-laguna-pool.md)

## Archived Background

These documents record prior exploration and should not drive the immediate hackathon plan:

- [RFC 001: Internal Setup](rfcs/001-internal-setup.md)
- [RFC 002: Training Dense Surrogate MoE Replacements](rfcs/002-training.md)
- [RFC 003: Evals](rfcs/003-evals.md)
- [RFC 004: Inventory](rfcs/004-inventory.md)
- [RFC 005: MoE-to-Dense Method Selection](rfcs/005-moe-to-dense-method-selection.md)
- [RFC 006: Parallel Densification Experiment Framework](rfcs/006-parallel-densification-experiment-framework.md)
- [Engineering Plan 001: Teacher Smoke Generation](eng_plans/001-teacher-smoke-generation.md)
- [Engineering Plan 002: Pool + HF Rollout ACP](eng_plans/002-pool-hf-rollout-acp.md)

## Near-Term Milestone

The next concrete milestone is:

```text
run 20 SWE-bench Verified rollouts through pool
serve Laguna XS.2 with vLLM behind a recording proxy
store full model/tool/streaming traces
extract a ToolSpec acceptance dataset
report token composition and predicted speedup versus DFlash baseline
```

Do not attempt a full SWE-bench leaderboard run before the trace corpus and profiler work.
