# vLLM SWE-Bench Rollout Data Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a vLLM-backed OpenAI-compatible Laguna XS.2 endpoint and build the task/sandbox/grader infrastructure needed to run 20+ realistic SWE-bench Verified rollouts through `pool`, collecting full model/tool traces for ToolSpec-style speculative decoding.

**Architecture:** `pool` runs inside disposable task sandboxes. A recording OpenAI proxy sits between `pool` and vLLM, logging requests/responses, streamed deltas, timing, and tokenized outputs. vLLM serves Laguna XS.2 for realistic small-batch latency and future drafter integration. SWE-bench task manifests and hidden graders live outside model-visible sandboxes.

**Tech Stack:** pool CLI, vLLM OpenAI server, Laguna XS.2, Python 3.11, uv, Hugging Face `datasets`, SWE-bench data, JSONL/PT artifacts, pytest.

---

## Executive Summary

We are moving from the HF smoke backend to a serving-oriented rollout stack:

```text
pool
  -> recording OpenAI-compatible proxy
  -> vLLM OpenAI-compatible Laguna server
  -> full rollout artifacts
  -> SWE-bench grading
  -> ToolSpec offline acceptance dataset
```

The target is not a full SWE-bench leaderboard run. The target is a high-quality trace corpus:

```text
20+ real repo tasks
full pool/tool/model events
streaming request/response logs
generated token spans
tool/action segmentation
grader outcomes
optional HF replay for activation traces
```

This is the dataset needed to decide whether a ToolSpec-style schema/retrieval drafter has legs for agentic coding.

## Backend Choice: vLLM First

Pick **vLLM first**, with SGLang as a follow-up parity target.

### Why vLLM First For Small-Batch Latency

vLLM is the lower-risk first serving backend for this project because the local vLLM checkout already contains Laguna-specific and Poolside-specific support:

```text
vllm/model_executor/models/laguna.py
vllm/tool_parsers/poolside_v1_tool_parser.py
vllm/reasoning/poolside_v1_reasoning_parser.py
```

That gives us several practical advantages:

- Laguna XS.2 has an inference implementation in vLLM.
- Poolside tool-call parsing is already represented in vLLM.
- Poolside reasoning parsing is already represented in vLLM.
- vLLM's OpenAI server is mature and directly compatible with `pool --api-url`.
- vLLM has strong prefix caching and continuous batching, which matter for small concurrent agent rollouts.
- vLLM is a plausible final integration point for any ToolSpec drafter.

For small-batch latency specifically, vLLM is attractive because agentic coding is often low-to-moderate concurrency rather than giant offline batches. We care about:

```text
p50 model-call latency
p95 model-call latency
time-to-first-token if streaming
decode TPOT
prefix reuse across multi-turn sessions
stable behavior under 2-8 concurrent pool tasks
```

vLLM is built for this serving profile and already has the relevant Poolside/Laguna plumbing. SGLang may be faster or more flexible for some speculative decoding internals, but the local evidence for Laguna-specific support is weaker, so it is a second backend after vLLM parity.

### When To Use HF Instead

HF/PyTorch remains the instrumentation workbench:

```text
activation hooks
exact PyTorch tensor capture
manual decode-loop experiments
HF replay of selected vLLM traces
```

Do not block rollout collection on activations. Collect vLLM traces first; replay selected prompts through HF later for activations.

## End State

A successful implementation gives us:

```text
configs/vllm_laguna_xs2.yaml
configs/recording_proxy_vllm.yaml
configs/swebench_verified_20.yaml

scripts/run_vllm_laguna_server.sh
scripts/run_recording_openai_proxy.py
scripts/build_swebench_task_manifests.py
scripts/prepare_task_sandbox.py
scripts/grade_task_sandbox.py
scripts/run_pool_swebench_rollout.py
scripts/run_swebench_rollout_batch.py
scripts/extract_toolspec_dataset.py

src/densify/openai_proxy/
src/densify/tasks/
src/densify/swebench/
src/densify/tooltrace/

tasks/swebench_verified/*.yaml
tasks/registry.jsonl

runs/swebench_pool_rollouts/{run_id}/...
```

## Data Layout

Follow RFC 007:

```text
tasks/
  registry.jsonl
  swebench_verified/
    astropy__astropy-12907.yaml
    django__django-xxxxx.yaml
  graders/
    swebench_verified/
      astropy__astropy-12907/
        grade.sh

envs/
  repo_templates/
    astropy__astropy/
      <base_commit>/
        repo/
        metadata.json
  build_cache/
    astropy__astropy/

sandboxes/
  pool_runs/
    {run_id}/
      repo/
      task.yaml
      pool_stdout.json
      pool_stderr.txt
      patch.diff
      grade_result.json
      model_artifacts/

runs/
  swebench_pool_rollouts/
    {run_id}/
      proxy_requests.jsonl
      proxy_stream_events.jsonl
      pool_command.sh
      model_calls/
      tool_calls.jsonl
      speculation_dataset.jsonl
```

Commit task manifests and scripts. Do not commit repo templates, sandboxes, rollouts, or grader outputs.

## Task Selection

Start with 20 SWE-bench Verified tasks across multiple Python repos.

Preferred mix:

```text
astropy/astropy: 3-4 tasks
django/django: 3-4 tasks
pytest-dev/pytest: 3-4 tasks
sympy/sympy: 3-4 tasks
matplotlib/matplotlib: 2-3 tasks
scikit-learn/scikit-learn: 2-3 tasks
```

Selection criteria:

- Python repo.
- Known `base_commit`.
- Problem statement is understandable.
- Test patch is available for grading but hidden from the model.
- Setup does not require extreme disk/time.
- Task likely causes file reads, searches, shell commands, and edits.

Use SWE-bench Verified before SWE-bench Lite if possible because the tasks are curated for reliable evaluation.

## Important Anti-Cheat Rule

The model may see:

```text
problem_statement
repo files at base_commit
public command output it produces itself
```

The model must not see:

```text
gold patch
test_patch
hidden grader script
expected fix
prior successful rollouts
grade_result.json
```

The hidden grader applies or uses SWE-bench `test_patch` only after the rollout finishes.

## Task 1: Add vLLM Server Config And Runbook

**Files:**

- Create: `configs/vllm_laguna_xs2.yaml`
- Create: `scripts/run_vllm_laguna_server.sh`
- Create: `docs/runbooks/vllm-laguna-pool.md`

- [ ] **Step 1: Add vLLM YAML config**

Create `configs/vllm_laguna_xs2.yaml`:

```yaml
model: poolside/Laguna-XS.2
served-model-name: hf-laguna-probe
host: 127.0.0.1
port: 8791
trust-remote-code: true
max-model-len: 131072
tensor-parallel-size: 1
gpu-memory-utilization: 0.90
enable-auto-tool-choice: true
tool-call-parser: poolside_v1
reasoning-parser: poolside_v1
default-chat-template-kwargs:
  enable_thinking: false
```

- [ ] **Step 2: Add launcher script**

Create `scripts/run_vllm_laguna_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export VLLM_USE_V1=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

uv run vllm serve \
  --config configs/vllm_laguna_xs2.yaml
```

- [ ] **Step 3: Make launcher executable**

Run:

```bash
chmod +x scripts/run_vllm_laguna_server.sh
```

- [ ] **Step 4: Document runbook**

Create `docs/runbooks/vllm-laguna-pool.md`:

```markdown
# vLLM Laguna + Pool Runbook

## Start vLLM

```bash
scripts/run_vllm_laguna_server.sh
```

## Smoke vLLM Directly

```bash
curl http://127.0.0.1:8791/v1/models
```

## Smoke Through Pool

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8791/v1 \
  -a default \
  -p "Reply with the single word READY." \
  -o json
```
```

- [ ] **Step 5: Verify on B300**

Run:

```bash
scripts/run_vllm_laguna_server.sh
```

In another shell:

```bash
curl -s http://127.0.0.1:8791/v1/models
POOLSIDE_API_KEY=dummy pool exec --sandbox disabled --api-url http://127.0.0.1:8791/v1 -a default -p "Reply with the single word READY." -o json
```

Expected:

```text
vLLM loads Laguna
pool exits 0
output contains READY or a clean equivalent
```

- [ ] **Step 6: Commit**

```bash
git add configs/vllm_laguna_xs2.yaml scripts/run_vllm_laguna_server.sh docs/runbooks/vllm-laguna-pool.md
git commit -m "Add vLLM Laguna pool runbook"
```

## Task 2: Add Recording OpenAI Proxy

**Files:**

- Create: `src/densify/openai_proxy/__init__.py`
- Create: `src/densify/openai_proxy/recording_proxy.py`
- Create: `scripts/run_recording_openai_proxy.py`
- Test: `tests/test_recording_proxy.py`

Purpose: log pool's OpenAI requests and vLLM responses without changing pool.

Target control flow:

```text
pool --api-url http://127.0.0.1:8792/v1
  -> recording proxy
  -> vLLM http://127.0.0.1:8791/v1
```

The proxy records:

```text
request body
response body
stream deltas
latency
model id
task id / run id from headers or config
```

- [ ] **Step 1: Implement proxy skeleton**

`recording_proxy.py` should:

```text
serve /v1/models by forwarding to upstream
serve /v1/chat/completions by forwarding to upstream
support streaming SSE
write proxy_requests.jsonl
write proxy_stream_events.jsonl
create model_calls/call_000001/request.json
create model_calls/call_000001/served_text.txt
create model_calls/call_000001/metadata.json
```

- [ ] **Step 2: Add CLI**

`scripts/run_recording_openai_proxy.py`:

```bash
uv run python scripts/run_recording_openai_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 8792 \
  --upstream-base-url http://127.0.0.1:8791/v1 \
  --output-dir runs/proxy_smoke
```

- [ ] **Step 3: Test against dummy upstream**

Use a tiny local HTTP server in `tests/test_recording_proxy.py`, not vLLM.

Expected:

```text
non-streaming and streaming responses are forwarded
request/response artifacts are written
```

- [ ] **Step 4: Verify with pool**

Run:

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8792/v1 \
  -a default \
  -p "Reply with the single word READY." \
  -o json
```

Expected:

```text
pool exits 0
runs/proxy_smoke/model_calls/call_000001/request.json exists
runs/proxy_smoke/model_calls/call_000001/served_text.txt exists
```

## Task 3: Add SWE-Bench Task Manifest Builder

**Files:**

- Create: `src/densify/swebench/__init__.py`
- Create: `src/densify/swebench/task_selection.py`
- Create: `scripts/build_swebench_task_manifests.py`
- Create: `configs/swebench_verified_20.yaml`
- Test: `tests/test_swebench_task_selection.py`

- [ ] **Step 1: Add selection config**

Create `configs/swebench_verified_20.yaml`:

```yaml
dataset: princeton-nlp/SWE-bench_Verified
split: test
output_dir: tasks/swebench_verified
registry_path: tasks/registry.jsonl
target_total: 20
repos:
  astropy/astropy: 4
  django/django: 4
  pytest-dev/pytest: 3
  sympy/sympy: 3
  matplotlib/matplotlib: 3
  scikit-learn/scikit-learn: 3
```

- [ ] **Step 2: Implement manifest builder**

Each output YAML:

```yaml
task_id: astropy__astropy-12907
suite: swebench_verified
repo: astropy/astropy
repo_id: astropy__astropy
base_commit: "<base_commit>"
problem_statement: |
  ...
visible_to_model:
  issue_statement: true
  public_tests: false
  hidden_tests: false
environment:
  template_path: envs/repo_templates/astropy__astropy/<base_commit>/repo
  setup_command: python -m pip install -e .
grader:
  hidden_command: uv run python scripts/grade_task_sandbox.py --task tasks/swebench_verified/astropy__astropy-12907.yaml --sandbox {sandbox}
limits:
  timeout_s: 1800
  max_turns: 30
metadata:
  dataset: princeton-nlp/SWE-bench_Verified
  instance_id: astropy__astropy-12907
```

Store `test_patch` and any gold fields in a non-model-visible grader metadata file:

```text
tasks/graders/swebench_verified/{task_id}/grader_metadata.json
```

Do not include `patch` or `test_patch` in the task prompt.

- [ ] **Step 3: Generate 20-task registry**

Run:

```bash
uv run python scripts/build_swebench_task_manifests.py --config configs/swebench_verified_20.yaml
```

Expected:

```text
20 YAML manifests written
tasks/registry.jsonl written
grader metadata written
```

## Task 4: Add Repo Template Builder

**Files:**

- Create: `src/densify/tasks/__init__.py`
- Create: `src/densify/tasks/manifest.py`
- Create: `src/densify/tasks/repo_templates.py`
- Create: `scripts/prepare_repo_templates.py`
- Test: `tests/test_task_manifest.py`

Purpose: clone repos once per `repo_id/base_commit`, then reuse clean templates.

- [ ] **Step 1: Implement manifest loader**

Load YAML manifests into a dataclass with:

```text
task_id
suite
repo
repo_id
base_commit
problem_statement
environment.template_path
grader.hidden_command
limits
```

- [ ] **Step 2: Implement repo template preparation**

For each task:

```text
if envs/repo_templates/{repo_id}/{base_commit}/repo exists:
  skip
else:
  git clone https://github.com/{repo}.git
  git checkout {base_commit}
  write metadata.json
```

- [ ] **Step 3: Run for 20 tasks**

Run:

```bash
uv run python scripts/prepare_repo_templates.py --registry tasks/registry.jsonl
```

Expected:

```text
unique repo/base_commit templates exist
templates are clean git worktrees
```

## Task 5: Add Sandbox Builder And Grader Runner

**Files:**

- Create: `src/densify/tasks/sandbox.py`
- Create: `src/densify/tasks/grader.py`
- Create: `scripts/prepare_task_sandbox.py`
- Create: `scripts/grade_task_sandbox.py`
- Test: `tests/test_sandbox_builder.py`

Purpose: every rollout gets a disposable copy.

- [ ] **Step 1: Prepare sandbox**

Given a task manifest:

```text
copy envs/repo_templates/{repo_id}/{base_commit}/repo -> sandboxes/pool_runs/{run_id}/repo
write sandboxes/pool_runs/{run_id}/task.yaml
write sandboxes/pool_runs/{run_id}/metadata.json
```

- [ ] **Step 2: Grade sandbox**

For SWE-bench Verified:

```text
run public/no-op checks if available
after pool finishes, apply hidden test_patch or invoke official SWE-bench harness
run relevant test command
write grade_result.json
```

First implementation can be conservative:

```text
record patch.diff
record whether repo is dirty
record placeholder grade_result with "not_graded_yet"
```

Then upgrade to actual SWE-bench grading once sandbox mechanics work.

- [ ] **Step 3: Test on synthetic task first**

Before using SWE-bench, create one tiny synthetic repo task to validate:

```text
template copy
pool run location
patch export
hidden grader outside repo
```

## Task 6: Add Pool SWE-Bench Rollout Runner

**Files:**

- Create: `src/densify/tasks/pool_runner.py`
- Create: `scripts/run_pool_swebench_rollout.py`
- Create: `scripts/run_swebench_rollout_batch.py`
- Test: `tests/test_pool_runner.py`

Single task command:

```bash
uv run python scripts/run_pool_swebench_rollout.py \
  --task tasks/swebench_verified/astropy__astropy-12907.yaml \
  --api-url http://127.0.0.1:8792/v1 \
  --output-dir runs/swebench_pool_rollouts \
  --max-turns 30
```

The runner should:

```text
create run_id
prepare sandbox
write pool prompt from task.problem_statement
run pool in sandbox repo
capture stdout/stderr
save pool command
export git diff as patch.diff
run grader
copy/provide proxy model artifacts under run_id
write rollout_summary.json
```

Batch command:

```bash
uv run python scripts/run_swebench_rollout_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8792/v1 \
  --limit 20 \
  --concurrency 1
```

Start with `--concurrency 1`. Move to concurrency `2-4` only after vLLM/proxy/pool artifacts are stable.

## Task 7: Add ToolSpec Dataset Extractor

**Files:**

- Create: `src/densify/tooltrace/__init__.py`
- Create: `src/densify/tooltrace/pool_parse.py`
- Create: `src/densify/tooltrace/speculation_dataset.py`
- Create: `scripts/extract_toolspec_dataset.py`
- Test: `tests/test_tooltrace_parse.py`

Purpose: convert rollouts into the offline ToolSpec analysis input.

Input:

```text
runs/swebench_pool_rollouts/*/model_calls/*/served_text.txt
runs/swebench_pool_rollouts/*/model_calls/*/request.json
pool stdout/events if available
```

Output:

```text
tool_calls.jsonl
speculation_dataset.jsonl
```

Initial parser should detect:

```text
<tool_call>...</tool_call>
<arg_key>...</arg_key>
<arg_value>...</arg_value>
shell command blocks if pool emits them differently
file-edit blocks if pool emits them differently
```

Each speculation row:

```json
{
  "run_id": "...",
  "task_id": "...",
  "call_id": "call_000004",
  "tool_name": "shell",
  "arguments": {"cmd": "pytest ..."},
  "target_text": "...",
  "target_tokens_path": "...",
  "segments": [
    {"kind": "schema", "start": 0, "end": 12},
    {"kind": "variable", "field": "cmd", "start": 12, "end": 41}
  ]
}
```

## Task 8: First End-to-End Acceptance Gate

Run:

```bash
# terminal 1
scripts/run_vllm_laguna_server.sh

# terminal 2
uv run python scripts/run_recording_openai_proxy.py \
  --listen-port 8792 \
  --upstream-base-url http://127.0.0.1:8791/v1 \
  --output-dir runs/proxy_vllm

# terminal 3
uv run python scripts/run_swebench_rollout_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8792/v1 \
  --limit 1 \
  --concurrency 1
```

Success:

```text
one SWE-bench sandbox created
pool exits cleanly or with captured failure
patch.diff exists
proxy model calls exist
served_text exists
rollout_summary.json exists
grade_result.json exists, even if not solved
```

Then scale:

```bash
uv run python scripts/run_swebench_rollout_batch.py \
  --registry tasks/registry.jsonl \
  --api-url http://127.0.0.1:8792/v1 \
  --limit 20 \
  --concurrency 1
```

## Risk Register

### Risk: vLLM Laguna Load Fails

Fallback:

```text
use existing HF backend for rollouts
keep recording proxy/task infra unchanged
try SGLang parity separately
```

### Risk: Pool Needs Extra Non-OpenAI Endpoints

The existing HF probe already implemented extra Poolside-ish endpoints. If raw vLLM lacks them, point pool at the recording proxy and have the proxy implement those endpoints while forwarding `/v1/chat/completions` to vLLM.

### Risk: Token IDs Are Not Available From vLLM

Fallback:

```text
tokenize served_text/generated_text with the Laguna tokenizer in the proxy
store approximate output token IDs
replay selected requests through HF for exact token/activation capture
```

### Risk: SWE-Bench Grading Takes Too Long

Fallback:

```text
first collect rollouts and patches
run grading async afterward
prioritize task trace quality over solved-task score
```

### Risk: Full SWE-Bench Env Setup Is Too Heavy

Fallback:

```text
use 5 real SWE-bench tasks plus 5 synthetic repo tasks
keep same artifact schema
```

## Definition of Done

This plan is complete when:

```text
vLLM serves Laguna behind an OpenAI-compatible endpoint
pool can talk to vLLM through the recording proxy
20 SWE-bench Verified task manifests exist
repo templates and sandboxes are reproducible
at least 1 real SWE-bench rollout completes end to end
batch runner can attempt 20 tasks
every attempted task writes full artifacts
ToolSpec extraction produces tool_calls.jsonl/speculation_dataset.jsonl
```

The next plan after this should implement offline ToolSpec acceptance analysis:

```text
schema FSM
retrieval index
draft/target exact-match measurement
accepted-token curves
estimated speedup report
```
