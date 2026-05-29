# Teacher Smoke Generation Implementation Plan

**Status:** Background/completed scaffold. Useful for HF smoke tests, but superseded for active rollout collection by [003: vLLM SWE-Bench Rollout Data Infrastructure](003-vllm-swebench-rollout-data-infra.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a Laguna XS.2 teacher model loaded through HF/PyTorch and generating sane Python coding outputs on a small, reproducible coding-task scaffold.

**Architecture:** The first milestone deliberately avoids densification, surrogate training, pool, SGLang, and vLLM. We prove the plain teacher path works in a normal PyTorch process, then use that same process for model introspection, prompt formatting, generation, and simple coding-eval scoring.

**Tech Stack:** Python 3.11, PyTorch, Transformers, Accelerate, Datasets, PyYAML, pytest, optional Hugging Face gated model auth.

---

## Executive Summary

This is the first engineering plan for the densification project. It answers the question:

> Can we load the default Laguna XS.2 teacher scaffold and make it generate plausible, testable Python coding answers before touching MoE replacement?

The output of this plan is not a trained model. The output is a working teacher-generation baseline:

```text
config
  -> load tokenizer + HF/PyTorch teacher
  -> inspect architecture and MoE layers
  -> run Python coding prompts
  -> extract code blocks
  -> run simple unit tests where available
  -> save JSONL generations and summary metrics
```

This baseline becomes the foundation for activation capture and training. If the teacher cannot be loaded, prompted, and scored in this scaffold, the later representation collection work will be brittle.

## What We Are Proving

### Primary Success Criterion

On one H100 80GB VM, using HF/PyTorch, we can:

1. Load the Laguna XS.2 teacher or a clearly declared architecture-compatible proxy.
2. Confirm the model exposes Transformer layers and MoE/MLP modules through `named_modules()`.
3. Generate completions for at least 20 Python coding prompts.
4. Save all prompts, generations, parse status, syntax status, and unit-test status to disk.
5. Produce a short run summary with sane generations visible in JSONL.

### Definition of "Sane"

For this milestone, "sane" means:

```text
>= 90% of prompts produce non-empty text
>= 70% of generations contain a Python-looking code block or function definition
>= 60% of extracted Python snippets parse with ast.parse
>= 30% of handwritten unit-test prompts pass their tests
no repeated-token collapse on the first 20 prompts
```

These are not final quality targets. They are smoke gates that tell us the model loading, prompting, decoding, and scoring pipeline is usable.

## Backend Decision For This Milestone

Use HF/PyTorch as the generation backend.

Reason:

```text
This milestone must exercise the same model object we later hook for activations.
We need model.named_modules(), forward hooks, and direct module access.
SGLang/vLLM are useful later, but they hide or transform the model internals.
```

SGLang/vLLM are out of scope for the first pass except as implementation references if HF remote-code loading fails.

pool is also out of scope for the first pass. pool is a coding-agent harness, not the generation engine. We can use pool later to produce agentic prompts, but the first scaffold should be a plain JSONL prompt file.

## Hardware Plan

Target VM:

```text
Provider: Prime Intellect / Prime Select
Budget: $200 credits
Main GPU: 1x H100 80GB
Host RAM: >= 128GB, prefer >= 256GB
Disk: >= 500GB local NVMe
OS: Ubuntu 22.04 or equivalent CUDA-ready image
```

Why one H100 first:

```text
The goal is path correctness, not throughput.
Multi-GPU introduces Accelerate/device_map complexity before we know the scaffold works.
One H100 should be enough for teacher inference if the HF checkpoint supports FP8 or reasonable sharding/offload.
```

Fallbacks:

```text
If full Laguna XS.2 does not load:
  use a smaller MoE proxy to validate the code path
  keep all output paths and schemas identical
  record the proxy in reports/run_summary.json

If generation OOMs:
  reduce max_new_tokens
  reduce batch size to 1
  enable device_map="auto"
  enable CPU offload only as a last resort
```

## Data Plan

### Dataset 1: Handwritten Python Smoke Set

Create a local JSONL file with 20-50 examples:

```text
data/prompts/python_smoke.jsonl
```

Each row:

```json
{
  "id": "sort_001",
  "prompt": "Write a Python function sort_numbers(xs) that returns the numbers in ascending order.",
  "entrypoint": "sort_numbers",
  "tests": [
    "assert sort_numbers([3, 1, 2]) == [1, 2, 3]",
    "assert sort_numbers([]) == []"
  ]
}
```

Use this first because it is deterministic, cheap, and debuggable.

### Dataset 2: HumanEval / MBPP Subset

Use after Dataset 1 passes. This gives a standard function-synthesis ladder without repo checkout overhead.

Recommended first slice:

```text
HumanEval first 20 tasks
MBPP sanitized first 50 tasks
```

### Dataset 3: Python SWE-bench Verified Tiny Subset

Use only after the teacher scaffold works. This is closer to the final coding-task story, but it is heavier.

Candidate construction:

```text
source: princeton-nlp/SWE-bench_Verified
filter: repositories that are primarily Python
first target size: 5 tasks
expanded target size: 20 tasks
```

For this RFC, SWE-bench is a prompt source, not the full agentic harness. We should not block teacher smoke generation on repo checkout, patch application, or long test runs.

## Output Artifacts

Every run writes:

```text
runs/teacher_smoke/{run_id}/
  config_resolved.yaml
  architecture.json
  generations.jsonl
  summary.json
  examples.md
```

`generations.jsonl` row:

```json
{
  "id": "sort_001",
  "prompt": "...",
  "raw_generation": "...",
  "extracted_code": "...",
  "parse_ok": true,
  "tests_ok": true,
  "test_stdout": "",
  "test_stderr": "",
  "latency_s": 4.2,
  "generated_tokens": 128
}
```

`architecture.json` fields:

```json
{
  "model_id": "poolside/Laguna-XS.2-FP8",
  "torch_dtype": "bfloat16",
  "num_parameters_seen": 33000000000,
  "num_transformer_layers": 40,
  "candidate_moe_modules": [
    {
      "name": "model.layers.1.mlp",
      "class_name": "LagunaSparseMoeBlock"
    }
  ]
}
```

## Repository and Artifact Layout

Use a normal `src/` Python package with `uv` as the environment manager. Keep source code, scripts, configs, immutable input data, generated rollouts, eval artifacts, and large checkpoints in separate folders. The split matters because this project will quickly produce large activation shards and run outputs that should not be committed.

### Top-Level Layout

```text
pyproject.toml
uv.lock
.python-version
.gitignore

configs/
  teacher_smoke_h100.yaml
  teacher_smoke_proxy.yaml
  activation_capture_h100.yaml
  train_surrogate_layer.yaml
  eval_tiny_coding.yaml

data/
  prompts/
    python_smoke.jsonl
    humaneval_tiny.jsonl
    mbpp_tiny.jsonl
    swebench_verified_python_tiny.jsonl
  rollouts/
    teacher_smoke/
    pool_agent/
  activations/
    teacher_laguna_xs2/
      layer_XX/
  evals/
    tiny_coding_results/
    humaneval_results/
    swebench_prompt_results/

inventory/
  models.md
  data.md
  hardware.md
  experiments.md
  checkpoints.md

reports/
  figures/
  tables/
  final_summary.md

runs/
  teacher_smoke/
  activation_capture/
  training/
  evals/
  inference_benchmarks/

checkpoints/
  surrogate_layers/
  densified_model/

src/densify/
  __init__.py
  config.py
  teacher_loader.py
  model_introspection.py
  prompt_data.py
  generation.py
  code_scoring.py
  run_artifacts.py
  activation_capture.py
  surrogate_modules.py
  training_data.py
  evals.py
  metrics.py

scripts/
  smoke_load_teacher.py
  run_teacher_smoke_eval.py
  build_python_smoke_prompts.py
  build_swebench_verified_tiny.py
  collect_rollouts.py
  capture_activations.py
  train_surrogate_layer.py
  assemble_densified_model.py
  run_eval.py

tests/
  test_prompt_data.py
  test_code_scoring.py
  test_run_artifacts.py
  test_model_introspection.py
  test_activation_capture.py
  test_surrogate_modules.py
```

### What Lives Where

`src/densify/` contains importable library code. Scripts should be thin wrappers around functions in this package.

`scripts/` contains executable entry points for the hackathon workflow:

```text
build_*                 -> create small prompt/eval datasets
smoke_load_teacher.py   -> check model loading and architecture introspection
run_teacher_smoke_eval  -> generate and score teacher outputs
collect_rollouts.py     -> create prompt/completion rollouts
capture_activations.py  -> replay prompts and save x_layer/y_teacher shards
train_surrogate_layer   -> train one dense replacement block
assemble_densified_model -> swap trained surrogates into a checkpoint
run_eval.py             -> evaluate teacher/student checkpoints
```

`configs/` contains small YAML files for reproducible runs. Every script should accept `--config` and write the resolved config into its run directory.

`data/prompts/` contains committed, small JSONL prompt files. These are inputs, not generated experiment outputs.

`data/rollouts/` contains generated prompt/completion traces. Do not commit large rollout files. Store a small sample only if needed for tests or documentation.

`data/activations/` contains activation shards used for surrogate training. Never commit this folder; it will become huge.

`data/evals/` contains generated eval outputs or converted benchmark subsets. Commit only tiny handcrafted eval definitions; do not commit benchmark result dumps unless they are small final reports.

`runs/` contains per-run artifacts: resolved configs, JSONL generations, summaries, logs, and examples. This is the main debugging surface. Do not commit routine run outputs.

`checkpoints/` contains trained surrogate layers and assembled densified checkpoints. Never commit model weights.

`inventory/` contains human-maintained markdown registries for models, datasets, hardware, experiments, and checkpoints. These should be committed because they explain what happened.

`reports/` contains final hackathon tables, figures, and summaries. Commit final small reports, not raw logs.

### File Responsibility Matrix

The top-level layout is intentionally broader than this first engineering plan. Plan 001 implements the teacher smoke path; later plans fill in activation capture, surrogate training, assembly, and full evals. Use this matrix to keep boundaries crisp.

#### Root Tooling Files

| File | Plan | Responsibility |
| --- | --- | --- |
| `.python-version` | 001 | Pins the local Python version to `3.11` for `uv`. |
| `pyproject.toml` | 001 | Defines package metadata, runtime deps, dev deps, pytest paths, and ruff settings. |
| `uv.lock` | 001 | Locks exact dependency versions after `uv sync --extra dev`. |
| `.gitignore` | 001 | Excludes virtualenvs, run outputs, rollouts, activations, checkpoints, and raw reports. |

#### Config Files

| File | Plan | Responsibility |
| --- | --- | --- |
| `configs/teacher_smoke_h100.yaml` | 001 | H100 teacher smoke config for Laguna XS.2: model id, dtype, prompt path, output path, generation settings. |
| `configs/teacher_smoke_proxy.yaml` | 001 | Smaller proxy-model config for validating code paths when full Laguna is unavailable. |
| `configs/activation_capture_h100.yaml` | 002 | Future config for replaying prompts through the teacher and capturing `(x_layer, y_teacher)` shards. |
| `configs/train_surrogate_layer.yaml` | 003 | Future config for training one dense surrogate layer from saved activations. |
| `configs/eval_tiny_coding.yaml` | 004 | Future config for running teacher/student coding evals against generated checkpoints. |

#### Library Modules

| File | Plan | Responsibility |
| --- | --- | --- |
| `src/densify/__init__.py` | 001 | Package marker and optional package version. No business logic. |
| `src/densify/config.py` | 001 | Dataclasses and YAML loading for run configs. Converts paths and generation settings into typed config objects. |
| `src/densify/teacher_loader.py` | 001 | Loads tokenizer and frozen teacher model with HF/PyTorch. Owns dtype mapping, `trust_remote_code`, `device_map`, eval mode, and parameter freezing. |
| `src/densify/model_introspection.py` | 001 | Scans `model.named_modules()` for candidate Transformer/MoE modules and writes architecture summaries. Later plans can tighten Laguna-specific detection here. |
| `src/densify/prompt_data.py` | 001 | Defines the prompt row schema and JSONL prompt loading. No model code and no generation code. |
| `src/densify/generation.py` | 001 | Formats coding prompts and calls `model.generate`. Returns raw text, latency, and generated-token counts. |
| `src/densify/code_scoring.py` | 001 | Extracts Python code from generations, checks syntax with `ast.parse`, and executes tiny unit tests in-process for smoke scoring. |
| `src/densify/run_artifacts.py` | 001 | Creates run directories and writes JSON/JSONL artifacts in stable schemas. |
| `src/densify/activation_capture.py` | 002 | Future module for registering forward pre-hooks/hooks, capturing `x_layer` and `y_teacher`, moving tensors to CPU, and writing activation shards. |
| `src/densify/surrogate_modules.py` | 003 | Future module defining `DenseSurrogateMLP` and initialization helpers. |
| `src/densify/training_data.py` | 003 | Future module for loading activation shards as PyTorch datasets/dataloaders. |
| `src/densify/evals.py` | 004 | Future module for reusable eval runners across tiny coding, HumanEval/MBPP, CruxEval, and SWE-bench prompt subsets. |
| `src/densify/metrics.py` | 004 | Future module for common metric calculations: parse rate, pass rate, latency, tokens/sec, reconstruction metrics, and summary tables. |

#### Script Entry Points

| File | Plan | Responsibility |
| --- | --- | --- |
| `scripts/build_python_smoke_prompts.py` | 001 | Writes the committed handcrafted Python smoke prompt file. |
| `scripts/smoke_load_teacher.py` | 001 | Loads tokenizer/model, prints basic metadata, and writes `architecture.json`. |
| `scripts/run_teacher_smoke_eval.py` | 001 | End-to-end teacher baseline: load prompts, generate completions, score code, and write run artifacts. |
| `scripts/build_swebench_verified_tiny.py` | 001 | Converts a small Python-filtered SWE-bench Verified slice into prompt JSONL. This is a prompt source only, not full SWE-bench evaluation. |
| `scripts/collect_rollouts.py` | 002 | Future script for generating/saving teacher rollouts from JSONL prompts or pool-derived prompts. |
| `scripts/capture_activations.py` | 002 | Future script for replaying prompts with hooks and saving activation shards. |
| `scripts/train_surrogate_layer.py` | 003 | Future script for training one surrogate layer from activation shards. |
| `scripts/assemble_densified_model.py` | 003 | Future script for swapping trained surrogates into the teacher architecture and writing a densified checkpoint manifest. |
| `scripts/run_eval.py` | 004 | Future script for evaluating teacher and densified checkpoints using a shared config. |

#### Data and Artifact Directories

| Path | Plan | Responsibility |
| --- | --- | --- |
| `data/prompts/*.jsonl` | 001+ | Small committed prompt definitions. Each row should have `id`, `prompt`, optional `entrypoint`, and optional `tests`. |
| `data/rollouts/teacher_smoke/` | 002 | Generated teacher continuations for later replay. Large files stay uncommitted. |
| `data/rollouts/pool_agent/` | 002 | Future pool/coding-agent trajectories converted into replayable prompt turns. |
| `data/activations/teacher_laguna_xs2/layer_XX/` | 002 | Activation shard storage by teacher/model/run/layer. Never committed. |
| `data/evals/*_results/` | 004 | Raw eval outputs, usually uncommitted unless converted into final report tables. |
| `runs/teacher_smoke/` | 001 | Per-run teacher smoke artifacts: resolved config, architecture summary, generations, metrics, and examples. |
| `runs/activation_capture/` | 002 | Future activation-capture logs and shard manifests. |
| `runs/training/` | 003 | Future surrogate training logs, losses, and checkpoint manifests. |
| `runs/evals/` | 004 | Future evaluation run summaries. |
| `runs/inference_benchmarks/` | 004 | Future memory/tokens/sec benchmark outputs. |
| `checkpoints/surrogate_layers/` | 003 | Future trained dense replacement layers. Never committed. |
| `checkpoints/densified_model/` | 003 | Future assembled densified model checkpoints. Never committed. |
| `inventory/*.md` | 002+ | Human-written registry of what models/data/hardware/checkpoints/experiments exist and why. |
| `reports/` | 004 | Final hackathon-facing tables, figures, and summary. |

#### Tests

| File | Plan | Responsibility |
| --- | --- | --- |
| `tests/test_prompt_data.py` | 001 | Verifies JSONL prompt loading and row schema. |
| `tests/test_code_scoring.py` | 001 | Verifies markdown code extraction, syntax checks, and smoke unit-test execution. |
| `tests/test_run_artifacts.py` | 001 | Verifies run directory creation and JSON/JSONL writing. |
| `tests/test_model_introspection.py` | 001 | Verifies candidate MoE/module scanning on fake modules. |
| `tests/test_activation_capture.py` | 002 | Future hook tests using a fake block to confirm input/output capture semantics. |
| `tests/test_surrogate_modules.py` | 003 | Future shape tests for dense surrogate MLPs and initialization helpers. |

### Git Tracking Policy

Commit:

```text
source code
scripts
configs
tests
small prompt JSONL files
inventory markdown
final report markdown/figures/tables
```

Do not commit:

```text
runs/
data/rollouts/
data/activations/
checkpoints/
large eval dumps
Hugging Face model cache
Prime VM logs
```

The `.gitignore` should include:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.ty/
runs/
data/rollouts/
data/activations/
data/evals/*_results/
checkpoints/
reports/raw/
```

### Tooling Plan

Use `uv` for environment creation and command execution:

```bash
uv venv --python 3.11
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ty check src tests scripts
```

Use `ruff` for linting and formatting. Use `ty` for fast type checks once the scaffold exists. Type checking should be advisory during the hackathon: helpful for catching schema mistakes, but not a blocker if model remote-code types are dynamic.

## Implementation Plan

### Task 1: Add Python Project Skeleton

**Files:**

- Create: `.python-version`
- Modify: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/densify/__init__.py`

- [ ] **Step 1: Create `.python-version`**

```text
3.11
```

- [ ] **Step 2: Update `.gitignore`**

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.ty/
runs/
data/rollouts/
data/activations/
data/evals/*_results/
checkpoints/
reports/raw/
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "laguna-xs2-densify"
version = "0.1.0"
description = "MoE densification experiments for Laguna XS.2"
requires-python = ">=3.11"
dependencies = [
  "accelerate>=0.34",
  "datasets>=2.20",
  "einops>=0.8",
  "numpy>=1.26",
  "pyyaml>=6.0",
  "safetensors>=0.4",
  "torch>=2.3",
  "tqdm>=4.66",
  "transformers>=4.44"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "ruff>=0.6",
  "ty>=0.0.1a16"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 4: Create `src/densify/__init__.py`**

```python
"""Utilities for Laguna XS.2 MoE densification experiments."""
```

- [ ] **Step 5: Install with uv**

Run:

```bash
uv venv --python 3.11
uv sync --extra dev
```

Expected:

```text
Resolved ...
Installed ...
```

- [ ] **Step 6: Verify toolchain**

Run:

```bash
uv run pytest --version
uv run ruff --version
uv run ty --version
```

Expected:

```text
pytest ...
ruff ...
ty ...
```

- [ ] **Step 7: Commit**

```bash
git add .python-version .gitignore pyproject.toml uv.lock src/densify/__init__.py
git commit -m "Add Python project skeleton"
```

### Task 2: Add Config Loading

**Files:**

- Create: `src/densify/config.py`
- Create: `configs/teacher_smoke_h100.yaml`
- Create: `configs/teacher_smoke_proxy.yaml`

- [ ] **Step 1: Implement config loader**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    batch_size: int


@dataclass(frozen=True)
class TeacherSmokeConfig:
    model_id: str
    torch_dtype: str
    trust_remote_code: bool
    device_map: str
    prompt_path: Path
    output_dir: Path
    generation: GenerationConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def load_teacher_smoke_config(path: str | Path) -> TeacherSmokeConfig:
    raw = load_yaml(path)
    gen = raw["generation"]
    return TeacherSmokeConfig(
        model_id=str(raw["model_id"]),
        torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        device_map=str(raw.get("device_map", "auto")),
        prompt_path=Path(raw["prompt_path"]),
        output_dir=Path(raw["output_dir"]),
        generation=GenerationConfig(
            max_new_tokens=int(gen.get("max_new_tokens", 256)),
            temperature=float(gen.get("temperature", 0.2)),
            top_p=float(gen.get("top_p", 0.95)),
            do_sample=bool(gen.get("do_sample", False)),
            batch_size=int(gen.get("batch_size", 1)),
        ),
    )
```

- [ ] **Step 2: Add H100 config**

```yaml
model_id: poolside/Laguna-XS.2-FP8
torch_dtype: bfloat16
trust_remote_code: true
device_map: auto
prompt_path: data/prompts/python_smoke.jsonl
output_dir: runs/teacher_smoke
generation:
  max_new_tokens: 256
  temperature: 0.2
  top_p: 0.95
  do_sample: false
  batch_size: 1
```

- [ ] **Step 3: Add proxy config**

```yaml
model_id: Qwen/Qwen1.5-MoE-A2.7B-Chat
torch_dtype: bfloat16
trust_remote_code: true
device_map: auto
prompt_path: data/prompts/python_smoke.jsonl
output_dir: runs/teacher_smoke_proxy
generation:
  max_new_tokens: 192
  temperature: 0.2
  top_p: 0.95
  do_sample: false
  batch_size: 1
```

- [ ] **Step 4: Commit**

```bash
git add src/densify/config.py configs/teacher_smoke_h100.yaml configs/teacher_smoke_proxy.yaml
git commit -m "Add teacher smoke config loader"
```

### Task 3: Add Python Smoke Prompt Dataset

**Files:**

- Create: `scripts/build_python_smoke_prompts.py`
- Generated: `data/prompts/python_smoke.jsonl`
- Test: `tests/test_prompt_data.py`
- Create: `src/densify/prompt_data.py`

- [ ] **Step 1: Implement prompt row loader**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingPrompt:
    id: str
    prompt: str
    entrypoint: str | None
    tests: list[str]


def load_jsonl_prompts(path: str | Path) -> list[CodingPrompt]:
    rows: list[CodingPrompt] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                CodingPrompt(
                    id=str(raw["id"]),
                    prompt=str(raw["prompt"]),
                    entrypoint=raw.get("entrypoint"),
                    tests=list(raw.get("tests", [])),
                )
            )
    if not rows:
        raise ValueError(f"No prompts loaded from {path}")
    return rows
```

- [ ] **Step 2: Add tests for prompt loading**

```python
import json

from densify.prompt_data import load_jsonl_prompts


def test_load_jsonl_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sort_001",
                "prompt": "Write sort_numbers.",
                "entrypoint": "sort_numbers",
                "tests": ["assert sort_numbers([2, 1]) == [1, 2]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_jsonl_prompts(path)

    assert prompts[0].id == "sort_001"
    assert prompts[0].entrypoint == "sort_numbers"
    assert prompts[0].tests == ["assert sort_numbers([2, 1]) == [1, 2]"]
```

- [ ] **Step 3: Add prompt builder script**

```python
from __future__ import annotations

import json
from pathlib import Path


PROMPTS = [
    {
        "id": "sort_001",
        "prompt": "Write a Python function sort_numbers(xs) that returns the numbers in ascending order.",
        "entrypoint": "sort_numbers",
        "tests": [
            "assert sort_numbers([3, 1, 2]) == [1, 2, 3]",
            "assert sort_numbers([]) == []",
        ],
    },
    {
        "id": "binary_search_001",
        "prompt": "Write a Python function binary_search(xs, target) that returns the index of target in sorted list xs, or -1 if target is absent.",
        "entrypoint": "binary_search",
        "tests": [
            "assert binary_search([1, 3, 5, 7], 5) == 2",
            "assert binary_search([1, 3, 5, 7], 2) == -1",
        ],
    },
    {
        "id": "diff_paths_001",
        "prompt": "Write a Python function changed_files(diff_text) that parses a unified diff string and returns a sorted list of changed file paths.",
        "entrypoint": "changed_files",
        "tests": [
            "d = 'diff --git a/a.py b/a.py\\n--- a/a.py\\n+++ b/a.py\\n@@ -1 +1 @@\\n-x\\n+y\\n'",
            "assert changed_files(d) == ['a.py']",
        ],
    },
    {
        "id": "off_by_one_001",
        "prompt": "Write a Python function count_items(xs) that returns the number of items in xs without using len().",
        "entrypoint": "count_items",
        "tests": [
            "assert count_items([]) == 0",
            "assert count_items(['a', 'b', 'c']) == 3",
        ],
    },
    {
        "id": "traceback_001",
        "prompt": "Write a Python function safe_divide(a, b) that returns None when b is zero and otherwise returns a / b.",
        "entrypoint": "safe_divide",
        "tests": [
            "assert safe_divide(6, 2) == 3",
            "assert safe_divide(6, 0) is None",
        ],
    },
]


def main() -> None:
    out = Path("data/prompts/python_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in PROMPTS:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(PROMPTS)} prompts to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate prompt file**

Run:

```bash
python scripts/build_python_smoke_prompts.py
```

Expected:

```text
Wrote 5 prompts to data/prompts/python_smoke.jsonl
```

- [ ] **Step 5: Run prompt tests**

Run:

```bash
pytest tests/test_prompt_data.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/densify/prompt_data.py tests/test_prompt_data.py scripts/build_python_smoke_prompts.py data/prompts/python_smoke.jsonl
git commit -m "Add Python smoke prompt dataset"
```

### Task 4: Add Teacher Loader

**Files:**

- Create: `src/densify/teacher_loader.py`
- Create: `scripts/smoke_load_teacher.py`

- [ ] **Step 1: Implement loader**

```python
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_teacher_model(
    model_id: str,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    device_map: str = "auto",
):
    dtype = DTYPES.get(torch_dtype)
    if dtype is None:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}; expected one of {sorted(DTYPES)}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
```

- [ ] **Step 2: Implement smoke loader script**

```python
from __future__ import annotations

import argparse

from densify.config import load_teacher_smoke_config
from densify.teacher_loader import load_teacher_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_teacher_smoke_config(args.config)
    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    model = load_teacher_model(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
    )

    print(f"model_id={cfg.model_id}")
    print(f"tokenizer_vocab={len(tokenizer)}")
    print(f"model_class={model.__class__.__name__}")
    print(f"num_parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run proxy load locally or on VM**

Run:

```bash
python scripts/smoke_load_teacher.py --config configs/teacher_smoke_proxy.yaml
```

Expected:

```text
model_id=...
tokenizer_vocab=...
model_class=...
num_parameters=...
```

- [ ] **Step 4: Run Laguna load on H100 VM**

Run:

```bash
python scripts/smoke_load_teacher.py --config configs/teacher_smoke_h100.yaml
```

Expected:

```text
model_id=poolside/Laguna-XS.2-FP8
tokenizer_vocab=...
model_class=...
num_parameters=...
```

- [ ] **Step 5: Commit**

```bash
git add src/densify/teacher_loader.py scripts/smoke_load_teacher.py
git commit -m "Add HF teacher loader"
```

### Task 5: Add Architecture Introspection

**Files:**

- Create: `src/densify/model_introspection.py`
- Modify: `scripts/smoke_load_teacher.py`
- Test: `tests/test_model_introspection.py`

- [ ] **Step 1: Implement module scanner**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleSummary:
    name: str
    class_name: str
    parameter_count: int


def count_direct_parameters(module: Any) -> int:
    return sum(param.numel() for param in module.parameters(recurse=False))


def find_candidate_moe_modules(model: Any) -> list[ModuleSummary]:
    candidates: list[ModuleSummary] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        child_names = {child_name.lower() for child_name, _ in module.named_children()}
        looks_like_moe = (
            "moe" in cls
            or "expert" in cls
            or "experts" in child_names
            or "gate" in child_names and "shared_expert" in child_names
        )
        if looks_like_moe:
            candidates.append(
                ModuleSummary(
                    name=name,
                    class_name=module.__class__.__name__,
                    parameter_count=count_direct_parameters(module),
                )
            )
    return candidates


def architecture_summary(model: Any, model_id: str, torch_dtype: str) -> dict[str, Any]:
    transformer_layers = [
        name
        for name, module in model.named_modules()
        if name.endswith(".layers") or module.__class__.__name__.lower().endswith("decoderlayer")
    ]
    candidates = find_candidate_moe_modules(model)
    return {
        "model_id": model_id,
        "torch_dtype": torch_dtype,
        "model_class": model.__class__.__name__,
        "num_parameters_seen": sum(p.numel() for p in model.parameters()),
        "num_transformer_layer_containers": len(transformer_layers),
        "candidate_moe_modules": [asdict(item) for item in candidates],
    }
```

- [ ] **Step 2: Add a unit test with fake modules**

```python
import torch

from densify.model_introspection import find_candidate_moe_modules


class FakeExpertBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)
        self.shared_expert = torch.nn.Linear(4, 4)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeExpertBlock()])


def test_find_candidate_moe_modules_detects_shared_expert_pattern():
    model = FakeModel()
    found = find_candidate_moe_modules(model)

    assert any(item.name == "layers.0" for item in found)
```

- [ ] **Step 3: Update smoke loader to write architecture summary**

Add this block after loading the model:

```python
import json
from pathlib import Path

from densify.model_introspection import architecture_summary

summary = architecture_summary(model, cfg.model_id, cfg.torch_dtype)
out_dir = Path(cfg.output_dir) / "load_smoke"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "architecture.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"candidate_moe_modules={len(summary['candidate_moe_modules'])}")
print(f"wrote={out_dir / 'architecture.json'}")
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_model_introspection.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/densify/model_introspection.py tests/test_model_introspection.py scripts/smoke_load_teacher.py
git commit -m "Add teacher architecture introspection"
```

### Task 6: Add Generation and Code Scoring

**Files:**

- Create: `src/densify/generation.py`
- Create: `src/densify/code_scoring.py`
- Test: `tests/test_code_scoring.py`

- [ ] **Step 1: Implement prompt formatting and generation**

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from densify.config import GenerationConfig
from densify.prompt_data import CodingPrompt


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_s: float
    generated_tokens: int


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def format_coding_prompt(prompt: CodingPrompt) -> str:
    return (
        "You are a careful Python coding assistant. "
        "Return a single Python code block with the requested function.\n\n"
        f"Task:\n{prompt.prompt}\n"
    )


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: CodingPrompt, cfg: GenerationConfig) -> GenerationResult:
    formatted = format_coding_prompt(prompt)
    device = first_model_device(model)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature if cfg.do_sample else None,
        top_p=cfg.top_p if cfg.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency_s = time.perf_counter() - start
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GenerationResult(
        text=text,
        latency_s=latency_s,
        generated_tokens=int(generated.numel()),
    )
```

- [ ] **Step 2: Implement code extraction and test scoring**

```python
from __future__ import annotations

import ast
import contextlib
import io
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeScore:
    extracted_code: str
    parse_ok: bool
    tests_ok: bool
    test_stdout: str
    test_stderr: str


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def score_code_generation(text: str, tests: list[str]) -> CodeScore:
    code = extract_python_code(text)
    parsed = parse_ok(code)
    if not parsed:
        return CodeScore(code, False, False, "", "syntax error")

    namespace: dict[str, object] = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace)
            for test in tests:
                exec(test, namespace)
    except Exception as exc:
        return CodeScore(code, True, False, stdout.getvalue(), f"{type(exc).__name__}: {exc}")

    return CodeScore(code, True, True, stdout.getvalue(), stderr.getvalue())
```

- [ ] **Step 3: Add scoring tests**

```python
from densify.code_scoring import extract_python_code, score_code_generation


def test_extract_python_code_from_markdown_block():
    text = "Here is code:\n```python\ndef f():\n    return 1\n```"
    assert extract_python_code(text) == "def f():\n    return 1"


def test_score_code_generation_passes_tests():
    score = score_code_generation(
        "```python\ndef add_one(x):\n    return x + 1\n```",
        ["assert add_one(2) == 3"],
    )
    assert score.parse_ok is True
    assert score.tests_ok is True


def test_score_code_generation_reports_failed_tests():
    score = score_code_generation(
        "```python\ndef add_one(x):\n    return x\n```",
        ["assert add_one(2) == 3"],
    )
    assert score.parse_ok is True
    assert score.tests_ok is False
    assert "AssertionError" in score.test_stderr
```

- [ ] **Step 4: Run scoring tests**

Run:

```bash
pytest tests/test_code_scoring.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/densify/generation.py src/densify/code_scoring.py tests/test_code_scoring.py
git commit -m "Add teacher generation scoring utilities"
```

### Task 7: Add Run Artifact Writer

**Files:**

- Create: `src/densify/run_artifacts.py`
- Test: `tests/test_run_artifacts.py`

- [ ] **Step 1: Implement artifact paths and JSONL writer**

```python
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base_dir) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row)) + "\n")
```

- [ ] **Step 2: Add tests**

```python
import json

from densify.run_artifacts import append_jsonl, new_run_dir, write_json


def test_new_run_dir_creates_unique_prefixed_dir(tmp_path):
    run_dir = new_run_dir(tmp_path, prefix="teacher")
    assert run_dir.exists()
    assert run_dir.name.startswith("teacher_")


def test_write_json_and_append_jsonl(tmp_path):
    json_path = tmp_path / "summary.json"
    jsonl_path = tmp_path / "rows.jsonl"

    write_json(json_path, {"ok": True})
    append_jsonl(jsonl_path, {"id": "a"})
    append_jsonl(jsonl_path, {"id": "b"})

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "a"}, {"id": "b"}]
```

- [ ] **Step 3: Run tests**

Run:

```bash
pytest tests/test_run_artifacts.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Commit**

```bash
git add src/densify/run_artifacts.py tests/test_run_artifacts.py
git commit -m "Add run artifact helpers"
```

### Task 8: Add End-to-End Teacher Smoke Eval Script

**Files:**

- Create: `scripts/run_teacher_smoke_eval.py`

- [ ] **Step 1: Implement script**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.code_scoring import score_code_generation
from densify.config import load_teacher_smoke_config
from densify.generation import generate_one
from densify.model_introspection import architecture_summary
from densify.prompt_data import load_jsonl_prompts
from densify.run_artifacts import append_jsonl, new_run_dir, write_json
from densify.teacher_loader import load_teacher_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_teacher_smoke_config(args.config)
    prompts = load_jsonl_prompts(cfg.prompt_path)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    run_dir = new_run_dir(cfg.output_dir, prefix="teacher_smoke")
    write_json(run_dir / "config_resolved.yaml.json", cfg)

    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    model = load_teacher_model(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
    )
    write_json(run_dir / "architecture.json", architecture_summary(model, cfg.model_id, cfg.torch_dtype))

    generations_path = run_dir / "generations.jsonl"
    counters = {
        "num_prompts": 0,
        "non_empty": 0,
        "python_like": 0,
        "parse_ok": 0,
        "tests_ok": 0,
        "total_generated_tokens": 0,
        "total_latency_s": 0.0,
    }

    examples: list[str] = []
    for prompt in prompts:
        result = generate_one(model, tokenizer, prompt, cfg.generation)
        score = score_code_generation(result.text, prompt.tests)
        row = {
            "id": prompt.id,
            "prompt": prompt.prompt,
            "raw_generation": result.text,
            "extracted_code": score.extracted_code,
            "parse_ok": score.parse_ok,
            "tests_ok": score.tests_ok,
            "test_stdout": score.test_stdout,
            "test_stderr": score.test_stderr,
            "latency_s": result.latency_s,
            "generated_tokens": result.generated_tokens,
        }
        append_jsonl(generations_path, row)

        counters["num_prompts"] += 1
        counters["non_empty"] += int(bool(result.text.strip()))
        counters["python_like"] += int("def " in score.extracted_code or "```python" in result.text.lower())
        counters["parse_ok"] += int(score.parse_ok)
        counters["tests_ok"] += int(score.tests_ok)
        counters["total_generated_tokens"] += result.generated_tokens
        counters["total_latency_s"] += result.latency_s

        if len(examples) < 3:
            examples.append(
                f"## {prompt.id}\n\nPrompt:\n{prompt.prompt}\n\nGeneration:\n```python\n{score.extracted_code}\n```\n"
            )

    n = max(counters["num_prompts"], 1)
    summary = {
        **counters,
        "non_empty_rate": counters["non_empty"] / n,
        "python_like_rate": counters["python_like"] / n,
        "parse_ok_rate": counters["parse_ok"] / n,
        "tests_ok_rate": counters["tests_ok"] / n,
        "tokens_per_second": counters["total_generated_tokens"] / max(counters["total_latency_s"], 1e-6),
    }
    write_json(run_dir / "summary.json", summary)
    Path(run_dir / "examples.md").write_text("\n".join(examples), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote={run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run proxy smoke eval**

Run:

```bash
python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_proxy.yaml --limit 2
```

Expected:

```text
{
  "num_prompts": 2,
  ...
}
wrote=runs/teacher_smoke_proxy/teacher_smoke_...
```

- [ ] **Step 3: Run Laguna teacher smoke eval on H100**

Run:

```bash
python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_h100.yaml --limit 5
```

Expected:

```text
summary.json exists
generations.jsonl exists
examples.md contains readable Python outputs
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_teacher_smoke_eval.py
git commit -m "Add end-to-end teacher smoke eval"
```

### Task 9: Add Python SWE-bench Verified Tiny Builder

**Files:**

- Create: `scripts/build_swebench_verified_tiny.py`

- [ ] **Step 1: Implement dataset builder**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


PYTHON_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default="data/prompts/swebench_verified_python_tiny.jsonl")
    args = parser.parse_args()

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in dataset:
            repo = row.get("repo", "")
            if repo not in PYTHON_REPOS:
                continue
            prompt = (
                "You are fixing a Python repository issue. "
                "Write a concise diagnosis and the likely patch strategy.\n\n"
                f"Repository: {repo}\n"
                f"Issue:\n{row.get('problem_statement', '')}\n"
            )
            f.write(
                json.dumps(
                    {
                        "id": row["instance_id"],
                        "prompt": prompt,
                        "entrypoint": None,
                        "tests": [],
                    }
                )
                + "\n"
            )
            count += 1
            if count >= args.limit:
                break

    print(f"Wrote {count} prompts to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build tiny SWE-bench prompt file**

Run:

```bash
python scripts/build_swebench_verified_tiny.py --limit 5
```

Expected:

```text
Wrote 5 prompts to data/prompts/swebench_verified_python_tiny.jsonl
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_swebench_verified_tiny.py data/prompts/swebench_verified_python_tiny.jsonl
git commit -m "Add Python SWE-bench Verified prompt builder"
```

### Task 10: Document Runbook and Gate Decision

**Files:**

- Create: `docs/runbooks/teacher-smoke-generation.md`
- Modify: `docs/MASTER_PLAN.md`

- [ ] **Step 1: Create runbook**

````markdown
# Teacher Smoke Generation Runbook

## Goal

Run the unmodified teacher model through HF/PyTorch on a small Python coding set and verify that generations are sane before activation capture or surrogate training.

## Commands

```bash
python -m pip install -e ".[dev]"
python scripts/build_python_smoke_prompts.py
python scripts/smoke_load_teacher.py --config configs/teacher_smoke_h100.yaml
python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_h100.yaml --limit 5
```

## Pass Gate

Promote to activation capture only if:

- `architecture.json` lists candidate MoE modules.
- `generations.jsonl` has one row per prompt.
- `summary.json` reports non-empty generations.
- At least three examples in `examples.md` are readable Python attempts.

## Failure Handling

- If the model does not load, run the proxy config and inspect Laguna remote-code support.
- If generations are empty, check tokenizer EOS/PAD config and chat template.
- If snippets fail parsing, inspect `examples.md` before changing prompts.
- If H100 memory is insufficient, retry with batch size 1 and lower `max_new_tokens`.
````

- [ ] **Step 2: Add link from master plan**

Add under `Minimal End-to-End Milestone`:

```markdown
Before activation capture, complete [Teacher Smoke Generation Implementation Plan](../eng_plans/001-teacher-smoke-generation.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/runbooks/teacher-smoke-generation.md docs/MASTER_PLAN.md
git commit -m "Document teacher smoke generation runbook"
```

## Early Spikes

Run these before spending serious GPU time:

1. **HF access spike**
   - Command: `python scripts/smoke_load_teacher.py --config configs/teacher_smoke_h100.yaml`
   - Answer: do we have auth and can HF remote code load the checkpoint?

2. **Architecture spike**
   - Artifact: `runs/teacher_smoke/load_smoke/architecture.json`
   - Answer: what are the exact module names for MoE blocks?

3. **Prompting spike**
   - Command: `python scripts/run_teacher_smoke_eval.py --config configs/teacher_smoke_h100.yaml --limit 5`
   - Answer: do default prompts produce code, or do we need a chat template / tool format?

4. **Memory spike**
   - Observe `nvidia-smi` during load and generation.
   - Answer: can we run generation and later hooks on one H100?

5. **SWE-bench prompt spike**
   - Command: `python scripts/build_swebench_verified_tiny.py --limit 5`
   - Answer: are issue prompts useful as text-generation smoke prompts, or should SWE-bench wait for pool?

## Handoff To Training RFC

Once this plan passes, [RFC 002: Training](../rfcs/002-training.md) can start with confidence that:

```text
teacher loading works
prompt format is usable
model internals are introspectable
generation artifacts are reproducible
we have a small Python coding scaffold for regression checks
```

The immediate next engineering plan should be activation capture for one MoE layer:

```text
teacher forward with hooks
  -> capture x_layer and y_teacher
  -> write activation shard
  -> verify tensor shapes and disk size
```
