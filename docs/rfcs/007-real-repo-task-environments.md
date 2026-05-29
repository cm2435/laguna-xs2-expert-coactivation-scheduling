# RFC 007: Real Repo Task Environments

## Purpose

Define how we store and run coding tasks so model rollouts happen in real repository environments without contaminating task metadata, graders, or future runs.

The core design is:

```text
central task store
  immutable task descriptions
  hidden/public graders
  repo source metadata

per-repo environment store
  clean cloned repo templates
  dependency setup scripts
  optional cached venvs/containers

per-run sandbox
  one disposable copy per model attempt
  pool is initialized and run inside this copy
  model sees the repo, not the hidden grader implementation
```

This matters because activation/rollout data should come from realistic coding behavior, not from prompts that accidentally leak solutions or graders.

## Motivation

For densification, our training data should include real coding trajectories:

```text
read files
inspect tests
run commands
edit code
observe failures
produce patches
```

But if we casually run everything in one mutable repo directory, we will lose reproducibility and may leak evaluation details into model context. We need an explicit layout that separates:

- task descriptions,
- repos,
- graders,
- sandboxes,
- model rollouts,
- final scored artifacts.

## Terms

### Task

A task is the immutable problem definition:

```text
task_id
repo_id
base_commit
prompt / issue statement
allowed files or scope if any
public tests visible to the model
hidden grader command not shown to the model
timeout / resource limits
```

### Repo Template

A clean checkout of a repository at a known commit. This is never mutated by a model run.

### Sandbox

A disposable working copy created from a repo template for one rollout attempt. Pool runs in the sandbox.

### Grader

A command or script that scores a completed sandbox after the model finishes. Hidden graders should live outside the sandbox or be copied in only after the model is done.

## Recommended Layout

```text
tasks/
  registry.jsonl
  tiny_repo_bugs/
    sort_bug_001.yaml
    diff_parser_001.yaml
  swebench_verified/
    astropy__astropy-12907.yaml
    astropy__astropy-13033.yaml
  graders/
    tiny_repo_bugs/
      sort_bug_001/
        public_tests.py
        hidden_tests.py
        grade.sh
    swebench_verified/
      astropy__astropy-12907/
        grade.sh

envs/
  repo_templates/
    astropy__astropy/
      5b0c.../
        repo/
        metadata.json
    tiny_repo_bugs/
      sort_bug_001/
        repo/
        metadata.json
  build_cache/
    astropy__astropy/
      py311/
        venv/ or wheelhouse/

sandboxes/
  pool_runs/
    {run_id}/
      repo/
      task.yaml
      public_tests/
      pool_logs/
      model_artifacts/
      patch.diff
      grade_result.json
```

Add these paths to `.gitignore` unless they contain small synthetic task definitions:

```text
envs/repo_templates/
envs/build_cache/
sandboxes/
```

Commit only:

```text
tasks/*.yaml
tasks/registry.jsonl
small synthetic graders
scripts that build envs/sandboxes
```

Do not commit large cloned repos, venvs, model artifacts, or full SWE-bench checkouts.

## Task Manifest Schema

Use YAML for human-authored tasks:

```yaml
task_id: tiny_sort_bug_001
suite: tiny_repo_bugs
repo_id: tiny_sort_bug_repo
base_commit: local
prompt: |
  The test suite is failing. Inspect the repository and fix the bug.
visible_to_model:
  issue_statement: true
  public_tests: true
  hidden_tests: false
environment:
  template_path: envs/repo_templates/tiny_repo_bugs/sort_bug_001/repo
  setup_command: python -m pip install -e .
  smoke_command: pytest tests/test_public.py -q
grader:
  public_command: pytest tests/test_public.py -q
  hidden_command: bash tasks/graders/tiny_repo_bugs/sort_bug_001/grade.sh
limits:
  timeout_s: 600
  max_turns: 20
  network: disabled
```

`registry.jsonl` should provide an index:

```json
{"task_id":"tiny_sort_bug_001","suite":"tiny_repo_bugs","manifest":"tasks/tiny_repo_bugs/sort_bug_001.yaml","repo_id":"tiny_sort_bug_repo"}
```

## Anti-Cheat Boundaries

The model may see:

```text
issue statement
repository files
public tests if the task includes them
stdout/stderr from commands it runs
```

The model must not see:

```text
hidden tests
grader scripts
gold patches
expected final answers
previous successful rollouts
score logs from hidden graders
```

Implementation rules:

1. Hidden graders live outside `sandboxes/.../repo`.
2. Public tests may be copied into the sandbox if the task is designed that way.
3. Hidden tests are mounted or copied only after the model finishes.
4. Pool runs from the sandbox repo directory.
5. Model artifacts are written outside the repo tree where possible.
6. If artifacts must be inside the sandbox, put them under `sandboxes/{run_id}/model_artifacts`, not inside `repo/`.

## Pool Runtime Model

Each rollout should execute like:

```text
1. read task manifest
2. copy repo template to new sandbox
3. copy public task files/tests into sandbox if needed
4. start pool in sandbox/repo
5. pool runs the task with our HF/PyTorch model backend
6. collect pool logs, model calls, token traces, activations
7. export patch.diff
8. run hidden grader outside model visibility
9. write grade_result.json
```

Conceptual command:

```bash
uv run python scripts/run_pool_task.py \
  --task tasks/tiny_repo_bugs/sort_bug_001.yaml \
  --model-backend hf_laguna \
  --output sandboxes/pool_runs
```

`run_pool_task.py` owns sandbox creation and grading. Pool owns the interactive agent loop inside the sandbox.

## Why Use One Subfolder Per Repo?

Yes, we should have one environment folder per repo/template.

Benefits:

- dependencies are installed once per repo family,
- base commit is explicit,
- repeated rollouts start from a clean state,
- multiple people can run different tasks without mutating each other's repos,
- graders can be tied to repo/task versions,
- activation data can reference `task_id`, `repo_id`, and `base_commit`.

The repo template should not be the same as the sandbox. Treat templates as read-only source material.

## Dataset Tiers

### Tier 0: Synthetic Tiny Repos

Hand-authored mini repos with 1-3 files and small tests.

Purpose:

```text
debug pool integration
debug sandboxing
debug hidden grader boundary
debug activation capture on real tool loops
```

Examples:

- sorting bug,
- off-by-one bug,
- diff parser bug,
- import/path bug,
- exception handling bug.

These should be committed if tiny enough, because they are our reliable smoke fixtures.

### Tier 1: Function Prompt Tasks

Current data:

```text
data/prompts/python_smoke.jsonl
```

Purpose:

```text
fast teacher/student generation sanity
no repo environment
cheap partial-swap eval
```

This is not a real environment tier.

### Tier 2: SWE-Bench Prompt-Only Tasks

Current data:

```text
data/prompts/swebench_verified_python_tiny.jsonl
```

Purpose:

```text
issue-understanding prompts
teacher continuation data
not a patch-validating benchmark yet
```

### Tier 3: SWE-Bench Real Repo Tasks

Use only once Tier 0 works.

Purpose:

```text
real repo patch attempts
real tests
real agentic trajectories
```

These should be stored as task manifests and repo templates, not as ad hoc checked-out directories.

## Grader Result Schema

Every task attempt writes:

```json
{
  "run_id": "20260529T193000Z_tiny_sort_bug_001_laguna_teacher",
  "task_id": "tiny_sort_bug_001",
  "repo_id": "tiny_sort_bug_repo",
  "base_commit": "local",
  "model_id": "poolside/Laguna-XS.2-FP8",
  "patch_path": "sandboxes/pool_runs/.../patch.diff",
  "public_tests": {
    "command": "pytest tests/test_public.py -q",
    "passed": true,
    "exit_code": 0
  },
  "hidden_tests": {
    "command": "bash tasks/graders/.../grade.sh",
    "passed": true,
    "exit_code": 0
  },
  "pool": {
    "turns": 8,
    "tool_calls": 14
  },
  "model_artifacts": {
    "tokens_captured": true,
    "activations_captured": false
  }
}
```

## Implementation Plan

### Step 1: Add Layout and Ignore Rules

Create:

```text
tasks/
tasks/graders/
envs/
sandboxes/
```

Ignore:

```text
envs/repo_templates/
envs/build_cache/
sandboxes/
```

### Step 2: Add One Tiny Repo Task

Create a tiny repo template:

```text
envs/repo_templates/tiny_repo_bugs/sort_bug_001/repo/
  pyproject.toml
  src/example_pkg/sorter.py
  tests/test_public.py
```

The bug should be simple but real.

Create hidden grader outside the repo:

```text
tasks/graders/tiny_repo_bugs/sort_bug_001/hidden_tests.py
tasks/graders/tiny_repo_bugs/sort_bug_001/grade.sh
```

Commit a tiny template only if it is small. If we do not want repo templates committed, create it with a builder script.

### Step 3: Add Sandbox Builder

Create:

```text
scripts/prepare_task_sandbox.py
```

Responsibilities:

```text
read task YAML
copy repo template to sandboxes/pool_runs/{run_id}/repo
copy public tests if needed
write resolved task.yaml
print sandbox path
```

### Step 4: Add Grader Runner

Create:

```text
scripts/grade_task_sandbox.py
```

Responsibilities:

```text
read task YAML
run public grader command inside sandbox/repo
run hidden grader command after model is done
write grade_result.json
```

### Step 5: Add Pool Task Runner

Create:

```text
scripts/run_pool_task.py
```

Responsibilities:

```text
prepare sandbox
launch pool in sandbox/repo
capture pool stdout/stderr
export patch.diff
run grader
write rollout summary
```

This script should later call the HF/PyTorch ACP backend from the Stage 2 engineering plan.

## Open Questions

1. Should tiny repo templates be committed or generated?
2. Should SWE-Bench environments be managed directly or via the official harness?
3. How do we keep pool's own logs out of the model-visible repo directory?
4. Should hidden graders run in Docker/venv to avoid dependency pollution?
5. How do we map pool tool-call turns to model activation captures cleanly?

## Recommendation

Use a central task store plus per-repo templates and per-run sandboxes.

Do not run pool directly inside the source repo or the repo template. Always create a disposable sandbox per rollout.

For the next implementation milestone, build one tiny synthetic repo task with public and hidden tests. Once that works, graduate to a five-task SWE-Bench real-repo subset.
