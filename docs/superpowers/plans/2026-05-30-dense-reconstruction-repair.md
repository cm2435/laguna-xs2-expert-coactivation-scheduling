# Dense Reconstruction Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Laguna XS.2 MoE-to-dense recovery pipeline so SFT starts from a competent reconstructed dense base, not a random routed-FFN placeholder.

**Architecture:** The repair has three gates. First, make the base checkpoint lineage explicit and impossible to mix up. Second, improve reconstruction with more and better-distributed data plus teacher logit KL. Third, only run tool-policy SFT once the dense base passes single-shot format and basic language sanity probes.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Laguna remote code, JSONL datasets, local/remote GPU training on `prime-gpu-31`.

---

## Current Diagnosis

The Sonnet recovery SFT failed for two reasons:

1. It accidentally trained from `checkpoints/laguna-xs2-dense-k8-copied-shell`, whose routed dense FFN weights are random placeholders.
2. The actual 2k reconstruction artifact, `runs/hf_upload/laguna-xs2-dense-k8-recon-2k`, is still undercooked for tool use: existing sanity checks show `0/5` rollout-prefix tool-call shape and repetitive output.

Therefore the next build should not start with more recovery SFT. It should repair and validate the dense reconstruction stage first.

## Files

- Modify: `scripts/train_dense_reconstruction.py`
  - Add optional teacher logit KL alongside MLP activation reconstruction.
  - Add train/validation metrics with base probes.
  - Add checkpoint metadata stating source init and reconstruction data mix.
- Create: `scripts/build_reconstruction_mixture.py`
  - Build a balanced JSONL mixture of code instruction rows, tool traces, and recovery/tool-format rows.
- Create: `scripts/run_base_tool_probe.py`
  - Single-shot base probe: can a checkpoint emit one valid Laguna tagged tool call?
- Modify: `scripts/train_dense_sft.py`
  - Add guardrails preventing SFT from accidentally using `copied-shell` placeholder unless explicitly allowed.
- Create: `tests/test_reconstruction_mixture.py`
- Create: `tests/test_base_tool_probe.py`
- Modify: `tests/test_train_dense_sft_utils.py`

---

### Task 1: Add Checkpoint Lineage Guardrails

**Files:**
- Modify: `scripts/train_dense_sft.py`
- Test: `tests/test_train_dense_sft_utils.py`

- [ ] **Step 1: Add a failing test that rejects placeholder bases**

Append this test to `tests/test_train_dense_sft_utils.py`:

```python
def test_rejects_copied_shell_placeholder_without_override(tmp_path):
    train = load_train_module()
    model_dir = tmp_path / "laguna-xs2-dense-k8-copied-shell"
    model_dir.mkdir()
    (model_dir / "copied_shell_report.json").write_text(
        '{"random_routed_dense_keys": 117, "copied_shared_expert_keys": 117}\n',
        encoding="utf-8",
    )

    try:
        train.assert_not_placeholder_base(str(model_dir), allow_placeholder_base=False)
    except SystemExit as exc:
        assert "random routed dense" in str(exc).lower()
    else:
        raise AssertionError("expected placeholder base rejection")


def test_allows_reconstructed_base_without_placeholder_report(tmp_path):
    train = load_train_module()
    model_dir = tmp_path / "laguna-xs2-dense-k8-recon-2k"
    model_dir.mkdir()

    train.assert_not_placeholder_base(str(model_dir), allow_placeholder_base=False)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run --no-sync pytest tests/test_train_dense_sft_utils.py::test_rejects_copied_shell_placeholder_without_override -q
```

Expected: fail because `assert_not_placeholder_base` does not exist.

- [ ] **Step 3: Implement the guard**

In `scripts/train_dense_sft.py`, add:

```python
def assert_not_placeholder_base(model_path: str, *, allow_placeholder_base: bool) -> None:
    if allow_placeholder_base:
        return
    report_path = Path(model_path) / "copied_shell_report.json"
    if not report_path.exists():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse copied shell report at {report_path}: {exc}") from exc
    random_keys = int(report.get("random_routed_dense_keys") or 0)
    if random_keys > 0:
        raise SystemExit(
            f"Refusing to SFT from placeholder base {model_path}: "
            f"{random_keys} random routed dense tensors remain. "
            "Use a reconstructed checkpoint or pass --allow-placeholder-base intentionally."
        )
```

Add an argument:

```python
parser.add_argument("--allow-placeholder-base", action="store_true")
```

Call the guard before loading the model:

```python
assert_not_placeholder_base(args.model, allow_placeholder_base=args.allow_placeholder_base)
```

- [ ] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/test_train_dense_sft_utils.py -q
```

Expected: all tests pass.

---

### Task 2: Build a Reconstruction Data Mixture

**Files:**
- Create: `scripts/build_reconstruction_mixture.py`
- Test: `tests/test_reconstruction_mixture.py`

- [ ] **Step 1: Write tests for mixture balancing**

Create `tests/test_reconstruction_mixture.py`:

```python
import json
from pathlib import Path

from scripts.build_reconstruction_mixture import build_mixture


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_mixture_interleaves_sources_and_records_source(tmp_path):
    code = tmp_path / "code.jsonl"
    tools = tmp_path / "tools.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    out = tmp_path / "mixed.jsonl"

    write_jsonl(code, [{"text": "code A"}, {"text": "code B"}])
    write_jsonl(tools, [{"messages": [{"role": "user", "content": "u"}]}])
    write_jsonl(recovery, [{"messages": [{"role": "assistant", "content": "<tool_call>shell</tool_call>"}]}])

    summary = build_mixture(
        code_path=code,
        tool_path=tools,
        recovery_path=recovery,
        output_path=out,
        max_rows=4,
        seed=123,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["_reconstruction_source"] for row in rows} == {"code", "tool", "recovery"}
    assert summary["rows_written"] == 4
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run --no-sync pytest tests/test_reconstruction_mixture.py -q
```

Expected: fail because the script does not exist.

- [ ] **Step 3: Implement the mixture builder**

Create `scripts/build_reconstruction_mixture.py`:

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tag_rows(rows: Iterable[dict], source: str) -> list[dict]:
    tagged = []
    for row in rows:
        clean = dict(row)
        clean["_reconstruction_source"] = source
        tagged.append(clean)
    return tagged


def round_robin(groups: list[list[dict]], max_rows: int) -> list[dict]:
    output: list[dict] = []
    index = 0
    while len(output) < max_rows and any(index < len(group) for group in groups):
        for group in groups:
            if index < len(group):
                output.append(group[index])
                if len(output) >= max_rows:
                    break
        index += 1
    return output


def build_mixture(
    *,
    code_path: Path,
    tool_path: Path,
    recovery_path: Path,
    output_path: Path,
    max_rows: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    groups = [
        tag_rows(read_jsonl(code_path), "code"),
        tag_rows(read_jsonl(tool_path), "tool"),
        tag_rows(read_jsonl(recovery_path), "recovery"),
    ]
    for group in groups:
        rng.shuffle(group)
    rows = round_robin(groups, max_rows=max_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows_written": len(rows),
        "sources": {source: sum(row["_reconstruction_source"] == source for row in rows) for source in ["code", "tool", "recovery"]},
        "seed": seed,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--tool-path", type=Path, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(json.dumps(build_mixture(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/test_reconstruction_mixture.py -q
```

Expected: pass.

---

### Task 3: Add Teacher Logit KL to Reconstruction

**Files:**
- Modify: `src/densify/reconstruction.py`
- Modify: `scripts/train_dense_reconstruction.py`
- Test: `tests/test_dense_reconstruction.py`

- [ ] **Step 1: Add a KL unit test on tiny models**

Append to `tests/test_dense_reconstruction.py`:

```python
def test_reconstruction_can_include_logit_kl():
    teacher = TinyModel()
    student = TinyModel()
    freeze_for_dense_reconstruction(student)
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }

    result = compute_parallel_reconstruction_loss(
        teacher,
        student,
        batch,
        layer_ids=[1],
        cosine_weight=0.0,
        logit_kl_weight=0.1,
    )

    assert result.loss.requires_grad
    assert "logit_kl" in result.per_layer[-1]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run --no-sync pytest tests/test_dense_reconstruction.py::test_reconstruction_can_include_logit_kl -q
```

Expected: fail because `logit_kl_weight` is not supported.

- [ ] **Step 3: Implement KL in `compute_parallel_reconstruction_loss`**

Change the signature:

```python
def compute_parallel_reconstruction_loss(
    teacher: nn.Module,
    student: nn.Module,
    batch: dict[str, torch.Tensor],
    layer_ids: list[int] | None = None,
    cosine_weight: float = 0.05,
    logit_kl_weight: float = 0.0,
) -> ReconstructionResult:
```

After layer losses are computed, add:

```python
    logit_kl = torch.zeros((), device=layer_losses[0].device, dtype=layer_losses[0].dtype)
    if logit_kl_weight:
        with torch.no_grad():
            teacher_logits = teacher(**batch).logits.detach()
        student_logits = student(**batch).logits
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        logit_kl, _ = _masked_mean(kl_per_token.to(layer_losses[0].dtype), attention_mask)
        layer_losses.append(logit_kl_weight * logit_kl)
        metrics[-1] = {"logit_kl": float(logit_kl.detach().cpu()), "token_count": int(batch["attention_mask"].sum().item())}
```

Note: this runs an extra student forward and should be used with small batch sizes. If memory bites, compute KL only every N steps in the training script instead of every step.

- [ ] **Step 4: Expose KL flags in the training script**

In `scripts/train_dense_reconstruction.py`, add:

```python
parser.add_argument("--logit-kl-weight", type=float, default=0.0)
```

Pass it into the loss call:

```python
result = compute_parallel_reconstruction_loss(
    teacher,
    student,
    batch,
    layer_ids=layer_ids,
    cosine_weight=args.cosine_weight,
    logit_kl_weight=args.logit_kl_weight,
)
```

Record it in `config`:

```python
"logit_kl_weight": args.logit_kl_weight,
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run --no-sync pytest tests/test_dense_reconstruction.py -q
```

Expected: pass.

---

### Task 4: Add Base One-Shot Tool Probe Script

**Files:**
- Create: `scripts/run_base_tool_probe.py`
- Test: `tests/test_base_tool_probe.py`

- [ ] **Step 1: Add parser-level tests**

Create `tests/test_base_tool_probe.py`:

```python
from scripts.run_base_tool_probe import summarize_generation


def test_summarize_generation_accepts_tagged_tool_call():
    text = "<think>Need to inspect.</think>\n<tool_call>shell\n<arg_key>command</arg_key>\n<arg_value>find . -name validators.py</arg_value>\n</tool_call>"
    summary = summarize_generation(text)
    assert summary["parseable_tool_call"] is True
    assert summary["tool_name"] == "shell"
    assert summary["has_required_arg"] is True


def test_summarize_generation_rejects_no_tool_call():
    summary = summarize_generation("the input is a list of strings")
    assert summary["parseable_tool_call"] is False
    assert summary["tool_name"] is None
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run --no-sync pytest tests/test_base_tool_probe.py -q
```

Expected: fail because the script does not exist.

- [ ] **Step 3: Implement script**

Create `scripts/run_base_tool_probe.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.pool_backend import parse_generated_tool_calls


REQUIRED_ARGS = {
    "shell": "command",
    "read_file": "path",
    "apply_patch": "patch",
    "exit": None,
}


def summarize_generation(text: str) -> dict:
    _, calls = parse_generated_tool_calls(text)
    if not calls:
        return {"parseable_tool_call": False, "tool_name": None, "has_required_arg": False}
    call = calls[0]
    name = call["function"]["name"]
    args = json.loads(call["function"].get("arguments") or "{}")
    required = REQUIRED_ARGS.get(name)
    has_required = required is None if name in REQUIRED_ARGS else False
    if required:
        has_required = bool(args.get(required))
    return {"parseable_tool_call": name in REQUIRED_ARGS, "tool_name": name, "has_required_arg": has_required, "arguments": args}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    args = parser.parse_args()

    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    messages = [
        {"role": "system", "content": "You are a coding agent. Emit exactly one Laguna tagged tool call. Valid tools: shell, read_file, apply_patch, exit."},
        {"role": "user", "content": "Repository: django/django. Issue: URLValidator accepts invalid characters in username/password. Inspect the repository to find the likely validator file. Emit one tool call."},
    ]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=True)
    inputs = {"input_ids": encoded.to(model.device)} if isinstance(encoded, torch.Tensor) else {key: value.to(model.device) for key, value in encoded.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    summary = summarize_generation(text)
    summary.update({"model_path": args.model_path, "generated_tokens": int(generated.numel()), "latency_s": time.perf_counter() - start})
    (args.output_dir / "generation.txt").write_text(text, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/test_base_tool_probe.py -q
```

Expected: pass.

---

### Task 5: Run Reconstruction v2

**Files:**
- Use: `scripts/train_dense_reconstruction.py`
- Use: `scripts/build_reconstruction_mixture.py`
- Use: `scripts/run_base_tool_probe.py`

- [ ] **Step 1: Build the mixture**

Run on the remote GPU host:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
.venv/bin/python scripts/build_reconstruction_mixture.py \
  --code-path data/sft/filtered_patch_sft_20260530.jsonl \
  --tool-path data/sft/rollout_sft_opus48_train80_seq16384_fulltarget.jsonl \
  --recovery-path data/sft/sonnet46_scale_256_p1_p12_sft_v2_finalmerge_noleaks_seq12288_shuf1337.jsonl \
  --output-path data/reconstruction/recon_mix_code_tool_recovery_seed1337.jsonl \
  --max-rows 20000 \
  --seed 1337
'
```

- [ ] **Step 2: Launch reconstruction v2 from placeholder once, intentionally**

Run:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
mkdir -p runs/reconstruction_v2_mix_kl_20260530 &&
.venv/bin/python scripts/train_dense_reconstruction.py \
  --teacher-model poolside/Laguna-XS.2 \
  --student-model checkpoints/laguna-xs2-dense-k8-copied-shell \
  --dataset json \
  --dataset-config data/reconstruction/recon_mix_code_tool_recovery_seed1337.jsonl \
  --split train \
  --output-dir runs/reconstruction_v2_mix_kl_20260530 \
  --seq-len 4096 \
  --batch-size 1 \
  --max-steps 10000 \
  --learning-rate 1e-4 \
  --cosine-weight 0.05 \
  --logit-kl-weight 0.02 \
  --log-every 25 \
  --save-every 1000 \
  --device cuda \
  --dtype bfloat16
'
```

If the `json` dataset invocation needs adjustment, use:

```python
load_dataset("json", data_files=str(path), split="train", streaming=args.streaming)
```

inside `scripts/train_dense_reconstruction.py`.

- [ ] **Step 3: Run base probes every 1000 steps**

For each checkpoint:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
.venv/bin/python scripts/run_base_tool_probe.py \
  --model-path runs/reconstruction_v2_mix_kl_20260530/checkpoint-step-1000 \
  --output-dir runs/reconstruction_v2_mix_kl_20260530/probe-step-1000 \
  --max-new-tokens 256 \
  --disable-cudnn-sdpa
'
```

Success criterion before SFT:

```text
parseable_tool_call = true
tool_name in {shell, read_file}
has_required_arg = true
generation is not repetitive token soup
```

Do not proceed to recovery SFT until at least one reconstruction checkpoint passes this one-shot probe.

---

### Task 6: Re-run Recovery SFT Only from a Passing Reconstructed Base

**Files:**
- Use: `scripts/train_dense_sft.py`
- Use: `scripts/run_dense_sft_sanity.py`
- Use: `scripts/run_coding_swebench_batch.py`

- [ ] **Step 1: Launch SFT from v2 reconstructed base**

Run:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
.venv/bin/python scripts/train_dense_sft.py \
  --model runs/reconstruction_v2_mix_kl_20260530/checkpoint-final \
  --tokenizer-model poolside/Laguna-XS.2 \
  --dataset data/sft/sonnet46_scale_256_p1_p12_sft_v2_finalmerge_noleaks_seq12288_shuf1337.jsonl \
  --output-dir runs/sft_recovery_from_recon_v2_20260530 \
  --seq-len 12288 \
  --batch-size 1 \
  --max-steps 1623 \
  --lr 5e-5 \
  --eval-every 100 \
  --eval-batches 16 \
  --train-norms \
  --train-lm-head \
  --disable-cudnn-sdpa
'
```

- [ ] **Step 2: Run sanity check**

Run:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
.venv/bin/python scripts/run_dense_sft_sanity.py \
  --model-path runs/sft_recovery_from_recon_v2_20260530/checkpoint-final \
  --tokenizer poolside/Laguna-XS.2 \
  --output-dir runs/sft_sanity_recovery_from_recon_v2_20260530 \
  --sft-dataset data/sft/sonnet46_scale_256_p1_p12_sft_v2_finalmerge_noleaks_seq12288_shuf1337.jsonl \
  --python-limit 0 \
  --swebench-limit 0 \
  --rollout-prefix-limit 20 \
  --max-new-tokens 256 \
  --seq-len 12288 \
  --disable-cudnn-sdpa
'
```

Gate to real rollouts:

```text
emits_tool_call_shape_rate >= 0.9
has_known_tool_name_rate >= 0.8
loop_warning_rows == 0 or clearly decreasing vs previous SFT
manual examples are not argument soup
```

- [ ] **Step 3: Run 3-task validation rollout**

Run:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
mkdir -p runs/openai_probe_recovery_from_recon_v2 &&
nohup .venv/bin/python scripts/run_openai_probe_server.py \
  --mode hf \
  --host 127.0.0.1 \
  --port 8765 \
  --model-id runs/sft_recovery_from_recon_v2_20260530/checkpoint-final \
  --tokenizer-id poolside/Laguna-XS.2 \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 512 \
  --no-sample \
  --disable-cudnn-sdpa \
  --output-dir runs/openai_probe_recovery_from_recon_v2/model_calls \
  --log-path runs/openai_probe_recovery_from_recon_v2/requests.jsonl \
  > runs/openai_probe_recovery_from_recon_v2/server.log 2>&1 &
'
```

Then:

```bash
ssh prime-gpu-31 '
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling &&
.venv/bin/python scripts/run_coding_swebench_batch.py \
  --registry tasks/registry_balanced_100_val20.jsonl \
  --api-url http://127.0.0.1:8765/v1 \
  --model hf-laguna-probe \
  --output-dir runs/real_val_rollouts_recovery_from_recon_v2_limit3_tok512_20260530 \
  --sandbox-root sandboxes/real_val_rollouts_recovery_from_recon_v2_limit3_tok512_20260530 \
  --limit 3 \
  --max-turns 8 \
  --max-tokens 512 \
  --temperature 0.0
'
```

---

## Decision Rules

Use these outcomes to choose the next big move:

- If reconstruction v2 still cannot emit one valid tool call: increase reconstruction scale and revisit architecture width `k=8`.
- If reconstruction v2 emits valid one-shot tool calls but SFT degrades: fix SFT masking/objective and checkpoint selection.
- If SFT emits valid calls but rollouts fail after observations: add off-policy KD / recovery SFT / RL.
- If original Laguna control works and dense v2 does not: harness is cleared; dense recovery is the bottleneck.

## Immediate Recommendation

Do not spend more compute on recovery SFT from `checkpoints/laguna-xs2-dense-k8-copied-shell`. First implement Task 1 guardrails, then train reconstruction v2 with tool/recovery data and KL. The prior Sonnet SFT run is useful as a negative control, but not as a repair path.
