from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.code_scoring import score_code_generation
from densify.eval_loop_warnings import generation_warning_flags
from densify.pool_backend import first_tagged_tool_call
from densify.prompt_data import load_jsonl_prompts


TOOL_CALL_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick qualitative sanity checks for a dense Laguna SFT checkpoint."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-prompt-path", default="data/prompts/python_smoke.jsonl")
    parser.add_argument("--swebench-prompt-path", default="data/prompts/swebench_verified_python_tiny.jsonl")
    parser.add_argument("--sft-dataset", default="data/sft/rollout_sft_opus48_train80_seq16384_fulltarget.jsonl")
    parser.add_argument("--python-limit", type=int, default=3)
    parser.add_argument("--swebench-limit", type=int, default=3)
    parser.add_argument("--rollout-prefix-limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()

    rows: list[dict[str, Any]] = []
    rows.extend(run_python_smoke(model, tokenizer, args))
    rows.extend(run_swebench_issue_smoke(model, tokenizer, args))
    rows.extend(run_rollout_prefix_smoke(model, tokenizer, args))

    summary = summarize(rows)
    (args.output_dir / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "examples.md").write_text(render_examples(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print_warnings(summary)
    print(f"wrote={args.output_dir}", flush=True)


def run_python_smoke(model, tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for prompt in load_jsonl_prompts(args.python_prompt_path)[: args.python_limit]:
        messages = [
            {
                "role": "user",
                "content": (
                    "Return a single Python code block with the requested function.\n\n"
                    f"Task:\n{prompt.prompt}"
                ),
            }
        ]
        text, latency_s, generated_tokens = generate_messages(model, tokenizer, messages, args)
        score = score_code_generation(text, prompt.tests)
        rows.append(
            {
                "kind": "python_smoke",
                "id": prompt.id,
                "prompt": prompt.prompt,
                "raw_generation": text,
                "extracted_code": score.extracted_code,
                "parse_ok": score.parse_ok,
                "tests_ok": score.tests_ok,
                "test_stdout": score.test_stdout,
                "test_stderr": score.test_stderr,
                "latency_s": latency_s,
                "generated_tokens": generated_tokens,
                "heuristics": {
                    **generation_heuristics(text),
                    **loop_warning_heuristics(text),
                },
            }
        )
    return rows


def run_swebench_issue_smoke(model, tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for prompt in load_jsonl_prompts(args.swebench_prompt_path)[: args.swebench_limit]:
        messages = [
            {
                "role": "system",
                "content": "You are a coding agent. Diagnose the repository issue and propose the likely patch strategy.",
            },
            {"role": "user", "content": prompt.prompt},
        ]
        text, latency_s, generated_tokens = generate_messages(model, tokenizer, messages, args)
        heuristics = {
            **generation_heuristics(text),
            **loop_warning_heuristics(text),
        }
        rows.append(
            {
                "kind": "swebench_issue",
                "id": prompt.id,
                "prompt": prompt.prompt,
                "raw_generation": text,
                "latency_s": latency_s,
                "generated_tokens": generated_tokens,
                "heuristics": {
                    **heuristics,
                    "mentions_patch": contains_any(text.lower(), ["patch", "change", "fix", "modify"]),
                    "mentions_file": contains_any(text.lower(), [".py", "file", "module"]),
                },
            }
        )
    return rows


def run_rollout_prefix_smoke(model, tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for row in select_rollout_prefix_rows(Path(args.sft_dataset), args.rollout_prefix_limit):
        messages = row["messages"][:-1]
        target = row["messages"][-1]
        text, latency_s, generated_tokens = generate_messages(model, tokenizer, messages, args)
        expected_tools = tool_names_from_message(target)
        generated_tools = tool_names_from_text(text)
        rows.append(
            {
                "kind": "rollout_prefix",
                "id": row["id"],
                "task_id": row.get("task_id"),
                "quality": row.get("quality"),
                "expected_target_preview": preview_message(target),
                "expected_tool_names": expected_tools,
                "generated_tool_names": generated_tools,
                "raw_generation": text,
                "latency_s": latency_s,
                "generated_tokens": generated_tokens,
                "heuristics": {
                    **generation_heuristics(text),
                    **loop_warning_heuristics(text),
                    "emits_tool_call_shape": bool(generated_tools) or '"tool_calls"' in text,
                    "uses_expected_tool_name": bool(set(expected_tools) & set(generated_tools)),
                    "looks_like_assistant_turn": contains_any(
                        text, ["tool_calls", "<assistant", "</assistant>", "read_file", "edit_file", "run_command"]
                    ),
                },
            }
        )
    return rows


@torch.inference_mode()
def generate_messages(model, tokenizer, messages: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, float, int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=args.enable_thinking,
        truncation=True,
        max_length=args.seq_len,
    )
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded.to(model.device)}
    else:
        inputs = {key: value.to(model.device) for key, value in encoded.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency_s = time.perf_counter() - start
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, latency_s, int(generated.numel())


def select_rollout_prefix_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    selected = []
    seen_tasks = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row.get("task_id")
            if task_id in seen_tasks:
                continue
            if row.get("quality") not in {"silver", "gold", "recovery_synthetic"}:
                continue
            messages = row.get("messages") or []
            if len(messages) < 3 or messages[-1].get("role") != "assistant":
                continue
            selected.append(row)
            seen_tasks.add(task_id)
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        raise ValueError(f"only found {len(selected)} rollout prefix rows in {path}")
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = Counter(str(row["kind"]) for row in rows)
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "by_kind": dict(by_kind),
        "non_empty_rate": mean_bool(bool(str(row["raw_generation"]).strip()) for row in rows),
        "total_generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "total_latency_s": sum(float(row["latency_s"]) for row in rows),
        "warning_counts": {
            "loop_warning_rows": sum(
                bool(row.get("heuristics", {}).get("repeated_tool_call")) for row in rows
            ),
            "error_marker_rows": sum(
                bool(row.get("heuristics", {}).get("contains_error_marker")) for row in rows
            ),
        },
    }
    summary["tokens_per_second"] = summary["total_generated_tokens"] / max(summary["total_latency_s"], 1e-6)
    for kind in sorted(by_kind):
        kind_rows = [row for row in rows if row["kind"] == kind]
        summary[kind] = summarize_kind(kind_rows)
    return summary


def summarize_kind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    heuristic_keys = sorted({key for row in rows for key in row.get("heuristics", {})})
    result = {"num_rows": len(rows)}
    for key in heuristic_keys:
        result[f"{key}_rate"] = mean_bool(bool(row.get("heuristics", {}).get(key)) for row in rows)
    if any("parse_ok" in row for row in rows):
        result["parse_ok_rate"] = mean_bool(bool(row.get("parse_ok")) for row in rows)
        result["tests_ok_rate"] = mean_bool(bool(row.get("tests_ok")) for row in rows)
    return result


def render_examples(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    parts = ["# Dense SFT Sanity Check", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for row in rows:
        parts.extend(
            [
                f"## {row['kind']} / {row['id']}",
                "",
                "Heuristics:",
                "```json",
                json.dumps(row.get("heuristics", {}), indent=2),
                "```",
                "",
            ]
        )
        if row.get("expected_tool_names") is not None:
            parts.extend(
                [
                    f"Expected tools: `{row.get('expected_tool_names')}`",
                    f"Generated tools: `{row.get('generated_tool_names')}`",
                    "",
                    "Expected target preview:",
                    "```text",
                    str(row.get("expected_target_preview", ""))[:1200],
                    "```",
                    "",
                ]
            )
        parts.extend(
            [
                "Generation:",
                "```text",
                str(row["raw_generation"])[:2400],
                "```",
                "",
            ]
        )
    return "\n".join(parts)


def generation_heuristics(text: str) -> dict[str, bool]:
    stripped = text.strip()
    return {
        "non_empty": bool(stripped),
        "has_think_tag": contains_any(text, ["<think>", "</think>"]),
        "has_tool_call_json": '"tool_calls"' in text,
        "has_known_tool_name": bool(tool_names_from_text(text)),
        "has_unclosed_assistant_tag": "<assistant" in text and "</assistant>" not in text,
        "looks_repetitive": looks_repetitive(text),
    }


def loop_warning_heuristics(text: str) -> dict[str, bool | int]:
    flags = generation_warning_flags(text)
    return {
        "repeated_tool_call": bool(flags["repeated_tool_call"]),
        "contains_error_marker": bool(flags["contains_error_marker"]),
        "tool_call_count": int(flags["tool_call_count"]),
        "max_consecutive_tool_call_repeat": int(flags["max_consecutive_tool_call_repeat"]),
    }


def print_warnings(summary: dict[str, Any]) -> None:
    warning_counts = summary.get("warning_counts") or {}
    loop_rows = int(warning_counts.get("loop_warning_rows") or 0)
    error_rows = int(warning_counts.get("error_marker_rows") or 0)
    if loop_rows:
        print(
            f"WARNING: validation generated repeated identical tool calls in {loop_rows} row(s).",
            flush=True,
        )
    if error_rows:
        print(
            f"WARNING: validation generations contained error markers in {error_rows} row(s).",
            flush=True,
        )


def tool_names_from_message(message: dict[str, Any]) -> list[str]:
    return [str(call.get("function", {}).get("name") or call.get("name") or "") for call in message.get("tool_calls") or []]


def tool_names_from_text(text: str) -> list[str]:
    json_names = TOOL_CALL_RE.findall(text)
    tagged = first_tagged_tool_call(text)
    if tagged is None:
        return json_names
    return json_names + [tagged[1]["function"]["name"]]


def preview_message(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return json.dumps({"tool_calls": message["tool_calls"]}, indent=2)[:2000]
    return str(message.get("content") or "")[:2000]


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def looks_repetitive(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 24:
        return False
    chunks = [" ".join(tokens[idx : idx + 6]) for idx in range(0, len(tokens) - 5, 6)]
    return len(chunks) - len(set(chunks)) >= 3


def mean_bool(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / max(len(values), 1)


if __name__ == "__main__":
    main()
