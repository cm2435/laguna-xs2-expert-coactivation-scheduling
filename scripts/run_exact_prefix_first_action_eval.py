from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from densify.pool_backend import parse_generated_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate first tool action on exact SFT prefixes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--dataset", default="data/sft/rollout_sft_opus48_train80_seq16384_fulltarget.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    rows = select_rows(Path(args.dataset), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to("cuda")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    results = [evaluate_row(idx, row, model, tokenizer, args) for idx, row in enumerate(rows, start=1)]
    summary = summarize(results)
    (args.output_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "examples.md").write_text(render_examples(results, summary))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"OUT={args.output_dir}", flush=True)


def select_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    seen_tasks = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("quality") not in {"silver", "gold"}:
                continue
            if row.get("metadata", {}).get("assistant_index") != 1:
                continue
            if not (row.get("messages") and row["messages"][-1].get("tool_calls")):
                continue
            task_id = row.get("task_id")
            if task_id in seen_tasks:
                continue
            rows.append(row)
            seen_tasks.add(task_id)
            if len(rows) >= limit:
                break
    return rows


@torch.inference_mode()
def evaluate_row(idx: int, row: dict, model, tokenizer, args: argparse.Namespace) -> dict:
    prefix = row["messages"][:-1]
    target = row["messages"][-1]
    expected_call = (target.get("tool_calls") or [{}])[0]
    expected_fn = expected_call.get("function", {})
    expected_args = expected_fn.get("arguments") or {}
    encoded = tokenizer.apply_chat_template(
        prefix,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = {"input_ids": encoded.to("cuda")} if isinstance(encoded, torch.Tensor) else {
        key: value.to("cuda") for key, value in encoded.items()
    }
    input_len = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency_s = time.perf_counter() - started
    generated = output[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    _, generated_calls = parse_generated_tool_calls(text)
    generated_call = generated_calls[0] if generated_calls else {}
    generated_fn = generated_call.get("function", {}) if generated_call else {}
    generated_args = parse_arguments(generated_fn.get("arguments") if generated_fn else None)
    expected_primary, generated_primary = primary_values(expected_args, generated_args)

    return {
        "idx": idx,
        "id": row.get("id"),
        "task_id": row.get("task_id"),
        "input_tokens": input_len,
        "generated_tokens": int(generated.numel()),
        "latency_s": latency_s,
        "expected_tool": expected_fn.get("name"),
        "generated_tool": generated_fn.get("name"),
        "parseable_tool_call": bool(generated_calls),
        "tool_match": expected_fn.get("name") == generated_fn.get("name"),
        "expected_arg_keys": sorted(expected_args.keys()) if isinstance(expected_args, dict) else [],
        "generated_arg_keys": sorted(generated_args.keys()) if isinstance(generated_args, dict) else [],
        "has_expected_required_key": bool(
            set(expected_args.keys()) & set(generated_args.keys())
            if isinstance(expected_args, dict) and isinstance(generated_args, dict)
            else False
        ),
        "expected_primary": expected_primary[:1000],
        "generated_primary": generated_primary[:1000],
        "primary_overlap": primary_overlap(expected_primary, generated_primary),
        "raw_generation": text,
    }


def parse_arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def primary_values(expected_args: dict, generated_args: dict) -> tuple[str, str]:
    for key in ["path", "command", "patch", "summary"]:
        if isinstance(expected_args, dict) and key in expected_args:
            return str(expected_args[key]), str(generated_args.get(key, ""))
    return "", ""


def primary_overlap(expected: str, generated: str) -> bool:
    if not expected or not generated:
        return False
    parts = [part for part in expected.replace("/", " ").replace("\\", " ").split() if len(part) > 3]
    return expected in generated or generated in expected or any(part in generated for part in parts[-4:])


def summarize(results: list[dict]) -> dict:
    n = max(len(results), 1)
    return {
        "num_rows": len(results),
        "parseable_tool_call_rate": sum(row["parseable_tool_call"] for row in results) / n,
        "tool_match_rate": sum(row["tool_match"] for row in results) / n,
        "has_expected_required_key_rate": sum(row["has_expected_required_key"] for row in results) / n,
        "primary_overlap_rate": sum(row["primary_overlap"] for row in results) / n,
        "expected_tool_counts": dict(Counter(row["expected_tool"] for row in results)),
        "generated_tool_counts": dict(Counter(row["generated_tool"] for row in results)),
    }


def render_examples(results: list[dict], summary: dict) -> str:
    parts = ["# Exact Prefix First-Action Eval", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for row in results:
        parts.extend(
            [
                f"## {row['idx']}. {row['task_id']}",
                "",
                f"Expected: `{row['expected_tool']}` `{row['expected_primary'][:240]}`",
                "",
                f"Generated: `{row['generated_tool']}` `{row['generated_primary'][:240]}`",
                "",
                "```text",
                row["raw_generation"][:1600],
                "```",
                "",
            ]
        )
    return "\n".join(parts)


if __name__ == "__main__":
    main()
