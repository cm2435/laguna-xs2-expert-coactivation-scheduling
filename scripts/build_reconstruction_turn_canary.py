from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from densify.reconstruction_data import format_sft_row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assistant_has_tool_call(message: dict[str, Any]) -> bool:
    return bool(message.get("role") == "assistant" and message.get("tool_calls"))


def wrap_child_thinking(content: Any, *, empty_thinking_close: bool = True) -> str:
    text = "" if content is None else str(content).strip()
    if "<think>" in text or "</think>" in text:
        return text
    if text:
        return f"<think>\n{text}\n</think>"
    return "</think>" if empty_thinking_close else ""


def normalize_assistant_surface(
    message: dict[str, Any],
    *,
    wrap_thinking: bool,
    empty_thinking_close: bool,
) -> dict[str, Any]:
    normalized = copy.deepcopy(message)
    if wrap_thinking:
        normalized["content"] = wrap_child_thinking(
            normalized.get("content"),
            empty_thinking_close=empty_thinking_close,
        )
    return normalized


def iter_turn_rows(
    source_rows: list[dict[str, Any]],
    *,
    wrap_thinking: bool = True,
    empty_thinking_close: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(source_rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        clean_messages = [message for message in messages if isinstance(message, dict)]
        for turn_index, message in enumerate(clean_messages):
            if not assistant_has_tool_call(message):
                continue
            target = normalize_assistant_surface(
                message,
                wrap_thinking=wrap_thinking,
                empty_thinking_close=empty_thinking_close,
            )
            turn_row = {
                "messages": copy.deepcopy(clean_messages[:turn_index]) + [target],
                "_reconstruction_source": row.get("_reconstruction_source", "turn_canary"),
                "_source_row_index": row_index,
                "_assistant_turn_index": turn_index,
                "_turn_canary": True,
            }
            output.append(turn_row)
    return output


def token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    text = format_sft_row(row)
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def build_turn_canary(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    max_unique_rows: int,
    repeat: int,
    seq_len: int,
    seed: int,
    wrap_thinking: bool = True,
    empty_thinking_close: bool = True,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    rows = iter_turn_rows(
        read_jsonl(input_path),
        wrap_thinking=wrap_thinking,
        empty_thinking_close=empty_thinking_close,
    )
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected_too_long = 0
    for row in rows:
        rendered = format_sft_row(row)
        if rendered in seen:
            continue
        seen.add(rendered)
        length = token_length(tokenizer, row)
        if length > seq_len:
            rejected_too_long += 1
            continue
        row["_token_length"] = length
        kept.append(row)
        if len(kept) >= max_unique_rows:
            break

    rng = random.Random(seed)
    rng.shuffle(kept)
    repeated: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        for row in kept:
            copy_row = copy.deepcopy(row)
            copy_row["_reconstruction_repeat"] = repeat_index
            repeated.append(copy_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row) + "\n" for row in repeated),
        encoding="utf-8",
    )
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "unique_rows": len(kept),
        "rows_written": len(repeated),
        "repeat": repeat,
        "seq_len": seq_len,
        "rejected_too_long": rejected_too_long,
        "wrap_thinking": wrap_thinking,
        "empty_thinking_close": empty_thinking_close,
        "token_lengths": [row["_token_length"] for row in kept],
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build short per-turn reconstruction rows for a tool-call overfit canary."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--max-unique-rows", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=1200)
    parser.add_argument("--seq-len", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-wrap-thinking", action="store_true")
    parser.add_argument("--no-empty-thinking-close", action="store_true")
    args = parser.parse_args()
    summary = build_turn_canary(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_unique_rows=args.max_unique_rows,
        repeat=args.repeat,
        seq_len=args.seq_len,
        seed=args.seed,
        wrap_thinking=not args.no_wrap_thinking,
        empty_thinking_close=not args.no_empty_thinking_close,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
