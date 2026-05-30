from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

from densify.rollout_sft.tokenize import split_sft_texts, tokenize_sft_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rollout SFT JSONL before training.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="poolside/Laguna-XS.2")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    summary = summarize_sft_rows(
        rows,
        tokenizer,
        seq_len=args.seq_len,
        max_examples=args.max_examples,
        enable_thinking=not args.disable_thinking,
    )
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def summarize_sft_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    seq_len: int,
    max_examples: int,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    quality_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    target_lengths: list[int] = []
    full_lengths: list[int] = []
    trainable_lengths: list[int] = []
    invalid_rows: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    template_examples: list[dict[str, Any]] = []
    tool_call_target_count = 0
    empty_target_count = 0

    for row in rows:
        row_id = str(row.get("id") or "<missing-id>")
        quality_counts[str(row.get("quality") or "unknown")] += 1
        messages = list(row.get("messages") or [])
        final_role = str(messages[-1].get("role") if messages else "<empty>")
        role_counts[final_role] += 1
        try:
            split = split_sft_texts(row, tokenizer, enable_thinking=enable_thinking)
            tokenized = tokenize_sft_row(row, tokenizer, seq_len, enable_thinking=enable_thinking)
        except Exception as exc:
            invalid_rows.append({"id": row_id, "error": str(exc)})
            continue

        target_text = split.target_text or _decode_trainable_tokens(tokenized, tokenizer)
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] if target_text else []
        target_lengths.append(len(target_ids))
        full_lengths.append(len(tokenized["input_ids"]))
        trainable = sum(1 for label in tokenized["labels"] if label != -100)
        trainable_lengths.append(trainable)
        target = messages[-1]
        if target.get("tool_calls"):
            tool_call_target_count += 1
        if not str(target.get("content") or "").strip() and not target.get("tool_calls"):
            empty_target_count += 1
        if len(examples) < max_examples:
            examples.append(_debug_example(row_id, split, tokenized, tokenizer))
        if len(template_examples) < max_examples:
            template_examples.append(compare_template_rendering(row, tokenizer, enable_thinking=enable_thinking))

    valid_count = len(rows) - len(invalid_rows)
    return {
        "row_count": len(rows),
        "valid_row_count": valid_count,
        "invalid_row_count": len(invalid_rows),
        "assistant_target_count": role_counts.get("assistant", 0),
        "non_assistant_target_count": len(rows) - role_counts.get("assistant", 0),
        "tool_call_target_count": tool_call_target_count,
        "empty_target_count": empty_target_count,
        "quality_counts": dict(quality_counts),
        "target_token_lengths": _length_stats(target_lengths),
        "full_token_lengths": _length_stats(full_lengths),
        "trainable_token_lengths": _length_stats(trainable_lengths),
        "invalid_rows": invalid_rows[:20],
        "examples": examples,
        "template_examples": template_examples,
    }


def compare_template_rendering(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    manual_split = split_sft_texts(row, use_chat_template=False)
    active_split = split_sft_texts(row, tokenizer, enable_thinking=enable_thinking)
    messages = list(row["messages"])
    has_chat_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if not has_chat_template:
        return {
            "id": row.get("id"),
            "has_chat_template": False,
            "manual_preview": manual_split.full_text[:1200],
        }

    try:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        return {
            "id": row.get("id"),
            "has_chat_template": True,
            "chat_template_error": str(exc),
            "manual_preview": manual_split.full_text[:1200],
        }

    manual_ids = tokenizer(manual_split.full_text, add_special_tokens=False)["input_ids"]
    chat_ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
    manual_count = len(manual_ids)
    chat_count = len(chat_ids)
    delta = chat_count - manual_count
    ratio = (chat_count / manual_count) if manual_count else None
    return {
        "id": row.get("id"),
        "has_chat_template": True,
        "active_render_source": active_split.source,
        "manual_token_count": manual_count,
        "chat_template_token_count": chat_count,
        "token_count_delta": delta,
        "chat_to_manual_token_ratio": ratio,
        "manual_preview": manual_split.full_text[:1200],
        "active_prefix_preview": active_split.prefix_text[-1200:],
        "active_target_preview": active_split.target_text[:1200],
        "chat_template_preview": str(chat_text)[:1200],
    }


def _length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": mean(values)}


def _debug_example(row_id: str, split, tokenized: dict[str, list[int]], tokenizer: Any) -> dict[str, Any]:
    labels = tokenized["labels"]
    input_ids = tokenized["input_ids"]
    first_trainable = next((idx for idx, label in enumerate(labels) if label != -100), None)
    return {
        "id": row_id,
        "prefix_preview": split.prefix_text[-1200:],
        "target_preview": split.target_text[:1200],
        "input_token_count": len(input_ids),
        "trainable_token_count": sum(1 for label in labels if label != -100),
        "first_trainable_index": first_trainable,
        "masked_decoded_tail": _decode_window(input_ids, labels, tokenizer, around=first_trainable),
    }


def _decode_trainable_tokens(tokenized: dict[str, list[int]], tokenizer: Any) -> str:
    trainable_ids = [
        token_id
        for token_id, label in zip(tokenized["input_ids"], tokenized["labels"], strict=False)
        if label != -100
    ]
    return tokenizer.decode(trainable_ids) if trainable_ids else ""


def _decode_window(
    input_ids: list[int],
    labels: list[int],
    tokenizer: Any,
    *,
    around: int | None,
    radius: int = 24,
) -> list[dict[str, Any]]:
    if around is None:
        return []
    start = max(0, around - radius)
    end = min(len(input_ids), around + radius)
    rows = []
    for idx in range(start, end):
        token_id = input_ids[idx]
        rows.append(
            {
                "index": idx,
                "token_id": token_id,
                "train": labels[idx] != -100,
                "decoded": tokenizer.decode([token_id]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
