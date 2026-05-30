from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

from densify.rollout_sft.tokenize import tokenize_sft_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter SFT rows that truncate away target tokens."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejects", type=Path)
    parser.add_argument("--model", default="poolside/Laguna-XS.2")
    parser.add_argument("--seq-len", type=int, default=12288)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    kept, rejected, lengths, trainable_lengths = filter_rows(
        rows,
        tokenizer,
        seq_len=args.seq_len,
        enable_thinking=not args.disable_thinking,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    if args.rejects:
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        args.rejects.write_text(
            "".join(json.dumps(row) + "\n" for row in rejected),
            encoding="utf-8",
        )

    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "seq_len": args.seq_len,
        "full_token_lengths": _stats(lengths),
        "trainable_token_lengths": _stats(trainable_lengths),
    }
    print(json.dumps(summary, indent=2))


def filter_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    seq_len: int,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    lengths: list[int] = []
    trainable_lengths: list[int] = []
    for row in rows:
        tokenized = tokenize_sft_row(row, tokenizer, seq_len, enable_thinking=enable_thinking)
        trainable = sum(label != -100 for label in tokenized["labels"])
        length = len(tokenized["input_ids"])
        if trainable <= 0:
            rejected.append(row)
            continue
        kept.append(row)
        lengths.append(length)
        trainable_lengths.append(trainable)
    return kept, rejected, lengths, trainable_lengths


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": mean(values)}


if __name__ == "__main__":
    main()
