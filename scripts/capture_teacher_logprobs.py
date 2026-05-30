from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import request

from transformers import AutoTokenizer

from densify.rollout_sft.logprobs import (
    build_kd_row,
    normalize_chat_top_logprobs,
    render_assistant_target,
    sampled_token_ids_from_chat_logprobs,
    split_last_assistant_target,
    target_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture teacher top-logprobs for rollout SFT rows.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-api-url", required=True)
    parser.add_argument("--model", default="laguna")
    parser.add_argument("--tokenizer", default="poolside/Laguna-XS.2")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"empty input: {args.input}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            context_messages, target_message = split_last_assistant_target(list(row["messages"]))
            target_text = render_assistant_target(target_message)
            if args.dry_run:
                top_logprobs = _dummy_top_logprobs(target_text, tokenizer)
            else:
                response = request_teacher_logprobs(
                    base_url=args.teacher_api_url,
                    model=args.model,
                    messages=context_messages,
                    max_tokens=max(1, len(tokenizer(target_text, add_special_tokens=False)["input_ids"])),
                    top_logprobs=args.top_logprobs,
                )
                top_logprobs = normalize_chat_top_logprobs(response, tokenizer, top_k=args.top_logprobs)
                sampled_ids = sampled_token_ids_from_chat_logprobs(response, tokenizer)
                expected_ids = target_token_ids(target_text, tokenizer)
                if sampled_ids != expected_ids:
                    raise SystemExit(
                        "teacher replay generated tokens that do not match the rollout target "
                        f"for row {row.get('id')}: sampled_len={len(sampled_ids)} target_len={len(expected_ids)}"
                    )
            kd_row = build_kd_row(
                row,
                tokenizer,
                top_logprobs,
                source="dry_run" if args.dry_run else "vllm_top_logprobs",
                top_k=args.top_logprobs,
            )
            handle.write(json.dumps(kd_row) + "\n")


def request_teacher_logprobs(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    top_logprobs: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _dummy_top_logprobs(target_text: str, tokenizer: Any) -> list[list[dict[str, float | int]]]:
    token_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    return [[{"token_id": int(token_id), "logprob": 0.0}] for token_id in token_ids]


if __name__ == "__main__":
    main()
