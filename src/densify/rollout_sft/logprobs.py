from __future__ import annotations

from typing import Any


def split_last_assistant_target(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return list(messages[:index]), dict(messages[index])
    raise ValueError("row has no assistant message to use as KD target")


def render_assistant_target(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if message.get("tool_calls"):
        import json

        tool_payload = json.dumps({"tool_calls": message["tool_calls"]}, sort_keys=True)
        return content + ("\n" if content else "") + tool_payload
    return content


def target_token_ids(target_text: str, tokenizer: Any) -> list[int]:
    return list(tokenizer(target_text, add_special_tokens=False)["input_ids"])


def normalize_chat_top_logprobs(
    response: dict[str, Any],
    tokenizer: Any,
    *,
    top_k: int,
) -> list[list[dict[str, float | int]]]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    logprobs = choices[0].get("logprobs") or {}
    content = logprobs.get("content") or []
    normalized: list[list[dict[str, float | int]]] = []
    for position in content:
        entries = position.get("top_logprobs") or []
        normalized_entries: list[dict[str, float | int]] = []
        for entry in entries[:top_k]:
            token_id = entry.get("token_id")
            if token_id is None:
                token = str(entry.get("token") or "")
                token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                if len(token_ids) != 1:
                    continue
                token_id = token_ids[0]
            normalized_entries.append({"token_id": int(token_id), "logprob": float(entry["logprob"])})
        normalized.append(normalized_entries)
    return normalized


def sampled_token_ids_from_chat_logprobs(response: dict[str, Any], tokenizer: Any) -> list[int]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    logprobs = choices[0].get("logprobs") or {}
    content = logprobs.get("content") or []
    token_ids: list[int] = []
    for position in content:
        token_id = position.get("token_id")
        if token_id is None:
            token = str(position.get("token") or "")
            encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(f"sampled token does not map to exactly one token id: {token!r}")
            token_id = encoded[0]
        token_ids.append(int(token_id))
    return token_ids


def build_kd_row(
    sft_row: dict[str, Any],
    tokenizer: Any,
    teacher_top_logprobs: list[list[dict[str, float | int]]],
    *,
    source: str,
    top_k: int,
) -> dict[str, Any]:
    context_messages, target_message = split_last_assistant_target(list(sft_row["messages"]))
    text = render_assistant_target(target_message)
    ids = target_token_ids(text, tokenizer)
    if len(ids) != len(teacher_top_logprobs):
        raise ValueError(
            f"target token count ({len(ids)}) does not match logprob positions ({len(teacher_top_logprobs)})"
        )
    return {
        "id": sft_row["id"],
        "task_id": sft_row.get("task_id"),
        "source_rollout": sft_row.get("source_rollout"),
        "context_messages": context_messages,
        "target_text": text,
        "target_token_ids": ids,
        "teacher_top_logprobs": teacher_top_logprobs,
        "metadata": {"logprob_source": source, "top_k": top_k},
    }
