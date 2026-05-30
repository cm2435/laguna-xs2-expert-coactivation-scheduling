from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFTTexts:
    prefix_text: str
    target_text: str
    full_text: str
    source: str = "manual"


def render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "user")
    if role == "assistant" and message.get("tool_calls"):
        return (
            "<assistant>\n"
            + str(message.get("content") or "")
            + "\n"
            + json.dumps({"tool_calls": message["tool_calls"]}, sort_keys=True)
            + "\n</assistant>"
        )
    if role == "tool":
        return (
            '<tool name="'
            + str(message.get("name") or "")
            + '">\n'
            + str(message.get("content") or "")
            + "\n</tool>"
        )
    return "<" + role + ">\n" + str(message.get("content") or "") + "\n</" + role + ">"


def render_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(render_message(message) for message in messages)


def split_sft_texts(
    row: dict[str, Any],
    tokenizer: Any | None = None,
    *,
    enable_thinking: bool = True,
    use_chat_template: bool = True,
) -> SFTTexts:
    messages = list(row["messages"])
    if not messages:
        raise ValueError("row contains no messages")
    prefix_messages = messages[:-1]
    target_message = messages[-1]
    if target_message.get("role") != "assistant":
        raise ValueError("final message must be assistant for SFT target")
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            if not isinstance(prefix_text, str) or not isinstance(full_text, str):
                raise TypeError("chat template did not return text")
            if not full_text.startswith(prefix_text):
                # Some templates do not make the generation prompt a byte-for-byte prefix of the full
                # assistant turn. In that case, use token common-prefix alignment in tokenize_sft_row.
                pass
            return SFTTexts(
                prefix_text=prefix_text,
                target_text=full_text[len(prefix_text) :] if full_text.startswith(prefix_text) else "",
                full_text=full_text,
                source="chat_template",
            )
        except TypeError:
            prefix_text = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(prefix_text, str) and isinstance(full_text, str):
                return SFTTexts(
                    prefix_text=prefix_text,
                    target_text=full_text[len(prefix_text) :] if full_text.startswith(prefix_text) else "",
                    full_text=full_text,
                    source="chat_template",
                )
    prefix_text = render_messages(prefix_messages)
    target_text = ("\n" if prefix_text else "") + render_message(target_message)
    full_text = prefix_text + target_text
    return SFTTexts(prefix_text=prefix_text, target_text=target_text, full_text=full_text)


def tokenize_sft_row(
    row: dict[str, Any],
    tokenizer: Any,
    seq_len: int,
    *,
    enable_thinking: bool = True,
    use_chat_template: bool = True,
) -> dict[str, list[int]]:
    texts = split_sft_texts(
        row,
        tokenizer,
        enable_thinking=enable_thinking,
        use_chat_template=use_chat_template,
    )
    add_special_tokens = texts.source != "chat_template"
    full_ids = tokenizer(
        texts.full_text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=seq_len,
    )["input_ids"]
    labels = [-100] * len(full_ids)
    prefix_ids = (
        tokenizer(texts.prefix_text, add_special_tokens=add_special_tokens)["input_ids"]
        if texts.prefix_text
        else []
    )
    prefix_len = min(len(prefix_ids), len(full_ids))
    if texts.source == "chat_template":
        prefix_len = _common_prefix_len(prefix_ids, full_ids)
    for idx in range(prefix_len, len(full_ids)):
        labels[idx] = full_ids[idx]
    return {"input_ids": full_ids, "labels": labels}


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count
