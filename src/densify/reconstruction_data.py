from __future__ import annotations

from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_role(role: str, content: str) -> str:
    normalized = role.lower().strip()
    tag = "assistant" if normalized == "assistant" else "user"
    return f"<{tag}>\n{content.strip()}\n</{tag}>"


def format_sft_row(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _as_text(message.get("content"))
            if content:
                parts.append(_format_role(_as_text(message.get("role", "user")), content))
        if parts:
            return "\n".join(parts)

    instruction = _as_text(row.get("instruction") or row.get("prompt") or row.get("question"))
    extra_input = _as_text(row.get("input"))
    output = _as_text(row.get("output") or row.get("completion") or row.get("response") or row.get("answer"))
    if extra_input:
        instruction = f"{instruction}\n\n{extra_input}" if instruction else extra_input
    if instruction and output:
        return f"<user>\n{instruction}\n</user>\n<assistant>\n{output}\n</assistant>"

    text = _as_text(row.get("text") or row.get("content"))
    if text:
        return text
    raise ValueError(f"Could not format SFT row with keys: {sorted(row)}")
