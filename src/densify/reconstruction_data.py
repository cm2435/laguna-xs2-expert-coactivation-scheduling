from __future__ import annotations

import json
from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_role(role: str, content: str) -> str:
    normalized = role.lower().strip()
    tag = "assistant" if normalized == "assistant" else "user"
    return f"<{tag}>\n{content.strip()}\n</{tag}>"


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if value is None:
        return {}
    return {"value": value}


def _format_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return ""
    name = _as_text(function.get("name"))
    if not name:
        return ""
    args = _tool_arguments(function.get("arguments"))
    lines = ["<tool_call>", name]
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = _as_text(value)
        lines.extend(
            [
                f"<arg_key>{key}</arg_key>",
                f"<arg_value>{rendered}</arg_value>",
            ]
        )
    lines.append("</tool_call>")
    return "\n".join(lines)


def _format_message(message: dict[str, Any]) -> str:
    role = _as_text(message.get("role", "user"))
    content_parts = []
    content = _as_text(message.get("content"))
    if content:
        content_parts.append(content)
    if role.lower().strip() == "assistant":
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                rendered = _format_tool_call(call)
                if rendered:
                    content_parts.append(rendered)
    if not content_parts:
        return ""
    return _format_role(role, "\n".join(content_parts))


def format_sft_row(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            rendered = _format_message(message)
            if rendered:
                parts.append(rendered)
        if parts:
            return "\n".join(parts)

    instruction = _as_text(row.get("instruction") or row.get("prompt") or row.get("question"))
    extra_input = _as_text(row.get("input"))
    output = _as_text(
        row.get("output") or row.get("completion") or row.get("response") or row.get("answer")
    )
    if extra_input:
        instruction = f"{instruction}\n\n{extra_input}" if instruction else extra_input
    if instruction and output:
        return f"<user>\n{instruction}\n</user>\n<assistant>\n{output}\n</assistant>"

    text = _as_text(row.get("text") or row.get("content"))
    if text:
        return text
    raise ValueError(f"Could not format SFT row with keys: {sorted(row)}")
