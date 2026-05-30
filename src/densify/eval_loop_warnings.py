from __future__ import annotations

import json
import re
from typing import Any

ERROR_MARKERS = (
    "file not found",
    "outside repo",
    "no such file",
    "no matches",
    "no output",
    "(no output)",
    "hunk failed",
    "context not found",
    "does not apply",
    "unrecognized input",
    "error:",
    "traceback",
)


def generation_warning_flags(text: str, *, repeat_threshold: int = 3) -> dict[str, Any]:
    keys = tagged_tool_call_keys(text)
    max_repeat = max_consecutive_repeat(keys)
    return {
        "tool_call_count": len(keys),
        "repeated_tool_call": max_repeat >= repeat_threshold,
        "max_consecutive_tool_call_repeat": max_repeat,
        "contains_error_marker": contains_error_marker(text),
    }


def trace_warning_flags(
    trace_rows: list[dict[str, Any]],
    *,
    repeat_threshold: int = 3,
) -> dict[str, Any]:
    keys = [_trace_tool_key(row) for row in trace_rows]
    max_repeat = max_consecutive_repeat(keys)
    repeat_after_error = False
    for previous, current in zip(trace_rows, trace_rows[1:], strict=False):
        if _trace_tool_key(previous) == _trace_tool_key(current) and contains_error_marker(
            str(previous.get("observation") or "")
        ):
            repeat_after_error = True
            break
    return {
        "tool_call_count": len(keys),
        "repeated_tool_call": max_repeat >= repeat_threshold,
        "repeat_after_error": repeat_after_error,
        "max_consecutive_tool_call_repeat": max_repeat,
    }


def tagged_tool_call_keys(text: str) -> list[str]:
    return [_tool_key(name, args) for name, args in tagged_tool_calls(text)]


def tagged_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        name, args = _parse_tagged_body(body)
        if name:
            calls.append((name, args))
    return calls


def max_consecutive_repeat(values: list[str]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    for previous, value in zip(values, values[1:], strict=False):
        if value == previous:
            current += 1
        else:
            current = 1
        best = max(best, current)
    return best


def contains_error_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ERROR_MARKERS)


def _parse_tagged_body(body: str) -> tuple[str, dict[str, str]]:
    lines = body.splitlines()
    name = ""
    while lines and not name:
        name = lines.pop(0).strip()
    args = {
        key.strip(): value.strip()
        for key, value in re.findall(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            body,
            flags=re.DOTALL,
        )
    }
    return name, args


def _trace_tool_key(row: dict[str, Any]) -> str:
    name = str(row.get("tool_name") or row.get("name") or "")
    args = row.get("arguments") or {}
    return _tool_key(name, args if isinstance(args, dict) else {"_raw": str(args)})


def _tool_key(name: str, args: dict[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": args},
        sort_keys=True,
        ensure_ascii=False,
    )
