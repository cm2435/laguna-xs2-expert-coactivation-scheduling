from __future__ import annotations

import json
import re
from dataclasses import dataclass

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    tool_name: str
    arguments: dict[str, str]
    raw_text: str
    start_char: int
    end_char: int


def parse_pool_tool_calls(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(0)
        inner = match.group(1).strip()
        if "\n" in inner:
            tool_name, args_text = inner.split("\n", 1)
        else:
            tool_name, args_text = inner, ""
        arguments = {
            key.strip(): value.strip()
            for key, value in ARG_RE.findall(args_text)
        }
        calls.append(
            ParsedToolCall(
                tool_name=tool_name.strip(),
                arguments=arguments,
                raw_text=raw,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return calls


def tool_call_to_json(call: ParsedToolCall, **extra) -> dict:
    return {
        **extra,
        "tool_name": call.tool_name,
        "arguments": call.arguments,
        "raw_text": call.raw_text,
        "start_char": call.start_char,
        "end_char": call.end_char,
        "arguments_json": json.dumps(call.arguments, ensure_ascii=False),
    }
