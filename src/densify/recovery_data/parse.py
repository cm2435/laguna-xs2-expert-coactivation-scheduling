from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from densify.recovery_data.schema import (
    PROMPT_FAMILIES,
    RecoveryExample,
    RecoveryMetadata,
)

SCHEME_LEAK_PATTERNS = [
    "data-generation",
    "data generation",
    "training target",
    "training example",
    "supervised training",
    "the prompt asked",
    "the scheme",
    "<scheme>",
    "child model",
    "student model",
    "synthetic trap",
]


def parse_recovery_text(text: str, *, fallback_id: str = "example") -> RecoveryExample:
    metadata = _parse_json_tag(text, "metadata_json")
    trajectory = _parse_json_tag(text, "trajectory_json")
    scheme = _extract_tag_text(text, "scheme")
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must be a JSON object")
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory_json must be a JSON object")
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("trajectory_json.messages must be a non-empty list")
    if _contains_scheme(str(trajectory)):
        raise ValueError("scheme leaked into trajectory_json")
    normalized_messages = [_normalize_message(message) for message in messages]
    for message in normalized_messages:
        if message.get("role") == "assistant" and not parse_message_tool_calls(message):
            raise ValueError("assistant message has no parseable tool call")
    if not any(parse_message_tool_calls(message) for message in normalized_messages):
        raise ValueError("trajectory contains no parseable Laguna tool call")

    prompt_family_id = str(metadata.get("prompt_family_id") or metadata.get("family_id") or "")
    family = PROMPT_FAMILIES.get(prompt_family_id, {})
    task_id = str(metadata.get("task_id") or fallback_id)
    repo = str(metadata.get("repo") or "")
    recovery_metadata = RecoveryMetadata(
        task_id=task_id,
        repo=repo,
        prompt_family_id=prompt_family_id,
        prompt_family_name=str(metadata.get("prompt_family_name") or family.get("name") or ""),
        example_type=str(metadata.get("example_type") or family.get("example_type") or ""),
        failure_type=str(metadata.get("failure_type") or family.get("failure_type") or ""),
        intended_first_action=str(
            metadata.get("intended_first_action") or family.get("target_tool") or ""
        ),
        recovery_action=str(metadata.get("recovery_action") or family.get("recovery_action") or ""),
        target_tool=str(metadata.get("target_tool") or family.get("target_tool") or ""),
        source=str(metadata.get("source") or "metacognitive_recovery"),
    )
    return RecoveryExample(
        id=str(metadata.get("id") or f"{task_id}:{prompt_family_id}:{fallback_id}"),
        metadata=recovery_metadata,
        scheme_audit=scheme,
        trajectory_messages=normalized_messages,
        stop_reason=str(trajectory.get("stop_reason") or ""),
        patch_nonempty=bool(trajectory.get("patch_nonempty")),
        notes=str(trajectory.get("notes") or ""),
        raw_text=text,
    )


def example_to_json(example: RecoveryExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "metadata": asdict(example.metadata),
        "scheme_audit": example.scheme_audit,
        "trajectory_messages": example.trajectory_messages,
        "stop_reason": example.stop_reason,
        "patch_nonempty": example.patch_nonempty,
        "notes": example.notes,
        "raw_text": example.raw_text,
    }


def example_from_json(row: dict[str, Any]) -> RecoveryExample:
    return RecoveryExample(
        id=str(row["id"]),
        metadata=RecoveryMetadata(**row["metadata"]),
        scheme_audit=str(row.get("scheme_audit") or ""),
        trajectory_messages=list(row.get("trajectory_messages") or []),
        stop_reason=str(row.get("stop_reason") or ""),
        patch_nonempty=bool(row.get("patch_nonempty")),
        notes=str(row.get("notes") or ""),
        raw_text=str(row.get("raw_text") or ""),
    )


def parse_message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("role") != "assistant":
        return []
    if isinstance(message.get("tool_calls"), list):
        return list(message["tool_calls"])
    call = first_tagged_tool_call(str(message.get("content") or ""))
    return [call] if call else []


def first_tagged_tool_call(text: str) -> dict[str, Any] | None:
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if match:
        body = match.group(1)
    else:
        start = text.find("<tool_call>")
        if start < 0:
            return None
        body = text[start + len("<tool_call>") :]
    lines = body.splitlines()
    tool_name = ""
    while lines and not tool_name:
        tool_name = lines.pop(0).strip()
    if not tool_name:
        return None
    arguments: dict[str, Any] = {}
    for arg_key, arg_value in re.findall(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        body,
        flags=re.DOTALL,
    ):
        arguments[arg_key.strip()] = arg_value.strip()
    return {
        "id": "generated_tool_1",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
    }


def child_think_blocks(message: dict[str, Any]) -> list[str]:
    if message.get("role") != "assistant":
        return []
    return re.findall(r"<think>(.*?)</think>", str(message.get("content") or ""), flags=re.DOTALL)


def has_scheme_leak(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in SCHEME_LEAK_PATTERNS)


def _parse_json_tag(text: str, tag: str) -> Any:
    body = _extract_tag_text(text, tag)
    if not body:
        raise ValueError(f"missing <{tag}> block")
    return json.loads(body)


def _extract_tag_text(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _contains_scheme(text: str) -> bool:
    return "<scheme>" in text.lower() or "</scheme>" in text.lower()


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("trajectory messages must be objects")
    role = str(message.get("role") or "")
    if role.startswith("assistant"):
        role = "assistant"
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported message role: {role}")
    normalized = dict(message)
    normalized["role"] = role
    normalized["content"] = (
        "" if normalized.get("content") is None else str(normalized.get("content"))
    )
    return normalized
