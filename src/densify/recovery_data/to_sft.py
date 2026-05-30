from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from densify.recovery_data.parse import RecoveryExample, example_from_json, parse_message_tool_calls
from densify.recovery_data.schema import ROW_WEIGHTS_BY_EXAMPLE_TYPE


def build_sft_rows_from_examples(examples: list[RecoveryExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.extend(build_sft_rows_from_example(example))
    return rows


def build_sft_rows_from_example(example: RecoveryExample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assistant_index = 0
    for index, message in enumerate(example.trajectory_messages):
        if message.get("role") != "assistant":
            continue
        calls = parse_message_tool_calls(message)
        if not calls:
            continue
        if _next_tool_observation_is_failure(example.trajectory_messages, index):
            continue
        assistant_index += 1
        target_messages = [
            _message_for_sft(item) for item in example.trajectory_messages[: index + 1]
        ]
        target_tool = _tool_name(calls[0])
        failed_observation = _prefix_has_failed_observation(target_messages[:-1])
        metadata = asdict(example.metadata)
        metadata.update(
            {
                "assistant_index": assistant_index,
                "target_role": "assistant",
                "target_tool": target_tool,
                "target_is_recovery": failed_observation,
                "stop_reason": example.stop_reason,
                "patch_nonempty": example.patch_nonempty,
                "notes": example.notes,
            }
        )
        rows.append(
            {
                "id": f"{example.id}:assistant_{assistant_index:04d}",
                "task_id": example.metadata.task_id,
                "source_rollout": example.id,
                "messages": target_messages,
                "quality": "recovery_synthetic",
                "weight": ROW_WEIGHTS_BY_EXAMPLE_TYPE.get(example.metadata.example_type, 2.0),
                "metadata": metadata,
            }
        )
    return rows


def read_parsed_examples(path: Path) -> list[RecoveryExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        examples.append(example_from_json(json.loads(line)))
    return examples


def write_sft_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _message_for_sft(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    # The Laguna chat template renders structured tool_calls itself. Recovery
    # examples already carry the desired Laguna tagged tool call in content, so
    # keeping tool_calls would either crash on JSON-string arguments or render a
    # duplicate call. SFT rows should train the known-good content format.
    normalized.pop("tool_calls", None)
    return normalized


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("function", {}).get("name") or call.get("name") or "")


def _prefix_has_failed_observation(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "tool":
            continue
        if _tool_observation_is_failure(str(message.get("content") or "")):
            return True
    return False


def _next_tool_observation_is_failure(messages: list[dict[str, Any]], assistant_index: int) -> bool:
    if assistant_index + 1 >= len(messages):
        return False
    next_message = messages[assistant_index + 1]
    if next_message.get("role") != "tool":
        return False
    return _tool_observation_is_failure(str(next_message.get("content") or ""))


def _tool_observation_is_failure(content: str) -> bool:
    lowered = content.lower()
    if "0 failed" in lowered or "no failures" in lowered:
        return False
    line_starts = (
        "error:",
        "error: ",
        "failed:",
        "failure:",
        "traceback ",
        "exception:",
    )
    explicit_markers = (
        "file not found",
        "outside repo",
        "no matches",
        "no output",
        "(no output)",
        "no results",
        "no occurrences",
        "hunk failed",
        "context not found",
        "does not apply",
        "unrecognized input",
        "patch apply failed",
        "patch did not apply",
        "jsondecodeerror",
    )
    lines = [line.strip().lower() for line in content.splitlines() if line.strip()]
    return any(
        line.startswith(line_starts) or any(marker in line for marker in explicit_markers)
        for line in lines
    )
