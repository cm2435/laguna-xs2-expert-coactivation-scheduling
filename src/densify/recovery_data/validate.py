from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from densify.recovery_data.parse import (
    child_think_blocks,
    has_scheme_leak,
    parse_message_tool_calls,
)
from densify.recovery_data.schema import KNOWN_TOOLS, RecoveryValidationResult
from densify.recovery_data.to_sft import _tool_observation_is_failure


def validate_recovery_dataset(
    examples: list[Any],
    rows: list[dict[str, Any]],
) -> RecoveryValidationResult:
    first_action_tool_counts: Counter[str] = Counter()
    example_type_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    recovery_action_counts: Counter[str] = Counter()
    target_tool_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()

    scheme_leak_count = 0
    think_leak_count = 0
    parseable_tool_call_rows = 0
    child_facing_think_rows = 0
    failed_observation_prefix_rows = 0
    empty_target_rows = 0
    unknown_tool_rows = 0

    for example in examples:
        metadata = example.metadata
        example_type_counts[metadata.example_type] += 1
        failure_type_counts[metadata.failure_type] += 1
        recovery_action_counts[metadata.recovery_action] += 1
        repo_counts[metadata.repo] += 1
        trajectory_text = " ".join(
            str(message.get("content") or "") for message in example.trajectory_messages
        )
        if has_scheme_leak(trajectory_text):
            scheme_leak_count += 1
        first_tool = _first_assistant_tool(example.trajectory_messages)
        if first_tool:
            first_action_tool_counts[first_tool] += 1

    for row in rows:
        target = row.get("messages", [{}])[-1]
        calls = parse_message_tool_calls(target)
        if calls:
            parseable_tool_call_rows += 1
            tool = str(calls[0].get("function", {}).get("name") or "")
            target_tool_counts[tool] += 1
            if tool not in KNOWN_TOOLS:
                unknown_tool_rows += 1
        else:
            empty_target_rows += 1
        thinks = child_think_blocks(target)
        if thinks:
            child_facing_think_rows += 1
        if any(has_scheme_leak(think) for think in thinks):
            think_leak_count += 1
        if _has_failed_observation(row.get("messages", [])[:-1]):
            failed_observation_prefix_rows += 1

    shell_explore = target_tool_counts.get("shell", 0)
    read_first = target_tool_counts.get("read_file", 0)
    total_supervised_targets = sum(target_tool_counts.values())
    read_file_fraction = (
        (read_first / total_supervised_targets) if total_supervised_targets else 0.0
    )
    passes_hard_balance = (
        shell_explore >= read_first
        and read_file_fraction <= 0.4
        and scheme_leak_count == 0
        and think_leak_count == 0
    )

    hard_failures: list[str] = []
    if scheme_leak_count:
        hard_failures.append("scheme_leak_count > 0")
    if think_leak_count:
        hard_failures.append("think blocks mention scheme/data generation/synthetic prompt")
    if read_first > shell_explore:
        hard_failures.append("supervised target read_file > shell/explore")
    if read_file_fraction > 0.4:
        hard_failures.append("supervised target read_file > 40%")
    if unknown_tool_rows:
        hard_failures.append("unknown tool names")
    if empty_target_rows:
        hard_failures.append("empty target assistant actions")

    return RecoveryValidationResult(
        num_rollouts=len(examples),
        num_rows=len(rows),
        first_action_tool_counts=dict(first_action_tool_counts),
        example_type_counts=dict(example_type_counts),
        failure_type_counts=dict(failure_type_counts),
        recovery_action_counts=dict(recovery_action_counts),
        target_tool_counts=dict(target_tool_counts),
        repo_counts=dict(repo_counts),
        scheme_leak_count=scheme_leak_count,
        think_leak_count=think_leak_count,
        parseable_tool_call_rows=parseable_tool_call_rows,
        child_facing_think_rows=child_facing_think_rows,
        failed_observation_prefix_rows=failed_observation_prefix_rows,
        empty_target_rows=empty_target_rows,
        unknown_tool_rows=unknown_tool_rows,
        passes_hard_balance=passes_hard_balance,
        hard_failures=hard_failures,
    )


def validation_to_json(result: RecoveryValidationResult) -> dict[str, Any]:
    return asdict(result)


def _first_assistant_tool(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = parse_message_tool_calls(message)
        if calls:
            return str(calls[0].get("function", {}).get("name") or "")
    return ""


def _has_failed_observation(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "tool":
            continue
        if _tool_observation_is_failure(str(message.get("content") or "")):
            return True
    return False
