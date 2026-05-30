from __future__ import annotations

import pytest
from recovery_fixtures import make_recovery_raw

from densify.recovery_data.parse import (
    child_think_blocks,
    has_scheme_leak,
    parse_message_tool_calls,
    parse_recovery_text,
)


def test_parse_recovery_text_extracts_metadata_scheme_and_tool_calls() -> None:
    example = parse_recovery_text(make_recovery_raw())

    assert example.id == "ex1"
    assert example.metadata.prompt_family_id == "P5"
    assert example.scheme_audit == "Plan the failure but discard this."
    assert example.trajectory_messages[2]["role"] == "assistant"
    calls = parse_message_tool_calls(example.trajectory_messages[2])
    assert calls[0]["function"]["name"] == "shell"
    assert "standard_duration_re" in calls[0]["function"]["arguments"]
    assert child_think_blocks(example.trajectory_messages[2]) == [
        "The file was not found, so I should search."
    ]


def test_parse_rejects_scheme_leak_inside_trajectory_json() -> None:
    raw = make_recovery_raw().replace("system", "<scheme>bad</scheme>")

    with pytest.raises(ValueError, match="scheme leaked"):
        parse_recovery_text(raw)


def test_parse_normalizes_openai_role_variant() -> None:
    raw = make_recovery_raw().replace('"role": "assistant"', '"role": "assistant to=tool"', 1)

    example = parse_recovery_text(raw)

    assert example.trajectory_messages[2]["role"] == "assistant"
    calls = parse_message_tool_calls(example.trajectory_messages[2])
    assert calls[0]["function"]["name"] == "shell"


def test_parse_rejects_empty_assistant_turns() -> None:
    assistant_message = (
        '"role": "assistant", "content": "<think>The file was not found, so I should search.'
        '</think>\\n<tool_call>shell\\n<arg_key>cmd</arg_key><arg_value>grep -R '
        '\\"standard_duration_re\\" -n django/</arg_value>\\n</tool_call>"'
    )
    raw = make_recovery_raw().replace(
        assistant_message,
        '"role": "assistant", "content": ""',
    )

    with pytest.raises(ValueError, match="assistant message has no parseable tool call"):
        parse_recovery_text(raw)


def test_scheme_leak_detector_allows_domain_words() -> None:
    assert not has_scheme_leak("tick labels should remain hidden on shared axes")
    assert not has_scheme_leak("X : array-like, shape. Training data for the estimator.")
    assert has_scheme_leak("The data generation prompt asked for this.")
