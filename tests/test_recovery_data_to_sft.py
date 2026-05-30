from __future__ import annotations

from recovery_fixtures import make_failure_then_recovery_raw, make_recovery_raw

from densify.recovery_data.parse import parse_recovery_text
from densify.recovery_data.to_sft import build_sft_rows_from_example


def test_build_sft_rows_from_example_splits_assistant_tool_targets() -> None:
    example = parse_recovery_text(make_recovery_raw())

    rows = build_sft_rows_from_example(example)

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "ex1:assistant_0001"
    assert row["quality"] == "recovery_synthetic"
    assert row["weight"] == 2.5
    assert row["messages"][-1]["role"] == "assistant"
    assert "tool_calls" not in row["messages"][-1]
    assert "<tool_call>shell" in row["messages"][-1]["content"]
    assert row["metadata"]["target_tool"] == "shell"


def test_build_sft_rows_does_not_target_synthetic_failed_action() -> None:
    example = parse_recovery_text(make_failure_then_recovery_raw())

    rows = build_sft_rows_from_example(example)

    assert len(rows) == 1
    row = rows[0]
    assert "tool_calls" not in row["messages"][-1]
    assert "<tool_call>shell" in row["messages"][-1]["content"]
    assert row["metadata"]["target_is_recovery"] is True
    prefix_tools = [message["role"] for message in row["messages"][:-1]]
    assert prefix_tools == ["system", "user", "assistant", "tool"]


def test_successful_pytest_output_with_failed_word_is_not_skipped() -> None:
    raw = make_recovery_raw().replace(
        "django/utils/dateparse.py:42:standard_duration_re",
        "1 passed, 0 failed in 0.12s",
    )
    example = parse_recovery_text(raw)

    rows = build_sft_rows_from_example(example)

    assert len(rows) == 1
    assert "<tool_call>shell" in rows[0]["messages"][-1]["content"]


def test_file_contents_with_not_found_phrase_are_not_marked_recovery() -> None:
    raw = make_recovery_raw().replace(
        "django/utils/dateparse.py:42:standard_duration_re",
        "def validate_error_message():\\n    return 'not found'  # literal user-facing text",
    )
    example = parse_recovery_text(raw)

    rows = build_sft_rows_from_example(example)

    assert len(rows) == 1
    assert rows[0]["metadata"]["target_is_recovery"] is False


def test_empty_search_output_marks_next_target_as_recovery() -> None:
    raw = make_failure_then_recovery_raw().replace(
        "file not found or outside repo: django/django/pull/10097",
        "(no output)",
    )
    example = parse_recovery_text(raw)

    rows = build_sft_rows_from_example(example)

    assert len(rows) == 1
    assert rows[0]["metadata"]["target_is_recovery"] is True
