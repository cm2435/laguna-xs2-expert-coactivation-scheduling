from __future__ import annotations

from densify.eval_loop_warnings import (
    generation_warning_flags,
    trace_warning_flags,
)


def tagged_call(name: str, key: str, value: str) -> str:
    return "\n".join(
        [
            f"<tool_call>{name}",
            f"<arg_key>{key}</arg_key>",
            f"<arg_value>{value}</arg_value>",
            "</tool_call>",
        ]
    )


def test_generation_warning_flags_detect_repeated_tagged_tool_calls() -> None:
    text = "\n".join(
        [
            tagged_call("apply_patch", "patch", "--- a/x\n+++ b/x"),
            tagged_call("apply_patch", "patch", "--- a/x\n+++ b/x"),
            tagged_call("apply_patch", "patch", "--- a/x\n+++ b/x"),
        ]
    )

    flags = generation_warning_flags(text)

    assert flags["repeated_tool_call"]
    assert flags["max_consecutive_tool_call_repeat"] == 3


def test_generation_warning_flags_accepts_different_progressing_calls() -> None:
    text = "\n".join(
        [
            tagged_call("shell", "command", "grep -R foo ."),
            tagged_call("read_file", "path", "src/app.py"),
            tagged_call("apply_patch", "patch", "--- a/src/app.py"),
        ]
    )

    flags = generation_warning_flags(text)

    assert not flags["repeated_tool_call"]
    assert flags["max_consecutive_tool_call_repeat"] == 1


def test_trace_warning_flags_detect_repeat_after_error_observation() -> None:
    trace = [
        {
            "tool_name": "read_file",
            "arguments": {"path": "django/django/pull/10097"},
            "observation": "file not found or outside repo: django/django/pull/10097",
        },
        {
            "tool_name": "read_file",
            "arguments": {"path": "django/django/pull/10097"},
            "observation": "file not found or outside repo: django/django/pull/10097",
        },
    ]

    flags = trace_warning_flags(trace)

    assert flags["repeat_after_error"]
    assert flags["max_consecutive_tool_call_repeat"] == 2
