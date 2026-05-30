from __future__ import annotations

# ruff: noqa: E501
import json


def make_recovery_raw(*, think: str = "The file was not found, so I should search.") -> str:
    metadata = {
        "id": "ex1",
        "task_id": "django__django-10999",
        "repo": "django/django",
        "prompt_family_id": "P5",
        "example_type": "failure_then_recovery",
        "failure_type": "file_not_found",
        "intended_first_action": "shell",
        "recovery_action": "shell_explore",
        "target_tool": "shell",
    }
    trajectory = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": (
                    f"<think>{think}</think>\n"
                    "<tool_call>shell\n"
                    "<arg_key>cmd</arg_key><arg_value>grep -R \"standard_duration_re\" -n django/</arg_value>\n"
                    "</tool_call>"
                ),
            },
            {"role": "tool", "content": "django/utils/dateparse.py:42:standard_duration_re"},
        ],
        "stop_reason": "resolved",
        "patch_nonempty": False,
        "notes": "ok",
    }
    return (
        "<example>\n"
        f"<metadata_json>{json.dumps(metadata)}</metadata_json>\n"
        "<scheme>Plan the failure but discard this.</scheme>\n"
        f"<trajectory_json>{json.dumps(trajectory)}</trajectory_json>\n"
        "</example>"
    )


def make_failure_then_recovery_raw() -> str:
    metadata = {
        "id": "ex2",
        "task_id": "django__django-10097",
        "repo": "django/django",
        "prompt_family_id": "P5",
        "example_type": "failure_then_recovery",
        "failure_type": "file_not_found",
        "intended_first_action": "shell",
        "recovery_action": "shell_explore",
        "target_tool": "shell",
    }
    trajectory = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": (
                    "<think>This URL looks path-like, so I will try it.</think>\n"
                    "<tool_call>read_file\n"
                    "<arg_key>path</arg_key><arg_value>django/django/pull/10097</arg_value>\n"
                    "</tool_call>"
                ),
            },
            {"role": "tool", "content": "file not found or outside repo: django/django/pull/10097"},
            {
                "role": "assistant",
                "content": (
                    "<think>The read failed because that was not a repo path. I should search for the issue symbol instead.</think>\n"
                    "<tool_call>shell\n"
                    "<arg_key>cmd</arg_key><arg_value>grep -R \"delete\" -n django/</arg_value>\n"
                    "</tool_call>"
                ),
            },
            {"role": "tool", "content": "django/db/models/deletion.py:10:delete"},
        ],
        "stop_reason": "resolved",
        "patch_nonempty": False,
        "notes": "ok",
    }
    return (
        "<example>\n"
        f"<metadata_json>{json.dumps(metadata)}</metadata_json>\n"
        "<scheme>Construct a failed read then recovery.</scheme>\n"
        f"<trajectory_json>{json.dumps(trajectory)}</trajectory_json>\n"
        "</example>"
    )
