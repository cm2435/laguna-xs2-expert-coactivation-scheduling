from __future__ import annotations

import json

from densify.on_policy.teacher_corrections import build_teacher_payload, correction_row_from_response


def test_build_teacher_payload_uses_original_request_messages_and_tools(tmp_path) -> None:
    request = tmp_path / "turn_0003.json"
    request.write_text(
        json.dumps(
            {
                "model": "student",
                "messages": [{"role": "user", "content": "fix bug"}],
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            }
        )
    )
    state = {
        "id": "run:turn_0003",
        "task_id": "task",
        "run_id": "run",
        "turn": 3,
        "selection_reason": "first_failed_tool",
        "request_path": str(request),
    }

    payload = build_teacher_payload(state, teacher_model="laguna")

    assert payload["model"] == "laguna"
    assert payload["messages"][0]["content"] == "fix bug"
    assert payload["tools"][0]["function"]["name"] == "shell"
    assert payload["temperature"] == 0.0


def test_correction_row_from_response_preserves_teacher_action() -> None:
    state = {
        "id": "run:turn_0003",
        "task_id": "task",
        "run_id": "run",
        "turn": 3,
        "selection_reason": "first_failed_tool",
    }
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
                }
            }
        ]
    }

    row = correction_row_from_response(state, response)

    assert row["id"] == "run:turn_0003:teacher"
    assert row["teacher_action"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert row["quality"] == "teacher_correction"
