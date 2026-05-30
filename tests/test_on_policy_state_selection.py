from __future__ import annotations

import json

from densify.on_policy.select_states import select_states_from_rollout


def test_select_states_from_rollout_picks_failed_edit_test_and_final(tmp_path) -> None:
    run = tmp_path / "runs" / "20260101T000000Z_demo"
    (run / "requests").mkdir(parents=True)
    for turn in range(1, 6):
        (run / "requests" / f"turn_{turn:04d}.json").write_text(
            json.dumps({"messages": [{"role": "user", "content": f"turn {turn}"}]})
        )
    (run / "rollout_summary.json").write_text(json.dumps({"task_id": "demo"}))
    (run / "tool_calls.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "turn": 1,
                        "tool_name": "read_file",
                        "ok": True,
                        "arguments": {"path": "a.py"},
                        "observation": "ok",
                    }
                ),
                json.dumps(
                    {
                        "turn": 2,
                        "tool_name": "shell",
                        "ok": False,
                        "arguments": {"command": "pytest -q"},
                        "observation": "failed",
                    }
                ),
                json.dumps(
                    {
                        "turn": 3,
                        "tool_name": "apply_patch",
                        "ok": True,
                        "arguments": {"patch": "*** Begin Patch"},
                        "observation": "patched",
                    }
                ),
            ]
        )
        + "\n"
    )
    (run / "summary.json").write_text(json.dumps({"success": False, "turns": 5}))

    states = select_states_from_rollout(run, max_states=4)

    assert [state["selection_reason"] for state in states] == [
        "first_tool_action",
        "first_failed_tool",
        "first_edit_attempt",
        "final_non_exit_turn",
    ]
    assert states[0]["task_id"] == "demo"
    assert states[-1]["turn"] == 5
