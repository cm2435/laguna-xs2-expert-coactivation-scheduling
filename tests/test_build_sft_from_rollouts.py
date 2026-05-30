from __future__ import annotations

import json

from densify.rollout_sft.build_dataset import (
    build_sft_rows,
    build_sft_rows_from_manifest,
    reconstruct_messages,
)


def make_fake_run(root, *, turns: int = 1):
    run = root / "runs" / "20260101T000000Z_demo"
    sandbox = root / "sandboxes" / "20260101T000000Z_demo"
    (run / "requests").mkdir(parents=True)
    (run / "responses").mkdir()
    sandbox.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"success": True, "turns": turns}))
    (run / "rollout_summary.json").write_text(json.dumps({"task_id": "demo"}))
    (run / "requests" / "turn_0001.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                ],
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            }
        )
    )
    tool_lines = []
    for turn in range(1, turns + 1):
        call_id = f"call_{turn}"
        command = "pytest -q" if turn == 1 else "sed -n '1,80p' a.py"
        (run / "responses" / f"turn_{turn:04d}.json").write_text(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": "shell",
                                            "arguments": json.dumps({"command": command}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        )
        tool_lines.append(
            json.dumps(
                {
                    "turn": turn,
                    "tool_call_id": call_id,
                    "tool_name": "shell",
                    "arguments": {"command": command},
                    "ok": True,
                    "observation": f"observation {turn}",
                }
            )
        )
    (run / "tool_calls.jsonl").write_text("\n".join(tool_lines) + "\n")
    (sandbox / "patch.diff").write_text("diff --git a/a.py b/a.py\n")
    return run, sandbox


def test_reconstruct_messages_preserves_assistant_and_tool_order(tmp_path) -> None:
    run, _ = make_fake_run(tmp_path)

    messages = reconstruct_messages(run)

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "shell"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"command": "pytest -q"}
    assert messages[3]["content"] == "observation 1"


def test_build_sft_rows_targets_assistant_not_tool_observation(tmp_path) -> None:
    make_fake_run(tmp_path)

    rows = build_sft_rows(tmp_path / "runs", tmp_path / "sandboxes")

    assert len(rows) == 1
    assert rows[0]["task_id"] == "demo"
    assert rows[0]["quality"] == "silver"
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["metadata"]["target_role"] == "assistant"


def test_build_sft_rows_emits_one_row_per_assistant_action(tmp_path) -> None:
    make_fake_run(tmp_path, turns=2)

    rows = build_sft_rows(tmp_path / "runs", tmp_path / "sandboxes")

    assert len(rows) == 2
    assert [row["id"] for row in rows] == [
        "20260101T000000Z_demo:assistant_0001",
        "20260101T000000Z_demo:assistant_0002",
    ]
    assert [row["messages"][-1]["role"] for row in rows] == ["assistant", "assistant"]
    assert [message["role"] for message in rows[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_build_sft_rows_can_require_patch_without_sandbox_dir(tmp_path) -> None:
    run, _ = make_fake_run(tmp_path)
    (run / "patch.diff").write_text("diff --git a/a.py b/a.py\n")

    rows = build_sft_rows(tmp_path / "runs", None, require_patch=True)

    assert len(rows) == 1
    assert rows[0]["metadata"]["patch_bytes"] > 0


def test_build_sft_rows_filters_large_patch_and_limits_after_first_edit(tmp_path) -> None:
    run, sandbox = make_fake_run(tmp_path, turns=3)
    (sandbox / "patch.diff").write_text("diff --git a/a.py b/a.py\n" + ("+x\n" * 100))
    # Add an edit on turn 2 so the builder can keep turns through turn 2 only.
    rows = (run / "tool_calls.jsonl").read_text().splitlines()
    rows[1] = json.dumps(
        {
            "turn": 2,
            "tool_call_id": "call_2",
            "tool_name": "apply_patch",
            "arguments": {"patch": "diff --git a/a.py b/a.py\n"},
            "ok": True,
            "observation": "updated a.py",
        }
    )
    (run / "tool_calls.jsonl").write_text("\n".join(rows) + "\n")

    rejected = build_sft_rows(
        tmp_path / "runs",
        tmp_path / "sandboxes",
        require_patch=True,
        max_patch_lines=10,
    )
    kept = build_sft_rows(
        tmp_path / "runs",
        tmp_path / "sandboxes",
        require_patch=True,
        max_patch_lines=200,
        turns_after_first_edit=0,
    )

    assert rejected == []
    assert len(kept) == 2
    assert all(row["metadata"]["assistant_index"] <= 2 for row in kept)


def test_build_sft_rows_from_manifest_uses_exact_banked_rollouts(tmp_path) -> None:
    run, sandbox = make_fake_run(tmp_path)
    (tmp_path / "runs" / "20260101T000001Z_extra").mkdir(parents=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "demo",
                "source_run": "main",
                "rollout_dir": str(run),
                "patch_path": str(sandbox / "patch.diff"),
                "harness_success": True,
                "patch_nonempty": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_sft_rows_from_manifest(manifest)

    assert len(rows) == 1
    assert rows[0]["source_rollout"] == str(run)
    assert rows[0]["metadata"]["manifest_source_run"] == "main"
