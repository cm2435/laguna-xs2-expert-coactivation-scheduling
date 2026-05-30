from __future__ import annotations

import json

from densify.rollout_sft.summary import classify_rollout_quality, summarize_rollout


def test_classify_rollout_quality_rejects_giant_patch() -> None:
    assert (
        classify_rollout_quality(
            success=True,
            patch_bytes=300_000,
            tool_call_count=10,
            blocked_command_count=0,
            deletion_file_count=0,
        )
        == "reject"
    )


def test_summarize_rollout_reads_patch_and_tools(tmp_path) -> None:
    run = tmp_path / "runs" / "20260101T000000Z_demo"
    sandbox = tmp_path / "sandboxes" / "20260101T000000Z_demo"
    run.mkdir(parents=True)
    sandbox.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"success": True, "turns": 3}))
    (run / "rollout_summary.json").write_text(json.dumps({"task_id": "demo"}))
    (run / "tool_calls.jsonl").write_text(
        json.dumps({"turn": 1, "tool_name": "read_file", "arguments": {}, "observation": "ok"})
        + "\n"
        + json.dumps({"turn": 2, "tool_name": "apply_patch", "arguments": {}, "observation": "ok"})
        + "\n",
        encoding="utf-8",
    )
    (sandbox / "patch.diff").write_text("diff --git a/a.py b/a.py\n")

    summary = summarize_rollout(run, tmp_path / "sandboxes")

    assert summary.task_id == "demo"
    assert summary.success is True
    assert summary.patch_bytes > 0
    assert summary.first_edit_turn == 2
    assert summary.quality == "silver"
