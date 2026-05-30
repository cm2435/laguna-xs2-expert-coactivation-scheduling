from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutSummary:
    run_id: str
    task_id: str
    success: bool
    turns: int
    tool_call_count: int
    patch_bytes: int
    patch_lines_changed: int
    finish_reason: str
    blocked_command_count: int
    deletion_file_count: int
    first_edit_turn: int | None
    first_test_turn: int | None
    quality: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def summarize_rollout(run_dir: Path, sandboxes_dir: Path | None = None) -> RolloutSummary:
    run_id = run_dir.name
    summary = _read_json(run_dir / "summary.json")
    rollout_summary = _read_json(run_dir / "rollout_summary.json")
    task_id = str(rollout_summary.get("task_id") or run_id.split("Z_", 1)[-1])
    tool_rows = _read_jsonl(run_dir / "tool_calls.jsonl")
    model_rows = _read_jsonl(run_dir / "model_turns.jsonl")

    sandbox_patch_path = sandboxes_dir / run_id / "patch.diff" if sandboxes_dir else None
    patch_path = sandbox_patch_path if sandbox_patch_path and sandbox_patch_path.exists() else run_dir / "patch.diff"
    patch_text = ""
    patch_bytes = 0
    if patch_path.exists():
        patch_bytes = patch_path.stat().st_size
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    deletion_file_count = patch_text.count("+++ /dev/null")
    patch_lines_changed = sum(
        1
        for line in patch_text.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )

    first_edit_turn = None
    first_test_turn = None
    blocked_count = 0
    for row in tool_rows:
        turn = int(row.get("turn") or 0)
        tool_name = str(row.get("tool_name") or "")
        arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
        command = str(arguments.get("command") or "")
        observation = str(row.get("observation") or "")
        if "blocked command" in observation:
            blocked_count += 1
        if first_edit_turn is None and tool_name in {"apply_patch", "write_file"}:
            first_edit_turn = turn
        if first_test_turn is None and tool_name == "shell" and _looks_like_test_command(command):
            first_test_turn = turn

    success = bool(summary.get("success"))
    turns = int(summary.get("turns") or len(model_rows))
    finish_reason = "exit_success" if success else ("max_turns" if summary else "incomplete")
    quality = classify_rollout_quality(
        success=success,
        patch_bytes=patch_bytes,
        patch_lines_changed=patch_lines_changed,
        tool_call_count=len(tool_rows),
        blocked_command_count=blocked_count,
        deletion_file_count=deletion_file_count,
    )
    return RolloutSummary(
        run_id=run_id,
        task_id=task_id,
        success=success,
        turns=turns,
        tool_call_count=len(tool_rows),
        patch_bytes=patch_bytes,
        patch_lines_changed=patch_lines_changed,
        finish_reason=finish_reason,
        blocked_command_count=blocked_count,
        deletion_file_count=deletion_file_count,
        first_edit_turn=first_edit_turn,
        first_test_turn=first_test_turn,
        quality=quality,
    )


def classify_rollout_quality(
    *,
    success: bool,
    patch_bytes: int,
    patch_lines_changed: int = 0,
    tool_call_count: int,
    blocked_command_count: int,
    deletion_file_count: int = 0,
) -> str:
    if tool_call_count == 0:
        return "reject"
    if patch_bytes > 200_000:
        return "reject"
    if patch_lines_changed > 2_000:
        return "reject"
    if deletion_file_count > 5:
        return "reject"
    if blocked_command_count > 5:
        return "reject"
    if success and patch_bytes > 0:
        return "silver"
    if tool_call_count > 0:
        return "bronze"
    return "reject"


def _looks_like_test_command(command: str) -> bool:
    needles = ("pytest", "tox", "unittest", "python -m pytest", "manage.py test")
    return any(needle in command for needle in needles)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
