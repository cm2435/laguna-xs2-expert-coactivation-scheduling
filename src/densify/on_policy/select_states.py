from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def select_states_from_rollout(run_dir: Path, *, max_states: int = 5) -> list[dict[str, Any]]:
    task_id = _task_id_for(run_dir)
    tool_rows = _read_jsonl(run_dir / "tool_calls.jsonl")
    model_rows = _read_jsonl(run_dir / "model_turns.jsonl")
    summary = _read_json(run_dir / "summary.json")
    selected: list[dict[str, Any]] = []
    seen_reasons: set[str] = set()

    def add(turn: int, reason: str) -> None:
        if len(selected) >= max_states or reason in seen_reasons:
            return
        request_path = run_dir / "requests" / f"turn_{turn:04d}.json"
        if not request_path.exists():
            return
        selected.append(
            {
                "id": f"{run_dir.name}:turn_{turn:04d}",
                "run_id": run_dir.name,
                "task_id": task_id,
                "turn": turn,
                "selection_reason": reason,
                "request_path": str(request_path),
            }
        )
        seen_reasons.add(reason)

    if tool_rows:
        add(int(tool_rows[0].get("turn") or 1), "first_tool_action")

    for row in tool_rows:
        if row.get("ok") is False:
            add(int(row.get("turn") or 0), "first_failed_tool")
            break

    for row in tool_rows:
        if str(row.get("tool_name") or "") in {"apply_patch", "write_file"}:
            add(int(row.get("turn") or 0), "first_edit_attempt")
            break

    if not summary.get("success"):
        final_turn = int(summary.get("turns") or 0)
        if model_rows:
            final_turn = int(model_rows[-1].get("turn") or final_turn or len(model_rows))
        add(final_turn, "final_non_exit_turn")

    for row in tool_rows:
        if str(row.get("tool_name") or "") == "shell":
            command = str((row.get("arguments") or {}).get("command") or "")
            if any(token in command for token in ("pytest", "tox", "unittest", "python -m pytest")):
                add(int(row.get("turn") or 0), "first_test_attempt")
                break

    return selected[:max_states]


def select_states_from_runs(runs_dir: Path, *, max_states_per_task: int = 5) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        states.extend(select_states_from_rollout(run_dir, max_states=max_states_per_task))
    return states


def write_states_jsonl(states: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(state) + "\n" for state in states), encoding="utf-8")


def _task_id_for(run_dir: Path) -> str:
    rollout_summary = _read_json(run_dir / "rollout_summary.json")
    return str(rollout_summary.get("task_id") or run_dir.name.split("Z_", 1)[-1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
