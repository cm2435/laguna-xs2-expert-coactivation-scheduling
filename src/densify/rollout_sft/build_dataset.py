from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from densify.rollout_sft.summary import summarize_rollout


def reconstruct_messages(run_dir: Path, max_turns: int | None = None) -> list[dict[str, Any]]:
    request_paths = sorted((run_dir / "requests").glob("turn_*.json"))
    response_paths = sorted((run_dir / "responses").glob("turn_*.json"))
    if not request_paths:
        return []

    first_request = _read_json(request_paths[0])
    messages = list(first_request.get("messages") or [])
    tool_rows_by_turn = _tool_rows_by_turn(run_dir / "tool_calls.jsonl")

    for turn, response_path in enumerate(response_paths, start=1):
        if max_turns is not None and turn > max_turns:
            break
        response = _read_json(response_path)
        choices = response.get("choices") or []
        if not choices:
            continue
        assistant = _normalize_assistant_message(dict(choices[0].get("message") or {}))
        assistant.setdefault("role", "assistant")
        messages.append(assistant)
        for tool in tool_rows_by_turn.get(turn, []):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool.get("tool_call_id") or ""),
                    "name": str(tool.get("tool_name") or ""),
                    "content": str(tool.get("observation") or ""),
                }
            )
    return messages


def build_sft_rows_from_manifest(
    manifest_path: Path,
    *,
    include_qualities: set[str] | None = None,
    max_turns: int | None = None,
    require_patch: bool = False,
    require_harness_success: bool = False,
    max_patch_bytes: int | None = None,
    max_patch_lines: int | None = None,
    turns_after_first_edit: int | None = None,
    exclude_source_contains: str | None = None,
) -> list[dict[str, Any]]:
    include_qualities = include_qualities or {"gold", "silver", "bronze"}
    rows: list[dict[str, Any]] = []
    for manifest_row in _read_jsonl(manifest_path):
        source_run = str(manifest_row.get("source_run") or "")
        if exclude_source_contains and exclude_source_contains in source_run:
            continue
        if require_harness_success and not bool(manifest_row.get("harness_success")):
            continue
        if require_patch and not bool(manifest_row.get("patch_nonempty")):
            continue

        run_dir = Path(str(manifest_row["rollout_dir"]))
        patch_path = Path(str(manifest_row.get("patch_path") or ""))
        sandboxes_dir = patch_path.parent.parent if patch_path.name == "patch.diff" else None
        summary = summarize_rollout(run_dir, sandboxes_dir)
        if require_patch and summary.patch_bytes <= 0:
            continue
        if max_patch_bytes is not None and summary.patch_bytes > max_patch_bytes:
            continue
        if max_patch_lines is not None and summary.patch_lines_changed > max_patch_lines:
            continue
        if summary.quality not in include_qualities:
            continue

        row_max_turns = max_turns
        if summary.first_edit_turn is not None and turns_after_first_edit is not None:
            edit_limit = summary.first_edit_turn + turns_after_first_edit
            row_max_turns = min(row_max_turns, edit_limit) if row_max_turns is not None else edit_limit
        messages = reconstruct_messages(run_dir, max_turns=row_max_turns)
        if not messages:
            continue
        weight = 1.0 if summary.quality in {"gold", "silver"} else 0.5
        for assistant_index, target_messages in enumerate(_assistant_action_prefixes(messages), start=1):
            metadata = summary.to_json()
            metadata.update(
                {
                    "assistant_index": assistant_index,
                    "target_role": "assistant",
                    "manifest_source_run": source_run,
                    "manifest_harness_success": bool(manifest_row.get("harness_success")),
                    "manifest_patch_nonempty": bool(manifest_row.get("patch_nonempty")),
                }
            )
            rows.append(
                {
                    "id": f"{summary.run_id}:assistant_{assistant_index:04d}",
                    "task_id": summary.task_id,
                    "source_rollout": str(run_dir),
                    "messages": target_messages,
                    "quality": summary.quality,
                    "weight": weight,
                    "metadata": metadata,
                }
            )
    return rows


def build_sft_rows(
    runs_dir: Path,
    sandboxes_dir: Path | None,
    *,
    include_qualities: set[str] | None = None,
    max_turns: int | None = None,
    require_patch: bool = False,
    max_patch_bytes: int | None = None,
    max_patch_lines: int | None = None,
    turns_after_first_edit: int | None = None,
) -> list[dict[str, Any]]:
    include_qualities = include_qualities or {"gold", "silver", "bronze"}
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        summary = summarize_rollout(run_dir, sandboxes_dir)
        if require_patch and summary.patch_bytes <= 0:
            continue
        if max_patch_bytes is not None and summary.patch_bytes > max_patch_bytes:
            continue
        if max_patch_lines is not None and summary.patch_lines_changed > max_patch_lines:
            continue
        if summary.quality not in include_qualities:
            continue
        row_max_turns = max_turns
        if summary.first_edit_turn is not None and turns_after_first_edit is not None:
            edit_limit = summary.first_edit_turn + turns_after_first_edit
            row_max_turns = min(row_max_turns, edit_limit) if row_max_turns is not None else edit_limit
        messages = reconstruct_messages(run_dir, max_turns=row_max_turns)
        if not messages:
            continue
        weight = 1.0 if summary.quality in {"gold", "silver"} else 0.5
        for assistant_index, target_messages in enumerate(_assistant_action_prefixes(messages), start=1):
            metadata = summary.to_json()
            metadata.update({"assistant_index": assistant_index, "target_role": "assistant"})
            rows.append(
                {
                    "id": f"{summary.run_id}:assistant_{assistant_index:04d}",
                    "task_id": summary.task_id,
                    "source_rollout": str(run_dir),
                    "messages": target_messages,
                    "quality": summary.quality,
                    "weight": weight,
                    "metadata": metadata,
                }
            )
    return rows


def write_sft_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    normalized_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_calls.append(tool_call)
            continue
        call = dict(tool_call)
        function = call.get("function")
        if isinstance(function, dict):
            function = dict(function)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {"arguments": arguments}
                function["arguments"] = parsed_arguments if isinstance(parsed_arguments, dict) else {"value": parsed_arguments}
            call["function"] = function
        normalized_calls.append(call)
    message["tool_calls"] = normalized_calls
    return message


def _assistant_action_prefixes(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            rows.append([dict(item) for item in messages[: index + 1]])
    return rows


def _tool_rows_by_turn(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.setdefault(int(row.get("turn") or 0), []).append(row)
    return rows
