from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


def build_teacher_payload(
    state: dict[str, Any],
    *,
    teacher_model: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    request = json.loads(Path(state["request_path"]).read_text(encoding="utf-8"))
    payload = {
        "model": teacher_model,
        "messages": request.get("messages") or [],
        "temperature": temperature,
    }
    if request.get("tools"):
        payload["tools"] = request["tools"]
        payload["tool_choice"] = request.get("tool_choice", "auto")
    return payload


def correction_row_from_response(
    state: dict[str, Any],
    response: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choices = response.get("choices") or []
    teacher_action = dict(choices[0].get("message") or {}) if choices else {}
    return {
        "id": f"{state['id']}:teacher",
        "task_id": state.get("task_id"),
        "run_id": state.get("run_id"),
        "turn": state.get("turn"),
        "selection_reason": state.get("selection_reason"),
        "context_messages": (payload or {}).get("messages", []),
        "teacher_action": teacher_action,
        "quality": "teacher_correction",
    }


def query_teacher(api_url: str, payload: dict[str, Any], *, timeout_s: int = 3600) -> dict[str, Any]:
    request = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read() or b"{}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
