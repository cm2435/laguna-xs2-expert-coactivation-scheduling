from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from densify.coding_harness.tools import ToolExecutor
from densify.run_artifacts import append_jsonl, write_json


@dataclass(frozen=True)
class HarnessConfig:
    repo: Path
    output_dir: Path
    api_url: str
    model: str
    task: str
    max_turns: int = 40
    temperature: float = 0.0
    command_timeout_s: int = 120
    output_limit_bytes: int = 20000


@dataclass(frozen=True)
class HarnessResult:
    success: bool
    turns: int
    output_dir: Path


SYSTEM_PROMPT = """You are a coding agent working in a real repository.
Use tools to inspect files, run tests, edit code, and leave the final patch in the working tree.
Prefer minimal changes. Read relevant files before editing. Run focused tests when possible.
Do not install packages or modify the system environment. If tests cannot run because dependencies
are missing, inspect the code statically, make the best repo-local fix, and exit with a summary.
Use repository-relative paths where possible.
When the task is complete, call exit with success=true. If blocked, call exit with success=false.
"""


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command in the repository root.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the repository with optional line bounds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "max_lines": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a complete file within the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a concise patch in Begin Patch / Update File format.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exit",
                "description": "End the rollout.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "summary": {"type": "string"},
                    },
                    "required": ["success"],
                },
            },
        },
    ]


def run_coding_harness(config: HarnessConfig) -> HarnessResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requests").mkdir(exist_ok=True)
    (output_dir / "responses").mkdir(exist_ok=True)
    executor = ToolExecutor(
        config.repo,
        command_timeout_s=config.command_timeout_s,
        output_limit_bytes=config.output_limit_bytes,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Repository root: {Path(config.repo).resolve()}\n\nTask:\n{config.task}",
        },
    ]

    for turn in range(1, config.max_turns + 1):
        request_payload = {
            "model": config.model,
            "messages": messages,
            "tools": tool_schemas(),
            "tool_choice": "auto",
            "temperature": config.temperature,
        }
        write_json(output_dir / "requests" / f"turn_{turn:04d}.json", request_payload)
        started = time.perf_counter()
        response_payload = post_chat_completion(config.api_url, request_payload)
        latency_s = time.perf_counter() - started
        write_json(output_dir / "responses" / f"turn_{turn:04d}.json", response_payload)

        message = extract_message(response_payload)
        message.setdefault("role", "assistant")
        messages.append(message)
        append_jsonl(
            output_dir / "model_turns.jsonl",
            {
                "turn": turn,
                "latency_s": latency_s,
                "content": message.get("content") or "",
                "tool_call_count": len(message.get("tool_calls") or []),
            },
        )

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("id") or f"turn_{turn}_tool")
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name", ""))
            arguments = parse_tool_arguments(function.get("arguments"))
            result = executor.execute(tool_name, arguments)
            append_jsonl(
                output_dir / "tool_calls.jsonl",
                {
                    "turn": turn,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "ok": result.ok,
                    "exit_code": result.exit_code,
                    "observation": result.output,
                },
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result.output,
                }
            )
            if tool_name == "exit":
                success = bool(arguments.get("success")) and result.ok
                write_json(
                    output_dir / "summary.json",
                    {"success": success, "turns": turn, "summary": arguments.get("summary", "")},
                )
                return HarnessResult(success, turn, output_dir)

    write_json(output_dir / "summary.json", {"success": False, "turns": config.max_turns})
    return HarnessResult(False, config.max_turns, output_dir)


def post_chat_completion(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed {exc.code}: {body}") from exc


def extract_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    return dict(choices[0].get("message") or {})


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
