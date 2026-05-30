from __future__ import annotations

import json
import os
import subprocess
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
    max_tokens: int | None = None
    command_timeout_s: int = 120
    output_limit_bytes: int = 20000
    api_key: str | None = None
    extra_headers: dict[str, str] | None = None


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
For apply_patch, prefer a normal unified diff with repo-relative paths, such as --- a/path.py and +++ b/path.py.
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
        request_payload = build_chat_completion_payload(config, messages)
        write_json(output_dir / "requests" / f"turn_{turn:04d}.json", request_payload)
        started = time.perf_counter()
        response_payload = post_chat_completion(
            config.api_url,
            request_payload,
            api_key=config.api_key,
            extra_headers=config.extra_headers,
        )
        latency_s = time.perf_counter() - started
        write_json(output_dir / "responses" / f"turn_{turn:04d}.json", response_payload)

        message = extract_message(response_payload)
        message.setdefault("role", "assistant")
        messages.append(
            prepare_assistant_message_for_history(
                message,
                preserve_reasoning_details=should_preserve_reasoning_details(config.model),
            )
        )
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
                prepare_tool_message_for_history(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result.output,
                    }
                )
            )
            if tool_name == "exit":
                success = bool(arguments.get("success")) and result.ok
                write_json(
                    output_dir / "summary.json",
                    {
                        "success": success,
                        "turns": turn,
                        "summary": arguments.get("summary", ""),
                        "stopped_reason": "exit",
                        **repo_patch_state(config.repo),
                    },
                )
                return HarnessResult(success, turn, output_dir)

    write_json(
        output_dir / "summary.json",
        {
            "success": False,
            "turns": config.max_turns,
            "stopped_reason": "max_turns",
            **repo_patch_state(config.repo),
        },
    )
    return HarnessResult(False, config.max_turns, output_dir)


def build_chat_completion_payload(config: HarnessConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": sanitize_messages_for_model(messages, config.model),
        "tools": tool_schemas(),
        "tool_choice": "auto",
    }
    if supports_temperature(config.model):
        payload["temperature"] = config.temperature
    if config.max_tokens is not None:
        if requires_max_completion_tokens(config.model):
            payload["max_completion_tokens"] = config.max_tokens
        else:
            payload["max_tokens"] = config.max_tokens
            payload["max_completion_tokens"] = config.max_tokens
    return payload


def sanitize_messages_for_model(messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    if should_preserve_reasoning_details(model):
        return messages
    return [strip_provider_reasoning_fields(message) for message in messages]


def strip_provider_reasoning_fields(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in message.items()
        if key
        not in {
            "reasoning",
            "reasoning_details",
            "reasoning_content",
        }
    }


def normalized_model_id(model: str) -> str:
    return model.lower().removeprefix("openrouter/")


def is_openai_gpt5_family(model: str) -> bool:
    normalized = normalized_model_id(model)
    return normalized.startswith("openai/gpt-5") or normalized.startswith("gpt-5")


def supports_temperature(model: str) -> bool:
    return not is_openai_gpt5_family(model)


def requires_max_completion_tokens(model: str) -> bool:
    return is_openai_gpt5_family(model)


def should_preserve_reasoning_details(model: str) -> bool:
    normalized = normalized_model_id(model)
    if is_openai_gpt5_family(normalized):
        return False
    return not normalized.startswith("openai/")


def repo_patch_state(repo: str | Path) -> dict[str, bool]:
    repo_path = Path(repo)
    diff = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--binary"],
        check=False,
        capture_output=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    status_lines = [
        line
        for line in (status.stdout or "").splitlines()
        if not line[3:].startswith(".poolside/")
    ]
    patch_nonempty = bool((diff.stdout or b"").strip())
    return {"repo_dirty": bool(status_lines), "patch_nonempty": patch_nonempty}


def post_chat_completion(
    api_url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/chat/completions"
    headers = {"content-type": "application/json"}
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    headers.update(extra_headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = json.loads(response.read() or b"{}")
            if "error" in payload:
                raise RuntimeError(f"chat completion failed: {payload['error']}")
            return payload
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed {exc.code}: {body}") from exc


def extract_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    return dict(choices[0].get("message") or {})


def prepare_assistant_message_for_history(
    message: dict[str, Any],
    *,
    preserve_reasoning_details: bool = True,
) -> dict[str, Any]:
    """Keep only Chat Completions input fields, preserving OpenRouter reasoning state."""
    prepared: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    tool_calls = [_prepare_tool_call(tool_call) for tool_call in message.get("tool_calls") or []]
    tool_calls = [tool_call for tool_call in tool_calls if tool_call]
    if tool_calls:
        prepared["tool_calls"] = tool_calls
    reasoning_details = message.get("reasoning_details")
    if preserve_reasoning_details and reasoning_details:
        prepared["reasoning_details"] = reasoning_details
    return prepared


def _prepare_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {}
    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}
    prepared = {
        "id": str(tool_call.get("id") or ""),
        "type": str(tool_call.get("type") or "function"),
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else "{}",
        },
    }
    return prepared if prepared["id"] and prepared["function"]["name"] else {}


def prepare_tool_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": str(message.get("tool_call_id") or ""),
        "content": str(message.get("content") or ""),
    }


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
