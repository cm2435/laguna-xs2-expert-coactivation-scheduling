from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from densify.coding_harness.runner import (
    HarnessConfig,
    build_chat_completion_payload,
    post_chat_completion,
    prepare_assistant_message_for_history,
    prepare_tool_message_for_history,
    run_coding_harness,
)
from densify.coding_harness.tools import ToolExecutor


def init_repo(path):
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "hello.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_tool_executor_runs_shell_and_applies_patch(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    executor = ToolExecutor(repo, command_timeout_s=5, output_limit_bytes=2048)

    shell = executor.execute("shell", {"command": f"{sys.executable} - <<'PY'\nprint('ready')\nPY"})
    assert shell.ok is True
    assert "ready" in shell.output

    patch = """*** Begin Patch
*** Update File: hello.py
@@
-VALUE = 1
+VALUE = 2
*** End Patch
"""
    edit = executor.execute("apply_patch", {"patch": patch})
    assert edit.ok is True
    assert (repo / "hello.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_tool_executor_accepts_laguna_shell_cmd_alias(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    executor = ToolExecutor(repo, command_timeout_s=5, output_limit_bytes=2048)

    shell = executor.execute("shell", {"cmd": f"{sys.executable} - <<'PY'\nprint('ready')\nPY"})

    assert shell.ok is True
    assert "ready" in shell.output


def test_tool_executor_applies_unified_diff_patch(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    executor = ToolExecutor(repo, command_timeout_s=5, output_limit_bytes=2048)

    patch = """diff --git a/hello.py b/hello.py
index 1d3bc16..d0a3882 100644
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 3
"""
    assert executor.looks_like_unified_patch(patch) is True
    edit = executor.execute("apply_patch", {"patch": patch})
    assert edit.ok is True
    assert "hello.py" in edit.output
    assert (repo / "hello.py").read_text(encoding="utf-8") == "VALUE = 3\n"


def test_tool_executor_accepts_absolute_repo_paths_and_blocks_installs(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    executor = ToolExecutor(repo, command_timeout_s=5, output_limit_bytes=2048)

    read = executor.execute("read_file", {"path": str(repo / "hello.py")})
    assert read.ok is True
    assert "VALUE = 1" in read.output

    blocked = executor.execute("shell", {"command": "pip install pyerfa"})
    assert blocked.ok is False
    assert "blocked command" in blocked.output

    compound = executor.execute("shell", {"command": "cd . && pip install pyerfa"})
    assert compound.ok is False
    assert "blocked command" in compound.output

    checkout = executor.execute("shell", {"command": "git checkout main"})
    assert checkout.ok is False
    assert "destructive git" in checkout.output

    compound_checkout = executor.execute("shell", {"command": "pwd && git checkout main"})
    assert compound_checkout.ok is False
    assert "destructive git" in compound_checkout.output


class DummyChatHandler(BaseHTTPRequestHandler):
    seen_payloads = []
    seen_headers = []
    call_count = 0

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload = json.loads(body or b"{}")
        type(self).seen_payloads.append(payload)
        type(self).seen_headers.append(dict(self.headers))
        type(self).call_count += 1
        if type(self).call_count == 1:
            self.write_json(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "shell",
                                            "arguments": json.dumps(
                                                {"command": "printf traced > trace.txt"}
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
            return
        self.write_json(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "exit",
                                        "arguments": json.dumps(
                                            {"success": True, "summary": "done"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    def write_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_harness_executes_openai_tool_calls_and_records_trace(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    DummyChatHandler.seen_payloads = []
    DummyChatHandler.seen_headers = []
    DummyChatHandler.call_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), DummyChatHandler)
    start_server(server)

    result = run_coding_harness(
        HarnessConfig(
            repo=repo,
            output_dir=tmp_path / "run",
            api_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="dummy",
            task="Create trace.txt using a tool.",
            max_turns=4,
        )
    )

    assert result.success is True
    assert (repo / "trace.txt").read_text(encoding="utf-8") == "traced"
    trace_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace_rows[0]["tool_name"] == "shell"
    assert trace_rows[0]["arguments"]["command"] == "printf traced > trace.txt"
    assert (tmp_path / "run" / "requests" / "turn_0001.json").exists()
    assert DummyChatHandler.seen_payloads[0]["tools"][0]["function"]["name"] == "shell"

    server.shutdown()


def test_post_chat_completion_sends_openrouter_headers():
    DummyChatHandler.seen_payloads = []
    DummyChatHandler.seen_headers = []
    DummyChatHandler.call_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), DummyChatHandler)
    start_server(server)

    post_chat_completion(
        f"http://127.0.0.1:{server.server_port}/v1",
        {"model": "openai/gpt-5.5", "messages": []},
        api_key="sk-test",
        extra_headers={"HTTP-Referer": "https://example.test", "X-Title": "Laguna Hackathon"},
    )

    headers = {key.lower(): value for key, value in DummyChatHandler.seen_headers[0].items()}
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["http-referer"] == "https://example.test"
    assert headers["x-title"] == "Laguna Hackathon"

    server.shutdown()


def test_prepare_assistant_message_for_history_preserves_openrouter_reasoning_details():
    message = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "reasoning": "visible summary that should not be replayed as a top-level field",
        "reasoning_details": [
            {
                "type": "reasoning.summary",
                "summary": "summary",
                "format": "azure-openai-responses-v1",
                "index": 0,
            },
            {
                "type": "reasoning.encrypted",
                "data": "encrypted",
                "format": "azure-openai-responses-v1",
                "index": 1,
            },
        ],
        "tool_calls": [
            {
                "type": "function",
                "index": 0,
                "id": "call_1",
                "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"},
            }
        ],
    }

    prepared = prepare_assistant_message_for_history(message)

    assert prepared == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"},
            }
        ],
        "reasoning_details": message["reasoning_details"],
    }


def test_prepare_assistant_message_for_history_can_strip_reasoning_details_for_openai():
    prepared = prepare_assistant_message_for_history(
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "encrypted"}],
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"},
                }
            ],
        },
        preserve_reasoning_details=False,
    )

    assert prepared == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"},
            }
        ],
    }


def test_gpt5_family_payload_uses_strict_chat_completion_fields(tmp_path):
    config = HarnessConfig(
        repo=tmp_path,
        output_dir=tmp_path / "run",
        api_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.5-mini",
        task="x",
        max_tokens=123,
        temperature=0.0,
    )
    payload = build_chat_completion_payload(
        config,
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "drop me",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "drop me"}],
                "reasoning_content": "drop me",
            },
        ],
    )

    assert payload["model"] == "openai/gpt-5.5-mini"
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 123
    assert payload["messages"][1] == {"role": "assistant", "content": ""}
    assert payload["tools"][0]["function"]["name"] == "shell"
    assert payload["tool_choice"] == "auto"


def test_non_openai_payload_preserves_reasoning_compat_fields(tmp_path):
    config = HarnessConfig(
        repo=tmp_path,
        output_dir=tmp_path / "run",
        api_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.6",
        task="x",
        max_tokens=123,
        temperature=0.0,
    )
    payload = build_chat_completion_payload(
        config,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_details": [{"data": "keep"}]},
        ],
    )

    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 123
    assert payload["max_completion_tokens"] == 123
    assert payload["messages"][1]["reasoning_details"] == [{"data": "keep"}]


def test_prepare_tool_message_for_history_drops_name_field():
    prepared = prepare_tool_message_for_history(
        {"role": "tool", "tool_call_id": "call_1", "name": "shell", "content": "ok"}
    )

    assert prepared == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


class PatchOnlyChatHandler(BaseHTTPRequestHandler):
    seen_payloads = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload = json.loads(body or b"{}")
        type(self).seen_payloads.append(payload)
        self.write_json(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_patch",
                                    "type": "function",
                                    "function": {
                                        "name": "apply_patch",
                                        "arguments": json.dumps(
                                            {
                                                "patch": (
                                                    "diff --git a/hello.py b/hello.py\n"
                                                    "index 1d3bc16..d00491f 100644\n"
                                                    "--- a/hello.py\n"
                                                    "+++ b/hello.py\n"
                                                    "@@ -1 +1 @@\n"
                                                    "-VALUE = 1\n"
                                                    "+VALUE = 4\n"
                                                )
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    def write_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def test_harness_max_turn_summary_records_patch_without_exit(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    PatchOnlyChatHandler.seen_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), PatchOnlyChatHandler)
    start_server(server)

    result = run_coding_harness(
        HarnessConfig(
            repo=repo,
            output_dir=tmp_path / "run",
            api_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="dummy",
            task="Patch hello.py but forget to exit.",
            max_turns=1,
        )
    )

    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert result.success is False
    assert summary["stopped_reason"] == "max_turns"
    assert summary["patch_nonempty"] is True
    assert summary["repo_dirty"] is True
    assert (repo / "hello.py").read_text(encoding="utf-8") == "VALUE = 4\n"

    server.shutdown()
