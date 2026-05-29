from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from densify.coding_harness.runner import HarnessConfig, run_coding_harness
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

    shell = executor.execute("shell", {"command": "python - <<'PY'\nprint('ready')\nPY"})
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


class DummyChatHandler(BaseHTTPRequestHandler):
    seen_payloads = []
    call_count = 0

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload = json.loads(body or b"{}")
        type(self).seen_payloads.append(payload)
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
