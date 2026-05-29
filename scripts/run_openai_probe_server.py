from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ProbeHandler(BaseHTTPRequestHandler):
    log_path: Path

    def agent_summary(self) -> dict[str, Any]:
        agent_id = "00000000-0000-4000-8000-000000000001"
        return {
            "id": agent_id,
            "name": "default",
            "display_name": "HF Laguna Probe",
            "default_model_id": self.model_summary()["id"],
            "model_id": self.model_summary()["id"],
            "models": [self.model_summary()],
        }

    def model_summary(self) -> dict[str, Any]:
        return {
            "id": "00000000-0000-4000-8000-000000000002",
            "name": "hf-laguna-probe",
            "display_name": "HF Laguna Probe",
            "context_window": 131072,
        }

    def do_GET(self) -> None:
        self.log_payload({"method": "GET", "path": self.path})
        if self.path == "/v1/v0/environment":
            self.write_json(
                200,
                {
                    "id": "localhost",
                    "type": "localhost",
                    "execution_environment_type": "localhost",
                    "available": True,
                },
            )
            return
        if self.path.startswith("/v1/v0/model/"):
            self.write_json(200, self.model_summary())
            return
        if "/sandbox-definitions" in self.path:
            sandbox_definition = {
                "id": "00000000-0000-4000-8000-000000000003",
                "sandbox_definition_id": "00000000-0000-4000-8000-000000000003",
                "sandboxDefinitionId": "00000000-0000-4000-8000-000000000003",
                "name": "localhost",
                "display_name": "Local",
                "displayName": "Local",
                "execution_environment_type": "localhost",
                "executionEnvironmentType": "localhost",
                "execution_environment_id": "localhost",
                "executionEnvironmentId": "localhost",
                "enabled": True,
                "Enabled": True,
            }
            self.write_json(
                200,
                {
                    "localhost": sandbox_definition,
                    "Local": sandbox_definition,
                    "sandbox_definitions": [sandbox_definition],
                    "sandboxDefinitions": [sandbox_definition],
                },
            )
            return
        if self.path.startswith("/v1/v0/agents/"):
            self.write_json(200, self.agent_summary())
            return
        if self.path.startswith("/v1/v0/agents?"):
            self.write_json(200, [self.agent_summary()])
            return
        if self.path == "/v1/v0/agents":
            self.write_json(200, [self.agent_summary()])
            return
        if self.path == "/v1/models":
            self.write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "hf-laguna-probe",
                            "object": "model",
                            "created": 0,
                            "owned_by": "densify",
                        }
                    ],
                },
            )
            return
        self.write_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload: Any = json.loads(body or b"{}")
        self.log_payload({"method": "POST", "path": self.path, "payload": payload})
        if self.path.endswith("/sessions"):
            self.write_json(
                200,
                {
                    "id": "00000000-0000-4000-8000-000000000004",
                    "agent_session_id": "00000000-0000-4000-8000-000000000004",
                    "status": "running",
                    "type": payload.get("type", "local"),
                },
            )
            return
        if self.path.endswith("/trajectory"):
            self.write_json(200, {"ok": True})
            return
        if self.path.endswith("/chat/completions"):
            if payload.get("stream"):
                self.write_chat_completion_stream(payload)
                return
            self.write_json(
                200,
                {
                    "id": "chatcmpl-probe",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", "hf-laguna-probe"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "READY"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
            return
        self.write_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def write_chat_completion_stream(self, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        model = payload.get("model", "hf-laguna-probe")
        chunks = [
            {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "READY"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def write_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_payload(self, payload: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"path": self.path, "payload": payload}) + "\n")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-path", default="runs/openai_probe/requests.jsonl")
    args = parser.parse_args()

    ProbeHandler.log_path = Path(args.log_path)
    server = ThreadingHTTPServer((args.host, args.port), ProbeHandler)
    print(f"serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
