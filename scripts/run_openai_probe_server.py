from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ProbeHandler(BaseHTTPRequestHandler):
    log_path: Path

    def do_GET(self) -> None:
        self.log_payload({"method": "GET", "path": self.path})
        if self.path == "/v1/v0/agents":
            self.write_json(
                200,
                {
                    "agents": [
                        {
                            "id": "default",
                            "name": "default",
                            "display_name": "HF Laguna Probe",
                            "models": [{"id": "hf-laguna-probe", "name": "hf-laguna-probe"}],
                        }
                    ]
                },
            )
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
        if self.path == "/v1/chat/completions":
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
