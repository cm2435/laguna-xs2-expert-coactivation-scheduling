from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

JSON = dict[str, Any]


@dataclass(frozen=True)
class JsonRpcRequest:
    id: str | int | None
    method: str
    params: JSON


def parse_jsonrpc_message(raw: str) -> JSON:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON-RPC payload must be an object")
    return payload


def request_from_message(message: JSON) -> JsonRpcRequest | None:
    method = message.get("method")
    if not isinstance(method, str):
        return None
    params = message.get("params") or {}
    if not isinstance(params, dict):
        params = {"value": params}
    return JsonRpcRequest(id=message.get("id"), method=method, params=params)


def response(message_id: str | int | None, result: Any) -> JSON:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def error_response(message_id: str | int | None, code: int, message: str) -> JSON:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def notification(method: str, params: JSON) -> JSON:
    return {"jsonrpc": "2.0", "method": method, "params": params}


class NdjsonConnection:
    def __init__(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def read_messages(self):
        for line in self.stdin:
            if not line.strip():
                continue
            message = parse_jsonrpc_message(line)
            self.log("in", message)
            yield message

    def send(self, message: JSON) -> None:
        self.log("out", message)
        self.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.stdout.flush()

    def log(self, direction: str, message: JSON) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"direction": direction, "message": message}) + "\n")

