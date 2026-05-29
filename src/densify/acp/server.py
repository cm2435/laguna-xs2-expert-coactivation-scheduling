from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from densify.acp.protocol import (
    NdjsonConnection,
    error_response,
    notification,
    request_from_message,
    response,
)


@dataclass
class DummyAcpServer:
    log_path: Path
    fixed_response: str = "READY"
    sessions: dict[str, dict] = field(default_factory=dict)

    def serve(self) -> None:
        conn = NdjsonConnection(log_path=self.log_path)
        for message in conn.read_messages():
            request = request_from_message(message)
            if request is None:
                continue

            if request.method == "initialize":
                conn.send(
                    response(
                        request.id,
                        {
                            "protocolVersion": 1,
                            "agentCapabilities": {
                                "loadSession": False,
                                "promptCapabilities": {},
                            },
                            "agentInfo": {
                                "name": "hf-laguna-dummy",
                                "title": "HF Laguna Dummy",
                                "version": "0.1.0",
                            },
                            "authMethods": [],
                        },
                    )
                )
                continue

            if request.method == "session/new":
                session_id = f"sess_{uuid4().hex}"
                self.sessions[session_id] = {
                    "cwd": request.params.get("cwd"),
                    "mcpServers": request.params.get("mcpServers", []),
                }
                conn.send(response(request.id, {"sessionId": session_id}))
                continue

            if request.method == "session/prompt":
                session_id = str(request.params.get("sessionId", ""))
                conn.send(
                    notification(
                        "session/update",
                        {
                            "sessionId": session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": self.fixed_response},
                            },
                        },
                    )
                )
                conn.send(response(request.id, {"stopReason": "end_turn"}))
                continue

            if request.method == "session/cancel":
                conn.send(response(request.id, {"stopReason": "cancelled"}))
                continue

            if request.method == "session/close":
                session_id = str(request.params.get("sessionId", ""))
                self.sessions.pop(session_id, None)
                conn.send(response(request.id, {}))
                continue

            conn.send(error_response(request.id, -32601, f"Method not found: {request.method}"))


def run_dummy_server(log_path: str | Path, fixed_response: str = "READY") -> None:
    DummyAcpServer(log_path=Path(log_path), fixed_response=fixed_response).serve()

