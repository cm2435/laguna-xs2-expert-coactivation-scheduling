from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from densify.run_artifacts import append_jsonl, write_json

AGENT_ID = "00000000-0000-4000-8000-000000000001"
MODEL_ID = "hf-laguna-probe"
SANDBOX_ID = "00000000-0000-4000-8000-000000000003"
SESSION_ID = "00000000-0000-4000-8000-000000000004"


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str
    output_dir: Path


@dataclass(frozen=True)
class ProxyCallRecord:
    call_id: str
    path: str
    status: int
    latency_s: float
    request_path: Path
    served_text_path: Path


def model_summary() -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "name": MODEL_ID,
        "display_name": "vLLM Laguna",
        "context_window": 131072,
    }


def agent_summary() -> dict[str, Any]:
    return {
        "id": AGENT_ID,
        "name": "default",
        "display_name": "vLLM Laguna",
        "default_model_id": MODEL_ID,
        "model_id": MODEL_ID,
        "models": [model_summary()],
    }


def sandbox_definitions() -> dict[str, Any]:
    sandbox_definition = {
        "id": SANDBOX_ID,
        "sandbox_definition_id": SANDBOX_ID,
        "sandboxDefinitionId": SANDBOX_ID,
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
    return {
        "localhost": sandbox_definition,
        "Local": sandbox_definition,
        "sandbox_definitions": [sandbox_definition],
        "sandboxDefinitions": [sandbox_definition],
    }


def extract_text_from_chat_completion(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    delta = first.get("delta") or {}
    if isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


class RecordingProxyHandler(BaseHTTPRequestHandler):
    config: ProxyConfig
    call_index: int = 0

    def do_GET(self) -> None:
        self.log_event({"method": "GET", "path": self.path})
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
            self.write_json(200, model_summary())
            return
        if "/sandbox-definitions" in self.path:
            self.write_json(200, sandbox_definitions())
            return
        if self.path.startswith("/v1/v0/agents/"):
            self.write_json(200, agent_summary())
            return
        if self.path.startswith("/v1/v0/agents?") or self.path == "/v1/v0/agents":
            self.write_json(200, [agent_summary()])
            return
        self.forward_get()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload: dict[str, Any] = json.loads(body or b"{}")
        self.log_event({"method": "POST", "path": self.path, "payload": payload})

        if self.path.endswith("/sessions"):
            self.write_json(
                200,
                {
                    "id": SESSION_ID,
                    "agent_session_id": SESSION_ID,
                    "status": "running",
                    "type": payload.get("type", "local"),
                },
            )
            return
        if self.path.endswith("/trajectory"):
            self.write_json(200, {"ok": True})
            return
        if self.path.endswith("/chat/completions"):
            self.forward_chat_completion(payload)
            return
        self.forward_post(payload)

    def forward_get(self) -> None:
        url = self.upstream_url()
        try:
            with urllib.request.urlopen(url, timeout=120) as upstream:
                data = upstream.read()
                self.write_raw(upstream.status, dict(upstream.headers), data)
        except urllib.error.HTTPError as exc:
            self.write_raw(exc.code, dict(exc.headers), exc.read())

    def forward_post(self, payload: dict[str, Any]) -> None:
        request_data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.upstream_url(),
            data=request_data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as upstream:
                data = upstream.read()
                self.write_raw(upstream.status, dict(upstream.headers), data)
        except urllib.error.HTTPError as exc:
            self.write_raw(exc.code, dict(exc.headers), exc.read())

    def forward_chat_completion(self, payload: dict[str, Any]) -> None:
        call_id = self.next_call_id()
        call_dir = self.config.output_dir / "model_calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        write_json(call_dir / "request.json", payload)

        started = time.perf_counter()
        if payload.get("stream"):
            self.forward_streaming_chat(call_id, call_dir, payload, started)
            return
        self.forward_non_streaming_chat(call_id, call_dir, payload, started)

    def forward_non_streaming_chat(
        self,
        call_id: str,
        call_dir: Path,
        payload: dict[str, Any],
        started: float,
    ) -> None:
        request_data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.upstream_url(),
            data=request_data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as upstream:
                body = upstream.read()
                status = upstream.status
                headers = dict(upstream.headers)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status = exc.code
            headers = dict(exc.headers)

        latency_s = time.perf_counter() - started
        response_payload = json.loads(body or b"{}")
        served_text = extract_text_from_chat_completion(response_payload)
        (call_dir / "served_text.txt").write_text(served_text, encoding="utf-8")
        write_json(call_dir / "response.json", response_payload)
        self.write_call_metadata(call_id, call_dir, status, latency_s, served_text)
        self.write_raw(status, headers, body)

    def forward_streaming_chat(
        self,
        call_id: str,
        call_dir: Path,
        payload: dict[str, Any],
        started: float,
    ) -> None:
        request_data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.upstream_url(),
            data=request_data,
            headers={"content-type": "application/json", "accept": "text/event-stream"},
            method="POST",
        )

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        text_parts: list[str] = []
        stream_path = call_dir / "stream_events.jsonl"
        status = 200
        try:
            with urllib.request.urlopen(request, timeout=3600) as upstream:
                status = upstream.status
                for raw_line in upstream:
                    self.wfile.write(raw_line)
                    self.wfile.flush()
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    append_jsonl(stream_path, {"event": data})
                    if data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text_parts.append(extract_text_from_chat_completion(chunk))
        except urllib.error.HTTPError as exc:
            status = exc.code
            error = exc.read().decode("utf-8", errors="replace")
            append_jsonl(stream_path, {"error": error, "status": status})
            self.wfile.write(f"data: {json.dumps({'error': error})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            self.close_connection = True

        latency_s = time.perf_counter() - started
        served_text = "".join(text_parts)
        (call_dir / "served_text.txt").write_text(served_text, encoding="utf-8")
        self.write_call_metadata(call_id, call_dir, status, latency_s, served_text)

    def write_call_metadata(
        self,
        call_id: str,
        call_dir: Path,
        status: int,
        latency_s: float,
        served_text: str,
    ) -> None:
        metadata = {
            "call_id": call_id,
            "path": self.path,
            "status": status,
            "latency_s": latency_s,
            "served_text_bytes": len(served_text.encode("utf-8")),
            "served_text_path": str(call_dir / "served_text.txt"),
        }
        write_json(call_dir / "metadata.json", metadata)
        append_jsonl(self.config.output_dir / "proxy_requests.jsonl", metadata)

    def upstream_url(self) -> str:
        return f"{self.config.upstream_base_url.rstrip('/')}{self.path.removeprefix('/v1')}"

    def next_call_id(self) -> str:
        type(self).call_index += 1
        return f"call_{type(self).call_index:06d}"

    def write_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_raw(self, status: int, headers: dict[str, Any], body: bytes) -> None:
        self.send_response(status)
        content_type = headers.get("content-type") or headers.get("Content-Type")
        if content_type:
            self.send_header("content-type", str(content_type))
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_event(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.config.output_dir / "proxy_events.jsonl", payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_proxy_server(
    host: str,
    port: int,
    upstream_base_url: str,
    output_dir: str | Path,
) -> ThreadingHTTPServer:
    config = ProxyConfig(upstream_base_url=upstream_base_url, output_dir=Path(output_dir))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    class ConfiguredRecordingProxyHandler(RecordingProxyHandler):
        pass

    ConfiguredRecordingProxyHandler.config = config
    ConfiguredRecordingProxyHandler.call_index = 0
    return ThreadingHTTPServer((host, port), ConfiguredRecordingProxyHandler)
