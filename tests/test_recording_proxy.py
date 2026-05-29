from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from densify.openai_proxy.recording_proxy import make_proxy_server


class DummyUpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/models":
            self.write_json({"object": "list", "data": [{"id": "dummy"}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        payload = json.loads(body or b"{}")
        if self.path == "/chat/completions" and payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            chunk = {
                "choices": [{"index": 0, "delta": {"content": "READY"}, "finish_reason": None}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if self.path == "/chat/completions":
            self.write_json({"choices": [{"message": {"content": "READY"}}]})
            return
        self.send_response(404)
        self.end_headers()

    def write_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def start_server(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_recording_proxy_forwards_non_streaming_chat(tmp_path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DummyUpstreamHandler)
    start_server(upstream)
    proxy = make_proxy_server(
        "127.0.0.1",
        0,
        f"http://127.0.0.1:{upstream.server_port}",
        tmp_path,
    )
    start_server(proxy)

    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read())

    assert payload["choices"][0]["message"]["content"] == "READY"
    assert (tmp_path / "model_calls" / "call_000001" / "request.json").exists()
    assert (tmp_path / "model_calls" / "call_000001" / "served_text.txt").read_text() == "READY"

    proxy.shutdown()
    upstream.shutdown()


def test_recording_proxy_forwards_streaming_chat(tmp_path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DummyUpstreamHandler)
    start_server(upstream)
    proxy = make_proxy_server(
        "127.0.0.1",
        0,
        f"http://127.0.0.1:{upstream.server_port}",
        tmp_path,
    )
    start_server(proxy)

    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
        data=json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode()

    assert "READY" in body
    assert (tmp_path / "model_calls" / "call_000001" / "served_text.txt").read_text() == "READY"
    assert (tmp_path / "model_calls" / "call_000001" / "stream_events.jsonl").exists()

    proxy.shutdown()
    upstream.shutdown()
