#!/usr/bin/env python3
"""Deterministic local OpenAI-compatible mock API for comparison benchmarks."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve as serve_websocket

MODEL_ID = "gpt-4o-mini"
RESPONSE_TEXT = "mock response"
STREAM_PARTS = ("mock ", "response")
TOOL_NAME = "lookup"
TOOL_ARGUMENTS = {"value": 1}
STRUCTURED_RESPONSE = {"value": 1}


def _usage(messages: list[dict[str, Any]]) -> dict[str, int]:
    prompt = max(1, sum(len(str(item.get("content", ""))) for item in messages) // 4)
    completion = 3
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class BenchmarkHTTPServer(ThreadingHTTPServer):
    # Keep the localhost mock from becoming the bottleneck in latency and
    # 16-way concurrency measurements.
    request_queue_size = 128

    def get_request(self) -> tuple[Any, Any]:
        import socket

        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mock"}],
                },
            )
            return
        self._json(404, {"error": {"message": "not found", "type": "mock_error"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "mock_error"}})
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        model = payload.get("model") or MODEL_ID

        if payload.get("stream"):
            created = int(time.time())
            events: list[bytes] = []
            for index, text in enumerate(STREAM_PARTS):
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                **({"role": "assistant"} if index == 0 else {}),
                                "content": text,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                events.append(
                    b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"
                )
            final = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": _usage(messages),
            }
            events.append(
                b"data: " + json.dumps(final, separators=(",", ":")).encode() + b"\n\n"
            )
            events.append(b"data: [DONE]\n\n")
            raw = b"".join(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            return

        if payload.get("response_format"):
            latest_user = next(
                (
                    str(item.get("content", ""))
                    for item in reversed(messages)
                    if item.get("role") == "user"
                ),
                "",
            )
            content = (
                '{"value":"invalid"}'
                if "[structured-invalid]" in latest_user
                and "[structured-repair]" not in latest_user
                else json.dumps(STRUCTURED_RESPONSE, separators=(",", ":"))
            )
            self._json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": _usage(messages),
                },
            )
            return

        if payload.get("tools") and not any(
            item.get("role") == "tool" for item in messages
        ):
            self._json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_mock_lookup",
                                "type": "function",
                                "function": {
                                    "name": TOOL_NAME,
                                    "arguments": json.dumps(
                                        TOOL_ARGUMENTS,
                                        separators=(",", ":"),
                                    ),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": _usage(messages),
                },
            )
            return

        self._json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": RESPONSE_TEXT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(messages),
            },
        )



class ResponsesWebSocketMockServer:
    """Deterministic Responses WebSocket mock used by Agent RT runtime benchmarks."""

    def __init__(self) -> None:
        self._response_counter = 0
        self.server = serve_websocket(self._handler, "127.0.0.1", 0)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="mock-openai-responses-ws",
            daemon=True,
        )
        self.thread.start()
        host, port = self.server.socket.getsockname()[:2]
        self.base_url = f"http://{host}:{port}/v1"

    def _handler(self, websocket: Any) -> None:
        try:
            self._handle_messages(websocket)
        except ConnectionClosed:
            pass

    def _handle_messages(self, websocket: Any) -> None:
        for raw in websocket:
            payload = json.loads(raw)
            if payload.get("type") != "response.create":
                websocket.send(json.dumps({
                    "type": "error",
                    "error": {"message": "expected response.create"},
                }))
                continue

            self._response_counter += 1
            response_id = f"resp_mock_{self._response_counter}"
            stream_id = payload.get("stream_id")
            model = payload.get("model") or MODEL_ID
            input_items = payload.get("input") or []
            text = (
                json.dumps(STRUCTURED_RESPONSE, separators=(",", ":"))
                if payload.get("text")
                else RESPONSE_TEXT
            )
            usage = {
                "input_tokens": max(1, len(json.dumps(input_items)) // 4),
                "output_tokens": 3,
            }
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

            if payload.get("tools") and not any(
                item.get("type") == "function_call_output"
                for item in input_items
                if isinstance(item, dict)
            ):
                arguments = json.dumps(TOOL_ARGUMENTS, separators=(",", ":"))
                websocket.send(json.dumps({
                    "type": "response.output_item.added",
                    "stream_id": stream_id,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_mock_lookup",
                        "call_id": "call_mock_lookup",
                        "name": TOOL_NAME,
                        "arguments": "",
                    },
                }, separators=(",", ":")))
                websocket.send(json.dumps({
                    "type": "response.function_call_arguments.delta",
                    "stream_id": stream_id,
                    "output_index": 0,
                    "delta": arguments,
                }, separators=(",", ":")))
                output = [{
                    "type": "function_call",
                    "id": "fc_mock_lookup",
                    "call_id": "call_mock_lookup",
                    "name": TOOL_NAME,
                    "arguments": arguments,
                }]
            else:
                for part in STREAM_PARTS if text == RESPONSE_TEXT else (text,):
                    websocket.send(json.dumps({
                        "type": "response.output_text.delta",
                        "stream_id": stream_id,
                        "delta": part,
                    }, separators=(",", ":")))
                output = [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }]

            websocket.send(json.dumps({
                "type": "response.completed",
                "stream_id": stream_id,
                "response": {
                    "id": response_id,
                    "model": model,
                    "output": output,
                    "usage": usage,
                },
            }, separators=(",", ":")))

    def shutdown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


def start_responses_websocket_server() -> ResponsesWebSocketMockServer:
    return ResponsesWebSocketMockServer()


def start_mock_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = BenchmarkHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, name="mock-openai", daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, thread, f"http://{host}:{port}/v1"


if __name__ == "__main__":
    server, _, base_url = start_mock_server()
    print(base_url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
