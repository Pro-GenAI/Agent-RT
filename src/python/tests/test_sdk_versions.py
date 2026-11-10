"""Real vendor-SDK integration against a local, offline wire-format double.

Most provider tests inject fake clients. These tests instead drive whichever
``openai`` / ``anthropic`` SDK version is installed against a loopback HTTP
server, so ``make test-deps-matrix`` can prove every supported SDK release
(mid-2024 onward) still works. No real model endpoint is ever contacted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from agent_rt import (
    AnthropicModelProvider,
    AnthropicProviderSettings,
    ContentPart,
    ModelMessage,
    ModelRequest,
    OpenAIModelProvider,
    OpenAIProviderSettings,
    ReasoningConfig,
    ToolDefinition,
)

MODEL = "sdk-matrix-model"
WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Look up the weather",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def _sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    lines: list[str] = []
    for event, data in events:
        if event is not None:
            lines.append(f"event: {event}")
        payload = data if isinstance(data, str) else json.dumps(data)
        lines.append(f"data: {payload}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def _openai_completion(body: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": "hello from openai"}
    finish = "stop"
    if body.get("tools"):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
        finish = "tool_calls"
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _openai_stream() -> bytes:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    usage = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    return _sse(
        [
            (None, chunk({"role": "assistant", "content": "hel"})),
            (None, chunk({"content": "lo"})),
            (None, chunk({}, "stop")),
            (None, usage),
            (None, "[DONE]"),
        ]
    )


def _anthropic_message(body: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": "hello from claude"}]
    stop = "end_turn"
    if body.get("tools"):
        content = [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ]
        stop = "tool_use"
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _anthropic_stream() -> bytes:
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
    }
    return _sse(
        [
            ("message_start", start),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hel"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


class _WireDouble(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, dict[str, Any]]]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any) -> None:
        self._reply(200, json.dumps(value).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        self.requests.append(("GET", path, {}))
        if path == "/v1/models":
            if "anthropic-version" in self.headers:
                self._json(
                    {
                        "data": [
                            {
                                "type": "model",
                                "id": MODEL,
                                "display_name": "Matrix",
                                "created_at": "2024-06-01T00:00:00Z",
                            }
                        ],
                        "has_more": False,
                        "first_id": MODEL,
                        "last_id": MODEL,
                    }
                )
            else:
                self._json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL,
                                "object": "model",
                                "created": 1,
                                "owned_by": "agent-rt",
                            }
                        ],
                    }
                )
            return
        self._reply(404, b'{"error": "not found"}', "application/json")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append(("POST", path, body))
        if path == "/v1/chat/completions":
            if body.get("stream"):
                self._reply(200, _openai_stream(), "text/event-stream")
            else:
                self._json(_openai_completion(body))
        elif path == "/v1/messages":
            if body.get("stream"):
                self._reply(200, _anthropic_stream(), "text/event-stream")
            else:
                self._json(_anthropic_message(body))
        elif path == "/v1/messages/count_tokens":
            self._json({"input_tokens": 42})
        else:
            self._reply(404, b'{"error": "not found"}', "application/json")


@pytest.fixture
def wire() -> Iterator[tuple[str, list[tuple[str, str, dict[str, Any]]]]]:
    recorded: list[tuple[str, str, dict[str, Any]]] = []
    handler = type("Handler", (_WireDouble,), {"requests": recorded})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", recorded
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(**kwargs: Any) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage("user", [ContentPart("text", "hi")])],
        model=MODEL,
        **kwargs,
    )


def _bodies(recorded: list[tuple[str, str, dict[str, Any]]], path: str) -> list[Any]:
    return [body for method, seen, body in recorded if seen == path]


async def test_openai_sdk_complete_stream_and_validate(wire: Any) -> None:
    pytest.importorskip("openai")
    base_url, recorded = wire
    settings = OpenAIProviderSettings(
        base_url=base_url + "/v1",
        api_key="test-key",  # pragma: allowlist secret
        default_model=MODEL,
        transport="sdk",
        websocket=False,
    )
    assert settings.validate_connection(check_default_model=True) == (MODEL,)

    provider = OpenAIModelProvider(settings)
    try:
        # reasoning_effort / max_completion_tokens postdate the oldest
        # supported SDKs; they must still reach the wire body.
        response = await provider.complete(
            _request(
                max_output_tokens=64,
                reasoning=ReasoningConfig(effort="low"),
            )
        )
        assert response.message.content[0].text == "hello from openai"
        assert response.usage is not None and response.usage.input_tokens == 5

        tool_response = await provider.complete(_request(tools=[WEATHER_TOOL]))
        assert tool_response.finish_reason == "tool_calls"
        assert tool_response.message.tool_calls[0].arguments == {"city": "Paris"}

        events = [event async for event in provider.stream(_request())]
        assert "".join(e.text or "" for e in events if e.type == "text_delta") == (
            "hello"
        )
        completed = [e.response for e in events if e.type == "completed"]
        assert completed and completed[0] is not None
        assert completed[0].usage is not None
        assert completed[0].usage.output_tokens == 2

        assert [entry.id for entry in await provider.list_models()] == [MODEL]
    finally:
        await provider.close()

    first, _, stream_body = _bodies(recorded, "/v1/chat/completions")
    assert first["reasoning_effort"] == "low"
    assert first["max_completion_tokens"] == 64
    assert "extra_body" not in first
    assert stream_body["stream_options"] == {"include_usage": True}


async def test_anthropic_sdk_complete_stream_validate_and_count(wire: Any) -> None:
    pytest.importorskip("anthropic")
    base_url, recorded = wire
    settings = AnthropicProviderSettings(
        base_url=base_url,
        api_key="test-key",  # pragma: allowlist secret
        default_model=MODEL,
    )
    assert settings.validate_connection(check_default_model=True) == (MODEL,)

    provider = AnthropicModelProvider(settings)
    try:
        # thinking / output_config postdate the oldest supported SDKs.
        response = await provider.complete(
            _request(
                max_output_tokens=64,
                reasoning=ReasoningConfig(
                    effort="low", thinking="enabled", budget_tokens=1024
                ),
            )
        )
        assert response.message.content[0].text == "hello from claude"
        assert response.usage is not None and response.usage.output_tokens == 3

        tool_response = await provider.complete(_request(tools=[WEATHER_TOOL]))
        assert tool_response.message.tool_calls[0].name == "get_weather"
        assert tool_response.message.tool_calls[0].arguments == {"city": "Paris"}

        events = [event async for event in provider.stream(_request())]
        assert "".join(e.text or "" for e in events if e.type == "text_delta") == (
            "hello"
        )

        assert await provider.count_tokens(_request()) == 42
        assert [entry.id for entry in await provider.list_models()] == [MODEL]
    finally:
        close = getattr(provider.client, "close", None)
        if close is not None:
            await close()

    first = _bodies(recorded, "/v1/messages")[0]
    assert first["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert first["output_config"] == {"effort": "low"}
    assert "extra_body" not in first
    assert _bodies(recorded, "/v1/messages/count_tokens")[0]["model"] == MODEL
