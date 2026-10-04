import asyncio
import json

import pytest

from agent_rt import (
    AgentConfig,
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ModelUsage,
    RateLimit,
    RateLimitExceededError,
)
from agent_rt_api import AgentRTAPIServer, APIQueueOverloadedError, create_app, serve


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        text = "echo:" + "".join(
            part.text or ""
            for message in request.messages
            if message.role == "user"
            for part in message.content
            if part.type == "text"
        )
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text=text),),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="stop",
        )

    async def stream(self, request):
        response = await self.complete(request)
        text = response.message.content[0].text or ""
        midpoint = max(1, len(text) // 2)
        yield ModelStreamEvent(type="text_delta", text=text[:midpoint])
        yield ModelStreamEvent(type="text_delta", text=text[midpoint:])
        yield ModelStreamEvent(type="completed", response=response)


class BlockingProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request):
        if request.model == "provider-a":
            self.started.set()
            await self.release.wait()
        return await super().complete(request)


class RateLimitedProvider(FakeProvider):
    async def complete(self, request):
        raise RateLimitExceededError("model:provider-a", 17.0)




def make_server(provider=None):
    loop = AgentLoop(provider or FakeProvider())
    agent = AgentConfig(
        name="assistant",
        instructions="You are a test agent.",
        model=ModelSettings(model="fake-model"),
    )
    return loop, {"assistant": agent}


class TestAgentRTAPIServer:
    async def test_server_resolves_default_and_overrides_settings(self):
        loop, agents = make_server()
        server = AgentRTAPIServer(loop, agents)
        model_id, result = await server.run(
            model=None,
            messages=(
                ModelMessage(
                    role="user", content=(ContentPart(type="text", text="hi"),)
                ),
            ),
            temperature=0.25,
            max_output_tokens=64,
        )
        assert model_id == "assistant"
        assert result.final_response.message.content[0].text == "echo:hi"

    async def test_auth_header(self):
        loop, agents = make_server()
        server = AgentRTAPIServer(loop, agents, api_keys={"secret"})
        assert not server.authorize_header(None)
        assert not server.authorize_header("Bearer wrong")
        assert server.authorize_header("Bearer secret")
        assert server.authorize_header("bearer secret")

    async def test_generation_cap_clamps_client_override(self):
        provider = FakeProvider()
        loop, agents = make_server(provider)
        server = AgentRTAPIServer(loop, agents, generation_cap=8)
        await server.run(
            model=None,
            messages=(
                ModelMessage(
                    role="user", content=(ContentPart(type="text", text="hi"),)
                ),
            ),
            max_output_tokens=10_000,
        )
        assert provider.requests[-1].max_output_tokens == 8

    def test_serve_rejects_unauthenticated_non_loopback_bind(self):
        loop, agents = make_server()
        with pytest.raises(ValueError, match="unauthenticated non-loopback"):
            serve(loop, agents, host="0.0.0.0")

    async def test_execution_queues_are_bounded_and_isolated_per_model(self):
        provider = BlockingProvider()
        loop = AgentLoop(provider)
        agents = {
            "a": AgentConfig(
                name="a", instructions="test", model=ModelSettings(model="provider-a")
            ),
            "b": AgentConfig(
                name="b", instructions="test", model=ModelSettings(model="provider-b")
            ),
        }
        server = AgentRTAPIServer(
            loop,
            agents,
            max_concurrent_requests=1,
            max_queued_requests=1,
            queue_wait_timeout_seconds=1.0,
        )
        messages = (
            ModelMessage(role="user", content=(ContentPart(type="text", text="hi"),)),
        )

        first = asyncio.create_task(server.run(model="a", messages=messages))
        await provider.started.wait()
        second = asyncio.create_task(server.run(model="a", messages=messages))
        await asyncio.sleep(0)
        assert server.execution_queues["a"].queued == 1

        model_id, _ = await server.run(model="b", messages=messages)
        assert model_id == "b"
        with pytest.raises(APIQueueOverloadedError):
            await server.run(model="a", messages=messages)

        provider.release.set()
        await first
        await second
        assert server.execution_queues["a"].running == 0
        assert server.execution_queues["a"].queued == 0


class TestAgentRTAPIHTTP:
    @classmethod
    def setup_class(cls):
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            pytest.skip("agent-rt[api] dependencies are not installed")
        cls.TestClient = TestClient

    def setup_method(self):
        loop, agents = make_server()
        self.client = self.TestClient(create_app(loop, agents))

    def test_security_defaults_hide_docs(self):
        assert self.client.get("/docs").status_code == 404
        assert self.client.get("/openapi.json").status_code == 404

    def test_health_reports_per_model_queue_capacity(self):
        payload = self.client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["queues"]["assistant"]["max_running"] == 32
        assert payload["queues"]["assistant"]["max_queued"] == 128

    def test_provider_rate_limit_returns_retry_after(self):
        loop, agents = make_server(RateLimitedProvider())
        client = self.TestClient(create_app(loop, agents))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "17"
        assert response.json()["error"]["type"] == "rate_limit_error"

    def test_streaming_rate_limit_uses_protocol_error_shapes(self):
        loop, agents = make_server(RateLimitedProvider())
        client = self.TestClient(create_app(loop, agents))

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        assert '"type":"rate_limit_error"' in body
        assert '"retry_after_seconds":17' in body
        assert "model:provider-a" not in body

        with client.stream(
            "POST",
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        assert "event: error" in body
        assert '"type":"rate_limit_error"' in body
        assert '"retry_after_seconds":17' in body

    def test_request_body_size_limit(self):
        loop, agents = make_server()
        client = self.TestClient(create_app(loop, agents, max_request_body_bytes=128))
        response = client.post(
            "/v1/chat/completions",
            content=json.dumps(
                {
                    "model": "assistant",
                    "messages": [{"role": "user", "content": "x" * 256}],
                }
            ),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413

    def test_api_rate_limit_returns_429(self):
        loop, agents = make_server()
        client = self.TestClient(
            create_app(
                loop,
                agents,
                request_rate_limit=RateLimit(limit=1, window_seconds=60.0),
            )
        )
        assert client.get("/v1/models").status_code == 200
        response = client.get("/v1/models")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_websocket_origin_and_message_size_limits(self):
        from starlette.websockets import WebSocketDisconnect

        loop, agents = make_server()
        client = self.TestClient(
            create_app(
                loop,
                agents,
                allowed_websocket_origins={"https://allowed.example"},
                max_websocket_message_bytes=128,
            )
        )

        with pytest.raises(
            WebSocketDisconnect
        ) as origin_error, client.websocket_connect(
            "/v1/responses",
            headers={"origin": "https://blocked.example"},
        ):
            pass
        assert origin_error.value.code == 4403

        with client.websocket_connect(
            "/v1/responses",
            headers={"origin": "https://allowed.example"},
        ) as websocket:
            websocket.send_json(
                {
                    "type": "response.create",
                    "model": "assistant",
                    "input": "x" * 256,
                }
            )
            with pytest.raises(WebSocketDisconnect) as size_error:
                websocket.receive_json()
        assert size_error.value.code == 1009

    def test_models_and_chat_completion(self):
        models = self.client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "assistant"

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "chat.completion"
        assert payload["model"] == "assistant"
        assert payload["choices"][0]["message"]["content"] == "echo:hello"
        assert payload["usage"]["total_tokens"] == 5

    def test_chat_sse_stream(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "stream"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        assert response.status_code == 200
        assert '"object":"chat.completion.chunk"' in body
        assert "data: [DONE]" in body

    def test_responses_http_and_sse(self):
        response = self.client.post(
            "/v1/responses",
            json={"model": "assistant", "input": "hello"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "response"
        assert payload["output"][0]["content"][0]["text"] == "echo:hello"

        with self.client.stream(
            "POST",
            "/v1/responses",
            json={"model": "assistant", "input": "stream", "stream": True},
        ) as streamed:
            body = "".join(streamed.iter_text())
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "event: response.completed" in body

    def test_anthropic_messages_http(self):
        response = self.client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "system": "Be concise.",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["type"] == "message"
        assert payload["role"] == "assistant"
        assert payload["model"] == "assistant"
        assert payload["content"] == [{"type": "text", "text": "echo:hello"}]
        assert payload["stop_reason"] == "end_turn"
        assert payload["usage"] == {"input_tokens": 3, "output_tokens": 2}

    def test_anthropic_messages_sse_stream(self):
        with self.client.stream(
            "POST",
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "stream"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        assert response.status_code == 200
        assert "event: message_start" in body
        assert '"type":"content_block_start"' in body
        assert '"type":"text_delta"' in body
        assert "event: message_delta" in body
        assert '"stop_reason":"end_turn"' in body
        assert "event: message_stop" in body

    def test_anthropic_models_shape(self):
        response = self.client.get(
            "/v1/models",
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["data"][0]["id"] == "assistant"
        assert payload["data"][0]["type"] == "model"
        assert not payload["has_more"]

    def test_anthropic_x_api_key_auth_and_error_shape(self):
        loop, agents = make_server()
        client = self.TestClient(create_app(loop, agents, api_keys={"secret"}))
        response = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 401
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "authentication_error"

        response = client.post(
            "/v1/messages",
            headers={
                "anthropic-version": "2023-06-01",
                "x-api-key": "secret",
            },
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200

    def test_anthropic_tool_history_and_request_tool_boundary(self):
        response = self.client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "lookup",
                                "input": {"q": "x"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "result",
                            },
                            {"type": "text", "text": "continue"},
                        ],
                    },
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "echo:continue"

        unsupported = self.client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "assistant",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "input_schema": {"type": "object"},
                    }
                ],
            },
        )
        assert unsupported.status_code == 400
        assert unsupported.json()["type"] == "error"
        assert "configured Agent RT runtime" in unsupported.json()["error"]["message"]

    def test_responses_websocket(self):
        with self.client.websocket_connect("/v1/responses") as websocket:
            websocket.send_json(
                {
                    "type": "response.create",
                    "stream_id": "request-1",
                    "model": "assistant",
                    "input": "ws",
                }
            )
            events = []
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event["type"] == "response.completed":
                    break
        assert all(event["stream_id"] == "request-1" for event in events)
        assert events[0]["type"] == "response.created"
        assert events[-1]["response"]["status"] == "completed"

    def test_bearer_auth(self):
        loop, agents = make_server()
        client = self.TestClient(create_app(loop, agents, api_keys={"secret"}))
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200

    def test_unknown_model_uses_openai_error_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 404
        assert "error" in response.json()
