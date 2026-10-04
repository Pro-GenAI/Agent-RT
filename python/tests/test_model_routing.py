import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent_rt import (
    MODEL_PROVIDER_ENV,
    AnthropicModelProvider,
    AnthropicProviderSettings,
    ContentPart,
    EmbeddingRequest,
    FirstMatchRoutingPolicy,
    ModelCatalogEntry,
    ModelDescriptor,
    ModelMessage,
    ModelRegistry,
    ModelRequest,
    ModelSettings,
    ModelTarget,
    OpenAIModelProvider,
    OpenAIProviderSettings,
    RateLimitExceededError,
    ReasoningConfig,
    RoutingRequirements,
    StructuredOutputRequirement,
    ToolDefinition,
    load_model,
)


class _AsyncSequence:
    def __init__(self, values):
        self._values = iter(values)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _model_client_factory(model_ids, calls=None, error=None):
    def factory(**kwargs):
        if calls is not None:
            calls.append(kwargs)

        def list_models():
            if error is not None:
                raise error
            return SimpleNamespace(
                data=[SimpleNamespace(id=model_id) for model_id in model_ids]
            )

        return SimpleNamespace(models=SimpleNamespace(list=list_models))

    return factory


class _FakeOpenAIAsyncClient:
    def __init__(self, response, stream, embedding_response=None, models=()):
        self.calls = []
        self.embedding_calls = []
        self.model_calls = []
        self.batch_calls = []
        self._models = tuple(models)

        async def create(**kwargs):
            self.calls.append(kwargs)
            return _AsyncSequence(stream) if kwargs.get("stream") else response

        async def create_embedding(**kwargs):
            self.embedding_calls.append(kwargs)
            return embedding_response

        async def list_models():
            self.model_calls.append(("list", None))
            return SimpleNamespace(data=list(self._models))

        async def retrieve_model(model):
            self.model_calls.append(("retrieve", model))
            for item in self._models:
                if item["id"] == model:
                    return item
            return {"id": model}

        async def create_batch(**kwargs):
            self.batch_calls.append(("create", kwargs))
            return {"id": "batch-openai", "status": "validating", "object": "batch"}

        async def retrieve_batch(batch_id):
            self.batch_calls.append(("retrieve", batch_id))
            return {"id": batch_id, "status": "completed", "object": "batch"}

        async def list_batches(**kwargs):
            self.batch_calls.append(("list", kwargs))
            return SimpleNamespace(
                data=[
                    {"id": "batch-openai", "status": "completed", "object": "batch"},
                    {
                        "id": "batch-openai-2",
                        "status": "in_progress",
                        "object": "batch",
                    },
                ]
            )

        async def cancel_batch(batch_id):
            self.batch_calls.append(("cancel", batch_id))
            return {"id": batch_id, "status": "cancelling", "object": "batch"}

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.embeddings = SimpleNamespace(create=create_embedding)
        self.models = SimpleNamespace(list=list_models, retrieve=retrieve_model)
        self.batches = SimpleNamespace(
            create=create_batch,
            retrieve=retrieve_batch,
            list=list_batches,
            cancel=cancel_batch,
        )


class _FakeAnthropicAsyncClient:
    def __init__(self, response, stream, token_count=11, models=()):
        self.calls = []
        self.token_count_calls = []
        self.model_calls = []
        self.batch_calls = []
        self._models = tuple(models)

        async def create(**kwargs):
            self.calls.append(kwargs)
            return _AsyncSequence(stream) if kwargs.get("stream") else response

        async def count_tokens(**kwargs):
            self.token_count_calls.append(kwargs)
            return SimpleNamespace(input_tokens=token_count)

        async def list_models():
            self.model_calls.append(("list", None))
            return SimpleNamespace(data=list(self._models))

        async def retrieve_model(model_id):
            self.model_calls.append(("retrieve", model_id))
            for item in self._models:
                if item["id"] == model_id:
                    return item
            return {"id": model_id}

        async def create_batch(**kwargs):
            self.batch_calls.append(("create", kwargs))
            return {
                "id": "msgbatch-test",
                "processing_status": "in_progress",
                "type": "message_batch",
            }

        async def retrieve_batch(message_batch_id):
            self.batch_calls.append(("retrieve", message_batch_id))
            return {
                "id": message_batch_id,
                "processing_status": "ended",
                "type": "message_batch",
            }

        async def list_batches(**kwargs):
            self.batch_calls.append(("list", kwargs))
            return SimpleNamespace(
                data=[
                    {
                        "id": "msgbatch-test",
                        "processing_status": "ended",
                        "type": "message_batch",
                    }
                ]
            )

        async def cancel_batch(message_batch_id):
            self.batch_calls.append(("cancel", message_batch_id))
            return {
                "id": message_batch_id,
                "processing_status": "canceling",
                "type": "message_batch",
            }

        async def batch_results(message_batch_id):
            self.batch_calls.append(("results", message_batch_id))
            return _AsyncSequence(
                [
                    SimpleNamespace(
                        custom_id="req-1", result=SimpleNamespace(type="succeeded")
                    )
                ]
            )

        batches = SimpleNamespace(
            create=create_batch,
            retrieve=retrieve_batch,
            list=list_batches,
            cancel=cancel_batch,
            results=batch_results,
        )
        self.messages = SimpleNamespace(
            create=create, count_tokens=count_tokens, batches=batches
        )
        self.models = SimpleNamespace(list=list_models, retrieve=retrieve_model)


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0
    paths: ClassVar[list[str]] = []

    def log_message(self, format, *args):
        return

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        type(self).paths.append(self.path)
        if self.path == "/v1/models":
            body = b'{"data":[{"id":"gpt-test","object":"model","created":1700000000,"owned_by":"test"}]}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models/gpt-test":
            body = b'{"id":"gpt-test","object":"model","created":1700000000,"owned_by":"test"}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/v1/batches?"):
            body = b'{"object":"list","data":[{"id":"batch-http","status":"completed","object":"batch"}]}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/batches/batch-http":
            body = b'{"id":"batch-http","status":"completed","object":"batch"}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b'{"error":"not found"}'
        self.send_response(404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).request_count += 1
        type(self).paths.append(self.path)
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        text = ""
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )

        if "rate429" in text:
            body = b'{"error":"rate limited"}'
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("retry-after", "60")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if "fail" in text:
            body = b'{"error":"boom"}'
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if "slow" in text:
            threading.Event().wait(1.0)

        if self.path == "/v1/batches":
            body = b'{"id":"batch-http","status":"validating","object":"batch"}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/batches/batch-http/cancel":
            body = b'{"id":"batch-http","status":"cancelling","object":"batch"}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/embeddings":
            values = payload.get("input")
            if not isinstance(values, list):
                values = [values]
            body = json.dumps(
                {
                    "object": "list",
                    "model": payload.get("model"),
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [float(index), 0.25],
                        }
                        for index, _ in enumerate(values)
                    ],
                    "usage": {
                        "prompt_tokens": len(values),
                        "total_tokens": len(values),
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if payload.get("stream"):
            body = (
                b'data: {"model":"gpt-test","choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n\n'
                b'data: {"model":"gpt-test","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(
            {
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {"content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        if "limited" in text:
            self.send_header("x-ratelimit-remaining-requests", "0")
            self.send_header("x-ratelimit-reset-requests", "50ms")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


class _OpenAICompatibleServer:
    def __enter__(self):
        _OpenAICompatibleHandler.request_count = 0
        _OpenAICompatibleHandler.paths = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}/v1"
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class TestModelRouting:
    def setup_method(self):
        self.fast = ModelDescriptor(
            model="fast-v1",
            provider="acme",
            capabilities=frozenset({"text"}),
            context_window=32_000,
            cost_per_million_tokens=1.0,
            latency_ms=200,
        )
        self.smart = ModelDescriptor(
            model="smart-v2",
            provider="acme",
            capabilities=frozenset({"text", "vision"}),
            context_window=128_000,
            cost_per_million_tokens=5.0,
            latency_ms=800,
            reasoning=True,
        )
        self.registry = ModelRegistry(
            models=[self.fast, self.smart],
            aliases={"default": ModelTarget("fast-v1", "acme")},
        )

    def test_alias_resolution_honors_provider_override(self):
        assert self.registry.resolve(ModelTarget("default")) == ModelTarget(
            "fast-v1", "acme"
        )
        assert self.registry.resolve(ModelTarget("default", "other")) == ModelTarget(
            "fast-v1", "other"
        )

    def test_fallback_chain_preserves_order(self):
        settings = ModelSettings(
            model="missing",
            provider="acme",
            fallback_models=(
                ModelTarget("default"),
                ModelTarget("smart-v2", "acme"),
            ),
        )
        assert self.registry.candidates(settings) == (self.fast, self.smart)

    def test_routing_skips_candidates_that_miss_requirements(self):
        settings = ModelSettings(
            model="default",
            fallback_models=(ModelTarget("smart-v2", "acme"),),
        )
        selected = self.registry.route(
            settings,
            RoutingRequirements(
                capabilities=frozenset({"vision"}),
                min_context_window=100_000,
                reasoning=True,
            ),
            FirstMatchRoutingPolicy(),
        )
        assert selected == self.smart

    def test_routing_raises_when_no_candidate_matches(self):
        with pytest.raises(LookupError):
            self.registry.route(
                ModelSettings(model="default"),
                RoutingRequirements(max_cost_per_million_tokens=0.5),
            )

    def test_openai_provider_settings_allow_custom_values(self):
        settings = OpenAIProviderSettings(
            base_url="https://gateway.example.test/v1/",
            api_key=" x ",
            default_model=" custom-model ",
        )

        assert settings.base_url == "https://gateway.example.test/v1"
        assert settings.client_options() == {
            "base_url": "https://gateway.example.test/v1",
            "api_key": "x",
        }
        assert settings.resolve_model() == "custom-model"
        assert settings.resolve_model("request-model") == "request-model"
        assert "api_key" not in repr(settings)

    def test_openai_provider_settings_validate_empty_values(self):
        with pytest.raises(ValueError):
            OpenAIProviderSettings(base_url=" ")
        settings = OpenAIProviderSettings(api_key=" ")
        assert settings.api_key is None
        assert settings.client_options() == {"base_url": "https://api.openai.com/v1"}
        with pytest.raises(ValueError):
            OpenAIProviderSettings(default_model=" ")
        with pytest.raises(ValueError):
            OpenAIProviderSettings().resolve_model()

    def test_openai_provider_settings_from_env_allow_blank_api_key_for_custom_base_url(
        self,
    ):
        calls = []
        settings = OpenAIProviderSettings.from_env(
            environ={
                "OPENAI_BASE_URL": "https://gateway.example.test/v1",
                "OPENAI_" + "API_KEY": "   ",
            },
            client_factory=_model_client_factory(("model-a",), calls),
        )
        assert settings.api_key is None
        assert calls == [
            {"base_url": "https://gateway.example.test/v1", "api_key": "not-provided"}
        ]

    def test_openai_provider_settings_from_env_require_api_key_for_default_base_url(
        self,
    ):
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIProviderSettings.from_env(
                environ={"OPENAI_" + "API_KEY": "   "}, validate=False
            )

    def test_openai_provider_settings_from_env_validate_startup_and_model(self):
        calls = []
        settings = OpenAIProviderSettings.from_env(
            environ={
                "OPENAI_BASE_URL": "https://gateway.example.test/v1/",
                "OPENAI_" + "API_KEY": " env-key ",
                "OPENAI_MODEL": " env-model ",
            },
            client_factory=_model_client_factory(("env-model", "other"), calls),
        )
        assert settings.base_url == "https://gateway.example.test/v1"
        assert settings.resolve_model() == "env-model"
        assert settings.websocket
        assert calls == [
            {"base_url": "https://gateway.example.test/v1", "api_key": "env-key"}
        ]

    def test_openai_provider_settings_websocket_env_opt_out(self):
        settings = OpenAIProviderSettings.from_env(
            environ={
                "OPENAI_BASE_URL": "https://gateway.example.test/v1",
                "OPENAI_WEBSOCKET": "off",
            },
            validate=False,
        )
        assert not settings.websocket
        with pytest.raises(ValueError, match="boolean environment value"):
            OpenAIProviderSettings.from_env(
                environ={
                    "OPENAI_BASE_URL": "https://gateway.example.test/v1",
                    "OPENAI_WEBSOCKET": "sometimes",
                },
                validate=False,
            )

    def test_openai_provider_settings_from_env_rejects_unavailable_model(self):
        with pytest.raises(ValueError, match="is not available"):
            OpenAIProviderSettings.from_env(
                environ={
                    "OPENAI_BASE_URL": "https://gateway.example.test/v1",
                    "OPENAI_MODEL": "missing-model",
                },
                client_factory=_model_client_factory(("other-model",)),
            )

    def test_openai_provider_settings_validate_base_url_before_startup_sdk_call(self):
        calls = []
        with pytest.raises(ValueError, match="absolute http"):
            OpenAIProviderSettings.from_env(
                environ={"OPENAI_BASE_URL": "not-a-url"},
                client_factory=_model_client_factory((), calls),
            )
        assert calls == []

    def test_openai_provider_settings_startup_rejects_sdk_failure(self):
        with pytest.raises(ValueError, match="401"):
            OpenAIProviderSettings.from_env(
                environ={"OPENAI_" + "API_KEY": "bad-key"},
                client_factory=_model_client_factory(
                    (), error=RuntimeError("HTTP 401")
                ),
            )

    def test_anthropic_provider_settings_from_env_validate_startup_and_model(self):
        calls = []
        settings = AnthropicProviderSettings.from_env(
            environ={
                "ANTHROPIC_BASE_URL": "https://anthropic.example.test/",
                "ANTHROPIC_" + "API_KEY": " a-key ",
                "ANTHROPIC_MODEL": " claude-test ",
            },
            client_factory=_model_client_factory(("claude-test",), calls),
        )
        assert settings.base_url == "https://anthropic.example.test"
        assert settings.resolve_model() == "claude-test"
        assert calls[0]["api_key"] == "a-key"

    def test_environment_provider_selection_uses_base_urls_then_openai_model_fallback(
        self,
    ):
        openai = load_model(
            environ={
                "OPENAI_BASE_URL": "https://openai.example.test/v1",
                "ANTHROPIC_BASE_URL": "https://anthropic.example.test",
            },
            validate=False,
            openai_async_client=object(),
        )
        assert isinstance(openai, OpenAIModelProvider)

        anthropic = load_model(
            environ={"ANTHROPIC_BASE_URL": "https://anthropic.example.test"},
            validate=False,
            anthropic_async_client=object(),
        )
        assert isinstance(anthropic, AnthropicModelProvider)

        openai_from_model = load_model(
            environ={"OPENAI_MODEL": "gpt-test", "OPENAI_" + "API_KEY": "test-key"},
            validate=False,
            openai_async_client=object(),
        )
        assert isinstance(openai_from_model, OpenAIModelProvider)

        invalid_environments = (
            {},
            {"OPENAI_" + "API_KEY": "key"},
            {"ANTHROPIC_" + "API_KEY": "key"},
            {"ANTHROPIC_MODEL": "claude-test"},
        )
        for environment in invalid_environments:
            with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
                load_model(environ=environment, validate=False)

    def test_model_provider_override_wins_over_auto_detection(self):
        anthropic = load_model(
            environ={
                MODEL_PROVIDER_ENV: " anthropic ",
                "OPENAI_BASE_URL": "https://openai.example.test/v1",
                "ANTHROPIC_BASE_URL": "https://anthropic.example.test",
            },
            validate=False,
            anthropic_async_client=object(),
        )
        assert isinstance(anthropic, AnthropicModelProvider)

        openai = load_model(
            environ={
                MODEL_PROVIDER_ENV: "OPENAI",
                "ANTHROPIC_BASE_URL": "https://anthropic.example.test",
                "OPENAI_" + "API_KEY": "test-key",
            },
            validate=False,
            openai_async_client=object(),
        )
        assert isinstance(openai, OpenAIModelProvider)

        with pytest.raises(ValueError, match="MODEL_PROVIDER"):
            load_model(environ={MODEL_PROVIDER_ENV: "unknown"}, validate=False)


class TestProviderResponse:
    def request(self, model=None, text="hello"):
        return ModelRequest(
            messages=(
                ModelMessage(
                    role="user", content=(ContentPart(type="text", text=text),)
                ),
            ),
            model=model,
        )

    async def test_openai_embeddings_response(self):
        embedding_response = {
            "object": "list",
            "model": "embed-test",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
        client = _FakeOpenAIAsyncClient({}, [], embedding_response=embedding_response)
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )

        result = await provider.embed(
            EmbeddingRequest(
                model="embed-test",
                input=["alpha", "beta"],
                dimensions=2,
                encoding_format="float",
            )
        )

        assert result.model == "embed-test"
        assert result.data[1].embedding == (0.3, 0.4)
        assert result.usage.total_tokens == 6
        assert client.embedding_calls[0] == {
            "model": "embed-test",
            "input": ["alpha", "beta"],
            "dimensions": 2,
            "encoding_format": "float",
        }

    async def test_openai_batch_api(self):
        client = _FakeOpenAIAsyncClient({}, [])
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )

        created = await provider.create_batch(
            input_file_id="file-input",
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"purpose": "test"},
        )
        retrieved = await provider.retrieve_batch(created.id)
        listed = await provider.list_batches(after="batch-0", limit=2)
        cancelled = await provider.cancel_batch(created.id)

        assert created.id == "batch-openai"
        assert created.status == "validating"
        assert retrieved.status == "completed"
        assert [item.id for item in listed] == ["batch-openai", "batch-openai-2"]
        assert cancelled.status == "cancelling"
        assert client.batch_calls[0][0] == "create"
        assert client.batch_calls[0][1]["input_file_id"] == "file-input"
        assert client.batch_calls[2] == ("list", {"after": "batch-0", "limit": 2})

    async def test_openai_models_api(self):
        client = _FakeOpenAIAsyncClient(
            {},
            [],
            models=(
                {
                    "id": "gpt-test",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "openai",
                },
            ),
        )
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )

        models = await provider.list_models()
        model = await provider.retrieve_model("gpt-test")

        assert models == (
            ModelCatalogEntry(
                id="gpt-test",
                created_at=models[0].created_at,
                object="model",
                owned_by="openai",
                raw=client._models[0],
            ),
        )
        assert model.id == "gpt-test"
        assert model.owned_by == "openai"
        assert client.model_calls == [("list", None), ("retrieve", "gpt-test")]

    async def test_openai_rate_limit_fails_fast_until_reset(self):
        client = _FakeOpenAIAsyncClient({}, [])
        calls = 0

        async def create(**kwargs):
            nonlocal calls
            calls += 1
            error = RuntimeError("rate limited")
            error.status_code = 429
            error.response = SimpleNamespace(headers={"retry-after": "60"}, status_code=429)
            raise error

        client.chat.completions.create = create
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )

        with pytest.raises(RateLimitExceededError) as first:
            await provider.complete(self.request())
        with pytest.raises(RateLimitExceededError) as second:
            await provider.complete(self.request())

        assert calls == 1
        assert first.value.retry_after_seconds > 50
        assert second.value.retry_after_seconds > 50

    async def test_openai_stream_rate_limit_sets_same_model_cooldown(self):
        client = _FakeOpenAIAsyncClient({}, [])
        calls = 0

        async def create(**kwargs):
            nonlocal calls
            calls += 1
            error = RuntimeError("rate limited")
            error.status_code = 429
            error.response = SimpleNamespace(
                headers={"retry-after": "60"}, status_code=429
            )
            raise error

        client.chat.completions.create = create
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )

        with pytest.raises(RateLimitExceededError):
            async for _ in provider.stream(self.request()):
                pass
        with pytest.raises(RateLimitExceededError):
            await provider.complete(self.request())
        assert calls == 1

    async def test_openai_non_streaming_response(self):
        response = {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        client = _FakeOpenAIAsyncClient(response, [])
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )
        result = await provider.complete(self.request())
        assert result.message.content[0].text == "done"
        assert result.message.tool_calls[0].arguments == {"q": "x"}
        assert result.finish_reason == "tool_calls"
        assert result.usage.total_tokens == 7
        assert "stream" not in client.calls[0]

    async def test_openai_reasoning_effort_request(self):
        response = {
            "model": "gpt-test",
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        }
        client = _FakeOpenAIAsyncClient(response, [])
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )
        request = ModelRequest(
            messages=self.request().messages,
            reasoning=ReasoningConfig(effort="high", summary="auto"),
        )

        await provider.complete(request)

        assert client.calls[0]["reasoning_effort"] == "high"
        assert "reasoning" not in client.calls[0]

    async def test_openai_caches_tool_and_structured_conversion_and_invalidates(self):
        response = {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ],
        }
        client = _FakeOpenAIAsyncClient(response, [])
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"),
            client=client,
        )
        first_tool = ToolDefinition(
            name="lookup",
            description="Lookup",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        first_output = StructuredOutputRequirement(
            name="result",
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        first_request = ModelRequest(
            messages=self.request().messages,
            tools=(first_tool,),
            structured_output=first_output,
        )
        await provider.complete(first_request)
        await provider.complete(first_request)
        assert client.calls[0]["messages"] is not client.calls[1]["messages"]
        assert client.calls[0]["messages"][0] is client.calls[1]["messages"][0]
        assert client.calls[0]["tools"] is client.calls[1]["tools"]
        assert client.calls[0]["response_format"] is client.calls[1]["response_format"]

        second_tool = ToolDefinition(
            name="lookup",
            description="Lookup v2",
            input_schema={"type": "object", "properties": {"id": {"type": "integer"}}},
        )
        second_output = StructuredOutputRequirement(
            name="result-v2",
            schema={"type": "object", "properties": {"value": {"type": "number"}}},
        )
        second_request = ModelRequest(
            messages=self.request().messages,
            tools=(second_tool,),
            structured_output=second_output,
        )
        await provider.complete(second_request)
        assert client.calls[1]["tools"] is not client.calls[2]["tools"]
        assert (
            client.calls[1]["response_format"] is not client.calls[2]["response_format"]
        )
        assert client.calls[2]["tools"][0]["function"]["description"] == "Lookup v2"
        assert client.calls[2]["response_format"]["json_schema"]["name"] == "result-v2"

    async def test_provider_message_prefix_cache_reuses_invalidates_and_bounds(self):
        messages = tuple(
            ModelMessage(
                role="user",
                content=(ContentPart(type="text", text=str(index)),),
            )
            for index in range(16)
        )

        openai = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"),
            client=object(),
        )
        first = openai._cached_messages(messages)
        appended_message = ModelMessage(
            role="assistant",
            content=(ContentPart(type="text", text="appended"),),
        )
        appended = openai._cached_messages(messages + (appended_message,))
        assert first[0] is appended[0]
        assert first[-1] is appended[-2]

        replacement = ModelMessage(
            role="user",
            content=(ContentPart(type="text", text="replacement"),),
        )
        changed_messages = (messages[0], replacement, *messages[2:], appended_message)
        changed = openai._cached_messages(changed_messages)
        assert appended[0] is changed[0]
        assert appended[1] is not changed[1]
        assert appended[2] is changed[2]

        oversized = tuple(
            ModelMessage(
                role="user",
                content=(ContentPart(type="text", text=f"large-{index}"),),
            )
            for index in range(1025)
        )
        openai._cached_messages(oversized)
        assert openai._message_prefix_messages == ()
        assert openai._message_prefix_payload == []

        anthropic = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"),
            client=object(),
        )
        anthropic_first = anthropic._cached_messages(messages)
        anthropic_appended = anthropic._cached_messages(messages + (appended_message,))
        assert anthropic_first[0] is anthropic_appended[0]
        anthropic._cached_messages(oversized)
        assert anthropic._message_prefix_messages == ()
        assert anthropic._message_prefix_payload == []

    async def test_openai_streaming_response(self):
        chunks = [
            {
                "model": "gpt-test",
                "choices": [{"delta": {"content": "hel"}, "finish_reason": None}],
            },
            {
                "model": "gpt-test",
                "choices": [
                    {
                        "delta": {
                            "content": "lo",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "gpt-test",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"x"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        client = _FakeOpenAIAsyncClient({}, chunks)
        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"), client=client
        )
        events = [event async for event in provider.stream(self.request())]
        assert [event.type for event in events] == [
            "text_delta",
            "text_delta",
            "tool_call_delta",
            "tool_call_delta",
            "completed",
        ]
        final = events[-1].response
        assert final.message.content[0].text == "hello"
        assert final.message.tool_calls[0].arguments == {"q": "x"}
        assert final.finish_reason == "tool_calls"
        assert client.calls[0]["stream"]

    async def test_openai_lazy_factory_reuses_client_and_closes_owned_client(self):
        response = {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        created = []

        def factory(**_kwargs):
            client = _FakeOpenAIAsyncClient(response, [])
            client.close_calls = 0

            async def close():
                client.close_calls += 1

            client.close = close
            created.append(client)
            return client

        provider = OpenAIModelProvider(
            OpenAIProviderSettings(default_model="gpt-test"),
            client_factory=factory,
        )
        first, second = await asyncio.gather(
            provider.complete(self.request()),
            provider.complete(self.request()),
        )
        assert first.message.content[0].text == "done"
        assert second.message.content[0].text == "done"
        assert len(created) == 1
        assert len(created[0].calls) == 2

        await provider.close()
        assert created[0].close_calls == 1

        await provider.complete(self.request())
        assert len(created) == 2

    def test_openai_compatible_proxy_selection_respects_environment(self, monkeypatch):
        import agent_rt as art

        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://secure-proxy.example:8443")
        monkeypatch.setenv("NO_PROXY", "api.internal.example")
        assert (
            art._compatible_proxy_url("http://external.example/v1")
            == "http://proxy.example:8080"
        )
        assert (
            art._compatible_proxy_url("https://external.example/v1")
            == "http://secure-proxy.example:8443"
        )
        assert art._compatible_proxy_url("https://api.internal.example/v1") is None

    def test_openai_compatible_proxy_selection_supports_all_proxy(self, monkeypatch):
        import agent_rt as art

        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ALL_PROXY", "proxy.example:9000")
        monkeypatch.setenv("NO_PROXY", "")
        assert (
            art._compatible_proxy_url("https://external.example/v1")
            == "http://proxy.example:9000"
        )

    async def test_openai_compatible_transport_completion_streaming_and_reuse(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            first_client = provider.client
            result = await provider.complete(self.request())
            assert result.message.content[0].text == "done"
            assert result.finish_reason == "stop"
            assert result.usage.total_tokens == 2
            events = [event async for event in provider.stream(self.request())]
            text = "".join(
                event.text or "" for event in events if event.type == "text_delta"
            )
            assert text == "hello"
            assert provider.client is first_client
            assert _OpenAICompatibleHandler.request_count == 2
            await provider.close()

    async def test_openai_compatible_transport_embeddings(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            try:
                result = await provider.embed(
                    EmbeddingRequest(model="embed-test", input=["a", "b"])
                )
            finally:
                await provider.close()

            assert result.model == "embed-test"
            assert result.data[1].embedding == (1.0, 0.25)
            assert result.usage.total_tokens == 2
            assert "/v1/embeddings" in _OpenAICompatibleHandler.paths

    async def test_openai_compatible_transport_batch_api(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            try:
                created = await provider.create_batch(
                    input_file_id="file-input",
                    endpoint="/v1/responses",
                    completion_window="24h",
                )
                retrieved = await provider.retrieve_batch(created.id)
                listed = await provider.list_batches(after="batch-0", limit=1)
                cancelled = await provider.cancel_batch(created.id)
            finally:
                await provider.close()

            assert created.id == "batch-http"
            assert created.status == "validating"
            assert retrieved.status == "completed"
            assert listed[0].id == "batch-http"
            assert cancelled.status == "cancelling"
            assert "/v1/batches" in _OpenAICompatibleHandler.paths
            assert "/v1/batches/batch-http" in _OpenAICompatibleHandler.paths
            assert any(
                path.startswith("/v1/batches?after=batch-0&limit=1")
                for path in _OpenAICompatibleHandler.paths
            )
            assert "/v1/batches/batch-http/cancel" in _OpenAICompatibleHandler.paths

    async def test_openai_compatible_transport_models_api(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            try:
                models = await provider.list_models()
                model = await provider.retrieve_model("gpt-test")
            finally:
                await provider.close()

            assert models[0].id == "gpt-test"
            assert models[0].owned_by == "test"
            assert model.id == "gpt-test"
            assert "/v1/models" in _OpenAICompatibleHandler.paths
            assert "/v1/models/gpt-test" in _OpenAICompatibleHandler.paths

    async def test_openai_compatible_transport_falls_back_when_websocket_is_unavailable(
        self,
    ):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                )
            )
            try:
                first = await provider.complete(self.request())
                second = await provider.complete(self.request())
            finally:
                await provider.close()

            assert first.message.content[0].text == "done"
            assert second.message.content[0].text == "done"
            assert _OpenAICompatibleHandler.paths.count("/v1/responses") == 1
            assert _OpenAICompatibleHandler.paths.count("/v1/chat/completions") == 2

    async def test_openai_model_validation_caches_http_fallback_after_models_call(self):
        with _OpenAICompatibleServer() as server:
            settings = OpenAIProviderSettings(
                base_url=server.base_url,
                default_model="gpt-test",
                transport="compatible",
            )
            validation_calls = []
            assert settings.validate_connection(
                check_default_model=True,
                client_factory=_model_client_factory(["gpt-test"], validation_calls),
                websocket_probe=lambda **_: False,
            ) == ("gpt-test",)
            assert len(validation_calls) == 1
            probe_count = _OpenAICompatibleHandler.paths.count("/v1/responses")
            provider = OpenAIModelProvider(settings)
            try:
                result = await provider.complete(self.request())
            finally:
                await provider.close()

            assert result.message.content[0].text == "done"
            assert validation_calls[0]["base_url"] == server.base_url
            assert probe_count == 0
            assert _OpenAICompatibleHandler.paths.count("/v1/responses") == 0
            assert "/v1/chat/completions" in _OpenAICompatibleHandler.paths

    async def test_openai_rate_limit_headers_gate_same_model_until_reset(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    websocket=False,
                )
            )
            await provider.complete(self.request(text="limited"))
            assert _OpenAICompatibleHandler.request_count == 1
            with pytest.raises(RateLimitExceededError):
                await provider.complete(self.request(text="limited again"))
            assert _OpenAICompatibleHandler.request_count == 1
            await asyncio.sleep(0.06)
            await provider.complete(self.request(text="after reset"))
            assert _OpenAICompatibleHandler.request_count == 2
            await provider.close()

    async def test_openai_compatible_429_sets_model_cooldown(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            with pytest.raises(RateLimitExceededError) as first:
                await provider.complete(self.request(text="rate429"))
            with pytest.raises(RateLimitExceededError) as second:
                await provider.complete(self.request(text="after limit"))
            assert _OpenAICompatibleHandler.request_count == 1
            assert first.value.retry_after_seconds > 50
            assert second.value.retry_after_seconds > 50
            await provider.close()

    async def test_openai_compatible_stream_429_sets_model_cooldown(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            with pytest.raises(RateLimitExceededError):
                async for _ in provider.stream(self.request(text="rate429")):
                    pass
            with pytest.raises(RateLimitExceededError):
                await provider.complete(self.request(text="after limit"))
            assert _OpenAICompatibleHandler.request_count == 1
            await provider.close()

    async def test_openai_compatible_transport_surfaces_http_failures(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            try:
                with pytest.raises(Exception, match="500"):
                    await provider.complete(self.request(text="fail"))
            finally:
                await provider.close()

    async def test_openai_compatible_transport_propagates_cancellation(self):
        with _OpenAICompatibleServer() as server:
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=server.base_url,
                    default_model="gpt-test",
                    transport="compatible",
                    websocket=False,
                )
            )
            try:
                task = asyncio.create_task(provider.complete(self.request(text="slow")))
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                await provider.close()

    async def test_anthropic_token_counting(self):
        client = _FakeAnthropicAsyncClient({}, [], token_count=23)
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        request = ModelRequest(
            messages=(
                ModelMessage(
                    role="user",
                    content=(ContentPart(type="text", text="hello"),),
                ),
            ),
            model="claude-test",
        )

        count = await provider.count_tokens(request)

        assert count == 23
        assert client.token_count_calls[0]["model"] == "claude-test"
        assert "max_tokens" not in client.token_count_calls[0]
        assert client.token_count_calls[0]["messages"][0]["role"] == "user"

    async def test_anthropic_batch_api(self):
        client = _FakeAnthropicAsyncClient({}, [])
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        requests = [
            {
                "custom_id": "req-1",
                "params": {
                    "model": "claude-test",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            }
        ]

        created = await provider.create_batch(requests=requests)
        retrieved = await provider.retrieve_batch(created.id)
        listed = await provider.list_batches(limit=1)
        cancelled = await provider.cancel_batch(created.id)
        results = await provider.batch_results(created.id)

        assert created.id == "msgbatch-test"
        assert created.status == "in_progress"
        assert retrieved.status == "ended"
        assert listed[0].id == "msgbatch-test"
        assert cancelled.status == "canceling"
        assert results[0].custom_id == "req-1"
        assert client.batch_calls[0] == ("create", {"requests": requests})
        assert client.batch_calls[-1] == ("results", "msgbatch-test")

    async def test_anthropic_models_api(self):
        client = _FakeAnthropicAsyncClient(
            {},
            [],
            models=(
                {
                    "id": "claude-test",
                    "type": "model",
                    "display_name": "Claude Test",
                    "created_at": "2026-01-01T00:00:00Z",
                },
            ),
        )
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )

        models = await provider.list_models()
        model = await provider.retrieve_model("claude-test")

        assert models[0].id == "claude-test"
        assert models[0].display_name == "Claude Test"
        assert model.id == "claude-test"
        assert client.model_calls == [("list", None), ("retrieve", "claude-test")]

    async def test_anthropic_rate_limit_fails_fast_until_reset(self):
        client = _FakeAnthropicAsyncClient({}, [])
        calls = 0

        async def create(**kwargs):
            nonlocal calls
            calls += 1
            error = RuntimeError("rate limited")
            error.status_code = 429
            error.response = SimpleNamespace(
                headers={"anthropic-ratelimit-requests-reset": "2099-01-01T00:00:00Z"},
                status_code=429,
            )
            raise error

        client.messages.create = create
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )

        with pytest.raises(RateLimitExceededError) as first:
            await provider.complete(self.request(model="claude-test"))
        with pytest.raises(RateLimitExceededError) as second:
            await provider.complete(self.request(model="claude-test"))

        assert calls == 1
        assert first.value.retry_after_seconds > 1
        assert second.value.retry_after_seconds > 1

    async def test_anthropic_non_streaming_response(self):
        response = {
            "model": "claude-test",
            "content": [
                {"type": "text", "text": "done"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "lookup",
                    "input": {"q": "x"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        client = _FakeAnthropicAsyncClient(response, [])
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        result = await provider.complete(self.request())
        assert result.message.content[0].text == "done"
        assert result.message.tool_calls[0].arguments == {"q": "x"}
        assert result.finish_reason == "tool_calls"
        assert result.usage.total_tokens == 7
        assert client.calls[0]["max_tokens"] == 4096

    async def test_anthropic_structured_output_request(self):
        response = {
            "model": "claude-test",
            "content": [{"type": "text", "text": '{"answer":"ok"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        client = _FakeAnthropicAsyncClient(response, [])
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        request = ModelRequest(
            messages=self.request().messages,
            structured_output=StructuredOutputRequirement(
                name="answer", schema=schema, strict=True
            ),
        )

        result = await provider.complete(request)

        assert result.message.content[0].text == '{"answer":"ok"}'
        assert client.calls[0]["output_config"] == {
            "format": {"type": "json_schema", "schema": schema}
        }

    async def test_anthropic_reasoning_request_merges_output_config(self):
        response = {
            "model": "claude-test",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        client = _FakeAnthropicAsyncClient(response, [])
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        request = ModelRequest(
            messages=self.request().messages,
            structured_output=StructuredOutputRequirement(schema=schema),
            reasoning=ReasoningConfig(effort="high", thinking="adaptive"),
        )

        await provider.complete(request)

        assert client.calls[0]["thinking"] == {"type": "adaptive"}
        assert client.calls[0]["output_config"] == {
            "format": {"type": "json_schema", "schema": schema},
            "effort": "high",
        }

    async def test_anthropic_legacy_thinking_budget_request(self):
        response = {
            "model": "claude-test",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        client = _FakeAnthropicAsyncClient(response, [])
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        request = ModelRequest(
            messages=self.request().messages,
            reasoning=ReasoningConfig(thinking="enabled", budget_tokens=2048),
        )

        await provider.complete(request)

        assert client.calls[0]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }

    async def test_anthropic_streaming_response(self):
        events_in = [
            {
                "type": "message_start",
                "message": {"model": "claude-test", "usage": {"input_tokens": 3}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ]
        client = _FakeAnthropicAsyncClient({}, events_in)
        provider = AnthropicModelProvider(
            AnthropicProviderSettings(default_model="claude-test"), client=client
        )
        events = [event async for event in provider.stream(self.request())]
        assert [event.type for event in events] == [
            "text_delta",
            "tool_call_delta",
            "completed",
        ]
        final = events[-1].response
        assert final.message.content[0].text == "hello"
        assert final.message.tool_calls[0].arguments == {"q": "x"}
        assert final.usage.total_tokens == 7
        assert client.calls[0]["stream"]
