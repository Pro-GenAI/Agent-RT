"""The OpenAI model provider."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from ext.runtime.models.conversion import (
    _finish_reason,
    _provider_tool_call,
    _provider_value,
)
from ext.runtime.models.interfaces import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    StructuredOutputRequirement,
)
from ext.runtime.models.openai_http import (
    _batch_job,
    _model_catalog_entry,
    _model_catalog_values,
    _OpenAICompatibleHTTPClient,
    _OpenAIRateLimitGate,
)
from ext.runtime.models.openai_payloads import (
    _openai_message,
    _openai_params,
    _openai_response,
    _openai_response_format,
    _openai_tool_params,
    _openai_usage,
    _response_headers,
)
from ext.runtime.models.provider_settings import OpenAIProviderSettings
from ext.runtime.models.sdk import (
    BatchJob,
    ModelCatalogEntry,
    _sdk_params,
    _set_ws_capability,
    _ws_capability,
)
from ext.runtime.resilience.rate_limits import RateLimitExceededError
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ContentPart, FinishReason, ModelMessage


class OpenAIModelProvider:
    name = "openai"

    def __init__(
        self,
        settings: OpenAIProviderSettings,
        *,
        client: Any = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")
        self.settings = settings
        self._client = client
        self._client_factory = client_factory
        websocket_capability = _ws_capability(settings.base_url)
        self._websocket_enabled = (
            settings.websocket
            and websocket_capability is not False
            and client is None
            and client_factory is None
        )
        self._websocket_transport: Any = None
        self._websocket_previous_response_id: str | None = None
        self._websocket_expected_prefix: tuple[ModelMessage, ...] = ()
        self._client_lock = threading.Lock()
        self._owns_client = False
        # Cache keys hold strong references so identity comparison cannot be
        # fooled by a recycled ``id()`` after the original object is freed.
        self._tool_conversion_key: tuple[ToolDefinition, ...] | None = None
        self._tool_conversion_value: list[dict[str, Any]] | None = None
        self._structured_conversion_key: StructuredOutputRequirement | None = None
        self._structured_conversion_value: dict[str, Any] | None = None
        self._message_conversion_cache: dict[
            int, tuple[ModelMessage, dict[str, Any]]
        ] = {}
        self._message_conversion_cache_limit = 512
        self._message_prefix_cache_min = 16
        self._message_prefix_cache_limit = 1024
        self._message_prefix_messages: tuple[ModelMessage, ...] = ()
        self._message_prefix_payload: list[dict[str, Any]] = []
        self._rate_limit_gate = _OpenAIRateLimitGate()

    @property
    def client(self) -> Any:
        return self._get_client()

    def _get_client(self) -> Any:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            client = self._client
            if client is None:
                factory = self._client_factory
                if factory is None:
                    if self.settings.transport == "sdk":
                        try:
                            from openai import AsyncOpenAI
                        except ModuleNotFoundError as exc:
                            if exc.name != "openai":
                                raise
                            raise RuntimeError(
                                "The OpenAI SDK is optional. Install "
                                "'agent-rt[openai]' to use "
                                "transport='sdk', or use the default "
                                "compatible transport."
                            ) from exc
                        factory = AsyncOpenAI
                    else:
                        factory = _OpenAICompatibleHTTPClient
                client = factory(
                    base_url=self.settings.base_url,
                    api_key=self.settings.api_key or "not-provided",
                )
                self._client = client
                self._owns_client = True
        return client

    async def _disable_websocket(self) -> None:
        self._websocket_enabled = False
        _set_ws_capability(self.settings.base_url, False)
        websocket_transport = self._websocket_transport
        if websocket_transport is not None:
            await websocket_transport.close()
        self._websocket_transport = None
        self._websocket_previous_response_id = None
        self._websocket_expected_prefix = ()

    async def close(self) -> None:
        websocket_transport = self._websocket_transport
        if websocket_transport is not None:
            await websocket_transport.close()
            self._websocket_transport = None
            self._websocket_previous_response_id = None
            self._websocket_expected_prefix = ()
        client = self._client
        if client is None or not self._owns_client:
            return
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        with self._client_lock:
            if self._client is client:
                self._client = None
                self._owns_client = False

    def _cached_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> list[dict[str, Any]]:
        use_prefix_cache = len(messages) >= self._message_prefix_cache_min
        cached_messages = self._message_prefix_messages if use_prefix_cache else ()
        cached_payload = self._message_prefix_payload if use_prefix_cache else []
        prefix_length = 0
        prefix_limit = min(len(messages), len(cached_messages))
        while (
            prefix_length < prefix_limit
            and messages[prefix_length] is cached_messages[prefix_length]
        ):
            prefix_length += 1

        payload = list(cached_payload[:prefix_length])
        cache = self._message_conversion_cache
        for message in messages[prefix_length:]:
            key = id(message)
            entry = cache.get(key)
            if entry is not None and entry[0] is message:
                converted = entry[1]
            else:
                converted = _openai_message(message)
                cache[key] = (message, converted)
                if len(cache) > self._message_conversion_cache_limit:
                    cache.pop(next(iter(cache)))
            payload.append(converted)

        if (
            self._message_prefix_cache_min
            <= len(messages)
            <= self._message_prefix_cache_limit
        ):
            self._message_prefix_messages = tuple(messages)
            self._message_prefix_payload = list(payload)
        else:
            self._message_prefix_messages = ()
            self._message_prefix_payload = []
        return payload

    def _cached_params(self, request: ModelRequest) -> dict[str, Any]:
        messages_payload = self._cached_messages(request.messages)
        tools_payload: list[dict[str, Any]] | None = None
        if request.tools:
            key = tuple(request.tools)
            cached_tools = self._tool_conversion_key
            if (
                cached_tools is None
                or len(cached_tools) != len(key)
                or any(left is not right for left, right in zip(cached_tools, key))
            ):
                self._tool_conversion_key = key
                self._tool_conversion_value = _openai_tool_params(request.tools)
            tools_payload = self._tool_conversion_value
        else:
            self._tool_conversion_key = None
            self._tool_conversion_value = None

        response_format: dict[str, Any] | None = None
        if request.structured_output is not None:
            if request.structured_output is not self._structured_conversion_key:
                self._structured_conversion_key = request.structured_output
                self._structured_conversion_value = _openai_response_format(
                    request.structured_output
                )
            response_format = self._structured_conversion_value
        else:
            self._structured_conversion_key = None
            self._structured_conversion_value = None

        return _openai_params(
            self.settings,
            request,
            messages_payload=messages_payload,
            tools_payload=tools_payload,
            response_format=response_format,
        )

    def _rate_limit_headers(self, client: Any, raw: Any = None) -> dict[str, str]:
        headers = _response_headers(raw) if raw is not None else {}
        completions = getattr(getattr(client, "chat", None), "completions", None)
        stored = getattr(completions, "last_response_headers", None)
        if stored:
            headers.update(
                {str(key).lower(): str(value) for key, value in dict(stored).items()}
            )
        return headers

    async def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Mapping[str, str] | None = None,
        output_expires_after: Mapping[str, Any] | None = None,
    ) -> BatchJob:
        resource = getattr(self._get_client(), "batches", None)
        create = getattr(resource, "create", None)
        if not callable(create):
            raise TypeError("OpenAI provider client does not support batches.create()")
        params: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata is not None:
            params["metadata"] = dict(metadata)
        if output_expires_after is not None:
            params["output_expires_after"] = dict(output_expires_after)
        return _batch_job(await create(**params))

    async def retrieve_batch(self, batch_id: str) -> BatchJob:
        resource = getattr(self._get_client(), "batches", None)
        retrieve = getattr(resource, "retrieve", None)
        if not callable(retrieve):
            raise TypeError(
                "OpenAI provider client does not support batches.retrieve()"
            )
        return _batch_job(await retrieve(batch_id))

    async def list_batches(
        self, *, after: str | None = None, limit: int | None = None
    ) -> tuple[BatchJob, ...]:
        resource = getattr(self._get_client(), "batches", None)
        list_batches = getattr(resource, "list", None)
        if not callable(list_batches):
            raise TypeError("OpenAI provider client does not support batches.list()")
        params: dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = limit
        page = await list_batches(**params)
        data = _provider_value(page, "data", page)
        return tuple(_batch_job(item) for item in data)

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        resource = getattr(self._get_client(), "batches", None)
        cancel = getattr(resource, "cancel", None)
        if not callable(cancel):
            raise TypeError("OpenAI provider client does not support batches.cancel()")
        return _batch_job(await cancel(batch_id))

    async def list_models(self) -> tuple[ModelCatalogEntry, ...]:
        resource = getattr(self._get_client(), "models", None)
        list_models = getattr(resource, "list", None)
        if not callable(list_models):
            raise TypeError("OpenAI provider client does not support models.list()")
        return _model_catalog_values(await list_models())

    async def retrieve_model(self, model: str) -> ModelCatalogEntry:
        resource = getattr(self._get_client(), "models", None)
        retrieve = getattr(resource, "retrieve", None)
        if not callable(retrieve):
            raise TypeError("OpenAI provider client does not support models.retrieve()")
        entry = _model_catalog_entry(await retrieve(model))
        if not entry.id:
            raise ValueError("OpenAI model response is missing id")
        return entry

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = self.settings.resolve_model(request.model)
        params: dict[str, Any] = {"model": model, "input": request.input}
        if request.dimensions is not None:
            params["dimensions"] = request.dimensions
        if request.encoding_format is not None:
            params["encoding_format"] = request.encoding_format
        resource = getattr(self._get_client(), "embeddings", None)
        create = getattr(resource, "create", None)
        if create is None:
            raise TypeError(
                "OpenAI provider client does not support embeddings.create()"
            )
        raw = await create(**params)
        data: list[EmbeddingItem] = []
        for index, item in enumerate(_provider_value(raw, "data", ()) or ()):
            value = _provider_value(item, "embedding")
            if isinstance(value, str):
                embedding: Sequence[float] | str = value
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                embedding = tuple(float(part) for part in value)
            else:
                raise ValueError(
                    "embedding response must contain a vector or base64 string"
                )
            data.append(
                EmbeddingItem(
                    index=int(_provider_value(item, "index", index)),
                    embedding=embedding,
                )
            )
        usage_value = _provider_value(raw, "usage")
        usage = None
        if usage_value is not None:
            prompt = _provider_value(usage_value, "prompt_tokens")
            total = _provider_value(usage_value, "total_tokens", prompt)
            usage = ModelUsage(input_tokens=prompt, total_tokens=total)
        return EmbeddingResponse(
            data=tuple(data),
            model=_provider_value(raw, "model", model),
            usage=usage,
            raw=raw,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        model = self.settings.resolve_model(request.model)
        self._rate_limit_gate.check(model)
        if self._websocket_enabled:
            from ext.transports.openai_ws import (
                OpenAIWebSocketUnavailableError,
                stream_model_response,
            )

            try:
                completed: ModelResponse | None = None
                async for event in stream_model_response(self, request):
                    if event.type == "completed":
                        completed = event.response
                if completed is None:
                    raise RuntimeError(
                        "OpenAI WebSocket did not produce a completed response"
                    )
                return completed
            except OpenAIWebSocketUnavailableError:
                await self._disable_websocket()
        client = self._get_client()
        completions = client.chat.completions
        try:
            raw_api = getattr(completions, "with_raw_response", None)
            if raw_api is not None:
                raw_response = await raw_api.create(
                    **_sdk_params(raw_api.create, self._cached_params(request))
                )
                headers = _response_headers(raw_response)
                parsed = raw_response.parse()
                response = await parsed if hasattr(parsed, "__await__") else parsed
                self._rate_limit_gate.update(model, headers)
            else:
                response = await completions.create(
                    **_sdk_params(completions.create, self._cached_params(request))
                )
                self._rate_limit_gate.update(
                    model, self._rate_limit_headers(client, response)
                )
        except BaseException as exc:
            headers = self._rate_limit_headers(client)
            error_response = getattr(exc, "response", None)
            headers.update(_response_headers(error_response))
            status_code = getattr(exc, "status_code", None)
            if status_code is None and error_response is not None:
                status_code = getattr(
                    error_response,
                    "status_code",
                    getattr(error_response, "status", None),
                )
            retry_after = self._rate_limit_gate.update(
                model, headers, force=status_code == 429
            )
            if status_code == 429:
                raise RateLimitExceededError(
                    "model:" + model, max(retry_after, 1.0)
                ) from exc
            raise
        return _openai_response(response)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        model_key = self.settings.resolve_model(request.model)
        self._rate_limit_gate.check(model_key)
        if self._websocket_enabled:
            from ext.transports.openai_ws import (
                OpenAIWebSocketUnavailableError,
                stream_model_response,
            )

            try:
                async for event in stream_model_response(self, request):
                    yield event
                return
            except OpenAIWebSocketUnavailableError:
                await self._disable_websocket()

        params = self._cached_params(request)
        params["stream"] = True
        # Chat Completions only reports token usage on streams when asked;
        # without it token budgets and cost ledgers would never see usage.
        params["stream_options"] = {"include_usage": True}
        client = self._get_client()
        try:
            create = client.chat.completions.create
            stream = await create(**_sdk_params(create, params))
        except BaseException as exc:
            headers = self._rate_limit_headers(client)
            error_response = getattr(exc, "response", None)
            headers.update(_response_headers(error_response))
            status_code = getattr(exc, "status_code", None)
            if status_code is None and error_response is not None:
                status_code = getattr(
                    error_response,
                    "status_code",
                    getattr(error_response, "status", None),
                )
            retry_after = self._rate_limit_gate.update(
                model_key, headers, force=status_code == 429
            )
            if status_code == 429:
                raise RateLimitExceededError(
                    "model:" + model_key, max(retry_after, 1.0)
                ) from exc
            raise
        text_parts: list[str] = []
        tool_state: dict[int, dict[str, Any]] = {}
        model: str | None = None
        finish: FinishReason | None = None
        usage: ModelUsage | None = None
        last_raw: Any = None
        try:
            async for chunk in stream:
                last_raw = chunk
                model = _provider_value(chunk, "model", model)
                chunk_usage = _openai_usage(_provider_value(chunk, "usage"))
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = _provider_value(chunk, "choices", ()) or ()
                if not choices:
                    continue
                choice = choices[0]
                reason = _provider_value(choice, "finish_reason")
                if reason is not None:
                    finish = _finish_reason(reason)
                delta = _provider_value(choice, "delta")
                text = _provider_value(delta, "content")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    yield ModelStreamEvent(type="text_delta", text=text, raw=chunk)
                for call in _provider_value(delta, "tool_calls", ()) or ():
                    index = int(_provider_value(call, "index", 0))
                    state = tool_state.get(index)
                    if state is None:
                        state = {"id": "", "name": "", "argument_parts": []}
                        tool_state[index] = state
                    call_id = _provider_value(call, "id")
                    if isinstance(call_id, str):
                        state["id"] = call_id
                    function = _provider_value(call, "function")
                    name = _provider_value(function, "name")
                    arguments_delta = _provider_value(function, "arguments")
                    if isinstance(name, str):
                        state["name"] = name
                    if isinstance(arguments_delta, str):
                        state["argument_parts"].append(arguments_delta)
                    yield ModelStreamEvent(
                        type="tool_call_delta",
                        tool_call_id=state["id"] or None,
                        tool_name=state["name"] or None,
                        arguments_delta=(
                            arguments_delta
                            if isinstance(arguments_delta, str)
                            else None
                        ),
                        raw=chunk,
                    )
        except BaseException as exc:
            headers = self._rate_limit_headers(client)
            headers.update(_response_headers(stream))
            error_response = getattr(exc, "response", None)
            headers.update(_response_headers(error_response))
            status_code = getattr(exc, "status_code", None)
            if status_code is None and error_response is not None:
                status_code = getattr(
                    error_response,
                    "status_code",
                    getattr(error_response, "status", None),
                )
            retry_after = self._rate_limit_gate.update(
                model_key, headers, force=status_code == 429
            )
            if status_code == 429:
                raise RateLimitExceededError(
                    "model:" + model_key, max(retry_after, 1.0)
                ) from exc
            raise
        tool_calls = tuple(
            _provider_tool_call(
                state["id"], state["name"], "".join(state["argument_parts"])
            )
            for _, state in sorted(tool_state.items())
        )
        response = ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(
                    (ContentPart(type="text", text="".join(text_parts)),)
                    if text_parts
                    else ()
                ),
                tool_calls=tool_calls,
            ),
            model=model,
            usage=usage,
            finish_reason=finish,
            raw=last_raw,
        )
        yield ModelStreamEvent(type="completed", response=response, raw=last_raw)
