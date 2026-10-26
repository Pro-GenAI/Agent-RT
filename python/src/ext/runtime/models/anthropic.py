"""The Anthropic model provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ext.runtime.models.anthropic_payloads import (
    _anthropic_message,
    _anthropic_params,
    _anthropic_response,
    _thinking_part,
)
from ext.runtime.models.conversion import (
    _finish_reason,
    _provider_tool_call,
    _provider_value,
)
from ext.runtime.models.interfaces import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from ext.runtime.models.openai_http import (
    _batch_job,
    _model_catalog_entry,
    _model_catalog_values,
    _OpenAIRateLimitGate,
)
from ext.runtime.models.openai_payloads import _response_headers
from ext.runtime.models.provider_settings import AnthropicProviderSettings
from ext.runtime.models.sdk import (
    BatchJob,
    ModelCatalogEntry,
    _sdk_list_models,
    _sdk_params,
)
from ext.runtime.resilience.rate_limits import RateLimitExceededError
from ext.runtime.types import ContentPart, FinishReason, ModelMessage


class AnthropicModelProvider:
    name = "anthropic"

    def __init__(
        self, settings: AnthropicProviderSettings, *, client: Any = None
    ) -> None:
        self.settings = settings
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ModuleNotFoundError as exc:
                if exc.name != "anthropic":
                    raise
                raise RuntimeError(
                    "The Anthropic SDK is optional. Install "
                    "'agent-rt[anthropic]' to use "
                    "AnthropicModelProvider without an injected client."
                ) from exc
            client = AsyncAnthropic(
                base_url=settings.base_url,
                api_key=settings.api_key or "not-provided",
            )
        self.client = client
        self._rate_limit_gate = _OpenAIRateLimitGate()
        self._message_conversion_cache: dict[
            int,
            tuple[
                ModelMessage,
                tuple[str | None, dict[str, Any] | None],
            ],
        ] = {}
        self._message_conversion_cache_limit = 512
        self._message_prefix_cache_min = 16
        self._message_prefix_cache_limit = 1024
        self._message_prefix_messages: tuple[ModelMessage, ...] = ()
        self._message_prefix_payload: list[tuple[str | None, dict[str, Any] | None]] = (
            []
        )

    def _cached_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> list[tuple[str | None, dict[str, Any] | None]]:
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

        converted = list(cached_payload[:prefix_length])
        cache = self._message_conversion_cache
        for message in messages[prefix_length:]:
            key = id(message)
            entry = cache.get(key)
            if entry is not None and entry[0] is message:
                value = entry[1]
            else:
                value = _anthropic_message(message)
                cache[key] = (message, value)
                if len(cache) > self._message_conversion_cache_limit:
                    cache.pop(next(iter(cache)))
            converted.append(value)

        if (
            self._message_prefix_cache_min
            <= len(messages)
            <= self._message_prefix_cache_limit
        ):
            self._message_prefix_messages = tuple(messages)
            self._message_prefix_payload = list(converted)
        else:
            self._message_prefix_messages = ()
            self._message_prefix_payload = []
        return converted

    def _cached_params(self, request: ModelRequest) -> dict[str, Any]:
        converted = self._cached_messages(request.messages)
        return _anthropic_params(
            self.settings,
            request,
            converted_messages=converted,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        model = self.settings.resolve_model(request.model)
        self._rate_limit_gate.check(model)
        try:
            create = self.client.messages.create
            response = await create(**_sdk_params(create, self._cached_params(request)))
        except BaseException as exc:
            error_response = getattr(exc, "response", None)
            headers = _response_headers(error_response)
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
        return _anthropic_response(response)

    def _message_batches_resource(self) -> Any:
        messages = getattr(self.client, "messages", None)
        resource = getattr(messages, "batches", None)
        if resource is None:
            # Anthropic SDKs 0.37-0.39 expose batches only under the beta
            # namespace; the request and response shapes are the same.
            beta_messages = getattr(
                getattr(self.client, "beta", None), "messages", None
            )
            resource = getattr(beta_messages, "batches", None)
        if resource is None:
            raise TypeError(
                "Anthropic provider client does not support messages.batches"
            )
        return resource

    async def create_batch(
        self,
        *,
        requests: Sequence[Mapping[str, Any]],
        user_profile_id: str | None = None,
    ) -> BatchJob:
        create = getattr(self._message_batches_resource(), "create", None)
        if not callable(create):
            raise TypeError(
                "Anthropic provider client does not support messages.batches.create()"
            )
        params: dict[str, Any] = {"requests": list(requests)}
        if user_profile_id is not None:
            params["user_profile_id"] = user_profile_id
        return _batch_job(await create(**params), anthropic=True)

    async def retrieve_batch(self, batch_id: str) -> BatchJob:
        retrieve = getattr(self._message_batches_resource(), "retrieve", None)
        if not callable(retrieve):
            raise TypeError(
                "Anthropic provider client does not support messages.batches.retrieve()"
            )
        return _batch_job(await retrieve(batch_id), anthropic=True)

    async def list_batches(
        self,
        *,
        after_id: str | None = None,
        before_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[BatchJob, ...]:
        list_batches = getattr(self._message_batches_resource(), "list", None)
        if not callable(list_batches):
            raise TypeError(
                "Anthropic provider client does not support messages.batches.list()"
            )
        params: dict[str, Any] = {}
        if after_id is not None:
            params["after_id"] = after_id
        if before_id is not None:
            params["before_id"] = before_id
        if limit is not None:
            params["limit"] = limit
        page = await list_batches(**params)
        data = _provider_value(page, "data", page)
        return tuple(_batch_job(item, anthropic=True) for item in data)

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        cancel = getattr(self._message_batches_resource(), "cancel", None)
        if not callable(cancel):
            raise TypeError(
                "Anthropic provider client does not support messages.batches.cancel()"
            )
        return _batch_job(await cancel(batch_id), anthropic=True)

    async def batch_results(self, batch_id: str) -> tuple[Any, ...]:
        results = getattr(self._message_batches_resource(), "results", None)
        if not callable(results):
            raise TypeError(
                "Anthropic provider client does not support messages.batches.results()"
            )
        decoder = await results(batch_id)
        return tuple([item async for item in decoder])

    async def list_models(self) -> tuple[ModelCatalogEntry, ...]:
        try:
            page = _sdk_list_models(self.client)
        except TypeError as exc:
            raise TypeError(
                "Anthropic provider client does not support models.list()"
            ) from exc
        return _model_catalog_values(await page)

    async def retrieve_model(self, model: str) -> ModelCatalogEntry:
        resource = getattr(self.client, "models", None)
        retrieve = getattr(resource, "retrieve", None)
        if not callable(retrieve):
            raise TypeError(
                "Anthropic provider client does not support models.retrieve()"
            )
        entry = _model_catalog_entry(await retrieve(model))
        if not entry.id:
            raise ValueError("Anthropic model response is missing id")
        return entry

    async def count_tokens(self, request: ModelRequest) -> int:
        params = self._cached_params(request)
        params.pop("max_tokens", None)
        params.pop("temperature", None)
        params.pop("output_config", None)
        count_tokens = getattr(self.client.messages, "count_tokens", None)
        if count_tokens is not None:
            response = await count_tokens(**_sdk_params(count_tokens, params))
        else:
            # Anthropic SDKs before 0.47 lack messages.count_tokens(); the
            # generic request helper reaches the same endpoint.
            raw_post = getattr(self.client, "post", None)
            if not callable(raw_post):
                raise TypeError(
                    "Anthropic provider client does not support "
                    "messages.count_tokens()"
                )
            response = await raw_post(
                "/v1/messages/count_tokens", cast_to=object, body=params
            )
        value = _provider_value(response, "input_tokens")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                "Anthropic token count response must include integer input_tokens"
            )
        return value

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        params = self._cached_params(request)
        model_key = self.settings.resolve_model(request.model)
        self._rate_limit_gate.check(model_key)
        params["stream"] = True
        try:
            create = self.client.messages.create
            stream = await create(**_sdk_params(create, params))
        except BaseException as exc:
            error_response = getattr(exc, "response", None)
            headers = _response_headers(error_response)
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
        thinking_state: dict[int, dict[str, Any]] = {}
        model: str | None = params["model"]
        finish: FinishReason | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        last_raw: Any = None
        async for event in stream:
            last_raw = event
            event_type = _provider_value(event, "type")
            if event_type == "message_start":
                message = _provider_value(event, "message")
                model = _provider_value(message, "model", model)
                usage = _provider_value(message, "usage")
                input_tokens = _provider_value(usage, "input_tokens", input_tokens)
            elif event_type == "content_block_start":
                index = int(_provider_value(event, "index", 0))
                block = _provider_value(event, "content_block")
                block_type = _provider_value(block, "type")
                if block_type == "text":
                    text = _provider_value(block, "text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                        yield ModelStreamEvent(type="text_delta", text=text, raw=event)
                elif block_type in ("thinking", "redacted_thinking"):
                    thinking_state[index] = {
                        "type": block_type,
                        "thinking": str(_provider_value(block, "thinking", "") or ""),
                        "signature": str(_provider_value(block, "signature", "") or ""),
                        "data": str(_provider_value(block, "data", "") or ""),
                    }
                elif block_type == "tool_use":
                    initial = _provider_value(block, "input", {})
                    tool_state[index] = {
                        "id": str(_provider_value(block, "id", "")),
                        "name": str(_provider_value(block, "name", "")),
                        "argument_parts": (
                            [json.dumps(initial, separators=(",", ":"))]
                            if initial
                            else []
                        ),
                    }
            elif event_type == "content_block_delta":
                index = int(_provider_value(event, "index", 0))
                delta = _provider_value(event, "delta")
                delta_type = _provider_value(delta, "type")
                if delta_type == "text_delta":
                    text = _provider_value(delta, "text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                        yield ModelStreamEvent(type="text_delta", text=text, raw=event)
                elif delta_type == "thinking_delta":
                    chunk = _provider_value(delta, "thinking")
                    state = thinking_state.setdefault(
                        index,
                        {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                            "data": "",
                        },
                    )
                    if isinstance(chunk, str) and chunk:
                        state["thinking"] += chunk
                        yield ModelStreamEvent(
                            type="reasoning_delta", text=chunk, raw=event
                        )
                elif delta_type == "signature_delta":
                    signature = _provider_value(delta, "signature")
                    state = thinking_state.setdefault(
                        index,
                        {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                            "data": "",
                        },
                    )
                    if isinstance(signature, str):
                        state["signature"] += signature
                elif delta_type == "input_json_delta":
                    partial = _provider_value(delta, "partial_json")
                    json_state = tool_state.get(index)
                    if json_state is None:
                        json_state = {"id": "", "name": "", "argument_parts": []}
                        tool_state[index] = json_state
                    if isinstance(partial, str):
                        json_state["argument_parts"].append(partial)
                    yield ModelStreamEvent(
                        type="tool_call_delta",
                        tool_call_id=json_state["id"] or None,
                        tool_name=json_state["name"] or None,
                        arguments_delta=partial if isinstance(partial, str) else None,
                        raw=event,
                    )
            elif event_type == "message_delta":
                delta = _provider_value(event, "delta")
                reason = _provider_value(delta, "stop_reason")
                if reason is not None:
                    finish = _finish_reason(reason, anthropic=True)
                usage = _provider_value(event, "usage")
                output_tokens = _provider_value(usage, "output_tokens", output_tokens)
        tool_calls = tuple(
            _provider_tool_call(
                str(state["id"]),
                str(state["name"]),
                "".join(state["argument_parts"]),
            )
            for _, state in sorted(tool_state.items())
        )
        total = (
            input_tokens + output_tokens
            if isinstance(input_tokens, int) and isinstance(output_tokens, int)
            else None
        )
        response = ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(
                    *(
                        _thinking_part(state, state["type"])
                        for _, state in sorted(thinking_state.items())
                    ),
                    *(
                        (ContentPart(type="text", text="".join(text_parts)),)
                        if text_parts
                        else ()
                    ),
                ),
                tool_calls=tool_calls,
            ),
            model=model,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
            ),
            finish_reason=finish,
            raw=last_raw,
        )
        yield ModelStreamEvent(type="completed", response=response, raw=last_raw)
