from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import secrets
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

from agent_rt import (
    AgentConfig,
    AgentLoop,
    AgentRunLimits,
    AgentRunResult,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    RateLimit,
    RateLimiter,
    RateLimitExceededError,
    ToolCall,
)

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "starlette"}:
        raise
    raise RuntimeError(
        "The Agent RT API server is optional. Install 'agent-rt[api]' "
        "to use agent_rt_api."
    ) from exc


LOGGER = logging.getLogger(__name__)

DEFAULT_API_RUN_LIMITS = AgentRunLimits(
    max_turns=16,
    max_tool_calls=64,
    timeout_seconds=120.0,
    max_total_tokens=100_000,
)
DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576
DEFAULT_MAX_WEBSOCKET_MESSAGE_BYTES = 1_048_576
DEFAULT_MAX_OUTPUT_TOKENS = 32_768
DEFAULT_REQUEST_RATE_LIMIT = RateLimit(limit=120, window_seconds=60.0)
DEFAULT_MAX_CONCURRENT_REQUESTS = 32
DEFAULT_MAX_QUEUED_REQUESTS = 128
DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS = 30.0


class _RequestBodyError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class _WebSocketPayloadTooLarge(ValueError):
    pass


class APIQueueOverloadedError(RuntimeError):
    def __init__(self, model: str, *, retry_after_seconds: float = 1.0) -> None:
        self.model = model
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        super().__init__("API execution queue is overloaded for model " + model)


class _ModelExecutionQueue:
    def __init__(self, *, max_running: int, max_queued: int, wait_timeout_seconds: float) -> None:
        self.max_running = max_running
        self.max_queued = max_queued
        self.wait_timeout_seconds = wait_timeout_seconds
        self._available = max_running
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    @property
    def running(self) -> int:
        return self.max_running - self._available

    @property
    def queued(self) -> int:
        return sum(1 for waiter in self._waiters if not waiter.done())

    async def acquire(self, model: str) -> None:
        async with self._lock:
            if self._available > 0:
                self._available -= 1
                return
            if self.queued >= self.max_queued:
                raise APIQueueOverloadedError(model)
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=self.wait_timeout_seconds)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            await self._withdraw(waiter)
            raise APIQueueOverloadedError(model) from exc
        except asyncio.CancelledError:
            await self._withdraw(waiter)
            raise

    async def _withdraw(self, waiter: asyncio.Future[None]) -> None:
        transferred = False
        async with self._lock:
            if waiter.done() and not waiter.cancelled():
                transferred = True
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    transferred = waiter.done() and not waiter.cancelled()
                if not waiter.done():
                    waiter.cancel()
        if transferred:
            await self.release()

    async def release(self) -> None:
        async with self._lock:
            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.done():
                    continue
                waiter.set_result(None)
                return
            self._available = min(self.max_running, self._available + 1)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "running": self.running,
            "queued": self.queued,
            "max_running": self.max_running,
            "max_queued": self.max_queued,
            "wait_timeout_seconds": self.wait_timeout_seconds,
        }


async def _read_json_object(request: Any, max_request_body_bytes: int) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise _RequestBodyError("Content-Length must be an integer.") from exc
        if declared_size < 0:
            raise _RequestBodyError("Content-Length must not be negative.")
        if declared_size > max_request_body_bytes:
            raise _RequestBodyError("Request body is too large.", status_code=413)

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > max_request_body_bytes:
            raise _RequestBodyError("Request body is too large.", status_code=413)
        chunks.append(chunk)

    raw = b"".join(chunks)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _RequestBodyError("Request body must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise _RequestBodyError("Request body must be a JSON object.")
    return value


async def _receive_websocket_json(
    websocket: Any,
    max_websocket_message_bytes: int,
) -> dict[str, Any]:
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(code=message.get("code", 1000))

    text_payload = message.get("text")
    bytes_payload = message.get("bytes")
    if text_payload is not None:
        raw = str(text_payload).encode("utf-8")
    elif bytes_payload is not None:
        raw = bytes(bytes_payload)
    else:
        raw = b""

    if len(raw) > max_websocket_message_bytes:
        raise _WebSocketPayloadTooLarge("WebSocket message is too large.")

    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("WebSocket message must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("WebSocket messages must be objects.")
    return value


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _client_identity(connection: Any) -> str:
    client = getattr(connection, "client", None)
    host = getattr(client, "host", None)
    return f"client:{host or 'unknown'}"


def _credential_digest(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"credential:{digest}"


def _text_part(text: str) -> tuple[ContentPart, ...]:
    return (ContentPart(type="text", text=text),) if text else ()


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError("unsupported content part: expected a string or object")
            item_type = item.get("type")
            text = item.get("text")
            if item_type in {None, "text", "input_text", "output_text"} and isinstance(
                text, str
            ):
                parts.append(text)
                continue
            # Never silently drop images/audio/files: the model would answer
            # without input the caller believes it received.
            raise ValueError(f"unsupported content part type: {item_type!r}")
        return "".join(parts)
    raise ValueError("message content must be a string or an array of text content parts")


def _tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("tool_calls must be an array")
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("tool_calls entries must be objects")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool_calls entries must include function objects")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("tool call arguments must contain valid JSON") from exc
        elif isinstance(arguments, Mapping):
            parsed = dict(arguments)
        else:
            raise ValueError("tool call arguments must be a JSON object or encoded object")
        if not isinstance(parsed, Mapping):
            raise ValueError("tool call arguments must decode to an object")
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=dict(parsed),
            )
        )
    return tuple(calls)


def _chat_messages(raw: Any) -> tuple[ModelMessage, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("messages must be an array")
    messages: list[ModelMessage] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("messages entries must be objects")
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        messages.append(
            ModelMessage(
                role=role,
                content=_text_part(_content_text(item.get("content"))),
                tool_calls=_tool_calls(item.get("tool_calls")),
                tool_call_id=(
                    str(item["tool_call_id"])
                    if item.get("tool_call_id") is not None
                    else None
                ),
            )
        )
    return tuple(messages)


def _responses_messages(raw: Any) -> tuple[ModelMessage, ...]:
    if isinstance(raw, str):
        return (ModelMessage(role="user", content=_text_part(raw)),)
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise ValueError("input must be a string or an array")
    messages: list[ModelMessage] = []
    for item in raw:
        if isinstance(item, str):
            messages.append(ModelMessage(role="user", content=_text_part(item)))
            continue
        if not isinstance(item, Mapping):
            raise ValueError("input entries must be strings or objects")
        item_type = item.get("type")
        if item_type in {None, "message"}:
            role = item.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"unsupported input role: {role!r}")
            messages.append(
                ModelMessage(role=role, content=_text_part(_content_text(item.get("content"))))
            )
            continue
        if item_type == "function_call":
            messages.append(
                ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=_tool_calls(
                        [{
                            "id": item.get("call_id") or item.get("id") or "",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }]
                    ),
                )
            )
            continue
        if item_type == "function_call_output":
            messages.append(
                ModelMessage(
                    role="tool",
                    content=_text_part(_content_text(item.get("output"))),
                    tool_call_id=str(item.get("call_id") or ""),
                )
            )
            continue
        raise ValueError(f"unsupported input item type: {item_type!r}")
    return tuple(messages)


def _message_text(message: ModelMessage | None) -> str:
    if message is None:
        return ""
    return "".join(part.text or "" for part in message.content if part.type == "text")


def _usage_dict(usage: ModelUsage | None, total_tokens: int = 0) -> dict[str, int]:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
        }
    prompt = usage.input_tokens or 0
    completion = usage.output_tokens or 0
    total = usage.total_tokens if usage.total_tokens is not None else prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _response_usage_dict(usage: ModelUsage | None, total_tokens: int = 0) -> dict[str, int]:
    chat = _usage_dict(usage, total_tokens)
    return {
        "input_tokens": chat["prompt_tokens"],
        "output_tokens": chat["completion_tokens"],
        "total_tokens": chat["total_tokens"],
    }


_LIMIT_TERMINATIONS = frozenset(
    {"max_turns", "max_tool_calls", "budget_exhausted", "timeout"}
)


def _finish_reason(response: ModelResponse | None, result: AgentRunResult) -> str:
    if result.termination_reason != "completed":
        # Limit-driven stops are truncations, not natural stops.
        return "length" if result.termination_reason in _LIMIT_TERMINATIONS else "stop"
    reason = response.finish_reason if response is not None else None
    if reason in {"tool_calls", "length", "content_filter"}:
        return reason
    return "stop"


def _chat_tool_calls(response: ModelResponse | None) -> list[dict[str, Any]] | None:
    if response is None or not response.message.tool_calls:
        return None
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(dict(call.arguments), separators=(",", ":")),
            },
        }
        for call in response.message.tool_calls
    ]


def _error_payload(message: str, *, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _anthropic_error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _anthropic_system_message(value: Any) -> tuple[ModelMessage, ...]:
    if value is None:
        return ()
    text = _content_text(value)
    return (ModelMessage(role="system", content=_text_part(text)),) if text else ()


def _anthropic_messages(raw: Any, *, system: Any = None) -> tuple[ModelMessage, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("messages must be an array")
    messages: list[ModelMessage] = list(_anthropic_system_message(system))
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("messages entries must be objects")
        role = item.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported Anthropic message role: {role!r}")
        content = item.get("content", "")
        if isinstance(content, str):
            messages.append(ModelMessage(role=role, content=_text_part(content)))
            continue
        if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
            raise ValueError("Anthropic message content must be a string or an array")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        def flush_text() -> None:
            if text_parts or tool_calls:
                messages.append(
                    ModelMessage(
                        role=role,
                        content=_text_part("".join(text_parts)),
                        tool_calls=tuple(tool_calls),
                    )
                )
                text_parts.clear()
                tool_calls.clear()

        for block in content:
            if not isinstance(block, Mapping):
                raise ValueError("Anthropic content blocks must be objects")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("Anthropic text blocks require a string text field")
                text_parts.append(text)
            elif block_type == "tool_use":
                if role != "assistant":
                    raise ValueError("tool_use blocks are only valid in assistant messages")
                tool_input = block.get("input", {})
                if not isinstance(tool_input, Mapping):
                    raise ValueError("tool_use input must be an object")
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=dict(tool_input),
                    )
                )
            elif block_type == "tool_result":
                if role != "user":
                    raise ValueError("tool_result blocks are only valid in user messages")
                flush_text()
                result_content = block.get("content", "")
                messages.append(
                    ModelMessage(
                        role="tool",
                        content=_text_part(_content_text(result_content)),
                        tool_call_id=str(block.get("tool_use_id") or ""),
                    )
                )
            elif block_type in {"image", "document"}:
                raise ValueError(
                    f"Anthropic {block_type} content blocks are not supported by this API adapter yet"
                )
            else:
                raise ValueError(f"unsupported Anthropic content block type: {block_type!r}")
        flush_text()
        if not content:
            messages.append(ModelMessage(role=role, content=()))
    return tuple(messages)


def _anthropic_stop_reason(response: ModelResponse | None, result: AgentRunResult) -> str:
    if result.termination_reason != "completed":
        return "max_tokens" if result.termination_reason in _LIMIT_TERMINATIONS else "end_turn"
    if response is not None and response.message.tool_calls:
        return "tool_use"
    if response is not None and response.finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _anthropic_content(
    response: ModelResponse | None,
    *,
    include_tool_calls: bool = True,
) -> list[dict[str, Any]]:
    if response is None:
        return []
    content: list[dict[str, Any]] = []
    text = _message_text(response.message)
    if text:
        content.append({"type": "text", "text": text})
    if not include_tool_calls:
        return content
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(call.arguments),
        }
        for call in response.message.tool_calls
    )
    return content


def _anthropic_usage(usage: ModelUsage | None) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens or 0 if usage is not None else 0,
        "output_tokens": usage.output_tokens or 0 if usage is not None else 0,
    }


class AgentRTAPIServer:
    """Expose Agent RT agents through OpenAI- and Anthropic-compatible APIs."""

    def __init__(
        self,
        loop: AgentLoop,
        agents: Mapping[str, AgentConfig],
        *,
        default_agent: str | None = None,
        api_keys: Collection[str] = (),
        limits: AgentRunLimits = DEFAULT_API_RUN_LIMITS,
        max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
        max_websocket_message_bytes: int = DEFAULT_MAX_WEBSOCKET_MESSAGE_BYTES,
        generation_cap: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_rate_limit: RateLimit = DEFAULT_REQUEST_RATE_LIMIT,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_queued_requests: int = DEFAULT_MAX_QUEUED_REQUESTS,
        queue_wait_timeout_seconds: float = DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS,
        allowed_websocket_origins: Collection[str] = (),
    ) -> None:
        if not agents:
            raise ValueError("at least one agent must be configured")
        if max_request_body_bytes < 1:
            raise ValueError("max_request_body_bytes must be positive")
        if max_websocket_message_bytes < 1:
            raise ValueError("max_websocket_message_bytes must be positive")
        if generation_cap < 1:
            raise ValueError("generation_cap must be positive")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if max_queued_requests < 0:
            raise ValueError("max_queued_requests must be non-negative")
        if queue_wait_timeout_seconds <= 0 or not math.isfinite(queue_wait_timeout_seconds):
            raise ValueError("queue_wait_timeout_seconds must be finite and positive")
        normalized_keys = tuple(dict.fromkeys(str(value) for value in api_keys))
        if any(not value for value in normalized_keys):
            raise ValueError("configured API keys must not be empty")
        normalized_origins = frozenset(str(value).strip() for value in allowed_websocket_origins)
        if any(not value for value in normalized_origins):
            raise ValueError("allowed WebSocket origins must not be empty")
        if "*" in normalized_origins:
            raise ValueError("wildcard WebSocket origins are not allowed")

        self.loop = loop
        self.agents = dict(agents)
        self.default_agent = default_agent or (next(iter(self.agents)) if len(self.agents) == 1 else None)
        if self.default_agent is not None and self.default_agent not in self.agents:
            raise ValueError("default_agent must name a configured agent")
        self.api_keys = normalized_keys
        self.limits = limits
        self.max_request_body_bytes = max_request_body_bytes
        self.max_websocket_message_bytes = max_websocket_message_bytes
        self.generation_cap = generation_cap
        self.allowed_websocket_origins = normalized_origins
        self.request_rate_limiter = RateLimiter(request_rate_limit)
        self.execution_queues = {
            model_id: _ModelExecutionQueue(
                max_running=max_concurrent_requests,
                max_queued=max_queued_requests,
                wait_timeout_seconds=queue_wait_timeout_seconds,
            )
            for model_id in self.agents
        }

    def resolve_agent(self, model: Any) -> tuple[str, AgentConfig]:
        if model is None:
            if self.default_agent is None:
                raise ValueError("model is required when multiple agents are configured")
            return self.default_agent, self.agents[self.default_agent]
        model_id = str(model)
        agent = self.agents.get(model_id)
        if agent is None:
            raise ValueError(f"unknown model/agent {model_id!r}")
        return model_id, agent

    def authorize_value(self, candidate: str | None) -> bool:
        if not self.api_keys:
            return True
        if candidate is None:
            return False
        # Compare bytes: str comparison raises TypeError on non-ASCII input,
        # which an unauthenticated client could use to force a 500.
        candidate_bytes = candidate.encode("utf-8", errors="surrogateescape")
        matched = False
        for configured in self.api_keys:
            matched = (
                secrets.compare_digest(
                    candidate_bytes,
                    configured.encode("utf-8", errors="surrogateescape"),
                )
                or matched
            )
        return matched

    def authorize_header(self, authorization: str | None) -> bool:
        if not self.api_keys:
            return True
        if not authorization:
            return False
        scheme, separator, value = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not value:
            return False
        return self.authorize_value(value)

    def request_identity(self, connection: Any) -> str:
        """Rate-limit identity: a verified credential, otherwise the client address.

        Unverified credentials never mint their own bucket, so rotating
        ``Authorization`` values cannot evade or flood the limiter.
        """
        if self.api_keys:
            authorization = connection.headers.get("authorization")
            if authorization and self.authorize_header(authorization):
                return _credential_digest(authorization)
            api_key = connection.headers.get("x-api-key")
            if api_key and self.authorize_value(api_key):
                return _credential_digest(api_key)
        return _client_identity(connection)

    def is_authorized(self, connection: Any) -> bool:
        if not self.api_keys:
            return True
        return self.authorize_header(
            connection.headers.get("authorization")
        ) or self.authorize_value(connection.headers.get("x-api-key"))

    def check_request_rate(self, identity: str) -> int:
        return self.request_rate_limiter.check("api:" + identity)

    def websocket_origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return True
        return origin in self.allowed_websocket_origins

    def prepare_run(
        self,
        *,
        model: Any,
        temperature: Any = None,
        max_output_tokens: Any = None,
    ) -> tuple[str, AgentConfig]:
        """Resolve the agent and validate per-request generation settings.

        Streaming endpoints call this before the response starts so malformed
        requests get a proper 4xx instead of a failure inside the stream.
        """
        model_id, agent = self.resolve_agent(model)
        settings = agent.model
        if temperature is not None:
            try:
                requested_temperature = float(temperature)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("temperature must be a number") from exc
            if not math.isfinite(requested_temperature):
                raise ValueError("temperature must be finite")
            settings = replace(settings, temperature=requested_temperature)
        if max_output_tokens is not None:
            try:
                requested_output = int(max_output_tokens)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max output tokens must be an integer") from exc
            if requested_output < 1:
                raise ValueError("max output tokens must be positive")
            settings = replace(
                settings,
                max_output_tokens=min(requested_output, self.generation_cap),
            )
        if settings is not agent.model:
            agent = replace(agent, model=settings)
        return model_id, agent

    async def run(
        self,
        *,
        model: Any,
        messages: Sequence[ModelMessage],
        temperature: Any = None,
        max_output_tokens: Any = None,
        stream_handler: Any = None,
    ) -> tuple[str, AgentRunResult]:
        model_id, agent = self.prepare_run(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        execution_queue = self.execution_queues[model_id]
        await execution_queue.acquire(model_id)
        try:
            if stream_handler is None:
                result = await self.loop.run(agent, messages, limits=self.limits)
            else:
                result = await self.loop.run_streaming(
                    agent,
                    messages,
                    stream_handler,
                    limits=self.limits,
                )
        finally:
            await execution_queue.release()
        return model_id, result

    def chat_response(self, model_id: str, result: AgentRunResult, request_id: str) -> dict[str, Any]:
        response = result.final_response
        message = response.message if response is not None else None
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": _message_text(message),
        }
        # Unexecuted tool calls from an interrupted run are server-side calls;
        # exposing them would invite clients to execute them.
        tool_calls = (
            _chat_tool_calls(response)
            if result.termination_reason == "completed"
            else None
        )
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": payload,
                "finish_reason": _finish_reason(response, result),
            }],
            "usage": _usage_dict(response.usage if response else None, result.total_tokens),
        }

    def responses_response(self, model_id: str, result: AgentRunResult, response_id: str) -> dict[str, Any]:
        response = result.final_response
        output: list[dict[str, Any]] = []
        if response is not None:
            text = _message_text(response.message)
            if text:
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }],
                })
            for call in response.message.tool_calls:
                output.append({
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), separators=(",", ":")),
                    "status": "completed",
                })
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed" if result.termination_reason == "completed" else "incomplete",
            "model": model_id,
            "output": output,
            "usage": _response_usage_dict(response.usage if response else None, result.total_tokens),
            "error": None,
            "incomplete_details": (
                None
                if result.termination_reason == "completed"
                else {"reason": result.termination_reason}
            ),
        }

    def anthropic_response(
        self,
        model_id: str,
        result: AgentRunResult,
        message_id: str,
    ) -> dict[str, Any]:
        response = result.final_response
        return {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": _anthropic_content(
                response,
                include_tool_calls=result.termination_reason == "completed",
            ),
            "stop_reason": _anthropic_stop_reason(response, result),
            "stop_sequence": None,
            "usage": _anthropic_usage(response.usage if response else None),
        }


def create_app(
    loop: AgentLoop,
    agents: Mapping[str, AgentConfig],
    *,
    default_agent: str | None = None,
    api_keys: Collection[str] = (),
    limits: AgentRunLimits = DEFAULT_API_RUN_LIMITS,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_websocket_message_bytes: int = DEFAULT_MAX_WEBSOCKET_MESSAGE_BYTES,
    generation_cap: int = DEFAULT_MAX_OUTPUT_TOKENS,
    request_rate_limit: RateLimit = DEFAULT_REQUEST_RATE_LIMIT,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_queued_requests: int = DEFAULT_MAX_QUEUED_REQUESTS,
    queue_wait_timeout_seconds: float = DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS,
    allowed_websocket_origins: Collection[str] = (),
    enable_docs: bool = False,
) -> Any:
    server = AgentRTAPIServer(
        loop,
        agents,
        default_agent=default_agent,
        api_keys=api_keys,
        limits=limits,
        max_request_body_bytes=max_request_body_bytes,
        max_websocket_message_bytes=max_websocket_message_bytes,
        generation_cap=generation_cap,
        request_rate_limit=request_rate_limit,
        max_concurrent_requests=max_concurrent_requests,
        max_queued_requests=max_queued_requests,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
        allowed_websocket_origins=allowed_websocket_origins,
    )
    app = FastAPI(
        title="Agent RT Compatible API",
        version="1.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    app.state.agent_rt_server = server

    @app.exception_handler(APIQueueOverloadedError)
    async def queue_overloaded(request: Request, exc: APIQueueOverloadedError) -> JSONResponse:
        headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after_seconds)))}
        if request.headers.get("anthropic-version") is not None:
            return JSONResponse(
                status_code=503,
                content=_anthropic_error_payload("Server is overloaded. Retry later.", error_type="overloaded_error"),
                headers=headers,
            )
        return JSONResponse(
            status_code=503,
            content=_error_payload("Server is overloaded. Retry later.", error_type="server_error"),
            headers=headers,
        )

    @app.exception_handler(RateLimitExceededError)
    async def provider_rate_limited(request: Request, exc: RateLimitExceededError) -> JSONResponse:
        headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after_seconds)))}
        if request.headers.get("anthropic-version") is not None:
            return JSONResponse(
                status_code=429,
                content=_anthropic_error_payload("Rate limit exceeded.", error_type="rate_limit_error"),
                headers=headers,
            )
        return JSONResponse(
            status_code=429,
            content=_error_payload("Rate limit exceeded.", error_type="rate_limit_error"),
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def openai_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, Mapping) and "error" in detail:
            content = dict(detail)
        else:
            content = _error_payload(str(detail))
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    def require_http_auth(request: Any) -> None:
        if not server.authorize_header(request.headers.get("authorization")):
            raise HTTPException(
                status_code=401,
                detail=_error_payload("Invalid API key.", error_type="authentication_error"),
            )

    def anthropic_authorized(request: Any) -> bool:
        if not server.api_keys:
            return True
        header_value = request.headers.get("x-api-key")
        if server.authorize_value(header_value):
            return True
        return server.authorize_header(request.headers.get("authorization"))

    def anthropic_http_error(
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        headers: Mapping[str, str] | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=_anthropic_error_payload(message, error_type=error_type),
            headers=dict(headers or {}),
        )

    async def body_json(request: Any) -> dict[str, Any]:
        try:
            return await _read_json_object(request, server.max_request_body_bytes)
        except _RequestBodyError as exc:
            message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else "Invalid request body."
            raise HTTPException(
                status_code=exc.status_code,
                detail=_error_payload(message),
            ) from exc

    def public_request_error(exc: Exception) -> tuple[int, str]:
        detail = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
        if detail.startswith("unknown model/agent"):
            return 404, "Unknown model/agent."
        if detail.startswith(
            ("unsupported content part", "temperature must", "max output tokens must")
        ):
            return 400, detail
        return 400, "Invalid request."

    def http_error(exc: Exception) -> Any:
        status, message = public_request_error(exc)
        return JSONResponse(status_code=status, content=_error_payload(message))

    def run_failure(exc: Exception, *, anthropic: bool = False) -> Any:
        """Map an exception raised while *executing* a run to an HTTP response.

        Validation happens before the run starts, so a ``ValueError`` here is an
        upstream/runtime fault (for example a malformed provider reply), not a
        client mistake; it must not be reported as ``400 Invalid request``.
        """
        LOGGER.exception("Agent run failed", exc_info=exc)
        if anthropic:
            return anthropic_http_error(
                "Internal server error.",
                status_code=500,
                error_type="api_error",
            )
        return JSONResponse(
            status_code=500,
            content=_error_payload("Internal server error.", error_type="server_error"),
        )

    def stream_error_payload(exc: BaseException, *, anthropic: bool = False) -> dict[str, Any]:
        if isinstance(exc, RateLimitExceededError):
            payload = (
                _anthropic_error_payload("Rate limit exceeded.", error_type="rate_limit_error")
                if anthropic
                else _error_payload("Rate limit exceeded.", error_type="rate_limit_error")
            )
            payload["retry_after_seconds"] = max(
                1, math.ceil(exc.retry_after_seconds)
            )
            return payload
        if isinstance(exc, APIQueueOverloadedError):
            payload = (
                _anthropic_error_payload("Server is overloaded. Retry later.", error_type="overloaded_error")
                if anthropic
                else _error_payload("Server is overloaded. Retry later.", error_type="server_error")
            )
            payload["retry_after_seconds"] = max(
                1, math.ceil(exc.retry_after_seconds)
            )
            return payload
        return (
            _anthropic_error_payload("Internal server error.", error_type="api_error")
            if anthropic
            else _error_payload("Internal server error.", error_type="server_error")
        )

    @app.middleware("http")
    async def enforce_request_rate(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/v1/"):
            try:
                server.check_request_rate(server.request_identity(request))
            except RateLimitExceededError as exc:
                headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after_seconds)))}
                if request.headers.get("anthropic-version") is not None:
                    return anthropic_http_error(
                        "Rate limit exceeded.",
                        status_code=429,
                        error_type="rate_limit_error",
                        headers=headers,
                    )
                return JSONResponse(
                    status_code=429,
                    content=_error_payload("Rate limit exceeded.", error_type="rate_limit_error"),
                    headers=headers,
                )
        return await call_next(request)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        if not server.is_authorized(request):
            # Agent names and queue depth are only for authenticated callers.
            return {"status": "ok"}
        return {
            "status": "ok",
            "queues": {
                model_id: queue.snapshot()
                for model_id, queue in server.execution_queues.items()
            },
        }

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        if request.headers.get("anthropic-version") is not None:
            if not anthropic_authorized(request):
                return anthropic_http_error(
                    "Invalid API key.",
                    status_code=401,
                    error_type="authentication_error",
                )
            data = [
                {
                    "id": model_id,
                    "type": "model",
                    "display_name": model_id,
                    "created_at": "1970-01-01T00:00:00Z",
                }
                for model_id in server.agents
            ]
            return {
                "data": data,
                "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
            }

        require_http_auth(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "agent-rt",
                }
                for model_id in server.agents
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        require_http_auth(request)
        body = await body_json(request)
        try:
            messages = _chat_messages(body.get("messages"))
            model_id, _ = server.prepare_run(
                model=body.get("model"),
                temperature=body.get("temperature"),
                max_output_tokens=body.get(
                    "max_completion_tokens", body.get("max_tokens")
                ),
            )
        except (TypeError, ValueError) as exc:
            return http_error(exc)

        if body.get("stream"):
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            async def generate() -> AsyncIterator[bytes]:
                queue: asyncio.Queue[Any] = asyncio.Queue()
                completed_result: AgentRunResult | None = None

                async def on_event(event: ModelStreamEvent) -> None:
                    await queue.put(event)

                async def execute() -> None:
                    nonlocal completed_result
                    try:
                        _, completed_result = await server.run(
                            model=body.get("model"),
                            messages=messages,
                            temperature=body.get("temperature"),
                            max_output_tokens=body.get("max_completion_tokens", body.get("max_tokens")),
                            stream_handler=on_event,
                        )
                    except BaseException as exc:
                        await queue.put(exc)
                    finally:
                        await queue.put(None)

                task = asyncio.create_task(execute())
                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        if isinstance(item, BaseException):
                            error = stream_error_payload(item)
                            yield f"data: {json.dumps(error, separators=(',', ':'))}\n\n".encode()
                            return
                        event: ModelStreamEvent = item
                        delta: dict[str, Any] = {}
                        if event.type == "text_delta" and event.text:
                            delta["content"] = event.text
                        elif event.type == "tool_call_delta":
                            delta["tool_calls"] = [{
                                "index": 0,
                                "id": event.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": event.tool_name,
                                    "arguments": event.arguments_delta or "",
                                },
                            }]
                        else:
                            continue
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()

                    if completed_result is not None:
                        response = completed_result.final_response
                        final_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": _finish_reason(response, completed_result),
                            }],
                        }
                        yield f"data: {json.dumps(final_chunk, separators=(',', ':'))}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                finally:
                    if not task.done():
                        task.cancel()

            return StreamingResponse(generate(), media_type="text/event-stream")

        try:
            model_id, result = await server.run(
                model=body.get("model"),
                messages=messages,
                temperature=body.get("temperature"),
                max_output_tokens=body.get("max_completion_tokens", body.get("max_tokens")),
            )
        except (TypeError, ValueError) as exc:
            return run_failure(exc)
        return server.chat_response(model_id, result, f"chatcmpl-{uuid.uuid4().hex}")

    async def run_responses_stream(body: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        messages = _responses_messages(body.get("input", ""))
        model_id, _ = server.resolve_agent(body.get("model"))
        response_id = f"resp_{uuid.uuid4().hex}"
        yield {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "in_progress",
                "model": model_id,
                "output": [],
            },
        }
        queue: asyncio.Queue[Any] = asyncio.Queue()
        result: AgentRunResult | None = None

        async def on_event(event: ModelStreamEvent) -> None:
            await queue.put(event)

        async def execute() -> None:
            nonlocal result
            try:
                _, result = await server.run(
                    model=body.get("model"),
                    messages=messages,
                    temperature=body.get("temperature"),
                    max_output_tokens=body.get("max_output_tokens"),
                    stream_handler=on_event,
                )
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                event: ModelStreamEvent = item
                if event.type == "text_delta" and event.text:
                    yield {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "delta": event.text,
                    }
                elif event.type == "tool_call_delta":
                    yield {
                        "type": "response.function_call_arguments.delta",
                        "response_id": response_id,
                        "call_id": event.tool_call_id,
                        "name": event.tool_name,
                        "delta": event.arguments_delta or "",
                    }
            if result is None:
                raise RuntimeError("stream ended without an Agent RT result")
            yield {
                "type": "response.completed",
                "response": server.responses_response(model_id, result, response_id),
            }
        finally:
            if not task.done():
                task.cancel()

    @app.post("/v1/responses")
    async def responses_endpoint(request: Request) -> Any:
        require_http_auth(request)
        body = await body_json(request)
        try:
            _responses_messages(body.get("input", ""))
            server.prepare_run(
                model=body.get("model"),
                temperature=body.get("temperature"),
                max_output_tokens=body.get("max_output_tokens"),
            )
            if body.get("stream"):
                async def generate() -> AsyncIterator[bytes]:
                    try:
                        async for event in run_responses_stream(body):
                            yield (
                                f"event: {event['type']}\n"
                                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                            ).encode()
                    except Exception as exc:
                        if not isinstance(
                            exc, (RateLimitExceededError, APIQueueOverloadedError)
                        ):
                            LOGGER.exception("Responses SSE stream failed")
                        error = {
                            "type": "error",
                            **stream_error_payload(exc),
                        }
                        yield f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n".encode()
                return StreamingResponse(generate(), media_type="text/event-stream")

            messages = _responses_messages(body.get("input", ""))
        except (TypeError, ValueError) as exc:
            return http_error(exc)
        try:
            model_id, result = await server.run(
                model=body.get("model"),
                messages=messages,
                temperature=body.get("temperature"),
                max_output_tokens=body.get("max_output_tokens"),
            )
        except (TypeError, ValueError) as exc:
            return run_failure(exc)
        return server.responses_response(model_id, result, f"resp_{uuid.uuid4().hex}")

    async def run_anthropic_stream(
        body: Mapping[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        messages = _anthropic_messages(body.get("messages"), system=body.get("system"))
        model_id, _ = server.resolve_agent(body.get("model"))
        message_id = f"msg_{uuid.uuid4().hex}"
        yield {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }

        queue: asyncio.Queue[Any] = asyncio.Queue()
        result: AgentRunResult | None = None
        next_index = 0
        text_index: int | None = None
        open_tool_indexes: dict[str, int] = {}

        async def on_event(event: ModelStreamEvent) -> None:
            await queue.put(event)

        async def execute() -> None:
            nonlocal result
            try:
                _, result = await server.run(
                    model=body.get("model"),
                    messages=messages,
                    temperature=body.get("temperature"),
                    max_output_tokens=body.get("max_tokens"),
                    stream_handler=on_event,
                )
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                event: ModelStreamEvent = item
                if event.type == "text_delta" and event.text:
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        yield {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    yield {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": event.text},
                    }
                elif event.type == "tool_call_delta":
                    call_key = event.tool_call_id or f"tool-{next_index}"
                    index = open_tool_indexes.get(call_key)
                    if index is None:
                        index = next_index
                        next_index += 1
                        open_tool_indexes[call_key] = index
                        yield {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": event.tool_call_id or "",
                                "name": event.tool_name or "",
                                "input": {},
                            },
                        }
                    if event.arguments_delta:
                        yield {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": event.arguments_delta,
                            },
                        }

            if result is None:
                raise RuntimeError("stream ended without an Agent RT result")

            if text_index is not None:
                yield {"type": "content_block_stop", "index": text_index}
            for index in open_tool_indexes.values():
                yield {"type": "content_block_stop", "index": index}

            response = result.final_response
            yield {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _anthropic_stop_reason(response, result),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": (
                        response.usage.output_tokens or 0
                        if response is not None
                        else 0
                    )
                },
            }
            yield {"type": "message_stop"}
        finally:
            if not task.done():
                task.cancel()

    @app.post("/v1/messages")
    async def anthropic_messages_endpoint(request: Request) -> Any:
        if not anthropic_authorized(request):
            return anthropic_http_error(
                "Invalid API key.",
                status_code=401,
                error_type="authentication_error",
            )
        try:
            body = await _read_json_object(request, server.max_request_body_bytes)
        except _RequestBodyError as exc:
            message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else "Invalid request body."
            return anthropic_http_error(
                message,
                status_code=exc.status_code,
            )

        if body.get("tools"):
            return anthropic_http_error(
                "Request-defined Anthropic tools are not supported; expose tools through "
                "the configured Agent RT runtime instead."
            )

        try:
            _anthropic_messages(body.get("messages"), system=body.get("system"))
            server.prepare_run(
                model=body.get("model"),
                temperature=body.get("temperature"),
                max_output_tokens=body.get("max_tokens"),
            )
        except (TypeError, ValueError) as exc:
            status, message = public_request_error(exc)
            error_type = "not_found_error" if status == 404 else "invalid_request_error"
            return anthropic_http_error(
                message,
                status_code=status,
                error_type=error_type,
            )

        if body.get("stream"):
            async def generate() -> AsyncIterator[bytes]:
                try:
                    async for event in run_anthropic_stream(body):
                        yield (
                            f"event: {event['type']}\n"
                            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                        ).encode()
                except Exception as exc:
                    if not isinstance(
                        exc, (RateLimitExceededError, APIQueueOverloadedError)
                    ):
                        LOGGER.exception("Anthropic SSE stream failed")
                    error = stream_error_payload(exc, anthropic=True)
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(error, separators=(',', ':'))}\n\n"
                    ).encode()

            return StreamingResponse(generate(), media_type="text/event-stream")

        try:
            messages = _anthropic_messages(body.get("messages"), system=body.get("system"))
        except (TypeError, ValueError) as exc:
            status, message = public_request_error(exc)
            error_type = "not_found_error" if status == 404 else "invalid_request_error"
            return anthropic_http_error(
                message,
                status_code=status,
                error_type=error_type,
            )
        try:
            model_id, result = await server.run(
                model=body.get("model"),
                messages=messages,
                temperature=body.get("temperature"),
                max_output_tokens=body.get("max_tokens"),
            )
        except (TypeError, ValueError) as exc:
            return run_failure(exc, anthropic=True)
        return server.anthropic_response(
            model_id,
            result,
            f"msg_{uuid.uuid4().hex}",
        )

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if not server.websocket_origin_allowed(origin):
            await websocket.close(code=4403, reason="Origin is not allowed")
            return

        client = getattr(websocket, "client", None)
        client_host = getattr(client, "host", None)
        try:
            server.check_request_rate(f"ws-connect:{client_host or 'unknown'}")
        except RateLimitExceededError:
            await websocket.close(code=4429, reason="Rate limit exceeded")
            return

        authorization = websocket.headers.get("authorization")
        if not server.authorize_header(authorization):
            await websocket.close(code=4401, reason="Invalid API key")
            return

        await websocket.accept()
        try:
            while True:
                try:
                    body = await _receive_websocket_json(
                        websocket,
                        server.max_websocket_message_bytes,
                    )
                except _WebSocketPayloadTooLarge:
                    await websocket.close(code=1009, reason="Message too large")
                    return
                except ValueError:
                    await websocket.send_json({
                        "type": "error",
                        **_error_payload("Invalid WebSocket message."),
                    })
                    continue

                try:
                    server.check_request_rate(server.request_identity(websocket))
                except RateLimitExceededError as exc:
                    await websocket.send_json({
                        "type": "error",
                        **_error_payload(
                            "Rate limit exceeded.",
                            error_type="rate_limit_error",
                        ),
                        "retry_after_seconds": max(1, math.ceil(exc.retry_after_seconds)),
                    })
                    continue

                if body.get("type") != "response.create":
                    await websocket.send_json({
                        "type": "error",
                        **_error_payload("Expected a response.create event."),
                    })
                    continue
                stream_id = body.get("stream_id")
                try:
                    _responses_messages(body.get("input", ""))
                    server.prepare_run(
                        model=body.get("model"),
                        temperature=body.get("temperature"),
                        max_output_tokens=body.get("max_output_tokens"),
                    )
                except (TypeError, ValueError) as exc:
                    _status, message = public_request_error(exc)
                    invalid = {"type": "error", **_error_payload(message)}
                    if stream_id is not None:
                        invalid["stream_id"] = stream_id
                    await websocket.send_json(invalid)
                    continue
                try:
                    async for event in run_responses_stream(body):
                        if stream_id is not None:
                            event = {"stream_id": stream_id, **event}
                        await websocket.send_json(event)
                except Exception as exc:
                    if not isinstance(exc, (RateLimitExceededError, APIQueueOverloadedError)):
                        LOGGER.exception("Responses WebSocket stream failed")
                    event = {
                        "type": "error",
                        **stream_error_payload(exc),
                    }
                    if stream_id is not None:
                        event["stream_id"] = stream_id
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            return

    return app


def serve(
    loop: AgentLoop,
    agents: Mapping[str, AgentConfig],
    *,
    default_agent: str | None = None,
    api_keys: Collection[str] = (),
    limits: AgentRunLimits = DEFAULT_API_RUN_LIMITS,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_websocket_message_bytes: int = DEFAULT_MAX_WEBSOCKET_MESSAGE_BYTES,
    generation_cap: int = DEFAULT_MAX_OUTPUT_TOKENS,
    request_rate_limit: RateLimit = DEFAULT_REQUEST_RATE_LIMIT,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_queued_requests: int = DEFAULT_MAX_QUEUED_REQUESTS,
    queue_wait_timeout_seconds: float = DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS,
    allowed_websocket_origins: Collection[str] = (),
    enable_docs: bool = False,
    allow_unauthenticated_non_loopback: bool = False,
    **uvicorn_options: Any,
) -> None:
    """Run the optional API server with Uvicorn."""

    if (
        not api_keys
        and not _is_loopback_host(host)
        and not allow_unauthenticated_non_loopback
    ):
        raise ValueError(
            "Refusing unauthenticated non-loopback API bind; configure API keys "
            "or explicitly set allow_unauthenticated_non_loopback=True."
        )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Agent RT API server is optional. Install 'agent-rt[api]' "
            "to use agent_rt_api.serve()."
        ) from exc

    app = create_app(
        loop,
        agents,
        default_agent=default_agent,
        api_keys=api_keys,
        limits=limits,
        max_request_body_bytes=max_request_body_bytes,
        max_websocket_message_bytes=max_websocket_message_bytes,
        generation_cap=generation_cap,
        request_rate_limit=request_rate_limit,
        max_concurrent_requests=max_concurrent_requests,
        max_queued_requests=max_queued_requests,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
        allowed_websocket_origins=allowed_websocket_origins,
        enable_docs=enable_docs,
    )
    uvicorn_options.setdefault("ws_max_size", max_websocket_message_bytes)
    uvicorn.run(app, host=host, port=port, **uvicorn_options)


__all__ = ("APIQueueOverloadedError", "AgentRTAPIServer", "create_app", "serve")
