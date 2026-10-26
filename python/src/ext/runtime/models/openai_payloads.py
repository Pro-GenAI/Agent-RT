"""OpenAI request/response payload conversion and rate-limit parsing."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from ext.runtime.models.conversion import (
    _finish_reason,
    _has_media,
    _message_text,
    _openai_content_parts,
    _provider_tool_call,
    _provider_value,
)
from ext.runtime.models.interfaces import (
    JSON_OBJECT_OUTPUT,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    StructuredOutputRequirement,
)
from ext.runtime.models.provider_settings import OpenAIProviderSettings
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ContentPart, ModelMessage, ToolCall


def _openai_message(message: ModelMessage) -> dict[str, Any]:
    text = _message_text(message)
    if message.role == "tool":
        if _has_media(message):
            raise ValueError("OpenAI chat cannot carry media content in tool messages")
        return {
            "role": "tool",
            "content": text,
            "tool_call_id": message.tool_call_id or "",
        }
    item: dict[str, Any] = {
        "role": message.role,
        "content": _openai_content_parts(message) if _has_media(message) else text,
    }
    if message.tool_calls:
        item["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        dict(call.arguments), separators=(",", ":")
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return item


def _openai_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return [_openai_message(message) for message in messages]


def _openai_tool_params(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            },
        }
        for tool in tools
    ]


def _openai_response_format(
    requirement: StructuredOutputRequirement,
) -> dict[str, Any]:
    if requirement == JSON_OBJECT_OUTPUT:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": requirement.name or "response",
            "schema": dict(requirement.schema),
            "strict": requirement.strict,
        },
    }


def _openai_params(
    settings: OpenAIProviderSettings,
    request: ModelRequest,
    *,
    messages_payload: list[dict[str, Any]] | None = None,
    tools_payload: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": settings.resolve_model(request.model),
        "messages": (
            messages_payload
            if messages_payload is not None
            else _openai_messages(request.messages)
        ),
    }
    if request.tools:
        params["tools"] = (
            tools_payload
            if tools_payload is not None
            else _openai_tool_params(request.tools)
        )
    if request.tool_selection and request.tool_selection.required:
        required = request.tool_selection.required
        params["tool_choice"] = (
            {"type": "function", "function": {"name": required[0]}}
            if len(required) == 1
            else "required"
        )
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.reasoning is not None and request.reasoning.effort is not None:
        params["reasoning_effort"] = request.reasoning.effort
    if request.max_output_tokens is not None:
        params["max_completion_tokens"] = request.max_output_tokens
    if request.structured_output is not None:
        params["response_format"] = (
            response_format
            if response_format is not None
            else _openai_response_format(request.structured_output)
        )
    return params


def _openai_usage(value: Any) -> ModelUsage | None:
    if value is None:
        return None
    prompt = _provider_value(value, "prompt_tokens")
    completion = _provider_value(value, "completion_tokens")
    total = _provider_value(value, "total_tokens")
    prompt_details = _provider_value(value, "prompt_tokens_details")
    completion_details = _provider_value(value, "completion_tokens_details")
    return ModelUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cached_tokens=(
            _provider_value(prompt_details, "cached_tokens") if prompt_details else None
        ),
        reasoning_tokens=(
            _provider_value(completion_details, "reasoning_tokens")
            if completion_details
            else None
        ),
    )


def _openai_response(value: Any) -> ModelResponse:
    choices = _provider_value(value, "choices", ())
    if not choices:
        raise ValueError("OpenAI completion response did not contain a choice")
    choice = choices[0]
    message = _provider_value(choice, "message")
    tool_calls: list[ToolCall] = []
    for call in _provider_value(message, "tool_calls", ()) or ():
        function = _provider_value(call, "function")
        tool_calls.append(
            _provider_tool_call(
                str(_provider_value(call, "id", "")),
                str(_provider_value(function, "name", "")),
                _provider_value(function, "arguments", "{}"),
            )
        )
    content = _provider_value(message, "content") or ""
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=(ContentPart(type="text", text=content),) if content else (),
            tool_calls=tuple(tool_calls),
        ),
        model=_provider_value(value, "model"),
        usage=_openai_usage(_provider_value(value, "usage")),
        finish_reason=_finish_reason(_provider_value(choice, "finish_reason")),
        raw=value,
    )


class _OpenAICompatibleHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = dict(headers)
        self.response = self


def _response_headers(value: Any) -> dict[str, str]:
    headers = getattr(value, "headers", None)
    if headers is None:
        return {}
    if hasattr(headers, "items"):
        return {str(key).lower(): str(item) for key, item in headers.items()}
    result: dict[str, str] = {}
    for key, item in headers:
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        if isinstance(item, bytes):
            item = item.decode("latin-1")
        result[str(key).lower()] = str(item)
    return result


def _parse_rate_limit_duration(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        # Retry-After commonly uses seconds. Very large numeric values are
        # accepted as Unix timestamps for compatible providers.
        if numeric > 1_000_000_000:
            return max(0.0, numeric - time.time())
        return max(0.0, numeric)
    try:
        timestamp = datetime.fromisoformat(text.replace("z", "+00:00"))
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            timestamp = None
    if timestamp is not None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, timestamp.timestamp() - time.time())
    total = 0.0
    position = 0
    pattern = re.compile(r"([0-9]*\.?[0-9]+)(ms|s|m|h|d)")
    for match in pattern.finditer(text):
        if match.start() != position:
            return None
        number = float(match.group(1))
        unit = match.group(2)
        total += (
            number * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
        )
        position = match.end()
    return total if position == len(text) and position > 0 else None
