"""Anthropic request/response payload conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ext.runtime.models.conversion import (
    PROVIDER_STATE_MIME,
    _anthropic_content_blocks,
    _finish_reason,
    _has_media,
    _is_provider_state,
    _message_text,
    _provider_value,
)
from ext.runtime.models.interfaces import ModelRequest, ModelResponse, ModelUsage
from ext.runtime.models.provider_settings import AnthropicProviderSettings
from ext.runtime.types import ContentPart, ModelMessage, ToolCall


def _anthropic_message(
    message: ModelMessage,
) -> tuple[str | None, dict[str, Any] | None]:
    text = _message_text(message)
    has_media = _has_media(message)
    if message.role == "system":
        if has_media:
            raise ValueError("Anthropic cannot carry media content in system messages")
        return (text or None, None)
    if message.role == "tool":
        return (
            None,
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": (
                            _anthropic_content_blocks(message) if has_media else text
                        ),
                    }
                ],
            },
        )
    content: list[dict[str, Any]] = []
    if message.role == "assistant":
        # Thinking blocks must lead the assistant turn that carries tool_use.
        content.extend(
            dict(part.data)
            for part in message.content
            if _is_provider_state(part) and isinstance(part.data, Mapping)
        )
    if has_media:
        content.extend(_anthropic_content_blocks(message))
    elif text:
        content.append({"type": "text", "text": text})
    if message.role == "assistant":
        content.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": dict(call.arguments),
            }
            for call in message.tool_calls
        )
    return (None, {"role": message.role, "content": content})


def _anthropic_params(
    settings: AnthropicProviderSettings,
    request: ModelRequest,
    *,
    converted_messages: (
        Sequence[tuple[str | None, dict[str, Any] | None]] | None
    ) = None,
) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    converted = (
        converted_messages
        if converted_messages is not None
        else tuple(_anthropic_message(message) for message in request.messages)
    )
    for system_text, payload in converted:
        if system_text:
            system_parts.append(system_text)
        if payload is not None:
            messages.append(payload)
    params: dict[str, Any] = {
        "model": settings.resolve_model(request.model),
        "messages": messages,
        "max_tokens": request.max_output_tokens or 4096,
    }
    if system_parts:
        params["system"] = "\n\n".join(system_parts)
    if request.tools:
        params["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ]
    if request.tool_selection and request.tool_selection.required:
        required = request.tool_selection.required
        params["tool_choice"] = (
            {"type": "tool", "name": required[0]}
            if len(required) == 1
            else {"type": "any"}
        )
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.reasoning is not None and request.reasoning.thinking is not None:
        thinking: dict[str, Any] = {"type": request.reasoning.thinking}
        if request.reasoning.budget_tokens is not None:
            thinking["budget_tokens"] = request.reasoning.budget_tokens
        params["thinking"] = thinking
    output_config: dict[str, Any] = {}
    if request.structured_output is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": dict(request.structured_output.schema),
        }
    if request.reasoning is not None and request.reasoning.effort is not None:
        output_config["effort"] = request.reasoning.effort
    if output_config:
        params["output_config"] = output_config
    return params


def _anthropic_usage(value: Any) -> ModelUsage | None:
    if value is None:
        return None
    input_tokens = _provider_value(value, "input_tokens")
    output_tokens = _provider_value(value, "output_tokens")
    total = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total = input_tokens + output_tokens
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _thinking_part(block: Any, block_type: str) -> ContentPart:
    """Keep a thinking/redacted_thinking block so it can be replayed verbatim."""
    payload: dict[str, Any] = {"type": block_type}
    if block_type == "thinking":
        payload["thinking"] = str(_provider_value(block, "thinking", "") or "")
        payload["signature"] = str(_provider_value(block, "signature", "") or "")
    else:
        payload["data"] = str(_provider_value(block, "data", "") or "")
    return ContentPart(type="json", data=payload, mime_type=PROVIDER_STATE_MIME)


def _anthropic_response(value: Any) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_parts: list[ContentPart] = []
    for block in _provider_value(value, "content", ()) or ():
        block_type = _provider_value(block, "type")
        if block_type in ("thinking", "redacted_thinking"):
            thinking_parts.append(_thinking_part(block, block_type))
        elif block_type == "text":
            text = _provider_value(block, "text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            arguments = _provider_value(block, "input", {})
            if not isinstance(arguments, Mapping):
                raise ValueError("Anthropic tool_use input must be an object")
            tool_calls.append(
                ToolCall(
                    id=str(_provider_value(block, "id", "")),
                    name=str(_provider_value(block, "name", "")),
                    arguments=dict(arguments),
                )
            )
    text = "".join(text_parts)
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=(
                *thinking_parts,
                *((ContentPart(type="text", text=text),) if text else ()),
            ),
            tool_calls=tuple(tool_calls),
        ),
        model=_provider_value(value, "model"),
        usage=_anthropic_usage(_provider_value(value, "usage")),
        finish_reason=_finish_reason(
            _provider_value(value, "stop_reason"), anthropic=True
        ),
        raw=value,
    )
