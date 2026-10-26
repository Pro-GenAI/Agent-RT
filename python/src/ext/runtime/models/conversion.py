"""Message, media, and tool-call conversion shared by providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ext.runtime.types import ContentPart, FinishReason, ModelMessage, ToolCall


def _provider_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


# Provider-private state carried through the transcript, e.g. Anthropic
# thinking blocks that must be replayed verbatim on the next tool turn. Marked
# with this MIME type, it is never rendered as user-visible text or treated as
# structured output, and other providers drop it.
PROVIDER_STATE_MIME = "application/vnd.anthropic.thinking+json"


_MEDIA_PART_TYPES = frozenset({"image", "audio", "video", "pdf", "document", "file"})


def _is_provider_state(part: ContentPart) -> bool:
    return part.mime_type == PROVIDER_STATE_MIME


def _message_text(message: ModelMessage) -> str:
    fragments: list[str] = []
    for part in message.content:
        if _is_provider_state(part) or part.type in _MEDIA_PART_TYPES:
            continue
        if part.text is not None:
            fragments.append(part.text)
        elif part.data is not None:
            fragments.append(json.dumps(part.data, ensure_ascii=False, default=str))
    return "".join(fragments)


def _has_media(message: ModelMessage) -> bool:
    return any(part.type in _MEDIA_PART_TYPES for part in message.content)


def _media_reference(part: ContentPart) -> tuple[str | None, str | None, str | None]:
    """Return ``(url, base64, mime_type)`` for an image/audio/pdf/file part.

    ``data`` may be an http(s) URL, a ``data:`` URL, a base64 string, raw bytes,
    or a mapping with ``url``/``data``. Raises if the part cannot be sent.
    """
    import base64

    data = part.data
    if isinstance(data, Mapping):
        data = data.get("url") or data.get("data")
    mime = part.mime_type
    if isinstance(data, (bytes, bytearray)):
        return None, base64.b64encode(bytes(data)).decode("ascii"), mime
    if isinstance(data, str) and data:
        if data.startswith(("http://", "https://")):
            return data, None, mime
        if data.startswith("data:"):
            header, _, payload = data.partition(",")
            media = header[5:].split(";")[0] or mime
            return None, payload, media
        return None, data, mime
    raise ValueError(f"{part.type} content part has no usable data")


def _data_url(base64_payload: str, mime: str | None, part_type: str) -> str:
    if not mime:
        raise ValueError(f"{part_type} content part needs a mime_type for inline data")
    return f"data:{mime};base64,{base64_payload}"


def _openai_content_parts(message: ModelMessage) -> list[dict[str, Any]]:
    if message.role != "user":
        raise ValueError(
            f"OpenAI chat cannot carry media content in {message.role} messages"
        )
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if _is_provider_state(part):
            continue
        if part.type not in _MEDIA_PART_TYPES:
            text = _message_text(ModelMessage(role=message.role, content=(part,)))
            if text:
                blocks.append({"type": "text", "text": text})
            continue
        url, payload, mime = _media_reference(part)
        if part.type == "image":
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url or _data_url(payload or "", mime, "image")
                    },
                }
            )
        elif part.type == "audio":
            if payload is None:
                raise ValueError("OpenAI chat audio input must be inline base64 data")
            audio_format = {
                "audio/wav": "wav",
                "audio/x-wav": "wav",
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
            }.get((mime or "").lower())
            if audio_format is None:
                raise ValueError(
                    "OpenAI chat audio input supports audio/wav and audio/mpeg"
                )
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": payload, "format": audio_format},
                }
            )
        elif part.type in {"pdf", "document", "file"}:
            if payload is None:
                raise ValueError("OpenAI chat file input must be inline base64 data")
            blocks.append(
                {
                    "type": "file",
                    "file": {
                        "file_data": _data_url(
                            payload, mime or "application/pdf", part.type
                        ),
                        "filename": (
                            "document.pdf"
                            if (mime or "application/pdf") == "application/pdf"
                            else "document"
                        ),
                    },
                }
            )
        else:
            raise ValueError(f"OpenAI chat does not support {part.type} content")
    return blocks


def _anthropic_content_blocks(message: ModelMessage) -> list[dict[str, Any]]:
    if message.role not in {"user", "tool"}:
        raise ValueError(
            f"Anthropic cannot carry media content in {message.role} messages"
        )
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if _is_provider_state(part):
            continue
        if part.type not in _MEDIA_PART_TYPES:
            text = _message_text(ModelMessage(role=message.role, content=(part,)))
            if text:
                blocks.append({"type": "text", "text": text})
            continue
        if part.type in {"audio", "video"}:
            raise ValueError(f"Anthropic does not support {part.type} content")
        url, payload, mime = _media_reference(part)
        if url is not None:
            source: dict[str, Any] = {"type": "url", "url": url}
        else:
            if not mime:
                raise ValueError(
                    f"{part.type} content part needs a mime_type for inline data"
                )
            source = {"type": "base64", "media_type": mime, "data": payload}
        blocks.append(
            {"type": "image" if part.type == "image" else "document", "source": source}
        )
    return blocks


def _tool_arguments(value: str | None) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool call arguments must decode to a JSON object")
    return parsed


def _parse_tool_call_arguments(
    value: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Parse model-produced tool arguments without raising on malformed JSON.

    Models routinely emit truncated or invalid argument strings (for example
    when a reply is cut off at ``max_tokens``). That is a model-correctable
    failure, so it is returned as an error string instead of aborting the run.
    """
    try:
        return _tool_arguments(value), None
    except ValueError as exc:  # JSONDecodeError is a ValueError
        return {}, f"tool call arguments are not a valid JSON object: {exc}"


def _provider_tool_call(call_id: str, name: str, raw_arguments: Any) -> ToolCall:
    arguments, error = _parse_tool_call_arguments(
        raw_arguments
        if isinstance(raw_arguments, str) or raw_arguments is None
        else json.dumps(raw_arguments)
    )
    return ToolCall(call_id, name, arguments, argument_error=error)


def _finish_reason(
    value: str | None, *, anthropic: bool = False
) -> FinishReason | None:
    if value is None:
        return None
    if anthropic:
        anthropic_reasons: dict[str, FinishReason] = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        return anthropic_reasons.get(value, "other")
    reasons: dict[str, FinishReason] = {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
        "content_filter": "content_filter",
    }
    return reasons.get(value, "other")
