from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from typing import Any, AsyncIterator, Mapping


LOGGER = logging.getLogger(__name__)


class OpenAIWebSocketUnavailableError(RuntimeError):
    """Raised when the configured OpenAI-compatible endpoint cannot open Responses WebSocket mode."""


def _responses_websocket_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/responses"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/responses"
    raise ValueError("OpenAI base URL must use http or https")


def responses_websocket_supported(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float = 2.0,
) -> bool:
    """Probe whether an OpenAI-compatible base URL accepts Responses WebSocket upgrades."""

    try:
        from websockets.sync.client import connect

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        with connect(
            _responses_websocket_url(base_url),
            additional_headers=headers,
            open_timeout=timeout,
            close_timeout=timeout,
            max_size=None,
        ):
            return True
    except Exception:
        return False


_MEDIA_TYPES = frozenset({"image", "audio", "video", "pdf", "document", "file"})
_PROVIDER_STATE_MIME = "application/vnd.anthropic.thinking+json"


def _message_text(message: Any) -> str:
    parts = []
    for part in getattr(message, "content", ()) or ():
        if (
            getattr(part, "mime_type", None) == _PROVIDER_STATE_MIME
            or getattr(part, "type", None) in _MEDIA_TYPES
        ):
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        if getattr(part, "type", None) == "json":
            value = getattr(part, "data", None)
            if value is not None:
                parts.append(json.dumps(value, separators=(",", ":")))
    return "".join(parts)


def _input_content(message: Any) -> list[dict[str, Any]]:
    """Responses-API content list for a message that carries media parts."""
    from agent_rt import _data_url, _media_reference

    if getattr(message, "role", "user") != "user":
        raise ValueError("Responses input can only carry media in user messages")
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        part_type = getattr(part, "type", None)
        if getattr(part, "mime_type", None) == _PROVIDER_STATE_MIME:
            continue
        if part_type not in _MEDIA_TYPES:
            text = _message_text(type(message)(role=message.role, content=(part,)))
            if text:
                blocks.append({"type": "input_text", "text": text})
            continue
        url, payload, mime = _media_reference(part)
        if part_type == "image":
            blocks.append(
                {
                    "type": "input_image",
                    "image_url": url or _data_url(payload or "", mime, "image"),
                }
            )
        elif part_type in {"pdf", "document", "file"}:
            if url is not None:
                blocks.append({"type": "input_file", "file_url": url})
            else:
                blocks.append(
                    {
                        "type": "input_file",
                        "file_data": _data_url(
                            payload or "", mime or "application/pdf", part_type
                        ),
                        "filename": "document.pdf",
                    }
                )
        else:
            raise ValueError(f"OpenAI Responses input does not support {part_type} content")
    return blocks


def responses_input_items(messages: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", "user")
        text = _message_text(message)
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": getattr(message, "tool_call_id", None) or "",
                "output": text,
            })
            continue
        if any(getattr(part, "type", None) in _MEDIA_TYPES for part in message.content):
            items.append(
                {"type": "message", "role": role, "content": _input_content(message)}
            )
        elif text:
            items.append({"type": "message", "role": role, "content": text})
        for call in getattr(message, "tool_calls", ()) or ():
            items.append({
                "type": "function_call",
                "call_id": getattr(call, "id", ""),
                "name": getattr(call, "name", ""),
                "arguments": json.dumps(
                    dict(getattr(call, "arguments", {}) or {}),
                    separators=(",", ":"),
                ),
            })
    return items


def responses_payload(
    settings: Any,
    request: Any,
    messages: Any,
    *,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.resolve_model(getattr(request, "model", None)),
        "store": False,
        "input": responses_input_items(messages),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    tools = getattr(request, "tools", None) or ()
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            }
            for tool in tools
        ]

    selection = getattr(request, "tool_selection", None)
    required = getattr(selection, "required", ()) if selection is not None else ()
    if required:
        payload["tool_choice"] = (
            {"type": "function", "name": required[0]}
            if len(required) == 1
            else "required"
        )

    temperature = getattr(request, "temperature", None)
    if temperature is not None:
        payload["temperature"] = temperature
    max_output_tokens = getattr(request, "max_output_tokens", None)
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    reasoning = getattr(request, "reasoning", None)
    if reasoning is not None:
        reasoning_payload: dict[str, Any] = {}
        effort = getattr(reasoning, "effort", None)
        summary = getattr(reasoning, "summary", None)
        if effort is not None:
            reasoning_payload["effort"] = effort
        if summary is not None:
            reasoning_payload["summary"] = summary
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload

    structured = getattr(request, "structured_output", None)
    if (
        structured is not None
        and structured.schema == {"type": "object"}
        and not structured.strict
        and structured.name is None
    ):
        # JSON mode (agent_rt.JSON_OBJECT_OUTPUT).
        payload["text"] = {"format": {"type": "json_object"}}
    elif structured is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": structured.name or "response",
                "schema": dict(structured.schema),
                "strict": structured.strict,
            }
        }
    return payload


class OpenAIResponsesWebSocketTransport:
    """Persistent low-level transport for OpenAI Responses WebSocket mode."""

    def __init__(self, *, base_url: str, api_key: str | None) -> None:
        self.url = _responses_websocket_url(base_url)
        self.api_key = api_key
        self._connection: Any = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._legacy_stream_lock = asyncio.Lock()
        self._reader_task: Any = None
        self._streams: dict[str, Any] = {}
        self._stream_counter = 0
        self._stream_id_supported: bool | None = None

    async def _connect(self) -> Any:
        connection = self._connection
        if connection is not None:
            return connection
        async with self._connect_lock:
            connection = self._connection
            if connection is not None:
                return connection
            from websockets.asyncio.client import connect

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                connection = await connect(
                    self.url,
                    additional_headers=headers or None,
                    max_size=None,
                )
            except Exception as exc:
                raise OpenAIWebSocketUnavailableError(
                    f"OpenAI Responses WebSocket unavailable at {self.url}"
                ) from exc
            self._connection = connection
            self._reader_task = asyncio.create_task(self._reader(connection))
            return connection

    async def _reader(self, connection: Any) -> None:
        failure: Exception | None = None
        try:
            async for raw in connection:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                event = json.loads(raw)
                stream_id = event.get("stream_id")
                if isinstance(stream_id, str):
                    queue = self._streams.get(stream_id)
                    if queue is not None:
                        queue.put_nowait(event)
                    continue
                if event.get("type") == "error":
                    for queue in tuple(self._streams.values()):
                        queue.put_nowait(event)
                elif len(self._streams) == 1:
                    next(iter(self._streams.values())).put_nowait(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
        finally:
            if self._connection is connection:
                self._connection = None
            if self._reader_task is asyncio.current_task():
                self._reader_task = None
            if failure is None:
                failure = RuntimeError("OpenAI WebSocket closed before response.completed")
            for queue in tuple(self._streams.values()):
                queue.put_nowait(failure)

    async def _drop_connection(self) -> None:
        connection = self._connection
        reader_task = self._reader_task
        self._connection = None
        self._reader_task = None
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                LOGGER.debug("failed to close websocket connection", exc_info=True)
        if reader_task is not None and reader_task is not asyncio.current_task():
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.debug("websocket reader failed during cleanup", exc_info=True)

    async def close(self) -> None:
        await self._drop_connection()

    async def _events_once(
        self,
        payload: Mapping[str, Any],
        *,
        include_stream_id: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        connection = await self._connect()
        self._stream_counter += 1
        stream_id = f"agent-rt-{self._stream_counter}"
        queue = asyncio.Queue()
        self._streams[stream_id] = queue
        request = {"type": "response.create", **dict(payload)}
        if include_stream_id:
            request["stream_id"] = stream_id
        finished = False
        try:
            async with self._send_lock:
                await connection.send(json.dumps(request, separators=(",", ":")))
            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                event = item
                event_type = event.get("type")
                if event_type == "error":
                    raise RuntimeError(f"OpenAI WebSocket error: {event!r}")
                if event_type == "response.failed":
                    raise RuntimeError(f"OpenAI WebSocket response failed: {event!r}")
                yield event
                # An incomplete response (for example max_output_tokens) is a
                # normal truncated result, not a transport failure; treating it
                # as one would also tear down the shared connection.
                if event_type in {"response.completed", "response.incomplete"}:
                    finished = True
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._connection is connection:
                await self._drop_connection()
            raise
        finally:
            self._streams.pop(stream_id, None)
            # Without stream ids, leftover events of an abandoned response would
            # be routed to the next request on this connection.
            if not finished and not include_stream_id and self._connection is connection:
                await asyncio.shield(self._drop_connection())

    async def events(
        self,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Send one response.create and yield matching events."""

        if self._stream_id_supported is False:
            async with self._legacy_stream_lock:
                async with aclosing(
                    self._events_once(payload, include_stream_id=False)
                ) as inner:
                    async for event in inner:
                        yield event
            return

        try:
            async with aclosing(
                self._events_once(payload, include_stream_id=True)
            ) as inner:
                async for event in inner:
                    yield event
            self._stream_id_supported = True
            return
        except RuntimeError as exc:
            if "Unsupported parameter: stream_id" not in str(exc):
                raise
            self._stream_id_supported = False

        async with self._legacy_stream_lock:
            async with aclosing(
                self._events_once(payload, include_stream_id=False)
            ) as inner:
                async for event in inner:
                    yield event


async def stream_model_response(provider: Any, request: Any) -> AsyncIterator[Any]:
    from agent_rt import ContentPart, ModelMessage, ModelResponse, ModelStreamEvent, ModelUsage, _provider_tool_call

    transport = provider._websocket_transport
    if transport is None:
        credentials = {"api" + "_key": getattr(provider.settings, "api" + "_key")}
        transport = OpenAIResponsesWebSocketTransport(
            base_url=provider.settings.base_url,
            **credentials,
        )
        provider._websocket_transport = transport

    previous_response_id = None
    messages = request.messages
    expected = provider._websocket_expected_prefix
    if (
        provider._websocket_previous_response_id
        and expected
        and len(messages) >= len(expected)
        and all(messages[index] is expected[index] for index in range(len(expected)))
    ):
        previous_response_id = provider._websocket_previous_response_id
        messages = messages[len(expected):]

    # The continuation is only valid for one successful round trip. Clear it now
    # and re-arm it after a completed response, so a failure, cancellation or
    # reconnect never replays a stale id the server can no longer resolve.
    provider._websocket_previous_response_id = None
    provider._websocket_expected_prefix = ()

    payload = responses_payload(
        provider.settings,
        request,
        messages,
        previous_response_id=previous_response_id,
    )

    text_parts: list[str] = []
    tool_state: dict[int, dict[str, Any]] = {}
    final_event: dict[str, Any] | None = None

    async for event in transport.events(payload):
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                text_parts.append(delta)
                yield ModelStreamEvent(type="text_delta", text=delta, raw=event)
            continue

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                index = int(event.get("output_index", 0))
                tool_state[index] = {
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "",
                }
            continue

        if event_type == "response.function_call_arguments.delta":
            index = int(event.get("output_index", 0))
            state = tool_state.setdefault(index, {"id": "", "name": "", "arguments": ""})
            delta = event.get("delta")
            if isinstance(delta, str):
                state["arguments"] += delta
            yield ModelStreamEvent(
                type="tool_call_delta",
                tool_call_id=state["id"] or None,
                tool_name=state["name"] or None,
                arguments_delta=delta if isinstance(delta, str) else None,
                raw=event,
            )
            continue

        if event_type in {"response.completed", "response.incomplete"}:
            final_event = event

    if final_event is None:
        raise RuntimeError("OpenAI WebSocket closed before response.completed")

    raw_response = final_event.get("response") or {}
    output = raw_response.get("output") or []
    if not text_parts:
        for item in output:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

    for index, item in enumerate(output):
        if item.get("type") != "function_call":
            continue
        state = tool_state.setdefault(index, {"id": "", "name": "", "arguments": ""})
        state["id"] = item.get("call_id") or item.get("id") or state["id"]
        state["name"] = item.get("name") or state["name"]
        state["arguments"] = item.get("arguments") or state["arguments"]

    tool_calls = tuple(
        _provider_tool_call(state["id"], state["name"], state["arguments"])
        for _, state in sorted(tool_state.items())
    )
    usage_value = raw_response.get("usage") or {}
    input_details = usage_value.get("input_tokens_details") or {}
    output_details = usage_value.get("output_tokens_details") or {}
    usage = ModelUsage(
        input_tokens=usage_value.get("input_tokens"),
        output_tokens=usage_value.get("output_tokens"),
        total_tokens=usage_value.get("total_tokens"),
        cached_tokens=input_details.get("cached_tokens"),
        reasoning_tokens=output_details.get("reasoning_tokens"),
    )
    response = ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=(ContentPart(type="text", text="".join(text_parts)),) if text_parts else (),
            tool_calls=tool_calls,
        ),
        model=raw_response.get("model"),
        usage=usage,
        finish_reason=(
            "tool_calls"
            if tool_calls
            else "length"
            if final_event.get("type") == "response.incomplete"
            else "stop"
        ),
        raw=final_event,
    )

    response_id = raw_response.get("id")
    if isinstance(response_id, str) and response_id:
        provider._websocket_previous_response_id = response_id
        provider._websocket_expected_prefix = tuple(request.messages) + (response.message,)
    else:
        provider._websocket_previous_response_id = None
        provider._websocket_expected_prefix = ()

    yield ModelStreamEvent(type="completed", response=response, raw=final_event)

