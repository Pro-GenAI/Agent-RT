"""Multimodal messages and realtime sessions."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.types import ContentPart, MessageRole, ModelMessage


@dataclass(frozen=True)
class MultimodalMessage:
    role: MessageRole
    parts: tuple[ContentPart, ...]

    def to_model_message(self) -> ModelMessage:
        return ModelMessage(role=self.role, content=self.parts)


RealtimeEventType = Literal[
    "audio_input",
    "audio_output",
    "speech_started",
    "speech_stopped",
    "text_delta",
    "response_completed",
    "interrupted",
]


@dataclass(frozen=True)
class RealtimeEvent:
    type: RealtimeEventType
    data: Any = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@runtime_checkable
class RealtimeTransport(Protocol):
    async def send(self, event: RealtimeEvent) -> None: ...
    def events(self) -> AsyncIterator[RealtimeEvent]: ...
    async def close(self) -> None: ...


class RealtimeSession:
    def __init__(self, transport: RealtimeTransport) -> None:
        self.transport = transport
        self.interrupted = False

    async def send_audio(self, data: Any) -> None:
        await self.transport.send(RealtimeEvent("audio_input", data))

    async def send_text(self, text: str) -> None:
        await self.transport.send(RealtimeEvent("text_delta", text))

    async def interrupt(self) -> None:
        self.interrupted = True
        await self.transport.send(RealtimeEvent("interrupted"))

    async def collect_until_complete(self) -> tuple[RealtimeEvent, ...]:
        collected: list[RealtimeEvent] = []
        async for event in self.transport.events():
            collected.append(event)
            if event.type in ("response_completed", "interrupted"):
                break
        return tuple(collected)

    async def close(self) -> None:
        await self.transport.close()
