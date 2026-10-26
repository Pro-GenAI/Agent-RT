"""Browser and computer-use sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

BrowserAction = Literal[
    "navigate",
    "search",
    "click",
    "fill",
    "download",
    "extract",
    "back",
    "forward",
    "reload",
]


@dataclass(frozen=True)
class BrowserState:
    url: str | None = None
    title: str | None = None
    history: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class BrowserBackend(Protocol):
    async def perform(
        self,
        action: BrowserAction,
        arguments: Mapping[str, Any],
        state: BrowserState,
    ) -> tuple[Any, BrowserState]: ...


class BrowserSession:
    def __init__(
        self,
        backend: BrowserBackend,
        state: BrowserState = BrowserState(),
    ) -> None:
        self.backend = backend
        self.state = state

    async def perform(self, action: BrowserAction, **arguments: Any) -> Any:
        result, self.state = await self.backend.perform(action, arguments, self.state)
        return result


ComputerAction = Literal["screenshot", "move", "click", "scroll", "key", "type", "drag"]


@dataclass(frozen=True)
class ComputerState:
    width: int
    height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ComputerBackend(Protocol):
    async def perform(
        self,
        action: ComputerAction,
        arguments: Mapping[str, Any],
        state: ComputerState,
    ) -> Any: ...


class ComputerSession:
    def __init__(self, backend: ComputerBackend, state: ComputerState) -> None:
        if state.width <= 0 or state.height <= 0:
            raise ValueError("computer dimensions must be positive")
        self.backend = backend
        self.state = state

    async def perform(self, action: ComputerAction, **arguments: Any) -> Any:
        return await self.backend.perform(action, arguments, self.state)
