"""Debug replay bundles and deterministic replay providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ext.runtime.models.interfaces import ModelRequest, ModelResponse


@dataclass(frozen=True)
class ReplayExchange:
    channel: Literal["model", "tool", "external"]
    key: str
    input: Any
    output: Any


@dataclass(frozen=True)
class ReplayBundle:
    replay_id: str
    state: Any = None
    events: tuple[Any, ...] = ()
    checkpoints: tuple[Any, ...] = ()
    exchanges: tuple[ReplayExchange, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DebugReplayStore:
    def __init__(self) -> None:
        self._bundles: dict[str, ReplayBundle] = {}

    def save(self, bundle: ReplayBundle) -> ReplayBundle:
        self._bundles[bundle.replay_id] = bundle
        return bundle

    def load(self, replay_id: str) -> ReplayBundle:
        if replay_id not in self._bundles:
            raise KeyError(replay_id)
        return self._bundles[replay_id]

    def reconstruct(self, replay_id: str) -> Mapping[str, Any]:
        bundle = self.load(replay_id)
        return {
            "state": bundle.state,
            "events": bundle.events,
            "checkpoints": bundle.checkpoints,
            "exchanges": bundle.exchanges,
            "metadata": bundle.metadata,
        }


class DeterministicReplay:
    def __init__(self, bundle: ReplayBundle) -> None:
        self.bundle = bundle
        self._positions: dict[tuple[str, str], int] = {}

    def next(
        self,
        channel: Literal["model", "tool", "external"],
        key: str,
        input: Any = None,
    ) -> Any:
        matches = [
            exchange
            for exchange in self.bundle.exchanges
            if exchange.channel == channel and exchange.key == key
        ]
        position_key = (channel, key)
        position = self._positions.get(position_key, 0)
        if position >= len(matches):
            raise LookupError(f"no recorded replay output for {channel}:{key}")
        exchange = matches[position]
        if input is not None and exchange.input != input:
            raise ValueError(f"replay input mismatch for {channel}:{key}")
        self._positions[position_key] = position + 1
        return exchange.output


class ReplayModelProvider:
    def __init__(
        self,
        replay: DeterministicReplay,
        *,
        key: str = "complete",
        name: str = "replay",
    ) -> None:
        self.replay = replay
        self.key = key
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        output = self.replay.next("model", self.key)
        if not isinstance(output, ModelResponse):
            raise TypeError("recorded model replay output must be ModelResponse")
        return output
