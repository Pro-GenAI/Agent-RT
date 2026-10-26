"""Model routing policies and the model registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ext.runtime.agent.config import ModelSettings


@dataclass(frozen=True)
class ModelTarget:
    model: str
    provider: str | None = None


@dataclass(frozen=True)
class ModelDescriptor(ModelTarget):
    capabilities: frozenset[str] = frozenset()
    context_window: int | None = None
    cost_per_million_tokens: float | None = None
    latency_ms: float | None = None
    reasoning: bool = False


@dataclass(frozen=True)
class RoutingRequirements:
    capabilities: frozenset[str] = frozenset()
    min_context_window: int | None = None
    max_cost_per_million_tokens: float | None = None
    max_latency_ms: float | None = None
    reasoning: bool | None = None


@runtime_checkable
class ModelRoutingPolicy(Protocol):
    def select(
        self,
        candidates: Sequence[ModelDescriptor],
        requirements: RoutingRequirements,
    ) -> ModelDescriptor: ...


class FirstMatchRoutingPolicy:
    def select(
        self,
        candidates: Sequence[ModelDescriptor],
        requirements: RoutingRequirements,
    ) -> ModelDescriptor:
        for candidate in candidates:
            if not requirements.capabilities.issubset(candidate.capabilities):
                continue
            if requirements.min_context_window is not None and (
                candidate.context_window is None
                or candidate.context_window < requirements.min_context_window
            ):
                continue
            if requirements.max_cost_per_million_tokens is not None and (
                candidate.cost_per_million_tokens is None
                or candidate.cost_per_million_tokens
                > requirements.max_cost_per_million_tokens
            ):
                continue
            if requirements.max_latency_ms is not None and (
                candidate.latency_ms is None
                or candidate.latency_ms > requirements.max_latency_ms
            ):
                continue
            if (
                requirements.reasoning is not None
                and candidate.reasoning != requirements.reasoning
            ):
                continue
            return candidate
        raise LookupError("no model candidate satisfies routing requirements")


class ModelRegistry:
    def __init__(
        self,
        models: Sequence[ModelDescriptor] = (),
        aliases: Mapping[str, ModelTarget] | None = None,
    ) -> None:
        self._models = {(model.provider, model.model): model for model in models}
        self._aliases = dict(aliases or {})

    def register(self, descriptor: ModelDescriptor) -> None:
        self._models[(descriptor.provider, descriptor.model)] = descriptor

    def alias(self, name: str, target: ModelTarget) -> None:
        self._aliases[name] = target

    def resolve(self, target: ModelTarget) -> ModelTarget:
        aliased = self._aliases.get(target.model)
        if aliased is None:
            return target
        return ModelTarget(
            model=aliased.model,
            provider=(
                target.provider if target.provider is not None else aliased.provider
            ),
        )

    def candidates(self, settings: ModelSettings) -> tuple[ModelDescriptor, ...]:
        targets = (
            ModelTarget(settings.model, settings.provider),
            *settings.fallback_models,
        )
        resolved: list[ModelDescriptor] = []
        for target in targets:
            concrete = self.resolve(target)
            descriptor = self._models.get((concrete.provider, concrete.model))
            if descriptor is not None:
                resolved.append(descriptor)
        return tuple(resolved)

    def route(
        self,
        settings: ModelSettings,
        requirements: RoutingRequirements = RoutingRequirements(),
        policy: ModelRoutingPolicy | None = None,
    ) -> ModelDescriptor:
        return (policy or FirstMatchRoutingPolicy()).select(
            self.candidates(settings), requirements
        )
