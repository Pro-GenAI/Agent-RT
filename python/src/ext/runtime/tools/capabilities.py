"""Capability descriptors, scoring, and the searchable capability catalog."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

CapabilityKind = Literal["tool", "skill", "connector", "file", "agent"]


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    kind: CapabilityKind
    name: str
    description: str = ""
    namespace: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilitySearchResult:
    capability: CapabilityDescriptor
    score: float


@runtime_checkable
class CapabilityScorer(Protocol):
    def score(self, query: str, capability: CapabilityDescriptor) -> float: ...


class LexicalCapabilityScorer:
    _token_pattern = re.compile(r"[a-z0-9]+")

    def score(self, query: str, capability: CapabilityDescriptor) -> float:
        query_terms = self._terms(query)
        if not query_terms:
            return 0.0
        haystack = " ".join(
            part
            for part in (
                capability.name,
                capability.namespace or "",
                capability.description,
                " ".join(
                    f"{key} {value}"
                    for key, value in capability.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                ),
            )
            if part
        )
        terms = self._terms(haystack)
        if not terms:
            return 0.0
        overlap = query_terms & terms
        if not overlap:
            return 0.0
        score = len(overlap) / len(query_terms)
        name_terms = self._terms(capability.name)
        if query_terms <= name_terms:
            score += 0.25
        return score

    def _terms(self, value: str) -> set[str]:
        return set(self._token_pattern.findall(value.lower()))


class CapabilityCatalog:
    def __init__(
        self,
        capabilities: Sequence[CapabilityDescriptor] = (),
        *,
        scorer: CapabilityScorer | None = None,
    ) -> None:
        self._capabilities: dict[str, CapabilityDescriptor] = {}
        self.scorer = scorer or LexicalCapabilityScorer()
        for capability in capabilities:
            self.register(capability)

    def register(
        self,
        capability: CapabilityDescriptor,
        *,
        replace: bool = False,
    ) -> None:
        if not capability.id.strip():
            raise ValueError("capability id must not be empty")
        if not capability.name.strip():
            raise ValueError("capability name must not be empty")
        if capability.id in self._capabilities and not replace:
            raise ValueError(f"capability already registered: {capability.id}")
        self._capabilities[capability.id] = capability

    def unregister(self, capability_id: str) -> CapabilityDescriptor:
        try:
            return self._capabilities.pop(capability_id)
        except KeyError as exc:
            raise KeyError(f"capability not registered: {capability_id}") from exc

    def search(
        self,
        query: str,
        *,
        kinds: frozenset[CapabilityKind] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[CapabilitySearchResult, ...]:
        if limit < 0:
            raise ValueError("capability search limit must be non-negative")
        results = []
        for capability in self._capabilities.values():
            if kinds is not None and capability.kind not in kinds:
                continue
            score = self.scorer.score(query, capability)
            if score > min_score:
                results.append(CapabilitySearchResult(capability, score))
        results.sort(
            key=lambda result: (
                -result.score,
                result.capability.kind,
                result.capability.id,
            )
        )
        return tuple(results[:limit])
