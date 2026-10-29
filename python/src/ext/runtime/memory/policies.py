"""Memory write and retrieval policies and scoped memory stores."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ext.runtime.memory.long_term import (
    LongTermMemoryStore,
    MemoryKind,
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from ext.runtime.types import JSONValue, MemoryScope, _validate_json_value


@dataclass(frozen=True)
class MemoryWriteCandidate:
    id: str
    kind: MemoryKind
    content: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    relevance: float = 1.0
    sensitivity: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("relevance", self.relevance),
            ("sensitivity", self.sensitivity),
            ("confidence", self.confidence),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        _validate_json_value(dict(self.metadata))


@dataclass(frozen=True)
class MemoryWriteDecision:
    persist: bool
    reason: str


@dataclass(frozen=True)
class MemoryWritePolicy:
    min_relevance: float = 0.5
    min_confidence: float = 0.5
    max_sensitivity: float = 0.5
    reject_duplicates: bool = True
    allowed_kinds: frozenset[MemoryKind] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("min_relevance", self.min_relevance),
            ("min_confidence", self.min_confidence),
            ("max_sensitivity", self.max_sensitivity),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def decide(
        self,
        candidate: MemoryWriteCandidate,
        store: LongTermMemoryStore,
    ) -> MemoryWriteDecision:
        if self.allowed_kinds is not None and candidate.kind not in self.allowed_kinds:
            return MemoryWriteDecision(False, "kind_not_allowed")
        if candidate.relevance < self.min_relevance:
            return MemoryWriteDecision(False, "relevance_below_threshold")
        if candidate.confidence < self.min_confidence:
            return MemoryWriteDecision(False, "confidence_below_threshold")
        if candidate.sensitivity > self.max_sensitivity:
            return MemoryWriteDecision(False, "sensitivity_above_threshold")
        if self.reject_duplicates:
            duplicates = store.search(
                MemorySearchQuery(
                    text=candidate.content,
                    scope=candidate.scope,
                    kinds=frozenset({candidate.kind}),
                    limit=10,
                )
            )
            normalized = " ".join(candidate.content.lower().split())
            if any(
                " ".join(result.record.content.lower().split()) == normalized
                for result in duplicates
            ):
                return MemoryWriteDecision(False, "duplicate")
        return MemoryWriteDecision(True, "accepted")

    def persist(
        self,
        candidate: MemoryWriteCandidate,
        store: LongTermMemoryStore,
    ) -> MemoryRecord | None:
        decision = self.decide(candidate, store)
        if not decision.persist:
            return None
        return store.write(
            MemoryRecord(
                id=candidate.id,
                kind=candidate.kind,
                content=candidate.content,
                scope=candidate.scope,
                metadata=candidate.metadata,
                tags=candidate.tags,
            )
        )


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    relevance_weight: float = 0.6
    recency_weight: float = 0.2
    confidence_weight: float = 0.2
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("relevance_weight", self.relevance_weight),
            ("recency_weight", self.recency_weight),
            ("confidence_weight", self.confidence_weight),
            ("min_confidence", self.min_confidence),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def search(
        self,
        store: LongTermMemoryStore,
        query: MemorySearchQuery,
    ) -> tuple[MemorySearchResult, ...]:
        candidates = store.search(
            MemorySearchQuery(
                text=query.text,
                scope=query.scope,
                kinds=query.kinds,
                tags=query.tags,
                limit=max(query.limit * 10, query.limit),
            )
        )
        if not candidates:
            return ()

        max_sequence = max(result.record.sequence for result in candidates) or 1
        ranked: list[MemorySearchResult] = []
        for result in candidates:
            confidence_value = result.record.metadata.get("confidence", 1.0)
            confidence = (
                float(confidence_value)
                if isinstance(confidence_value, (int, float))
                and not isinstance(confidence_value, bool)
                else 1.0
            )
            if confidence < self.min_confidence:
                continue
            recency = result.record.sequence / max_sequence
            score = (
                result.score * self.relevance_weight
                + recency * self.recency_weight
                + confidence * self.confidence_weight
            )
            ranked.append(MemorySearchResult(record=result.record, score=score))

        ranked.sort(
            key=lambda item: (-item.score, -item.record.sequence, item.record.id)
        )
        return tuple(ranked[: query.limit])


class ScopedMemoryStore:
    def __init__(self, store: LongTermMemoryStore, scope: MemoryScope) -> None:
        self.store = store
        self.scope = scope
        self._prefix = (
            json.dumps(
                ["memory", scope.tenant], ensure_ascii=False, separators=(",", ":")
            )
            + "::"
            if scope.tenant is not None
            else ""
        )

    def _assert_scope(self, scope: MemoryScope) -> None:
        if not self.scope.contains(scope):
            raise PermissionError("memory scope is outside the bound scope")

    def _key(self, memory_id: str) -> str:
        return self._prefix + memory_id

    def _visible(self, record: MemoryRecord | None) -> MemoryRecord | None:
        if record is None or not self.scope.contains(record.scope):
            return None
        if not record.id.startswith(self._prefix):
            return None
        return replace(record, id=record.id[len(self._prefix) :])

    def read(self, memory_id: str) -> MemoryRecord | None:
        return self._visible(self.store.read(self._key(memory_id)))

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self._assert_scope(record.scope)
        stored = self.store.write(replace(record, id=self._key(record.id)))
        return replace(stored, id=record.id)

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        if self.read(memory_id) is None:
            raise KeyError(f"memory not found: {memory_id}")
        self._assert_scope(record.scope)
        stored = self.store.update(
            self._key(memory_id), replace(record, id=self._key(record.id))
        )
        return replace(stored, id=record.id)

    def delete(self, memory_id: str) -> bool:
        if self.read(memory_id) is None:
            return False
        return self.store.delete(self._key(memory_id))

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        effective_scope = query.scope or self.scope
        if not self.scope.contains(effective_scope):
            raise PermissionError("memory query scope is outside the bound scope")
        results = self.store.search(replace(query, scope=effective_scope))
        return tuple(
            replace(result, record=record)
            for result in results
            if (record := self._visible(result.record)) is not None
        )
