"""Semantic and episodic memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ext.runtime.memory.long_term import (
    LongTermMemoryStore,
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from ext.runtime.types import JSONValue, MemoryScope, _validate_json_value


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


class SemanticMemory:
    def __init__(
        self,
        store: LongTermMemoryStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def write(
        self,
        *,
        memory_id: str,
        content: str,
        scope: MemoryScope = MemoryScope(),
        metadata: Mapping[str, JSONValue] | None = None,
        tags: Sequence[str] = (),
    ) -> MemoryRecord:
        embedding = tuple(
            float(value) for value in self.embedding_provider.embed(content)
        )
        return self.store.write(
            MemoryRecord(
                id=memory_id,
                kind="semantic",
                content=content,
                scope=scope,
                metadata=metadata or {},
                tags=tuple(tags),
                embedding=embedding,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> tuple[MemorySearchResult, ...]:
        query_vector = tuple(
            float(value) for value in self.embedding_provider.embed(query)
        )
        candidates = self.store.search(
            MemorySearchQuery(
                scope=scope,
                kinds=frozenset({"semantic"}),
                limit=1000000,
            )
        )
        ranked: list[MemorySearchResult] = []
        for candidate in candidates:
            embedding = candidate.record.embedding
            if embedding is None or len(embedding) != len(query_vector):
                continue
            ranked.append(
                MemorySearchResult(
                    record=candidate.record,
                    score=self._cosine(query_vector, embedding),
                )
            )
        ranked.sort(
            key=lambda item: (-item.score, -item.record.sequence, item.record.id)
        )
        return tuple(ranked[:limit])

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class Episode:
    id: str
    task: str
    outcome: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    decisions: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("episode id must not be empty")
        if not self.task.strip():
            raise ValueError("episode task must not be empty")
        if not self.outcome.strip():
            raise ValueError("episode outcome must not be empty")
        _validate_json_value(dict(self.metadata))


class EpisodicMemory:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store

    def remember(self, episode: Episode) -> MemoryRecord:
        payload: dict[str, JSONValue] = {
            "task": episode.task,
            "outcome": episode.outcome,
            "decisions": list(episode.decisions),
            "actions": list(episode.actions),
            "trace": list(episode.trace),
        }
        metadata = dict(episode.metadata)
        metadata["episode"] = payload
        searchable = " ".join(
            (
                episode.task,
                episode.outcome,
                *episode.decisions,
                *episode.actions,
                *episode.trace,
            )
        )
        return self.store.write(
            MemoryRecord(
                id=episode.id,
                kind="episode",
                content=searchable,
                scope=episode.scope,
                metadata=metadata,
                tags=("episode",),
            )
        )

    def search(
        self,
        text: str,
        *,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> tuple[MemorySearchResult, ...]:
        return self.store.search(
            MemorySearchQuery(
                text=text,
                scope=scope,
                kinds=frozenset({"episode"}),
                limit=limit,
            )
        )
