"""Long-term memory records, queries, and the in-memory store."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from ext.runtime.types import JSONValue, MemoryScope, _validate_json_value

MemoryKind = Literal["fact", "semantic", "episode", "procedure"]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    kind: MemoryKind
    content: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    sequence: int = 0
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at_ms: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("memory id must not be empty")
        if not self.content.strip():
            raise ValueError("memory content must not be empty")
        _validate_json_value(dict(self.metadata))
        if self.created_at_ms < 0:
            raise ValueError("memory created_at_ms must be non-negative")
        if self.expires_at_ms is not None and self.expires_at_ms < self.created_at_ms:
            raise ValueError("memory expires_at_ms must not precede created_at_ms")
        if self.schema_version < 1:
            raise ValueError("memory schema_version must be at least 1")
        if self.embedding is not None:
            if not self.embedding:
                raise ValueError("memory embedding must not be empty")
            for value in self.embedding:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("memory embedding values must be numbers")
                if not float("-inf") < float(value) < float("inf"):
                    raise ValueError("memory embedding values must be finite")


@dataclass(frozen=True)
class MemorySearchQuery:
    text: str | None = None
    scope: MemoryScope | None = None
    kinds: frozenset[MemoryKind] | None = None
    tags: frozenset[str] = frozenset()
    limit: int = 10

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("memory search limit must be non-negative")


@dataclass(frozen=True)
class MemorySearchResult:
    record: MemoryRecord
    score: float


@runtime_checkable
class LongTermMemoryStore(Protocol):
    def read(self, memory_id: str) -> MemoryRecord | None: ...
    def write(self, record: MemoryRecord) -> MemoryRecord: ...
    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord: ...
    def delete(self, memory_id: str) -> bool: ...
    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]: ...


class InMemoryLongTermMemoryStore:
    _token_pattern = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._sequence = 0

    def read(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        if record.id in self._records:
            raise ValueError(f"memory already exists: {record.id}")
        self._sequence += 1
        stored = MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=self._sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.expires_at_ms,
            schema_version=record.schema_version,
        )
        self._records[stored.id] = stored
        return stored

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        if memory_id not in self._records:
            raise KeyError(f"memory not found: {memory_id}")
        if record.id != memory_id:
            raise ValueError("memory update cannot change id")
        previous = self._records[memory_id]
        stored = MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=previous.sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.expires_at_ms,
            schema_version=record.schema_version,
        )
        self._records[memory_id] = stored
        return stored

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        query_terms = self._terms(query.text or "")
        results: list[MemorySearchResult] = []
        for record in self._records.values():
            if query.scope is not None and not query.scope.contains(record.scope):
                continue
            if query.kinds is not None and record.kind not in query.kinds:
                continue
            if query.tags and not query.tags.issubset(record.tags):
                continue

            if query_terms:
                record_terms = self._terms(
                    record.content
                    + " "
                    + " ".join(record.tags)
                    + " "
                    + " ".join(
                        f"{key} {value}"
                        for key, value in record.metadata.items()
                        if isinstance(value, (str, int, float, bool))
                    )
                )
                overlap = query_terms & record_terms
                if not overlap:
                    continue
                score = len(overlap) / len(query_terms)
            else:
                score = 1.0
            results.append(MemorySearchResult(record=record, score=score))

        results.sort(
            key=lambda item: (-item.score, -item.record.sequence, item.record.id)
        )
        return tuple(results[: query.limit])

    def _terms(self, value: str) -> set[str]:
        return set(self._token_pattern.findall(value.lower()))
