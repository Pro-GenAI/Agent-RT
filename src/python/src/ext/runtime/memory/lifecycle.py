"""Memory lifecycle policies and procedural memory."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ext.runtime.memory.long_term import (
    LongTermMemoryStore,
    MemoryKind,
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from ext.runtime.types import JSONValue, MemoryScope, _validate_json_value

MemoryMigration = Callable[[MemoryRecord], MemoryRecord]


@dataclass(frozen=True)
class MemoryLifecyclePolicy:
    default_retention_ms: int | None = None

    def __post_init__(self) -> None:
        if self.default_retention_ms is not None and self.default_retention_ms < 1:
            raise ValueError("default_retention_ms must be positive")


class LifecycleMemoryStore:
    def __init__(
        self,
        store: LongTermMemoryStore,
        *,
        policy: MemoryLifecyclePolicy | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryLifecyclePolicy()
        self.clock = clock or (lambda: int(time.time() * 1000))
        self._migrations: dict[int, MemoryMigration] = {}

    def register_migration(
        self,
        from_version: int,
        migration: MemoryMigration,
    ) -> None:
        if from_version < 1:
            raise ValueError("migration version must be at least 1")
        self._migrations[from_version] = migration

    def _expired(self, record: MemoryRecord) -> bool:
        return record.expires_at_ms is not None and record.expires_at_ms <= self.clock()

    def _apply_default_retention(self, record: MemoryRecord) -> MemoryRecord:
        if record.expires_at_ms is not None or self.policy.default_retention_ms is None:
            return record
        return MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=record.sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.created_at_ms + self.policy.default_retention_ms,
            schema_version=record.schema_version,
        )

    def read(self, memory_id: str) -> MemoryRecord | None:
        record = self.store.read(memory_id)
        if record is None:
            return None
        if self._expired(record):
            self.store.delete(memory_id)
            return None
        return record

    def write(self, record: MemoryRecord) -> MemoryRecord:
        return self.store.write(self._apply_default_retention(record))

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        return self.store.update(
            memory_id,
            self._apply_default_retention(record),
        )

    def delete(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        self.purge_expired()
        return self.store.search(query)

    def purge_expired(self) -> int:
        expired = [
            result.record.id
            for result in self.store.search(MemorySearchQuery(limit=1_000_000))
            if self._expired(result.record)
        ]
        for memory_id in expired:
            self.store.delete(memory_id)
        return len(expired)

    def migrate(self, memory_id: str, *, target_version: int) -> MemoryRecord:
        record = self.read(memory_id)
        if record is None:
            raise KeyError(f"memory not found: {memory_id}")
        if target_version < record.schema_version:
            raise ValueError("memory migrations cannot move backwards")
        while record.schema_version < target_version:
            migration = self._migrations.get(record.schema_version)
            if migration is None:
                raise ValueError(
                    f"no migration registered from version {record.schema_version}"
                )
            migrated = migration(record)
            if migrated.id != record.id:
                raise ValueError("memory migration cannot change id")
            if migrated.schema_version != record.schema_version + 1:
                raise ValueError("memory migration must advance exactly one version")
            record = migrated
        return self.store.update(memory_id, record)

    def compact(
        self,
        memory_ids: Sequence[str],
        *,
        compacted_id: str,
        content: str,
        kind: MemoryKind = "fact",
        scope: MemoryScope = MemoryScope(),
        metadata: Mapping[str, JSONValue] | None = None,
        tags: Sequence[str] = (),
        delete_sources: bool = True,
    ) -> MemoryRecord:
        if not memory_ids:
            raise ValueError("memory compaction requires at least one source")
        sources = []
        for memory_id in memory_ids:
            record = self.read(memory_id)
            if record is None:
                raise KeyError(f"memory not found: {memory_id}")
            sources.append(record)
        compacted_metadata = dict(metadata or {})
        compacted_metadata["compacted_from"] = list(memory_ids)
        result = self.write(
            MemoryRecord(
                id=compacted_id,
                kind=kind,
                content=content,
                scope=scope,
                metadata=compacted_metadata,
                tags=tuple(tags),
            )
        )
        if delete_sources:
            for memory_id in memory_ids:
                self.store.delete(memory_id)
        return result


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    instructions: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    script: str | None = None
    template: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("procedure id must not be empty")
        if not self.name.strip():
            raise ValueError("procedure name must not be empty")
        if not self.instructions.strip():
            raise ValueError("procedure instructions must not be empty")
        _validate_json_value(dict(self.metadata))


class ProceduralMemory:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store

    def remember(self, procedure: Procedure) -> MemoryRecord:
        payload: dict[str, JSONValue] = {
            "name": procedure.name,
            "instructions": procedure.instructions,
            "script": procedure.script,
            "template": procedure.template,
        }
        metadata = dict(procedure.metadata)
        metadata["procedure"] = payload
        searchable = " ".join(
            part
            for part in (
                procedure.name,
                procedure.instructions,
                procedure.script or "",
                procedure.template or "",
            )
            if part
        )
        return self.store.write(
            MemoryRecord(
                id=procedure.id,
                kind="procedure",
                content=searchable,
                scope=procedure.scope,
                metadata=metadata,
                tags=("procedure", *procedure.tags),
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
                kinds=frozenset({"procedure"}),
                limit=limit,
            )
        )
