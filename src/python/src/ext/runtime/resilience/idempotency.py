"""Idempotency records and stores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class IdempotencyRecord:
    scope: str
    key: str
    value: Any
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.scope.strip():
            raise ValueError("idempotency scope must not be empty")
        if not self.key.strip():
            raise ValueError("idempotency key must not be empty")


@runtime_checkable
class IdempotencyStore(Protocol):
    def get(self, scope: str, key: str) -> IdempotencyRecord | None: ...
    def put(self, record: IdempotencyRecord) -> IdempotencyRecord: ...


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], IdempotencyRecord] = {}

    def get(self, scope: str, key: str) -> IdempotencyRecord | None:
        return self._records.get((scope, key))

    def put(self, record: IdempotencyRecord) -> IdempotencyRecord:
        identity = (record.scope, record.key)
        existing = self._records.get(identity)
        if existing is not None:
            return existing
        self._records[identity] = record
        return record
