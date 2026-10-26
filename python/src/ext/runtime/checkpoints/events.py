"""Durable event logs, retention, and tenant-scoped event stores."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, runtime_checkable

from ext.runtime.authorization.grants import TenantContext
from ext.runtime.guardrails.privacy import PrivacyRedactor
from ext.runtime.types import JSONValue, _validate_json_value

DurableEventType = Literal[
    "state_changed",
    "model_requested",
    "model_completed",
    "tool_requested",
    "tool_completed",
    "approval_requested",
    "approval_resolved",
    "lifecycle_transition",
    "checkpoint_saved",
]


@dataclass(frozen=True)
class DurableEvent:
    event_id: str
    task_id: str
    type: DurableEventType
    payload: Mapping[str, JSONValue] = field(default_factory=dict)
    sequence: int = 0
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        _validate_json_value(dict(self.payload))


@runtime_checkable
class EventStore(Protocol):
    def append(self, event: DurableEvent) -> DurableEvent: ...
    def list(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[DurableEvent, ...]: ...


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[DurableEvent]] = {}
        self._event_ids: set[str] = set()

    def append(self, event: DurableEvent) -> DurableEvent:
        if event.event_id in self._event_ids:
            raise ValueError(f"event already exists: {event.event_id}")
        stream = self._events.setdefault(event.task_id, [])
        stored = DurableEvent(
            event_id=event.event_id,
            task_id=event.task_id,
            type=event.type,
            payload=event.payload,
            sequence=len(stream) + 1,
            occurred_at_ms=event.occurred_at_ms,
        )
        stream.append(stored)
        self._event_ids.add(stored.event_id)
        return stored

    def list(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[DurableEvent, ...]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return tuple(
            event
            for event in self._events.get(task_id, ())
            if event.sequence > after_sequence
        )


@dataclass(frozen=True)
class EventRetentionPolicy:
    ttl_ms: int | None = None
    archive_after_ms: int | None = None
    ephemeral: bool = False
    suppress_event_types: frozenset[DurableEventType] = frozenset()

    def __post_init__(self) -> None:
        if self.ttl_ms is not None and self.ttl_ms < 1:
            raise ValueError("ttl_ms must be positive")
        if self.archive_after_ms is not None and self.archive_after_ms < 1:
            raise ValueError("archive_after_ms must be positive")


class RetainedEventStore:
    def __init__(
        self,
        policy: EventRetentionPolicy = EventRetentionPolicy(),
        *,
        redactor: PrivacyRedactor | None = None,
    ) -> None:
        self.policy = policy
        self.redactor = redactor or PrivacyRedactor()
        self._events: dict[str, list[DurableEvent]] = {}
        self._archive: dict[str, list[DurableEvent]] = {}
        self._event_ids: set[str] = set()
        # Sequences are per task and strictly increasing for the store's
        # lifetime. Deriving them from the visible stream would reuse numbers
        # once archival or expiry shrinks it, hiding new events from readers
        # that resume with ``after_sequence``.
        self._last_sequence: dict[str, int] = {}

    def append(self, event: DurableEvent) -> DurableEvent:
        payload = self.redactor.redact(dict(event.payload))
        sanitized = DurableEvent(
            event_id=event.event_id,
            task_id=event.task_id,
            type=event.type,
            payload=payload,
            sequence=event.sequence,
            occurred_at_ms=event.occurred_at_ms,
        )
        if self.policy.ephemeral or event.type in self.policy.suppress_event_types:
            return sanitized
        if event.event_id in self._event_ids:
            raise ValueError(f"event already exists: {event.event_id}")
        stream = self._events.setdefault(event.task_id, [])
        sequence = self._last_sequence.get(event.task_id, 0) + 1
        self._last_sequence[event.task_id] = sequence
        stored = replace(sanitized, sequence=sequence)
        stream.append(stored)
        self._event_ids.add(stored.event_id)
        return stored

    def list(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
        now_ms: int | None = None,
    ) -> tuple[DurableEvent, ...]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        now = int(time.time() * 1000) if now_ms is None else now_ms
        ttl = self.policy.ttl_ms
        return tuple(
            event
            for event in self._events.get(task_id, ())
            if event.sequence > after_sequence
            and (ttl is None or now - event.occurred_at_ms < ttl)
        )

    def archive_due(self, *, now_ms: int | None = None) -> int:
        if self.policy.archive_after_ms is None:
            return 0
        now = int(time.time() * 1000) if now_ms is None else now_ms
        moved = 0
        for task_id, events in list(self._events.items()):
            keep: list[DurableEvent] = []
            for event in events:
                if now - event.occurred_at_ms >= self.policy.archive_after_ms:
                    self._archive.setdefault(task_id, []).append(event)
                    moved += 1
                else:
                    keep.append(event)
            self._events[task_id] = keep
        return moved

    def archived(self, task_id: str) -> tuple[DurableEvent, ...]:
        return tuple(self._archive.get(task_id, ()))

    def purge_expired(self, *, now_ms: int | None = None) -> int:
        if self.policy.ttl_ms is None:
            return 0
        now = int(time.time() * 1000) if now_ms is None else now_ms
        removed = 0
        for bucket in (self._events, self._archive):
            for task_id, events in list(bucket.items()):
                keep = [
                    event
                    for event in events
                    if now - event.occurred_at_ms < self.policy.ttl_ms
                ]
                removed += len(events) - len(keep)
                bucket[task_id] = keep
        return removed


class TenantEventStore:
    def __init__(self, store: EventStore, tenant: TenantContext) -> None:
        self.store = store
        self.tenant = tenant

    def append(self, event: DurableEvent) -> DurableEvent:
        return self.store.append(
            replace(
                event,
                event_id=self.tenant.qualify("event", event.event_id),
                task_id=self.tenant.task_id(event.task_id),
            )
        )

    def list(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[DurableEvent, ...]:
        return self.store.list(
            self.tenant.task_id(task_id),
            after_sequence=after_sequence,
        )
