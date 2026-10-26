"""Leased work queues."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ext.runtime.types import JSONValue, _validate_json_value


@dataclass(frozen=True)
class WorkQueueItem:
    item_id: str
    payload: JSONValue
    priority: int = 0
    max_attempts: int = 3
    attempts: int = 0
    available_at_ms: int = 0
    lease_owner: str | None = None
    lease_expires_at_ms: int | None = None
    enqueued_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("queue item_id must not be empty")
        if self.max_attempts < 1:
            raise ValueError("queue max_attempts must be at least 1")
        if self.attempts < 0:
            raise ValueError("queue attempts must be non-negative")
        _validate_json_value(self.payload)


class InMemoryWorkQueue:
    def __init__(
        self,
        *,
        max_active_leases: int | None = None,
        max_leases_per_worker: int | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if max_active_leases is not None and max_active_leases < 1:
            raise ValueError("max_active_leases must be at least 1")
        if max_leases_per_worker is not None and max_leases_per_worker < 1:
            raise ValueError("max_leases_per_worker must be at least 1")
        self.max_active_leases = max_active_leases
        self.max_leases_per_worker = max_leases_per_worker
        self.clock = clock or (lambda: int(time.time() * 1000))
        self._items: dict[str, WorkQueueItem] = {}
        self._completed: set[str] = set()
        self._failed: set[str] = set()

    def contains(self, item_id: str) -> bool:
        """True if the item is queued, leased, completed, or failed."""
        return (
            item_id in self._items
            or item_id in self._completed
            or item_id in self._failed
        )

    def enqueue(self, item: WorkQueueItem) -> WorkQueueItem:
        if item.item_id in self._items or item.item_id in self._completed:
            raise ValueError(f"queue item already exists: {item.item_id}")
        self._items[item.item_id] = item
        return item

    def release_expired(self) -> int:
        now = self.clock()
        released = 0
        for item_id, item in tuple(self._items.items()):
            if (
                item.lease_owner is not None
                and item.lease_expires_at_ms is not None
                and item.lease_expires_at_ms <= now
            ):
                if item.attempts >= item.max_attempts:
                    # Every attempt crashed or timed out: dead-letter it instead
                    # of leaving an item that can never be leased again.
                    del self._items[item_id]
                    self._failed.add(item_id)
                    released += 1
                    continue
                self._items[item_id] = WorkQueueItem(
                    item_id=item.item_id,
                    payload=item.payload,
                    priority=item.priority,
                    max_attempts=item.max_attempts,
                    attempts=item.attempts,
                    available_at_ms=item.available_at_ms,
                    enqueued_at_ms=item.enqueued_at_ms,
                )
                released += 1
        return released

    def lease(
        self,
        worker_id: str,
        *,
        lease_ms: int,
        limit: int = 1,
    ) -> tuple[WorkQueueItem, ...]:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_ms < 1:
            raise ValueError("lease_ms must be positive")
        if limit < 1:
            raise ValueError("lease limit must be at least 1")
        self.release_expired()
        now = self.clock()
        active = [item for item in self._items.values() if item.lease_owner is not None]
        global_slots = (
            limit
            if self.max_active_leases is None
            else max(0, self.max_active_leases - len(active))
        )
        worker_active = sum(1 for item in active if item.lease_owner == worker_id)
        worker_slots = (
            limit
            if self.max_leases_per_worker is None
            else max(0, self.max_leases_per_worker - worker_active)
        )
        take = min(limit, global_slots, worker_slots)
        if take <= 0:
            return ()

        available = [
            item
            for item in self._items.values()
            if item.lease_owner is None
            and item.available_at_ms <= now
            and item.attempts < item.max_attempts
        ]
        available.sort(
            key=lambda item: (-item.priority, item.enqueued_at_ms, item.item_id)
        )
        leased: list[WorkQueueItem] = []
        for item in available[:take]:
            updated = WorkQueueItem(
                item_id=item.item_id,
                payload=item.payload,
                priority=item.priority,
                max_attempts=item.max_attempts,
                attempts=item.attempts + 1,
                available_at_ms=item.available_at_ms,
                lease_owner=worker_id,
                lease_expires_at_ms=now + lease_ms,
                enqueued_at_ms=item.enqueued_at_ms,
            )
            self._items[item.item_id] = updated
            leased.append(updated)
        return tuple(leased)

    def ack(self, item_id: str, worker_id: str) -> bool:
        item = self._items.get(item_id)
        if item is None or item.lease_owner != worker_id:
            return False
        self._items.pop(item_id)
        self._completed.add(item_id)
        return True

    def fail(
        self,
        item_id: str,
        worker_id: str,
        *,
        retry_delay_ms: int = 0,
    ) -> bool:
        if retry_delay_ms < 0:
            raise ValueError("retry_delay_ms must be non-negative")
        item = self._items.get(item_id)
        if item is None or item.lease_owner != worker_id:
            return False
        if item.attempts >= item.max_attempts:
            self._items.pop(item_id)
            self._failed.add(item_id)
            return True
        self._items[item_id] = WorkQueueItem(
            item_id=item.item_id,
            payload=item.payload,
            priority=item.priority,
            max_attempts=item.max_attempts,
            attempts=item.attempts,
            available_at_ms=self.clock() + retry_delay_ms,
            enqueued_at_ms=item.enqueued_at_ms,
        )
        return True

    def list(self) -> tuple[WorkQueueItem, ...]:
        return tuple(
            sorted(
                self._items.values(),
                key=lambda item: (-item.priority, item.enqueued_at_ms, item.item_id),
            )
        )
