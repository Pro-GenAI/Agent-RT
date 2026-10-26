"""Scheduled tasks and external event triggers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ext.runtime.tasks.queue import InMemoryWorkQueue, WorkQueueItem
from ext.runtime.types import JSONValue, _validate_json_value


@dataclass(frozen=True)
class ScheduledTask:
    schedule_id: str
    payload: JSONValue
    next_run_at_ms: int
    interval_ms: int | None = None
    max_runs: int | None = None
    runs: int = 0

    def __post_init__(self) -> None:
        if not self.schedule_id.strip():
            raise ValueError("schedule_id must not be empty")
        if self.next_run_at_ms < 0:
            raise ValueError("next_run_at_ms must be non-negative")
        if self.interval_ms is not None and self.interval_ms < 1:
            raise ValueError("interval_ms must be positive")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        _validate_json_value(self.payload)


class InMemoryScheduler:
    def __init__(self) -> None:
        self._scheduled: dict[str, ScheduledTask] = {}

    def schedule(self, task: ScheduledTask) -> ScheduledTask:
        if task.schedule_id in self._scheduled:
            raise ValueError(f"schedule already exists: {task.schedule_id}")
        self._scheduled[task.schedule_id] = task
        return task

    def cancel(self, schedule_id: str) -> bool:
        return self._scheduled.pop(schedule_id, None) is not None

    def get(self, schedule_id: str) -> ScheduledTask | None:
        return self._scheduled.get(schedule_id)

    def due(self, now_ms: int) -> tuple[ScheduledTask, ...]:
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")
        due = [
            task for task in self._scheduled.values() if task.next_run_at_ms <= now_ms
        ]
        due.sort(key=lambda task: (task.next_run_at_ms, task.schedule_id))
        emitted: list[ScheduledTask] = []
        for task in due:
            emitted.append(task)
            runs = task.runs + 1
            if task.interval_ms is None or (
                task.max_runs is not None and runs >= task.max_runs
            ):
                self._scheduled.pop(task.schedule_id, None)
                continue
            next_run = task.next_run_at_ms
            if next_run <= now_ms:
                # Jump straight past now; stepping one interval at a time can
                # take billions of iterations after a long pause.
                next_run += (
                    (now_ms - next_run) // task.interval_ms + 1
                ) * task.interval_ms
            self._scheduled[task.schedule_id] = ScheduledTask(
                schedule_id=task.schedule_id,
                payload=task.payload,
                next_run_at_ms=next_run,
                interval_ms=task.interval_ms,
                max_runs=task.max_runs,
                runs=runs,
            )
        return tuple(emitted)


@dataclass(frozen=True)
class ExternalEvent:
    event_id: str
    source: str
    type: str
    payload: JSONValue
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("external event_id must not be empty")
        if not self.source.strip():
            raise ValueError("external event source must not be empty")
        if not self.type.strip():
            raise ValueError("external event type must not be empty")
        _validate_json_value(self.payload)


@dataclass(frozen=True)
class EventTriggerRule:
    trigger_id: str
    source: str
    event_type: str
    task_prefix: str = "event"
    priority: int = 0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("trigger_id", self.trigger_id),
            ("source", self.source),
            ("event_type", self.event_type),
            ("task_prefix", self.task_prefix),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.max_attempts < 1:
            raise ValueError("event trigger max_attempts must be at least 1")

    def matches(self, event: ExternalEvent) -> bool:
        return self.source == event.source and self.event_type == event.type


class EventTriggerDispatcher:
    def __init__(
        self,
        queue: InMemoryWorkQueue,
        *,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.queue = queue
        self.clock = clock or (lambda: int(time.time() * 1000))
        self._rules: dict[str, EventTriggerRule] = {}
        self._seen_events: set[str] = set()

    def register(self, rule: EventTriggerRule) -> None:
        if rule.trigger_id in self._rules:
            raise ValueError(f"event trigger already exists: {rule.trigger_id}")
        self._rules[rule.trigger_id] = rule

    def unregister(self, trigger_id: str) -> bool:
        return self._rules.pop(trigger_id, None) is not None

    def dispatch(self, event: ExternalEvent) -> tuple[WorkQueueItem, ...]:
        if event.event_id in self._seen_events:
            return ()
        enqueued: list[WorkQueueItem] = []
        now = self.clock()
        for rule in sorted(self._rules.values(), key=lambda item: item.trigger_id):
            if not rule.matches(event):
                continue
            item = WorkQueueItem(
                item_id=f"{rule.task_prefix}:{rule.trigger_id}:{event.event_id}",
                payload={
                    "trigger_id": rule.trigger_id,
                    "event_id": event.event_id,
                    "source": event.source,
                    "type": event.type,
                    "payload": event.payload,
                },
                priority=rule.priority,
                max_attempts=rule.max_attempts,
                available_at_ms=now,
                enqueued_at_ms=now,
            )
            if self.queue.contains(item.item_id):
                continue  # already enqueued by an earlier, partially failed dispatch
            self.queue.enqueue(item)
            enqueued.append(item)
        # Mark the event seen only once every rule has been enqueued, so a
        # failure part-way leaves it eligible for redelivery.
        self._seen_events.add(event.event_id)
        return tuple(enqueued)
