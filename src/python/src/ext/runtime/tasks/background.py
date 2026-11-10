"""Task lifecycle and background task management."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ext.runtime.checkpoints.events import DurableEvent, EventStore
from ext.runtime.types import CancellationToken, JSONValue

if TYPE_CHECKING:
    import asyncio
else:
    from ext.runtime.types import asyncio


TaskStatus = Literal[
    "submitted",
    "queued",
    "running",
    "waiting_for_input",
    "waiting_for_approval",
    "completed",
    "failed",
    "canceled",
]


@dataclass(frozen=True)
class TaskLifecycleState:
    task_id: str
    status: TaskStatus = "submitted"
    version: int = 0

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.version < 0:
            raise ValueError("task lifecycle version must be non-negative")


class TaskLifecycle:
    _transitions: Mapping[TaskStatus, frozenset[TaskStatus]] = {
        "submitted": frozenset({"queued", "running", "canceled"}),
        "queued": frozenset({"running", "canceled"}),
        "running": frozenset(
            {
                "waiting_for_input",
                "waiting_for_approval",
                "completed",
                "failed",
                "canceled",
            }
        ),
        "waiting_for_input": frozenset({"running", "canceled"}),
        "waiting_for_approval": frozenset({"running", "failed", "canceled"}),
        "completed": frozenset(),
        "failed": frozenset(),
        "canceled": frozenset(),
    }

    def __init__(
        self,
        state: TaskLifecycleState,
        *,
        event_store: EventStore | None = None,
    ) -> None:
        self.state = state
        self.event_store = event_store

    def transition(
        self,
        status: TaskStatus,
        *,
        reason: str | None = None,
    ) -> TaskLifecycleState:
        if status == self.state.status:
            return self.state
        allowed = self._transitions[self.state.status]
        if status not in allowed:
            raise ValueError(
                f"invalid task transition: {self.state.status} -> {status}"
            )
        previous = self.state
        self.state = TaskLifecycleState(
            task_id=previous.task_id,
            status=status,
            version=previous.version + 1,
        )
        if self.event_store is not None:
            payload: dict[str, JSONValue] = {
                "from": previous.status,
                "to": status,
                "version": self.state.version,
            }
            if reason is not None:
                payload["reason"] = reason
            self.event_store.append(
                DurableEvent(
                    event_id=(f"{self.state.task_id}:lifecycle:{self.state.version}"),
                    task_id=self.state.task_id,
                    type="lifecycle_transition",
                    payload=payload,
                )
            )
        return self.state


BackgroundTaskRunner = Callable[
    [Callable[[float, str | None], None], CancellationToken],
    Awaitable[Any],
]


@dataclass(frozen=True)
class BackgroundTaskSnapshot:
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    message: str | None = None
    result: Any = None
    error: str | None = None
    submitted_at_ms: int = 0
    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("background task_id must not be empty")
        if self.progress < 0 or self.progress > 1:
            raise ValueError("background task progress must be between 0 and 1")


class BackgroundTaskManager:
    def __init__(self) -> None:
        self._snapshots: dict[str, BackgroundTaskSnapshot] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._tokens: dict[str, CancellationToken] = {}

    def submit(
        self,
        task_id: str,
        runner: BackgroundTaskRunner,
    ) -> BackgroundTaskSnapshot:
        if not task_id.strip():
            raise ValueError("background task_id must not be empty")
        if task_id in self._snapshots:
            raise ValueError(f"background task already exists: {task_id}")
        now = int(time.time() * 1000)
        initial = BackgroundTaskSnapshot(
            task_id=task_id,
            status="queued",
            submitted_at_ms=now,
        )
        self._snapshots[task_id] = initial
        token = CancellationToken()
        self._tokens[task_id] = token

        async def execute() -> None:
            started = int(time.time() * 1000)
            self._snapshots[task_id] = BackgroundTaskSnapshot(
                task_id=task_id,
                status="running",
                progress=0.0,
                submitted_at_ms=now,
                started_at_ms=started,
            )

            def report(progress: float, message: str | None = None) -> None:
                if progress < 0 or progress > 1:
                    raise ValueError("background task progress must be between 0 and 1")
                current = self._snapshots[task_id]
                if current.status != "running":
                    return
                self._snapshots[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    status="running",
                    progress=progress,
                    message=message,
                    submitted_at_ms=current.submitted_at_ms,
                    started_at_ms=current.started_at_ms,
                )

            try:
                result = await runner(report, token)
            except asyncio.CancelledError:
                completed = int(time.time() * 1000)
                current = self._snapshots[task_id]
                self._snapshots[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    status="canceled",
                    progress=current.progress,
                    message=current.message,
                    submitted_at_ms=current.submitted_at_ms,
                    started_at_ms=current.started_at_ms,
                    completed_at_ms=completed,
                )
                raise
            except Exception as exc:
                completed = int(time.time() * 1000)
                current = self._snapshots[task_id]
                self._snapshots[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    status="failed",
                    progress=current.progress,
                    message=current.message,
                    error=str(exc),
                    submitted_at_ms=current.submitted_at_ms,
                    started_at_ms=current.started_at_ms,
                    completed_at_ms=completed,
                )
            else:
                completed = int(time.time() * 1000)
                current = self._snapshots[task_id]
                status: TaskStatus = "canceled" if token.is_cancelled else "completed"
                self._snapshots[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    status=status,
                    progress=1.0 if status == "completed" else current.progress,
                    message=current.message,
                    result=result if status == "completed" else None,
                    submitted_at_ms=current.submitted_at_ms,
                    started_at_ms=current.started_at_ms,
                    completed_at_ms=completed,
                )

        self._tasks[task_id] = asyncio.create_task(execute())
        return initial

    def get(self, task_id: str) -> BackgroundTaskSnapshot | None:
        return self._snapshots.get(task_id)

    async def wait(self, task_id: str) -> BackgroundTaskSnapshot:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"background task not found: {task_id}")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Swallow only the job's own cancellation. If *we* were cancelled
            # (for example by wait_for) the job keeps running and the
            # cancellation must propagate.
            if not task.cancelled():
                raise
        return self._snapshots[task_id]

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        token = self._tokens.get(task_id)
        if task is None or token is None or task.done():
            return False
        token.cancel()
        task.cancel()
        current = self._snapshots[task_id]
        if current.status == "queued":
            # The coroutine was cancelled before it ever ran, so its own
            # cancellation handler will never record the new state.
            self._snapshots[task_id] = BackgroundTaskSnapshot(
                task_id=task_id,
                status="canceled",
                progress=current.progress,
                message=current.message,
                submitted_at_ms=current.submitted_at_ms,
                completed_at_ms=int(time.time() * 1000),
            )
        return True

    def list(self) -> tuple[BackgroundTaskSnapshot, ...]:
        return tuple(
            sorted(
                self._snapshots.values(),
                key=lambda item: (item.submitted_at_ms, item.task_id),
            )
        )
