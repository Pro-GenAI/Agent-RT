"""Audit records, audit trails, and tool lifecycle audit hooks."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.guardrails.privacy import PrivacyRedactor
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ToolCall

AuditAction = Literal["requested", "authorized", "executed", "changed", "canceled"]


AuditOutcome = Literal["success", "denied", "error"]


@dataclass(frozen=True)
class AuditRecord:
    id: str
    actor_id: str
    action: AuditAction
    resource: str
    outcome: AuditOutcome = "success"
    details: Mapping[str, Any] = field(default_factory=dict)
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@runtime_checkable
class AuditTrail(Protocol):
    def append(self, record: AuditRecord) -> AuditRecord: ...
    def list(self) -> tuple[AuditRecord, ...]: ...


class InMemoryAuditTrail:
    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._ids: set[str] = set()

    def append(self, record: AuditRecord) -> AuditRecord:
        if record.id in self._ids:
            raise ValueError(f"audit record already exists: {record.id}")
        self._records.append(record)
        self._ids.add(record.id)
        return record

    def list(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)


ToolAuditPhase = Literal["start", "success", "error"]


@dataclass(frozen=True)
class ToolAuditEvent:
    phase: ToolAuditPhase
    call: ToolCall
    definition: ToolDefinition
    value: Any = None
    error: BaseException | None = None
    request_context: Mapping[str, Any] = field(default_factory=dict)


ToolPreCallHook = Callable[
    [ToolCall, ToolDefinition],
    Awaitable[ToolCall | None],
]


ToolPostCallHook = Callable[
    [ToolCall, ToolDefinition, Any],
    Awaitable[Any],
]


ToolErrorHook = Callable[
    [ToolCall, ToolDefinition, BaseException],
    Awaitable[None],
]


ToolAuditHook = Callable[[ToolAuditEvent], Awaitable[None]]


@dataclass(frozen=True)
class ToolLifecycleHooks:
    pre_call: ToolPreCallHook | None = None
    post_call: ToolPostCallHook | None = None
    on_error: ToolErrorHook | None = None
    audit: ToolAuditHook | None = None


def make_audit_trail_hook(
    trail: AuditTrail,
    *,
    redactor: PrivacyRedactor | None = None,
) -> ToolAuditHook:
    sanitizer = redactor or PrivacyRedactor()

    async def hook(event: ToolAuditEvent) -> None:
        actor = event.request_context.get("actor_id", "unknown")
        actor_id = actor if isinstance(actor, str) and actor else "unknown"
        action: AuditAction = (
            "requested"
            if event.phase == "start"
            else "executed" if event.phase == "success" else "canceled"
        )
        details = {
            "call_id": event.call.id,
            "side_effect": event.definition.side_effect,
            "arguments": sanitizer.redact(dict(event.call.arguments)),
        }
        if event.phase == "success":
            details["result"] = sanitizer.redact(event.value)
        elif event.error is not None:
            details["error"] = sanitizer.redact(str(event.error))
        trail.append(
            AuditRecord(
                id=f"audit:{event.call.id}:{event.phase}:{len(trail.list()) + 1}",
                actor_id=actor_id,
                action=action,
                resource=event.call.name,
                outcome="error" if event.phase == "error" else "success",
                details=details,
            )
        )

    return hook
