"""Human approval requests, grants, stores, and the approval manager."""

from __future__ import annotations

import fnmatch
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.observability.audit import AuditRecord, AuditTrail
from ext.runtime.tools.definitions import ToolDefinition, ToolSideEffect
from ext.runtime.types import ToolCall

ApprovalDecision = Literal["allow", "deny"]


ApprovalScope = Literal["once", "session", "durable"]


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    call: ToolCall
    side_effect: ToolSideEffect
    reason: str
    session_id: str | None = None


def _canonical_tool_arguments(arguments: Mapping[str, Any]) -> str:
    """Canonical JSON text used to bind an approval to exact tool arguments."""
    return json.dumps(
        dict(arguments), sort_keys=True, separators=(",", ":"), default=str
    )


@dataclass(frozen=True)
class ApprovalGrant:
    id: str
    decision: ApprovalDecision
    scope: ApprovalScope
    tool_pattern: str
    session_id: str | None = None
    call_id: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    # When set on a ``once`` grant, the grant only applies to a call whose
    # canonical arguments are identical, so a reused call id cannot smuggle
    # different arguments past the human who approved the call.
    arguments_json: str | None = None


class ApprovalRequiredError(RuntimeError):
    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        super().__init__(request.reason)


class ApprovalDeniedError(PermissionError):
    pass


@runtime_checkable
class ApprovalStore(Protocol):
    def add(self, grant: ApprovalGrant) -> ApprovalGrant: ...
    def decision(
        self,
        *,
        tool: str,
        call_id: str,
        session_id: str | None = None,
        arguments_json: str | None = None,
    ) -> ApprovalDecision | None: ...
    def revoke(self, grant_id: str) -> bool: ...
    def list(self) -> tuple[ApprovalGrant, ...]: ...


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._grants: dict[str, ApprovalGrant] = {}

    def add(self, grant: ApprovalGrant) -> ApprovalGrant:
        if grant.scope == "session" and not grant.session_id:
            raise ValueError("session approval requires session_id")
        if grant.scope == "once" and not grant.call_id:
            raise ValueError("once approval requires call_id")
        self._grants[grant.id] = grant
        return grant

    def decision(
        self,
        *,
        tool: str,
        call_id: str,
        session_id: str | None = None,
        arguments_json: str | None = None,
    ) -> ApprovalDecision | None:
        for grant in reversed(tuple(self._grants.values())):
            if not fnmatch.fnmatchcase(tool, grant.tool_pattern):
                continue
            if grant.scope == "session" and grant.session_id != session_id:
                continue
            if grant.scope == "once" and grant.call_id != call_id:
                continue
            if (
                grant.scope == "once"
                and grant.arguments_json is not None
                and grant.arguments_json != arguments_json
            ):
                continue
            if grant.scope == "once":
                self._grants.pop(grant.id, None)
            return grant.decision
        return None

    def revoke(self, grant_id: str) -> bool:
        return self._grants.pop(grant_id, None) is not None

    def list(self) -> tuple[ApprovalGrant, ...]:
        return tuple(self._grants.values())


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class ApprovalManager:
    def __init__(
        self,
        store: ApprovalStore | None = None,
        *,
        required_side_effects: Sequence[ToolSideEffect] = (
            "consequential",
            "destructive",
        ),
        audit_trail: AuditTrail | None = None,
        notifier: Callable[[ApprovalRequest], None] | None = None,
    ) -> None:
        self.notifier = notifier
        self.store = store or InMemoryApprovalStore()
        self.required_side_effects = frozenset(required_side_effects)
        self.audit_trail = audit_trail

    def requires_approval(self, definition: ToolDefinition) -> bool:
        return (
            definition.side_effect in self.required_side_effects
            or definition.metadata.get("approval_required") is True
        )

    def check(
        self,
        call: ToolCall,
        definition: ToolDefinition,
        *,
        session_id: str | None = None,
    ) -> None:
        if not self.requires_approval(definition):
            return
        lookup: dict[str, Any] = {
            "tool": call.name,
            "call_id": call.id,
            "session_id": session_id,
        }
        if _accepts_keyword(self.store.decision, "arguments_json"):
            lookup["arguments_json"] = _canonical_tool_arguments(call.arguments)
        decision = self.store.decision(**lookup)
        if decision == "allow":
            return
        if decision == "deny":
            raise ApprovalDeniedError(f"approval denied for tool: {call.name}")
        request = ApprovalRequest(
            id=f"approval:{session_id or 'global'}:{call.id}",
            call=call,
            side_effect=definition.side_effect,
            reason=f"human approval required for {definition.side_effect} tool: {call.name}",
            session_id=session_id,
        )
        if self.notifier is not None:
            self.notifier(request)
        raise ApprovalRequiredError(request)

    def resolve(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        *,
        scope: ApprovalScope = "once",
        grant_id: str | None = None,
        actor_id: str = "human",
    ) -> ApprovalGrant:
        grant = ApprovalGrant(
            id=grant_id or f"grant:{request.id}:{scope}",
            decision=decision,
            scope=scope,
            tool_pattern=request.call.name,
            session_id=request.session_id if scope == "session" else None,
            call_id=request.call.id if scope == "once" else None,
            arguments_json=(
                _canonical_tool_arguments(request.call.arguments)
                if scope == "once"
                else None
            ),
        )
        stored = self.store.add(grant)
        if self.audit_trail is not None:
            self.audit_trail.append(
                AuditRecord(
                    id=f"audit:{stored.id}",
                    actor_id=actor_id,
                    action="authorized",
                    resource=request.call.name,
                    outcome="success" if decision == "allow" else "denied",
                    details={
                        "approval_id": request.id,
                        "scope": scope,
                        "call_id": request.call.id,
                    },
                )
            )
        return stored
