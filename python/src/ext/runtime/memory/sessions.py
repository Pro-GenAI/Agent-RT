"""Short-term and tenant-scoped session memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ext.runtime.authorization.grants import TenantContext
from ext.runtime.types import ModelMessage, SessionRef, WorkflowState


# Lazily loaded session and long-term memory implementations.
@dataclass(frozen=True)
class SessionSnapshot:
    session: SessionRef
    messages: tuple[ModelMessage, ...] = ()
    workflow_state: WorkflowState | None = None


class ShortTermSessionMemory:
    def __init__(self, *, max_messages: int = 100) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: dict[SessionRef, list[ModelMessage]] = {}
        self._states: dict[SessionRef, WorkflowState] = {}

    def append_messages(
        self,
        session: SessionRef,
        messages: Sequence[ModelMessage],
    ) -> None:
        bucket = self._messages.setdefault(session, [])
        bucket.extend(messages)
        if len(bucket) > self.max_messages:
            del bucket[: -self.max_messages]

    def set_workflow_state(
        self,
        session: SessionRef,
        state: WorkflowState | None,
    ) -> None:
        if state is None:
            self._states.pop(session, None)
        else:
            self._states[session] = state

    def snapshot(self, session: SessionRef) -> SessionSnapshot:
        return SessionSnapshot(
            session=session,
            messages=tuple(self._messages.get(session, ())),
            workflow_state=self._states.get(session),
        )

    def clear(self, session: SessionRef) -> None:
        self._messages.pop(session, None)
        self._states.pop(session, None)


class TenantSessionMemory:
    def __init__(self, memory: ShortTermSessionMemory, tenant: TenantContext) -> None:
        self.memory = memory
        self.tenant = tenant

    def append_messages(
        self, session: SessionRef, messages: Sequence[ModelMessage]
    ) -> None:
        self.memory.append_messages(self.tenant.session_ref(session), messages)

    def set_workflow_state(
        self, session: SessionRef, state: WorkflowState | None
    ) -> None:
        self.memory.set_workflow_state(self.tenant.session_ref(session), state)

    def snapshot(self, session: SessionRef) -> SessionSnapshot:
        return self.memory.snapshot(self.tenant.session_ref(session))

    def clear(self, session: SessionRef) -> None:
        self.memory.clear(self.tenant.session_ref(session))
