"""Progress reporting and CLI, API, IDE, and chat integration surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ext.runtime.authorization.approvals import ApprovalDecision, ApprovalRequest
from ext.runtime.types import ContentPart, ModelMessage, SessionRef
from ext.runtime.workspace.filesystem import WorkspaceFiles

if TYPE_CHECKING:
    from ext.runtime.memory.sessions import SessionSnapshot, ShortTermSessionMemory


ProgressStatus = Literal[
    "queued",
    "running",
    "waiting_for_input",
    "waiting_for_approval",
    "completed",
    "failed",
    "canceled",
]


@dataclass(frozen=True)
class ProgressEvent:
    task_id: str
    status: ProgressStatus
    message: str = ""
    active_step: str | None = None
    completed_steps: tuple[str, ...] = ()
    pending_approval_id: str | None = None
    waiting_on: str | None = None
    progress: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("progress must be between 0 and 1")


ProgressListener = Callable[[ProgressEvent], Any]


class ProgressReporter:
    def __init__(self) -> None:
        self._listeners: list[ProgressListener] = []
        self._events: list[ProgressEvent] = []

    def subscribe(self, listener: ProgressListener) -> None:
        self._listeners.append(listener)

    def emit(self, event: ProgressEvent) -> ProgressEvent:
        self._events.append(event)
        for listener in tuple(self._listeners):
            listener(event)
        return event

    @property
    def events(self) -> tuple[ProgressEvent, ...]:
        return tuple(self._events)


@dataclass(frozen=True)
class RuntimeRequest:
    agent: str
    messages: tuple[ModelMessage, ...]
    session_id: str | None = None
    task_id: str | None = None
    structured: bool = False
    stream: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeResponse:
    task_id: str
    session_id: str | None
    result: Any
    events: tuple[ProgressEvent, ...] = ()


RuntimeExecutor = Callable[[RuntimeRequest], Awaitable[Any]]


class CLIInterface:
    def __init__(
        self,
        executor: RuntimeExecutor,
        *,
        session_memory: ShortTermSessionMemory | None = None,
    ) -> None:
        self.executor = executor
        self.session_memory = session_memory

    async def run(
        self,
        request: RuntimeRequest,
        *,
        stdin: str | None = None,
    ) -> RuntimeResponse:
        effective = request
        if stdin is not None:
            effective = replace(
                request,
                messages=request.messages
                + (
                    ModelMessage(
                        role="user",
                        content=(ContentPart(type="text", text=stdin),),
                    ),
                ),
            )
        result = await self.executor(effective)
        if self.session_memory is not None and effective.session_id is not None:
            session = SessionRef(effective.session_id, "cli")
            self.session_memory.append_messages(session, effective.messages)
        return RuntimeResponse(
            task_id=effective.task_id or "cli-task",
            session_id=effective.session_id,
            result=result,
        )

    def resume(self, session_id: str) -> SessionSnapshot | None:
        if self.session_memory is None:
            return None
        return self.session_memory.snapshot(SessionRef(session_id, "cli"))


APIOperation = Literal[
    "run",
    "session.get",
    "task.get",
    "task.cancel",
    "events.list",
    "artifact.get",
    "approval.resolve",
    "status.get",
]


@dataclass(frozen=True)
class APIRequest:
    operation: APIOperation
    payload: Mapping[str, Any] = field(default_factory=dict)


APIOperationHandler = Callable[[Mapping[str, Any]], Awaitable[Any]]


class APIInterface:
    def __init__(self) -> None:
        self._handlers: dict[str, APIOperationHandler] = {}

    def register(
        self,
        operation: APIOperation,
        handler: APIOperationHandler,
    ) -> None:
        self._handlers[operation] = handler

    async def handle(self, request: APIRequest) -> Any:
        handler = self._handlers.get(request.operation)
        if handler is None:
            raise KeyError(request.operation)
        return await handler(request.payload)


@dataclass(frozen=True)
class IDEDiagnostic:
    path: str
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class IDEContext:
    workspace_id: str
    current_file: str | None = None
    selection: str | None = None
    diagnostics: tuple[IDEDiagnostic, ...] = ()
    diff: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IDEProgress:
    task_id: str
    message: str
    fraction: float | None = None


class IDEIntegration:
    def __init__(self, workspace: WorkspaceFiles) -> None:
        self.workspace = workspace

    def read_current(self, context: IDEContext) -> str | None:
        if context.current_file is None:
            return None
        return self.workspace.read_text(context.current_file)

    def apply_patch(self, path: str, patch: str) -> None:
        self.workspace.apply_unified_patch(path, patch)


# Lazily loaded interfaces/observability ledgers that are not needed by AgentLoop.
@dataclass(frozen=True)
class ChatEnvelope:
    channel: str
    user_id: str
    thread_id: str
    text: str
    message_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatReply:
    channel: str
    thread_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatAdapter(Protocol):
    async def send(self, reply: ChatReply) -> Any: ...


class ChatSessionBridge:
    def __init__(
        self,
        adapter: ChatAdapter,
        memory: ShortTermSessionMemory,
    ) -> None:
        self.adapter = adapter
        self.memory = memory

    def session_ref(self, envelope: ChatEnvelope) -> SessionRef:
        return SessionRef(
            session_id=f"{envelope.channel}:{envelope.user_id}",
            thread_id=envelope.thread_id,
        )

    def ingest(self, envelope: ChatEnvelope) -> SessionRef:
        session = self.session_ref(envelope)
        self.memory.append_messages(
            session,
            (
                ModelMessage(
                    role="user",
                    content=(ContentPart(type="text", text=envelope.text),),
                ),
            ),
        )
        return session

    async def reply(self, envelope: ChatEnvelope, text: str) -> Any:
        return await self.adapter.send(
            ChatReply(
                channel=envelope.channel,
                thread_id=envelope.thread_id,
                text=text,
            )
        )


@dataclass(frozen=True)
class ApprovalPresentation:
    approval_id: str
    title: str
    summary: str
    consequences: tuple[str, ...] = ()
    diff: str | None = None
    choices: tuple[ApprovalDecision, ...] = ("allow", "deny")
    metadata: Mapping[str, Any] = field(default_factory=dict)


def approval_presentation(
    request: ApprovalRequest,
    *,
    title: str | None = None,
    summary: str | None = None,
    consequences: Sequence[str] = (),
    diff: str | None = None,
) -> ApprovalPresentation:
    return ApprovalPresentation(
        approval_id=request.id,
        title=title or f"Approve {request.call.name}",
        summary=summary or request.reason,
        consequences=tuple(consequences),
        diff=diff,
        metadata={
            "tool": request.call.name,
            "side_effect": request.side_effect,
            "session_id": request.session_id,
        },
    )
