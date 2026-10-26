"""Remote agents: cards, directories, tasks, artifacts, transports, and clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class RemoteAgentCard:
    id: str
    name: str
    endpoint: str
    description: str = ""
    modalities: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RemoteAgentDirectory(Protocol):
    async def discover(self) -> Sequence[RemoteAgentCard]: ...


class RemoteAgentRegistry:
    def __init__(self, directory: RemoteAgentDirectory) -> None:
        self.directory = directory
        self._agents: dict[str, RemoteAgentCard] = {}

    async def refresh(self) -> tuple[RemoteAgentCard, ...]:
        discovered = tuple(await self.directory.discover())
        self._agents = {agent.id: agent for agent in discovered}
        return discovered

    def get(self, agent_id: str) -> RemoteAgentCard:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        return self._agents[agent_id]


@dataclass(frozen=True)
class RemoteMessage:
    id: str
    role: str
    parts: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


RemoteTaskStatus = Literal[
    "submitted",
    "working",
    "input_required",
    "completed",
    "failed",
    "canceled",
]


@dataclass(frozen=True)
class RemoteTask:
    id: str
    agent_id: str
    status: RemoteTaskStatus = "submitted"
    message: RemoteMessage | None = None
    error: str | None = None
    version: int = 0


class RemoteTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, RemoteTask] = {}

    def upsert(self, task: RemoteTask) -> RemoteTask:
        existing = self._tasks.get(task.id)
        if existing is not None and task.version < existing.version:
            raise ValueError("remote task version cannot move backwards")
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> RemoteTask:
        if task_id not in self._tasks:
            raise KeyError(task_id)
        return self._tasks[task_id]


@dataclass(frozen=True)
class RemoteArtifact:
    id: str
    task_id: str
    name: str
    media_type: str
    data: Any
    version: int = 1
    complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RemoteArtifactStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, list[RemoteArtifact]] = {}

    def put(self, artifact: RemoteArtifact) -> RemoteArtifact:
        versions = self._artifacts.setdefault(artifact.id, [])
        if versions and artifact.version <= versions[-1].version:
            raise ValueError("remote artifact version must increase")
        versions.append(artifact)
        return artifact

    def latest(self, artifact_id: str) -> RemoteArtifact:
        versions = self._artifacts.get(artifact_id)
        if not versions:
            raise KeyError(artifact_id)
        return versions[-1]


@dataclass(frozen=True)
class RemoteEvent:
    type: Literal["status", "message", "artifact", "callback"]
    task_id: str
    payload: Mapping[str, Any]


@runtime_checkable
class RemoteAgentTransport(Protocol):
    async def send_message(
        self,
        agent: RemoteAgentCard,
        message: RemoteMessage,
    ) -> Mapping[str, Any]: ...

    async def get_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> Mapping[str, Any]: ...

    async def cancel_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> Mapping[str, Any]: ...

    def stream_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> AsyncIterator[Mapping[str, Any]]: ...


CapabilitySet = frozenset[str]


@dataclass(frozen=True)
class CapabilityNegotiation:
    local: CapabilitySet
    remote: CapabilitySet
    required: CapabilitySet = frozenset()

    @property
    def negotiated(self) -> CapabilitySet:
        available = self.local & self.remote
        missing = self.required - available
        if missing:
            raise ValueError(f"required capabilities unavailable: {sorted(missing)}")
        return available


class RemoteAgentClient:
    def __init__(
        self,
        registry: RemoteAgentRegistry,
        transport: RemoteAgentTransport,
        *,
        task_store: RemoteTaskStore | None = None,
        artifact_store: RemoteArtifactStore | None = None,
        local_capabilities: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.task_store = task_store or RemoteTaskStore()
        self.artifact_store = artifact_store or RemoteArtifactStore()
        self.local_capabilities = frozenset(local_capabilities)

    def negotiate(
        self,
        agent_id: str,
        *,
        required: Sequence[str] = (),
    ) -> CapabilitySet:
        card = self.registry.get(agent_id)
        return CapabilityNegotiation(
            local=self.local_capabilities,
            remote=card.capabilities,
            required=frozenset(required),
        ).negotiated

    @staticmethod
    def _parse_task(agent_id: str, payload: Mapping[str, Any]) -> RemoteTask:
        status = str(payload.get("status", "submitted"))
        if status not in {
            "submitted",
            "working",
            "input_required",
            "completed",
            "failed",
            "canceled",
        }:
            raise ValueError(f"unsupported remote task status: {status}")
        task_id = str(payload.get("id", "")).strip()
        if not task_id:
            raise ValueError("remote task id is required")
        return RemoteTask(
            id=task_id,
            agent_id=agent_id,
            status=status,  # type: ignore[arg-type]
            error=str(payload["error"]) if payload.get("error") is not None else None,
            version=int(payload.get("version", 0)),
        )

    def _persist_artifact(
        self,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> RemoteArtifact:
        artifact = RemoteArtifact(
            id=str(payload["id"]),
            task_id=task_id,
            name=str(payload.get("name", payload["id"])),
            media_type=str(payload.get("mediaType", "application/octet-stream")),
            data=payload.get("data"),
            version=int(payload.get("version", 1)),
            complete=bool(payload.get("complete", True)),
            metadata=(
                dict(payload.get("metadata", {}))
                if isinstance(payload.get("metadata", {}), Mapping)
                else {}
            ),
        )
        return self.artifact_store.put(artifact)

    async def send(
        self,
        agent_id: str,
        message: RemoteMessage,
    ) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.send_message(card, message)
        task = self.task_store.upsert(self._parse_task(agent_id, payload))
        artifacts = payload.get("artifacts", ())
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
            for artifact in artifacts:
                if isinstance(artifact, Mapping):
                    self._persist_artifact(task.id, artifact)
        return task

    async def refresh_task(self, agent_id: str, task_id: str) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.get_task(card, task_id)
        return self.task_store.upsert(self._parse_task(agent_id, payload))

    async def cancel(self, agent_id: str, task_id: str) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.cancel_task(card, task_id)
        return self.task_store.upsert(self._parse_task(agent_id, payload))

    async def stream(
        self,
        agent_id: str,
        task_id: str,
        *,
        callback: Callable[[RemoteEvent], Awaitable[None]] | None = None,
    ) -> tuple[RemoteEvent, ...]:
        card = self.registry.get(agent_id)
        events: list[RemoteEvent] = []
        async for payload in self.transport.stream_task(card, task_id):
            event_type = str(payload.get("type", "status"))
            if event_type not in {"status", "message", "artifact", "callback"}:
                continue
            event = RemoteEvent(
                type=event_type,  # type: ignore[arg-type]
                task_id=task_id,
                payload=dict(payload),
            )
            events.append(event)
            if event_type == "status":
                self.task_store.upsert(self._parse_task(agent_id, payload))
            elif event_type == "artifact":
                artifact = payload.get("artifact")
                if isinstance(artifact, Mapping):
                    self._persist_artifact(task_id, artifact)
            if callback is not None:
                await callback(event)
        return tuple(events)
