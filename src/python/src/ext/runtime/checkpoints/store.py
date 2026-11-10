"""Agent checkpoints and checkpoint stores."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ext.runtime.agent.config import AgentConfig
from ext.runtime.agent.limits import AgentRunResult
from ext.runtime.types import (
    JSONValue,
    ModelMessage,
    WorkflowState,
    _validate_json_value,
)

if TYPE_CHECKING:
    from ext.runtime.workspace.filesystem import ArtifactStatus


@dataclass(frozen=True)
class AgentCheckpoint:
    checkpoint_id: str
    agent_name: str
    messages: tuple[ModelMessage, ...]
    workflow_state: WorkflowState | None = None
    turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    status: ArtifactStatus = "draft"
    retention_until_ms: int | None = None
    location: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be empty")
        if not self.agent_name.strip():
            raise ValueError("checkpoint agent_name must not be empty")
        if self.turns < 0 or self.tool_calls < 0 or self.total_tokens < 0:
            raise ValueError("checkpoint counters must be non-negative")
        _validate_json_value(dict(self.metadata))


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, checkpoint: AgentCheckpoint) -> AgentCheckpoint: ...
    def load(self, checkpoint_id: str) -> AgentCheckpoint | None: ...
    def delete(self, checkpoint_id: str) -> bool: ...
    def list(self, *, agent_name: str | None = None) -> tuple[AgentCheckpoint, ...]: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, AgentCheckpoint] = {}

    def save(self, checkpoint: AgentCheckpoint) -> AgentCheckpoint:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        return self._checkpoints.get(checkpoint_id)

    def delete(self, checkpoint_id: str) -> bool:
        return self._checkpoints.pop(checkpoint_id, None) is not None

    def list(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[AgentCheckpoint, ...]:
        checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if agent_name is None or checkpoint.agent_name == agent_name
        ]
        checkpoints.sort(
            key=lambda checkpoint: (
                checkpoint.created_at_ms,
                checkpoint.checkpoint_id,
            ),
            reverse=True,
        )
        return tuple(checkpoints)


def checkpoint_from_result(
    checkpoint_id: str,
    agent: AgentConfig,
    result: AgentRunResult,
    *,
    workflow_state: WorkflowState | None = None,
    metadata: Mapping[str, JSONValue] | None = None,
) -> AgentCheckpoint:
    return AgentCheckpoint(
        checkpoint_id=checkpoint_id,
        agent_name=agent.name,
        messages=result.messages,
        workflow_state=workflow_state,
        turns=result.turns,
        tool_calls=result.tool_calls,
        total_tokens=result.total_tokens,
        metadata=metadata or {},
    )
