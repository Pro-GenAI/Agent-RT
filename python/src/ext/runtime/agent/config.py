"""Agent configuration, model settings, and compiled execution plans."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ext.runtime.models.interfaces import (
    ModelStreamEvent,
    ModelUsage,
    StructuredOutputRequirement,
)
from ext.runtime.tools.definitions import (
    ToolDefinition,
    ToolSelectionPolicy,
    ToolSelectionRequirement,
)
from ext.runtime.types import CancellationToken, ToolCall

if TYPE_CHECKING:
    from ext.runtime.models.routing import ModelTarget


StreamEventHandler = Callable[[ModelStreamEvent], Awaitable[None]]


@dataclass(frozen=True)
class ModelSettings:
    model: str
    provider: str | None = None
    fallback_models: Sequence[ModelTarget] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentOutputRequirements:
    format: Literal["text", "json"] | None = None
    schema: Mapping[str, Any] | None = None
    max_repair_attempts: int = 1


@dataclass(frozen=True)
class AgentConfig:
    name: str
    instructions: str
    model: ModelSettings
    role: str | None = None
    description: str | None = None
    capabilities: Sequence[str] = ()
    output: AgentOutputRequirements | None = None
    tool_policy: ToolSelectionPolicy | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Strings, custom objects, stores or callables; see MemorySource.
    memory: Sequence[Any] = ()


@dataclass(frozen=True)
class CompiledExecutionPlan:
    model_settings: ModelSettings
    visible_tools: tuple[ToolDefinition, ...]
    tool_policy: ToolSelectionPolicy | None
    structured_output: StructuredOutputRequirement | None
    tool_selection: ToolSelectionRequirement | None
    visible_tool_names: frozenset[str]
    tool_registry_version: int | None
    dynamic_tool_filter: bool
    simple_text_fast_path: bool
    active_stages: tuple[str, ...]


@runtime_checkable
class ToolExecutor(Protocol):
    async def execute(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken | None = None,
    ) -> Any: ...


@runtime_checkable
class TokenUsageRecorder(Protocol):
    def record_model_usage(
        self,
        usage: ModelUsage,
        **attribution: Any,
    ) -> Any: ...
