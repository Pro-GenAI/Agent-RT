"""Provider-neutral model requests, responses, embeddings, and streaming events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.context.items import PromptCacheHint
from ext.runtime.tools.definitions import ToolDefinition, ToolSelectionRequirement
from ext.runtime.types import CancellationToken, FinishReason, ModelMessage


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    input: Any
    model: str | None = None
    dimensions: int | None = None
    encoding_format: str | None = None


@dataclass(frozen=True)
class EmbeddingItem:
    index: int
    embedding: Sequence[float] | str


@dataclass(frozen=True)
class EmbeddingResponse:
    data: Sequence[EmbeddingItem]
    model: str | None = None
    usage: ModelUsage | None = None
    raw: Any = None


@runtime_checkable
class EmbeddingModelProvider(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


@dataclass(frozen=True)
class StructuredOutputRequirement:
    schema: Mapping[str, Any]
    name: str | None = None
    strict: bool = True


# JSON mode (OpenAI `response_format={"type": "json_object"}`): any JSON
# object, without a schema. OpenAI-compatible providers send JSON mode for it.
JSON_OBJECT_OUTPUT = StructuredOutputRequirement(
    schema={"type": "object"}, name=None, strict=False
)


@dataclass(frozen=True)
class ReasoningConfig:
    effort: str | None = None
    summary: str | None = None
    thinking: Literal["adaptive", "enabled", "disabled"] | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.thinking not in (None, "adaptive", "enabled", "disabled"):
            raise ValueError(
                "reasoning thinking must be adaptive, enabled, or disabled"
            )
        if self.effort is not None and not self.effort.strip():
            raise ValueError("reasoning effort must not be empty")
        if self.summary is not None and not self.summary.strip():
            raise ValueError("reasoning summary must not be empty")
        if self.budget_tokens is not None and self.budget_tokens < 1:
            raise ValueError("reasoning budget_tokens must be at least 1")
        if self.budget_tokens is not None and self.thinking != "enabled":
            raise ValueError("reasoning budget_tokens requires thinking='enabled'")


@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[ModelMessage]
    model: str | None = None
    tools: Sequence[ToolDefinition] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    structured_output: StructuredOutputRequirement | None = None
    reasoning: ReasoningConfig | None = None
    tool_selection: ToolSelectionRequirement | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_cache: PromptCacheHint | None = None
    cancellation_token: CancellationToken | None = None


@dataclass(frozen=True)
class ModelResponse:
    message: ModelMessage
    model: str | None = None
    usage: ModelUsage | None = None
    finish_reason: FinishReason | None = None
    raw: Any = None


@runtime_checkable
class TokenCountingModelProvider(Protocol):
    async def count_tokens(self, request: ModelRequest) -> int: ...


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


StreamEventType = Literal[
    "text_delta",
    "reasoning_delta",
    "status",
    "tool_call_delta",
    "completed",
]


@dataclass(frozen=True)
class ModelStreamEvent:
    type: StreamEventType
    text: str | None = None
    status: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    response: ModelResponse | None = None
    raw: Any = None


@runtime_checkable
class StreamingModelProvider(ModelProvider, Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...
