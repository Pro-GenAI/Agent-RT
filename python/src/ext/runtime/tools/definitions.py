"""Tool definitions, side-effect classes, execution limits, and tool selection policy."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ext.runtime.settings import runtime_settings
from ext.runtime.types import CancellationToken

ToolSideEffect = Literal[
    "none", "read", "write", "reversible", "consequential", "destructive"
]


ToolErrorBehavior = Literal["raise", "return_error"]


ToolExecutionMode = Literal["parallel", "sequential"]


ToolHandler = Callable[
    [Mapping[str, Any], CancellationToken | None],
    Awaitable[Any],
]


EXECUTION_TIMEOUT_ENV = "AGENT_RT_EXECUTION_TIMEOUT_SECONDS"


EXECUTION_MEMORY_ENV = "AGENT_RT_EXECUTION_MEMORY_BYTES"


EXECUTION_CPU_ENV = "AGENT_RT_EXECUTION_CPU_SECONDS"


@dataclass(frozen=True)
class ToolExecutionLimits:
    timeout_seconds: float | None = None
    memory_bytes: int | None = None
    cpu_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("memory_bytes", self.memory_bytes),
            ("cpu_seconds", self.cpu_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")


def tool_execution_limits_from_env(
    environment: Mapping[str, str] | None = None,
) -> ToolExecutionLimits:
    env = runtime_settings() if environment is None else environment

    def parse_float(name: str) -> float | None:
        raw = env.get(name)
        if raw is None:
            return None
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
        return value

    def parse_int(name: str) -> int | None:
        raw = env.get(name)
        if raw is None:
            return None
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    return ToolExecutionLimits(
        timeout_seconds=parse_float(EXECUTION_TIMEOUT_ENV),
        memory_bytes=parse_int(EXECUTION_MEMORY_ENV),
        cpu_seconds=parse_float(EXECUTION_CPU_ENV),
    )


@dataclass(frozen=True)
class ToolExecutionContext:
    services: Mapping[str, Any] = field(default_factory=dict)
    request_context: Mapping[str, Any] = field(default_factory=dict)
    cancellation_token: CancellationToken | None = None
    limits: ToolExecutionLimits = field(default_factory=ToolExecutionLimits)


ContextualToolHandler = Callable[
    [Mapping[str, Any], ToolExecutionContext],
    Awaitable[Any],
]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    side_effect: ToolSideEffect = "none"
    error_behavior: ToolErrorBehavior = "raise"
    timeout_seconds: float | None = None
    memory_bytes: int | None = None
    cpu_seconds: float | None = None
    execution_mode: ToolExecutionMode = "parallel"

    def __post_init__(self) -> None:
        validate_tool_definition(self)


def validate_tool_definition(tool: ToolDefinition) -> None:
    if not tool.name.strip():
        raise ValueError("tool name must not be empty")
    if not tool.description.strip():
        raise ValueError("tool description must not be empty")
    if tool.input_schema.get("type") not in (None, "object"):
        raise ValueError("tool input_schema must describe an object")
    if tool.side_effect not in (
        "none",
        "read",
        "write",
        "reversible",
        "consequential",
        "destructive",
    ):
        raise ValueError(f"unsupported tool side effect: {tool.side_effect}")
    if tool.error_behavior not in ("raise", "return_error"):
        raise ValueError(f"unsupported tool error behavior: {tool.error_behavior}")
    if tool.timeout_seconds is not None and tool.timeout_seconds < 0:
        raise ValueError("tool timeout_seconds must be non-negative")
    if tool.memory_bytes is not None and tool.memory_bytes < 0:
        raise ValueError("tool memory_bytes must be non-negative")
    if tool.cpu_seconds is not None and tool.cpu_seconds < 0:
        raise ValueError("tool cpu_seconds must be non-negative")
    if tool.execution_mode not in ("parallel", "sequential"):
        raise ValueError(f"unsupported tool execution mode: {tool.execution_mode}")


def _effective_tool_execution_limits(
    tool: ToolDefinition | None,
) -> ToolExecutionLimits:
    defaults = tool_execution_limits_from_env()
    if tool is None:
        return defaults
    return ToolExecutionLimits(
        timeout_seconds=(
            tool.timeout_seconds
            if tool.timeout_seconds is not None
            else defaults.timeout_seconds
        ),
        memory_bytes=(
            tool.memory_bytes
            if tool.memory_bytes is not None
            else defaults.memory_bytes
        ),
        cpu_seconds=(
            tool.cpu_seconds if tool.cpu_seconds is not None else defaults.cpu_seconds
        ),
    )


@dataclass(frozen=True)
class ToolSelectionPolicy:
    allowed: frozenset[str] | None = None
    denied: frozenset[str] = frozenset()
    required: frozenset[str] = frozenset()
    preferred: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = self.denied & self.required
        if overlap:
            raise ValueError(
                "required tools cannot also be denied: " + ", ".join(sorted(overlap))
            )
        if self.allowed is not None:
            missing = self.required - self.allowed
            if missing:
                raise ValueError(
                    "required tools must be allowed: " + ", ".join(sorted(missing))
                )

    def permits(self, name: str) -> bool:
        if name in self.denied:
            return False
        return self.allowed is None or name in self.allowed

    def filter_definitions(
        self,
        definitions: Sequence[ToolDefinition],
    ) -> tuple[ToolDefinition, ...]:
        return tuple(tool for tool in definitions if self.permits(tool.name))


@dataclass(frozen=True)
class ToolSelectionRequirement:
    required: tuple[str, ...] = ()
    preferred: tuple[str, ...] = ()
