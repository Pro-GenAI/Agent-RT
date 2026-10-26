"""Shared runtime primitives: JSON values, messages, tool calls, cancellation, and the lazy asyncio shim."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


ContentPartType = Literal[
    "text", "image", "audio", "video", "pdf", "document", "file", "json"
]


FinishReason = Literal[
    "stop", "tool_calls", "length", "content_filter", "error", "other"
]


JSONScalar = str | int | float | bool | None


JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class _LazyAsyncio:
    def __init__(self) -> None:
        self._module: Any = None

    def __getattr__(self, name: str) -> Any:
        module = self._module
        if module is None:
            import asyncio as module

            self._module = module
        return getattr(module, name)


if TYPE_CHECKING:
    import asyncio
else:
    asyncio = _LazyAsyncio()


async def _default_async_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite numbers are not serializable")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: workflow state object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path}: unsupported workflow state value type {type(value).__name__}"
    )


@dataclass(frozen=True)
class WorkflowState:
    state_type: str
    version: int
    data: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_type.strip():
            raise ValueError("workflow state_type must not be empty")
        if self.version < 1:
            raise ValueError("workflow state version must be at least 1")
        _validate_json_value(dict(self.data))

    def to_json_value(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "state_type": self.state_type,
            "version": self.version,
            "data": dict(self.data),
        }
        _validate_json_value(payload)
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_json_value(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json_value(
        cls,
        value: Mapping[str, Any],
        *,
        expected_state_type: str | None = None,
    ) -> WorkflowState:
        state_type = value.get("state_type")
        version = value.get("version")
        data = value.get("data")
        if not isinstance(state_type, str):
            raise ValueError("workflow state_type must be a string")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("workflow state version must be an integer")
        if not isinstance(data, Mapping):
            raise ValueError("workflow state data must be an object")
        if expected_state_type is not None and state_type != expected_state_type:
            raise ValueError(
                f"workflow state type mismatch: expected {expected_state_type}, got {state_type}"
            )
        return cls(state_type=state_type, version=version, data=dict(data))

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_state_type: str | None = None,
    ) -> WorkflowState:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("workflow state payload is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("workflow state payload must be a JSON object")
        return cls.from_json_value(
            value,
            expected_state_type=expected_state_type,
        )


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    thread_id: str

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.thread_id.strip():
            raise ValueError("thread_id must not be empty")


@dataclass(frozen=True)
class MemoryScope:
    user: str | None = None
    tenant: str | None = None
    agent: str | None = None
    project: str | None = None
    workspace: str | None = None
    task: str | None = None

    def contains(self, other: MemoryScope) -> bool:
        for name in ("user", "tenant", "agent", "project", "workspace", "task"):
            expected = getattr(self, name)
            if expected is not None and getattr(other, name) != expected:
                return False
        return True


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class RunCancelled(Exception):
    pass


@dataclass(frozen=True)
class ContentPart:
    type: ContentPartType
    text: str | None = None
    data: Any = None
    mime_type: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]
    # Set by provider adapters when the model's raw arguments were not a valid
    # JSON object. The call must not run; the loop reports the error to the
    # model so it can repair the call instead of aborting the whole run.
    argument_error: str | None = None


@dataclass(frozen=True)
class ModelMessage:
    role: MessageRole
    content: Sequence[ContentPart]
    tool_calls: Sequence[ToolCall] = ()
    tool_call_id: str | None = None
