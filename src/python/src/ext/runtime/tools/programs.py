"""Programmatic tool execution with per-call timeouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ext.runtime.tools.definitions import _effective_tool_execution_limits
from ext.runtime.types import CancellationToken, RunCancelled, ToolCall

if TYPE_CHECKING:
    import asyncio

    from ext.runtime.tools.registry import ToolRegistry
else:
    from ext.runtime.types import asyncio


class ToolTimeoutError(TimeoutError):
    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"tool {tool_name} exceeded timeout of {timeout_seconds} seconds"
        )


@dataclass(frozen=True)
class ToolProgramResult:
    call: ToolCall
    value: Any


class ToolProgramExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        calls: Sequence[ToolCall],
        *,
        concurrent: bool = False,
        cancellation_token: CancellationToken | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> tuple[ToolProgramResult, ...]:
        if concurrent:
            values = await asyncio.gather(
                *(
                    self._execute_one(
                        call,
                        cancellation_token,
                        request_context,
                    )
                    for call in calls
                )
            )
        else:
            values = []
            for call in calls:
                values.append(
                    await self._execute_one(
                        call,
                        cancellation_token,
                        request_context,
                    )
                )
        return tuple(
            ToolProgramResult(call, value) for call, value in zip(calls, values)
        )

    async def _execute_one(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken | None,
        request_context: Mapping[str, Any] | None,
    ) -> Any:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise RunCancelled
        registered = self.registry.get(call.name)
        timeout_seconds = _effective_tool_execution_limits(
            registered.definition
        ).timeout_seconds
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ToolTimeoutError(call.name, timeout_seconds)

        operation = self.registry.execute(
            call,
            cancellation_token,
            request_context,
        )
        if timeout_seconds is None:
            return await operation
        try:
            return await asyncio.wait_for(operation, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise ToolTimeoutError(call.name, timeout_seconds) from exc


def _tool_timeout_payload(
    call: ToolCall,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    return {
        "error": {
            "type": "tool_timeout",
            "tool": call.name,
            "timeout_seconds": timeout_seconds,
        }
    }
