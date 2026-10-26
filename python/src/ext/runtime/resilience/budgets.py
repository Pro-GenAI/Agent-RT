"""Execution budgets, deadlines, loop detection, and finalizers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal

from ext.runtime.types import ModelMessage

TerminationReason = Literal[
    "completed",
    "cancelled",
    "stop_requested",
    "max_turns",
    "max_tool_calls",
    "timeout",
    "budget_exhausted",
    "waiting_for_approval",
    "loop_detected",
    "model_response_failure",
]


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionBudgetLimits:
    max_model_calls: int | None = None
    max_retries: int | None = None
    max_subagents: int | None = None
    max_cost: float | None = None


class ExecutionBudget:
    def __init__(self, limits: ExecutionBudgetLimits = ExecutionBudgetLimits()) -> None:
        self.limits = limits
        self.model_calls = 0
        self.retries = 0
        self.subagents = 0
        self.cost = 0.0

    @staticmethod
    def _consume(
        current: float, amount: float, limit: float | None, label: str
    ) -> int | float:
        if amount < 0:
            raise ValueError(f"{label} amount must be non-negative")
        next_value = current + amount
        if limit is not None and next_value > limit:
            raise BudgetExceededError(f"{label} budget exceeded")
        return next_value

    def consume_model_call(self, amount: int = 1) -> None:
        self.model_calls = int(
            self._consume(
                self.model_calls, amount, self.limits.max_model_calls, "model call"
            )
        )

    def consume_retry(self, amount: int = 1) -> None:
        self.retries = int(
            self._consume(self.retries, amount, self.limits.max_retries, "retry")
        )

    def consume_subagent(self, amount: int = 1) -> None:
        self.subagents = int(
            self._consume(self.subagents, amount, self.limits.max_subagents, "subagent")
        )

    def consume_cost(self, amount: float) -> None:
        self.cost = float(
            self._consume(self.cost, amount, self.limits.max_cost, "cost")
        )


@dataclass(frozen=True)
class Deadline:
    expires_at: float
    clock: Callable[[], float] = field(default=monotonic, compare=False, repr=False)

    @classmethod
    def after(
        cls, seconds: float, *, clock: Callable[[], float] = monotonic
    ) -> Deadline:
        if seconds < 0:
            raise ValueError("deadline duration must be non-negative")
        return cls(clock() + seconds, clock)

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    @property
    def expired(self) -> bool:
        return self.remaining_seconds() <= 0


@dataclass
class LoopDetector:
    repeat_threshold: int = 3
    _last_signature: str | None = field(default=None, init=False, repr=False)
    _repeat_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold must be at least 2")

    def reset(self) -> None:
        """Forget the observed trajectory (``AgentLoop`` calls this per run)."""
        self._last_signature = None
        self._repeat_count = 0

    def observe(self, message: ModelMessage) -> bool:
        signature = json.dumps(
            {
                "content": [
                    {
                        "type": part.type,
                        "text": part.text,
                        "data": part.data,
                    }
                    for part in message.content
                ],
                "tools": [
                    {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                    for call in message.tool_calls
                ],
            },
            sort_keys=True,
            default=str,
        )
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1
        return self._repeat_count >= self.repeat_threshold


Finalizer = Callable[[], Any]


async def run_finalizers(finalizers: Sequence[Finalizer]) -> None:
    first_error: BaseException | None = None
    for finalizer in reversed(tuple(finalizers)):
        try:
            value = finalizer()
            if hasattr(value, "__await__"):
                await value
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
