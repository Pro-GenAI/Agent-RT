"""Failure classification, retry policies, and recovery routing."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ext.runtime.authorization.approvals import (
    ApprovalDeniedError,
    ApprovalRequiredError,
)
from ext.runtime.authorization.permissions import PermissionDeniedError
from ext.runtime.guardrails.results import GuardrailViolationError
from ext.runtime.resilience.rate_limits import CircuitOpenError, RateLimitExceededError
from ext.runtime.tools.programs import ToolTimeoutError
from ext.runtime.tools.validation import (
    StructuredOutputValidationError,
    ToolArgumentValidationError,
)
from ext.runtime.types import _default_async_sleep

if TYPE_CHECKING:
    import asyncio

    from ext.runtime.resilience.budgets import ExecutionBudget
else:
    from ext.runtime.types import asyncio


FailureKind = Literal[
    "transient_dependency",
    "model_correctable",
    "user_correctable",
    "policy",
    "terminal_system",
]


@dataclass(frozen=True)
class FailureDisposition:
    kind: FailureKind
    retryable: bool
    reason: str


def classify_failure(error: BaseException) -> FailureDisposition:
    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            ConnectionError,
            ToolTimeoutError,
            CircuitOpenError,
            RateLimitExceededError,
        ),
    ):
        return FailureDisposition(
            "transient_dependency", True, "dependency or timeout failure"
        )
    if isinstance(
        error, (ToolArgumentValidationError, StructuredOutputValidationError)
    ):
        return FailureDisposition(
            "model_correctable", False, "model output or tool arguments can be repaired"
        )
    if isinstance(error, ApprovalRequiredError):
        return FailureDisposition(
            "user_correctable", False, "human input or approval is required"
        )
    if isinstance(
        error, (PermissionDeniedError, ApprovalDeniedError, GuardrailViolationError)
    ):
        return FailureDisposition(
            "policy", False, "policy or authorization denied the operation"
        )
    if isinstance(error, (ValueError, KeyError)):
        return FailureDisposition(
            "user_correctable", False, "request or configuration can be corrected"
        )
    return FailureDisposition(
        "terminal_system", False, "unclassified terminal system failure"
    )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.1
    multiplier: float = 2.0
    max_delay_seconds: float = 5.0
    jitter_ratio: float = 0.0
    retry_kinds: frozenset[FailureKind] = frozenset({"transient_dependency"})

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay_for_retry(self, retry_number: int, *, random_value: float = 0.5) -> float:
        if retry_number < 1:
            raise ValueError("retry_number must be at least 1")
        base = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * (self.multiplier ** (retry_number - 1)),
        )
        if self.jitter_ratio == 0:
            return base
        spread = base * self.jitter_ratio
        return max(0.0, base - spread + (2 * spread * random_value))


class RetryExecutor:
    def __init__(
        self,
        default_policy: RetryPolicy = RetryPolicy(),
        *,
        operation_policies: Mapping[str, RetryPolicy] | None = None,
        classifier: Callable[[BaseException], FailureDisposition] = classify_failure,
        budget: ExecutionBudget | None = None,
    ) -> None:
        self.default_policy = default_policy
        self.operation_policies = dict(operation_policies or {})
        self.classifier = classifier
        self.budget = budget

    async def execute(
        self,
        operation_name: str,
        operation: Callable[[int], Awaitable[Any]],
        *,
        sleep: Callable[[float], Awaitable[Any]] = _default_async_sleep,
        random_value: Callable[[], float] = random.random,
    ) -> Any:
        policy = self.operation_policies.get(operation_name, self.default_policy)
        attempt = 1
        while True:
            try:
                return await operation(attempt)
            except Exception as exc:
                disposition = self.classifier(exc)
                if (
                    attempt >= policy.max_attempts
                    or disposition.kind not in policy.retry_kinds
                    or not disposition.retryable
                ):
                    raise
                if self.budget is not None:
                    self.budget.consume_retry()
                await sleep(
                    policy.delay_for_retry(
                        attempt,
                        random_value=random_value(),
                    )
                )
                attempt += 1


RecoveryAction = Literal[
    "retry",
    "repair_arguments",
    "alternate_tool",
    "switch_model",
    "request_user_input",
    "escalate",
]


@dataclass(frozen=True)
class RecoveryContext:
    attempts_exhausted: bool = False
    alternate_tool_available: bool = False
    model_fallback_available: bool = False


class RecoveryRouter:
    def route(
        self,
        failure: FailureDisposition,
        context: RecoveryContext = RecoveryContext(),
    ) -> RecoveryAction:
        if failure.kind == "transient_dependency":
            if not context.attempts_exhausted:
                return "retry"
            if context.alternate_tool_available:
                return "alternate_tool"
            if context.model_fallback_available:
                return "switch_model"
            return "escalate"
        if failure.kind == "model_correctable":
            return "repair_arguments"
        if failure.kind == "user_correctable":
            return "request_user_input"
        if failure.kind == "policy":
            return "escalate"
        if context.model_fallback_available:
            return "switch_model"
        return "escalate"
