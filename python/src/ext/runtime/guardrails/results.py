"""Guardrail results, violations, and blocked-action errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

GuardrailAction = Literal["allow", "transform", "block"]


@dataclass(frozen=True)
class GuardrailResult:
    action: GuardrailAction = "allow"
    value: Any = None
    reason: str | None = None
    classifications: tuple[str, ...] = ()


class GuardrailViolationError(RuntimeError):
    def __init__(self, reason: str, classifications: Sequence[str] = ()) -> None:
        self.classifications = tuple(classifications)
        super().__init__(reason)


class ActionBlockedError(GuardrailViolationError):
    """Raised when an action blocker rejects a proposed tool call."""


def _apply_guardrail_result(original: Any, result: GuardrailResult) -> Any:
    if result.action == "block":
        raise GuardrailViolationError(
            result.reason or "guardrail blocked value",
            result.classifications,
        )
    if result.action == "transform":
        return result.value
    if result.action != "allow":
        raise ValueError(f"unsupported guardrail action: {result.action}")
    return original
