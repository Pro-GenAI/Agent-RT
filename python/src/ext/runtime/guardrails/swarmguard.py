"""Opt-in SwarmGuard PII detection for tool-call arguments.

Matches email addresses or explicit synthetic canaries recursively without logging values.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ext.runtime.guardrails.results import GuardrailResult
from ext.runtime.guardrails.tool_guardrails import ToolInputGuardrail
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ToolCall

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)


def contains_pii(value: Any, *, canaries: Sequence[str] = ()) -> bool:
    """Deeply scan input values, without exposing matches in diagnostics."""
    if isinstance(value, str):
        return bool(_EMAIL.search(value)) or any(
            canary and canary in value for canary in canaries
        )
    if isinstance(value, Mapping):
        return any(contains_pii(item, canaries=canaries) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_pii(item, canaries=canaries) for item in value)
    return False


@dataclass(frozen=True)
class UnexpectedPIIPolicy:
    """Default-deny sensitive arguments unless the named tool is exempted."""

    exempt_tools: frozenset[str] = frozenset()
    canaries: tuple[str, ...] = ()

    def unexpected(self, name: str, arguments: Mapping[str, Any]) -> bool:
        return name not in self.exempt_tools and contains_pii(
            arguments, canaries=self.canaries
        )


def make_unexpected_pii_guardrail(
    policy: UnexpectedPIIPolicy = UnexpectedPIIPolicy(),
) -> ToolInputGuardrail:
    def guard(call: ToolCall, _definition: ToolDefinition) -> GuardrailResult:
        if policy.unexpected(call.name, call.arguments):
            return GuardrailResult(
                action="block",
                reason="unexpected sensitive data in tool arguments",
                classifications=("swarmguard", "unexpected-pii"),
            )
        return GuardrailResult()

    return guard
