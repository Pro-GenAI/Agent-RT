"""Agent run limits, secure presets, and run results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ext.runtime.authorization.approvals import ApprovalManager
from ext.runtime.authorization.permissions import PermissionEngine, PermissionRule
from ext.runtime.models.interfaces import ModelResponse
from ext.runtime.resilience.budgets import TerminationReason
from ext.runtime.resilience.retry import FailureDisposition
from ext.runtime.tools.registry import ToolRegistry
from ext.runtime.types import ModelMessage


@dataclass(frozen=True)
class AgentRunLimits:
    max_turns: int = 16
    max_tool_calls: int = 64
    timeout_seconds: float | None = None
    max_total_tokens: int | None = None
    concurrent_tool_calls: bool = False


# Finite bounds for untrusted or production runs (the API server's defaults).
SECURE_RUN_LIMITS = AgentRunLimits(
    max_turns=16,
    max_tool_calls=64,
    timeout_seconds=120.0,
    max_total_tokens=100_000,
)


def secure_tool_registry(
    permission_rules: Sequence[PermissionRule] = (),
    *,
    approval_manager: ApprovalManager | None = None,
    **options: Any,
) -> ToolRegistry:
    """ToolRegistry preset for untrusted or production use.

    Permissions deny by default: only ``none``/``read`` tools are allowed until
    ``permission_rules`` allow more (deny rules always win), and
    consequential/destructive tools still need human approval.
    """
    return ToolRegistry(
        permission_engine=PermissionEngine(
            (
                PermissionRule("allow", side_effects=("none", "read")),
                *permission_rules,
            )
        ),
        approval_manager=approval_manager or ApprovalManager(),
        **options,
    )


@dataclass(frozen=True)
class AgentRunResult:
    messages: tuple[ModelMessage, ...]
    final_response: ModelResponse | None
    termination_reason: TerminationReason
    turns: int
    tool_calls: int
    total_tokens: int
    structured_output: Any = None
    failure: FailureDisposition | None = None
