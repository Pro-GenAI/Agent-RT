"""Data classification and exfiltration guardrails for tool inputs and outputs."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ext.runtime.guardrails.results import GuardrailResult, GuardrailViolationError
from ext.runtime.guardrails.tool_guardrails import (
    ToolInputGuardrail,
    ToolOutputGuardrail,
)
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ToolCall

DataClassification = Literal["public", "internal", "confidential", "restricted"]


@dataclass(frozen=True)
class DataEgressRequest:
    classification: DataClassification
    channel: str
    target: str | None = None


class DataExfiltrationPolicy:
    def __init__(
        self, allowed: Mapping[DataClassification, Sequence[str]] | None = None
    ) -> None:
        self.allowed = {
            "public": ("*",),
            "internal": ("model", "log:internal", "tool:internal", "network:internal"),
            "confidential": ("tool:approved", "network:approved"),
            "restricted": (),
        }
        if allowed is not None:
            self.allowed.update({key: tuple(value) for key, value in allowed.items()})

    def check(self, request: DataEgressRequest) -> None:
        destination = (
            request.channel
            if request.target is None
            else f"{request.channel}:{request.target}"
        )
        patterns = self.allowed.get(request.classification, ())
        if not any(fnmatch.fnmatchcase(destination, pattern) for pattern in patterns):
            raise GuardrailViolationError(
                f"{request.classification} data cannot leave through {destination}",
                ("data-exfiltration", request.classification),
            )


def make_tool_input_exfiltration_guardrail(
    policy: DataExfiltrationPolicy,
    classify: Callable[[ToolCall], DataClassification],
    target: Callable[[ToolCall], str] = lambda call: call.name,
) -> ToolInputGuardrail:
    def guard(call: ToolCall, _definition: ToolDefinition) -> GuardrailResult:
        policy.check(DataEgressRequest(classify(call), "tool", target(call)))
        return GuardrailResult()

    return guard


def make_tool_output_exfiltration_guardrail(
    policy: DataExfiltrationPolicy,
    classify: Callable[[Any], DataClassification],
    *,
    channel: str = "model",
    target: str | None = None,
) -> ToolOutputGuardrail:
    def guard(
        value: Any, _call: ToolCall, _definition: ToolDefinition
    ) -> GuardrailResult:
        policy.check(DataEgressRequest(classify(value), channel, target))
        return GuardrailResult()

    return guard
