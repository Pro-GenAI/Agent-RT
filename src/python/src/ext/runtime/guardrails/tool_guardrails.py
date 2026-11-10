"""Tool input/output guardrails, including Agent Action Guard."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ext.runtime.guardrails.results import GuardrailResult
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ToolCall

if TYPE_CHECKING:
    import asyncio
else:
    from ext.runtime.types import asyncio


ToolInputGuardrail = Callable[[ToolCall, ToolDefinition], GuardrailResult]


ToolOutputGuardrail = Callable[..., GuardrailResult]


async def _call_maybe_async(function: Callable[..., Any], *args: Any) -> Any:
    """Call a hook, preferring its ``acall`` twin, and await awaitable results.

    Hooks that do blocking I/O (Decision-model clients) expose ``acall`` so the
    loop can run them off the event loop while staying callable synchronously.
    """
    acall = getattr(function, "acall", None)
    result = acall(*args) if callable(acall) else function(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _call_tool_output_guardrail(
    guardrail: ToolOutputGuardrail,
    value: Any,
    call: ToolCall,
    definition: ToolDefinition,
    request_context: Mapping[str, Any],
) -> GuardrailResult:
    """Run a sync or async output guardrail (async ones keep blocking I/O off the loop)."""
    acall = getattr(guardrail, "acall", None)
    if callable(acall):
        return await _call_maybe_async(
            guardrail, value, call, definition, request_context
        )
    try:
        signature = inspect.signature(guardrail)
        signature.bind(value, call, definition, request_context)
    except (TypeError, ValueError):
        result = guardrail(value, call, definition)
    else:
        result = guardrail(value, call, definition, request_context)
    if inspect.isawaitable(result):
        result = await result
    return result


DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL = "agent-action-guard"


AgentActionGuardClassifier = Callable[[Mapping[str, Any]], tuple[str | None, float]]


def make_agent_action_guard_tool_input_guardrail(
    classify: AgentActionGuardClassifier | None = None,
) -> ToolInputGuardrail:
    classifier = classify

    def guard(call: ToolCall, _definition: ToolDefinition) -> GuardrailResult:
        nonlocal classifier
        if classifier is None:
            try:
                from agent_action_guard import is_action_harmful
            except ModuleNotFoundError as exc:
                if exc.name != "agent_action_guard":
                    raise
                raise RuntimeError(
                    "Agent Action Guard is not installed. Reinstall "
                    "'agent-rt' (it is a core dependency) to use the "
                    "default model-backed tool input guardrail."
                ) from exc
            classifier = is_action_harmful

        label, confidence = classifier(
            {
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": dict(call.arguments),
                },
            }
        )
        if label:
            return GuardrailResult(
                action="block",
                reason=f"Agent Action Guard blocked tool input ({confidence:.3f})",
                classifications=("tool-input:blocked", "agent-action-guard", label),
            )
        return GuardrailResult(classifications=("agent-action-guard",))

    async def acall(call: ToolCall, definition: ToolDefinition) -> GuardrailResult:
        # The classifier (and its first-use model load) blocks; keep it off the event loop.
        return await asyncio.to_thread(guard, call, definition)

    guard.acall = acall  # type: ignore[attr-defined]
    return guard


def make_default_tool_input_guardrail(
    *,
    model: str = DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
    classify: AgentActionGuardClassifier | None = None,
) -> ToolInputGuardrail:
    if model != DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL:
        raise ValueError(f"unsupported tool input guardrail model: {model}")
    return make_agent_action_guard_tool_input_guardrail(classify)
