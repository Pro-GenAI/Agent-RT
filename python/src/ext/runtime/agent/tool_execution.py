"""Agent loop tool-call execution: visibility, approval, guardrails, and timeouts."""

from __future__ import annotations

import hashlib
import json
from time import monotonic
from typing import TYPE_CHECKING

from ext.runtime.authorization.approvals import ApprovalRequiredError
from ext.runtime.resilience.budgets import (
    TerminationReason,
)
from ext.runtime.resilience.idempotency import IdempotencyRecord
from ext.runtime.tools.definitions import (
    _effective_tool_execution_limits,
)
from ext.runtime.tools.programs import ToolTimeoutError, _tool_timeout_payload
from ext.runtime.tools.registry import RegisteredTool
from ext.runtime.tools.todos import marshal_tool_result
from ext.runtime.tools.validation import (
    ToolArgumentValidationError,
    _tool_validation_error_payload,
    validate_tool_arguments,
)
from ext.runtime.types import (
    ContentPart,
    ModelMessage,
    RunCancelled,
    ToolCall,
)

if TYPE_CHECKING:
    import asyncio

    from ext.runtime.agent.loop import AgentLoop
    from ext.runtime.agent.run_steps import _RunState
else:
    from ext.runtime.types import asyncio


async def execute_tool_call(
    self: AgentLoop,
    run: _RunState,
    call: ToolCall,
) -> tuple[ModelMessage | None, TerminationReason | None]:
    if run.cancellation_token is not None and run.cancellation_token.is_cancelled:
        return None, "cancelled"
    if run.stop_requested is not None and run.stop_requested():
        return None, "stop_requested"
    if (
        run.limits.timeout_seconds is not None
        and monotonic() - run.started >= run.limits.timeout_seconds
    ):
        return None, "timeout"
    if (
        (run.tool_policy is not None and not run.tool_policy.permits(call.name))
        or (
            self.capability_grant is not None
            and call.name not in run.active_visible_tool_names
        )
        or (
            self.tool_filter is not None
            and call.name not in run.active_visible_tool_names
        )
    ):
        return (
            ModelMessage(
                role="tool",
                content=(
                    ContentPart(
                        type="json",
                        data=_tool_validation_error_payload(
                            call,
                            ("tool is not permitted by selection policy",),
                        ),
                    ),
                ),
                tool_call_id=call.id,
            ),
            None,
        )

    if call.argument_error is not None:
        return (
            ModelMessage(
                role="tool",
                content=(
                    ContentPart(
                        type="json",
                        data=_tool_validation_error_payload(
                            call, (call.argument_error,)
                        ),
                    ),
                ),
                tool_call_id=call.id,
            ),
            None,
        )

    registered: RegisteredTool | None = None
    if self.tool_registry is not None:
        try:
            registered = self.tool_registry.get(call.name)
            if call.name not in run.active_visible_tool_names:
                return (
                    ModelMessage(
                        role="tool",
                        content=(
                            ContentPart(
                                type="json",
                                data=_tool_validation_error_payload(
                                    call,
                                    (
                                        "tool is not permitted by active visibility policy",
                                    ),
                                ),
                            ),
                        ),
                        tool_call_id=call.id,
                    ),
                    None,
                )
            if not registered.enabled:
                raise ToolArgumentValidationError(
                    call.name,
                    ("tool is disabled",),
                )
            validate_tool_arguments(registered.definition, call.arguments)
        except (KeyError, ToolArgumentValidationError) as exc:
            issues = (
                exc.issues
                if isinstance(exc, ToolArgumentValidationError)
                else (f"tool is not registered: {call.name}",)
            )
            return (
                ModelMessage(
                    role="tool",
                    content=(
                        ContentPart(
                            type="json",
                            data=_tool_validation_error_payload(call, issues),
                        ),
                    ),
                    tool_call_id=call.id,
                ),
                None,
            )

    run.emit_event(
        "tool_requested",
        {"tool": call.name, "tool_call_id": call.id},
    )
    idempotency_scope = (
        f"{run.task_id}:tool:{call.name}" if run.task_id is not None else None
    )
    # Bind the key to the arguments so a reused call id with different
    # arguments is executed instead of replaying a stale result.
    idempotency_key = (
        f"{call.id}:"
        + hashlib.sha256(
            json.dumps(
                dict(call.arguments),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
    )
    if self.idempotency_store is not None and idempotency_scope is not None:
        existing = self.idempotency_store.get(
            idempotency_scope,
            idempotency_key,
        )
        if existing is not None:
            run.emit_event(
                "tool_completed",
                {
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "replayed": True,
                },
            )
            return (
                ModelMessage(
                    role="tool",
                    content=marshal_tool_result(existing.value),
                    tool_call_id=call.id,
                ),
                None,
            )

    try:
        if (
            registered is not None
            and self.tool_registry is not None
            and (
                registered.handler is not None
                or registered.contextual_handler is not None
            )
        ):
            effective_tool_context = dict(run.tool_context)
            effective_tool_context["user_prompt"] = run.user_prompt
            if run.active_deadline is not None:
                effective_tool_context["deadline"] = run.active_deadline
                effective_tool_context["deadline_remaining_seconds"] = (
                    run.active_deadline.remaining_seconds()
                )
            tool_awaitable = self.tool_registry.execute(
                call,
                run.cancellation_token,
                effective_tool_context,
            )
        elif self.tool_executor is not None:
            tool_awaitable = (
                self.tool_executor.execute(call, run.cancellation_token)
                if run.cancellation_token is not None
                else self.tool_executor.execute(call)
            )
        else:
            raise RuntimeError(
                "model requested tools but no tool executor or registered handler is configured"
            )
        tool_timeout = _effective_tool_execution_limits(
            registered.definition if registered is not None else None
        ).timeout_seconds
        value = await run.await_tool_operation(
            tool_awaitable,
            call,
            tool_timeout,
        )
        if self.idempotency_store is not None and idempotency_scope is not None:
            self.idempotency_store.put(
                IdempotencyRecord(
                    scope=idempotency_scope,
                    key=idempotency_key,
                    value=value,
                )
            )
        run.emit_event(
            "tool_completed",
            {
                "tool": call.name,
                "tool_call_id": call.id,
                "replayed": False,
            },
        )
    except ApprovalRequiredError as exc:
        run.emit_event(
            "approval_requested",
            {
                "approval_id": exc.request.id,
                "tool": call.name,
                "tool_call_id": call.id,
                "side_effect": exc.request.side_effect,
                "reason": exc.request.reason,
            },
        )
        return None, "waiting_for_approval"
    except RunCancelled:
        return None, "cancelled"
    except ToolTimeoutError as exc:
        return (
            ModelMessage(
                role="tool",
                content=(
                    ContentPart(
                        type="json",
                        data=_tool_timeout_payload(
                            call,
                            exc.timeout_seconds,
                        ),
                    ),
                ),
                tool_call_id=call.id,
            ),
            None,
        )
    except asyncio.TimeoutError:
        # Only the run deadline ends the run. A TimeoutError raised by
        # the tool's own I/O is an ordinary tool failure.
        if run.deadline_hit:
            return None, "timeout"
        raise

    return (
        ModelMessage(
            role="tool",
            content=marshal_tool_result(value),
            tool_call_id=call.id,
        ),
        None,
    )
