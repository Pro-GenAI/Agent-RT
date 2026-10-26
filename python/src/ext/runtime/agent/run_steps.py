"""Agent loop run steps: deadlines, model requests, token usage, and approvals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import TYPE_CHECKING, Any

from ext.runtime.agent.config import (
    AgentConfig,
    StreamEventHandler,
)
from ext.runtime.agent.limits import AgentRunLimits
from ext.runtime.models.interfaces import (
    ModelRequest,
    ModelResponse,
    StreamingModelProvider,
)
from ext.runtime.resilience.budgets import (
    Deadline,
)
from ext.runtime.tools.definitions import (
    ToolSelectionPolicy,
)
from ext.runtime.tools.programs import ToolTimeoutError
from ext.runtime.types import (
    CancellationToken,
    ModelMessage,
    RunCancelled,
    ToolCall,
)

if TYPE_CHECKING:
    import asyncio

    from ext.runtime.agent.loop import AgentLoop
else:
    from ext.runtime.types import asyncio


@dataclass
class _RunState:
    """Values one agent run shares with its step functions.

    deadline_hit, tool_policy and active_visible_tool_names change
    while the run is in progress, so the loop reads and writes them here.
    """

    agent: AgentConfig
    limits: AgentRunLimits
    stop_requested: Callable[[], bool] | None
    on_stream_event: StreamEventHandler | None
    cancellation_token: CancellationToken | None
    tool_context: Mapping[str, Any]
    task_id: str | None
    user_prompt: str
    history: list[ModelMessage]
    started: float
    active_deadline: Deadline | None
    deadline_checks_enabled: bool
    deadline_hit: bool
    tool_policy: ToolSelectionPolicy | None
    active_visible_tool_names: set[str]
    emit_event: Callable[..., None]
    await_with_deadline: Callable[..., Awaitable[Any]] = field(init=False)
    await_tool_operation: Callable[..., Awaitable[Any]] = field(init=False)
    record_token_usage: Callable[[ModelResponse, ModelRequest], None] = field(
        init=False
    )


def record_stage(
    self: AgentLoop,
    stage: str,
    stage_started: float | None,
    **labels: str,
) -> None:
    if self.metrics is None or stage_started is None:
        return
    self.metrics.record(
        "agent_loop.stage.duration_ms",
        (monotonic() - stage_started) * 1000,
        kind="histogram",
        labels={"stage": stage, **labels},
    )


async def await_with_deadline(run: _RunState, awaitable: Any) -> Any:
    task = asyncio.ensure_future(awaitable)
    cancellation_waiter: asyncio.Task[None] | None = None
    if run.cancellation_token is not None:
        if run.cancellation_token.is_cancelled:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise RunCancelled
        cancellation_waiter = asyncio.create_task(run.cancellation_token.wait())

    remaining: float | None = None
    if run.limits.timeout_seconds is not None:
        remaining = run.limits.timeout_seconds - (monotonic() - run.started)
    if run.active_deadline is not None:
        deadline_remaining = run.active_deadline.remaining_seconds()
        remaining = (
            deadline_remaining
            if remaining is None
            else min(remaining, deadline_remaining)
        )
    if remaining is not None and remaining <= 0:
        task.cancel()
        if cancellation_waiter is not None:
            cancellation_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await task
        run.deadline_hit = True
        raise asyncio.TimeoutError

    waiters = {task}
    if cancellation_waiter is not None:
        waiters.add(cancellation_waiter)
    try:
        done, _ = await asyncio.wait(
            waiters,
            timeout=remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        # Cancellation of the outer AgentLoop task (including the
        # Ctrl+C path used by asyncio.run()) must not orphan the
        # provider request or a running tool task.
        task.cancel()
        if cancellation_waiter is not None:
            cancellation_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if cancellation_waiter is not None:
            with suppress(asyncio.CancelledError):
                await cancellation_waiter
        raise

    if task in done:
        if cancellation_waiter is not None:
            cancellation_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_waiter
        return await task

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    if cancellation_waiter is not None:
        cancellation_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await cancellation_waiter
    if run.cancellation_token is not None and run.cancellation_token.is_cancelled:
        raise RunCancelled
    run.deadline_hit = True
    raise asyncio.TimeoutError


async def await_tool_operation(
    run: _RunState,
    awaitable: Any,
    call: ToolCall,
    timeout_seconds: float | None,
) -> Any:
    if timeout_seconds is None:
        return await run.await_with_deadline(awaitable)

    run_remaining: float | None = None
    if run.limits.timeout_seconds is not None:
        run_remaining = run.limits.timeout_seconds - (monotonic() - run.started)
    if run.active_deadline is not None:
        deadline_remaining = run.active_deadline.remaining_seconds()
        run_remaining = (
            deadline_remaining
            if run_remaining is None
            else min(run_remaining, deadline_remaining)
        )
    if run_remaining is not None and run_remaining <= 0:
        if hasattr(awaitable, "close"):
            awaitable.close()
        run.deadline_hit = True
        raise asyncio.TimeoutError

    if timeout_seconds <= 0:
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise ToolTimeoutError(call.name, timeout_seconds)

    tool_wins = run_remaining is None or timeout_seconds <= run_remaining
    try:
        return await asyncio.wait_for(
            run.await_with_deadline(awaitable),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        if tool_wins:
            raise ToolTimeoutError(call.name, timeout_seconds) from exc
        raise


def record_token_usage(
    self: AgentLoop, run: _RunState, response: ModelResponse, request: ModelRequest
) -> None:
    if self.token_ledger is None or response.usage is None:
        return
    attribution: dict[str, Any] = {
        "task_id": run.task_id,
        "agent_id": run.agent.name,
        "model": response.model or request.model or run.agent.model.model,
    }
    user_id = run.tool_context.get("user_id")
    tenant_id = run.tool_context.get("tenant_id")
    if isinstance(user_id, str) and user_id:
        attribution["user_id"] = user_id
    if isinstance(tenant_id, str) and tenant_id:
        attribution["tenant_id"] = tenant_id
    self.token_ledger.record_model_usage(response.usage, **attribution)


async def complete_request(
    self: AgentLoop, run: _RunState, request: ModelRequest
) -> ModelResponse:
    if self.execution_budget is not None:
        self.execution_budget.consume_model_call()
    if run.active_deadline is not None:
        request = replace(
            request,
            metadata={
                **dict(request.metadata),
                "deadline_remaining_seconds": run.active_deadline.remaining_seconds(),
            },
        )
    if self.rate_limiter is not None:
        keys = [
            f"model:{request.model or run.agent.model.model}",
            f"provider:{self.provider.name}",
        ]
        user_id = run.tool_context.get("user_id")
        tenant_id = run.tool_context.get("tenant_id")
        if isinstance(user_id, str) and user_id:
            keys.append(f"user:{user_id}")
        if isinstance(tenant_id, str) and tenant_id:
            keys.append(f"tenant:{tenant_id}")
        self.rate_limiter.check_many(keys)
    run.emit_event(
        "model_requested",
        {"model": request.model or "", "message_count": len(request.messages)},
    )
    if run.on_stream_event is None:
        operation = self.provider.complete(request)
        completed_response = (
            await run.await_with_deadline(operation)
            if run.deadline_checks_enabled
            else await operation
        )
        if self.execution_budget is not None and self.cost_estimator is not None:
            self.execution_budget.consume_cost(
                max(0.0, float(self.cost_estimator(completed_response)))
            )
        run.record_token_usage(completed_response, request)
        run.emit_event(
            "model_completed",
            {
                "model": completed_response.model or request.model or "",
                "finish_reason": completed_response.finish_reason or "other",
            },
        )
        return completed_response
    if not isinstance(self.provider, StreamingModelProvider):
        raise TypeError("streaming requires a provider that implements stream()")

    iterator = self.provider.stream(request).__aiter__()
    completed: ModelResponse | None = None
    while True:
        try:
            next_event = iterator.__anext__()
            event = (
                await run.await_with_deadline(next_event)
                if run.deadline_checks_enabled
                else await next_event
            )
        except StopAsyncIteration:
            break
        await run.on_stream_event(event)
        if event.type == "completed":
            if event.response is None:
                raise RuntimeError("completed stream event must include a response")
            completed = event.response

    if completed is None:
        raise RuntimeError("model stream ended without a completed response")
    if self.execution_budget is not None and self.cost_estimator is not None:
        self.execution_budget.consume_cost(
            max(0.0, float(self.cost_estimator(completed)))
        )
    run.record_token_usage(completed, request)
    run.emit_event(
        "model_completed",
        {
            "model": completed.model or request.model or "",
            "finish_reason": completed.finish_reason or "other",
        },
    )
    return completed


def pending_approval_calls(
    run: _RunState,
) -> list[tuple[ToolCall, int]]:
    """Calls of the last assistant turn that paused for approval.

    A run that stops with ``waiting_for_approval`` leaves a
    ``tool_not_executed`` placeholder for each unexecuted call. When the
    run is resumed those calls are executed (their approval is checked
    again) instead of asking the model to re-issue them under a new call
    id, which would orphan a ``once`` approval.
    """
    last_assistant = next(
        (
            index
            for index in range(len(run.history) - 1, -1, -1)
            if run.history[index].role == "assistant"
        ),
        None,
    )
    if last_assistant is None or not run.history[last_assistant].tool_calls:
        return []
    placeholders: dict[str, int] = {}
    for index in range(last_assistant + 1, len(run.history)):
        message = run.history[index]
        if message.role != "tool" or message.tool_call_id is None:
            return []
        error = next(
            (
                part.data.get("error")
                for part in message.content
                if part.type == "json" and isinstance(part.data, Mapping)
            ),
            None,
        )
        if (
            isinstance(error, Mapping)
            and error.get("type") == "tool_not_executed"
            and error.get("reason") == "waiting_for_approval"
        ):
            placeholders[message.tool_call_id] = index
    return [
        (call, placeholders[call.id])
        for call in run.history[last_assistant].tool_calls
        if call.id in placeholders
    ]
