"""The agent loop."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from time import monotonic
from typing import TYPE_CHECKING, Any

from ext.runtime.agent import plan_compiler, run_steps, tool_execution
from ext.runtime.agent.config import (
    AgentConfig,
    CompiledExecutionPlan,
    StreamEventHandler,
    TokenUsageRecorder,
    ToolExecutor,
)
from ext.runtime.agent.limits import AgentRunLimits, AgentRunResult
from ext.runtime.agent.run_steps import _RunState
from ext.runtime.authorization.grants import CapabilityGrant
from ext.runtime.checkpoints.events import DurableEvent, DurableEventType, EventStore
from ext.runtime.checkpoints.store import AgentCheckpoint
from ext.runtime.context.assembler import ContextAssembler
from ext.runtime.context.items import (
    ContextItem,
    ContextSelectionPolicy,
    PromptInjectionDefense,
)
from ext.runtime.guardrails.boundary import (
    BoundaryGuardrailPolicy,
    InputGuardrail,
    OutputGuardrail,
    make_default_input_guardrail,
    make_default_output_guardrail,
)
from ext.runtime.guardrails.results import _apply_guardrail_result
from ext.runtime.guardrails.tool_guardrails import _call_maybe_async
from ext.runtime.memory.sources import resolve_agent_memory
from ext.runtime.models.interfaces import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from ext.runtime.observability.metrics import RuntimeMetrics
from ext.runtime.resilience.budgets import (
    BudgetExceededError,
    Deadline,
    ExecutionBudget,
    Finalizer,
    LoopDetector,
    TerminationReason,
    run_finalizers,
)
from ext.runtime.resilience.idempotency import IdempotencyStore
from ext.runtime.resilience.rate_limits import RateLimiter
from ext.runtime.resilience.retry import FailureDisposition
from ext.runtime.tools.definitions import (
    ToolSelectionRequirement,
)
from ext.runtime.tools.registry import ToolRegistry
from ext.runtime.tools.toolbase import Toolbase, ToolFilterContext, ToolVisibilityFilter
from ext.runtime.tools.validation import (
    StructuredOutputValidationError,
    _validate_structured_message,
)
from ext.runtime.types import (
    CancellationToken,
    ContentPart,
    JSONValue,
    ModelMessage,
    RunCancelled,
    ToolCall,
    WorkflowState,
)

if TYPE_CHECKING:
    import asyncio
else:
    from ext.runtime.types import asyncio


class AgentLoop:
    def __init__(
        self,
        provider: ModelProvider,
        tool_executor: ToolExecutor | None = None,
        tool_registry: ToolRegistry | Toolbase | None = None,
        tool_filter: ToolVisibilityFilter | None = None,
        context_assembler: ContextAssembler | None = None,
        event_store: EventStore | None = None,
        idempotency_store: IdempotencyStore | None = None,
        capability_grant: CapabilityGrant | None = None,
        input_guardrails: Sequence[InputGuardrail] = (),
        output_guardrails: Sequence[OutputGuardrail] = (),
        prompt_injection_defense: PromptInjectionDefense | None = None,
        rate_limiter: RateLimiter | None = None,
        execution_budget: ExecutionBudget | None = None,
        deadline: Deadline | None = None,
        loop_detector: LoopDetector | None = None,
        cost_estimator: Callable[[ModelResponse], float] | None = None,
        finalizers: Sequence[Finalizer] = (),
        metrics: RuntimeMetrics | None = None,
        token_ledger: TokenUsageRecorder | None = None,
        response_failure_classifier: (
            Callable[[ModelResponse], FailureDisposition | None] | None
        ) = None,
        boundary_guardrail_policy: (
            BoundaryGuardrailPolicy | None
        ) = BoundaryGuardrailPolicy(),
    ) -> None:
        self.provider = provider
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry
        self.tool_filter = (
            tool_registry.visibility_filter(tool_filter)
            if isinstance(tool_registry, Toolbase)
            else tool_filter
        )
        self.context_assembler = context_assembler or ContextAssembler()
        self.event_store = event_store
        self.idempotency_store = idempotency_store
        self.capability_grant = capability_grant
        default_inputs = (
            (make_default_input_guardrail(boundary_guardrail_policy),)
            if boundary_guardrail_policy is not None
            else ()
        )
        default_outputs = (
            (make_default_output_guardrail(boundary_guardrail_policy),)
            if boundary_guardrail_policy is not None
            else ()
        )
        self._default_input_guardrail_count = len(default_inputs)
        self._default_output_guardrail_count = len(default_outputs)
        self.input_guardrails = default_inputs + tuple(input_guardrails)
        self.output_guardrails = default_outputs + tuple(output_guardrails)
        self.prompt_injection_defense = (
            prompt_injection_defense or PromptInjectionDefense()
        )
        self.rate_limiter = rate_limiter
        self.execution_budget = execution_budget
        self.deadline = deadline
        self.loop_detector = loop_detector
        self.cost_estimator = cost_estimator
        self.finalizers = tuple(finalizers)
        self.metrics = metrics
        self.token_ledger = token_ledger
        self.response_failure_classifier = response_failure_classifier

    def compile_plan(
        self,
        agent: AgentConfig,
        context_items: Sequence[ContextItem] = (),
        context_policy: ContextSelectionPolicy | None = None,
    ) -> CompiledExecutionPlan:
        return plan_compiler.compile_plan(self, agent, context_items, context_policy)

    async def run(
        self,
        agent: AgentConfig,
        messages: Sequence[ModelMessage],
        *,
        limits: AgentRunLimits = AgentRunLimits(),
        stop_requested: Callable[[], bool] | None = None,
        on_stream_event: StreamEventHandler | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_context: Mapping[str, Any] = {},
        workflow_state: WorkflowState | None = None,
        context_items: Sequence[ContextItem] = (),
        context_policy: ContextSelectionPolicy | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        resume_checkpoint: AgentCheckpoint | None = None,
        task_id: str | None = None,
    ) -> AgentRunResult:
        try:
            return await self._run_impl(
                agent,
                messages,
                limits=limits,
                stop_requested=stop_requested,
                on_stream_event=on_stream_event,
                cancellation_token=cancellation_token,
                tool_context=tool_context,
                workflow_state=workflow_state,
                context_items=context_items,
                context_policy=context_policy,
                context_metadata=context_metadata,
                resume_checkpoint=resume_checkpoint,
                task_id=task_id,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            # asyncio.run() translates Ctrl+C into cancellation of the main
            # task before re-raising KeyboardInterrupt. Mark the shared token
            # as cancelled as well so provider/tool implementations that own
            # work outside the immediate await chain can tear it down.
            if cancellation_token is not None:
                cancellation_token.cancel()
            raise
        finally:
            await run_finalizers(self.finalizers)

    async def _run_impl(
        self,
        agent: AgentConfig,
        messages: Sequence[ModelMessage],
        *,
        limits: AgentRunLimits = AgentRunLimits(),
        stop_requested: Callable[[], bool] | None = None,
        on_stream_event: StreamEventHandler | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_context: Mapping[str, Any] = {},
        workflow_state: WorkflowState | None = None,
        context_items: Sequence[ContextItem] = (),
        context_policy: ContextSelectionPolicy | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        resume_checkpoint: AgentCheckpoint | None = None,
        task_id: str | None = None,
    ) -> AgentRunResult:
        user_prompt = ""
        for message in reversed(messages):
            if message.role == "user":
                user_prompt = "".join(
                    part.text or "" for part in message.content if part.type == "text"
                )
                break
        if agent.memory:
            context_items = (
                *context_items,
                *await resolve_agent_memory(agent.memory, user_prompt),
            )

        guarded_messages: Sequence[ModelMessage] = tuple(messages)
        for guardrail in self.input_guardrails:
            guarded = _apply_guardrail_result(
                guarded_messages,
                guardrail(guarded_messages),
            )
            if not isinstance(guarded, Sequence):
                raise TypeError(
                    "input guardrail transform must return a message sequence"
                )
            guarded_messages = tuple(guarded)

        if resume_checkpoint is not None:
            if resume_checkpoint.agent_name != agent.name:
                raise ValueError(
                    f"checkpoint belongs to agent {resume_checkpoint.agent_name}, not {agent.name}"
                )
            history = list(resume_checkpoint.messages)
            if workflow_state is None:
                workflow_state = resume_checkpoint.workflow_state
            turns = resume_checkpoint.turns
            tool_calls = resume_checkpoint.tool_calls
            total_tokens = resume_checkpoint.total_tokens
        else:
            history = [
                ModelMessage(
                    role="system",
                    content=(ContentPart(type="text", text=agent.instructions),),
                ),
                *guarded_messages,
            ]
            turns = 0
            tool_calls = 0
            total_tokens = 0
        started = monotonic()
        # Trajectory state must never leak between runs (or concurrent runs
        # sharing one AgentLoop), so each run observes its own copy.
        loop_detector: LoopDetector | None = None
        if self.loop_detector is not None:
            loop_detector = copy.copy(self.loop_detector)
            loop_detector.reset()
        deadline_hit = False
        active_deadline = self.deadline
        if active_deadline is None and limits.timeout_seconds is not None:
            active_deadline = Deadline.after(limits.timeout_seconds)
        deadline_checks_enabled = (
            cancellation_token is not None
            or limits.timeout_seconds is not None
            or active_deadline is not None
        )
        response: ModelResponse | None = None
        structured_output: Any = None
        repair_attempts = 0
        events_enabled = self.event_store is not None and task_id is not None
        plan = self.compile_plan(agent, context_items, context_policy)
        tool_policy = plan.tool_policy

        def is_simple_text_message(message: ModelMessage) -> bool:
            return (
                message.role != "tool"
                and not message.tool_calls
                and all(part.type == "text" for part in message.content)
            )

        simple_text_request = (
            plan.simple_text_fast_path
            and workflow_state is None
            and not context_items
            and context_policy is None
            and not context_metadata
            and all(is_simple_text_message(message) for message in history)
        )
        active_visible_tool_names: set[str] = set()

        record_stage = partial(run_steps.record_stage, self)

        event_counter = (
            len(self.event_store.list(task_id))
            if self.event_store is not None and task_id is not None
            else 0
        )

        def emit_event(
            event_type: DurableEventType,
            payload: Mapping[str, JSONValue] | None = None,
        ) -> None:
            nonlocal event_counter
            if self.event_store is None or task_id is None:
                return
            # The counter starts from the visible event count, which can lag
            # behind the ids already used once a retention policy has archived
            # or expired events, so skip ids that are taken.
            while True:
                event_counter += 1
                try:
                    self.event_store.append(
                        DurableEvent(
                            event_id=f"{task_id}:run:{event_counter}",
                            task_id=task_id,
                            type=event_type,
                            payload=payload or {},
                        )
                    )
                    return
                except ValueError as exc:
                    if "already exists" not in str(exc):
                        raise

        if events_enabled:
            emit_event(
                "lifecycle_transition",
                {"from": "submitted", "to": "running"},
            )

        response_failure: FailureDisposition | None = None

        def finish(reason: TerminationReason) -> AgentRunResult:
            terminal = (
                "completed"
                if reason == "completed"
                else (
                    "canceled"
                    if reason == "cancelled"
                    else (
                        "waiting_for_approval"
                        if reason == "waiting_for_approval"
                        else "failed"
                    )
                )
            )
            if events_enabled:
                emit_event(
                    "lifecycle_transition",
                    {"from": "running", "to": terminal, "reason": reason},
                )
            return AgentRunResult(
                messages=tuple(history),
                final_response=response,
                termination_reason=reason,
                turns=turns,
                tool_calls=tool_calls,
                total_tokens=total_tokens,
                structured_output=structured_output,
                failure=response_failure,
            )

        run = _RunState(
            agent=agent,
            limits=limits,
            stop_requested=stop_requested,
            on_stream_event=on_stream_event,
            cancellation_token=cancellation_token,
            tool_context=tool_context,
            task_id=task_id,
            user_prompt=user_prompt,
            history=history,
            started=started,
            active_deadline=active_deadline,
            deadline_checks_enabled=deadline_checks_enabled,
            deadline_hit=deadline_hit,
            tool_policy=tool_policy,
            active_visible_tool_names=active_visible_tool_names,
            emit_event=emit_event,
        )
        run.await_with_deadline = partial(run_steps.await_with_deadline, run)
        run.await_tool_operation = partial(run_steps.await_tool_operation, run)
        run.record_token_usage = partial(run_steps.record_token_usage, self, run)
        execute_tool_call = partial(tool_execution.execute_tool_call, self, run)
        complete_request = partial(run_steps.complete_request, self, run)
        pending_approval_calls = partial(run_steps.pending_approval_calls, run)

        if resume_checkpoint is not None:
            pending = pending_approval_calls()
            if pending:
                if self.tool_filter is not None:
                    selected_names = set(
                        await _call_maybe_async(
                            self.tool_filter,
                            ToolFilterContext(
                                agent=agent,
                                messages=tuple(history),
                                turn=turns,
                                tool_calls=tool_calls,
                                runtime_context=tool_context,
                            ),
                            plan.visible_tools,
                        )
                    )
                    run.active_visible_tool_names = {
                        tool.name
                        for tool in plan.visible_tools
                        if tool.name in selected_names
                    }
                else:
                    run.active_visible_tool_names = set(plan.visible_tool_names)
                if tool_calls + len(pending) > limits.max_tool_calls:
                    return finish("max_tool_calls")
                for call, index in pending:
                    resumed_message, resumed_termination = await execute_tool_call(call)
                    if resumed_termination is not None:
                        return finish(resumed_termination)
                    if resumed_message is not None:
                        history[index] = resumed_message
                        tool_calls += 1

        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                return finish("cancelled")
            if stop_requested is not None and stop_requested():
                return finish("stop_requested")
            if turns >= limits.max_turns:
                return finish("max_turns")
            if (
                limits.max_total_tokens is not None
                and total_tokens >= limits.max_total_tokens
            ):
                return finish("budget_exhausted")
            if (
                limits.timeout_seconds is not None
                and monotonic() - started >= limits.timeout_seconds
            ):
                return finish("timeout")
            if active_deadline is not None and active_deadline.expired:
                return finish("timeout")
            if (
                self.tool_registry is not None
                and self.tool_registry.version != plan.tool_registry_version
            ):
                plan = self.compile_plan(agent, context_items, context_policy)
                run.tool_policy = plan.tool_policy
                simple_text_request = (
                    plan.simple_text_fast_path
                    and workflow_state is None
                    and not context_items
                    and context_policy is None
                    and not context_metadata
                    and all(is_simple_text_message(message) for message in history)
                )

            visible_tools = plan.visible_tools
            tool_selection = plan.tool_selection
            if self.tool_filter is not None:
                context = ToolFilterContext(
                    agent=agent,
                    messages=tuple(history),
                    turn=turns,
                    tool_calls=tool_calls,
                    runtime_context=tool_context,
                )
                selected_names = set(
                    await _call_maybe_async(self.tool_filter, context, visible_tools)
                )
                visible_tools = tuple(
                    tool for tool in visible_tools if tool.name in selected_names
                )
                run.active_visible_tool_names = {tool.name for tool in visible_tools}
                if run.tool_policy is not None:
                    missing_required = (
                        run.tool_policy.required - run.active_visible_tool_names
                    )
                    if missing_required:
                        raise ValueError(
                            "required tools are unavailable: "
                            + ", ".join(sorted(missing_required))
                        )
                    tool_selection = ToolSelectionRequirement(
                        required=tuple(sorted(run.tool_policy.required)),
                        preferred=tuple(
                            name
                            for name in run.tool_policy.preferred
                            if name in run.active_visible_tool_names
                        ),
                    )
                else:
                    tool_selection = None
            else:
                run.active_visible_tool_names = set(plan.visible_tool_names)

            structured_requirement = plan.structured_output
            request_assembly_started = monotonic() if self.metrics is not None else None
            if simple_text_request:
                request = ModelRequest(
                    messages=tuple(history),
                    model=plan.model_settings.model,
                    temperature=plan.model_settings.temperature,
                    max_output_tokens=plan.model_settings.max_output_tokens,
                    metadata=dict(plan.model_settings.metadata),
                    cancellation_token=cancellation_token,
                )
            else:
                request = self.context_assembler.assemble_request(
                    agent,
                    tuple(history),
                    tools=visible_tools,
                    workflow_state=workflow_state,
                    context_items=context_items,
                    runtime_metadata=context_metadata,
                    policy=context_policy,
                    structured_output=structured_requirement,
                    tool_selection=tool_selection,
                    cancellation_token=cancellation_token,
                )
            record_stage(
                "request_assembly",
                request_assembly_started,
                path="fast" if simple_text_request else "general",
            )
            model_call_started = monotonic() if self.metrics is not None else None
            try:
                response = await complete_request(request)
            except RunCancelled:
                return finish("cancelled")
            except asyncio.TimeoutError:
                if not run.deadline_hit:
                    raise
                return finish("timeout")
            except BudgetExceededError:
                return finish("budget_exhausted")
            finally:
                record_stage("model_call", model_call_started)

            output_guardrail_started = (
                monotonic()
                if self.metrics is not None and self.output_guardrails
                else None
            )
            guarded_message = response.message
            for output_guardrail in self.output_guardrails:
                guarded = _apply_guardrail_result(
                    guarded_message,
                    output_guardrail(guarded_message),
                )
                if not isinstance(guarded, ModelMessage):
                    raise TypeError(
                        "output guardrail transform must return ModelMessage"
                    )
                guarded_message = guarded
            record_stage("output_guardrails", output_guardrail_started)
            if guarded_message is not response.message:
                response = replace(response, message=guarded_message)

            turns += 1
            history.append(response.message)
            if simple_text_request:
                simple_text_request = is_simple_text_message(response.message)
            if loop_detector is not None and loop_detector.observe(response.message):
                return finish("loop_detected")
            if response.usage is not None:
                used = response.usage.total_tokens
                if used is None:
                    used = (response.usage.input_tokens or 0) + (
                        response.usage.output_tokens or 0
                    )
                total_tokens += used
            if (
                limits.max_total_tokens is not None
                and total_tokens >= limits.max_total_tokens
            ):
                return finish("budget_exhausted")
            if self.response_failure_classifier is not None:
                response_failure = await _call_maybe_async(
                    self.response_failure_classifier, response
                )
                if response_failure is not None:
                    return finish("model_response_failure")

            calls = tuple(response.message.tool_calls)
            if calls and self.tool_registry is None and self.tool_executor is None:
                raise RuntimeError(
                    "model requested tool calls but no tool executor is configured"
                )
            if not calls:
                requirements = agent.output
                if requirements is not None and (
                    requirements.format == "json" or requirements.schema is not None
                ):
                    try:
                        structured_output = _validate_structured_message(
                            response.message,
                            requirements,
                        )
                    except StructuredOutputValidationError as exc:
                        if (
                            repair_attempts < max(requirements.max_repair_attempts, 0)
                            and turns < limits.max_turns
                        ):
                            repair_attempts += 1
                            history.append(
                                ModelMessage(
                                    role="user",
                                    content=(
                                        ContentPart(
                                            type="text",
                                            text=(
                                                "Your previous response did not satisfy the required "
                                                "structured output. Return corrected JSON only. "
                                                "Validation errors: "
                                                + "; ".join(exc.issues)
                                            ),
                                        ),
                                    ),
                                )
                            )
                            continue
                        raise
                return finish("completed")
            answered: set[int] = set()

            def record_result(call: ToolCall, message: ModelMessage) -> None:
                nonlocal tool_calls
                tool_calls += 1
                history.append(message)
                answered.add(id(call))  # noqa: B023 - closures run within this turn

            def finish_turn(termination: TerminationReason) -> AgentRunResult:
                # Keep the transcript provider-valid: every tool call the model
                # made gets a tool message, even when the run stops early.
                for call in calls:  # noqa: B023
                    if id(call) in answered:  # noqa: B023
                        continue
                    history.append(
                        ModelMessage(
                            role="tool",
                            content=(
                                ContentPart(
                                    type="json",
                                    data={
                                        "error": {
                                            "type": "tool_not_executed",
                                            "tool": call.name,
                                            "reason": termination,
                                        }
                                    },
                                ),
                            ),
                            tool_call_id=call.id,
                        )
                    )
                return finish(termination)

            async def run_batch(
                batch: Sequence[ToolCall],
            ) -> TerminationReason | None:
                outcomes = await asyncio.gather(
                    *(execute_tool_call(item) for item in batch),
                    return_exceptions=True,
                )
                failure: BaseException | None = None
                termination: TerminationReason | None = None
                for item, outcome in zip(batch, outcomes):
                    if isinstance(outcome, BaseException):
                        failure = failure or outcome
                        continue
                    message, reason = outcome
                    if reason is not None:
                        termination = termination or reason
                    elif message is not None:
                        record_result(item, message)
                if failure is not None:
                    raise failure
                return termination

            if tool_calls + len(calls) > limits.max_tool_calls:
                return finish_turn("max_tool_calls")

            if limits.concurrent_tool_calls and len(calls) > 1:
                parallel_batch: list[ToolCall] = []
                for call in calls:
                    sequential = False
                    if self.tool_registry is not None:
                        try:
                            sequential = (
                                self.tool_registry.get(
                                    call.name
                                ).definition.execution_mode
                                == "sequential"
                            )
                        except KeyError:
                            sequential = False

                    if sequential:
                        if parallel_batch:
                            termination = await run_batch(parallel_batch)
                            parallel_batch = []
                            if termination is not None:
                                return finish_turn(termination)

                        tool_message, termination = await execute_tool_call(call)
                        if termination is not None:
                            return finish_turn(termination)
                        if tool_message is not None:
                            record_result(call, tool_message)
                    else:
                        parallel_batch.append(call)

                if parallel_batch:
                    termination = await run_batch(parallel_batch)
                    if termination is not None:
                        return finish_turn(termination)
            else:
                for call in calls:
                    tool_message, termination = await execute_tool_call(call)
                    if termination is not None:
                        return finish_turn(termination)
                    if tool_message is not None:
                        record_result(call, tool_message)

    async def run_streaming(
        self,
        agent: AgentConfig,
        messages: Sequence[ModelMessage],
        on_event: StreamEventHandler,
        *,
        limits: AgentRunLimits = AgentRunLimits(),
        stop_requested: Callable[[], bool] | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_context: Mapping[str, Any] = {},
        workflow_state: WorkflowState | None = None,
        context_items: Sequence[ContextItem] = (),
        context_policy: ContextSelectionPolicy | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        resume_checkpoint: AgentCheckpoint | None = None,
        task_id: str | None = None,
    ) -> AgentRunResult:
        return await self.run(
            agent,
            messages,
            limits=limits,
            stop_requested=stop_requested,
            on_stream_event=on_event,
            cancellation_token=cancellation_token,
            tool_context=tool_context,
            workflow_state=workflow_state,
            context_items=context_items,
            context_policy=context_policy,
            context_metadata=context_metadata,
            resume_checkpoint=resume_checkpoint,
            task_id=task_id,
        )
