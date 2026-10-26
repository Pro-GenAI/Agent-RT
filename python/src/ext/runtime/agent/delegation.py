"""Agent invocation, agents as tools, handoffs, subagents, and supervisor teams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ext.runtime.agent.config import AgentConfig
from ext.runtime.agent.limits import AgentRunResult
from ext.runtime.authorization.grants import CapabilityGrant
from ext.runtime.context.items import ContextItem
from ext.runtime.resilience.budgets import ExecutionBudget
from ext.runtime.tools.definitions import (
    ContextualToolHandler,
    ToolDefinition,
    ToolExecutionContext,
)
from ext.runtime.types import ContentPart, ModelMessage, WorkflowState


@dataclass(frozen=True)
class AgentInvocation:
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    allowed_tools: frozenset[str] = frozenset()
    capability_grant: CapabilityGrant | None = None
    execution_budget: ExecutionBudget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentInvoker(Protocol):
    async def invoke(
        self,
        agent: AgentConfig,
        invocation: AgentInvocation,
    ) -> AgentRunResult: ...


@dataclass(frozen=True)
class AgentTool:
    definition: ToolDefinition
    handler: ContextualToolHandler


def agent_as_tool(
    agent: AgentConfig,
    invoker: AgentInvoker,
    *,
    name: str | None = None,
    description: str | None = None,
) -> AgentTool:
    tool_name = name or f"agent.{agent.name}"

    async def handler(
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        prompt = arguments.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("agent tool requires a non-empty input")
        result = await invoker.invoke(
            agent,
            AgentInvocation(
                messages=(
                    ModelMessage(
                        role="user",
                        content=(ContentPart(type="text", text=prompt),),
                    ),
                ),
                metadata={
                    "parent_tool": tool_name,
                    "parent_request_context": dict(context.request_context),
                },
            ),
        )
        if result.structured_output is not None:
            return result.structured_output
        if result.final_response is not None:
            texts = [
                part.text
                for part in result.final_response.message.content
                if part.type == "text" and part.text is not None
            ]
            if texts:
                return "".join(texts)
        return {
            "termination_reason": result.termination_reason,
            "turns": result.turns,
        }

    return AgentTool(
        definition=ToolDefinition(
            name=tool_name,
            description=description
            or agent.description
            or f"Delegate bounded work to {agent.name}.",
            input_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
            side_effect="none",
        ),
        handler=handler,
    )


@dataclass(frozen=True)
class HandoffRequest:
    target: str
    messages: tuple[ModelMessage, ...]
    reason: str
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandoffResult:
    owner: str
    result: AgentRunResult
    reason: str


class HandoffManager:
    def __init__(
        self,
        specialists: Mapping[str, tuple[AgentConfig, AgentInvoker]],
    ) -> None:
        self.specialists = dict(specialists)

    async def handoff(self, request: HandoffRequest) -> HandoffResult:
        if request.target not in self.specialists:
            raise KeyError(request.target)
        agent, invoker = self.specialists[request.target]
        result = await invoker.invoke(
            agent,
            AgentInvocation(
                messages=request.messages,
                context_items=request.context_items,
                workflow_state=request.workflow_state,
                metadata={
                    **dict(request.metadata),
                    "handoff_reason": request.reason,
                    "handoff_target": request.target,
                },
            ),
        )
        return HandoffResult(
            owner=request.target,
            result=result,
            reason=request.reason,
        )


@dataclass(frozen=True)
class SubagentSpec:
    agent: AgentConfig
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    allowed_tools: frozenset[str] = frozenset()
    capability_grant: CapabilityGrant | None = None
    execution_budget: ExecutionBudget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class IsolatedSubagentRunner:
    def __init__(
        self,
        invoker: AgentInvoker,
        *,
        parent_budget: ExecutionBudget | None = None,
    ) -> None:
        self.invoker = invoker
        self.parent_budget = parent_budget

    async def run(self, spec: SubagentSpec) -> AgentRunResult:
        if self.parent_budget is not None:
            self.parent_budget.consume_subagent()
        return await self.invoker.invoke(
            spec.agent,
            AgentInvocation(
                messages=spec.messages,
                context_items=spec.context_items,
                workflow_state=spec.workflow_state,
                allowed_tools=spec.allowed_tools,
                capability_grant=spec.capability_grant,
                execution_budget=spec.execution_budget,
                metadata={**dict(spec.metadata), "isolated_subagent": True},
            ),
        )


@dataclass(frozen=True)
class WorkerAssignment:
    id: str
    worker: str
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerResult:
    assignment_id: str
    worker: str
    result: AgentRunResult


class SupervisorWorkerTeam:
    def __init__(
        self,
        workers: Mapping[str, tuple[AgentConfig, IsolatedSubagentRunner]],
    ) -> None:
        self.workers = dict(workers)

    async def delegate(
        self,
        assignments: Sequence[WorkerAssignment],
    ) -> tuple[WorkerResult, ...]:
        results: list[WorkerResult] = []
        for assignment in assignments:
            if assignment.worker not in self.workers:
                raise KeyError(assignment.worker)
            agent, runner = self.workers[assignment.worker]
            result = await runner.run(
                SubagentSpec(
                    agent=agent,
                    messages=assignment.messages,
                    context_items=assignment.context_items,
                    metadata={
                        **dict(assignment.metadata),
                        "assignment_id": assignment.id,
                        "supervised": True,
                    },
                )
            )
            results.append(
                WorkerResult(
                    assignment_id=assignment.id,
                    worker=assignment.worker,
                    result=result,
                )
            )
        return tuple(results)
