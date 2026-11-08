"""Governed task delegation with bounded depth, reviewer reports and tracing.

The invoker must execute AgentConfig.tool_policy (as AgentLoop does). The
swarm validates permissions against a trusted tool catalog before invoking it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4

from ext.runtime.agent.config import AgentConfig
from ext.runtime.agent.delegation import (
    AgentInvocation,
    AgentInvoker,
    IsolatedSubagentRunner,
    SubagentSpec,
)
from ext.runtime.agent.limits import AgentRunResult
from ext.runtime.authorization.grants import CapabilityGrant
from ext.runtime.observability.metrics import RuntimeMetrics
from ext.runtime.observability.tracing import TraceRecorder
from ext.runtime.resilience.budgets import ExecutionBudget
from ext.runtime.tools.definitions import ToolDefinition, ToolSelectionPolicy
from ext.runtime.types import ContentPart, ModelMessage

ToolPermission = Literal["read", "write"]


def _text(message: str) -> ModelMessage:
    return ModelMessage(role="user", content=(ContentPart(type="text", text=message),))


def _response(result: AgentRunResult | None) -> str | None:
    if result is None:
        return None
    if result.structured_output is not None:
        return json.dumps(result.structured_output, default=str)
    if result.final_response is None:
        return None
    return "".join(
        part.text or ""
        for part in result.final_response.message.content
        if part.type == "text"
    )


@dataclass(frozen=True)
class ReviewerReport:
    response: str | None
    error: str | None = None


@dataclass(frozen=True)
class SubagentOutcome:
    task_id: str
    agent: str
    depth: int
    result: AgentRunResult | None
    response: str | None
    review: ReviewerReport
    error: str | None = None


class SubagentScope:
    """An invocation-scoped handle. Root can fan out; depth-2 has one child."""

    def __init__(
        self,
        swarm: SubagentSwarm,
        *,
        depth: int,
        history: tuple[ModelMessage, ...],
        permissions: Mapping[str, ToolPermission],
        span_id: str,
        parent_agent: str,
        owner_agent: str,
    ) -> None:
        self._swarm = swarm
        self.depth = depth
        self.history = history
        self.permissions = dict(permissions)
        self.span_id = span_id
        self.parent_agent = parent_agent
        self.owner_agent = owner_agent
        self._spawned = 0

    async def spawn(
        self,
        task: str,
        agent: AgentConfig,
        *,
        allowed_tools: Mapping[str, ToolPermission],
        reviewer: AgentConfig | None = None,
    ) -> SubagentOutcome:
        """Delegate a task with the entire inherited chat history."""
        if not task.strip():
            raise ValueError("subagent task must not be empty")
        child_permissions = dict(allowed_tools)
        swarm = self._swarm
        with swarm._lock:
            if self.depth >= 3:
                swarm.metrics.increment("subagent.denied", labels={"reason": "depth"})
                raise ValueError("subagent depth limit (3) exceeded")
            if self.depth > 1 and self._spawned >= 1:
                swarm.metrics.increment("subagent.denied", labels={"reason": "fanout"})
                raise ValueError("a subagent may launch only one child")
            for name, level in child_permissions.items():
                if level not in ("read", "write") or name not in swarm.catalog:
                    raise PermissionError(f"unknown tool or access level: {name}")
                parent_level = self.permissions.get(name)
                if parent_level is None or (parent_level == "read" and level == "write"):
                    raise PermissionError(f"cannot escalate tool permission: {name}")
                side_effect = swarm.catalog[name].side_effect
                if level == "read" and side_effect not in ("none", "read"):
                    raise PermissionError(f"tool is not read-only: {name}")
                if agent.tool_policy is not None and not agent.tool_policy.permits(name):
                    raise PermissionError(f"agent policy denies tool: {name}")
            if agent.tool_policy and not agent.tool_policy.required.issubset(child_permissions):
                raise PermissionError("required tools unavailable to subagent")
            # Reserve the slot before any await, even for concurrent siblings.
            if swarm.budget is not None:
                swarm.budget.consume_subagent()
            self._spawned += 1
            task_id = uuid4().hex
            span_id = uuid4().hex
            history = (*self.history, _text(task))
            scope = SubagentScope(
                swarm,
                depth=self.depth + 1,
                history=history,
                permissions=child_permissions,
                span_id=span_id,
                parent_agent=self.owner_agent,
                owner_agent=agent.name,
            )
            swarm.traces.start(
                span_id=span_id,
                trace_id=swarm.trace_id,
                parent_span_id=self.span_id,
                name=agent.name,
                kind="subagent",
                task_id=task_id,
                attributes={
                    "depth": scope.depth,
                    "parent_agent": self.parent_agent,
                    "tool_count": len(child_permissions),
                },
            )
            swarm._active += 1
            swarm.metrics.increment("subagent.started", labels={"agent": agent.name})
            swarm.metrics.record("subagent.active", swarm._active, kind="gauge")
        return await swarm._execute(
            scope=scope,
            task_id=task_id,
            agent=agent,
            reviewer=reviewer or swarm.reviewer,
        )


class SubagentSwarm:
    """Shared execution/monitoring root; each child has an independent scope.

    Depth counts the main agent as 1. Children at depth 2 may each launch one
    depth-3 child; depth-3 agents (and all reviewers) cannot launch agents.
    """

    def __init__(
        self,
        invoker: AgentInvoker,
        *,
        chat_history: Sequence[ModelMessage],
        tool_definitions: Sequence[ToolDefinition],
        parent_permissions: Mapping[str, ToolPermission],
        reviewer: AgentConfig,
        budget: ExecutionBudget | None = None,
        capability_grant: CapabilityGrant | None = None,
        traces: TraceRecorder | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self.invoker = invoker
        self.reviewer = reviewer
        self.budget = budget
        self.capability_grant = capability_grant
        self.catalog = {tool.name: tool for tool in tool_definitions}
        if len(self.catalog) != len(tool_definitions):
            raise ValueError("duplicate tool definitions")
        for name, level in parent_permissions.items():
            if name not in self.catalog or level not in ("read", "write"):
                raise ValueError(f"invalid parent tool permission: {name}")
            if level == "read" and self.catalog[name].side_effect not in ("none", "read"):
                raise ValueError(f"tool is not read-only: {name}")
            if capability_grant is not None and not capability_grant.allows_tool(name):
                raise PermissionError(f"parent grant denies tool: {name}")
        self._lock = threading.RLock()
        self._active = 0
        self.traces = traces or TraceRecorder()
        self.metrics = metrics or RuntimeMetrics()
        self.trace_id = uuid4().hex
        root_id = uuid4().hex
        with self._lock:
            self.traces.start(
                span_id=root_id,
                trace_id=self.trace_id,
                name="main-agent",
                kind="agent",
            )
        self.root = SubagentScope(
            self,
            depth=1,
            history=tuple(chat_history),
            permissions=parent_permissions,
            span_id=root_id,
            parent_agent="main-agent",
            owner_agent="main-agent",
        )

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    async def deploy(
        self,
        task: str,
        agent: AgentConfig,
        *,
        allowed_tools: Mapping[str, ToolPermission],
        reviewer: AgentConfig | None = None,
    ) -> SubagentOutcome:
        return await self.root.spawn(
            task, agent, allowed_tools=allowed_tools, reviewer=reviewer
        )

    async def _execute(
        self,
        *,
        scope: SubagentScope,
        task_id: str,
        agent: AgentConfig,
        reviewer: AgentConfig,
    ) -> SubagentOutcome:
        names = frozenset(scope.permissions)
        existing = agent.tool_policy
        restricted = replace(
            agent,
            tool_policy=ToolSelectionPolicy(
                allowed=names,
                denied=existing.denied if existing else frozenset(),
                required=existing.required if existing else frozenset(),
                preferred=existing.preferred if existing else (),
            ),
        )
        inherited = self.capability_grant
        grant = CapabilityGrant(
            tools=tuple(sorted(names)),
            paths=inherited.paths if inherited else (),
            networks=inherited.networks if inherited else (),
            credential_ids=inherited.credential_ids if inherited else (),
        )
        result: AgentRunResult | None = None
        error: str | None = None
        review = ReviewerReport(None)
        try:
            try:
                result = await IsolatedSubagentRunner(self.invoker).run(
                    SubagentSpec(
                        agent=restricted,
                        messages=scope.history,
                        allowed_tools=names,
                        capability_grant=grant,
                        execution_budget=self.budget,
                        metadata={
                            "task_id": task_id,
                            "trace_id": self.trace_id,
                            "span_id": scope.span_id,
                            "depth": scope.depth,
                            "parent_agent": scope.parent_agent,
                            "tool_permissions": dict(scope.permissions),
                        },
                        subagents=scope,
                    )
                )
                if result.termination_reason != "completed":
                    error = f"worker terminated: {result.termination_reason}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            # Always review: successful work can still contain mistakes.
            review_id = uuid4().hex
            with self._lock:
                self.traces.start(
                    span_id=review_id,
                    trace_id=self.trace_id,
                    parent_span_id=scope.span_id,
                    name=reviewer.name,
                    kind="agent",
                    task_id=task_id,
                    attributes={"role": "reviewer", "depth": scope.depth},
                )
                self.metrics.increment("subagent.review.started")
            try:
                report_context = json.dumps({
                    "task_id": task_id,
                    "task": scope.history[-1].content[0].text,
                    "worker": agent.name,
                    "termination": result.termination_reason if result else None,
                    "response": _response(result),
                    "error": error,
                }, default=str)
                review_result = await self.invoker.invoke(
                    replace(
                        reviewer,
                        tool_policy=ToolSelectionPolicy(allowed=frozenset()),
                    ),
                    AgentInvocation(
                        messages=(*scope.history, _text(
                            "Review the delegated task for mistakes, failed actions, "
                            "policy violations and incomplete work. Report findings "
                            "explicitly; do not perform tools or delegate.\n" + report_context
                        )),
                        allowed_tools=frozenset(),
                        capability_grant=CapabilityGrant(),
                        metadata={
                            "task_id": task_id,
                            "trace_id": self.trace_id,
                            "span_id": review_id,
                            "reviewed_span_id": scope.span_id,
                        },
                    ),
                )
                review = ReviewerReport(
                    _response(review_result),
                    None if review_result.termination_reason == "completed" else
                    f"reviewer terminated: {review_result.termination_reason}",
                )
            except Exception as exc:
                review = ReviewerReport(None, f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self.traces.finish(
                        review_id,
                        status="error" if review.error else "ok",
                        attributes={"has_findings": bool(review.response)},
                    )
                    self.metrics.increment(
                        "subagent.review.failed" if review.error else "subagent.review.completed"
                    )
            return SubagentOutcome(
                task_id=task_id,
                agent=agent.name,
                depth=scope.depth,
                result=result,
                response=_response(result),
                review=review,
                error=error,
            )
        finally:
            with self._lock:
                finished = self.traces.finish(
                    scope.span_id,
                    status="error" if error else "ok",
                    attributes={
                        "termination": result.termination_reason if result else "error",
                        "review_failed": bool(review.error),
                    },
                )
                self.metrics.increment(
                    "subagent.failed" if error else "subagent.completed",
                    labels={"agent": agent.name},
                )
                self.metrics.record(
                    "subagent.duration_ms", float(finished.duration_ms or 0),
                    kind="histogram", labels={"agent": agent.name},
                )
                self._active -= 1
                self.metrics.record("subagent.active", self._active, kind="gauge")

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close swarm while subagents are active")
            if self.traces.get(self.root.span_id).status == "running":
                self.traces.finish(self.root.span_id)
