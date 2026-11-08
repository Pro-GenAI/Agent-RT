"""Offline regression tests for governed subagent delegation and concurrent review."""

import asyncio

import pytest

from agent_rt import (
    AgentConfig,
    AgentRunResult,
    CapabilityGrant,
    ContentPart,
    ExecutionBudget,
    ExecutionBudgetLimits,
    ModelMessage,
    ModelSettings,
    SubagentSwarm,
    ToolDefinition,
)


def config(name: str) -> AgentConfig:
    return AgentConfig(name=name, instructions="Finish assigned task", model=ModelSettings(model="mock"))


def tools() -> list[ToolDefinition]:
    return [
        ToolDefinition("lookup", "Read data", {"type": "object"}, side_effect="read"),
        ToolDefinition("edit", "Write data", {"type": "object"}, side_effect="write"),
    ]


def ok(name: str) -> AgentRunResult:
    return AgentRunResult(
        messages=(),
        final_response=None,
        termination_reason="completed",
        turns=1,
        tool_calls=0,
        total_tokens=0,
        structured_output={"agent": name},
    )


@pytest.mark.asyncio
async def test_full_history_nested_depth_review_and_concurrency():
    seen = []
    active = 0
    peak = 0

    class FakeInvoker:
        async def invoke(self, agent, invocation):
            nonlocal active, peak
            seen.append((agent.name, invocation))
            if agent.name == "reviewer":
                assert invocation.allowed_tools == frozenset()
                assert invocation.capability_grant.tools == ()
                assert invocation.subagents is None
                assert agent.tool_policy.allowed == frozenset()
                return ok("reviewed")
            active += 1
            peak = max(peak, active)
            try:
                assert invocation.messages[0].content[0].text == "all prior chat"
                assert invocation.subagents is not None
                if agent.name == "worker":
                    assert invocation.subagents.depth == 2
                    nested = await invocation.subagents.spawn(
                        "grandchild task", config("grandchild"),
                        allowed_tools={"lookup": "read"},
                    )
                    assert nested.response == '{"agent": "grandchild"}'
                    assert nested.review.response is not None
                    with pytest.raises(ValueError, match="only one child"):
                        await invocation.subagents.spawn(
                            "another", config("grandchild"), allowed_tools={}
                        )
                elif agent.name == "grandchild":
                    assert invocation.subagents.depth == 3
                    with pytest.raises(ValueError, match="depth limit"):
                        await invocation.subagents.spawn(
                            "forbidden", config("worker"), allowed_tools={}
                        )
                await asyncio.sleep(0)
                return ok(agent.name)
            finally:
                active -= 1

    history = [ModelMessage(role="user", content=(ContentPart(type="text", text="all prior chat"),))]
    budget = ExecutionBudget(ExecutionBudgetLimits(max_subagents=4))
    swarm = SubagentSwarm(
        FakeInvoker(),
        chat_history=history,
        tool_definitions=tools(),
        parent_permissions={"lookup": "read", "edit": "write"},
        reviewer=config("reviewer"),
        budget=budget,
    )
    results = await asyncio.gather(
        swarm.deploy("work A", config("worker"), allowed_tools={"lookup": "read"}),
        swarm.deploy("work B", config("worker"), allowed_tools={"lookup": "read"}),
    )
    assert all(outcome.response == '{"agent": "worker"}' for outcome in results)
    assert all(outcome.review.response == '{"agent": "reviewed"}' for outcome in results)
    assert all(outcome.error is None for outcome in results)
    assert peak >= 2
    assert swarm.active == 0
    assert budget.subagents == 4
    spans = swarm.traces.spans(trace_id=swarm.trace_id)
    assert len([span for span in spans if span.kind == "subagent"]) == 4
    assert len([span for span in spans if span.attributes.get("role") == "reviewer"]) == 4
    assert all(span.status == "ok" for span in spans)
    assert swarm.metrics.total("subagent.started") == 4
    assert swarm.metrics.total("subagent.review.completed") == 4
    assert max(swarm.metrics.values("subagent.active")) >= 2
    swarm.close()


@pytest.mark.asyncio
async def test_permissions_fail_closed_without_consuming_slots():
    class FakeInvoker:
        async def invoke(self, agent, invocation):
            return ok(agent.name)

    swarm = SubagentSwarm(
        FakeInvoker(),
        chat_history=(),
        tool_definitions=tools(),
        parent_permissions={"lookup": "read", "edit": "write"},
        reviewer=config("reviewer"),
        capability_grant=CapabilityGrant(tools=("lookup", "edit")),
    )
    with pytest.raises(PermissionError, match="not read-only"):
        await swarm.deploy("task", config("worker"), allowed_tools={"edit": "read"})
    with pytest.raises(PermissionError, match="escalate"):
        await swarm.deploy("task", config("worker"), allowed_tools={"missing": "write"})
    # The initial invalid attempts don't allocate a child slot.
    assert swarm.metrics.total("subagent.started") == 0
    result = await swarm.deploy("task", config("worker"), allowed_tools={"lookup": "read"})
    assert result.error is None
    assert result.result is not None
    swarm.close()
    with pytest.raises(PermissionError, match="parent grant"):
        SubagentSwarm(
            FakeInvoker(), chat_history=(), tool_definitions=tools(),
            parent_permissions={"edit": "write"},
            reviewer=config("reviewer"),
            capability_grant=CapabilityGrant(tools=("lookup",)),
        )


@pytest.mark.asyncio
async def test_worker_and_reviewer_failures_are_attributed():
    class FakeInvoker:
        async def invoke(self, agent, invocation):
            if agent.name == "worker":
                raise RuntimeError("worker failure")
            raise RuntimeError("reviewer failure")

    swarm = SubagentSwarm(
        FakeInvoker(), chat_history=(), tool_definitions=tools(),
        parent_permissions={}, reviewer=config("reviewer"),
    )
    result = await swarm.deploy("task", config("worker"), allowed_tools={})
    assert result.result is None
    assert result.error == "RuntimeError: worker failure"
    assert result.review.error == "RuntimeError: reviewer failure"
    assert swarm.metrics.total("subagent.failed") == 1
    assert swarm.metrics.total("subagent.review.failed") == 1
    assert len([s for s in swarm.traces.spans() if s.status == "error"]) == 2
    swarm.close()
