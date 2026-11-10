"""Agent routing, round-robin, speaker-selected, swarm, hierarchical, and speculative teams."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.agent.config import AgentConfig
from ext.runtime.agent.delegation import (
    AgentInvocation,
    AgentInvoker,
    IsolatedSubagentRunner,
    SubagentSpec,
)
from ext.runtime.agent.limits import AgentRunResult
from ext.runtime.types import asyncio


@dataclass(frozen=True)
class AgentRoute:
    name: str
    agent: AgentConfig
    invoker: AgentInvoker
    intents: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()


class AgentRouter:
    def __init__(self, routes: Sequence[AgentRoute]) -> None:
        self.routes = tuple(routes)

    def select(
        self,
        *,
        intent: str | None = None,
        required_capabilities: Sequence[str] = (),
    ) -> AgentRoute:
        required = set(required_capabilities)
        candidates: list[tuple[int, AgentRoute]] = []
        for route in self.routes:
            if required - set(route.capabilities):
                continue
            score = len(required & set(route.capabilities))
            if intent is not None and intent in route.intents:
                score += 1000
            candidates.append((score, route))
        if not candidates:
            raise LookupError("no matching agent route")
        candidates.sort(key=lambda item: (-item[0], item[1].name))
        return candidates[0][1]

    async def route(
        self,
        invocation: AgentInvocation,
        *,
        intent: str | None = None,
        required_capabilities: Sequence[str] = (),
    ) -> tuple[str, AgentRunResult]:
        selected = self.select(
            intent=intent,
            required_capabilities=required_capabilities,
        )
        return selected.name, await selected.invoker.invoke(
            selected.agent,
            invocation,
        )


@dataclass(frozen=True)
class TeamMember:
    name: str
    agent: AgentConfig
    invoker: AgentInvoker


@dataclass(frozen=True)
class TeamTurn:
    speaker: str
    result: AgentRunResult


class RoundRobinTeam:
    def __init__(self, members: Sequence[TeamMember]) -> None:
        if not members:
            raise ValueError("round-robin team requires at least one member")
        self.members = tuple(members)
        self._index = 0

    async def step(self, invocation: AgentInvocation) -> TeamTurn:
        member = self.members[self._index]
        self._index = (self._index + 1) % len(self.members)
        result = await member.invoker.invoke(member.agent, invocation)
        return TeamTurn(member.name, result)

    async def run(
        self,
        invocations: Sequence[AgentInvocation],
    ) -> tuple[TeamTurn, ...]:
        return tuple([await self.step(invocation) for invocation in invocations])


SpeakerSelector = Callable[
    [Sequence[TeamMember], AgentInvocation, Sequence[TeamTurn]],
    Awaitable[str],
]


class ModelSelectedSpeakerTeam:
    def __init__(
        self,
        members: Sequence[TeamMember],
        selector: SpeakerSelector,
    ) -> None:
        if not members:
            raise ValueError("speaker team requires at least one member")
        self.members = tuple(members)
        self.selector = selector
        self.history: list[TeamTurn] = []

    async def step(self, invocation: AgentInvocation) -> TeamTurn:
        selected_name = await self.selector(
            self.members,
            invocation,
            tuple(self.history),
        )
        member = next(
            (
                candidate
                for candidate in self.members
                if candidate.name == selected_name
            ),
            None,
        )
        if member is None:
            raise LookupError(f"selector chose unknown speaker: {selected_name}")
        turn = TeamTurn(
            member.name,
            await member.invoker.invoke(member.agent, invocation),
        )
        self.history.append(turn)
        return turn


@dataclass(frozen=True)
class SwarmDecision:
    next_agent: str | None = None
    reason: str = ""


SwarmHandoffPolicy = Callable[
    [str, "AgentRunResult"],
    Awaitable[SwarmDecision],
]


class SwarmTeam:
    def __init__(
        self,
        members: Mapping[str, tuple[AgentConfig, AgentInvoker]],
        handoff_policy: SwarmHandoffPolicy,
    ) -> None:
        self.members = dict(members)
        self.handoff_policy = handoff_policy

    async def run(
        self,
        start: str,
        invocation: AgentInvocation,
        *,
        max_handoffs: int = 8,
    ) -> tuple[TeamTurn, ...]:
        if start not in self.members:
            raise KeyError(start)
        turns: list[TeamTurn] = []
        current = start
        for _ in range(max_handoffs + 1):
            agent, invoker = self.members[current]
            result = await invoker.invoke(agent, invocation)
            turns.append(TeamTurn(current, result))
            decision = await self.handoff_policy(current, result)
            if decision.next_agent is None:
                return tuple(turns)
            if decision.next_agent not in self.members:
                raise KeyError(decision.next_agent)
            current = decision.next_agent
        raise RuntimeError("swarm handoff limit exceeded")


@runtime_checkable
class TeamNode(Protocol):
    async def execute(self, invocation: AgentInvocation) -> Any: ...


@dataclass(frozen=True)
class AgentTeamNode:
    agent: AgentConfig
    invoker: AgentInvoker

    async def execute(self, invocation: AgentInvocation) -> AgentRunResult:
        return await self.invoker.invoke(self.agent, invocation)


class HierarchicalTeam:
    def __init__(
        self,
        children: Mapping[str, TeamNode],
        selector: Callable[[AgentInvocation], Awaitable[str]],
    ) -> None:
        self.children = dict(children)
        self.selector = selector

    async def execute(self, invocation: AgentInvocation) -> Any:
        child_name = await self.selector(invocation)
        if child_name not in self.children:
            raise KeyError(child_name)
        return await self.children[child_name].execute(invocation)


class ParallelSubagentExecutor:
    def __init__(self, runner: IsolatedSubagentRunner) -> None:
        self.runner = runner

    async def run(
        self,
        specs: Sequence[SubagentSpec],
    ) -> tuple[AgentRunResult, ...]:
        return tuple(await asyncio.gather(*(self.runner.run(spec) for spec in specs)))


@dataclass(frozen=True)
class MapReduceResult:
    mapped: tuple[AgentRunResult, ...]
    reduced: Any


class MapReduceOrchestrator:
    def __init__(
        self,
        runner: IsolatedSubagentRunner,
        reducer: Callable[[Sequence[AgentRunResult]], Awaitable[Any]],
    ) -> None:
        self.runner = runner
        self.reducer = reducer

    async def run(
        self,
        specs: Sequence[SubagentSpec],
    ) -> MapReduceResult:
        mapped = tuple(await asyncio.gather(*(self.runner.run(spec) for spec in specs)))
        return MapReduceResult(
            mapped=mapped,
            reduced=await self.reducer(mapped),
        )


@dataclass(frozen=True)
class SpeculativeBranch:
    name: str
    spec: SubagentSpec


@dataclass(frozen=True)
class SpeculativeBranchResult:
    name: str
    result: AgentRunResult
    score: float


@dataclass(frozen=True)
class SpeculativeResult:
    branches: tuple[SpeculativeBranchResult, ...]
    selected: SpeculativeBranchResult | None = None
    merged: Any = None


class SpeculativeOrchestrator:
    def __init__(
        self,
        runner: IsolatedSubagentRunner,
        scorer: Callable[[str, AgentRunResult], Awaitable[float]],
        *,
        merger: (
            Callable[[Sequence[SpeculativeBranchResult]], Awaitable[Any]] | None
        ) = None,
    ) -> None:
        self.runner = runner
        self.scorer = scorer
        self.merger = merger

    async def run(
        self,
        branches: Sequence[SpeculativeBranch],
        *,
        mode: Literal["select", "merge"] = "select",
    ) -> SpeculativeResult:
        if not branches:
            raise ValueError("speculative execution requires at least one branch")
        if mode == "merge" and self.merger is None:
            raise ValueError("merge mode requires a merger")
        if mode not in ("select", "merge"):
            raise ValueError("unsupported speculative mode")
        results = tuple(
            await asyncio.gather(*(self.runner.run(branch.spec) for branch in branches))
        )
        scored_items: list[SpeculativeBranchResult] = []
        for branch, result in zip(branches, results):
            scored_items.append(
                SpeculativeBranchResult(
                    name=branch.name,
                    result=result,
                    score=float(await self.scorer(branch.name, result)),
                )
            )
        scored = tuple(scored_items)
        if mode == "merge":
            if self.merger is None:
                raise RuntimeError("merge mode lost its configured merger")
            return SpeculativeResult(
                branches=scored,
                merged=await self.merger(scored),
            )
        selected = min(scored, key=lambda item: (-item.score, item.name))
        return SpeculativeResult(
            branches=scored,
            selected=selected,
        )
