from __future__ import annotations

import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ext.compat.openai_agents import Agent as _CompatAgent
from ext.compat.openai_agents import Runner as _CompatRunner


class Process(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"


class Agent:
    def __init__(
        self,
        *,
        role: str,
        goal: str,
        backstory: str = "",
        llm: Any = None,
        tools: Sequence[Any] = (),
        verbose: bool = False,
        allow_delegation: bool = False,
        provider: Any = None,
        max_iter: int = 16,
        **kwargs: Any,
    ) -> None:
        if not str(role).strip():
            raise ValueError("agent role must not be empty")
        self.role = str(role)
        self.goal = str(goal)
        self.backstory = str(backstory)
        self.llm = llm
        self.tools = list(tools)
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.provider = provider
        self.max_iter = max_iter
        self.options = dict(kwargs)

    def _agent(self) -> _CompatAgent:
        model = self.llm if isinstance(self.llm, str) else "gpt-4o-mini"
        instructions = "\n".join(
            part
            for part in (
                f"Role: {self.role}",
                f"Goal: {self.goal}",
                self.backstory,
            )
            if part
        )
        return _CompatAgent(
            name=self.role,
            instructions=instructions,
            model=model,
            provider=self.provider,
            tools=self.tools,
        )


@dataclass
class Task:
    description: str
    expected_output: str = ""
    agent: Agent | None = None
    context: Sequence[Task] = ()
    async_execution: bool = False
    output: Any = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("task description must not be empty")


@dataclass(frozen=True)
class TaskOutput:
    raw: str
    description: str = ""
    agent: str | None = None

    def __str__(self) -> str:
        return self.raw


@dataclass(frozen=True)
class CrewOutput:
    raw: str
    tasks_output: tuple[TaskOutput, ...]

    def __str__(self) -> str:
        return self.raw


class Crew:
    def __init__(
        self,
        *,
        agents: Sequence[Agent] = (),
        tasks: Sequence[Task] = (),
        process: Process | str = Process.sequential,
        verbose: bool = False,
        manager_agent: Agent | None = None,
        **kwargs: Any,
    ) -> None:
        self.agents = list(agents)
        self.tasks = list(tasks)
        self.process = Process(process)
        self.verbose = verbose
        self.manager_agent = manager_agent
        self.options = dict(kwargs)

    def _resolve_agent(self, task: Task) -> Agent:
        if task.agent is not None:
            return task.agent
        if len(self.agents) == 1:
            return self.agents[0]
        raise ValueError("task.agent is required when a Crew contains multiple agents")

    async def kickoff_async(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
    ) -> CrewOutput:
        if self.process is Process.hierarchical:
            raise NotImplementedError(
                "CrewAI hierarchical process compatibility is not implemented; "
                "use Process.sequential"
            )
        if any(task.async_execution for task in self.tasks):
            raise NotImplementedError(
                "CrewAI async_execution task scheduling is not implemented; "
                "use kickoff_async with sequential tasks"
            )

        rendered_inputs = dict(inputs or {})
        completed: dict[int, TaskOutput] = {}
        outputs: list[TaskOutput] = []

        for task in self.tasks:
            agent = self._resolve_agent(task)
            context_outputs = [
                completed[id(item)].raw
                for item in task.context
                if id(item) in completed
            ]
            try:
                description = task.description.format(**rendered_inputs)
            except (KeyError, IndexError, ValueError):
                description = task.description
            prompt_parts = [description]
            if task.expected_output:
                prompt_parts.append("Expected output: " + task.expected_output)
            if context_outputs:
                prompt_parts.append(
                    "Context from prior tasks:\n" + "\n\n".join(context_outputs)
                )
            result = await _CompatRunner.run(
                agent._agent(),
                "\n\n".join(prompt_parts),
                max_turns=agent.max_iter,
            )
            output = TaskOutput(
                raw=str(result.final_output),
                description=task.description,
                agent=agent.role,
            )
            task.output = output
            completed[id(task)] = output
            outputs.append(output)

        return CrewOutput(
            raw=outputs[-1].raw if outputs else "",
            tasks_output=tuple(outputs),
        )

    def kickoff(self, *, inputs: Mapping[str, Any] | None = None) -> CrewOutput:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.kickoff_async(inputs=inputs))
        raise RuntimeError(
            "Crew.kickoff cannot run inside an active event loop; use kickoff_async"
        )


def install_crewai_compat(parent: Any) -> None:
    module = types.ModuleType(f"{parent.__name__}.crewai")
    exports = {
        "Agent": Agent,
        "Crew": Crew,
        "CrewOutput": CrewOutput,
        "Process": Process,
        "Task": Task,
        "TaskOutput": TaskOutput,
    }
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    sys.modules[module.__name__] = module
    parent.crewai = module


__all__ = (
    "Agent",
    "Crew",
    "CrewOutput",
    "Process",
    "Task",
    "TaskOutput",
    "install_crewai_compat",
)
