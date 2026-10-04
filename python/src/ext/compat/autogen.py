from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ext.compat.openai_agents import Agent as _CompatAgent
from ext.compat.openai_agents import Runner as _CompatRunner


@dataclass(frozen=True)
class TextMessage:
    content: str
    source: str
    type: str = "TextMessage"
    models_usage: Any = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskResult:
    messages: list[TextMessage]
    stop_reason: str | None = None


class OpenAIChatCompletionClient:
    def __init__(self, *, model: str, provider: Any = None, **kwargs: Any) -> None:
        self.model = model
        self.provider = provider
        self.options = dict(kwargs)

    async def close(self) -> None:
        return None


class AssistantAgent:
    def __init__(
        self,
        name: str,
        model_client: OpenAIChatCompletionClient | None = None,
        *,
        description: str = "",
        system_message: str = "",
        tools: Sequence[Any] = (),
        model_client_stream: bool = False,
        max_tool_iterations: int = 16,
        **kwargs: Any,
    ) -> None:
        if not str(name).strip():
            raise ValueError("agent name must not be empty")
        self.name = str(name)
        self.model_client = model_client
        self.description = description
        self.system_message = system_message
        self.tools = list(tools)
        self.model_client_stream = model_client_stream
        self.max_tool_iterations = max_tool_iterations
        self.options = dict(kwargs)

    def _agent(self) -> _CompatAgent:
        provider = self.model_client.provider if self.model_client is not None else None
        model = (
            self.model_client.model if self.model_client is not None else "gpt-4o-mini"
        )
        return _CompatAgent(
            name=self.name,
            instructions=self.system_message,
            model=model,
            provider=provider,
            tools=self.tools,
            handoff_description=self.description or None,
        )

    async def run(self, *, task: Any = None, **_: Any) -> TaskResult:
        if task is None:
            raise ValueError("AssistantAgent.run requires task")
        result = await _CompatRunner.run(
            self._agent(),
            str(task),
            max_turns=self.max_tool_iterations,
        )
        return TaskResult(
            messages=[
                TextMessage(content=str(task), source="user"),
                TextMessage(content=str(result.final_output), source=self.name),
            ],
            stop_reason=result.raw_result.termination_reason,
        )

    async def run_stream(self, *, task: Any = None, **kwargs: Any):
        result = await self.run(task=task, **kwargs)
        for message in result.messages:
            yield message
        yield result


class TerminationCondition:
    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        return None

    def __or__(self, other: TerminationCondition) -> OrTermination:
        return OrTermination(self, other)


class TextMentionTermination(TerminationCondition):
    def __init__(self, text: str, sources: Sequence[str] | None = None) -> None:
        self.text = text
        self.sources = set(sources or ())

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        for message in reversed(messages):
            if self.sources and message.source not in self.sources:
                continue
            if self.text in message.content:
                return f"Text {self.text!r} mentioned"
        return None


class MaxMessageTermination(TerminationCondition):
    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        if len(messages) >= self.max_messages:
            return f"Maximum {self.max_messages} messages reached"
        return None


class OrTermination(TerminationCondition):
    def __init__(self, left: TerminationCondition, right: TerminationCondition) -> None:
        self.left = left
        self.right = right

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        return self.left.should_stop(messages) or self.right.should_stop(messages)


class RoundRobinGroupChat:
    def __init__(
        self,
        participants: Sequence[AssistantAgent],
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        **_: Any,
    ) -> None:
        if not participants:
            raise ValueError("RoundRobinGroupChat requires at least one participant")
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.participants = list(participants)
        self.termination_condition = termination_condition
        self.max_turns = max_turns or 16

    async def run(self, *, task: Any = None, **_: Any) -> TaskResult:
        if task is None:
            raise ValueError(
                "continued AutoGen team state is not implemented; provide task explicitly"
            )
        messages = [TextMessage(content=str(task), source="user")]
        stop_reason: str | None = None
        for turn in range(self.max_turns):
            participant = self.participants[turn % len(self.participants)]
            history = "\n".join(f"{item.source}: {item.content}" for item in messages)
            result = await _CompatRunner.run(
                participant._agent(),
                history,
                max_turns=participant.max_tool_iterations,
            )
            messages.append(
                TextMessage(content=str(result.final_output), source=participant.name)
            )
            if self.termination_condition is not None:
                stop_reason = self.termination_condition.should_stop(messages)
                if stop_reason is not None:
                    break
        if stop_reason is None:
            stop_reason = f"Maximum {self.max_turns} turns reached"
        return TaskResult(messages=messages, stop_reason=stop_reason)

    async def run_stream(self, *, task: Any = None, **kwargs: Any):
        result = await self.run(task=task, **kwargs)
        for message in result.messages:
            yield message
        yield result


def install_autogen_compat(parent: Any) -> None:
    root_name = f"{parent.__name__}.autogen_agentchat"
    root = types.ModuleType(root_name)
    root.__path__ = ()
    agents = types.ModuleType(f"{root_name}.agents")
    teams = types.ModuleType(f"{root_name}.teams")
    conditions = types.ModuleType(f"{root_name}.conditions")
    messages = types.ModuleType(f"{root_name}.messages")

    agents.AssistantAgent = AssistantAgent
    agents.__all__ = ("AssistantAgent",)
    teams.RoundRobinGroupChat = RoundRobinGroupChat
    teams.__all__ = ("RoundRobinGroupChat",)
    conditions.TextMentionTermination = TextMentionTermination
    conditions.MaxMessageTermination = MaxMessageTermination
    conditions.__all__ = ("MaxMessageTermination", "TextMentionTermination")
    messages.TextMessage = TextMessage
    messages.__all__ = ("TextMessage",)

    root.AssistantAgent = AssistantAgent
    root.RoundRobinGroupChat = RoundRobinGroupChat
    root.TaskResult = TaskResult
    root.TextMessage = TextMessage
    root.TextMentionTermination = TextMentionTermination
    root.MaxMessageTermination = MaxMessageTermination
    root.agents = agents
    root.teams = teams
    root.conditions = conditions
    root.messages = messages
    root.__all__ = (
        "AssistantAgent",
        "MaxMessageTermination",
        "RoundRobinGroupChat",
        "TaskResult",
        "TextMentionTermination",
        "TextMessage",
    )

    model_name = f"{parent.__name__}.autogen_ext.models.openai"
    ext = types.ModuleType(f"{parent.__name__}.autogen_ext")
    ext.__path__ = ()
    models = types.ModuleType(f"{parent.__name__}.autogen_ext.models")
    models.__path__ = ()
    openai = types.ModuleType(model_name)
    openai.OpenAIChatCompletionClient = OpenAIChatCompletionClient
    openai.__all__ = ("OpenAIChatCompletionClient",)
    models.openai = openai
    ext.models = models

    for module in (root, agents, teams, conditions, messages, ext, models, openai):
        sys.modules[module.__name__] = module
    parent.autogen_agentchat = root
    parent.autogen_ext = ext


__all__ = (
    "AssistantAgent",
    "MaxMessageTermination",
    "OpenAIChatCompletionClient",
    "RoundRobinGroupChat",
    "TaskResult",
    "TextMentionTermination",
    "TextMessage",
    "install_autogen_compat",
)
