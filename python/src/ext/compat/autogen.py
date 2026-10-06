from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import time
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

    def dump(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, data: dict[str, Any]) -> TextMessage:
        fields = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})

    def to_text(self) -> str:
        return self.content

    def to_model_text(self) -> str:
        return self.content


@dataclass(frozen=True)
class TaskResult:
    messages: list[TextMessage]
    stop_reason: str | None = None


@dataclass(frozen=True)
class Response:
    chat_message: TextMessage
    inner_messages: list[Any] | None = None


def _component(provider: str, config: dict[str, Any]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        provider=provider,
        component_type="termination",
        version=1,
        component_version=1,
        config=config,
    )


class OpenAIChatCompletionClient:
    def __init__(self, *, model: str, provider: Any = None, **kwargs: Any) -> None:
        self.model = model
        self._provider = provider
        self.options = dict(kwargs)

    @property
    def provider(self) -> Any:
        # Built lazily, so constructing a client never needs credentials.
        # None lets the runner use the environment-configured provider.
        if self._provider is None:
            self._provider = self._build_provider()
        return self._provider

    def _build_provider(self) -> Any:
        return None

    async def close(self) -> None:
        return None


class AnthropicChatCompletionClient(OpenAIChatCompletionClient):
    def _build_provider(self) -> Any:
        from ext.compat.base import _anthropic_provider

        return _anthropic_provider(
            model=self.model,
            credential=self.options.get("api_" + "key"),
            base_url=self.options.get("base_url"),
        )


class OllamaChatCompletionClient(OpenAIChatCompletionClient):
    def _build_provider(self) -> Any:
        import os

        from ext.compat.base import _openai_provider

        host = (
            self.options.get("host")
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434"
        )
        if "://" not in host:
            host = f"http://{host}"
        return _openai_provider(
            model=self.model,
            credential="ollama",
            base_url=f"{host.rstrip('/')}/v1",
        )


def _accepts_cancellation_token(func: Any) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        item
        for item in parameters
        if item.kind in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2 or any(
        item.kind is item.VAR_POSITIONAL for item in parameters
    )


class UserProxyAgent:
    """Human-in-the-loop agent: every reply comes from ``input_func``."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "A human user",
        input_func: Any = None,
    ) -> None:
        if not str(name).strip():
            raise ValueError("agent name must not be empty")
        self.name = str(name)
        self.description = description
        self.input_func = input_func

    async def _read(self, prompt: str, cancellation_token: Any) -> str:
        if self.input_func is None:
            return await asyncio.to_thread(input, prompt)
        if _accepts_cancellation_token(self.input_func):
            value = self.input_func(prompt, cancellation_token)
        else:
            value = self.input_func(prompt)
        if inspect.isawaitable(value):
            value = await value
        return str(value)

    async def on_messages(
        self, messages: Sequence[Any], cancellation_token: Any = None
    ) -> Response:
        answer = await self._read("Enter your response: ", cancellation_token)
        return Response(chat_message=TextMessage(content=answer, source=self.name))

    async def run(
        self, *, task: Any = None, cancellation_token: Any = None, **_: Any
    ) -> TaskResult:
        messages = (
            [] if task is None else [TextMessage(content=str(task), source="user")]
        )
        response = await self.on_messages(messages, cancellation_token)
        return TaskResult(messages=[*messages, response.chat_message])


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

    def _component_config(self) -> dict[str, Any]:
        return {}

    def dump_component(self) -> types.SimpleNamespace:
        return _component(
            f"autogen_agentchat.conditions.{type(self).__name__}",
            self._component_config(),
        )


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

    def _component_config(self) -> dict[str, Any]:
        return {"text": self.text, "sources": sorted(self.sources) or None}


class MaxMessageTermination(TerminationCondition):
    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        if len(messages) >= self.max_messages:
            return f"Maximum {self.max_messages} messages reached"
        return None

    def _component_config(self) -> dict[str, Any]:
        return {"max_messages": self.max_messages}


class TokenUsageTermination(TerminationCondition):
    def __init__(
        self,
        max_total_token: int | None = None,
        max_prompt_token: int | None = None,
        max_completion_token: int | None = None,
    ) -> None:
        if (
            max_total_token is None
            and max_prompt_token is None
            and max_completion_token is None
        ):
            raise ValueError("at least one token limit must be set")
        self.max_total_token = max_total_token
        self.max_prompt_token = max_prompt_token
        self.max_completion_token = max_completion_token

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        prompt = completion = 0
        for message in messages:
            usage = getattr(message, "models_usage", None)
            prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion += int(getattr(usage, "completion_tokens", 0) or 0)
        limits = (
            (self.max_total_token, prompt + completion, "total"),
            (self.max_prompt_token, prompt, "prompt"),
            (self.max_completion_token, completion, "completion"),
        )
        for limit, used, label in limits:
            if limit is not None and used >= limit:
                return f"Token usage limit reached: {label} {used} >= {limit}"
        return None

    def _component_config(self) -> dict[str, Any]:
        return {
            "max_total_token": self.max_total_token,
            "max_prompt_token": self.max_prompt_token,
            "max_completion_token": self.max_completion_token,
        }


class TimeoutTermination(TerminationCondition):
    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._started: float | None = None

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        now = time.monotonic()
        if self._started is None:
            self._started = now
        if now - self._started >= self.timeout_seconds:
            return f"Timeout of {self.timeout_seconds} seconds reached"
        return None

    def _component_config(self) -> dict[str, Any]:
        return {"timeout_seconds": self.timeout_seconds}


class OrTermination(TerminationCondition):
    def __init__(self, left: TerminationCondition, right: TerminationCondition) -> None:
        self.left = left
        self.right = right

    def should_stop(self, messages: Sequence[TextMessage]) -> str | None:
        return self.left.should_stop(messages) or self.right.should_stop(messages)

    def _component_config(self) -> dict[str, Any]:
        conditions = []
        for item in (self.left, self.right):
            if isinstance(item, OrTermination):
                conditions.extend(item._component_config()["conditions"])
            else:
                conditions.append(vars(item.dump_component()))
        return {"conditions": conditions}

    def dump_component(self) -> types.SimpleNamespace:
        return _component(
            "autogen_agentchat.base.OrTerminationCondition", self._component_config()
        )


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

    def dump_component(self) -> types.SimpleNamespace:
        termination = self.termination_condition
        return types.SimpleNamespace(
            provider="autogen_agentchat.teams.RoundRobinGroupChat",
            component_type="team",
            version=1,
            component_version=1,
            config={
                "participants": [{"name": item.name} for item in self.participants],
                "termination_condition": (
                    vars(termination.dump_component())
                    if termination is not None
                    else None
                ),
                "max_turns": self.max_turns,
            },
        )


async def Console(stream: Any, **_: Any) -> Any:
    """Print streamed messages and return the final TaskResult/Response."""
    last = None
    async for item in stream:
        if isinstance(item, (TaskResult, Response)):
            last = item
            continue
        content = getattr(item, "content", None)
        if content is not None:
            print(
                f"---------- {getattr(item, 'source', '')} ----------\n{content}",
                flush=True,
            )
    return last


def install_autogen_compat(parent: Any) -> None:
    root_name = f"{parent.__name__}.autogen_agentchat"
    root = types.ModuleType(root_name)
    root.__path__ = ()
    agents = types.ModuleType(f"{root_name}.agents")
    teams = types.ModuleType(f"{root_name}.teams")
    conditions = types.ModuleType(f"{root_name}.conditions")
    messages = types.ModuleType(f"{root_name}.messages")

    base = types.ModuleType(f"{root_name}.base")
    ui = types.ModuleType(f"{root_name}.ui")

    agents.AssistantAgent = AssistantAgent
    agents.UserProxyAgent = UserProxyAgent
    agents.__all__ = ("AssistantAgent", "UserProxyAgent")
    teams.RoundRobinGroupChat = RoundRobinGroupChat
    teams.__all__ = ("RoundRobinGroupChat",)
    conditions.TextMentionTermination = TextMentionTermination
    conditions.MaxMessageTermination = MaxMessageTermination
    conditions.TokenUsageTermination = TokenUsageTermination
    conditions.TimeoutTermination = TimeoutTermination
    conditions.__all__ = (
        "MaxMessageTermination",
        "TextMentionTermination",
        "TimeoutTermination",
        "TokenUsageTermination",
    )
    messages.TextMessage = TextMessage
    messages.__all__ = ("TextMessage",)
    base.Response = Response
    base.TaskResult = TaskResult
    base.TerminationCondition = TerminationCondition
    base.__all__ = ("Response", "TaskResult", "TerminationCondition")
    ui.Console = Console
    ui.__all__ = ("Console",)

    root.AssistantAgent = AssistantAgent
    root.RoundRobinGroupChat = RoundRobinGroupChat
    root.TaskResult = TaskResult
    root.TextMessage = TextMessage
    root.TextMentionTermination = TextMentionTermination
    root.MaxMessageTermination = MaxMessageTermination
    root.UserProxyAgent = UserProxyAgent
    root.agents = agents
    root.teams = teams
    root.conditions = conditions
    root.messages = messages
    root.base = base
    root.ui = ui
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
    anthropic = types.ModuleType(f"{parent.__name__}.autogen_ext.models.anthropic")
    anthropic.AnthropicChatCompletionClient = AnthropicChatCompletionClient
    anthropic.__all__ = ("AnthropicChatCompletionClient",)
    ollama = types.ModuleType(f"{parent.__name__}.autogen_ext.models.ollama")
    ollama.OllamaChatCompletionClient = OllamaChatCompletionClient
    ollama.__all__ = ("OllamaChatCompletionClient",)
    models.openai = openai
    models.anthropic = anthropic
    models.ollama = ollama
    ext.models = models

    for module in (
        root,
        agents,
        teams,
        conditions,
        messages,
        base,
        ui,
        ext,
        models,
        openai,
        anthropic,
        ollama,
    ):
        sys.modules[module.__name__] = module
    parent.autogen_agentchat = root
    parent.autogen_ext = ext


__all__ = (
    "AnthropicChatCompletionClient",
    "AssistantAgent",
    "Console",
    "MaxMessageTermination",
    "OllamaChatCompletionClient",
    "OpenAIChatCompletionClient",
    "Response",
    "RoundRobinGroupChat",
    "TaskResult",
    "TextMentionTermination",
    "TextMessage",
    "TimeoutTermination",
    "TokenUsageTermination",
    "UserProxyAgent",
    "install_autogen_compat",
)
