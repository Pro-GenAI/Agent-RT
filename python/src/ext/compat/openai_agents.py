from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, get_args, get_origin

TContext = TypeVar("TContext")


def _rt():
    return sys.modules["agent_rt"]


def _function_style_name(name: str) -> str:
    """The SDK's default handoff/agent-tool naming: `Billing-EU` -> `billing_eu`."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", re.sub(r"\s", "_", name)).lower()


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple, set, frozenset):
        return {
            "type": "array",
            "items": _schema_for_annotation(args[0] if args else Any),
        }
    if origin in (dict, Mapping):
        return {"type": "object"}
    if origin is not None and type(None) in args:
        values = [
            _schema_for_annotation(item) for item in args if item is not type(None)
        ]
        return {"anyOf": [*values, {"type": "null"}]}
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }.get(annotation, {})


def _callable_schema(func: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    try:
        signature = inspect.signature(func)
        annotations = inspect.get_annotations(func, eval_str=True)
    except (TypeError, ValueError, NameError):
        signature = inspect.Signature()
        annotations = {}
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "context", "ctx", "run_context"}:
            continue
        annotation = annotations.get(parameter.name, parameter.annotation)
        properties[parameter.name] = _schema_for_annotation(annotation)
        if parameter.default is inspect.Signature.empty:
            required.append(parameter.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


class AgentsException(Exception):
    """Base class of the SDK's `agents.exceptions` hierarchy."""


class MaxTurnsExceeded(AgentsException):
    pass


class ModelBehaviorError(AgentsException):
    pass


class UserError(AgentsException):
    pass


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunContextWrapper(Generic[TContext]):
    """Run context handed to guardrails, matching the SDK shape."""

    context: TContext = None  # type: ignore[assignment]
    usage: Usage = field(default_factory=Usage)


@dataclass
class GuardrailFunctionOutput:
    output_info: Any = None
    tripwire_triggered: bool = False


@dataclass(frozen=True)
class InputGuardrail:
    guardrail_function: Any
    name: str | None = None
    run_in_parallel: bool = True

    def get_name(self) -> str:
        return self.name or getattr(self.guardrail_function, "__name__", "guardrail")


@dataclass(frozen=True)
class OutputGuardrail:
    guardrail_function: Any
    name: str | None = None

    def get_name(self) -> str:
        return self.name or getattr(self.guardrail_function, "__name__", "guardrail")


@dataclass(frozen=True)
class InputGuardrailResult:
    guardrail: InputGuardrail
    output: GuardrailFunctionOutput


@dataclass(frozen=True)
class OutputGuardrailResult:
    guardrail: OutputGuardrail
    agent_output: Any
    agent: Any
    output: GuardrailFunctionOutput


class InputGuardrailTripwireTriggered(AgentsException):
    def __init__(self, guardrail_result: InputGuardrailResult) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.get_name()} triggered tripwire"
        )


class OutputGuardrailTripwireTriggered(AgentsException):
    def __init__(self, guardrail_result: OutputGuardrailResult) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.get_name()} triggered tripwire"
        )


def input_guardrail(
    func: Any = None, *, name: str | None = None, run_in_parallel: bool = True, **_: Any
):
    def decorate(target: Any) -> InputGuardrail:
        return InputGuardrail(target, name=name, run_in_parallel=run_in_parallel)

    return decorate(func) if callable(func) else decorate


def output_guardrail(func: Any = None, *, name: str | None = None, **_: Any):
    def decorate(target: Any) -> OutputGuardrail:
        return OutputGuardrail(target, name=name)

    return decorate(func) if callable(func) else decorate


async def _guardrail_output(function: Any, *args: Any) -> GuardrailFunctionOutput:
    result = function(*args)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, GuardrailFunctionOutput):
        raise UserError("guardrail functions must return GuardrailFunctionOutput")
    return result


def set_default_openai_key(key: str | None, use_for_tracing: bool = True) -> None:
    """Agent RT's OpenAI provider reads OPENAI_API_KEY, the SDK's default key."""
    if key:
        os.environ["OPENAI_API_KEY"] = key


def set_tracing_disabled(disabled: bool) -> None:
    """Accepted for migration parity; Agent RT never exports SDK traces."""


def enable_verbose_stdout_logging() -> None:
    """Accepted for migration parity; Agent RT logging is configured separately."""


@contextlib.contextmanager
def trace(workflow_name: str, *args: Any, **kwargs: Any):
    """A no-op SDK trace scope; Agent RT never exports SDK traces."""
    yield SimpleTrace(workflow_name)


@dataclass(frozen=True)
class SimpleTrace:
    name: str


class RunState:
    """Importable stand-in: SDK run-state pause/resume is not implemented."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OpenAI Agents RunState compatibility is not implemented in Python; "
            "use Agent RT approval checkpoints"
        )

    @classmethod
    async def from_json(cls, *args: Any, **kwargs: Any) -> RunState:
        return cls()


class MCPServerStreamableHttp:
    """Importable stand-in: remote MCP must use an injected Agent RT MCPClient."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OpenAI Agents MCP server compatibility is not implemented; "
            "use an explicitly injected Agent RT MCPClient"
        )


MCPServerSse = MCPServerStdio = MCPServerStreamableHttp


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    params_json_schema: Mapping[str, Any]
    callback: Any

    async def invoke(self, arguments: Mapping[str, Any]) -> Any:
        result = self.callback(**dict(arguments))
        if inspect.isawaitable(result):
            result = await result
        return result


def function_tool(
    func: Any = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    **_: Any,
):
    def decorate(target: Any) -> FunctionTool:
        name = name_override or getattr(target, "__name__", "tool")
        return FunctionTool(
            name=name,
            # The SDK accepts undocumented functions; Agent RT tool
            # definitions need a non-empty description.
            description=description_override or inspect.getdoc(target) or name,
            params_json_schema=_callable_schema(target),
            callback=target,
        )

    return decorate(func) if callable(func) else decorate


@dataclass(frozen=True)
class Handoff:
    agent: Agent
    tool_name_override: str | None = None
    tool_description_override: str | None = None


def handoff(
    agent: Agent,
    *,
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    **_: Any,
) -> Handoff:
    return Handoff(agent, tool_name_override, tool_description_override)


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: str = "",
        model: Any = None,
        tools: Sequence[Any] = (),
        handoffs: Sequence[Any] = (),
        handoff_description: str | None = None,
        output_type: Any = None,
        provider: Any = None,
        input_guardrails: Sequence[Any] = (),
        output_guardrails: Sequence[Any] = (),
        **kwargs: Any,
    ) -> None:
        if not str(name).strip():
            raise ValueError("agent name must not be empty")
        self.name = str(name)
        self.instructions = str(instructions or "")
        self.model = model or kwargs.pop("model_name", None) or "gpt-4o-mini"
        self.tools = list(tools)
        self.handoffs = list(handoffs)
        self.handoff_description = handoff_description
        self.output_type = output_type
        self.provider = provider
        self.input_guardrails = [
            item if isinstance(item, InputGuardrail) else InputGuardrail(item)
            for item in input_guardrails
        ]
        self.output_guardrails = [
            item if isinstance(item, OutputGuardrail) else OutputGuardrail(item)
            for item in output_guardrails
        ]
        self.metadata = dict(kwargs)

    def as_tool(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        **_: Any,
    ) -> FunctionTool:
        target = self

        async def invoke(input: str) -> Any:
            result = await Runner.run(target, input)
            return result.final_output

        return FunctionTool(
            name=tool_name or _function_style_name(self.name),
            description=tool_description
            or self.handoff_description
            or f"Delegate bounded work to {self.name}.",
            params_json_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
            callback=invoke,
        )


@dataclass(frozen=True)
class RunResult:
    final_output: Any
    last_agent: Agent
    history: tuple[Any, ...]
    raw_result: Any

    def final_output_as(self, _type: Any, **_: Any) -> Any:
        return self.final_output


def _message(role: str, text: str):
    rt = _rt()
    return rt.ModelMessage(
        role=role,
        content=(rt.ContentPart(type="text", text=text),),
    )


def _input_messages(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (_message("user", value),)
    if isinstance(value, Sequence):
        out = []
        for item in value:
            if isinstance(item, _rt().ModelMessage):
                out.append(item)
                continue
            if isinstance(item, Mapping):
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                if isinstance(content, Sequence) and not isinstance(content, str):
                    content = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, Mapping)
                    )
                out.append(_message(role, str(content)))
                continue
            out.append(_message("user", str(item)))
        return tuple(out)
    return (_message("user", str(value)),)


def _extract_text(result: Any) -> Any:
    if result.structured_output is not None:
        return result.structured_output
    response = result.final_response
    if response is None:
        return ""
    text = "".join(
        part.text or "" for part in response.message.content if part.type == "text"
    )
    return text


def _normalize_tool(value: Any) -> FunctionTool:
    if isinstance(value, FunctionTool):
        return value
    if callable(value):
        return function_tool(value)
    raise TypeError("OpenAI Agents compatibility supports callable/function tools only")


def _handoff_tool(value: Any, state: dict[str, Agent]) -> FunctionTool:
    item = value if isinstance(value, Handoff) else Handoff(value)
    if not isinstance(item.agent, Agent):
        raise TypeError("handoffs must contain Agent or Handoff values")
    name = item.tool_name_override or _function_style_name(
        "transfer_to_" + item.agent.name
    )

    async def invoke(input: str = "") -> Any:
        nested = await Runner.run(item.agent, input or "Continue the current task.")
        state["last_agent"] = nested.last_agent
        return nested.final_output

    return FunctionTool(
        name=name,
        description=item.tool_description_override
        or item.agent.handoff_description
        or f"Transfer work to {item.agent.name}.",
        params_json_schema={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "additionalProperties": False,
        },
        callback=invoke,
    )


class Runner:
    @staticmethod
    async def run(
        starting_agent: Agent,
        input: Any,
        *,
        max_turns: int = 16,
        context: Any = None,
        **_: Any,
    ) -> RunResult:
        if not isinstance(starting_agent, Agent):
            raise TypeError("starting_agent must be an Agent")
        rt = _rt()
        run_context = RunContextWrapper(context=context)
        # Input guardrails finish before the first model call, so a tripwire
        # stops the run before any model or tool side effect.
        for guardrail in starting_agent.input_guardrails:
            output = await _guardrail_output(
                guardrail.guardrail_function, run_context, starting_agent, input
            )
            if output.tripwire_triggered:
                raise InputGuardrailTripwireTriggered(
                    InputGuardrailResult(guardrail=guardrail, output=output)
                )
        provider = starting_agent.provider or rt.load_model()
        registry = rt.ToolRegistry()
        state = {"last_agent": starting_agent}

        tools = [_normalize_tool(tool) for tool in starting_agent.tools]
        tools.extend(_handoff_tool(item, state) for item in starting_agent.handoffs)
        for item in tools:

            async def handler(arguments, _cancellation_token=None, tool=item):
                return await tool.invoke(arguments)

            registry.register(
                rt.ToolDefinition(
                    name=item.name,
                    description=item.description,
                    input_schema=dict(item.params_json_schema),
                ),
                handler=handler,
            )

        model_name = (
            starting_agent.model
            if isinstance(starting_agent.model, str)
            else getattr(starting_agent.model, "model", None)
            or getattr(provider, "model", None)
            or "gpt-4o-mini"
        )
        config = rt.AgentConfig(
            name=starting_agent.name,
            instructions=starting_agent.instructions,
            description=starting_agent.handoff_description,
            model=rt.ModelSettings(model=str(model_name)),
        )
        result = await rt.AgentLoop(
            provider,
            tool_registry=registry if tools else None,
        ).run(
            config,
            _input_messages(input),
            limits=rt.AgentRunLimits(max_turns=max_turns),
            tool_context={"context": context} if context is not None else {},
        )
        if result.termination_reason == "max_turns":
            raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")
        final_output = _extract_text(result)
        for guardrail in starting_agent.output_guardrails:
            output = await _guardrail_output(
                guardrail.guardrail_function, run_context, starting_agent, final_output
            )
            if output.tripwire_triggered:
                raise OutputGuardrailTripwireTriggered(
                    OutputGuardrailResult(
                        guardrail=guardrail,
                        agent_output=final_output,
                        agent=starting_agent,
                        output=output,
                    )
                )
        return RunResult(
            final_output=final_output,
            last_agent=state["last_agent"],
            history=result.messages,
            raw_result=result,
        )

    @staticmethod
    def run_sync(starting_agent: Agent, input: Any, **kwargs: Any) -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(Runner.run(starting_agent, input, **kwargs))
        raise RuntimeError(
            "Runner.run_sync cannot run inside an active event loop; use Runner.run"
        )

    @staticmethod
    def run_streamed(starting_agent: Agent, input: Any, **kwargs: Any):
        raise NotImplementedError(
            "OpenAI Agents streamed-run compatibility is not implemented; use Runner.run"
        )


async def run(agent: Agent, input: Any, **kwargs: Any) -> RunResult:
    return await Runner.run(agent, input, **kwargs)


def install_openai_agents_compat(parent: Any) -> None:
    import types

    module = types.ModuleType(f"{parent.__name__}.agents")
    exceptions = {
        "AgentsException": AgentsException,
        "InputGuardrailTripwireTriggered": InputGuardrailTripwireTriggered,
        "MaxTurnsExceeded": MaxTurnsExceeded,
        "ModelBehaviorError": ModelBehaviorError,
        "OutputGuardrailTripwireTriggered": OutputGuardrailTripwireTriggered,
        "UserError": UserError,
    }
    guardrails = {
        "GuardrailFunctionOutput": GuardrailFunctionOutput,
        "InputGuardrail": InputGuardrail,
        "InputGuardrailResult": InputGuardrailResult,
        "OutputGuardrail": OutputGuardrail,
        "OutputGuardrailResult": OutputGuardrailResult,
        "input_guardrail": input_guardrail,
        "output_guardrail": output_guardrail,
    }
    tracing = {"set_tracing_disabled": set_tracing_disabled, "trace": trace}
    mcp = {
        "MCPServerSse": MCPServerSse,
        "MCPServerStdio": MCPServerStdio,
        "MCPServerStreamableHttp": MCPServerStreamableHttp,
    }
    exports = {
        "Agent": Agent,
        "FunctionTool": FunctionTool,
        "Handoff": Handoff,
        "RunContextWrapper": RunContextWrapper,
        "RunResult": RunResult,
        "RunState": RunState,
        "Runner": Runner,
        "Usage": Usage,
        "enable_verbose_stdout_logging": enable_verbose_stdout_logging,
        "function_tool": function_tool,
        "handoff": handoff,
        "run": run,
        "set_default_openai_key": set_default_openai_key,
        **exceptions,
        **guardrails,
        **tracing,
    }
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    # Upstream `agents` is a package; register the submodules migrated code
    # imports from (`from agents.exceptions import MaxTurnsExceeded`).
    module.__path__ = []
    for name, values in (
        ("exceptions", exceptions),
        ("guardrail", guardrails),
        ("run_context", {"RunContextWrapper": RunContextWrapper}),
        ("usage", {"Usage": Usage}),
        ("tracing", tracing),
        ("mcp", mcp),
    ):
        submodule = types.ModuleType(f"{module.__name__}.{name}")
        submodule.__dict__.update(values)
        submodule.__all__ = tuple(sorted(values))
        sys.modules[submodule.__name__] = submodule
        sys.modules[f"{parent.__name__}.openai_agents.{name}"] = submodule
        setattr(module, name, submodule)
    sys.modules[module.__name__] = module
    sys.modules[f"{parent.__name__}.openai_agents"] = module
    parent.agents = module
    parent.openai_agents = module


__all__ = (
    "Agent",
    "AgentsException",
    "FunctionTool",
    "GuardrailFunctionOutput",
    "Handoff",
    "InputGuardrail",
    "InputGuardrailTripwireTriggered",
    "MaxTurnsExceeded",
    "OutputGuardrail",
    "OutputGuardrailTripwireTriggered",
    "RunContextWrapper",
    "RunResult",
    "Runner",
    "function_tool",
    "handoff",
    "input_guardrail",
    "install_openai_agents_compat",
    "output_guardrail",
    "run",
)
