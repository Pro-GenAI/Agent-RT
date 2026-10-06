from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
from collections.abc import Mapping

from ext.compat._format import safe_format
from ext.compat.base import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    _anthropic_provider,
    _langchain_tool_schema,
    _openai_provider,
    _parse_structured_value,
    _request,
    _run_sync,
    _schema_mapping,
)


def _messages_input(value):
    if isinstance(value, str):
        return [HumanMessage(value)]
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if isinstance(value, BaseMessage):
        return [value]
    return list(value)


def _usage_metadata(usage):
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _ai_message(response):
    calls = [
        {
            "id": call.id,
            "name": call.name,
            "args": dict(call.arguments),
            "type": "tool_call",
            **(
                {"argument_error": call.argument_error}
                if getattr(call, "argument_error", None)
                else {}
            ),
        }
        for call in response.message.tool_calls
    ]
    metadata = {
        "model": response.model,
        "finish_reason": response.finish_reason,
    }
    content = [
        {
            "type": part.type,
            **({"text": part.text} if part.text is not None else {}),
            **({"data": part.data} if part.data is not None else {}),
            **({"mime_type": part.mime_type} if part.mime_type is not None else {}),
        }
        for part in response.message.content
    ]
    return AIMessage(
        (
            content
            if any(part.get("type") != "text" for part in content)
            else "".join(str(part.get("text", "")) for part in content)
        ),
        tool_calls=calls,
        additional_kwargs=dict(metadata),
        response_metadata=dict(metadata),
        usage_metadata=_usage_metadata(response.usage),
    )


class _StructuredRunnable:
    def __init__(self, model, schema, *, include_raw=False):
        self.model = model
        self.schema = schema
        self.include_raw = include_raw

    async def ainvoke(self, messages, **kwargs):
        raw = await self.model.ainvoke(messages, **kwargs)
        try:
            parsed = _parse_structured_value(self.schema, raw.text)
        except Exception as exc:
            if self.include_raw:
                return {"raw": raw, "parsed": None, "parsing_error": exc}
            raise
        if self.include_raw:
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return parsed

    def invoke(self, messages, **kwargs):
        return _run_sync(self.ainvoke(messages, **kwargs))

    async def abatch(self, inputs, **kwargs):
        return await asyncio.gather(*(self.ainvoke(item, **kwargs) for item in inputs))

    def batch(self, inputs, **kwargs):
        return _run_sync(self.abatch(inputs, **kwargs))

    def with_retry(self, *, stop_after_attempt=3, **_):
        return _RetryRunnable(self, max_attempts=int(stop_after_attempt))

    def with_fallbacks(self, fallbacks, **_):
        return _FallbackRunnable(self, fallbacks)

    def pipe(self, next_runnable):
        return _SequenceRunnable((self, next_runnable))

    def configurable_fields(self, **fields):
        return _ConfigurableRunnable(self, fields)


async def _emit_callbacks(config, event_type, *, input_value, output=None, error=None):
    if not config:
        return
    callbacks = config.get("callbacks", ()) if isinstance(config, Mapping) else ()
    event = {
        "type": event_type,
        "input": input_value,
        "output": output,
        "error": error,
    }
    for callback in callbacks:
        if callable(callback):
            result = callback(event)
        else:
            method_name = {
                "start": "handle_chain_start",
                "end": "handle_chain_end",
                "error": "handle_chain_error",
            }[event_type]
            handler = getattr(callback, method_name, None)
            if not callable(handler):
                continue
            result = handler(event)
        if inspect.isawaitable(result):
            await result


class _SequenceRunnable:
    def __init__(self, steps):
        self.steps = tuple(steps)

    async def ainvoke(self, value, config=None, **kwargs):
        current = value
        for step in self.steps:
            current = await step.ainvoke(current, config=config, **kwargs)
        return current

    def invoke(self, value, config=None, **kwargs):
        return _run_sync(self.ainvoke(value, config=config, **kwargs))

    def pipe(self, next_runnable):
        return _SequenceRunnable((*self.steps, next_runnable))


class _RetryRunnable:
    def __init__(self, runnable, max_attempts=3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.runnable = runnable
        self.max_attempts = max_attempts

    async def ainvoke(self, value, config=None, **kwargs):
        last_error = None
        for _ in range(self.max_attempts):
            try:
                return await self.runnable.ainvoke(value, config=config, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise last_error

    def invoke(self, value, config=None, **kwargs):
        return _run_sync(self.ainvoke(value, config=config, **kwargs))


class _FallbackRunnable:
    def __init__(self, runnable, fallbacks):
        self.runnable = runnable
        self.fallbacks = tuple(fallbacks)

    async def ainvoke(self, value, config=None, **kwargs):
        last_error = None
        for candidate in (self.runnable, *self.fallbacks):
            try:
                return await candidate.ainvoke(value, config=config, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise last_error

    def invoke(self, value, config=None, **kwargs):
        return _run_sync(self.ainvoke(value, config=config, **kwargs))


class _ConfigurableRunnable:
    def __init__(self, runnable, fields):
        self.runnable = runnable
        self.fields = dict(fields)

    async def ainvoke(self, value, config=None, **kwargs):
        configurable = (
            config.get("configurable", {}) if isinstance(config, Mapping) else {}
        )
        mapped = dict(kwargs)
        for field, spec in self.fields.items():
            config_key = spec if isinstance(spec, str) else getattr(spec, "id", field)
            if config_key in configurable:
                mapped[field] = configurable[config_key]
        return await self.runnable.ainvoke(value, config=config, **mapped)

    def invoke(self, value, config=None, **kwargs):
        return _run_sync(self.ainvoke(value, config=config, **kwargs))


class RunnableLambda:
    def __init__(self, func):
        if not callable(func):
            raise TypeError("RunnableLambda requires a callable")
        self.func = func

    async def ainvoke(self, value, config=None, **kwargs):
        try:
            result = self.func(value, config=config, **kwargs)
        except TypeError:
            result = self.func(value)
        if inspect.isawaitable(result):
            result = await result
        return result

    def invoke(self, value, config=None, **kwargs):
        return _run_sync(self.ainvoke(value, config=config, **kwargs))

    def pipe(self, next_runnable):
        return _SequenceRunnable((self, next_runnable))


class RunnablePassthrough:
    async def ainvoke(self, value, config=None, **kwargs):
        del config, kwargs
        return value

    def invoke(self, value, config=None, **kwargs):
        del config, kwargs
        return value

    def pipe(self, next_runnable):
        return _SequenceRunnable((self, next_runnable))


class MessagesPlaceholder:
    def __init__(self, variable_name, *, optional=False):
        self.variable_name = str(variable_name)
        self.optional = bool(optional)


class ChatPromptTemplate:
    def __init__(self, messages):
        self.messages = tuple(messages)

    @classmethod
    def from_messages(cls, messages):
        return cls(messages)

    def format_messages(self, **values):
        output = []
        for item in self.messages:
            if isinstance(item, MessagesPlaceholder):
                if item.variable_name not in values:
                    if item.optional:
                        continue
                    raise KeyError(item.variable_name)
                output.extend(_messages_input(values[item.variable_name]))
                continue
            if isinstance(item, BaseMessage):
                output.append(item)
                continue
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    "chat prompt messages must be BaseMessage, MessagesPlaceholder, or (role, template)"
                )
            role, template = item
            content = safe_format(template, values)
            normalized = {"human": "user", "ai": "assistant"}.get(str(role), str(role))
            if normalized == "system":
                output.append(SystemMessage(content))
            elif normalized == "assistant":
                output.append(AIMessage(content))
            elif normalized == "tool":
                output.append(ToolMessage(content, tool_call_id=str(values.get("tool_call_id", ""))))
            else:
                output.append(HumanMessage(content))
        return output

    def format(self, **values):
        return "\n".join(message.text for message in self.format_messages(**values))

    async def ainvoke(self, values, config=None, **kwargs):
        del config, kwargs
        if not isinstance(values, Mapping):
            raise TypeError("ChatPromptTemplate input must be a mapping")
        return self.format_messages(**dict(values))

    def invoke(self, values, config=None, **kwargs):
        del config, kwargs
        if not isinstance(values, Mapping):
            raise TypeError("ChatPromptTemplate input must be a mapping")
        return self.format_messages(**dict(values))

    def pipe(self, next_runnable):
        return _SequenceRunnable((self, next_runnable))


class ChatOpenAI:
    def __init__(
        self,
        model,
        *,
        base_url=None,
        temperature=None,
        max_tokens=None,
        provider=None,
        tools=(),
        response_format=None,
        **kwargs,
    ):
        credential = kwargs.pop("api_" + "key", None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _openai_provider(
            model=model, credential=credential, base_url=base_url, provider=provider
        )
        self._tools = tuple(tools)
        self._response_format = response_format
        self._bound_kwargs = dict(kwargs)

    def _clone(self, *, tools=None, response_format=None, bound_kwargs=None):
        clone = object.__new__(type(self))
        clone.model = self.model
        clone.temperature = self.temperature
        clone.max_tokens = self.max_tokens
        clone.provider = self.provider
        clone._tools = self._tools if tools is None else tuple(tools)
        clone._response_format = (
            self._response_format if response_format is None else response_format
        )
        clone._bound_kwargs = (
            dict(self._bound_kwargs) if bound_kwargs is None else dict(bound_kwargs)
        )
        return clone

    def bind(self, **kwargs):
        options = {**self._bound_kwargs, **kwargs}
        response_format = options.pop("response_format", self._response_format)
        return self._clone(
            response_format=response_format,
            bound_kwargs=options,
        )

    def bind_tools(self, tools, **kwargs):
        return self._clone(
            tools=tuple(tools),
            bound_kwargs={**self._bound_kwargs, **kwargs},
        )

    def with_retry(self, *, stop_after_attempt=3, **_):
        return _RetryRunnable(self, max_attempts=int(stop_after_attempt))

    def with_fallbacks(self, fallbacks, **_):
        return _FallbackRunnable(self, fallbacks)

    def pipe(self, next_runnable):
        return _SequenceRunnable((self, next_runnable))

    def configurable_fields(self, **fields):
        return _ConfigurableRunnable(self, fields)

    def with_structured_output(
        self,
        schema,
        *,
        include_raw=False,
        method=None,
        strict=None,
        **_,
    ):
        del method, strict
        return _StructuredRunnable(
            self._clone(response_format=schema),
            schema,
            include_raw=include_raw,
        )

    def _make_request(self, messages, kwargs):
        options = {**self._bound_kwargs, **kwargs}
        tools = tuple(
            _langchain_tool_schema(item) for item in options.get("tools", self._tools)
        )
        response_format = options.get("response_format", self._response_format)
        if isinstance(response_format, Mapping) and not any(
            key in response_format for key in ("schema", "json_schema", "format")
        ):
            response_format = {"schema": dict(response_format)}
        return _request(
            model=self.model,
            messages=_messages_input(messages),
            temperature=options.get("temperature", self.temperature),
            max_tokens=options.get("max_tokens", self.max_tokens),
            tools=tools,
            response_format=response_format,
        )

    async def ainvoke(self, messages, config=None, **kwargs):
        await _emit_callbacks(config, "start", input_value=messages)
        try:
            response = await self.provider.complete(
                self._make_request(messages, kwargs)
            )
            output = _ai_message(response)
        except Exception as exc:
            await _emit_callbacks(config, "error", input_value=messages, error=exc)
            raise
        await _emit_callbacks(config, "end", input_value=messages, output=output)
        return output

    def invoke(self, messages, config=None, **kwargs):
        return _run_sync(self.ainvoke(messages, config=config, **kwargs))

    async def abatch(self, inputs, config=None, **kwargs):
        configs = list(config) if isinstance(config, (list, tuple)) else None
        shared = config if isinstance(config, Mapping) else None
        max_concurrency = int((shared or {}).get("max_concurrency", len(inputs) or 1))
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run(index, item):
            async with semaphore:
                item_config = configs[index] if configs is not None else shared
                return await self.ainvoke(item, config=item_config, **kwargs)

        return await asyncio.gather(
            *(run(index, item) for index, item in enumerate(inputs))
        )

    def batch(self, inputs, config=None, **kwargs):
        return _run_sync(self.abatch(inputs, config=config, **kwargs))

    async def astream(self, messages, **kwargs):
        request = self._make_request(messages, kwargs)
        stream = getattr(self.provider, "stream", None)
        if not callable(stream):
            yield AIMessageChunk((await self.ainvoke(messages, **kwargs)).text)
            return
        tool_buffers = {}
        async for event in stream(request):
            if event.type == "text_delta" and event.text:
                yield AIMessageChunk(event.text)
            elif event.type == "tool_call_delta":
                call_id = event.tool_call_id or ""
                current = tool_buffers.setdefault(
                    call_id, {"name": event.tool_name or "", "arguments": ""}
                )
                if event.tool_name:
                    current["name"] = event.tool_name
                current["arguments"] += event.arguments_delta or ""
                try:
                    args = json.loads(current["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield AIMessageChunk(
                    "",
                    tool_calls=[
                        {
                            "id": call_id,
                            "name": current["name"],
                            "args": args,
                            "type": "tool_call_chunk",
                        }
                    ],
                    additional_kwargs={
                        "arguments_delta": event.arguments_delta or "",
                        "arguments": current["arguments"],
                    },
                )
            elif event.type == "completed" and event.response is not None:
                response = event.response
                yield AIMessageChunk(
                    "",
                    response_metadata={
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                    },
                    usage_metadata=_usage_metadata(response.usage),
                )

    def stream(self, messages, **kwargs):
        async def collect():
            return [chunk async for chunk in self.astream(messages, **kwargs)]

        return iter(_run_sync(collect()))


class ChatAnthropic(ChatOpenAI):
    def __init__(
        self,
        model,
        *,
        base_url=None,
        temperature=None,
        max_tokens=None,
        provider=None,
        tools=(),
        response_format=None,
        **kwargs,
    ):
        credential = kwargs.pop("api_" + "key", None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _anthropic_provider(
            model=model, credential=credential, base_url=base_url, provider=provider
        )
        self._tools = tuple(tools)
        self._response_format = response_format
        self._bound_kwargs = dict(kwargs)


def init_chat_model(model, *, model_provider=None, **kwargs):
    provider_name = model_provider
    model_name = model
    if isinstance(model, str) and ":" in model:
        prefix, candidate = model.split(":", 1)
        if prefix in {"openai", "anthropic"}:
            provider_name = provider_name or prefix
            model_name = candidate
    if provider_name == "anthropic":
        return ChatAnthropic(model_name, **kwargs)
    if provider_name in (None, "openai"):
        return ChatOpenAI(model_name, **kwargs)
    raise ValueError(
        f"unsupported LangChain compatibility model provider: {provider_name}"
    )


def tool(name_or_callable=None, *, description=None, args_schema=None):
    def decorate(func, name=None):
        func.name = name or getattr(func, "__name__", "tool")
        func.description = description or inspect.getdoc(func) or ""
        func.args_schema = args_schema

        def invoke(arguments):
            if isinstance(arguments, Mapping):
                return func(**dict(arguments))
            return func(arguments)

        async def ainvoke(arguments):
            result = invoke(arguments)
            if inspect.isawaitable(result):
                return await result
            return result

        func.invoke = invoke
        func.ainvoke = ainvoke
        return func

    if callable(name_or_callable):
        return decorate(name_or_callable)
    if isinstance(name_or_callable, str):
        return lambda func: decorate(func, name_or_callable)
    if name_or_callable is None:
        return decorate
    raise TypeError("tool expects a callable or optional tool name")


def _tool_name(value):
    schema = _langchain_tool_schema(value)
    function = schema.get("function", schema)
    return str(function.get("name", ""))


async def _execute_tool(value, arguments, runtime=None):
    if callable(value):
        kwargs = dict(arguments)
        if runtime is not None:
            try:
                parameters = inspect.signature(value).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "runtime" in parameters:
                kwargs["runtime"] = runtime
            if "store" in parameters:
                kwargs["store"] = runtime.store
            if "context" in parameters:
                kwargs["context"] = runtime.context
            if "state" in parameters:
                kwargs["state"] = runtime.state
        result = value(**kwargs)
    elif callable(getattr(value, "ainvoke", None)):
        payload = dict(arguments)
        if runtime is not None:
            payload["runtime"] = runtime
        return await value.ainvoke(payload)
    elif callable(getattr(value, "invoke", None)):
        payload = dict(arguments)
        if runtime is not None:
            payload["runtime"] = runtime
        result = value.invoke(payload)
    else:
        raise TypeError(
            f"tool {_tool_name(value)!r} has schema but no executable callable"
        )
    if inspect.isawaitable(result):
        return await result
    return result


def _tool_result_text(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class ProviderStrategy:
    def __init__(self, schema, *, strict=True, max_retries=1):
        self.schema = schema
        self.strict = bool(strict)
        self.max_retries = int(max_retries)


class ToolStrategy:
    def __init__(
        self,
        schema,
        *,
        tool_name="structured_response",
        handle_errors=True,
        max_retries=2,
        tool_message_content=None,
    ):
        self.schema = schema
        self.tool_name = str(tool_name)
        self.handle_errors = bool(handle_errors)
        self.max_retries = int(max_retries)
        self.tool_message_content = tool_message_content


class AutoStrategy:
    def __init__(self, schema, *, strict=True, max_retries=1):
        self.schema = schema
        self.strict = bool(strict)
        self.max_retries = int(max_retries)


def _response_strategy(value):
    if value is None:
        return None
    if isinstance(value, (ProviderStrategy, ToolStrategy, AutoStrategy)):
        return value
    return AutoStrategy(value)


def _strategy_schema(strategy):
    return strategy.schema if strategy is not None else None


def _strategy_tool(strategy):
    if not isinstance(strategy, ToolStrategy):
        return None
    schema = _schema_mapping(strategy.schema)
    if schema is None and isinstance(strategy.schema, Mapping):
        schema = dict(strategy.schema)
    if schema is None:
        raise TypeError("ToolStrategy requires JSON Schema or a Pydantic-style schema")
    return {
        "type": "function",
        "function": {
            "name": strategy.tool_name,
            "description": "Return the final structured response.",
            "parameters": schema,
        },
    }


class MemorySaver:
    def __init__(self):
        self._threads = {}

    def get(self, thread_id):
        value = self._threads.get(str(thread_id))
        if value is None:
            return None
        return {
            "messages": list(value.get("messages", ())),
            "state": dict(value.get("state", {})),
        }

    def put(self, thread_id, value):
        self._threads[str(thread_id)] = {
            "messages": list(value.get("messages", ())),
            "state": dict(value.get("state", {})),
        }
        return value

    def delete(self, thread_id):
        return self._threads.pop(str(thread_id), None) is not None


class InMemoryStore:
    def __init__(self):
        self._values = {}

    def put(self, namespace, key, value):
        ns = (
            tuple(namespace)
            if isinstance(namespace, (list, tuple))
            else (str(namespace),)
        )
        self._values[(ns, str(key))] = value

    def get(self, namespace, key):
        ns = (
            tuple(namespace)
            if isinstance(namespace, (list, tuple))
            else (str(namespace),)
        )
        return self._values.get((ns, str(key)))

    def delete(self, namespace, key):
        ns = (
            tuple(namespace)
            if isinstance(namespace, (list, tuple))
            else (str(namespace),)
        )
        return self._values.pop((ns, str(key)), None) is not None

    def search(self, namespace):
        ns = (
            tuple(namespace)
            if isinstance(namespace, (list, tuple))
            else (str(namespace),)
        )
        return [
            {"namespace": key_ns, "key": key, "value": value}
            for (key_ns, key), value in self._values.items()
            if key_ns[: len(ns)] == ns
        ]


class AgentRuntime:
    def __init__(self, *, context=None, state=None, store=None, config=None):
        self.context = context
        self.state = state if state is not None else {}
        self.store = store
        self.config = config or {}


class AgentMiddleware:
    async def before_model(self, runtime, messages):
        return messages

    async def after_model(self, runtime, response):
        return response

    async def wrap_tool_call(self, runtime, call, handler):
        return await handler(call)

    async def after_agent(self, runtime, result):
        return result


class ModelCallLimitMiddleware(AgentMiddleware):
    def __init__(self, max_calls):
        self.max_calls = int(max_calls)
        self.calls = 0

    async def before_model(self, runtime, messages):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError(f"model call limit exceeded: {self.max_calls}")
        return messages


class ToolCallLimitMiddleware(AgentMiddleware):
    def __init__(self, max_calls):
        self.max_calls = int(max_calls)
        self.calls = 0

    async def wrap_tool_call(self, runtime, call, handler):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError(f"tool call limit exceeded: {self.max_calls}")
        return await handler(call)


class HumanInTheLoopMiddleware(AgentMiddleware):
    def __init__(self, approval):
        self.approval = approval

    async def wrap_tool_call(self, runtime, call, handler):
        decision = self.approval(call, runtime)
        if inspect.isawaitable(decision):
            decision = await decision
        if not decision:
            raise RuntimeError(f"tool call {call['name']!r} requires approval")
        return await handler(call)


class PIIMiddleware(AgentMiddleware):
    def __init__(self, patterns=()):
        import re

        self.patterns = tuple(re.compile(pattern) for pattern in patterns)

    async def before_model(self, runtime, messages):
        for message in messages:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            redacted = content
            for pattern in self.patterns:
                redacted = pattern.sub("[REDACTED]", redacted)
            if redacted != content:
                message.content = redacted
        return messages


class SummarizationMiddleware(AgentMiddleware):
    def __init__(self, max_messages=20):
        self.max_messages = int(max_messages)

    async def before_model(self, runtime, messages):
        if len(messages) <= self.max_messages:
            return messages
        keep = max(2, self.max_messages - 1)
        prefix = messages[:-keep]
        summary = "\n".join(getattr(item, "text", str(item)) for item in prefix)
        return [SystemMessage("Conversation summary:\n" + summary), *messages[-keep:]]


class LLMToolSelectorMiddleware(AgentMiddleware):
    def __init__(self, selector):
        self.selector = selector

    def select_tools(self, runtime, tools):
        selected = self.selector(runtime, tools)
        return tuple(selected)


def _validate_schema_value(schema, value, label):
    if schema is None:
        return value
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(value)
    if isinstance(schema, Mapping):
        required = schema.get("required", ())
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{label} missing required fields: {', '.join(missing)}")
    return value


def _thread_id(config):
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable", {})
    if isinstance(configurable, Mapping):
        value = configurable.get("thread_id")
        return str(value) if value is not None else None
    return None


def _checkpoint_get(checkpointer, thread_id):
    if checkpointer is None or thread_id is None:
        return None
    getter = getattr(checkpointer, "get", None)
    if callable(getter):
        try:
            return getter(thread_id)
        except TypeError:
            return getter({"configurable": {"thread_id": thread_id}})
    loader = getattr(checkpointer, "load", None)
    if callable(loader):
        checkpoint = loader(thread_id)
        if checkpoint is None:
            return None
        return {
            "messages": [
                _from_model_message(message) for message in checkpoint.messages
            ],
            "state": dict(checkpoint.metadata.get("state", {})),
        }
    raise TypeError("checkpointer must expose get()/put() or Agent RT load()/save()")


def _checkpoint_put(checkpointer, thread_id, messages, state):
    if checkpointer is None or thread_id is None:
        return
    value = {"messages": list(messages), "state": dict(state)}
    putter = getattr(checkpointer, "put", None)
    if callable(putter):
        try:
            putter(thread_id, value)
        except TypeError:
            putter({"configurable": {"thread_id": thread_id}}, value)
        return
    saver = getattr(checkpointer, "save", None)
    if callable(saver):
        from agent_rt import AgentCheckpoint

        native = _request(messages, model="langchain-checkpoint").messages
        saver(
            AgentCheckpoint(
                checkpoint_id=thread_id,
                agent_name="langchain-compat",
                messages=tuple(native),
                metadata={"state": dict(state)},
            )
        )
        return
    raise TypeError("checkpointer must expose get()/put() or Agent RT load()/save()")


def _from_model_message(message):
    text = "".join(part.text or "" for part in message.content if part.type == "text")
    if message.role == "assistant":
        calls = [
            {
                "id": call.id,
                "name": call.name,
                "args": dict(call.arguments),
                "type": "tool_call",
            }
            for call in message.tool_calls
        ]
        return AIMessage(text, tool_calls=calls)
    if message.role == "system":
        return SystemMessage(text)
    if message.role == "tool":
        return ToolMessage(text, tool_call_id=message.tool_call_id or "")
    return HumanMessage(text)


class _CompatAgent:
    def __init__(
        self,
        *,
        model,
        tools=(),
        system_prompt=None,
        response_format=None,
        max_iterations=25,
        middleware=(),
        checkpointer=None,
        store=None,
        context_schema=None,
        state_schema=None,
    ):
        self.model = init_chat_model(model) if isinstance(model, str) else model
        self.tools = tuple(tools)
        self.system_prompt = system_prompt
        self.response_strategy = _response_strategy(response_format)
        self.response_format = _strategy_schema(self.response_strategy)
        self.max_iterations = max_iterations
        self.middleware = tuple(middleware or ())
        self.checkpointer = checkpointer
        self.store = store
        self.context_schema = context_schema
        self.state_schema = state_schema
        self._tool_map = {_tool_name(item): item for item in self.tools}
        self._bound_model = (
            self.model.bind_tools(self.tools) if self.tools else self.model
        )

    async def _run(self, inputs, *, config=None, emit=None):
        incoming = _messages_input(inputs)
        thread_id = _thread_id(config)
        checkpoint = _checkpoint_get(self.checkpointer, thread_id)
        state = dict(checkpoint.get("state", {})) if checkpoint else {}
        if isinstance(inputs, Mapping):
            supplied_state = inputs.get("state")
            if isinstance(supplied_state, Mapping):
                state.update(supplied_state)
        state = _validate_schema_value(self.state_schema, state, "state")
        context = None
        if isinstance(config, Mapping):
            configurable = config.get("configurable", {})
            if isinstance(configurable, Mapping):
                context = configurable.get(
                    "context", configurable.get("runtime_context")
                )
        context = _validate_schema_value(self.context_schema, context or {}, "context")
        runtime = AgentRuntime(
            context=context, state=state, store=self.store, config=config
        )
        messages = list(checkpoint.get("messages", ())) if checkpoint else []
        if messages and incoming and getattr(incoming[0], "role", None) == "system":
            incoming = incoming[1:]
        messages.extend(incoming)
        if self.system_prompt and (
            not messages or getattr(messages[0], "role", None) != "system"
        ):
            messages.insert(0, SystemMessage(self.system_prompt))
        structured_retries = 0
        for _ in range(self.max_iterations):
            model_messages = messages
            for item in self.middleware:
                hook = getattr(item, "before_model", None)
                if callable(hook):
                    model_messages = await hook(runtime, model_messages)
            selected_tools = self.tools
            for item in self.middleware:
                selector = getattr(item, "select_tools", None)
                if callable(selector):
                    selected_tools = tuple(selector(runtime, selected_tools))
            strategy_tool = _strategy_tool(self.response_strategy)
            model_tools = (
                (*selected_tools, strategy_tool) if strategy_tool else selected_tools
            )
            bound_model = (
                self.model.bind_tools(model_tools) if model_tools else self.model
            )
            response_kwargs = {}
            if self.response_strategy is not None and not isinstance(
                self.response_strategy, ToolStrategy
            ):
                response_kwargs["response_format"] = self.response_format
            response = await bound_model.ainvoke(model_messages, **response_kwargs)
            for item in reversed(self.middleware):
                hook = getattr(item, "after_model", None)
                if callable(hook):
                    response = await hook(runtime, response)
            messages.append(response)
            if emit is not None:
                await emit("model", [response])
            if isinstance(self.response_strategy, ToolStrategy):
                structured_call = next(
                    (
                        call
                        for call in response.tool_calls
                        if call["name"] == self.response_strategy.tool_name
                    ),
                    None,
                )
                if structured_call is not None:
                    try:
                        parsed = _parse_structured_value(
                            self.response_strategy.schema,
                            json.dumps(structured_call.get("args", {})),
                        )
                    except Exception as exc:
                        if (
                            not self.response_strategy.handle_errors
                            or structured_retries >= self.response_strategy.max_retries
                        ):
                            raise
                        structured_retries += 1
                        error_message = ToolMessage(
                            f"Structured response validation failed: {exc}",
                            tool_call_id=structured_call["id"],
                            name=self.response_strategy.tool_name,
                        )
                        messages.append(error_message)
                        if emit is not None:
                            await emit("tools", [error_message])
                        continue
                    result = {
                        "messages": messages,
                        "state": runtime.state,
                        "structured_response": parsed,
                    }
                    for item in reversed(self.middleware):
                        hook = getattr(item, "after_agent", None)
                        if callable(hook):
                            result = await hook(runtime, result)
                    _checkpoint_put(
                        self.checkpointer,
                        thread_id,
                        result.get("messages", messages),
                        runtime.state,
                    )
                    return result
            if not response.tool_calls:
                result = {"messages": messages, "state": runtime.state}
                if self.response_format is not None:
                    try:
                        result["structured_response"] = _parse_structured_value(
                            self.response_format, response.text
                        )
                    except Exception:
                        retry_limit = getattr(self.response_strategy, "max_retries", 0)
                        if structured_retries >= retry_limit:
                            raise
                        structured_retries += 1
                        messages.append(
                            SystemMessage(
                                "The previous structured response failed validation. "
                                "Return only a valid response matching the required schema."
                            )
                        )
                        continue
                for item in reversed(self.middleware):
                    hook = getattr(item, "after_agent", None)
                    if callable(hook):
                        result = await hook(runtime, result)
                _checkpoint_put(
                    self.checkpointer,
                    thread_id,
                    result.get("messages", messages),
                    runtime.state,
                )
                return result
            tool_messages = []
            for call in response.tool_calls:
                name = call["name"]
                if name not in self._tool_map:
                    raise RuntimeError(f"model requested unknown tool {name!r}")
                if call.get("argument_error"):
                    # Malformed model arguments are never executed (not even with
                    # defaults); the model gets the error and can retry.
                    rejected = ToolMessage(
                        f"Tool call rejected: {call['argument_error']}",
                        tool_call_id=call["id"],
                        name=name,
                    )
                    messages.append(rejected)
                    tool_messages.append(rejected)
                    continue

                async def execute(current_call, tool_name=name):
                    return await _execute_tool(
                        self._tool_map[tool_name], current_call.get("args", {}), runtime
                    )

                handler = execute
                for item in reversed(self.middleware):
                    hook = getattr(item, "wrap_tool_call", None)
                    if not callable(hook):
                        continue
                    inner = handler

                    async def wrapped(current_call, hook=hook, inner=inner):
                        return await hook(runtime, current_call, inner)

                    handler = wrapped
                output = await handler(call)
                tool_message = ToolMessage(
                    _tool_result_text(output),
                    tool_call_id=call["id"],
                    name=name,
                )
                messages.append(tool_message)
                tool_messages.append(tool_message)
            if emit is not None:
                await emit("tools", tool_messages)
        raise RuntimeError(
            f"LangChain compatibility agent exceeded {self.max_iterations} iterations"
        )

    async def ainvoke(self, inputs, config=None, **_):
        return await self._run(inputs, config=config)

    def invoke(self, inputs, config=None, **kwargs):
        return _run_sync(self.ainvoke(inputs, config=config, **kwargs))

    async def astream(self, inputs, config=None, *, stream_mode="updates", **_):
        queue = asyncio.Queue()

        async def emit(node, messages):
            if stream_mode == "updates":
                await queue.put({node: {"messages": messages}})
            elif stream_mode == "messages":
                for message in messages:
                    await queue.put((message, {"langgraph_node": node}))
            else:
                await queue.put({node: {"messages": messages}})

        async def run():
            await self._run(inputs, config=config, emit=emit)

        task = asyncio.create_task(run())
        try:
            while not task.done() or not queue.empty():
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=0.05)
                except (asyncio.TimeoutError, TimeoutError):
                    continue
            await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def stream(self, inputs, config=None, *, stream_mode="updates", **kwargs):
        async def collect():
            return [
                chunk
                async for chunk in self.astream(
                    inputs,
                    config=config,
                    stream_mode=stream_mode,
                    **kwargs,
                )
            ]

        return iter(_run_sync(collect()))


def create_agent(
    model,
    tools=(),
    *,
    system_prompt=None,
    response_format=None,
    middleware=(),
    checkpointer=None,
    store=None,
    context_schema=None,
    state_schema=None,
    **kwargs,
):
    return _CompatAgent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        response_format=response_format,
        max_iterations=int(kwargs.pop("max_iterations", 25)),
        middleware=middleware,
        checkpointer=checkpointer,
        store=store,
        context_schema=context_schema,
        state_schema=state_schema,
    )


from ext.compat.langchain_retrieval import (
    AgentRTEmbeddings,
    AgentRTRetriever,
    AgentRTVectorStore,
    Chroma,
    ContextualCompressionRetriever,
    CSVLoader,
    DirectoryLoader,
    Document,
    EnsembleRetriever,
    ExternalVectorStoreRetriever,
    JSONLoader,
    JsonOutputParser,
    MapReduceDocumentsChain,
    MemoryVectorStore,
    Milvus,
    Pinecone,
    PineconeVectorStore,
    PromptTemplate,
    PydanticOutputParser,
    Qdrant,
    QdrantVectorStore,
    RecursiveCharacterTextSplitter,
    RefineDocumentsChain,
    RetrievalChain,
    StringOutputParser,
    StrOutputParser,
    StuffDocumentsChain,
    TextLoader,
    Weaviate,
    WeaviateVectorStore,
    create_map_reduce_documents_chain,
    create_refine_documents_chain,
    create_retrieval_chain,
    create_stuff_documents_chain,
)


class OpenAIEmbeddings(AgentRTEmbeddings):
    """``langchain_openai.OpenAIEmbeddings`` over an Agent RT OpenAI provider."""

    def __init__(
        self,
        model="text-embedding-3-small",
        *,
        base_url=None,
        provider=None,
        dimensions=None,
        **kwargs,
    ):
        credential = kwargs.pop("api_" + "key", None)
        base_url = base_url or kwargs.pop("openai_api_base", None)
        super().__init__(
            _openai_provider(
                model=model,
                credential=credential,
                base_url=base_url,
                provider=provider,
            ),
            model,
        )
        self.dimensions = dimensions


def init_embeddings(model, *, provider=None, **kwargs):
    provider_name = provider
    model_name = model
    if isinstance(model, str) and ":" in model:
        provider_name, model_name = model.split(":", 1)
    if provider_name in (None, "openai"):
        return OpenAIEmbeddings(model_name, **kwargs)
    raise ValueError(
        f"unsupported LangChain compatibility embeddings provider: {provider_name}"
    )


# Upstream LangChain v1 submodules that resolve to the combined compatibility
# module, so `from langchain.chat_models.base import init_chat_model` keeps
# its upstream path after the `agent_rt.` prefix.
_LANGCHAIN_SUBMODULES = (
    "agents",
    "agents.middleware",
    "agents.structured_output",
    "chat_models",
    "chat_models.base",
    "embeddings",
    "messages",
    "tools",
)


def _module(name, **exports):
    module = types.ModuleType(name)
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    return module


def install_langchain_compat(parent):
    module = _module(
        f"{parent.__name__}.langchain",
        AIMessage=AIMessage,
        AIMessageChunk=AIMessageChunk,
        BaseMessage=BaseMessage,
        ChatAnthropic=ChatAnthropic,
        ChatOpenAI=ChatOpenAI,
        ChatPromptTemplate=ChatPromptTemplate,
        HumanMessage=HumanMessage,
        MessagesPlaceholder=MessagesPlaceholder,
        RunnableLambda=RunnableLambda,
        RunnablePassthrough=RunnablePassthrough,
        SystemMessage=SystemMessage,
        ToolMessage=ToolMessage,
        MemorySaver=MemorySaver,
        InMemoryStore=InMemoryStore,
        AgentRuntime=AgentRuntime,
        AgentMiddleware=AgentMiddleware,
        ModelCallLimitMiddleware=ModelCallLimitMiddleware,
        ToolCallLimitMiddleware=ToolCallLimitMiddleware,
        HumanInTheLoopMiddleware=HumanInTheLoopMiddleware,
        PIIMiddleware=PIIMiddleware,
        SummarizationMiddleware=SummarizationMiddleware,
        LLMToolSelectorMiddleware=LLMToolSelectorMiddleware,
        ProviderStrategy=ProviderStrategy,
        ToolStrategy=ToolStrategy,
        AutoStrategy=AutoStrategy,
        create_agent=create_agent,
        init_chat_model=init_chat_model,
        init_embeddings=init_embeddings,
        BaseChatModel=ChatOpenAI,
        _ConfigurableModel=ChatOpenAI,
        OpenAIEmbeddings=OpenAIEmbeddings,
        tool=tool,
        AgentRTEmbeddings=AgentRTEmbeddings,
        AgentRTRetriever=AgentRTRetriever,
        AgentRTVectorStore=AgentRTVectorStore,
        Chroma=Chroma,
        Milvus=Milvus,
        Pinecone=Pinecone,
        PineconeVectorStore=PineconeVectorStore,
        Qdrant=Qdrant,
        QdrantVectorStore=QdrantVectorStore,
        Weaviate=Weaviate,
        WeaviateVectorStore=WeaviateVectorStore,
        ContextualCompressionRetriever=ContextualCompressionRetriever,
        CSVLoader=CSVLoader,
        DirectoryLoader=DirectoryLoader,
        Document=Document,
        EnsembleRetriever=EnsembleRetriever,
        ExternalVectorStoreRetriever=ExternalVectorStoreRetriever,
        JSONLoader=JSONLoader,
        JsonOutputParser=JsonOutputParser,
        MapReduceDocumentsChain=MapReduceDocumentsChain,
        MemoryVectorStore=MemoryVectorStore,
        PydanticOutputParser=PydanticOutputParser,
        PromptTemplate=PromptTemplate,
        RecursiveCharacterTextSplitter=RecursiveCharacterTextSplitter,
        RefineDocumentsChain=RefineDocumentsChain,
        RetrievalChain=RetrievalChain,
        StrOutputParser=StrOutputParser,
        StringOutputParser=StringOutputParser,
        StuffDocumentsChain=StuffDocumentsChain,
        TextLoader=TextLoader,
        create_map_reduce_documents_chain=create_map_reduce_documents_chain,
        create_refine_documents_chain=create_refine_documents_chain,
        create_retrieval_chain=create_retrieval_chain,
        create_stuff_documents_chain=create_stuff_documents_chain,
    )
    sys.modules[module.__name__] = module
    parent.langchain = module
    module.__path__ = ()
    # The combined module stands in for each submodule; a self-referencing
    # attribute (`langchain.chat_models.base`) is what makes dotted access work.
    for submodule in _LANGCHAIN_SUBMODULES:
        for part in submodule.split("."):
            setattr(module, part, module)
        sys.modules[f"{module.__name__}.{submodule}"] = module
    integrations = ("chroma", "milvus", "qdrant", "weaviate", "pine" + "cone")
    for integration in integrations:
        alias = f"{parent.__name__}.langchain_{integration}"
        sys.modules[alias] = module
        setattr(parent, f"langchain_{integration}", module)


__all__ = (
    "AIMessage",
    "AIMessageChunk",
    "BaseMessage",
    "ChatAnthropic",
    "ChatOpenAI",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
    "create_agent",
    "init_chat_model",
    "install_langchain_compat",
    "tool",
)
