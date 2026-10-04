from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, get_args, get_origin


def _rt():
    return sys.modules["agent_rt"]


def _credential(kwargs):
    return kwargs.pop("api_" + "key", None)


def _text(message):
    return "".join(part.text or "" for part in message.content if part.type == "text")


def _run_sync(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "synchronous compatibility methods cannot run inside an active event loop; use the async method instead"
    )


def _role(value):
    raw = getattr(value, "value", value)
    return {"human": "user", "ai": "assistant"}.get(str(raw), str(raw))


def _content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                out.append(item["text"])
        return "".join(out)
    return "" if value is None else str(value)


def _model_message(role, content, *, tool_call_id=None, tool_calls=()):
    rt = _rt()
    calls = []
    for call in tool_calls:
        if isinstance(call, rt.ToolCall):
            calls.append(call)
            continue
        if isinstance(call, Mapping):
            function = call.get("function", call)
            if not isinstance(function, Mapping):
                continue
            arguments = function.get("arguments", call.get("args", {}))
            call_id = call.get("id", "")
            call_name = function.get("name", call.get("name", ""))
        else:
            function = getattr(call, "function", None)
            if function is None:
                continue
            arguments = getattr(function, "arguments", {})
            call_id = getattr(call, "id", "")
            call_name = getattr(function, "name", getattr(call, "name", ""))
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        calls.append(
            rt.ToolCall(
                id=str(call_id),
                name=str(call_name),
                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
            )
        )
    parts = []
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(rt.ContentPart(type="text", text=item))
                continue
            if not isinstance(item, Mapping):
                continue
            part_type = str(item.get("type", "text"))
            if part_type in {
                "text",
                "image",
                "audio",
                "video",
                "pdf",
                "document",
                "file",
                "json",
            }:
                parts.append(
                    rt.ContentPart(
                        type=part_type,
                        text=(
                            str(item.get("text"))
                            if item.get("text") is not None
                            else None
                        ),
                        data=item.get("data", item.get("source")),
                        mime_type=(
                            str(item.get("mime_type", item.get("mimeType")))
                            if item.get("mime_type", item.get("mimeType")) is not None
                            else None
                        ),
                    )
                )
    else:
        text = _content(content)
        if text:
            parts.append(rt.ContentPart(type="text", text=text))
    return rt.ModelMessage(
        role=_role(role),
        content=tuple(parts),
        tool_calls=tuple(calls),
        tool_call_id=tool_call_id,
    )


def _messages(values):
    out = []
    for value in values:
        if isinstance(value, _rt().ModelMessage):
            out.append(value)
        elif isinstance(value, Mapping):
            out.append(
                _model_message(
                    value.get("role", "user"),
                    value.get("content", ""),
                    tool_call_id=value.get("tool_call_id"),
                    tool_calls=value.get("tool_calls", ()),
                )
            )
        elif isinstance(value, tuple) and len(value) == 2:
            out.append(_model_message(value[0], value[1]))
        else:
            extra = getattr(value, "additional_kwargs", {}) or {}
            out.append(
                _model_message(
                    getattr(value, "role", getattr(value, "type", "user")),
                    getattr(value, "content", ""),
                    tool_call_id=getattr(
                        value, "tool_call_id", extra.get("tool_call_id")
                    ),
                    tool_calls=getattr(
                        value, "tool_calls", extra.get("tool_calls", ())
                    ),
                )
            )
    return tuple(out)


def _json_schema_for_annotation(annotation):
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple, set, frozenset):
        item = args[0] if args else Any
        return {"type": "array", "items": _json_schema_for_annotation(item)}
    if origin is dict:
        return {"type": "object"}
    if origin is not None and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            schema = _json_schema_for_annotation(non_none[0])
            return {"anyOf": [schema, {"type": "null"}]}
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }.get(annotation, {})


def _langchain_tool_schema(value):
    if isinstance(value, Mapping):
        return value
    name = getattr(value, "name", None) or getattr(value, "__name__", None)
    description = getattr(value, "description", None) or inspect.getdoc(value) or ""
    args_schema = getattr(value, "args_schema", None)
    schema = _schema_mapping(args_schema) if args_schema is not None else None
    target = value
    if not callable(target) and callable(getattr(value, "invoke", None)):
        target = value.invoke
    if schema is None and callable(target):
        properties = {}
        required = []
        annotations = {}
        try:
            parameters = inspect.signature(target).parameters.values()
            annotations = inspect.get_annotations(target, eval_str=True)
        except (TypeError, ValueError):
            parameters = ()
        except (NameError, AttributeError):
            annotations = {}
        for parameter in parameters:
            if parameter.name in {"self", "runtime", "config"}:
                continue
            annotation = annotations.get(parameter.name, parameter.annotation)
            properties[parameter.name] = _json_schema_for_annotation(annotation)
            if parameter.default is inspect.Signature.empty:
                required.append(parameter.name)
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
    if not name:
        raise TypeError("tool must be a mapping, callable, or object with a name")
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(description),
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


def _tool_definition(value):
    rt = _rt()
    function = value.get("function", value)
    if not isinstance(function, Mapping):
        raise TypeError("tool definition must be a mapping")
    schema = function.get(
        "parameters", function.get("input_schema", {"type": "object", "properties": {}})
    )
    return rt.ToolDefinition(
        name=str(function.get("name", value.get("name", ""))),
        description=str(function.get("description", value.get("description", ""))),
        input_schema=(
            dict(schema)
            if isinstance(schema, Mapping)
            else {"type": "object", "properties": {}}
        ),
    )


_CODE_TOOL_NAME = "agent_rt_code_execution"
_SHELL_TOOL_NAME = "agent_rt_shell_execution"


def _code_tool_kind(value):
    if not isinstance(value, Mapping):
        return None
    tool_type = str(value.get("type", ""))
    if tool_type == "code_interpreter" or tool_type.startswith("code_execution"):
        return "code"
    if tool_type in {"shell", "local_shell"}:
        return "shell"
    return None


def _code_tool_definition(kind):
    if kind == "shell":
        return {
            "type": "function",
            "function": {
                "name": _SHELL_TOOL_NAME,
                "description": "Execute a command in the configured Agent RT sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "argv": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                },
            },
        }
    return {
        "type": "function",
        "function": {
            "name": _CODE_TOOL_NAME,
            "description": "Execute code in the configured Agent RT sandbox interpreter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "runtime": {"type": "string", "default": "python"},
                },
                "required": ["code"],
            },
        },
    }


def _compat_tools(tools):
    converted = []
    local_kinds = set()
    for tool in tools or ():
        kind = _code_tool_kind(tool)
        if kind is None:
            converted.append(tool)
            continue
        local_kinds.add(kind)
        converted.append(_code_tool_definition(kind))
    return tuple(converted), frozenset(local_kinds)


_FILE_SEARCH_TOOL_NAME = "agent_rt_file_search"


def _file_search_tool(value):
    return isinstance(value, Mapping) and value.get("type") == "file_search"


def _file_search_definition():
    return {
        "type": "function",
        "function": {
            "name": _FILE_SEARCH_TOOL_NAME,
            "description": "Search files using the configured Agent RT retrieval providers.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _file_search_binding(value, retrieval_registry):
    if retrieval_registry is None:
        raise RuntimeError(
            "file_search compatibility requires retrieval_registry=RetrievalRegistry(...)"
        )
    vector_store_ids = value.get("vector_store_ids", ())
    if not isinstance(vector_store_ids, Sequence) or isinstance(
        vector_store_ids, (str, bytes, bytearray)
    ):
        raise TypeError("file_search vector_store_ids must be a sequence")
    names = tuple(str(name) for name in vector_store_ids if str(name))
    if not names:
        raise ValueError("file_search requires at least one vector_store_id")
    max_results = value.get("max_num_results", 10)
    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or max_results < 1
    ):
        raise ValueError("file_search max_num_results must be a positive integer")
    filters = value.get("filters", {})
    if filters is None:
        filters = {}
    if not isinstance(filters, Mapping):
        raise TypeError("file_search filters must be a mapping")
    ranking = value.get("ranking_options", {})
    score_threshold = None
    if isinstance(ranking, Mapping) and ranking.get("score_threshold") is not None:
        score_threshold = float(ranking["score_threshold"])
    return {
        "kind": "file",
        "registry": retrieval_registry,
        "names": names,
        "max_results": max_results,
        "filters": dict(filters),
        "score_threshold": score_threshold,
    }


async def _execute_file_search(binding, call):
    query = call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("file_search requires a non-empty query")
    rt = _rt()
    merged = []
    for name in binding["names"]:
        results = await binding["registry"].search(
            name,
            rt.RetrievalQuery(
                text=query,
                limit=binding["max_results"],
                filters=binding["filters"],
            ),
        )
        for result in results:
            if (
                binding["score_threshold"] is not None
                and result.score is not None
                and result.score < binding["score_threshold"]
            ):
                continue
            merged.append((name, result))
    merged.sort(
        key=lambda item: (
            item[1].score is not None,
            item[1].score if item[1].score is not None else float("-inf"),
        ),
        reverse=True,
    )
    selected = merged[: binding["max_results"]]
    return {
        "query": query,
        "results": [
            {
                "file_id": result.id,
                "filename": result.title,
                "text": (
                    result.content
                    if isinstance(result.content, str)
                    else _serialize_tool_result(result.content)
                ),
                "score": result.score,
                "uri": result.uri,
                "attributes": dict(result.metadata),
                "vector_store_id": name,
            }
            for name, result in selected
        ],
    }


_WEB_SEARCH_TOOL_NAME = "agent_rt_web_search"


def _web_search_tool(value):
    if not isinstance(value, Mapping):
        return False
    tool_type = str(value.get("type", ""))
    return tool_type == "web_search" or tool_type.startswith(
        ("web_search_preview", "web_search_")
    )


def _web_search_definition():
    return {
        "type": "function",
        "function": {
            "name": _WEB_SEARCH_TOOL_NAME,
            "description": "Search the web using the configured Agent RT web-search provider.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _web_search_binding(value, provider):
    if provider is None:
        provider = _rt().EnvironmentWebSearchProvider.from_environment()
    max_results = value.get("max_num_results", value.get("max_results", 10))
    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or max_results < 1
    ):
        max_results = 10
    filters = {}
    if isinstance(value.get("filters"), Mapping):
        filters.update(value["filters"])
    for key in (
        "allowed_domains",
        "blocked_domains",
        "user_location",
        "search_context_size",
    ):
        if value.get(key) is not None:
            filters[key] = value[key]
    return {
        "kind": "web",
        "provider": provider,
        "max_results": max_results,
        "filters": filters,
    }


async def _execute_web_search(binding, call):
    query = call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web_search requires a non-empty query")
    results = await binding["provider"].search(
        _rt().RetrievalQuery(
            text=query,
            limit=binding["max_results"],
            filters=binding["filters"],
        )
    )
    return {
        "query": query,
        "results": [
            {
                "id": result.id,
                "title": result.title,
                "text": (
                    result.content
                    if isinstance(result.content, str)
                    else _serialize_tool_result(result.content)
                ),
                "score": result.score,
                "url": result.uri,
                "metadata": dict(result.metadata),
            }
            for result in results[: binding["max_results"]]
        ],
    }


def _mcp_tool_kind(value):
    if not isinstance(value, Mapping):
        return None
    tool_type = str(value.get("type", ""))
    if tool_type == "mcp":
        return "openai"
    if tool_type == "mcp_toolset":
        return "anthropic"
    return None


def _mcp_server_label(value):
    kind = _mcp_tool_kind(value)
    if kind == "openai":
        return str(value.get("server_label", ""))
    if kind == "anthropic":
        return str(value.get("mcp_server_name", ""))
    return ""


def _validate_anthropic_mcp_servers(mcp_servers, tools):
    servers = tuple(mcp_servers or ())
    toolsets = [tool for tool in (tools or ()) if _mcp_tool_kind(tool) == "anthropic"]
    if not servers and not toolsets:
        return
    names = []
    for server in servers:
        if not isinstance(server, Mapping):
            raise TypeError("Anthropic mcp_servers entries must be mappings")
        name = str(server.get("name", ""))
        if not name:
            raise ValueError("Anthropic MCP server requires a non-empty name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Anthropic MCP server names must be unique")
    references = [_mcp_server_label(tool) for tool in toolsets]
    for name in names:
        if references.count(name) != 1:
            raise ValueError(
                f"Anthropic MCP server {name!r} must be referenced by exactly one mcp_toolset"
            )
    for reference in references:
        if reference not in names:
            raise ValueError(
                f"Anthropic mcp_toolset references undeclared MCP server {reference!r}"
            )


def _allowed_mcp_tool_names(value):
    allowed = value.get("allowed_tools") if isinstance(value, Mapping) else None
    if allowed is None and isinstance(value, Mapping):
        allowed = (
            value.get("tool_configuration", {}).get("allowed_tools")
            if isinstance(value.get("tool_configuration"), Mapping)
            else None
        )
    if allowed is None:
        return None
    if isinstance(allowed, Mapping):
        allowed = allowed.get("tool_names", allowed.get("tools"))
    if isinstance(allowed, Sequence) and not isinstance(
        allowed, (str, bytes, bytearray)
    ):
        return frozenset(str(name) for name in allowed)
    # Anything else (a string, or a filter such as {"read_only": true}) cannot be
    # enforced here; ignoring it would silently expose every tool.
    raise ValueError(
        "MCP allowed_tools must be a list of tool names (or {'tool_names': [...]}); "
        "other filters such as read_only are not supported"
    )


def _require_mcp_approval_supported(value):
    if _mcp_tool_kind(value) != "openai":
        return
    approval = value.get("require_approval")
    if approval is None or approval == "never":
        return
    raise ValueError(
        "MCP require_approval is not enforced by compatibility mode; pass "
        "require_approval='never' explicitly or gate the MCP client with an "
        "Agent RT AuthorizedMCPClient policy"
    )


def _mcp_tool_enabled(value, name):
    if not isinstance(value, Mapping):
        return True
    if _mcp_tool_kind(value) == "openai":
        allowed = _allowed_mcp_tool_names(value)
        return allowed is None or name in allowed
    if _mcp_tool_kind(value) == "anthropic":
        default_config = value.get("default_config", {})
        default_enabled = (
            bool(default_config.get("enabled", True))
            if isinstance(default_config, Mapping)
            else True
        )
        configs = value.get("configs", {})
        config = None
        if isinstance(configs, Mapping):
            config = configs.get(name)
        elif isinstance(configs, Sequence) and not isinstance(
            configs, (str, bytes, bytearray)
        ):
            for item in configs:
                if isinstance(item, Mapping) and str(item.get("name", "")) == name:
                    config = item
                    break
        if isinstance(config, Mapping) and "enabled" in config:
            return bool(config["enabled"])
        return default_enabled
    return True


def _mcp_safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(value))


def _mcp_function_name(server_label, tool_name):
    return f"mcp__{_mcp_safe_name(server_label)}__{_mcp_safe_name(tool_name)}"


async def _ensure_mcp_initialized(client):
    inner = getattr(client, "client", client)
    if hasattr(inner, "server_info") and inner.server_info is None:
        initialize = getattr(inner, "initialize", None)
        if callable(initialize):
            await initialize()


async def _expand_compat_tools(
    tools, mcp_clients, retrieval_registry=None, web_search_provider=None
):
    converted = []
    local_kinds = set()
    mcp_bindings = {}
    retrieval_bindings = {}
    for tool in tools or ():
        if _file_search_tool(tool):
            converted.append(_file_search_definition())
            retrieval_bindings[_FILE_SEARCH_TOOL_NAME] = _file_search_binding(
                tool, retrieval_registry
            )
            local_kinds.add("file_search")
            continue
        if _web_search_tool(tool):
            converted.append(_web_search_definition())
            retrieval_bindings[_WEB_SEARCH_TOOL_NAME] = _web_search_binding(
                tool, web_search_provider
            )
            local_kinds.add("web_search")
            continue
        mcp_kind = _mcp_tool_kind(tool)
        if mcp_kind is not None:
            server_label = _mcp_server_label(tool)
            if not server_label:
                raise ValueError("MCP tool declaration requires a server label/name")
            client = (mcp_clients or {}).get(server_label)
            if client is None:
                raise RuntimeError(
                    f"MCP compatibility requires mcp_clients[{server_label!r}] with an Agent RT MCP client"
                )
            _require_mcp_approval_supported(tool)
            await _ensure_mcp_initialized(client)
            list_tools = getattr(client, "list_tools", None)
            if not callable(list_tools):
                raise TypeError(
                    f"MCP client {server_label!r} does not support list_tools()"
                )
            discovered = await list_tools()
            for discovered_tool in discovered:
                if not _mcp_tool_enabled(tool, discovered_tool.name):
                    continue
                function_name = _mcp_function_name(server_label, discovered_tool.name)
                if function_name in mcp_bindings:
                    raise ValueError(
                        f"MCP tool name collision for {function_name!r}; rename the server or tool"
                    )
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": discovered_tool.description,
                            "parameters": dict(discovered_tool.input_schema),
                        },
                    }
                )
                mcp_bindings[function_name] = (
                    client,
                    discovered_tool.name,
                    server_label,
                )
            local_kinds.add("mcp")
            continue
        kind = _code_tool_kind(tool)
        if kind is None:
            converted.append(tool)
            continue
        local_kinds.add(kind)
        converted.append(_code_tool_definition(kind))
    return (
        tuple(converted),
        frozenset(local_kinds),
        mcp_bindings,
        retrieval_bindings,
    )


def _serialize_tool_result(value):
    try:
        return json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


async def _execute_mcp_tool(binding, call):
    client, tool_name, _server_label = binding
    call_tool = getattr(client, "call_tool", None)
    if not callable(call_tool):
        raise TypeError("configured Agent RT MCP client does not support call_tool()")
    return await call_tool(tool_name, dict(call.arguments))


def _sandbox_result_payload(result):
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout.decode("utf-8", errors="replace"),
        "stderr": result.stderr.decode("utf-8", errors="replace"),
        "duration_ms": result.duration_ms,
        "truncated": result.truncated,
    }


async def _execute_local_code_tool(session, call):
    if session is None:
        raise RuntimeError(
            "code execution compatibility requires sandbox_session=SandboxSession(...)"
        )
    if call.name == _CODE_TOOL_NAME:
        code = call.arguments.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("code execution tool requires a non-empty code string")
        runtime = call.arguments.get("runtime", "python")
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("code execution runtime must be a non-empty string")
        result = await session.run_code(runtime, code)
        return _sandbox_result_payload(result)
    if call.name == _SHELL_TOOL_NAME:
        argv = call.arguments.get("argv")
        command = call.arguments.get("command")
        if argv is None and isinstance(command, str):
            argv = ("sh", "-lc", command)
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes, bytearray)):
            raise ValueError("shell execution requires command or argv")
        result = await session.execute(
            _rt().SandboxCommand(
                argv=tuple(str(part) for part in argv),
                cwd=str(call.arguments.get("cwd", "")),
            )
        )
        return _sandbox_result_payload(result)
    raise KeyError(call.name)


def _is_local_code_call(call):
    return call.name in {_CODE_TOOL_NAME, _SHELL_TOOL_NAME}


async def _complete_with_local_code_tools(
    provider,
    request,
    *,
    sandbox_session=None,
    mcp_bindings=None,
    mcp_execution_log=None,
    retrieval_bindings=None,
    file_search_execution_log=None,
    web_search_execution_log=None,
    final_history_out=None,
    max_rounds=8,
):
    mcp_bindings = dict(mcp_bindings or {})
    mcp_execution_log = mcp_execution_log if mcp_execution_log is not None else []
    retrieval_bindings = dict(retrieval_bindings or {})
    file_search_execution_log = (
        file_search_execution_log if file_search_execution_log is not None else []
    )
    web_search_execution_log = (
        web_search_execution_log if web_search_execution_log is not None else []
    )
    if sandbox_session is None and not mcp_bindings and not retrieval_bindings:
        response = await provider.complete(request)
        if final_history_out is not None:
            final_history_out[:] = [*request.messages, response.message]
        return response
    current = request
    rt = _rt()
    for _ in range(max_rounds + 1):
        response = await provider.complete(current)
        calls = tuple(response.message.tool_calls)
        local_calls = tuple(
            call
            for call in calls
            if (
                _is_local_code_call(call)
                or call.name in mcp_bindings
                or call.name in retrieval_bindings
            )
        )
        if not local_calls:
            if final_history_out is not None:
                final_history_out[:] = [*current.messages, response.message]
            return response
        if len(local_calls) != len(calls):
            if final_history_out is not None:
                final_history_out[:] = [*current.messages, response.message]
            return response
        history = list(current.messages)
        history.append(response.message)
        for call in local_calls:
            if call.name in mcp_bindings:
                binding = mcp_bindings[call.name]
                payload = await _execute_mcp_tool(binding, call)
                _client, tool_name, server_label = binding
                mcp_execution_log.append(
                    {
                        "id": call.id,
                        "name": tool_name,
                        "server_label": server_label,
                        "arguments": dict(call.arguments),
                        "output": payload,
                    }
                )
            elif call.name in retrieval_bindings:
                retrieval_binding = retrieval_bindings[call.name]
                if retrieval_binding.get("kind") == "web":
                    payload = await _execute_web_search(retrieval_binding, call)
                    web_search_execution_log.append(
                        {
                            "id": call.id,
                            "query": payload["query"],
                            "results": payload["results"],
                        }
                    )
                else:
                    payload = await _execute_file_search(retrieval_binding, call)
                    file_search_execution_log.append(
                        {
                            "id": call.id,
                            "query": payload["query"],
                            "results": payload["results"],
                        }
                    )
            else:
                payload = await _execute_local_code_tool(sandbox_session, call)
            history.append(
                rt.ModelMessage(
                    role="tool",
                    content=(
                        rt.ContentPart(
                            type="text",
                            text=_serialize_tool_result(payload),
                        ),
                    ),
                    tool_call_id=call.id,
                )
            )
        current = rt.ModelRequest(
            model=current.model,
            messages=tuple(history),
            tools=current.tools,
            temperature=current.temperature,
            max_output_tokens=current.max_output_tokens,
            structured_output=current.structured_output,
            tool_selection=current.tool_selection,
            metadata=current.metadata,
            prompt_cache=current.prompt_cache,
            cancellation_token=current.cancellation_token,
        )
    raise RuntimeError("code execution exceeded maximum tool rounds")


async def _openai_chat_local_stream(
    provider,
    request,
    session,
    max_rounds,
    mcp_bindings=None,
    retrieval_bindings=None,
):
    response = await _complete_with_local_code_tools(
        provider,
        request,
        sandbox_session=session,
        mcp_bindings=mcp_bindings,
        retrieval_bindings=retrieval_bindings,
        max_rounds=max_rounds,
    )
    text = _text(response.message)
    if text:
        yield SimpleNamespace(
            id=None,
            object="chat.completion.chunk",
            model=response.model,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=text, tool_calls=[]),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
    yield SimpleNamespace(
        id=None,
        object="chat.completion.chunk",
        model=response.model,
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=None, tool_calls=[]),
                finish_reason=response.finish_reason,
            )
        ],
        usage=_usage_namespace(response.usage),
    )


async def _openai_responses_local_stream(
    provider,
    request,
    session,
    max_rounds,
    mcp_bindings=None,
    retrieval_bindings=None,
    *,
    client=None,
    response_id=None,
):
    final_history = []
    file_search_calls = []
    web_search_calls = []
    response = await _complete_with_local_code_tools(
        provider,
        request,
        sandbox_session=session,
        mcp_bindings=mcp_bindings,
        retrieval_bindings=retrieval_bindings,
        file_search_execution_log=file_search_calls,
        web_search_execution_log=web_search_calls,
        final_history_out=final_history,
        max_rounds=max_rounds,
    )
    if client is not None and response_id is not None:
        _store_response_state(client, response_id, final_history)
    text = _text(response.message)
    if text:
        yield SimpleNamespace(type="response.output_text.delta", delta=text)
    yield SimpleNamespace(
        type="response.completed",
        response=_openai_response_object(
            response,
            response_id=response_id,
            file_search_calls=file_search_calls,
            web_search_calls=web_search_calls,
        ),
    )


async def _anthropic_local_event_stream(
    provider,
    request,
    session,
    max_rounds,
    mcp_bindings=None,
    retrieval_bindings=None,
):
    response = await _complete_with_local_code_tools(
        provider,
        request,
        sandbox_session=session,
        mcp_bindings=mcp_bindings,
        retrieval_bindings=retrieval_bindings,
        max_rounds=max_rounds,
    )
    yield SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            id=None,
            type="message",
            role="assistant",
            model=response.model,
            content=[],
            usage=None,
        ),
    )
    text = _text(response.message)
    if text:
        yield SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text=text),
        )
    yield SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=response.finish_reason),
        usage=_usage_namespace(response.usage, anthropic=True),
    )
    yield SimpleNamespace(type="message_stop")


def _schema_mapping(value):
    if isinstance(value, Mapping):
        return dict(value)
    model_json_schema = getattr(value, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
        if isinstance(schema, Mapping):
            return dict(schema)
    schema = getattr(value, "schema", None)
    if callable(schema):
        schema = schema()
        if isinstance(schema, Mapping):
            return dict(schema)
    return None


def _structured(value):
    if value is None:
        return None
    rt = _rt()
    if not isinstance(value, Mapping):
        schema = _schema_mapping(value)
        if schema is None:
            return None
        return rt.StructuredOutputRequirement(
            schema=schema,
            name=getattr(value, "__name__", None),
            strict=True,
        )

    payload = value
    if isinstance(payload.get("format"), Mapping):
        payload = payload["format"]
    if isinstance(payload.get("json_schema"), Mapping):
        payload = payload["json_schema"]

    if payload.get("type") == "json_schema" and isinstance(
        payload.get("schema"), Mapping
    ):
        return rt.StructuredOutputRequirement(
            schema=dict(payload["schema"]),
            name=str(payload.get("name")) if payload.get("name") else None,
            strict=bool(payload.get("strict", True)),
        )
    if "schema" in payload and isinstance(payload["schema"], Mapping):
        return rt.StructuredOutputRequirement(
            schema=dict(payload["schema"]),
            name=str(payload.get("name")) if payload.get("name") else None,
            strict=bool(payload.get("strict", True)),
        )
    return None


def _reasoning_config(
    *, reasoning=None, reasoning_effort=None, thinking=None, output_config=None
):
    rt = _rt()
    effort = reasoning_effort
    summary = None
    if isinstance(reasoning, Mapping):
        if reasoning.get("effort") is not None:
            effort = reasoning.get("effort")
        if reasoning.get("summary") is not None:
            summary = reasoning.get("summary")
    thinking_type = None
    budget_tokens = None
    if isinstance(thinking, Mapping):
        if thinking.get("type") is not None:
            thinking_type = str(thinking.get("type"))
        if thinking.get("budget_tokens") is not None:
            budget_tokens = int(thinking.get("budget_tokens"))
    if isinstance(output_config, Mapping) and output_config.get("effort") is not None:
        effort = output_config.get("effort")
    if (
        effort is None
        and summary is None
        and thinking_type is None
        and budget_tokens is None
    ):
        return None
    return rt.ReasoningConfig(
        effort=str(effort) if effort is not None else None,
        summary=str(summary) if summary is not None else None,
        thinking=thinking_type,
        budget_tokens=budget_tokens,
    )


def _request(
    *,
    model,
    messages,
    temperature=None,
    max_tokens=None,
    tools=(),
    response_format=None,
    reasoning=None,
):
    rt = _rt()
    return rt.ModelRequest(
        model=model,
        messages=_messages(messages),
        tools=tuple(_tool_definition(t) for t in tools),
        temperature=temperature,
        max_output_tokens=max_tokens,
        structured_output=_structured(response_format),
        reasoning=reasoning,
    )


def _openai_provider(*, model, credential, base_url, provider=None):
    if provider is not None:
        return provider
    rt = _rt()
    env = os.environ
    settings = {
        "base_url": base_url
        or env.get(rt.OPENAI_BASE_URL_ENV, rt.DEFAULT_OPENAI_BASE_URL),
        "default_model": model or env.get(rt.OPENAI_MODEL_ENV),
        "api_" + "key": credential or env.get(rt.OPENAI_API_KEY_ENV),
    }
    return rt.OpenAIModelProvider(rt.OpenAIProviderSettings(**settings))


def _anthropic_provider(*, model, credential, base_url, provider=None):
    if provider is not None:
        return provider
    rt = _rt()
    env = os.environ
    settings = {
        "base_url": base_url
        or env.get(rt.ANTHROPIC_BASE_URL_ENV, rt.DEFAULT_ANTHROPIC_BASE_URL),
        "default_model": model or env.get(rt.ANTHROPIC_MODEL_ENV),
        "api_" + "key": credential or env.get(rt.ANTHROPIC_API_KEY_ENV),
    }
    return rt.AnthropicModelProvider(rt.AnthropicProviderSettings(**settings))


@dataclass
class _LCMessage:
    content: Any
    role: str = "user"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage_metadata: dict[str, Any] | None = None
    id: str | None = None
    name: str | None = None

    @property
    def type(self):
        return {"user": "human", "assistant": "ai"}.get(self.role, self.role)

    @property
    def text(self):
        return _content(self.content)

    @property
    def content_blocks(self):
        if isinstance(self.content, Sequence) and not isinstance(
            self.content, (str, bytes, bytearray)
        ):
            return list(self.content)
        return [{"type": "text", "text": self.text}] if self.text else []


BaseMessage = _LCMessage


class HumanMessage(_LCMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content=content, role="user", **kwargs)


class SystemMessage(_LCMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content=content, role="system", **kwargs)


class AIMessage(_LCMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content=content, role="assistant", **kwargs)


class AIMessageChunk(AIMessage):
    def __add__(self, other):
        if not isinstance(other, AIMessageChunk):
            return NotImplemented
        return AIMessageChunk(
            self.text + other.text,
            tool_calls=[*self.tool_calls, *other.tool_calls],
            additional_kwargs={**self.additional_kwargs, **other.additional_kwargs},
            response_metadata={**self.response_metadata, **other.response_metadata},
            usage_metadata=other.usage_metadata or self.usage_metadata,
        )


class ToolMessage(_LCMessage):
    def __init__(self, content, *, tool_call_id, **kwargs):
        super().__init__(
            content=content, role="tool", tool_call_id=tool_call_id, **kwargs
        )


def _lc_input_messages(value):
    if isinstance(value, str):
        return [HumanMessage(value)]
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if isinstance(value, _LCMessage):
        return [value]
    return list(value)


def _lc_usage_metadata(usage):
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _lc_ai_message(response):
    calls = [
        {"id": c.id, "name": c.name, "args": dict(c.arguments), "type": "tool_call"}
        for c in response.message.tool_calls
    ]
    metadata = {
        "model": response.model,
        "finish_reason": response.finish_reason,
    }
    return AIMessage(
        _text(response.message),
        tool_calls=calls,
        additional_kwargs=dict(metadata),
        response_metadata=dict(metadata),
        usage_metadata=_lc_usage_metadata(response.usage),
    )


def _validate_json_schema_value(schema, value, path="$"):
    if not isinstance(schema, Mapping):
        return
    expected = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, Sequence)
        and not isinstance(item, (str, bytes, bytearray)),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if (
        isinstance(expected, str)
        and expected in type_checks
        and not type_checks[expected](value)
    ):
        raise ValueError(f"{path} must be {expected}")
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and value not in enum:
        raise ValueError(f"{path} must be one of the allowed enum values")
    if expected == "object" and isinstance(value, Mapping):
        required = schema.get("required", ())
        if isinstance(required, Sequence):
            for key in required:
                if key not in value:
                    raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child in properties.items():
                if key in value:
                    _validate_json_schema_value(child, value[key], f"{path}.{key}")
    if (
        expected == "array"
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema_value(item_schema, item, f"{path}[{index}]")


def _parse_structured_value(schema, text):
    value = json.loads(text)
    validator = getattr(schema, "model_validate", None)
    if callable(validator):
        return validator(value)
    validator = getattr(schema, "parse_obj", None)
    if callable(validator):
        return validator(value)
    if inspect.isclass(schema) and not isinstance(schema, Mapping):
        try:
            if isinstance(value, Mapping):
                return schema(**value)
        except TypeError:
            pass
    if isinstance(schema, Mapping):
        _validate_json_schema_value(schema, value)
    return value


class _LCStructuredRunnable:
    def __init__(self, model, schema, include_raw=False):
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
        credential = _credential(kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _openai_provider(
            model=model, credential=credential, base_url=base_url, provider=provider
        )
        self._tools = tuple(tools)
        self._response_format = response_format

    def bind_tools(self, tools, **_):
        return self._clone(tools=tuple(tools))

    def bind(self, **kwargs):
        return self._clone(
            response_format=kwargs.get("response_format", self._response_format)
        )

    def _clone(self, *, tools=None, response_format=None):
        clone = object.__new__(type(self))
        clone.model = self.model
        clone.temperature = self.temperature
        clone.max_tokens = self.max_tokens
        clone.provider = self.provider
        clone._tools = self._tools if tools is None else tuple(tools)
        clone._response_format = (
            self._response_format if response_format is None else response_format
        )
        return clone

    async def ainvoke(self, messages, **kwargs):
        response = await self.provider.complete(
            _request(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                tools=kwargs.get("tools", self._tools),
                response_format=kwargs.get("response_format", self._response_format),
            )
        )
        calls = [
            {"id": c.id, "name": c.name, "args": dict(c.arguments), "type": "tool_call"}
            for c in response.message.tool_calls
        ]
        return AIMessage(
            _text(response.message),
            tool_calls=calls,
            additional_kwargs={
                "model": response.model,
                "finish_reason": response.finish_reason,
            },
        )

    def invoke(self, messages, **kwargs):
        return _run_sync(self.ainvoke(messages, **kwargs))


class ChatAnthropic(ChatOpenAI):
    def __init__(
        self,
        model,
        *,
        base_url=None,
        temperature=None,
        max_tokens=None,
        provider=None,
        **kwargs,
    ):
        credential = _credential(kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _anthropic_provider(
            model=model, credential=credential, base_url=base_url, provider=provider
        )
        self._tools = tuple(kwargs.get("tools", ()))
        self._response_format = kwargs.get("response_format")


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: str | MessageRole
    content: str | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    message: ChatMessage
    raw: Any = None
    delta: str | None = None


@dataclass
class CompletionResponse:
    text: str
    raw: Any = None
    delta: str | None = None


class LlamaIndexOpenAI:
    def __init__(
        self,
        model,
        *,
        api_base=None,
        temperature=None,
        max_tokens=None,
        provider=None,
        **kwargs,
    ):
        credential = _credential(kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _openai_provider(
            model=model, credential=credential, base_url=api_base, provider=provider
        )

    async def achat(self, messages, **kwargs):
        response = await self.provider.complete(
            _request(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                tools=kwargs.get("tools", ()),
                response_format=kwargs.get("response_format"),
            )
        )
        tool_calls = [
            SimpleNamespace(
                id=c.id,
                function=SimpleNamespace(
                    name=c.name, arguments=json.dumps(dict(c.arguments))
                ),
            )
            for c in response.message.tool_calls
        ]
        return ChatResponse(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=_text(response.message),
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=response.raw,
        )

    def chat(self, messages, **kwargs):
        return _run_sync(self.achat(messages, **kwargs))

    async def acomplete(self, prompt, **kwargs):
        response = await self.achat(
            [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
        )
        return CompletionResponse(text=response.message.content or "", raw=response.raw)

    def complete(self, prompt, **kwargs):
        return _run_sync(self.acomplete(prompt, **kwargs))


class LlamaIndexAnthropic(LlamaIndexOpenAI):
    def __init__(
        self,
        model,
        *,
        api_base=None,
        temperature=None,
        max_tokens=None,
        provider=None,
        **kwargs,
    ):
        credential = _credential(kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _anthropic_provider(
            model=model, credential=credential, base_url=api_base, provider=provider
        )


def _usage_namespace(usage, *, anthropic=False):
    if usage is None:
        return None
    if anthropic:
        return SimpleNamespace(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    return SimpleNamespace(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def _openai_tool_call(call):
    return SimpleNamespace(
        id=call.id,
        type="function",
        function=SimpleNamespace(
            name=call.name,
            arguments=json.dumps(dict(call.arguments)),
        ),
    )


def _openai_chat_response(response):
    message = SimpleNamespace(
        role="assistant",
        content=_text(response.message),
        tool_calls=[_openai_tool_call(c) for c in response.message.tool_calls],
    )
    return SimpleNamespace(
        id=None,
        object="chat.completion",
        model=response.model,
        choices=[
            SimpleNamespace(
                index=0,
                message=message,
                finish_reason=response.finish_reason,
            )
        ],
        usage=_usage_namespace(response.usage),
    )


async def _provider_stream(provider, request):
    stream = getattr(provider, "stream", None)
    if callable(stream):
        async for event in stream(request):
            yield event
        return
    response = await provider.complete(request)
    text = _text(response.message)
    if text:
        yield _rt().ModelStreamEvent(type="text_delta", text=text)
    yield _rt().ModelStreamEvent(type="completed", response=response)


async def _openai_chat_stream(provider, request):
    async for event in _provider_stream(provider, request):
        if event.type == "text_delta":
            yield SimpleNamespace(
                id=None,
                object="chat.completion.chunk",
                model=request.model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content=event.text, tool_calls=[]),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        elif event.type == "tool_call_delta":
            function = SimpleNamespace(
                name=event.tool_name,
                arguments=event.arguments_delta or "",
            )
            call = SimpleNamespace(
                index=0,
                id=event.tool_call_id,
                type="function",
                function=function,
            )
            yield SimpleNamespace(
                id=None,
                object="chat.completion.chunk",
                model=request.model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content=None, tool_calls=[call]),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        elif event.type == "completed" and event.response is not None:
            yield SimpleNamespace(
                id=None,
                object="chat.completion.chunk",
                model=event.response.model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content=None, tool_calls=[]),
                        finish_reason=event.response.finish_reason,
                    )
                ],
                usage=_usage_namespace(event.response.usage),
            )


async def _collect_async_iterator(iterator):
    return [item async for item in iterator]


class _AsyncOpenAICompletions:
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        *,
        model,
        messages,
        temperature=None,
        max_tokens=None,
        max_completion_tokens=None,
        tools=(),
        response_format=None,
        reasoning_effort=None,
        stream=False,
        **_,
    ):
        (
            compat_tools,
            local_kinds,
            mcp_bindings,
            retrieval_bindings,
        ) = await _expand_compat_tools(
            tools,
            self._client.mcp_clients,
            self._client.retrieval_registry,
            self._client.web_search_provider,
        )
        request = _request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=(
                max_completion_tokens
                if max_completion_tokens is not None
                else max_tokens
            ),
            tools=compat_tools,
            response_format=response_format,
            reasoning=_reasoning_config(reasoning_effort=reasoning_effort),
        )
        if (
            local_kinds.intersection({"code", "shell"})
            and self._client.sandbox_session is None
        ):
            raise RuntimeError(
                "code execution compatibility requires sandbox_session=SandboxSession(...)"
            )
        if stream:
            if local_kinds:
                return _openai_chat_local_stream(
                    self._client.provider,
                    request,
                    self._client.sandbox_session,
                    self._client.max_code_tool_rounds,
                    mcp_bindings,
                    retrieval_bindings,
                )
            return _openai_chat_stream(self._client.provider, request)
        response = await _complete_with_local_code_tools(
            self._client.provider,
            request,
            sandbox_session=(
                self._client.sandbox_session
                if local_kinds.intersection({"code", "shell"})
                else None
            ),
            mcp_bindings=mcp_bindings,
            retrieval_bindings=retrieval_bindings,
            max_rounds=self._client.max_code_tool_rounds,
        )
        return _openai_chat_response(response)


class _SyncOpenAICompletions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        stream = bool(kwargs.get("stream"))
        result = _run_sync(self._client._async.chat.completions.create(**kwargs))
        if stream:
            return iter(_run_sync(_collect_async_iterator(result)))
        return result


def _responses_messages(value, instructions=None):
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return messages
    for item in value or ():
        if not isinstance(item, Mapping):
            messages.append(item)
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
            continue
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id", item.get("id", "")),
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
            continue
        messages.append(item)
    return messages


def _response_format_from_text(text):
    if not isinstance(text, Mapping):
        return None
    value = text.get("format")
    if not isinstance(value, Mapping):
        return None
    if value.get("type") == "json_schema":
        return {"json_schema": value}
    return None


def _openai_response_object(
    response,
    mcp_calls=(),
    response_id=None,
    file_search_calls=(),
    web_search_calls=(),
):
    output = [
        SimpleNamespace(
            id=call["id"],
            type="mcp_call",
            approval_request_id=None,
            arguments=json.dumps(call["arguments"]),
            error=None,
            name=call["name"],
            output=_serialize_tool_result(call["output"]),
            server_label=call["server_label"],
        )
        for call in mcp_calls
    ]
    output.extend(
        SimpleNamespace(
            id=call["id"],
            type="file_search_call",
            status="completed",
            queries=[call["query"]],
            results=[
                SimpleNamespace(
                    file_id=result["file_id"],
                    filename=result["filename"],
                    score=result["score"],
                    text=result["text"],
                    attributes=result["attributes"],
                )
                for result in call["results"]
            ],
        )
        for call in file_search_calls
    )
    output.extend(
        SimpleNamespace(
            id=call["id"],
            type="web_search_call",
            status="completed",
            action=SimpleNamespace(
                type="search",
                query=call["query"],
                sources=[
                    SimpleNamespace(
                        type="url",
                        url=result["url"],
                        title=result["title"],
                    )
                    for result in call["results"]
                    if result["url"]
                ],
            ),
            results=[
                SimpleNamespace(
                    title=result["title"],
                    url=result["url"],
                    text=result["text"],
                    score=result["score"],
                )
                for result in call["results"]
            ],
        )
        for call in web_search_calls
    )
    text = _text(response.message)
    if text:
        output.append(
            SimpleNamespace(
                id=None,
                type="message",
                role="assistant",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
            )
        )
    output.extend(
        SimpleNamespace(
            id=call.id,
            type="function_call",
            call_id=call.id,
            name=call.name,
            arguments=json.dumps(dict(call.arguments)),
            status="completed",
        )
        for call in response.message.tool_calls
    )
    usage = response.usage
    return SimpleNamespace(
        id=response_id,
        object="response",
        status="completed",
        model=response.model,
        output=output,
        output_text=text,
        usage=(
            None
            if usage is None
            else SimpleNamespace(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        ),
        error=None,
    )


def _response_history(client, previous_response_id):
    if previous_response_id is None:
        return ()
    state = client._response_states.get(previous_response_id)
    if state is None:
        raise ValueError(f"unknown previous_response_id {previous_response_id!r}")
    return state


def _next_response_id(client):
    client._response_sequence += 1
    return f"resp_agent_rt_{client._response_sequence}"


def _store_response_state(client, response_id, messages):
    client._response_states[response_id] = tuple(messages)
    while len(client._response_states) > client.max_response_states:
        client._response_states.pop(next(iter(client._response_states)))


async def _openai_responses_stream(provider, request, *, client=None, response_id=None):
    async for event in _provider_stream(provider, request):
        if event.type == "text_delta":
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta=event.text or "",
            )
        elif event.type == "tool_call_delta":
            yield SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id=event.tool_call_id,
                name=event.tool_name,
                delta=event.arguments_delta or "",
            )
        elif event.type == "completed" and event.response is not None:
            if client is not None and response_id is not None:
                _store_response_state(
                    client,
                    response_id,
                    tuple(request.messages) + (event.response.message,),
                )
            yield SimpleNamespace(
                type="response.completed",
                response=_openai_response_object(
                    event.response, response_id=response_id
                ),
            )


class _AsyncOpenAIResponses:
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        *,
        model,
        input,
        instructions=None,
        temperature=None,
        max_output_tokens=None,
        tools=(),
        text=None,
        reasoning=None,
        previous_response_id=None,
        stream=False,
        **_,
    ):
        (
            compat_tools,
            local_kinds,
            mcp_bindings,
            retrieval_bindings,
        ) = await _expand_compat_tools(
            tools,
            self._client.mcp_clients,
            self._client.retrieval_registry,
            self._client.web_search_provider,
        )
        prior_messages = _response_history(self._client, previous_response_id)
        current_messages = _messages(_responses_messages(input, instructions))
        request = _request(
            model=model,
            messages=tuple(prior_messages) + tuple(current_messages),
            temperature=temperature,
            max_tokens=max_output_tokens,
            tools=compat_tools,
            response_format=_response_format_from_text(text),
            reasoning=_reasoning_config(reasoning=reasoning),
        )
        if (
            local_kinds.intersection({"code", "shell"})
            and self._client.sandbox_session is None
        ):
            raise RuntimeError(
                "code execution compatibility requires sandbox_session=SandboxSession(...)"
            )
        response_id = _next_response_id(self._client)
        if stream:
            if local_kinds:
                return _openai_responses_local_stream(
                    self._client.provider,
                    request,
                    self._client.sandbox_session,
                    self._client.max_code_tool_rounds,
                    mcp_bindings,
                    retrieval_bindings,
                    client=self._client,
                    response_id=response_id,
                )
            return _openai_responses_stream(
                self._client.provider,
                request,
                client=self._client,
                response_id=response_id,
            )
        mcp_calls = []
        file_search_calls = []
        web_search_calls = []
        final_history = []
        response = await _complete_with_local_code_tools(
            self._client.provider,
            request,
            sandbox_session=(
                self._client.sandbox_session
                if local_kinds.intersection({"code", "shell"})
                else None
            ),
            mcp_bindings=mcp_bindings,
            mcp_execution_log=mcp_calls,
            retrieval_bindings=retrieval_bindings,
            file_search_execution_log=file_search_calls,
            web_search_execution_log=web_search_calls,
            final_history_out=final_history,
            max_rounds=self._client.max_code_tool_rounds,
        )
        _store_response_state(self._client, response_id, final_history)
        return _openai_response_object(
            response,
            mcp_calls,
            response_id=response_id,
            file_search_calls=file_search_calls,
            web_search_calls=web_search_calls,
        )


class _SyncOpenAIResponses:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        stream = bool(kwargs.get("stream"))
        result = _run_sync(self._client._async.responses.create(**kwargs))
        if stream:
            return iter(_run_sync(_collect_async_iterator(result)))
        return result


def _openai_embedding_response(response):
    return SimpleNamespace(
        object="list",
        model=response.model,
        data=[
            SimpleNamespace(
                object="embedding",
                index=item.index,
                embedding=item.embedding,
            )
            for item in response.data
        ],
        usage=(
            None
            if response.usage is None
            else SimpleNamespace(
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.total_tokens,
            )
        ),
    )


class _AsyncOpenAIEmbeddings:
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        *,
        model,
        input,
        encoding_format=None,
        dimensions=None,
        **_,
    ):
        embed = getattr(self._client.provider, "embed", None)
        if not callable(embed):
            raise TypeError("configured Agent RT provider does not support embeddings")
        response = await embed(
            _rt().EmbeddingRequest(
                model=model,
                input=input,
                encoding_format=encoding_format,
                dimensions=dimensions,
            )
        )
        return _openai_embedding_response(response)


def _openai_batch_shape(job):
    if getattr(job, "raw", None) is not None:
        return job.raw
    return SimpleNamespace(id=job.id, object="batch", status=job.status)


def _anthropic_batch_shape(job):
    if getattr(job, "raw", None) is not None:
        return job.raw
    return SimpleNamespace(
        id=job.id, type="message_batch", processing_status=job.status
    )


class _AsyncOpenAIBatches:
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        *,
        input_file_id,
        endpoint,
        completion_window="24h",
        metadata=None,
        output_expires_after=None,
        **_,
    ):
        create = getattr(self._client.provider, "create_batch", None)
        if not callable(create):
            raise TypeError(
                "configured Agent RT provider does not support batch creation"
            )
        return _openai_batch_shape(
            await create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata=metadata,
                output_expires_after=output_expires_after,
            )
        )

    async def retrieve(self, batch_id, **_):
        retrieve = getattr(self._client.provider, "retrieve_batch", None)
        if not callable(retrieve):
            raise TypeError(
                "configured Agent RT provider does not support batch retrieval"
            )
        return _openai_batch_shape(await retrieve(batch_id))

    async def list(self, *, after=None, limit=None, **_):
        list_batches = getattr(self._client.provider, "list_batches", None)
        if not callable(list_batches):
            raise TypeError(
                "configured Agent RT provider does not support batch listing"
            )
        data = [
            _openai_batch_shape(job)
            for job in await list_batches(after=after, limit=limit)
        ]
        return SimpleNamespace(
            object="list",
            data=data,
            has_more=False,
            first_id=(data[0].id if data else None),
            last_id=(data[-1].id if data else None),
        )

    async def cancel(self, batch_id, **_):
        cancel = getattr(self._client.provider, "cancel_batch", None)
        if not callable(cancel):
            raise TypeError(
                "configured Agent RT provider does not support batch cancellation"
            )
        return _openai_batch_shape(await cancel(batch_id))


class _SyncOpenAIBatches:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        return _run_sync(self._client._async.batches.create(**kwargs))

    def retrieve(self, batch_id, **kwargs):
        return _run_sync(self._client._async.batches.retrieve(batch_id, **kwargs))

    def list(self, **kwargs):
        return _run_sync(self._client._async.batches.list(**kwargs))

    def cancel(self, batch_id, **kwargs):
        return _run_sync(self._client._async.batches.cancel(batch_id, **kwargs))


class _AsyncAnthropicBatchResults:
    def __init__(self, values):
        self._values = tuple(values)

    def __aiter__(self):
        async def iterator():
            for value in self._values:
                yield value

        return iterator()


class _SyncAnthropicBatchResults:
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self):
        return iter(self._values)


class _AsyncAnthropicMessageBatches:
    def __init__(self, client):
        self._client = client

    async def create(self, *, requests, user_profile_id=None, **_):
        create = getattr(self._client.provider, "create_batch", None)
        if not callable(create):
            raise TypeError(
                "configured Agent RT provider does not support batch creation"
            )
        return _anthropic_batch_shape(
            await create(requests=requests, user_profile_id=user_profile_id)
        )

    async def retrieve(self, message_batch_id, **_):
        retrieve = getattr(self._client.provider, "retrieve_batch", None)
        if not callable(retrieve):
            raise TypeError(
                "configured Agent RT provider does not support batch retrieval"
            )
        return _anthropic_batch_shape(await retrieve(message_batch_id))

    async def list(self, *, after_id=None, before_id=None, limit=None, **_):
        list_batches = getattr(self._client.provider, "list_batches", None)
        if not callable(list_batches):
            raise TypeError(
                "configured Agent RT provider does not support batch listing"
            )
        data = [
            _anthropic_batch_shape(job)
            for job in await list_batches(
                after_id=after_id, before_id=before_id, limit=limit
            )
        ]
        return SimpleNamespace(
            data=data,
            has_more=False,
            first_id=(data[0].id if data else None),
            last_id=(data[-1].id if data else None),
        )

    async def cancel(self, message_batch_id, **_):
        cancel = getattr(self._client.provider, "cancel_batch", None)
        if not callable(cancel):
            raise TypeError(
                "configured Agent RT provider does not support batch cancellation"
            )
        return _anthropic_batch_shape(await cancel(message_batch_id))

    async def results(self, message_batch_id, **_):
        results = getattr(self._client.provider, "batch_results", None)
        if not callable(results):
            raise TypeError(
                "configured Agent RT provider does not support batch results"
            )
        return _AsyncAnthropicBatchResults(await results(message_batch_id))


class _SyncAnthropicMessageBatches:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        return _run_sync(self._client._async.messages.batches.create(**kwargs))

    def retrieve(self, message_batch_id, **kwargs):
        return _run_sync(
            self._client._async.messages.batches.retrieve(message_batch_id, **kwargs)
        )

    def list(self, **kwargs):
        return _run_sync(self._client._async.messages.batches.list(**kwargs))

    def cancel(self, message_batch_id, **kwargs):
        return _run_sync(
            self._client._async.messages.batches.cancel(message_batch_id, **kwargs)
        )

    def results(self, message_batch_id, **kwargs):
        async_results = _run_sync(
            self._client._async.messages.batches.results(message_batch_id, **kwargs)
        )
        values = _run_sync(_collect_async_iterator(async_results))
        return _SyncAnthropicBatchResults(values)


def _openai_model_shape(entry):
    created = None
    if entry.created_at is not None:
        try:
            created = int(entry.created_at.timestamp())
        except (AttributeError, TypeError, ValueError, OverflowError):
            created = None
    return SimpleNamespace(
        id=entry.id,
        object=entry.object or "model",
        created=created,
        owned_by=entry.owned_by,
    )


def _anthropic_model_shape(entry):
    return SimpleNamespace(
        id=entry.id,
        type="model",
        display_name=entry.display_name or entry.id,
        created_at=entry.created_at,
    )


class _AsyncOpenAIModels:
    def __init__(self, client):
        self._client = client

    async def list(self, **_):
        list_models = getattr(self._client.provider, "list_models", None)
        if not callable(list_models):
            raise TypeError(
                "configured Agent RT provider does not support model listing"
            )
        data = [_openai_model_shape(entry) for entry in await list_models()]
        return SimpleNamespace(object="list", data=data)

    async def retrieve(self, model, **_):
        retrieve = getattr(self._client.provider, "retrieve_model", None)
        if not callable(retrieve):
            raise TypeError(
                "configured Agent RT provider does not support model retrieval"
            )
        return _openai_model_shape(await retrieve(model))


class _SyncOpenAIModels:
    def __init__(self, client):
        self._client = client

    def list(self, **kwargs):
        return _run_sync(self._client._async.models.list(**kwargs))

    def retrieve(self, model, **kwargs):
        return _run_sync(self._client._async.models.retrieve(model, **kwargs))


class _AsyncAnthropicModels:
    def __init__(self, client):
        self._client = client

    async def list(self, **_):
        list_models = getattr(self._client.provider, "list_models", None)
        if not callable(list_models):
            raise TypeError(
                "configured Agent RT provider does not support model listing"
            )
        data = [_anthropic_model_shape(entry) for entry in await list_models()]
        return SimpleNamespace(
            data=data,
            has_more=False,
            first_id=(data[0].id if data else None),
            last_id=(data[-1].id if data else None),
        )

    async def retrieve(self, model_id, **_):
        retrieve = getattr(self._client.provider, "retrieve_model", None)
        if not callable(retrieve):
            raise TypeError(
                "configured Agent RT provider does not support model retrieval"
            )
        return _anthropic_model_shape(await retrieve(model_id))


class _SyncAnthropicModels:
    def __init__(self, client):
        self._client = client

    def list(self, **kwargs):
        return _run_sync(self._client._async.models.list(**kwargs))

    def retrieve(self, model_id, **kwargs):
        return _run_sync(self._client._async.models.retrieve(model_id, **kwargs))


class _SyncOpenAIEmbeddings:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        return _run_sync(self._client._async.embeddings.create(**kwargs))


class AsyncOpenAI:
    def __init__(
        self,
        *,
        base_url=None,
        provider=None,
        sandbox_session=None,
        mcp_clients=None,
        retrieval_registry=None,
        web_search_provider=None,
        max_code_tool_rounds=8,
        max_response_states=128,
        **kwargs,
    ):
        credential = _credential(kwargs)
        self.sandbox_session = sandbox_session
        self.mcp_clients = dict(mcp_clients or {})
        self.retrieval_registry = retrieval_registry
        self.web_search_provider = web_search_provider
        self.max_code_tool_rounds = max_code_tool_rounds
        self.max_response_states = max(1, int(max_response_states))
        self._response_states = {}
        self._response_sequence = 0
        self.provider = _openai_provider(
            model=None,
            credential=credential,
            base_url=base_url,
            provider=provider,
        )
        self.chat = SimpleNamespace(completions=_AsyncOpenAICompletions(self))
        self.responses = _AsyncOpenAIResponses(self)
        self.embeddings = _AsyncOpenAIEmbeddings(self)
        self.models = _AsyncOpenAIModels(self)
        self.batches = _AsyncOpenAIBatches(self)


class OpenAI:
    def __init__(self, *, base_url=None, provider=None, **kwargs):
        self._async = AsyncOpenAI(base_url=base_url, provider=provider, **kwargs)
        self.provider = self._async.provider
        self.chat = SimpleNamespace(completions=_SyncOpenAICompletions(self))
        self.responses = _SyncOpenAIResponses(self)
        self.embeddings = _SyncOpenAIEmbeddings(self)
        self.models = _SyncOpenAIModels(self)
        self.batches = _SyncOpenAIBatches(self)


def _anthropic_response_shape(response, mcp_calls=(), web_search_calls=()):
    text = _text(response.message)
    blocks = []
    for call in mcp_calls:
        blocks.append(
            SimpleNamespace(
                type="mcp_tool_use",
                id=call["id"],
                name=call["name"],
                server_name=call["server_label"],
                input=dict(call["arguments"]),
            )
        )
        blocks.append(
            SimpleNamespace(
                type="mcp_tool_result",
                tool_use_id=call["id"],
                is_error=False,
                content=[
                    SimpleNamespace(
                        type="text", text=_serialize_tool_result(call["output"])
                    )
                ],
            )
        )
    for call in web_search_calls:
        blocks.append(
            SimpleNamespace(
                type="server_tool_use",
                id=call["id"],
                name="web_search",
                input={"query": call["query"]},
            )
        )
        blocks.append(
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id=call["id"],
                content=[
                    SimpleNamespace(
                        type="web_search_result",
                        title=result["title"],
                        url=result["url"],
                        encrypted_content=None,
                        page_age=None,
                    )
                    for result in call["results"]
                    if result["url"]
                ],
            )
        )
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    blocks.extend(
        SimpleNamespace(
            type="tool_use",
            id=c.id,
            name=c.name,
            input=dict(c.arguments),
        )
        for c in response.message.tool_calls
    )
    return SimpleNamespace(
        id=None,
        type="message",
        role="assistant",
        model=response.model,
        content=blocks,
        stop_reason=response.finish_reason,
        usage=_usage_namespace(response.usage, anthropic=True),
    )


def _anthropic_messages(messages, system=None):
    rt = _rt()
    result = []
    if system is not None:
        result.append(_model_message("system", _content(system)))
    for message in messages:
        if not isinstance(message, Mapping):
            result.append(message)
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, Sequence) or isinstance(
            content, (str, bytes, bytearray)
        ):
            result.append(_model_message(role, content))
            continue
        text_parts = []
        tool_calls = []
        tool_results = []
        for block in content:
            if not isinstance(block, Mapping):
                text_parts.append(_content(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(_content(block.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    rt.ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=(
                            dict(block.get("input", {}))
                            if isinstance(block.get("input", {}), Mapping)
                            else {}
                        ),
                    )
                )
            elif block_type == "tool_result":
                tool_results.append(
                    _model_message(
                        "tool",
                        block.get("content", ""),
                        tool_call_id=str(block.get("tool_use_id", "")),
                    )
                )
        if text_parts or tool_calls:
            result.append(
                rt.ModelMessage(
                    role=_role(role),
                    content=(
                        (rt.ContentPart(type="text", text="".join(text_parts)),)
                        if text_parts
                        else ()
                    ),
                    tool_calls=tuple(tool_calls),
                )
            )
        result.extend(tool_results)
    return result


def _anthropic_request(
    *,
    model,
    messages,
    max_tokens,
    system=None,
    temperature=None,
    tools=(),
    output_config=None,
    output_format=None,
    thinking=None,
):
    structured = output_config if output_config is not None else output_format
    return _request(
        model=model,
        messages=_anthropic_messages(messages, system),
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        response_format=structured,
        reasoning=_reasoning_config(thinking=thinking, output_config=output_config),
    )


async def _anthropic_event_stream(provider, request):
    started = False
    async for event in _provider_stream(provider, request):
        if not started:
            started = True
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    id=None,
                    type="message",
                    role="assistant",
                    model=request.model,
                    content=[],
                    usage=None,
                ),
            )
        if event.type == "text_delta":
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=event.text or ""),
            )
        elif event.type == "tool_call_delta":
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta",
                    partial_json=event.arguments_delta or "",
                ),
                tool_use_id=event.tool_call_id,
                tool_name=event.tool_name,
            )
        elif event.type == "completed" and event.response is not None:
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason=event.response.finish_reason),
                usage=_usage_namespace(event.response.usage, anthropic=True),
            )
            yield SimpleNamespace(type="message_stop")


class _AsyncAnthropicMessageStream:
    def __init__(self, provider, request):
        self._provider = provider
        self._request = request
        self._iterator = None
        self._events = []
        self._final_message = None

    def __aiter__(self):
        if self._iterator is None:
            self._iterator = self._iterate()
        return self._iterator

    async def _iterate(self):
        started = False
        tool_started = False
        async for event in _provider_stream(self._provider, self._request):
            self._events.append(event)
            if not started:
                started = True
                yield SimpleNamespace(
                    type="message_start",
                    message=SimpleNamespace(
                        id=None,
                        type="message",
                        role="assistant",
                        model=self._request.model,
                        content=[],
                        usage=None,
                    ),
                )
            if event.type == "text_delta":
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=SimpleNamespace(type="text_delta", text=event.text or ""),
                )
            elif event.type == "tool_call_delta":
                if not tool_started:
                    tool_started = True
                    yield SimpleNamespace(
                        type="content_block_start",
                        index=0,
                        content_block=SimpleNamespace(
                            type="tool_use",
                            id=event.tool_call_id,
                            name=event.tool_name,
                            input={},
                        ),
                    )
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=SimpleNamespace(
                        type="input_json_delta",
                        partial_json=event.arguments_delta or "",
                    ),
                )
            elif event.type == "completed" and event.response is not None:
                self._final_message = _anthropic_response_shape(event.response)
                yield SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason=event.response.finish_reason),
                    usage=_usage_namespace(event.response.usage, anthropic=True),
                )
                yield SimpleNamespace(type="message_stop")

    @property
    def text_stream(self):
        async def text_iterator():
            async for event in self:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text

        return text_iterator()

    async def get_final_message(self):
        if self._final_message is None:
            async for _ in self:
                pass
        return self._final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncAnthropicLocalMessageStream:
    def __init__(
        self,
        provider,
        request,
        session,
        max_rounds,
        mcp_bindings=None,
        retrieval_bindings=None,
    ):
        self._provider = provider
        self._request = request
        self._session = session
        self._max_rounds = max_rounds
        self._mcp_bindings = dict(mcp_bindings or {})
        self._retrieval_bindings = dict(retrieval_bindings or {})
        self._iterator = None
        self._text_parts = []
        self._final_message = None
        self._model = request.model
        self._usage = None
        self._stop_reason = None

    def __aiter__(self):
        if self._iterator is None:
            self._iterator = self._iterate()
        return self._iterator

    async def _iterate(self):
        async for event in _anthropic_local_event_stream(
            self._provider,
            self._request,
            self._session,
            self._max_rounds,
            self._mcp_bindings,
            self._retrieval_bindings,
        ):
            if event.type == "message_start":
                self._model = event.message.model
            elif (
                event.type == "content_block_delta" and event.delta.type == "text_delta"
            ):
                self._text_parts.append(event.delta.text)
            elif event.type == "message_delta":
                self._stop_reason = event.delta.stop_reason
                self._usage = event.usage
            elif event.type == "message_stop":
                text = "".join(self._text_parts)
                self._final_message = SimpleNamespace(
                    id=None,
                    type="message",
                    role="assistant",
                    model=self._model,
                    content=[SimpleNamespace(type="text", text=text)] if text else [],
                    stop_reason=self._stop_reason,
                    usage=self._usage,
                )
            yield event

    @property
    def text_stream(self):
        async def text_iterator():
            async for event in self:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text

        return text_iterator()

    async def get_final_message(self):
        if self._final_message is None:
            async for _ in self:
                pass
        return self._final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncAnthropicDeferredMessageStream:
    def __init__(self, client, kwargs):
        self._client = client
        self._kwargs = dict(kwargs)
        self._delegate = None

    async def _prepare(self):
        if self._delegate is not None:
            return self._delegate
        kwargs = dict(self._kwargs)
        tools = kwargs.get("tools", ())
        mcp_servers = kwargs.pop("mcp_servers", ())
        kwargs.pop("betas", None)
        _validate_anthropic_mcp_servers(mcp_servers, tools)
        (
            compat_tools,
            local_kinds,
            mcp_bindings,
            retrieval_bindings,
        ) = await _expand_compat_tools(
            tools, self._client.mcp_clients, None, self._client.web_search_provider
        )
        kwargs["tools"] = compat_tools
        request = _anthropic_request(**kwargs)
        code_kinds = local_kinds.intersection({"code", "shell"})
        if code_kinds and self._client.sandbox_session is None:
            raise RuntimeError(
                "code execution compatibility requires sandbox_session=SandboxSession(...)"
            )
        if local_kinds:
            self._delegate = _AsyncAnthropicLocalMessageStream(
                self._client.provider,
                request,
                self._client.sandbox_session if code_kinds else None,
                self._client.max_code_tool_rounds,
                mcp_bindings,
                retrieval_bindings,
            )
        else:
            self._delegate = _AsyncAnthropicMessageStream(
                self._client.provider, request
            )
        return self._delegate

    def __aiter__(self):
        async def iterator():
            delegate = await self._prepare()
            async for event in delegate:
                yield event

        return iterator()

    @property
    def text_stream(self):
        async def iterator():
            delegate = await self._prepare()
            async for text in delegate.text_stream:
                yield text

        return iterator()

    async def get_final_message(self):
        delegate = await self._prepare()
        return await delegate.get_final_message()

    async def __aenter__(self):
        await self._prepare()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SyncAnthropicMessageStream:
    def __init__(self, async_stream):
        self._async_stream = async_stream
        self._events = None

    def _materialize(self):
        if self._events is None:
            self._events = _run_sync(_collect_async_iterator(self._async_stream))
        return self._events

    def __iter__(self):
        return iter(self._materialize())

    @property
    def text_stream(self):
        return (
            event.delta.text
            for event in self._materialize()
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        )

    def get_final_message(self):
        self._materialize()
        getter = getattr(self._async_stream, "get_final_message", None)
        if callable(getter):
            return _run_sync(getter())
        return self._async_stream._final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _AsyncAnthropicMessages:
    def __init__(self, client):
        self._client = client
        self.batches = _AsyncAnthropicMessageBatches(client)

    async def create(
        self,
        *,
        model,
        messages,
        max_tokens,
        system=None,
        temperature=None,
        tools=(),
        mcp_servers=(),
        output_config=None,
        output_format=None,
        thinking=None,
        stream=False,
        **_,
    ):
        _validate_anthropic_mcp_servers(mcp_servers, tools)
        (
            compat_tools,
            local_kinds,
            mcp_bindings,
            retrieval_bindings,
        ) = await _expand_compat_tools(
            tools, self._client.mcp_clients, None, self._client.web_search_provider
        )
        request = _anthropic_request(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            tools=compat_tools,
            output_config=output_config,
            output_format=output_format,
            thinking=thinking,
        )
        if (
            local_kinds.intersection({"code", "shell"})
            and self._client.sandbox_session is None
        ):
            raise RuntimeError(
                "code execution compatibility requires sandbox_session=SandboxSession(...)"
            )
        if stream:
            if local_kinds:
                return _anthropic_local_event_stream(
                    self._client.provider,
                    request,
                    self._client.sandbox_session,
                    self._client.max_code_tool_rounds,
                    mcp_bindings,
                    retrieval_bindings,
                )
            return _anthropic_event_stream(self._client.provider, request)
        mcp_calls = []
        web_search_calls = []
        response = await _complete_with_local_code_tools(
            self._client.provider,
            request,
            sandbox_session=(
                self._client.sandbox_session
                if local_kinds.intersection({"code", "shell"})
                else None
            ),
            mcp_bindings=mcp_bindings,
            mcp_execution_log=mcp_calls,
            retrieval_bindings=retrieval_bindings,
            web_search_execution_log=web_search_calls,
            max_rounds=self._client.max_code_tool_rounds,
        )
        return _anthropic_response_shape(response, mcp_calls, web_search_calls)

    async def count_tokens(
        self,
        *,
        model,
        messages,
        system=None,
        tools=(),
        **_,
    ):
        count_tokens = getattr(self._client.provider, "count_tokens", None)
        if not callable(count_tokens):
            raise TypeError(
                "configured Agent RT provider does not support token counting"
            )
        request = _anthropic_request(
            model=model,
            messages=messages,
            max_tokens=1,
            system=system,
            tools=tools,
        )
        value = await count_tokens(request)
        return SimpleNamespace(input_tokens=value)

    def stream(self, **kwargs):
        return _AsyncAnthropicDeferredMessageStream(self._client, kwargs)


class _SyncAnthropicMessages:
    def __init__(self, client):
        self._client = client
        self.batches = _SyncAnthropicMessageBatches(client)

    def create(self, **kwargs):
        stream = bool(kwargs.get("stream"))
        result = _run_sync(self._client._async.messages.create(**kwargs))
        if stream:
            return iter(_run_sync(_collect_async_iterator(result)))
        return result

    def count_tokens(self, **kwargs):
        return _run_sync(self._client._async.messages.count_tokens(**kwargs))

    def stream(self, **kwargs):
        return _SyncAnthropicMessageStream(
            self._client._async.messages.stream(**kwargs)
        )


class AsyncAnthropic:
    def __init__(
        self,
        *,
        base_url=None,
        provider=None,
        sandbox_session=None,
        mcp_clients=None,
        web_search_provider=None,
        max_code_tool_rounds=8,
        **kwargs,
    ):
        credential = _credential(kwargs)
        self.sandbox_session = sandbox_session
        self.mcp_clients = dict(mcp_clients or {})
        self.web_search_provider = web_search_provider
        self.max_code_tool_rounds = max_code_tool_rounds
        self.provider = _anthropic_provider(
            model=None,
            credential=credential,
            base_url=base_url,
            provider=provider,
        )
        self.messages = _AsyncAnthropicMessages(self)
        self.models = _AsyncAnthropicModels(self)
        self.beta = SimpleNamespace(messages=self.messages)


class Anthropic:
    def __init__(self, *, base_url=None, provider=None, **kwargs):
        self._async = AsyncAnthropic(base_url=base_url, provider=provider, **kwargs)
        self.provider = self._async.provider
        self.messages = _SyncAnthropicMessages(self)
        self.models = _SyncAnthropicModels(self)
        self.beta = SimpleNamespace(messages=self.messages)


def _module(name, **exports):
    module = types.ModuleType(name)
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    return module


def install_compat_submodules(parent):
    modules = {
        "langchain": _module(
            f"{parent.__name__}.langchain",
            ChatOpenAI=ChatOpenAI,
            ChatAnthropic=ChatAnthropic,
            HumanMessage=HumanMessage,
            AIMessage=AIMessage,
            SystemMessage=SystemMessage,
            ToolMessage=ToolMessage,
        ),
        "llamaindex": _module(
            f"{parent.__name__}.llamaindex",
            OpenAI=LlamaIndexOpenAI,
            Anthropic=LlamaIndexAnthropic,
            ChatMessage=ChatMessage,
            ChatResponse=ChatResponse,
            CompletionResponse=CompletionResponse,
            MessageRole=MessageRole,
        ),
        "openai": _module(
            f"{parent.__name__}.openai", OpenAI=OpenAI, AsyncOpenAI=AsyncOpenAI
        ),
        "anthropic": _module(
            f"{parent.__name__}.anthropic",
            Anthropic=Anthropic,
            AsyncAnthropic=AsyncAnthropic,
        ),
    }
    for short, module in modules.items():
        sys.modules[module.__name__] = module
        setattr(parent, short, module)

    from ext.compat.autogen import install_autogen_compat
    from ext.compat.crewai import install_crewai_compat
    from ext.compat.langchain import install_langchain_compat
    from ext.compat.llamaindex import install_llamaindex_compat
    from ext.compat.openai_agents import install_openai_agents_compat

    install_langchain_compat(parent)
    install_llamaindex_compat(parent)
    install_openai_agents_compat(parent)
    install_autogen_compat(parent)
    install_crewai_compat(parent)

    # Exact migration aliases: preserve the upstream module path and only
    # prepend the agent_rt. prefix. Keep the historical flattened aliases above for
    # backwards compatibility.
    langchain = sys.modules[f"{parent.__name__}.langchain"]
    sys.modules[f"{parent.__name__}.langchain_openai"] = langchain
    sys.modules[f"{parent.__name__}.langchain_anthropic"] = langchain
    setattr(parent, "langchain_openai", langchain)
    setattr(parent, "langchain_anthropic", langchain)

    llamaindex = sys.modules[f"{parent.__name__}.llamaindex"]
    llama_index = _module(
        f"{parent.__name__}.llama_index",
        **{name: getattr(llamaindex, name) for name in llamaindex.__all__},
    )
    llama_index.__path__ = ()
    llama_llms = _module(f"{parent.__name__}.llama_index.llms")
    llama_llms.__path__ = ()
    sys.modules[llama_index.__name__] = llama_index
    sys.modules[llama_llms.__name__] = llama_llms
    sys.modules[f"{parent.__name__}.llama_index.llms.openai"] = llamaindex
    sys.modules[f"{parent.__name__}.llama_index.llms.anthropic"] = llamaindex
    llama_llms.openai = llamaindex
    llama_llms.anthropic = llamaindex
    llama_index.llms = llama_llms
    setattr(parent, "llama_index", llama_index)
