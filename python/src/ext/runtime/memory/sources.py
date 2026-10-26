"""Agent memory sources and memory backend adapters."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ext.runtime.context.items import ContextItem, TrustLevel
from ext.runtime.types import ContentPart, MemoryScope


@dataclass(frozen=True)
class MemorySource:
    """Options for one ``AgentConfig.memory`` entry.

    ``source`` is a string, any custom object (dataclass, pydantic model,
    mapping, ...), a memory store or retriever (Agent RT, LangChain,
    LangGraph, LlamaIndex, AutoGen, OpenAI Agents sessions, mem0), or a
    callable ``(query) -> value(s)`` that may be async. Stores, retrievers and
    callables are searched with the latest user prompt on every run and
    default to untrusted; static values default to trusted.
    ``search_kwargs`` are passed to the source's search call (for example a
    mem0 ``user_id`` or a LangGraph ``namespace``).
    """

    source: Any
    name: str | None = None
    trust: TrustLevel | None = None
    limit: int = 5
    scope: MemoryScope | None = None
    search_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("memory limit must be at least 1")
        if self.name is not None and not self.name.strip():
            raise ValueError("memory source name must not be empty")


def _memory_json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _memory_item(value: Any) -> Any:
    """Reduce records, documents, nodes and messages from any framework to
    their content plus role/tags/metadata; other values pass through."""
    if hasattr(value, "node") and hasattr(value, "score"):  # LlamaIndex NodeWithScore
        value = value.node
    if isinstance(getattr(value, "page_content", None), str):  # LangChain Document
        content = value.page_content
    elif callable(getattr(value, "get_content", None)):  # LlamaIndex node
        content = value.get_content()
    elif hasattr(value, "content") and any(
        hasattr(value, marker)
        for marker in ("kind", "role", "mime_type", "additional_kwargs")
    ):  # Agent RT record, chat message, AutoGen MemoryContent
        content = value.content
    elif all(hasattr(value, name) for name in ("key", "value", "namespace")):
        # LangGraph store Item.
        return {"key": value.key, "value": value.value}
    else:
        return value
    fields: dict[str, Any] = {}
    role = getattr(value, "role", None)
    if role is None and hasattr(value, "additional_kwargs"):
        role = getattr(value, "type", None)  # LangChain message
    if role is not None:
        fields["role"] = getattr(role, "value", role)
    for name in ("tags", "metadata"):
        extra = getattr(value, name, None)
        if extra:
            fields[name] = list(extra) if name == "tags" else dict(extra)
    if not fields and isinstance(content, str):
        return content
    if getattr(value, "kind", None) is not None and "role" not in fields:
        fields["kind"] = value.kind
    return {"content": content, **fields}


def _memory_part(value: Any) -> ContentPart:
    # Store records and documents carry embeddings and bookkeeping the model
    # does not need.
    value = _memory_item(value)
    if isinstance(value, str):
        return ContentPart(type="text", text=value)
    data = json.loads(json.dumps(value, default=_memory_json_default))
    return ContentPart(type="json", data=data)


def _has(value: Any, *names: str) -> bool:
    return all(callable(getattr(value, name, None)) for name in names)


async def _memory_call(
    value: Any, names: Sequence[str], *args: Any, **kwargs: Any
) -> Any:
    # Prefer the async twin (a*) when a framework offers one.
    method = next(
        getattr(value, name) for name in names if callable(getattr(value, name, None))
    )
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def _memory_lookup(value: Any, query: str, source: MemorySource) -> Any:
    """Search a dynamic memory source; return ``None`` for static values."""
    kwargs = dict(source.search_kwargs)
    limit = source.limit
    if isinstance(value, (str, bytes, Mapping, type)):
        return None
    if _has(value, "recall", "remember"):  # load_memory() backend
        return list(await _memory_call(value, ("recall",), query, limit))
    if _has(value, "search", "write", "read"):  # Agent RT long-term store
        from ext.runtime.optional import MemorySearchQuery

        results = value.search(
            MemorySearchQuery(text=query or None, scope=source.scope, limit=limit)
        )
        return [getattr(result, "record", result) for result in results]
    if _has(value, "search", "put", "get"):  # LangGraph BaseStore
        namespace = tuple(kwargs.pop("namespace", ()))
        return await _memory_call(
            value,
            ("asearch", "search"),
            namespace,
            query=query or None,
            limit=limit,
            **kwargs,
        )
    if _has(value, "search", "add", "get_all"):  # mem0 Memory / AsyncMemory / client
        # mem0 2.x takes top_k (scope via filters=); 1.x took limit.
        size = (
            "limit"
            if "limit" in inspect.signature(value.search).parameters
            else "top_k"
        )
        found = await _memory_call(value, ("search",), query, **{size: limit}, **kwargs)
        return found.get("results", []) if isinstance(found, Mapping) else found
    if _has(value, "query", "add", "update_context"):  # AutoGen Memory
        found = await _memory_call(value, ("query",), query, **kwargs)
        return list(getattr(found, "results", found))
    if _has(value, "get_items", "add_items"):  # OpenAI Agents Session
        return list(await _memory_call(value, ("get_items",), limit=limit))
    if _has(value, "get_all", "put"):  # LlamaIndex Memory / ChatMemoryBuffer
        found = await _memory_call(value, ("aget", "get"), input=query or None)
        return list(found)[-limit:]
    if _has(value, "add_message") and hasattr(value, "messages"):
        # LangChain chat message history.
        found = (
            await _memory_call(value, ("aget_messages",))
            if _has(value, "aget_messages")
            else value.messages
        )
        return list(found)[-limit:]
    if _has(value, "retrieve") or _has(value, "aretrieve"):  # LlamaIndex retriever
        return list(await _memory_call(value, ("aretrieve", "retrieve"), query))
    retriever = "Retriever" in type(value).__name__
    if retriever and (_has(value, "invoke") or _has(value, "ainvoke")):
        # LangChain retriever.
        return list(await _memory_call(value, ("ainvoke", "invoke"), query))
    if callable(value):
        result = value(query)
        return await result if inspect.isawaitable(result) else result
    return None


async def resolve_agent_memory(
    memory: Sequence[Any],
    query: str = "",
) -> tuple[ContextItem, ...]:
    """Turn ``AgentConfig.memory`` entries into context items for one run."""
    items: list[ContextItem] = []
    for index, entry in enumerate(memory):
        source = entry if isinstance(entry, MemorySource) else MemorySource(entry)
        name = source.name or f"memory-{index + 1}"
        found = await _memory_lookup(source.source, query, source)
        dynamic = found is not None
        value = found if dynamic else source.source
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if dynamic:
            values = values[: source.limit]
        trust = source.trust or ("untrusted" if dynamic else "trusted")
        for position, item in enumerate(values, start=1):
            if item is None:
                continue
            items.append(
                ContextItem(
                    id=f"memory:{name}:{position}",
                    kind="retrieved",
                    content=(_memory_part(item),),
                    metadata={"memory_source": name},
                    trust=trust,
                )
            )
    return tuple(items)
