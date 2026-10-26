"""Memory packages (mem0, memU, Zep, Letta, Supermemory) behind one interface.

``load_memory()`` returns a ``MemoryBackend`` that can go straight into
``AgentConfig.memory`` (it is searched with the user's prompt every run) and
whose ``remember(messages)`` stores a finished conversation. Each package is an
optional extra, imported only when its backend is loaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

DEFAULT_MEMORY_BACKEND = "mem0"

# backend -> (import name, pip extra, allowed options)
MEMORY_BACKENDS: dict[str, tuple[str, str, frozenset[str]]] = {
    "mem0": (
        "mem0",
        "mem0",
        frozenset({"api_key", "host", "config", "user_id", "agent_id", "run_id"}),
    ),
    "memu": (
        "memu",
        "memu",
        frozenset(
            {
                "user_id",
                "llm_profiles",
                "blob_config",
                "database_config",
                "memorize_config",
                "retrieve_config",
                "user_config",
            }
        ),
    ),
    "zep": ("zep_cloud", "zep", frozenset({"api_key", "base_url", "graph_id", "thread_id"})),
    "letta": ("letta_client", "letta", frozenset({"api_key", "base_url", "agent_id"})),
    "supermemory": (
        "supermemory",
        "supermemory",
        frozenset({"api_key", "base_url", "namespace"}),
    ),
}

Recall = Callable[[str, int], Awaitable[Sequence[Any]]]
Remember = Callable[[list[dict[str, str]]], Awaitable[Any]]


class MemoryBackend:
    """A memory package exposed as ``recall(query, limit)`` / ``remember(messages)``."""

    def __init__(self, name: str, client: Any, recall: Recall, remember: Remember) -> None:
        self.name = name
        self.client = client
        self._recall = recall
        self._remember = remember

    async def recall(self, query: str, limit: int = 5) -> list[Any]:
        return list(await self._recall(query, limit))[:limit]

    async def remember(self, messages: Any) -> Any:
        """Store user/assistant turns (``ModelMessage``s, role/content dicts or text)."""
        turns = _conversation(messages)
        return await self._remember(turns) if turns else None

    def __repr__(self) -> str:
        return f"MemoryBackend({self.name!r})"


def _conversation(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    turns = []
    for message in messages:
        if isinstance(message, Mapping):
            role, content = message.get("role"), message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if not isinstance(content, str) and content is not None:
                content = "".join(
                    getattr(part, "text", None) or "" for part in content
                )
        # System prompts and tool output are not conversation memory.
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append({"role": role, "content": content})
    return turns


def _transcript(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def _require(options: Mapping[str, Any], name: str, backend: str) -> Any:
    value = options.get(name)
    if not value:
        raise ValueError(f"memory backend {backend!r} requires option {name!r}")
    return value


async def _in_thread(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # The vendor SDKs are blocking; keep them off the event loop.
    return await asyncio.to_thread(function, *args, **kwargs)


def _mem0(options: Mapping[str, Any]) -> MemoryBackend:
    import mem0

    if options.get("api_key"):
        client: Any = mem0.MemoryClient(api_key=options["api_key"], host=options.get("host"))
    elif options.get("config"):
        client = mem0.Memory.from_config(dict(options["config"]))
    else:
        client = mem0.Memory()
    entity = {
        key: options[key] for key in ("user_id", "agent_id", "run_id") if options.get(key)
    } or {"user_id": "default"}

    async def recall(query: str, limit: int) -> list[Any]:
        found = await _in_thread(client.search, query, filters=entity, top_k=limit)
        results = found.get("results", []) if isinstance(found, Mapping) else found
        return [item.get("memory", item) if isinstance(item, Mapping) else item for item in results]

    async def remember(turns: list[dict[str, str]]) -> Any:
        return await _in_thread(client.add, turns, **entity)

    return MemoryBackend("mem0", client, recall, remember)


def _memu(options: Mapping[str, Any]) -> MemoryBackend:
    import memu

    client = memu.MemoryService(
        **{key: value for key, value in options.items() if key != "user_id"}
    )
    user = {"user_id": options["user_id"]} if options.get("user_id") else None

    async def recall(query: str, limit: int) -> list[Any]:
        found = await client.retrieve(
            [{"role": "user", "content": {"text": query}}], where=user
        )
        return [item.get("summary", item) for item in found.get("items", [])][:limit]

    async def remember(turns: list[dict[str, str]]) -> Any:
        # memU ingests conversations from a JSON resource, not inline messages.
        handle, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(turns, file)
            return await client.memorize(
                resource_url=path, modality="conversation", user=user
            )
        finally:
            os.unlink(path)

    return MemoryBackend("memu", client, recall, remember)


def _zep(options: Mapping[str, Any]) -> MemoryBackend:
    from zep_cloud.client import Zep
    from zep_cloud.types import AddMessage

    kwargs = {key: options[key] for key in ("api_key", "base_url") if options.get(key)}
    client = Zep(**kwargs)

    async def recall(query: str, limit: int) -> list[Any]:
        graph_id = _require(options, "graph_id", "zep")
        page = await _in_thread(client.graph.search_edges, graph_id, query=query, page_size=limit)
        return [edge.fact for edge in list(getattr(page, "items", None) or [])[:limit]]

    async def remember(turns: list[dict[str, str]]) -> Any:
        thread_id = _require(options, "thread_id", "zep")
        return await _in_thread(
            client.thread.add_messages,
            thread_id,
            messages=[AddMessage(role=turn["role"], content=turn["content"]) for turn in turns],
        )

    return MemoryBackend("zep", client, recall, remember)


def _letta(options: Mapping[str, Any]) -> MemoryBackend:
    from letta_client import Letta

    kwargs = {key: options[key] for key in ("api_key", "base_url") if options.get(key)}
    client = Letta(**kwargs)
    agent_id = _require(options, "agent_id", "letta")

    async def recall(query: str, limit: int) -> list[Any]:
        found = await _in_thread(client.agents.passages.search, agent_id, query=query, top_k=limit)
        return [result.content for result in found.results]

    async def remember(turns: list[dict[str, str]]) -> Any:
        return await _in_thread(client.agents.passages.create, agent_id, text=_transcript(turns))

    return MemoryBackend("letta", client, recall, remember)


def _supermemory(options: Mapping[str, Any]) -> MemoryBackend:
    from supermemory import Supermemory

    kwargs = {key: options[key] for key in ("api_key", "base_url") if options.get(key)}
    client = Supermemory(**kwargs)
    namespace = options.get("namespace") or "default"

    async def recall(query: str, limit: int) -> list[Any]:
        found = await _in_thread(client.search, namespace, query=query, limit=limit)
        return [result.memory or result.chunk for result in found.results]

    async def remember(turns: list[dict[str, str]]) -> Any:
        return await _in_thread(client.add, namespace, content=_transcript(turns))

    return MemoryBackend("supermemory", client, recall, remember)


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], MemoryBackend]] = {
    "mem0": _mem0,
    "memu": _memu,
    "zep": _zep,
    "letta": _letta,
    "supermemory": _supermemory,
}


def load_memory(
    config: Any = None,
    *,
    backend: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> MemoryBackend:
    """Load a memory package: ``backend``/``options`` win over the config's
    ``agent_rt_settings.memory``; the default backend is mem0."""
    from ext.config import require_resolved

    settings = getattr(config, "settings", {}).get("memory") or {}
    name = backend or settings.get("backend") or DEFAULT_MEMORY_BACKEND
    if name not in MEMORY_BACKENDS:
        raise ValueError(
            f"unknown memory backend {name!r}; expected one of: {', '.join(MEMORY_BACKENDS)}"
        )
    module, extra, allowed = MEMORY_BACKENDS[name]
    chosen = dict(settings.get("options") or {}) if options is None else dict(options)
    unknown = sorted(set(chosen) - allowed)
    if unknown:
        raise ValueError(f"memory backend {name!r}: unknown option(s): {', '.join(unknown)}")
    require_resolved(chosen)
    try:
        return _BUILDERS[name](chosen)
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] != module:
            raise
        hint = " (memU needs Python 3.13+)" if name == "memu" else ""
        raise ImportError(
            f"memory backend {name!r} is not installed{hint}: pip install 'agent-rt[{extra}]'"
        ) from exc


__all__ = ["DEFAULT_MEMORY_BACKEND", "MEMORY_BACKENDS", "MemoryBackend", "load_memory"]
