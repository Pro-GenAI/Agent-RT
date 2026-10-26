import json
from pathlib import Path

from agent_rt.autogen_agentchat.agents import AssistantAgent
from agent_rt.autogen_ext.models.openai import OpenAIChatCompletionClient
from agent_rt.langchain import AIMessage, Document, HumanMessage, InMemoryStore
from agent_rt.llama_index.core.llms import ChatMessage
from agent_rt.llama_index.core.memory import Memory
from agent_rt.llama_index.core.schema import NodeWithScore, TextNode

from agent_rt import (
    ContentPart,
    MemorySource,
    ModelMessage,
    ModelResponse,
    resolve_agent_memory,
)


def parts(items):
    return [item.content[0].text or item.content[0].data for item in items]


class FakeLangChainRetriever:
    async def ainvoke(self, query):
        return [
            Document(page_content=f"doc for {query}"),
            Document(page_content="tagged", metadata={"source": "wiki"}),
        ]


class FakeLlamaIndexRetriever:
    def retrieve(self, query):
        return [NodeWithScore(node=TextNode(text=f"node for {query}"), score=0.9)]


class FakeChatHistory:  # langchain_core BaseChatMessageHistory shape
    def __init__(self):
        self.messages = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("bye")]

    def add_message(self, message):
        self.messages.append(message)


class FakeMem0:  # mem0.Memory shape
    def __init__(self):
        self.calls = []

    def add(self, *args, **kwargs): ...

    def get_all(self, **kwargs): ...

    def search(self, query, *, top_k=20, filters=None, **kwargs):  # mem0 2.x
        self.calls.append((query, {"top_k": top_k, "filters": filters, **kwargs}))
        return {"results": [{"id": "1", "memory": "likes tea"}]}


class FakeAutoGenMemory:  # autogen_core.memory.Memory shape
    class Content:
        def __init__(self, content):
            self.content = content
            self.mime_type = "text/plain"
            self.metadata = None

    class Result:
        def __init__(self, results):
            self.results = results

    async def add(self, content): ...

    async def update_context(self, model_context): ...

    async def query(self, query, **kwargs):
        return self.Result([self.Content(f"autogen {query}")])


class FakeSession:  # agents.memory.Session shape
    async def get_items(self, limit=None):
        items = [{"role": "user", "content": "earlier"}]
        return items[-limit:] if limit else items

    async def add_items(self, items): ...


async def test_memory_sources_from_other_frameworks():
    store = InMemoryStore()
    store.put(("users", "u1"), "prefs", {"units": "metric"})
    store.put(("users", "u2"), "prefs", {"units": "imperial"})
    llama_memory = Memory.from_defaults()
    llama_memory.put(ChatMessage(role="user", content="remember blue"))
    mem0 = FakeMem0()

    items = await resolve_agent_memory(
        [
            FakeLangChainRetriever(),
            FakeLlamaIndexRetriever(),
            MemorySource(store, search_kwargs={"namespace": ("users", "u1")}),
            MemorySource(FakeChatHistory(), limit=2),
            llama_memory,
            MemorySource(mem0, search_kwargs={"filters": {"user_id": "u1"}}),
            FakeAutoGenMemory(),
            FakeSession(),
        ],
        "colors",
    )

    assert parts(items) == [
        "doc for colors",
        {"content": "tagged", "metadata": {"source": "wiki"}},
        "node for colors",
        {"namespace": ["users", "u1"], "key": "prefs", "value": {"units": "metric"}},
        {"content": "hello", "role": "assistant"},
        {"content": "bye", "role": "user"},
        {"content": "remember blue", "role": "user"},
        {"id": "1", "memory": "likes tea"},
        "autogen colors",
        {"role": "user", "content": "earlier"},
    ]
    assert {item.trust for item in items} == {"untrusted"}
    assert mem0.calls == [("colors", {"top_k": 5, "filters": {"user_id": "u1"}})]


async def test_autogen_assistant_agent_passes_memory_to_the_run():
    class Provider:
        name = "queue"

        def __init__(self):
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            return ModelResponse(
                message=ModelMessage(
                    role="assistant", content=(ContentPart(type="text", text="ok"),)
                )
            )

    provider = Provider()
    agent = AssistantAgent(
        "assistant",
        model_client=OpenAIChatCompletionClient(model="m", provider=provider),
        memory=["The user is called Ada.", FakeAutoGenMemory()],
    )
    await agent.run(task="who am I?")

    context = provider.requests[0].messages[0].content[0].data["context"]
    assert context["retrieved_data"][0]["content"][0]["text"] == (
        "The user is called Ada."
    )
    untrusted = provider.requests[0].messages[1].content[0].data
    assert untrusted["untrusted_context"]["retrieved_data"][0]["content"][0][
        "text"
    ] == ("autogen who am I?")


async def test_upstream_langchain_message_shape_uses_type_as_role():
    class UpstreamAIMessage:  # langchain_core AIMessage has `type`, not `role`
        type = "ai"
        content = "hello"
        additional_kwargs = None

    items = await resolve_agent_memory([UpstreamAIMessage()])
    assert parts(items) == [{"content": "hello", "role": "ai"}]
    assert items[0].trust == "trusted"


# --- load_memory(): memory packages behind one interface -------------------

import sys
import types

import pytest

from agent_rt import load_memory
from ext.config import ConfigError, parse_config


class Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def install(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


async def test_load_memory_defaults_to_mem0_and_scopes_by_user(monkeypatch):
    client = types.SimpleNamespace(
        search=Recorder({"results": [{"memory": "likes tea"}]}),
        add=Recorder({"results": []}),
    )
    install(monkeypatch, "mem0", Memory=lambda: client, MemoryClient=None)

    memory = load_memory(options={"user_id": "u1"})
    assert memory.name == "mem0"
    assert await memory.recall("drinks?", 3) == ["likes tea"]
    assert client.search.calls == [
        (("drinks?",), {"filters": {"user_id": "u1"}, "top_k": 3})
    ]

    await memory.remember(
        [
            {"role": "system", "content": "secret instructions"},
            ModelMessage(role="user", content=(ContentPart(type="text", text="hi"),)),
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert client.add.calls == [
        (
            (
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            ),
            {"user_id": "u1"},
        )
    ]
    # A backend is also an AgentConfig.memory source.
    items = await resolve_agent_memory([MemorySource(memory, limit=2)], "drinks?")
    assert parts(items) == ["likes tea"]
    assert items[0].trust == "untrusted"


async def test_load_memory_other_backends(monkeypatch, tmp_path):
    # memU ingests a JSON conversation file.
    seen = {}

    class MemoryService:
        def __init__(self, **options):
            seen["options"] = options

        async def retrieve(self, queries, where=None):
            seen["retrieve"] = (queries, where)
            return {"items": [{"summary": "prefers metric"}]}

        async def memorize(self, *, resource_url, modality, user=None):
            text = Path(resource_url).read_text(encoding="utf-8")
            seen["memorize"] = (json.loads(text), modality, user)

    install(monkeypatch, "memu", MemoryService=MemoryService)
    memu = load_memory(
        backend="memu",
        options={
            "user_id": "u1",
            "database_config": {"metadata_store": {"provider": "inmemory"}},
        },
    )
    assert seen["options"] == {
        "database_config": {"metadata_store": {"provider": "inmemory"}}
    }
    assert await memu.recall("units") == ["prefers metric"]
    assert seen["retrieve"] == (
        [{"role": "user", "content": {"text": "units"}}],
        {"user_id": "u1"},
    )
    await memu.remember("use metric")
    assert seen["memorize"] == (
        [{"role": "user", "content": "use metric"}],
        "conversation",
        {"user_id": "u1"},
    )

    # Zep: graph search for recall, thread messages for remember.
    zep_client = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            search_edges=Recorder(
                types.SimpleNamespace(items=[types.SimpleNamespace(fact="F1")])
            )
        ),
        thread=types.SimpleNamespace(add_messages=Recorder()),
    )
    install(monkeypatch, "zep_cloud")
    install(monkeypatch, "zep_cloud.client", Zep=lambda **kwargs: zep_client)
    install(monkeypatch, "zep_cloud.types", AddMessage=lambda **kwargs: kwargs)
    zep = load_memory(
        backend="zep", options={"api_key": "k", "graph_id": "g", "thread_id": "t"}
    )
    assert await zep.recall("q", 2) == ["F1"]
    assert zep_client.graph.search_edges.calls == [
        (("g",), {"query": "q", "page_size": 2})
    ]
    await zep.remember("hi")
    assert zep_client.thread.add_messages.calls == [
        (("t",), {"messages": [{"role": "user", "content": "hi"}]})
    ]

    # Letta archival passages.
    letta_client = types.SimpleNamespace(
        agents=types.SimpleNamespace(
            passages=types.SimpleNamespace(
                search=Recorder(
                    types.SimpleNamespace(results=[types.SimpleNamespace(content="P1")])
                ),
                create=Recorder(),
            )
        )
    )
    install(monkeypatch, "letta_client", Letta=lambda **kwargs: letta_client)
    letta = load_memory(backend="letta", options={"agent_id": "a1"})
    assert await letta.recall("q", 4) == ["P1"]
    assert letta_client.agents.passages.search.calls == [
        (("a1",), {"query": "q", "top_k": 4})
    ]
    await letta.remember(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert letta_client.agents.passages.create.calls == [
        (("a1",), {"text": "user: hi\nassistant: yo"})
    ]

    # Supermemory namespaces.
    result = types.SimpleNamespace(memory=None, chunk="C1")
    sm_client = types.SimpleNamespace(
        search=Recorder(types.SimpleNamespace(results=[result])), add=Recorder()
    )
    install(monkeypatch, "supermemory", Supermemory=lambda **kwargs: sm_client)
    sm = load_memory(backend="supermemory", options={"namespace": "u1"})
    assert await sm.recall("q", 1) == ["C1"]
    assert sm_client.search.calls == [(("u1",), {"query": "q", "limit": 1})]


def test_load_memory_errors_and_config(monkeypatch):
    with pytest.raises(ValueError, match="unknown memory backend"):
        load_memory(backend="nope")
    with pytest.raises(ValueError, match="unknown option"):
        load_memory(backend="letta", options={"agent": "a"})
    with pytest.raises(ValueError, match="requires option 'agent_id'"):
        install(monkeypatch, "letta_client", Letta=lambda **kwargs: None)
        load_memory(backend="letta", options={})
    monkeypatch.setitem(sys.modules, "zep_cloud", None)
    with pytest.raises(ImportError, match=r"pip install 'agent-rt\[zep\]'"):
        load_memory(backend="zep", options={"graph_id": "g"})

    config = parse_config(
        "agent_rt_settings:\n  memory:\n    backend: supermemory\n"
        "    options: {namespace: u1, api_key: os.environ/SM_KEY}\n",
        environ={},
    )
    install(monkeypatch, "supermemory", Supermemory=lambda **kwargs: kwargs)
    # The unset key is deferred until the backend is loaded.
    with pytest.raises(ConfigError, match="SM_KEY"):
        load_memory(config)
    with pytest.raises(ConfigError, match="memory.backend must be one of"):
        parse_config("agent_rt_settings:\n  memory: {backend: nope}\n", environ={})
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(
            "agent_rt_settings:\n  memory: {options: {graph_id: g}}\n", environ={}
        )
