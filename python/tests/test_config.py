from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

import agent_rt
from agent_rt import (
    AgentConfig,
    AgentLoop,
    AgentRTConfig,
    ConfigError,
    ConfigNotFoundError,
    ConfiguredModelProvider,
    ContentPart,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    OpenAIProviderSettings,
    RetrievalRegistry,
    RetrievalResult,
    ToolCall,
    ToolRegistry,
    configure_runtime,
    load_config,
    load_model,
    register_file_search_tool,
    resolve_vector_stores,
    runtime_settings,
    vector_store_registry_from_config,
)
from agent_rt_api import AgentRTAPIServer, build_from_config
from ext.config import config_search_paths, discover_config_path, parse_config

CONFIG = textwrap.dedent("""
    model_list:
      - model_name: claude-sonnet-5
        litellm_params:
          model: anthropic/claude-sonnet-5
          api_key: os.environ/TEST_ANTHROPIC_KEY
      - model_name: claude-sonnet-5
        agent_rt_params:
          model: openai/gpt-backup
          api_key: os.environ/TEST_OPENAI_KEY
      - model_name: embed-small
        agent_rt_params:
          model: openai/text-embedding-3-small
          api_key: os.environ/TEST_OPENAI_KEY

    vector_store_registry:
      - vector_store_name: docs-kb
        litellm_params:
          vector_store_id: T37J8R4WTM
          custom_llm_provider: qdrant
          api_base: https://qdrant.example.test
          collection: docs
          embedding_model: embed-small
          vector_store_description: Docs
          vector_store_metadata:
            source: https://example.test/docs
      - vector_store_name: private-kb
        agent_rt_params:
          vector_store_id: PRIVATE1
          custom_llm_provider: chroma
          api_base: https://chroma.example.test
          collection: private
          allowed_models: [embed-small]

    agent_rt_settings:
      default_model: claude-sonnet-5
      execution_limits:
        timeout_seconds: 7
      sandbox:
        allowed_domains: [a.example.test, b.example.test]
        allow_http: false
    """)

ENVIRONMENT = {"TEST_ANTHROPIC_KEY": "ak", "TEST_OPENAI_KEY": "ok"}


def parse(text: str = CONFIG, environ=None) -> AgentRTConfig:
    return parse_config(text, environ=ENVIRONMENT if environ is None else environ)


class TestDiscovery:
    def test_explicit_path_wins_and_must_exist(self, tmp_path):
        explicit = tmp_path / "custom.yaml"
        explicit.write_text("model_list: []\n", encoding="utf-8")
        (tmp_path / "agent-rt.yaml").write_text("model_list: []\n", encoding="utf-8")

        assert discover_config_path(explicit, cwd=tmp_path, home=tmp_path) == explicit
        with pytest.raises(ConfigError, match="not found"):
            discover_config_path(tmp_path / "nope.yaml")

    def test_current_folder_beats_home_folder(self, tmp_path):
        cwd = tmp_path / "work"
        home = tmp_path / "home"
        (home / ".agent-rt").mkdir(parents=True)
        cwd.mkdir()
        home_file = home / ".agent-rt" / "config.yaml"
        home_file.write_text("{}", encoding="utf-8")
        assert discover_config_path(cwd=cwd, home=home) == home_file

        local = cwd / "agent-rt.yaml"
        local.write_text("{}", encoding="utf-8")
        assert discover_config_path(cwd=cwd, home=home) == local

    def test_missing_default_lists_searched_locations(self, tmp_path):
        with pytest.raises(ConfigNotFoundError) as error:
            discover_config_path(cwd=tmp_path, home=tmp_path / "h")
        for candidate in config_search_paths(cwd=tmp_path, home=tmp_path / "h"):
            assert str(candidate) in str(error.value)

    def test_load_config_reads_file_and_records_source(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(CONFIG, encoding="utf-8")
        config = load_config(path, environ=ENVIRONMENT)
        assert config.source == path
        assert config.model_names == ("claude-sonnet-5", "embed-small")


class TestParsing:
    def test_secrets_resolve_from_environment_references(self):
        config = parse()
        assert config.models[0].api_key == "ak"
        assert config.models[1].api_key == "ok"
        assert "ak" not in repr(config.models[0])

    def test_missing_secret_names_the_key_but_not_a_value(self):
        config = parse(environ={})  # model and store keys fail only when used
        provider = ConfiguredModelProvider(config, provider_factory=lambda _: None)
        with pytest.raises(
            ConfigError, match=r"model_list\[0\]\.litellm_params\.api_key"
        ):
            provider._provider(config.models[0])
        with pytest.raises(ConfigError, match=r"web_search\.api_key.*NOPE"):
            parse(
                "agent_rt_settings:\n  web_search: {api_key: os.environ/NOPE}\n",
                environ={},
            )

    async def test_unset_secret_in_unused_or_failed_deployment_does_not_block(self):
        openai = RecordingProvider("openai")
        provider = load_model(
            parse(environ={"TEST_OPENAI_KEY": "ok"}),
            provider_factory=lambda deployment: openai,
        )
        response = await provider.complete(user_request("claude-sonnet-5"))
        assert response.message.content[0].text == "openai"

    def test_server_secrets_are_checked_when_serving(self):
        config = parse(
            "agent_rt_settings:\n  server: {api_keys: [os.environ/NOPE]}\n", environ={}
        )
        with pytest.raises(ConfigError, match=r"server\.api_keys\[0\].*NOPE"):
            config.server_options()

    def test_store_name_may_equal_its_id(self):
        config = parse(
            CONFIG.replace(
                "vector_store_name: docs-kb", "vector_store_name: T37J8R4WTM"
            )
        )
        assert config.vector_store("T37J8R4WTM").name == "T37J8R4WTM"

    def test_duplicate_names_become_ordered_fallbacks(self):
        config = parse()
        names = [item.provider for item in config.deployments("claude-sonnet-5")]
        assert names == ["anthropic", "openai"]

    def test_unknown_keys_and_providers_are_rejected(self):
        with pytest.raises(ConfigError, match="unknown key"):
            parse("surprise: 1\n")
        with pytest.raises(ConfigError, match="unsupported provider"):
            parse(
                "model_list:\n  - model_name: x\n    agent_rt_params: {model: bedrock/x}\n"
            )
        with pytest.raises(ConfigError, match="<provider>/<model>"):
            parse(
                "model_list:\n  - model_name: x\n    agent_rt_params: {model: plain}\n"
            )

    def test_unsafe_yaml_tags_are_rejected(self):
        with pytest.raises(ConfigError, match="invalid YAML"):
            parse("model_list: !!python/object/apply:os.system ['true']\n")

    def test_vector_store_validation(self):
        config = parse()
        store = config.vector_store("T37J8R4WTM")
        assert store is config.vector_store("docs-kb")
        assert store.metadata == {"source": "https://example.test/docs"}
        with pytest.raises(ConfigError, match="not in model_list"):
            parse(
                CONFIG.replace(
                    "embedding_model: embed-small", "embedding_model: missing"
                )
            )
        with pytest.raises(ConfigError, match="unsupported"):
            parse(
                CONFIG.replace(
                    "custom_llm_provider: qdrant", "custom_llm_provider: azure"
                )
            )
        with pytest.raises(ConfigError, match="duplicate vector store"):
            parse(CONFIG.replace("PRIVATE1", "T37J8R4WTM"))

    def test_default_model_must_exist(self):
        with pytest.raises(ConfigError, match="default_model"):
            parse(
                CONFIG.replace("default_model: claude-sonnet-5", "default_model: nope")
            )

    def test_runtime_environment_comes_from_settings_only(self):
        env = parse().runtime_environment()
        assert env["AGENT_RT_EXECUTION_TIMEOUT_SECONDS"] == "7"
        assert (
            env["AGENT_RT_SANDBOX_ALLOWED_DOMAINS"] == "a.example.test,b.example.test"
        )
        assert env["AGENT_RT_SANDBOX_ALLOW_HTTP"] == "false"

    def test_configure_runtime_feeds_implicit_defaults(self):
        previous = runtime_settings()
        try:
            configure_runtime(parse())
            assert runtime_settings()["AGENT_RT_EXECUTION_TIMEOUT_SECONDS"] == "7"
        finally:
            agent_rt._RUNTIME_SETTINGS = previous


def text_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(
            role="assistant", content=(ContentPart(type="text", text=text),)
        ),
        finish_reason="stop",
    )


class RecordingProvider:
    def __init__(self, label: str, fail: bool = False) -> None:
        self.name = label
        self.label = label
        self.fail = fail
        self.requests: list = []

    async def complete(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"{self.label} down")
        return text_response(self.label)

    async def stream(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"{self.label} down")
        yield SimpleNamespace(type="completed", response=text_response(self.label))

    async def list_models(self):
        if self.fail:
            raise RuntimeError(f"{self.label} down")
        # "claude-sonnet-5" collides with a configured model_name, which must win.
        return (
            SimpleNamespace(id=f"{self.label}-extra"),
            SimpleNamespace(id="claude-sonnet-5"),
        )

    async def embed(self, request):
        self.requests.append(request)
        return EmbeddingResponse(data=(EmbeddingItem(index=0, embedding=(0.1, 0.2)),))


def user_request(model=None) -> ModelRequest:
    return ModelRequest(
        messages=(
            ModelMessage(role="user", content=(ContentPart(type="text", text="hi"),)),
        ),
        model=model,
    )


class TestConfiguredModelProvider:
    def make(self, anthropic_down=False):
        providers = {
            "anthropic": RecordingProvider("anthropic", fail=anthropic_down),
            "openai": RecordingProvider("openai"),
        }
        provider = load_model(
            parse(), provider_factory=lambda deployment: providers[deployment.provider]
        )
        return provider, providers

    async def test_routes_by_model_name_and_rewrites_underlying_model(self):
        provider, providers = self.make()
        response = await provider.complete(user_request("claude-sonnet-5"))
        assert response.message.content[0].text == "anthropic"
        assert providers["anthropic"].requests[0].model == "claude-sonnet-5"
        assert providers["openai"].requests == []

    async def test_default_model_used_when_unspecified(self):
        provider, providers = self.make()
        await provider.complete(user_request())
        assert providers["anthropic"].requests

    async def test_falls_back_to_next_deployment(self):
        provider, providers = self.make(anthropic_down=True)
        response = await provider.complete(user_request("claude-sonnet-5"))
        assert response.message.content[0].text == "openai"
        assert providers["openai"].requests[0].model == "gpt-backup"

    async def test_stream_falls_back_before_first_event(self):
        provider, _ = self.make(anthropic_down=True)
        events = [
            event async for event in provider.stream(user_request("claude-sonnet-5"))
        ]
        assert events[-1].response.message.content[0].text == "openai"

    async def test_discovered_models_route_to_the_endpoint_that_listed_them(self):
        provider, providers = self.make(anthropic_down=True)
        # The down endpoint is skipped, and each endpoint is asked once.
        assert await provider.discover_models() == ("openai-extra",)
        assert provider.discovery_errors == [
            "claude-sonnet-5: RuntimeError: anthropic down"
        ]
        await provider.complete(user_request("openai-extra"))
        assert providers["openai"].requests[-1].model == "openai-extra"
        await provider.complete(user_request("claude-sonnet-5"))
        assert providers["anthropic"].requests[-1].model == "claude-sonnet-5"

    async def test_unknown_model_lists_configured_models(self):
        provider, _ = self.make()
        with pytest.raises(
            ValueError, match="configured models: claude-sonnet-5, embed-small"
        ):
            await provider.complete(user_request("nope"))

    async def test_embed_routes_to_the_named_embedding_model(self):
        provider, providers = self.make()
        await provider.embed(EmbeddingRequest(input="x", model="embed-small"))
        assert providers["openai"].requests[0].model == "text-embedding-3-small"

    async def test_model_listing_uses_public_names(self):
        provider, _ = self.make()
        listed = await provider.list_models()
        assert [entry.id for entry in listed] == ["claude-sonnet-5", "embed-small"]
        assert provider.settings.default_model == "claude-sonnet-5"

    def test_default_factory_builds_real_providers_lazily(self):
        provider = load_model(parse())
        assert isinstance(provider, ConfiguredModelProvider)
        first = provider._provider(provider.config.models[0])
        assert type(first).__name__ == "AnthropicModelProvider"
        second = provider._provider(provider.config.models[1])
        assert second.settings.default_model == "gpt-backup"
        assert second.settings.api_key == "ok"

    def test_empty_model_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one model"):
            load_model(parse("{}"))

    def test_openai_settings_validate_connection_checks_default_model(self):
        def factory(**_kwargs):
            return SimpleNamespace(
                models=SimpleNamespace(
                    list=lambda: SimpleNamespace(data=[SimpleNamespace(id="gpt-a")])
                )
            )

        settings = OpenAIProviderSettings(
            base_url="https://compatible.example.test/v1",
            default_model="gpt-missing",
            websocket=False,
        )
        with pytest.raises(ValueError, match="not available"):
            settings.validate_connection(
                check_default_model=True, client_factory=factory
            )


class StubStore:
    kind = "knowledge"

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score
        self.queries: list = []

    async def search(self, query):
        self.queries.append(query)
        return [
            RetrievalResult(
                id=self.label + "-1",
                title=self.label,
                content=f"passage from {self.label}",
                score=self.score,
            )
        ]


class TestVectorStores:
    def test_registry_registers_every_store_under_name_and_id(self):
        registry = vector_store_registry_from_config(parse())
        assert isinstance(registry, RetrievalRegistry)
        assert {"docs-kb", "T37J8R4WTM", "private-kb", "PRIVATE1"} <= set(
            registry._providers
        )

    def test_embedding_is_pinned_to_configured_model(self):
        registry = vector_store_registry_from_config(
            parse(), embedding_provider=RecordingProvider("emb")
        )
        assert (
            registry._providers["docs-kb"].embedding_provider._model_name
            == "embed-small"
        )

    def test_resolve_rejects_unknown_and_disallowed_stores(self):
        config = parse()
        resolved = resolve_vector_stores(config, ["docs-kb", "T37J8R4WTM"])
        assert [store.name for store in resolved] == ["docs-kb"]
        with pytest.raises(ValueError, match="unknown vector store"):
            resolve_vector_stores(config, ["missing"])
        with pytest.raises(ValueError, match="not available to this model"):
            resolve_vector_stores(config, ["PRIVATE1"], model_name="claude-sonnet-5")
        assert resolve_vector_stores(config, ["PRIVATE1"], model_name="embed-small")
        with pytest.raises(ValueError, match="at least one"):
            resolve_vector_stores(config, [])

    async def test_file_search_merges_stores_by_score(self):
        config = parse()
        registry = RetrievalRegistry()
        registry.register("docs-kb", StubStore("docs", 0.4))
        registry.register("private-kb", StubStore("private", 0.9))
        tools = ToolRegistry()
        register_file_search_tool(
            tools, registry, tuple(config.vector_stores), max_results=2
        )

        output = await tools.execute(
            ToolCall(id="1", name="file_search", arguments={"query": "what?"})
        )
        rows = output
        assert [row["vector_store"] for row in rows] == ["private-kb", "docs-kb"]


class ToolCallingProvider:
    """First turn calls file_search; second turn answers from the tool result."""

    name = "scripted"

    def __init__(self) -> None:
        self.requests: list = []

    async def complete(self, request):
        self.requests.append(request)
        if not any(message.role == "tool" for message in request.messages):
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id="c1", name="file_search", arguments={"query": "litellm"}
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )
        return text_response("answered")


class TestApiFileSearch:
    def make_server(self, provider=None):
        registry = RetrievalRegistry()
        docs = StubStore("docs", 0.5)
        private = StubStore("private", 0.5)
        for name in ("docs-kb", "T37J8R4WTM"):
            registry.register(name, docs)
        for name in ("private-kb", "PRIVATE1"):
            registry.register(name, private)
        agents = {
            name: AgentConfig(
                name=name, instructions="x", model=ModelSettings(model=name)
            )
            for name in ("claude-sonnet-5", "embed-small")
        }
        loop = AgentLoop(provider or ToolCallingProvider())
        return (
            AgentRTAPIServer(loop, agents, config=parse(), retrieval=registry),
            private,
        )

    def test_non_file_search_tools_are_rejected(self):
        server, _ = self.make_server()
        with pytest.raises(ValueError, match="except file_search"):
            server.file_search_request(
                [{"type": "function", "function": {}}], "claude-sonnet-5"
            )
        with pytest.raises(ValueError, match="list of strings"):
            server.file_search_request([{"type": "file_search"}], "claude-sonnet-5")

    def test_unknown_store_and_disallowed_model_are_rejected(self):
        server, _ = self.make_server()
        with pytest.raises(ValueError, match="unknown vector store"):
            server.file_search_request(
                [{"type": "file_search", "vector_store_ids": ["evil"]}],
                "claude-sonnet-5",
            )
        with pytest.raises(ValueError, match="not available to this model"):
            server.file_search_request(
                [{"type": "file_search", "vector_store_ids": ["PRIVATE1"]}],
                "claude-sonnet-5",
            )

    def test_unconfigured_server_rejects_file_search(self):
        agent = AgentConfig(name="a", instructions="x", model=ModelSettings(model="a"))
        server = AgentRTAPIServer(AgentLoop(ToolCallingProvider()), {"a": agent})
        with pytest.raises(ValueError, match="not configured"):
            server.file_search_request(
                [{"type": "file_search", "vector_store_ids": ["x"]}], "a"
            )

    async def test_run_with_file_search_uses_request_scoped_tool(self):
        provider = ToolCallingProvider()
        server, private = self.make_server(provider)
        docs = server.retrieval._providers["docs-kb"]
        message = ModelMessage(
            role="user", content=(ContentPart(type="text", text="What is litellm?"),)
        )

        model_id, result = await server.run(
            model="claude-sonnet-5",
            messages=(message,),
            tools=[{"type": "file_search", "vector_store_ids": ["T37J8R4WTM"]}],
        )
        assert model_id == "claude-sonnet-5"
        assert result.final_response.message.content[0].text == "answered"
        assert [tool.name for tool in provider.requests[0].tools] == ["file_search"]
        assert [q.text for q in docs.queries] == ["litellm"]
        assert private.queries == []
        tool_message = provider.requests[1].messages[-1]
        assert "passage from docs" in str(tool_message.content[0].data)
        assert server.loop.tool_registry is None

    def test_build_from_config_creates_agent_per_model(self, monkeypatch):
        # build_from_config calls configure_runtime; restore the global afterwards.
        monkeypatch.setattr(agent_rt, "_RUNTIME_SETTINGS", runtime_settings())
        built = build_from_config(
            parse(), provider_factory=lambda _d: RecordingProvider("x")
        )
        assert set(built["agents"]) == {"claude-sonnet-5", "embed-small"}
        assert built["default_agent"] == "claude-sonnet-5"
        assert built["retrieval"] is not None


async def test_openai_agents_compat_reports_an_invalid_config(tmp_path, monkeypatch):
    from agent_rt.agents import Agent, Runner

    (tmp_path / "agent-rt.yaml").write_text("surprise: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConfigError, match="unknown key"):
        await Runner.run(Agent(name="a"), "hi")


async def test_openai_agents_compat_falls_back_for_an_unconfigured_model(
    tmp_path, monkeypatch
):
    from agent_rt.agents import Agent, Runner

    import ext.compat.base as compat_base

    (tmp_path / "agent-rt.yaml").write_text(
        "model_list:\n  - model_name: only-this\n    agent_rt_params: {model: openai/x, api_key: k}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    vendor = RecordingProvider("vendor env")
    monkeypatch.setattr(compat_base, "_openai_provider", lambda **_kwargs: vendor)
    result = await Runner.run(Agent(name="a", model="gpt-4o-mini"), "hi")
    assert result.final_output == "vendor env"
    assert vendor.requests[0].model == "gpt-4o-mini"
