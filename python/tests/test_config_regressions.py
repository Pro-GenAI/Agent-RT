"""Offline regression coverage for YAML Bedrock stores and deferred configuration."""

import httpx
import pytest
from openai import AsyncOpenAI

from agent_rt import (
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    RetrievalQuery,
    ToolCall,
    vector_store_registry_from_config,
)
from agent_rt_api import build_from_config, create_app
from ext.config import ConfigError, parse_config

BEDROCK_YAML = """
model_list:
  - model_name: claude-sonnet-5
    litellm_params: {model: anthropic/claude-sonnet-5}
vector_store_registry:
  - vector_store_name: docs-kb
    litellm_params:
      custom_llm_provider: bedrock
      vector_store_id: T37J8R4WTM
      vector_store_description: Documentation
      options: {region: us-west-2}
"""


class FakeBedrock:
    def __init__(self):
        self.calls = []

    def retrieve(self, **params):
        self.calls.append(params)
        return {
            "retrievalResults": [
                {
                    "content": {"text": "LiteLLM is a gateway."},
                    "location": {
                        "webLocation": {"url": "https://www.litellm.com/docs"}
                    },
                    "score": 0.8,
                }
            ]
        }


class FakeModel:
    name = "offline"

    async def complete(self, request):
        if not any(msg.role == "tool" for msg in request.messages):
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id="call1",
                            name="file_search",
                            arguments={"query": "LiteLLM"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="LiteLLM is a gateway."),),
            ),
            finish_reason="stop",
        )

    async def stream(self, request):
        response = await self.complete(request)
        if response.message.content:
            yield ModelStreamEvent(type="text_delta", text="LiteLLM is a gateway.")
        yield ModelStreamEvent(type="completed", response=response)


@pytest.mark.asyncio
async def test_bedrock_retrieval_and_aliases():
    config = parse_config(BEDROCK_YAML, environ={})
    fake = FakeBedrock()
    registry = vector_store_registry_from_config(
        config,
        bedrock_client_factory=lambda _: fake,
    )
    assert registry._providers["docs-kb"] is registry._providers["T37J8R4WTM"]
    assert fake.calls == []
    rows = await registry.search("T37J8R4WTM", RetrievalQuery(text="LiteLLM", limit=3))
    assert rows[0].content == "LiteLLM is a gateway."
    assert rows[0].uri == "https://www.litellm.com/docs"
    assert fake.calls[0] == {
        "knowledgeBaseId": "T37J8R4WTM",
        "retrievalQuery": {"text": "LiteLLM"},
        "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 3}},
    }


@pytest.mark.asyncio
async def test_unused_store_secret_is_deferred():
    config = parse_config(
        BEDROCK_YAML + """
  - vector_store_name: optional
    litellm_params:
      custom_llm_provider: qdrant
      vector_store_id: unused
      api_base: os.environ/UNSET_TEST_URL
      collection: docs
""",
        environ={},
    )
    built = build_from_config(
        config,
        provider_factory=lambda _: FakeModel(),
        bedrock_client_factory=lambda _: FakeBedrock(),
    )
    with pytest.raises(ConfigError, match="UNSET_TEST_URL"):
        await built["retrieval"].search("unused", RetrievalQuery(text="x"))
    assert await built["retrieval"].search("docs-kb", RetrievalQuery(text="LiteLLM"))


def test_config_repr_redacts_sensitive_nested_values():
    config = parse_config(
        BEDROCK_YAML + """
agent_rt_settings:
  server:
    api_keys: [redacted-test-placeholder]
""",
        environ={},
    )
    assert "redacted-test-placeholder" not in repr(config)
    assert "redacted-test-placeholder" in config.settings["server"]["api_keys"]


@pytest.mark.asyncio
async def test_openai_sdk_file_search_over_http_and_sse():
    config = parse_config(BEDROCK_YAML, environ={})
    fake = FakeBedrock()
    built = build_from_config(
        config,
        provider_factory=lambda _: FakeModel(),
        bedrock_client_factory=lambda _: fake,
    )
    app = create_app(
        built["loop"], built["agents"], config=config, retrieval=built["retrieval"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        sdk = AsyncOpenAI(
            base_url="http://test/v1", api_key="offline", http_client=http_client
        )
        request = {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
            "tools": [{"type": "file_search", "vector_store_ids": ["T37J8R4WTM"]}],
        }
        answer = await sdk.chat.completions.create(**request)
        assert answer.choices[0].message.content == "LiteLLM is a gateway."
        streamed = await sdk.chat.completions.create(**request, stream=True)
        chunks = [chunk async for chunk in streamed]
        assert "LiteLLM is a gateway." in "".join(
            choice.delta.content or "" for chunk in chunks for choice in chunk.choices
        )
    assert len(fake.calls) == 2
