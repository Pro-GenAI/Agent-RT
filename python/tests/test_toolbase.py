import pytest

from agent_rt import (
    TOOL_SEARCH_NAME,
    AgentConfig,
    ContentPart,
    ModelMessage,
    ModelSettings,
    Toolbase,
    ToolCall,
    ToolDefinition,
    ToolFilterContext,
    ToolRegistry,
)


class FakeMCPClient:
    def __init__(self):
        self.server_info = None
        self.initialized = False
        self.calls = []

    async def initialize(self):
        self.initialized = True
        self.server_info = {"name": "docs"}

    async def list_tools(self):
        return (
            {
                "name": "weather",
                "description": "Look up weather forecasts.",
                "input_schema": {"type": "object"},
            },
            {
                "name": "dangerous",
                "description": "Dangerous remote action.",
                "input_schema": {"type": "object"},
            },
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"tool": name, "arguments": arguments}


def _agent():
    return AgentConfig(
        name="toolbase-test",
        instructions="Use tools.",
        model=ModelSettings(model="fake-model"),
    )


@pytest.mark.asyncio
async def test_toolbase_aggregates_local_and_mcp_tools_and_filters_unsafe_tools():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calendar",
            description="Read calendar events.",
            input_schema={"type": "object"},
        )
    )
    registry.register(
        ToolDefinition(
            name="unsafe_local",
            description="Unsafe local action.",
            input_schema={"type": "object"},
        )
    )
    client = FakeMCPClient()

    async def safety_filter(tools):
        return tuple(
            tool.name
            for tool in tools
            if tool.name not in {"unsafe_local", "mcp.docs.dangerous"}
        )

    toolbase = await Toolbase.initialize(
        tool_registry=registry,
        mcp_clients={"docs": client},
        safety_filter=safety_filter,
    )

    assert client.initialized
    assert {tool.name for tool in toolbase.definitions()} == {
        "calendar",
        "mcp.docs.weather",
        TOOL_SEARCH_NAME,
    }
    with pytest.raises(KeyError, match="safety-screened"):
        toolbase.get("unsafe_local")
    with pytest.raises(KeyError):
        registry.get("mcp.docs.dangerous")

    result = await toolbase.execute(
        ToolCall(
            id="mcp-call",
            name="mcp.docs.weather",
            arguments={"city": "Bengaluru"},
        )
    )
    assert result == {
        "tool": "weather",
        "arguments": {"city": "Bengaluru"},
    }
    assert client.calls == [("weather", {"city": "Bengaluru"})]


@pytest.mark.asyncio
async def test_tool_search_promotes_safe_tool_pruned_by_relevance_filter():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calendar",
            description="Read calendar events.",
            input_schema={"type": "object"},
        )
    )
    registry.register(
        ToolDefinition(
            name="weather",
            description="Look up weather forecasts and temperatures.",
            input_schema={"type": "object"},
        )
    )

    def select_calendar(_context, _tools):
        return ("calendar",)

    toolbase = await Toolbase.initialize(
        tool_registry=registry,
        selection_filter=select_calendar,
    )
    visibility_filter = toolbase.visibility_filter()
    definitions = toolbase.definitions()
    initial = await visibility_filter(
        ToolFilterContext(
            agent=_agent(),
            messages=(),
            turn=0,
            tool_calls=0,
        ),
        definitions,
    )
    assert initial == ("calendar", TOOL_SEARCH_NAME)

    search_call = ToolCall(
        id="search-1",
        name=TOOL_SEARCH_NAME,
        arguments={"query": "weather forecast"},
    )
    search_result = await toolbase.execute(search_call)
    assert [tool["name"] for tool in search_result["tools"]] == ["weather"]

    messages = (
        ModelMessage(role="assistant", content=(), tool_calls=(search_call,)),
        ModelMessage(
            role="tool",
            content=(ContentPart(type="json", data=search_result),),
            tool_call_id=search_call.id,
        ),
    )
    after_search = await visibility_filter(
        ToolFilterContext(
            agent=_agent(),
            messages=messages,
            turn=1,
            tool_calls=1,
        ),
        definitions,
    )
    assert after_search == ("calendar", "weather", TOOL_SEARCH_NAME)


@pytest.mark.asyncio
async def test_toolbase_fails_closed_when_screened_tool_is_replaced():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="safe",
            description="Safe tool.",
            input_schema={"type": "object"},
        )
    )
    toolbase = await Toolbase.initialize(tool_registry=registry)

    registry.register(
        ToolDefinition(
            name="safe",
            description="Replacement definition.",
            input_schema={"type": "object"},
        ),
        replace=True,
    )

    assert {tool.name for tool in toolbase.definitions()} == {TOOL_SEARCH_NAME}
    with pytest.raises(KeyError, match="safety-screened"):
        toolbase.get("safe")
