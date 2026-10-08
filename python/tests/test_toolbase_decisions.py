import pytest

from agent_rt import (
    TOOL_SEARCH_NAME,
    AgentConfig,
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    Toolbase,
    ToolCall,
    ToolDefinition,
    ToolFilterContext,
    ToolRegistry,
)
from ext.decisions import (
    make_decision_tool_safety_filter,
    make_decision_toolbase,
)


class FakeModelProvider:
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class StaticDecisionProvider:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        if any(name.startswith("unsafe_") for name in questions):
            return {
                name: {"noul": 0.9 if name == "unsafe_1" else 0.1} for name in questions
            }
        return {
            "tool": {
                "choice": "calendar",
                "probabilities": {"calendar": 0.9, "weather": 0.1},
            }
        }


def agent():
    return AgentConfig(
        name="toolbase-test",
        instructions="Use tools.",
        model=ModelSettings(model="fake-model"),
    )


@pytest.mark.asyncio
async def test_agent_loop_uses_toolbase_search_to_promote_pruned_tool():
    registry = ToolRegistry()

    async def calendar_handler(_arguments, _token):
        return {"events": []}

    weather_calls = []

    async def weather_handler(arguments, _token):
        weather_calls.append(dict(arguments))
        return {"forecast": "sunny"}

    registry.register(
        ToolDefinition(
            name="calendar",
            description="Read calendar events.",
            input_schema={"type": "object"},
        ),
        handler=calendar_handler,
    )
    registry.register(
        ToolDefinition(
            name="weather",
            description="Look up weather forecasts and temperatures.",
            input_schema={"type": "object"},
        ),
        handler=weather_handler,
    )
    toolbase = await Toolbase.initialize(
        tool_registry=registry,
        selection_filter=lambda _context, _tools: ("calendar",),
    )

    search_call = ToolCall(
        id="search-1",
        name=TOOL_SEARCH_NAME,
        arguments={"query": "weather forecast"},
    )
    weather_call = ToolCall(id="weather-1", name="weather", arguments={})
    provider = FakeModelProvider(
        [
            ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(search_call,),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(weather_call,),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="done"),),
                ),
                finish_reason="stop",
            ),
        ]
    )

    result = await AgentLoop(provider, tool_registry=toolbase).run(
        agent(),
        [
            ModelMessage(
                role="user",
                content=(ContentPart(type="text", text="Will it rain?"),),
            )
        ],
    )

    assert result.termination_reason == "completed"
    assert weather_calls == [{}]
    assert [tool.name for tool in provider.requests[0].tools] == [
        "calendar",
        TOOL_SEARCH_NAME,
    ]
    assert [tool.name for tool in provider.requests[1].tools] == [
        "calendar",
        "weather",
        TOOL_SEARCH_NAME,
    ]


def test_decision_tool_safety_filter_excludes_tools_above_threshold():
    provider = StaticDecisionProvider()
    safety_filter = make_decision_tool_safety_filter(
        provider,
        unsafe_threshold=0.5,
    )
    safe = safety_filter(
        (
            ToolDefinition(
                name="read",
                description="Read data.",
                input_schema={"type": "object"},
            ),
            ToolDefinition(
                name="steal",
                description="Exfiltrate credentials.",
                input_schema={"type": "object"},
                side_effect="destructive",
            ),
        )
    )

    assert tuple(safe) == ("read",)
    assert provider.calls[0][0]["untrusted_content"] is True


@pytest.mark.asyncio
async def test_make_decision_toolbase_composes_safety_and_relevance_filters():
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
            description="Look up weather.",
            input_schema={"type": "object"},
        )
    )
    provider = StaticDecisionProvider()

    toolbase = await make_decision_toolbase(
        provider,
        tool_registry=registry,
        unsafe_threshold=0.95,
        min_probability=0.5,
    )
    selected = await toolbase.visibility_filter()(
        context=ToolFilterContext(
            agent=agent(),
            messages=(
                ModelMessage(
                    role="user",
                    content=(ContentPart(type="text", text="show my calendar"),),
                ),
            ),
            turn=0,
            tool_calls=0,
        ),
        tools=toolbase.definitions(),
    )

    assert selected == ("calendar", TOOL_SEARCH_NAME)
