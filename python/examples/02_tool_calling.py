import asyncio

from agent_rt import (
    AgentLoop,
    ToolDefinition,
    ToolRegistry,
    load_model,
)

from _common import make_agent, response_text, user_message


async def main() -> None:
    registry = ToolRegistry()

    async def weather(arguments, cancellation_token):
        city = arguments["city"]
        # Replace this deterministic demo data with a real API in your application.
        return {"city": city, "temperature_c": 28, "condition": "clear"}

    registry.register(
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            input_schema={
                "type": "object",
                "required": ["city"],
                "properties": {"city": {"type": "string"}},
                "additionalProperties": False,
            },
            side_effect="read",
        ),
        handler=weather,
    )

    provider = load_model()
    agent = make_agent(
        "weather-assistant",
        "Use get_weather when the user asks about weather. Summarize the tool result.",
    )

    result = await AgentLoop(provider, tool_registry=registry).run(
        agent,
        [user_message("What is the weather in Chennai?")],
    )

    print(response_text(result.final_response))
    print(f"tool_calls={result.tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
