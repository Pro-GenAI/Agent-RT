import asyncio

from agent_rt import ToolCall, ToolDefinition, ToolRegistry


async def main() -> None:
    registry = ToolRegistry()

    async def search(arguments, cancellation_token):
        return {"query": arguments["query"], "matches": ["alpha", "beta"]}

    definition = ToolDefinition(
        name="search",
        description="Search a small demo catalog.",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
        side_effect="read",
    )

    registry.register(definition, namespace="catalog", handler=search)
    print("visible tools:", [tool.name for tool in registry.definitions()])

    result = await registry.execute(
        ToolCall(id="demo-1", name="catalog.search", arguments={"query": "agent"})
    )
    print("result:", result)

    registry.disable("catalog.search")
    print("after disable:", [tool.name for tool in registry.definitions()])


if __name__ == "__main__":
    asyncio.run(main())
