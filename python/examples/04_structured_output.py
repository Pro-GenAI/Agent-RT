import asyncio
import json

from agent_rt import (
    AgentConfig,
    AgentLoop,
    AgentOutputRequirements,
    ModelSettings,
    load_model,
)

from _common import configured_model, user_message


async def main() -> None:
    provider = load_model()
    agent = AgentConfig(
        name="extractor",
        instructions="Extract the requested fields and return only valid JSON.",
        model=ModelSettings(model=configured_model()),
        output=AgentOutputRequirements(
            format="json",
            schema={
                "type": "object",
                "required": ["language", "difficulty", "topics"],
                "properties": {
                    "language": {"type": "string"},
                    "difficulty": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            max_repair_attempts=1,
        ),
    )

    result = await AgentLoop(provider).run(
        agent,
        [user_message("Create a beginner Python learning plan covering functions and classes.")],
    )

    print(json.dumps(result.structured_output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
