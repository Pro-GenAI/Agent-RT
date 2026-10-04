import asyncio

from agent_rt import AgentLoop, load_model

from _common import make_agent, response_text, user_message


async def main() -> None:
    provider = load_model()
    agent = make_agent(
        "basic",
        "Answer clearly and concisely. Do not use tools.",
    )

    result = await AgentLoop(provider).run(
        agent,
        [user_message("Explain what an agent harness does in two sentences.")],
    )

    print(response_text(result.final_response))
    print(f"termination={result.termination_reason} turns={result.turns}")


if __name__ == "__main__":
    asyncio.run(main())
