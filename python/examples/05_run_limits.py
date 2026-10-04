import asyncio

from agent_rt import AgentLoop, AgentRunLimits, load_model

from _common import make_agent, response_text, user_message


async def main() -> None:
    provider = load_model()
    agent = make_agent("bounded", "Answer briefly.")

    limits = AgentRunLimits(
        max_turns=1,
        max_tool_calls=0,
        timeout_seconds=30,
        max_total_tokens=500,
    )

    result = await AgentLoop(provider).run(
        agent,
        [user_message("Name two reasons to put explicit limits around an agent loop.")],
        limits=limits,
    )

    print(response_text(result.final_response))
    print(
        f"termination={result.termination_reason} "
        f"turns={result.turns} tools={result.tool_calls} tokens={result.total_tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
