import asyncio

from agent_rt import AgentLoop, load_model

from _common import make_agent, user_message


async def main() -> None:
    provider = load_model()
    agent = make_agent("streaming", "Answer in a short paragraph.")

    async def on_event(event) -> None:
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)

    result = await AgentLoop(provider).run_streaming(
        agent,
        [user_message("Give me three practical uses for an agent harness.")],
        on_event,
    )

    print()
    print(f"termination={result.termination_reason} tokens={result.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
