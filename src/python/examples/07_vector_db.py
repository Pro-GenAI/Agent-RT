import asyncio
import sys

from agent_rt import RetrievalQuery, vector_store_registry_from_config

from _common import example_config, example_provider


async def main() -> None:
    # Every vector_store_registry entry in agent-rt.yaml is registered under its
    # name and id; each store embeds queries with its configured embedding_model.
    config = example_config()
    if not config.vector_stores:
        raise RuntimeError("Add a vector_store_registry entry to your agent-rt.yaml.")
    registry = vector_store_registry_from_config(
        config,
        embedding_provider=example_provider(),
    )

    store = sys.argv[1] if len(sys.argv) > 1 else config.vector_stores[0].name
    query = sys.argv[2] if len(sys.argv) > 2 else "What is Agent RT?"
    results = await registry.search(store, RetrievalQuery(query, limit=5))

    print(f"vector_store={store} results={len(results)}")
    for result in results:
        print(f"- {result.score!s:>8}  {result.title}")
        if result.uri:
            print(f"  {result.uri}")
        print(f"  {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
