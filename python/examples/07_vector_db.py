import asyncio
import os

from agent_rt import (
    RetrievalQuery,
    VectorDBProviderRegistry,
    load_model,
    register_popular_vector_db_backends,
    vector_db_provider_from_environment,
)


async def main() -> None:
    # OpenAI/OpenAI-compatible providers implement Agent RT's embedding contract.
    # Alternatively, pass a precomputed numeric vector as filters={"vector": [...]}.
    embedding_provider = load_model()
    if not hasattr(embedding_provider, "embed"):
        raise RuntimeError(
            "This example needs an embedding-capable provider (for example OpenAI) "
            "or a precomputed filters.vector."
        )

    registry = VectorDBProviderRegistry()
    register_popular_vector_db_backends(
        registry,
        embedding_provider=embedding_provider,
    )

    # AGENT_RT_VECTOR_DB chooses chroma|milvus|pinecone|qdrant|weaviate.
    # Changing only the environment selects a different registered backend.
    provider = vector_db_provider_from_environment(registry)

    query = os.environ.get("AGENT_RT_VECTOR_DB_QUERY", "What is Agent RT?")
    limit = int(os.environ.get("AGENT_RT_VECTOR_DB_LIMIT", "5"))
    results = await provider.search(RetrievalQuery(query, limit=limit))

    print(f"backend={os.environ['AGENT_RT_VECTOR_DB']} results={len(results)}")
    for result in results:
        print(f"- {result.score!s:>8}  {result.title}")
        if result.uri:
            print(f"  {result.uri}")
        print(f"  {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
