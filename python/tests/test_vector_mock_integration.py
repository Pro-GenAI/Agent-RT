from __future__ import annotations

import pytest

from agent_rt import EnvironmentVectorDBProvider, RetrievalQuery
from tests.helpers.vector_mock import QueryResult, VectorEntry, VectorMock

BACKENDS = ("chroma", "milvus", "pinecone", "qdrant", "weaviate")


def test_vector_mock_matches_aimock_fixture_lifecycle() -> None:
    mock = VectorMock()
    assert mock.health() == {"status": "ok", "collections": 0}

    mock.addCollection("docs", {"dimension": 3}).upsert(
        "docs",
        [
            VectorEntry(
                id="doc-1",
                values=[0.1, 0.2, 0.3],
                metadata={"text": "stored fixture"},
            )
        ],
    )
    assert mock.health() == {"status": "ok", "collections": 1}
    assert mock.vectors["docs"]["doc-1"].values == [0.1, 0.2, 0.3]

    mock.deleteCollection("docs")
    assert mock.health() == {"status": "ok", "collections": 0}

    mock.add_collection("docs", dimension=3).reset()
    assert mock.health() == {"status": "ok", "collections": 0}


def _collection(backend: str) -> str:
    return "Docs" if backend == "weaviate" else "docs"


def _fixture_result(backend: str) -> QueryResult:
    return QueryResult(
        id=f"{backend}-1",
        score=0.9,
        metadata={
            "text": f"{backend} mock hit",
            "title": f"{backend.title()} result",
            "url": f"https://example.test/{backend}",
        },
    )


@pytest.mark.parametrize("backend", BACKENDS)
async def test_core_vector_db_backends_against_vector_mock(backend: str) -> None:
    collection = _collection(backend)
    seen = []
    mock = VectorMock().add_collection(collection, dimension=3)
    mock.on_query(
        collection,
        lambda query: seen.append(query) or [_fixture_result(backend)],
    )
    url = mock.start()
    try:
        provider = EnvironmentVectorDBProvider.from_environment(
            {
                "AGENT_RT_VECTOR_DB": backend,
                "AGENT_RT_VECTOR_DB_COLLECTION": collection,
                "AGENT_RT_VECTOR_DB_URL": url,
            }
        )
        results = await provider.search(
            RetrievalQuery(
                "agent runtime",
                limit=1,
                filters={"vector": [0.1, 0.2, 0.3]},
            )
        )
    finally:
        mock.stop()

    assert len(results) == 1
    assert results[0].id == f"{backend}-1"
    assert results[0].content == f"{backend} mock hit"
    assert results[0].metadata["provider"] == backend
    assert seen[0].vector == [0.1, 0.2, 0.3]
    assert seen[0].top_k == 1


class _LangChainEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        assert text == "agent runtime"
        return [0.1, 0.2, 0.3]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_langchain_vector_store_backends_against_vector_mock(
    backend: str,
) -> None:
    from agent_rt.langchain_chroma import Chroma
    from agent_rt.langchain_milvus import Milvus
    from agent_rt.langchain_pinecone import PineconeVectorStore
    from agent_rt.langchain_qdrant import QdrantVectorStore
    from agent_rt.langchain_weaviate import WeaviateVectorStore

    classes = {
        "chroma": Chroma,
        "milvus": Milvus,
        "pinecone": PineconeVectorStore,
        "qdrant": QdrantVectorStore,
        "weaviate": WeaviateVectorStore,
    }
    collection = _collection(backend)
    seen = []
    mock = VectorMock().add_collection(collection, dimension=3)
    mock.on_query(
        collection,
        lambda query: seen.append(query) or [_fixture_result(backend)],
    )
    url = mock.start()
    try:
        store = classes[backend](
            _LangChainEmbeddings(),
            environment={
                "AGENT_RT_VECTOR_DB_URL": url,
                "AGENT_RT_VECTOR_DB_COLLECTION": collection,
            },
        )
        documents = await store.asimilarity_search("agent runtime", k=1)
    finally:
        mock.stop()

    assert len(documents) == 1
    assert documents[0].page_content == f"{backend} mock hit"
    assert documents[0].metadata["provider"] == backend
    assert seen[0].vector == [0.1, 0.2, 0.3]
    assert seen[0].top_k == 1


@pytest.mark.parametrize("backend", BACKENDS)
async def test_llamaindex_vector_store_backends_against_vector_mock(
    backend: str,
) -> None:
    from agent_rt.llama_index.vector_stores.chroma import ChromaVectorStore
    from agent_rt.llama_index.vector_stores.milvus import MilvusVectorStore
    from agent_rt.llama_index.vector_stores.pinecone import PineconeVectorStore
    from agent_rt.llama_index.vector_stores.qdrant import QdrantVectorStore
    from agent_rt.llama_index.vector_stores.weaviate import WeaviateVectorStore

    classes = {
        "chroma": ChromaVectorStore,
        "milvus": MilvusVectorStore,
        "pinecone": PineconeVectorStore,
        "qdrant": QdrantVectorStore,
        "weaviate": WeaviateVectorStore,
    }
    collection = _collection(backend)
    seen = []
    mock = VectorMock().add_collection(collection, dimension=3)
    mock.on_query(
        collection,
        lambda query: seen.append(query) or [_fixture_result(backend)],
    )
    url = mock.start()
    try:
        store = classes[backend](
            environment={
                "AGENT_RT_VECTOR_DB_URL": url,
                "AGENT_RT_VECTOR_DB_COLLECTION": collection,
            }
        )
        nodes = await store.aquery(
            {
                "query_embedding": [0.1, 0.2, 0.3],
                "query_str": "agent runtime",
                "similarity_top_k": 1,
            }
        )
    finally:
        mock.stop()

    assert len(nodes) == 1
    assert nodes[0].node.text == f"{backend} mock hit"
    assert nodes[0].node.metadata["provider"] == backend
    assert nodes[0].score == pytest.approx(0.9)
    assert seen[0].vector == [0.1, 0.2, 0.3]
    assert seen[0].top_k == 1
