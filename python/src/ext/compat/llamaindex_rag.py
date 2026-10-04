from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_rt import EmbeddingRequest, FileSystem
from ext.compat.llamaindex_prompts import Settings


@dataclass
class Document:
    text: str = ""
    id_: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


@dataclass
class TextNode:
    text: str
    id_: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    ref_doc_id: str | None = None

    def get_content(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


@dataclass
class NodeWithScore:
    node: TextNode
    score: float | None = None


class SimpleDirectoryReader:
    def __init__(
        self,
        *,
        file_system: FileSystem,
        input_dir: str = "",
        required_exts: Sequence[str] | None = None,
        recursive: bool = False,
    ) -> None:
        self.file_system = file_system
        self.input_dir = input_dir
        self.required_exts = tuple(required_exts or ())
        self.recursive = recursive

    def load_data(self) -> list[Document]:
        if self.recursive:
            pattern = f"{self.input_dir}/**/*" if self.input_dir else "**/*"
            paths = self.file_system.glob(pattern)
        else:
            paths = tuple(
                entry.path
                for entry in self.file_system.list(self.input_dir)
                if not entry.is_directory
            )
        extensions = tuple(
            value.lower() if value.startswith(".") else f".{value.lower()}"
            for value in self.required_exts
        )
        documents: list[Document] = []
        for path in paths:
            if extensions and not path.lower().endswith(extensions):
                continue
            documents.append(
                Document(
                    text=self.file_system.read(path).decode("utf-8"),
                    id_=path,
                    metadata={"file_path": path},
                )
            )
        return documents


class SentenceSplitter:
    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 20) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self.chunk_size])
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def get_nodes_from_documents(self, documents: Sequence[Document]) -> list[TextNode]:
        nodes: list[TextNode] = []
        for document in documents:
            for index, chunk in enumerate(self.split_text(document.text)):
                doc_id = document.id_ or "document"
                nodes.append(
                    TextNode(
                        text=chunk,
                        id_=f"{doc_id}-chunk-{index}",
                        metadata=dict(document.metadata),
                        ref_doc_id=document.id_,
                    )
                )
        return nodes

    def transform(self, nodes: Sequence[TextNode]) -> list[TextNode]:
        result: list[TextNode] = []
        for node in nodes:
            for index, chunk in enumerate(self.split_text(node.text)):
                result.append(
                    TextNode(
                        text=chunk,
                        id_=f"{node.id_ or 'node'}-chunk-{index}",
                        metadata=dict(node.metadata),
                        ref_doc_id=node.ref_doc_id,
                    )
                )
        return result


class AgentRTEmbedding:
    def __init__(self, provider: Any, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    async def aget_text_embedding_batch(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        response = await self.provider.embed(
            EmbeddingRequest(input=list(texts), model=self.model)
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors: list[list[float]] = []
        for item in ordered:
            if isinstance(item.embedding, str):
                raise TypeError("LlamaIndex compatibility requires numeric embeddings")
            vectors.append([float(value) for value in item.embedding])
        return vectors

    async def aget_text_embedding(self, text: str) -> list[float]:
        return (await self.aget_text_embedding_batch([text]))[0]

    async def aget_query_embedding(self, query: str) -> list[float]:
        return await self.aget_text_embedding(query)


class IngestionPipeline:
    def __init__(
        self,
        *,
        transformations: Sequence[Any] | None = None,
        embed_model: AgentRTEmbedding | None = None,
    ) -> None:
        self.transformations = list(transformations or ())
        self.embed_model = embed_model

    async def arun(
        self,
        *,
        documents: Sequence[Document] | None = None,
        nodes: Sequence[TextNode] | None = None,
    ) -> list[TextNode]:
        current = list(nodes or ())
        if not current and documents is not None:
            current = (
                [
                    TextNode(
                        text=doc.text,
                        id_=doc.id_,
                        metadata=dict(doc.metadata),
                        ref_doc_id=doc.id_,
                    )
                    for doc in documents
                ]
                if self.transformations
                else SentenceSplitter().get_nodes_from_documents(documents)
            )
        for transformation in self.transformations:
            current = list(transformation.transform(current))
        if self.embed_model is not None and current:
            vectors = await self.embed_model.aget_text_embedding_batch(
                [node.text for node in current]
            )
            current = [
                TextNode(
                    text=node.text,
                    id_=node.id_,
                    metadata=dict(node.metadata),
                    embedding=vector,
                    ref_doc_id=node.ref_doc_id,
                )
                for node, vector in zip(current, vectors, strict=True)
            ]
        return current


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(value * value for value in left))
    rn = math.sqrt(sum(value * value for value in right))
    return dot / (ln * rn) if ln and rn else 0.0


class SimpleVectorStore:
    def __init__(self) -> None:
        self._nodes: dict[str, TextNode] = {}

    def add(self, nodes: Sequence[TextNode]) -> list[str]:
        ids: list[str] = []
        for index, node in enumerate(nodes):
            if node.embedding is None:
                raise ValueError("nodes must be embedded before insertion")
            node_id = node.id_ or f"node-{len(self._nodes) + index}"
            self._nodes[node_id] = node
            ids.append(node_id)
        return ids

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        similarity_top_k: int = 2,
        filters: Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        candidates = [
            node
            for node in self._nodes.values()
            if not filters or all(node.metadata.get(k) == v for k, v in filters.items())
        ]
        scored = [
            NodeWithScore(
                node=node, score=_cosine(query_embedding, node.embedding or ())
            )
            for node in candidates
        ]
        return sorted(scored, key=lambda item: item.score or 0.0, reverse=True)[
            :similarity_top_k
        ]

    def delete(self, ref_doc_id: str) -> None:
        self._nodes = {
            key: node
            for key, node in self._nodes.items()
            if node.ref_doc_id != ref_doc_id and key != ref_doc_id
        }


class AgentRTVectorStore:
    """LlamaIndex-shaped read/query vector store backed by Agent RT."""

    backend: str | None = None

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        index_name: str | None = None,
        url: str | None = None,
        environment: Mapping[str, str] | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.backend:
            raise TypeError("AgentRTVectorStore requires a concrete backend subclass")
        env = dict(os.environ if environment is None else environment)
        env["AGENT_RT_VECTOR_DB"] = self.backend
        resolved_collection = collection_name or index_name
        if resolved_collection:
            env["AGENT_RT_VECTOR_DB_COLLECTION"] = resolved_collection
        if url:
            env["AGENT_RT_VECTOR_DB_URL"] = url
        credential = kwargs.pop("api_key", None)
        if credential:
            env["AGENT_RT_VECTOR_DB_API_KEY"] = str(credential)
        for key in (
            "namespace",
            "tenant",
            "database",
            "content_field",
            "title_field",
            "uri_field",
            "filter",
        ):
            value = kwargs.pop(key, None)
            if value is not None:
                env[f"AGENT_RT_VECTOR_DB_OPTION_{key.upper()}"] = str(value)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported vector store compatibility options: {unknown}")
        self.environment = env
        from ext.runtime.optional import EnvironmentVectorDBProvider

        self.provider = EnvironmentVectorDBProvider.from_environment(env, client=client)

    async def aquery(
        self,
        query: Any,
        *,
        similarity_top_k: int = 2,
        filters: Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        query_embedding = getattr(query, "query_embedding", None)
        query_str = getattr(query, "query_str", None)
        if isinstance(query, Mapping):
            query_embedding = query.get("query_embedding", query_embedding)
            query_str = query.get("query_str", query_str)
            similarity_top_k = int(query.get("similarity_top_k", similarity_top_k))
            filters = query.get("filters", filters)
        elif isinstance(query, Sequence) and not isinstance(query, (str, bytes, bytearray)):
            query_embedding = query
        request_filters = dict(filters or {})
        if query_embedding is not None:
            request_filters["vector"] = [float(value) for value in query_embedding]
        text = str(query_str or "vector query")
        from ext.runtime.optional import RetrievalQuery

        results = await self.provider.search(
            RetrievalQuery(
                text=text,
                limit=similarity_top_k,
                filters=request_filters,
            )
        )
        return [
            NodeWithScore(
                node=TextNode(
                    text=(
                        item.content if isinstance(item.content, str) else str(item.content)
                    ),
                    id_=item.id,
                    metadata={
                        **dict(item.metadata),
                        "title": item.title,
                        **({"url": item.uri} if item.uri else {}),
                    },
                ),
                score=item.score,
            )
            for item in results
        ]

    def query(
        self,
        query: Any,
        *,
        similarity_top_k: int = 2,
        filters: Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        from ext.compat.base import _run_sync

        return _run_sync(
            self.aquery(
                query,
                similarity_top_k=similarity_top_k,
                filters=filters,
            )
        )

    def add(self, _nodes: Sequence[TextNode]) -> list[str]:
        if not _nodes:
            return []
        raise NotImplementedError(
            "Agent RT vector DB compatibility currently supports retrieval from existing collections; "
            "write/upsert remains backend-native"
        )

    def delete(self, _ref_doc_id: str) -> None:
        raise NotImplementedError(
            "Agent RT vector DB compatibility currently supports retrieval from existing collections; "
            "delete remains backend-native"
        )


class ChromaVectorStore(AgentRTVectorStore):
    backend = "chroma"


class PineconeVectorStore(AgentRTVectorStore):
    backend = "pinecone"


class QdrantVectorStore(AgentRTVectorStore):
    backend = "qdrant"


class MilvusVectorStore(AgentRTVectorStore):
    backend = "milvus"


class WeaviateVectorStore(AgentRTVectorStore):
    backend = "weaviate"


class StorageContext:
    def __init__(self, vector_store: Any = None) -> None:
        self.vector_store = vector_store or SimpleVectorStore()

    @classmethod
    def from_defaults(
        cls, *, vector_store: Any = None, **_: Any
    ) -> StorageContext:
        return cls(vector_store)


class VectorIndexRetriever:
    def __init__(
        self,
        index: VectorStoreIndex,
        *,
        similarity_top_k: int = 2,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.filters = filters

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        embedding = await self.index.embed_model.aget_query_embedding(query)
        vector_store = self.index.storage_context.vector_store
        if hasattr(vector_store, "aquery"):
            return await vector_store.aquery(
                {
                    "query_embedding": embedding,
                    "query_str": query,
                    "similarity_top_k": self.similarity_top_k,
                    "filters": self.filters,
                }
            )
        return vector_store.query(
            embedding,
            similarity_top_k=self.similarity_top_k,
            filters=self.filters,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        from ext.compat.base import _run_sync

        return _run_sync(self.aretrieve(query))


class AgentRTRetriever:
    def __init__(
        self,
        provider: Any = None,
        *,
        registry: Any = None,
        provider_name: str | None = None,
        similarity_top_k: int = 2,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.provider_name = provider_name
        self.similarity_top_k = similarity_top_k
        self.filters = dict(filters or {})

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        from ext.runtime.optional import RetrievalQuery

        request = RetrievalQuery(
            text=query, limit=self.similarity_top_k, filters=self.filters
        )
        if self.provider is not None:
            results = await self.provider.search(request)
        elif self.registry is not None and self.provider_name:
            results = await self.registry.search(self.provider_name, request)
        else:
            raise ValueError(
                "AgentRTRetriever requires provider or registry/provider_name"
            )
        return [
            NodeWithScore(
                TextNode(
                    id_=result.id,
                    text=(
                        result.content
                        if isinstance(result.content, str)
                        else str(result.content)
                    ),
                    metadata={
                        **dict(result.metadata),
                        "title": result.title,
                        **({"uri": result.uri} if result.uri else {}),
                    },
                ),
                result.score,
            )
            for result in results
        ]

    def retrieve(self, query: str) -> list[NodeWithScore]:
        from ext.compat.base import _run_sync

        return _run_sync(self.aretrieve(query))


@dataclass
class Response:
    response: str
    source_nodes: list[NodeWithScore] = field(default_factory=list)

    def __str__(self) -> str:
        return self.response


class ResponseSynthesizer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def asynthesize(self, query: str, nodes: Sequence[NodeWithScore]) -> Response:
        context = "\n\n".join(item.node.text for item in nodes)
        prompt = f"Use only this context to answer.\nContext:\n{context}\nQuery: {query}\nAnswer:"
        completion = await self.llm.acomplete(prompt)
        return Response(completion.text, list(nodes))


class RetrieverQueryEngine:
    def __init__(
        self,
        *,
        retriever: Any,
        llm: Any | None = None,
        response_synthesizer: ResponseSynthesizer | None = None,
    ) -> None:
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer or (
            ResponseSynthesizer(llm) if llm is not None else None
        )

    async def aquery(self, query: str) -> Response:
        nodes = await self.retriever.aretrieve(query)
        if self.response_synthesizer is None:
            return Response("\n\n".join(item.node.text for item in nodes), nodes)
        return await self.response_synthesizer.asynthesize(query, nodes)


class VectorStoreIndex:
    def __init__(
        self,
        nodes: Sequence[TextNode],
        *,
        embed_model: AgentRTEmbedding,
        storage_context: StorageContext | None = None,
    ) -> None:
        self.embed_model = embed_model
        self.storage_context = storage_context or StorageContext.from_defaults()
        self.storage_context.vector_store.add(nodes)

    @classmethod
    def from_vector_store(
        cls,
        vector_store: Any,
        *,
        embed_model: AgentRTEmbedding | None = None,
    ) -> VectorStoreIndex:
        active_embed_model = embed_model or Settings.embed_model
        if active_embed_model is None:
            raise ValueError(
                "VectorStoreIndex.from_vector_store requires embed_model or Settings.embed_model"
            )
        return cls(
            [],
            embed_model=active_embed_model,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
        )

    @classmethod
    async def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        embed_model: AgentRTEmbedding | None = None,
        transformations: Sequence[Any] | None = None,
        storage_context: StorageContext | None = None,
    ) -> VectorStoreIndex:
        active_embed_model = embed_model or Settings.embed_model
        if active_embed_model is None:
            raise ValueError(
                "VectorStoreIndex.from_documents requires embed_model or Settings.embed_model"
            )
        nodes = await IngestionPipeline(
            transformations=transformations, embed_model=active_embed_model
        ).arun(documents=documents)
        return cls(
            nodes, embed_model=active_embed_model, storage_context=storage_context
        )

    def as_retriever(self, **kwargs: Any) -> VectorIndexRetriever:
        return VectorIndexRetriever(self, **kwargs)

    def as_query_engine(
        self, *, llm: Any | None = None, **kwargs: Any
    ) -> RetrieverQueryEngine:
        return RetrieverQueryEngine(
            retriever=self.as_retriever(**kwargs), llm=llm or Settings.llm
        )
