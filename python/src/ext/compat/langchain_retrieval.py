from __future__ import annotations

import csv
import inspect
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_rt import EmbeddingRequest, FileSystem
from ext.compat._format import safe_format


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


class TextLoader:
    def __init__(self, path: str, *, file_system: FileSystem) -> None:
        self.path = path
        self.file_system = file_system

    def load(self) -> list[Document]:
        return [
            Document(
                page_content=self.file_system.read(self.path).decode("utf-8"),
                metadata={"source": self.path},
                id=self.path,
            )
        ]


class JSONLoader:
    def __init__(
        self,
        path: str,
        *,
        file_system: FileSystem,
        content_key: str | None = None,
    ) -> None:
        self.path = path
        self.file_system = file_system
        self.content_key = content_key

    def load(self) -> list[Document]:
        value = json.loads(self.file_system.read(self.path).decode("utf-8"))
        items = value if isinstance(value, list) else [value]
        documents: list[Document] = []
        for index, item in enumerate(items):
            content = (
                item.get(self.content_key)
                if self.content_key and isinstance(item, Mapping)
                else item
            )
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": self.path, "index": index},
                    id=f"{self.path}:{index}",
                )
            )
        return documents


class CSVLoader:
    def __init__(self, path: str, *, file_system: FileSystem) -> None:
        self.path = path
        self.file_system = file_system

    def load(self) -> list[Document]:
        text = self.file_system.read(self.path).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        documents: list[Document] = []
        for index, row in enumerate(reader):
            page_content = "\n".join(f"{key}: {value}" for key, value in row.items())
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={"source": self.path, "row": index},
                    id=f"{self.path}:{index}",
                )
            )
        return documents


class DirectoryLoader:
    def __init__(
        self,
        path: str,
        *,
        file_system: FileSystem,
        recursive: bool = False,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self.path = path
        self.file_system = file_system
        self.recursive = recursive
        self.extensions = tuple(extensions or ())

    def load(self) -> list[Document]:
        if self.recursive:
            pattern = f"{self.path}/**/*" if self.path else "**/*"
            paths = self.file_system.glob(pattern)
        else:
            paths = tuple(
                item.path
                for item in self.file_system.list(self.path)
                if not item.is_directory
            )
        extensions = tuple(
            value.lower() if value.startswith(".") else f".{value.lower()}"
            for value in self.extensions
        )
        docs: list[Document] = []
        for path in paths:
            if extensions and not path.lower().endswith(extensions):
                continue
            docs.extend(TextLoader(path, file_system=self.file_system).load())
        return docs


class RecursiveCharacterTextSplitter:
    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        output: list[Document] = []
        step = self.chunk_size - self.chunk_overlap
        for document in documents:
            for index, start in enumerate(range(0, len(document.page_content), step)):
                chunk = document.page_content[start : start + self.chunk_size]
                if not chunk:
                    break
                output.append(
                    Document(
                        page_content=chunk,
                        metadata={**document.metadata, "chunk": index},
                        id=f"{document.id}:{index}" if document.id else None,
                    )
                )
                if start + self.chunk_size >= len(document.page_content):
                    break
        return output

    def create_documents(self, texts: Sequence[str]) -> list[Document]:
        return self.split_documents([Document(page_content=text) for text in texts])


class AgentRTEmbeddings:
    def __init__(self, provider: Any, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self.provider.embed(
            EmbeddingRequest(input=list(texts), model=self.model)
        )
        vectors: list[list[float]] = []
        for item in sorted(response.data, key=lambda value: value.index):
            if isinstance(item.embedding, str):
                raise TypeError("LangChain compatibility requires numeric embeddings")
            vectors.append([float(value) for value in item.embedding])
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        from ext.compat.base import _run_sync

        return _run_sync(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        from ext.compat.base import _run_sync

        return _run_sync(self.aembed_query(text))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(value * value for value in left))
    rn = math.sqrt(sum(value * value for value in right))
    return dot / (ln * rn) if ln and rn else 0.0


class MemoryVectorStore:
    def __init__(self, embeddings: AgentRTEmbeddings) -> None:
        self.embeddings = embeddings
        self._records: list[tuple[Document, list[float]]] = []

    @classmethod
    async def afrom_documents(
        cls, documents: Sequence[Document], embeddings: AgentRTEmbeddings
    ) -> MemoryVectorStore:
        store = cls(embeddings)
        await store.aadd_documents(documents)
        return store

    async def aadd_documents(self, documents: Sequence[Document]) -> list[str]:
        vectors = await self.embeddings.aembed_documents(
            [document.page_content for document in documents]
        )
        ids: list[str] = []
        for index, (document, vector) in enumerate(
            zip(documents, vectors, strict=True)
        ):
            self._records.append((document, vector))
            ids.append(document.id or f"doc-{len(self._records) + index}")
        return ids

    async def asimilarity_search(self, query: str, k: int = 4) -> list[Document]:
        vector = await self.embeddings.aembed_query(query)
        scored = sorted(
            (
                (document, _cosine(vector, embedding))
                for document, embedding in self._records
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [document for document, _score in scored[:k]]

    def as_retriever(self, *, k: int = 4) -> _VectorRetriever:
        return _VectorRetriever(self, k)


class AgentRTVectorStore:
    """LangChain-shaped vector store backed by Agent RT vector DB retrieval."""

    backend: str | None = None

    def __init__(
        self,
        embedding: Any = None,
        *,
        embedding_function: Any = None,
        collection_name: str | None = None,
        index_name: str | None = None,
        url: str | None = None,
        environment: Mapping[str, str] | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.backend:
            raise TypeError("AgentRTVectorStore requires a concrete backend subclass")
        self.embeddings = embedding or embedding_function or kwargs.pop("embeddings", None)
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

    async def _query_vector(self, text: str) -> list[float] | None:
        if self.embeddings is None:
            return None
        if hasattr(self.embeddings, "aembed_query"):
            value = self.embeddings.aembed_query(text)
        elif hasattr(self.embeddings, "embed_query"):
            value = self.embeddings.embed_query(text)
        else:
            raise TypeError(
                "embedding must expose embed_query() or aembed_query() for LangChain vector-store compatibility"
            )
        if inspect.isawaitable(value):
            value = await value
        return [float(item) for item in value]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | None]]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported similarity search options: {unknown}")
        filters = dict(filter or {})
        vector = await self._query_vector(query)
        if vector is not None:
            filters["vector"] = vector
        from ext.runtime.optional import RetrievalQuery

        results = await self.provider.search(
            RetrievalQuery(text=query, limit=k, filters=filters)
        )
        return [
            (
                Document(
                    page_content=(
                        item.content if isinstance(item.content, str) else str(item.content)
                    ),
                    id=item.id,
                    metadata={
                        **dict(item.metadata),
                        "title": item.title,
                        **({"source": item.uri} if item.uri else {}),
                        **({"score": item.score} if item.score is not None else {}),
                    },
                ),
                item.score,
            )
            for item in results
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            document
            for document, _score in await self.asimilarity_search_with_score(
                query, k, filter=filter, **kwargs
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        from ext.compat.base import _run_sync

        return _run_sync(self.asimilarity_search(query, k, filter=filter, **kwargs))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | None]]:
        from ext.compat.base import _run_sync

        return _run_sync(
            self.asimilarity_search_with_score(query, k, filter=filter, **kwargs)
        )

    def as_retriever(self, *, search_kwargs: Mapping[str, Any] | None = None):
        values = dict(search_kwargs or {})
        k = int(values.pop("k", 4))
        return ExternalVectorStoreRetriever(self, k=k, search_kwargs=values)


class Chroma(AgentRTVectorStore):
    backend = "chroma"


class PineconeVectorStore(AgentRTVectorStore):
    backend = "pinecone"


Pinecone = PineconeVectorStore


class QdrantVectorStore(AgentRTVectorStore):
    backend = "qdrant"


Qdrant = QdrantVectorStore


class Milvus(AgentRTVectorStore):
    backend = "milvus"


class WeaviateVectorStore(AgentRTVectorStore):
    backend = "weaviate"


Weaviate = WeaviateVectorStore


class _VectorRetriever:
    def __init__(self, store: MemoryVectorStore, k: int) -> None:
        self.store = store
        self.k = k

    async def ainvoke(self, query: str) -> list[Document]:
        return await self.store.asimilarity_search(query, self.k)

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await self.ainvoke(query)


class AgentRTRetriever:
    def __init__(
        self,
        provider: Any = None,
        *,
        registry: Any = None,
        provider_name: str | None = None,
        k: int = 4,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.provider_name = provider_name
        self.k = k
        self.filters = dict(filters or {})

    async def ainvoke(self, query: str) -> list[Document]:
        from ext.runtime.optional import RetrievalQuery

        request = RetrievalQuery(text=query, limit=self.k, filters=self.filters)
        if self.provider is not None:
            results = await self.provider.search(request)
        elif self.registry is not None and self.provider_name:
            results = await self.registry.search(self.provider_name, request)
        else:
            raise ValueError(
                "AgentRTRetriever requires provider or registry/provider_name"
            )
        return [
            Document(
                page_content=(
                    item.content if isinstance(item.content, str) else str(item.content)
                ),
                id=item.id,
                metadata={
                    **dict(item.metadata),
                    "title": item.title,
                    **({"source": item.uri} if item.uri else {}),
                    **({"score": item.score} if item.score is not None else {}),
                },
            )
            for item in results
        ]

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await self.ainvoke(query)


class ExternalVectorStoreRetriever:
    """Bridge an injected LangChain-like vector store without importing its vendor SDK."""

    def __init__(self, vector_store: Any, *, k: int = 4, search_kwargs=None) -> None:
        self.vector_store = vector_store
        self.k = k
        self.search_kwargs = dict(search_kwargs or {})

    async def ainvoke(self, query: str) -> list[Document]:
        kwargs = {"k": self.k, **self.search_kwargs}
        if hasattr(self.vector_store, "asimilarity_search"):
            result = self.vector_store.asimilarity_search(query, **kwargs)
        elif hasattr(self.vector_store, "similarity_search"):
            result = self.vector_store.similarity_search(query, **kwargs)
        else:
            raise TypeError(
                "vector_store must expose similarity_search() or asimilarity_search()"
            )
        if inspect.isawaitable(result):
            result = await result
        return list(result)

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await self.ainvoke(query)


def _document_key(document: Document) -> str:
    if document.id:
        return f"id:{document.id}"
    return json.dumps(
        [document.page_content, document.metadata], sort_keys=True, default=str
    )


class EnsembleRetriever:
    """Combine retrievers with weighted reciprocal-rank fusion."""

    def __init__(
        self,
        retrievers: Sequence[Any],
        weights: Sequence[float] | None = None,
        *,
        c: int = 60,
        k: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        if c < 1:
            raise ValueError("c must be at least 1")
        resolved = tuple(float(value) for value in (weights or [1.0] * len(retrievers)))
        if len(resolved) != len(retrievers):
            raise ValueError("weights must match retrievers")
        if any(value < 0 for value in resolved) or not any(resolved):
            raise ValueError("weights must contain at least one positive value")
        self.retrievers = tuple(retrievers)
        self.weights = resolved
        self.c = c
        self.k = k

    async def ainvoke(self, query: str) -> list[Document]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            if hasattr(retriever, "ainvoke"):
                ranked = await retriever.ainvoke(query)
            elif hasattr(retriever, "aget_relevant_documents"):
                ranked = await retriever.aget_relevant_documents(query)
            else:
                raise TypeError(
                    "retrievers must expose ainvoke() or aget_relevant_documents()"
                )
            for rank, document in enumerate(ranked, start=1):
                key = _document_key(document)
                documents.setdefault(key, document)
                scores[key] = scores.get(key, 0.0) + weight / (self.c + rank)
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        if self.k is not None:
            ordered = ordered[: self.k]
        return [documents[key] for key in ordered]

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await self.ainvoke(query)


class ContextualCompressionRetriever:
    """Retrieve documents and pass them through an injected compressor/reranker."""

    def __init__(self, base_retriever: Any, base_compressor: Any) -> None:
        self.base_retriever = base_retriever
        self.base_compressor = base_compressor

    async def ainvoke(self, query: str) -> list[Document]:
        if hasattr(self.base_retriever, "ainvoke"):
            documents = await self.base_retriever.ainvoke(query)
        else:
            documents = await self.base_retriever.aget_relevant_documents(query)
        if hasattr(self.base_compressor, "acompress_documents"):
            compressed = self.base_compressor.acompress_documents(documents, query)
        elif hasattr(self.base_compressor, "compress_documents"):
            compressed = self.base_compressor.compress_documents(documents, query)
        else:
            raise TypeError(
                "base_compressor must expose compress_documents() or acompress_documents()"
            )
        if inspect.isawaitable(compressed):
            compressed = await compressed
        return list(compressed)

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await self.ainvoke(query)


def _output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    return str(content)


class StrOutputParser:
    def parse(self, text: str) -> str:
        return text

    def invoke(self, value: Any) -> str:
        return self.parse(_output_text(value))

    async def ainvoke(self, value: Any) -> str:
        return self.invoke(value)


StringOutputParser = StrOutputParser


class JsonOutputParser:
    def parse(self, text: str) -> Any:
        return json.loads(text.strip())

    def invoke(self, value: Any) -> Any:
        return self.parse(_output_text(value))

    async def ainvoke(self, value: Any) -> Any:
        return self.invoke(value)


class PydanticOutputParser(JsonOutputParser):
    def __init__(self, pydantic_object: Any) -> None:
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> Any:
        value = super().parse(text)
        if hasattr(self.pydantic_object, "model_validate"):
            return self.pydantic_object.model_validate(value)
        if hasattr(self.pydantic_object, "parse_obj"):
            return self.pydantic_object.parse_obj(value)
        return self.pydantic_object(**value)


class PromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = template

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        return cls(template)

    def format(self, **values: Any) -> str:
        return safe_format(self.template, values)


class StuffDocumentsChain:
    def __init__(
        self, *, llm: Any, prompt: PromptTemplate, document_separator: str = "\n\n"
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.document_separator = document_separator

    async def ainvoke(self, inputs: Mapping[str, Any]) -> str:
        documents = inputs.get("context", ())
        context = self.document_separator.join(
            document.page_content for document in documents
        )
        rendered = self.prompt.format(**{**dict(inputs), "context": context})
        response = await self.llm.ainvoke(rendered)
        return str(getattr(response, "text", getattr(response, "content", response)))


def create_stuff_documents_chain(
    llm: Any,
    prompt: PromptTemplate,
    *,
    document_separator: str = "\n\n",
) -> StuffDocumentsChain:
    return StuffDocumentsChain(
        llm=llm, prompt=prompt, document_separator=document_separator
    )


class RetrievalChain:
    def __init__(self, retriever: Any, combine_docs_chain: StuffDocumentsChain) -> None:
        self.retriever = retriever
        self.combine_docs_chain = combine_docs_chain

    async def ainvoke(self, inputs: str | Mapping[str, Any]) -> dict[str, Any]:
        values = {"input": inputs} if isinstance(inputs, str) else dict(inputs)
        context = await self.retriever.ainvoke(str(values["input"]))
        answer = await self.combine_docs_chain.ainvoke({**values, "context": context})
        return {"context": context, "answer": answer}


class MapReduceDocumentsChain:
    def __init__(self, *, llm: Any, map_prompt: PromptTemplate, reduce_prompt: PromptTemplate) -> None:
        self.llm = llm
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt

    async def ainvoke(self, inputs: Mapping[str, Any]) -> str:
        documents = list(inputs.get("context", ()))
        mapped = []
        for document in documents:
            response = await self.llm.ainvoke(
                self.map_prompt.format(**{**dict(inputs), "context": document.page_content})
            )
            mapped.append(str(getattr(response, "text", getattr(response, "content", response))))
        reduced = self.reduce_prompt.format(
            **{**dict(inputs), "context": "\n\n".join(mapped)}
        )
        response = await self.llm.ainvoke(reduced)
        return str(getattr(response, "text", getattr(response, "content", response)))


class RefineDocumentsChain:
    def __init__(
        self,
        *,
        llm: Any,
        initial_prompt: PromptTemplate,
        refine_prompt: PromptTemplate,
    ) -> None:
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.refine_prompt = refine_prompt

    async def ainvoke(self, inputs: Mapping[str, Any]) -> str:
        documents = list(inputs.get("context", ()))
        if not documents:
            return ""
        first = self.initial_prompt.format(
            **{**dict(inputs), "context": documents[0].page_content}
        )
        response = await self.llm.ainvoke(first)
        answer = str(getattr(response, "text", getattr(response, "content", response)))
        for document in documents[1:]:
            rendered = self.refine_prompt.format(
                **{
                    **dict(inputs),
                    "context": document.page_content,
                    "existing_answer": answer,
                }
            )
            response = await self.llm.ainvoke(rendered)
            answer = str(getattr(response, "text", getattr(response, "content", response)))
        return answer


def create_map_reduce_documents_chain(
    llm: Any,
    map_prompt: PromptTemplate,
    reduce_prompt: PromptTemplate,
) -> MapReduceDocumentsChain:
    return MapReduceDocumentsChain(
        llm=llm, map_prompt=map_prompt, reduce_prompt=reduce_prompt
    )


def create_refine_documents_chain(
    llm: Any,
    initial_prompt: PromptTemplate,
    refine_prompt: PromptTemplate,
) -> RefineDocumentsChain:
    return RefineDocumentsChain(
        llm=llm, initial_prompt=initial_prompt, refine_prompt=refine_prompt
    )


def create_retrieval_chain(
    retriever: Any, combine_docs_chain: StuffDocumentsChain
) -> RetrievalChain:
    return RetrievalChain(retriever, combine_docs_chain)
