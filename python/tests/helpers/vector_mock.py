from __future__ import annotations

import json
import re
import threading
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class VectorQuery:
    vector: list[float] | None
    top_k: int
    filters: Any
    collection: str


@dataclass(frozen=True)
class VectorEntry:
    id: str
    values: list[float]
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    metadata: Mapping[str, Any] | None = None
    values: list[float] | None = None


class VectorMock:
    """Dependency-free local vector DB mock modeled after aimock's VectorMock fixture API.

    It supports the five HTTP query shapes used by Agent RT: Chroma v2, Milvus,
    Pinecone, Qdrant, and Weaviate. The mock is intended for tests only.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.collections: dict[str, int] = {}
        self.vectors: dict[str, dict[str, VectorEntry]] = {}
        self.query_handlers: dict[
            str, Sequence[QueryResult] | Callable[[VectorQuery], Sequence[QueryResult]]
        ] = {}
        self.requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def add_collection(self, name: str, *, dimension: int) -> VectorMock:
        self.collections[name] = dimension
        self.vectors[name] = {}
        return self

    def addCollection(self, name: str, options: Mapping[str, Any]) -> VectorMock:
        return self.add_collection(name, dimension=int(options["dimension"]))

    def upsert(self, collection: str, vectors: Sequence[VectorEntry]) -> VectorMock:
        entries = list(vectors)
        if collection not in self.collections:
            dimension = len(entries[0].values) if entries else 0
            self.add_collection(collection, dimension=dimension)
        bucket = self.vectors.setdefault(collection, {})
        for entry in entries:
            bucket[entry.id] = entry
        return self

    def delete_collection(self, name: str) -> VectorMock:
        self.collections.pop(name, None)
        self.vectors.pop(name, None)
        self.query_handlers.pop(name, None)
        return self

    def deleteCollection(self, name: str) -> VectorMock:
        return self.delete_collection(name)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "collections": len(self.collections)}

    def on_query(
        self,
        collection: str,
        results: Sequence[QueryResult] | Callable[[VectorQuery], Sequence[QueryResult]],
    ) -> VectorMock:
        self.query_handlers[collection] = results
        return self

    def onQuery(
        self,
        collection: str,
        results: Sequence[QueryResult] | Callable[[VectorQuery], Sequence[QueryResult]],
    ) -> VectorMock:
        return self.on_query(collection, results)

    def reset(self) -> VectorMock:
        self.collections.clear()
        self.vectors.clear()
        self.query_handlers.clear()
        self.requests.clear()
        return self

    def get_requests(self) -> list[dict[str, Any]]:
        return list(self.requests)

    def getRequests(self) -> list[dict[str, Any]]:
        return self.get_requests()

    def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("VectorMock server already started")

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                owner._handle(self)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_port = int(self._server.server_address[1])
        self.url = f"http://{self.host}:{actual_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is None:
            raise RuntimeError("VectorMock server not started")
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        self.url = None
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def _read_json(self, request: BaseHTTPRequestHandler) -> dict[str, Any]:
        length = int(request.headers.get("content-length", "0") or 0)
        raw = request.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("request body must be a JSON object")
        return value

    def _write_json(
        self, request: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]
    ) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        request.send_response(status)
        request.send_header("content-type", "application/json")
        request.send_header("content-length", str(len(encoded)))
        request.end_headers()
        request.wfile.write(encoded)

    def _resolve(
        self,
        collection: str,
        *,
        vector: list[float] | None,
        top_k: int,
        filters: Any,
    ) -> list[QueryResult]:
        handler = self.query_handlers.get(collection)
        if handler is None:
            return []
        query = VectorQuery(
            vector=vector,
            top_k=top_k,
            filters=filters,
            collection=collection,
        )
        results = handler(query) if callable(handler) else handler
        return list(results)[:top_k]

    def _pinecone_collection(self, body: Mapping[str, Any]) -> str:
        namespace = body.get("namespace")
        if isinstance(namespace, str) and namespace:
            return namespace
        if "default" in self.query_handlers:
            return "default"
        if len(self.query_handlers) == 1:
            return next(iter(self.query_handlers))
        return "default"

    def _handle(self, request: BaseHTTPRequestHandler) -> None:
        try:
            body = self._read_json(request)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            self._write_json(request, 400, {"error": str(exc)})
            return

        path = urllib.parse.urlsplit(request.path).path
        self.requests.append(
            {
                "method": "POST",
                "path": path,
                "headers": dict(request.headers.items()),
                "body": body,
            }
        )

        qdrant = re.fullmatch(r"/collections/([^/]+)/points/search", path)
        if qdrant:
            collection = urllib.parse.unquote(qdrant.group(1))
            results = self._resolve(
                collection,
                vector=_number_list(body.get("vector")),
                top_k=int(body.get("limit", 10)),
                filters=body.get("filter"),
            )
            self._write_json(
                request,
                200,
                {
                    "result": [
                        {
                            "id": result.id,
                            "score": result.score,
                            "payload": dict(result.metadata or {}),
                        }
                        for result in results
                    ]
                },
            )
            return

        if path == "/query":
            collection = self._pinecone_collection(body)
            results = self._resolve(
                collection,
                vector=_number_list(body.get("vector")),
                top_k=int(body.get("topK", 10)),
                filters=body.get("filter"),
            )
            self._write_json(
                request,
                200,
                {
                    "matches": [
                        {
                            "id": result.id,
                            "score": result.score,
                            "metadata": dict(result.metadata or {}),
                        }
                        for result in results
                    ]
                },
            )
            return

        if path == "/v2/vectordb/entities/search":
            collection = str(body.get("collectionName", ""))
            data = body.get("data")
            vector = _number_list(data[0]) if isinstance(data, list) and data else None
            results = self._resolve(
                collection,
                vector=vector,
                top_k=int(body.get("limit", 10)),
                filters=body.get("filter"),
            )
            self._write_json(
                request,
                200,
                {
                    "data": [
                        {
                            "id": result.id,
                            "score": result.score,
                            **dict(result.metadata or {}),
                        }
                        for result in results
                    ]
                },
            )
            return

        if path == "/v1/graphql":
            graph_query = str(body.get("query", ""))
            match = re.search(r"Get\s*\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", graph_query)
            collection = match.group(1) if match else ""
            limit_match = re.search(r"limit:\s*(\d+)", graph_query)
            limit = int(limit_match.group(1)) if limit_match else 10
            vector_match = re.search(r"vector:\s*\[([^\]]*)\]", graph_query)
            vector = (
                [
                    float(item.strip())
                    for item in vector_match.group(1).split(",")
                    if item.strip()
                ]
                if vector_match
                else None
            )
            results = self._resolve(
                collection,
                vector=vector,
                top_k=limit,
                filters=None,
            )
            items = []
            for result in results:
                metadata = dict(result.metadata or {})
                items.append(
                    {
                        **metadata,
                        "_additional": {
                            "id": result.id,
                            "certainty": result.score,
                        },
                    }
                )
            self._write_json(request, 200, {"data": {"Get": {collection: items}}})
            return

        chroma = re.fullmatch(
            r"/api/v2/tenants/[^/]+/databases/[^/]+/collections/([^/]+)/query",
            path,
        )
        if chroma:
            collection = urllib.parse.unquote(chroma.group(1))
            embeddings = body.get("query_embeddings")
            vector = (
                _number_list(embeddings[0])
                if isinstance(embeddings, list) and embeddings
                else None
            )
            results = self._resolve(
                collection,
                vector=vector,
                top_k=int(body.get("n_results", 10)),
                filters=body.get("where"),
            )
            self._write_json(
                request,
                200,
                {
                    "ids": [[result.id for result in results]],
                    "documents": [[_content(result.metadata) for result in results]],
                    "metadatas": [[dict(result.metadata or {}) for result in results]],
                    "distances": [
                        [
                            (
                                max(0.0, (1.0 / result.score) - 1.0)
                                if result.score > 0
                                else 1.0
                            )
                            for result in results
                        ]
                    ],
                    "uris": [
                        [
                            str((result.metadata or {}).get("url", ""))
                            for result in results
                        ]
                    ],
                },
            )
            return

        self._write_json(request, 404, {"error": "not found"})


def _number_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def _content(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""
    value = metadata.get("text", metadata.get("content", ""))
    return str(value)
