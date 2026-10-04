const http = require("node:http");
const { VectorMock: AimockVectorMock } = require("@copilotkit/aimock");

class VectorMock extends AimockVectorMock {
  constructor(options = {}) {
    super(options);
    this._options = options;
    this._queryHandlers = new Map();
    this._collections = new Map();
    this._requests = [];
    this._server = null;
    this.url = null;
    this._primaryCollection = null;
  }

  addCollection(name, options) {
    this._collections.set(name, Number(options.dimension));
    if (this._primaryCollection === null && name !== "default") {
      this._primaryCollection = name;
      super.addCollection("default", options);
    }
    super.addCollection(name, options);
    return this;
  }

  upsert(collection, vectors) {
    super.upsert(collection, vectors);
    if (this._primaryCollection === collection) {
      super.upsert("default", vectors);
    }
    return this;
  }

  deleteCollection(name) {
    super.deleteCollection(name);
    this._collections.delete(name);
    this._queryHandlers.delete(name);
    if (this._primaryCollection === name) {
      super.deleteCollection("default");
      this._primaryCollection = null;
    }
    return this;
  }

  health() {
    return { status: "ok", collections: this._collections.size };
  }

  onQuery(collection, results) {
    this._queryHandlers.set(collection, results);
    if (this._primaryCollection === collection) {
      super.onQuery("default", results);
    }
    super.onQuery(collection, results);
    return this;
  }

  getRequests() {
    return [...this._requests];
  }

  reset() {
    super.reset();
    this._queryHandlers.clear();
    this._collections.clear();
    this._requests = [];
    this._primaryCollection = null;
    return this;
  }

  async start() {
    if (this._server) throw new Error("Server already started");
    const host = this._options.host ?? "127.0.0.1";
    const port = this._options.port ?? 0;

    this._server = http.createServer(async (req, res) => {
      const url = new URL(req.url ?? "/", `http://${host}`);
      const path = url.pathname;
      this._requests.push({ method: req.method ?? "GET", path, headers: req.headers });

      if (isAgentRTExtraRoute(path)) {
        const body = await readJson(req, res);
        if (body === null) return;
        this._requests[this._requests.length - 1].body = body;
        this._handleAgentRTExtra(req, res, path, body);
        return;
      }

      const handled = await super.handleRequest(req, res, path);
      if (!handled) {
        jsonResponse(res, 404, { error: "not found" });
      }
    });

    await new Promise((resolve, reject) => {
      this._server.once("error", reject);
      this._server.listen(port, host, resolve);
    });
    const address = this._server.address();
    if (!address || typeof address === "string") throw new Error("VectorMock failed to bind");
    this.url = `http://${host}:${address.port}`;
    return this.url;
  }

  async stop() {
    if (!this._server) throw new Error("Server not started");
    const server = this._server;
    this._server = null;
    this.url = null;
    await new Promise((resolve, reject) => {
      server.close((error) => error ? reject(error) : resolve());
    });
  }

  _resolve(collection, query) {
    const handler = this._queryHandlers.get(collection);
    if (!handler) return [];
    const results = typeof handler === "function" ? handler(query) : handler;
    return [...results].slice(0, query.topK ?? 10);
  }

  _handleAgentRTExtra(_req, res, path, body) {
    if (path === "/v2/vectordb/entities/search") {
      const collection = String(body.collectionName ?? "");
      const results = this._resolve(collection, {
        collection,
        vector: Array.isArray(body.data?.[0]) ? body.data[0] : undefined,
        topK: Number(body.limit ?? 10),
        filter: body.filter,
      });
      jsonResponse(res, 200, {
        data: results.map((result) => ({
          id: result.id,
          score: result.score,
          ...(result.metadata ?? {}),
        })),
      });
      return;
    }

    if (path === "/v1/graphql") {
      const queryText = String(body.query ?? "");
      const collection = /Get\s*\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(/.exec(queryText)?.[1] ?? "";
      const limit = Number(/limit:\s*(\d+)/.exec(queryText)?.[1] ?? 10);
      const vectorText = /vector:\s*\[([^\]]*)\]/.exec(queryText)?.[1];
      const vector = vectorText
        ? vectorText.split(",").map((value) => Number(value.trim())).filter(Number.isFinite)
        : undefined;
      const results = this._resolve(collection, {
        collection,
        vector,
        topK: limit,
        filter: undefined,
      });
      jsonResponse(res, 200, {
        data: {
          Get: {
            [collection]: results.map((result) => ({
              ...(result.metadata ?? {}),
              _additional: { id: result.id, certainty: result.score },
            })),
          },
        },
      });
      return;
    }

    const chroma = /^\/api\/v2\/tenants\/[^/]+\/databases\/[^/]+\/collections\/([^/]+)\/query$/.exec(path);
    if (chroma) {
      const collection = decodeURIComponent(chroma[1]);
      const results = this._resolve(collection, {
        collection,
        vector: Array.isArray(body.query_embeddings?.[0]) ? body.query_embeddings[0] : undefined,
        topK: Number(body.n_results ?? 10),
        filter: body.where,
      });
      jsonResponse(res, 200, {
        ids: [results.map((result) => result.id)],
        documents: [results.map((result) => content(result.metadata))],
        metadatas: [results.map((result) => result.metadata ?? {})],
        distances: [results.map((result) => result.score > 0 ? Math.max(0, (1 / result.score) - 1) : 1)],
        uris: [results.map((result) => String(result.metadata?.url ?? ""))],
      });
      return;
    }

    jsonResponse(res, 404, { error: "not found" });
  }
}

function isAgentRTExtraRoute(path) {
  return path === "/v2/vectordb/entities/search"
    || path === "/v1/graphql"
    || /^\/api\/v2\/tenants\/[^/]+\/databases\/[^/]+\/collections\/[^/]+\/query$/.test(path);
}

async function readJson(req, res) {
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString() || "{}");
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
      jsonResponse(res, 400, { error: "request body must be a JSON object" });
      return null;
    }
    return parsed;
  } catch (error) {
    jsonResponse(res, 400, { error: String(error) });
    return null;
  }
}

function jsonResponse(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function content(metadata) {
  return String(metadata?.text ?? metadata?.content ?? "");
}

module.exports = { VectorMock };
