const test = require("node:test");
const assert = require("node:assert/strict");

const {
  EnvironmentVectorDBProvider,
} = require("../dist/index.js");
const { VectorMock } = require("./vector-mock.js");

const BACKENDS = ["chroma", "milvus", "pinecone", "qdrant", "weaviate"];

test("VectorMock preserves the aimock fixture lifecycle", () => {
  const mock = new VectorMock();
  assert.deepEqual(mock.health(), { status: "ok", collections: 0 });

  mock
    .addCollection("docs", { dimension: 3 })
    .upsert("docs", [{
      id: "doc-1",
      values: [0.1, 0.2, 0.3],
      metadata: { text: "stored fixture" },
    }]);
  assert.deepEqual(mock.health(), { status: "ok", collections: 1 });

  mock.deleteCollection("docs");
  assert.deepEqual(mock.health(), { status: "ok", collections: 0 });

  mock.addCollection("docs", { dimension: 3 }).reset();
  assert.deepEqual(mock.health(), { status: "ok", collections: 0 });
});

function collectionFor(backend) {
  return backend === "weaviate" ? "Docs" : "docs";
}

function resultFor(backend) {
  return {
    id: `${backend}-1`,
    score: 0.9,
    metadata: {
      text: `${backend} mock hit`,
      title: `${backend} result`,
      url: `https://example.test/${backend}`,
    },
  };
}

for (const backend of BACKENDS) {
  test(`core vector DB ${backend} works against VectorMock`, async () => {
    const collection = collectionFor(backend);
    const seen = [];
    const mock = new VectorMock()
      .addCollection(collection, { dimension: 3 })
      .onQuery(collection, (query) => {
        seen.push(query);
        return [resultFor(backend)];
      });
    const url = await mock.start();

    try {
      const provider = EnvironmentVectorDBProvider.fromEnvironment({
        AGENT_RT_VECTOR_DB: backend,
        AGENT_RT_VECTOR_DB_COLLECTION: collection,
        AGENT_RT_VECTOR_DB_URL: url,
      });
      const results = await provider.search({
        text: "agent runtime",
        limit: 1,
        filters: { vector: [0.1, 0.2, 0.3] },
      });
      assert.equal(results.length, 1);
      assert.equal(results[0].id, `${backend}-1`);
      assert.equal(results[0].content, `${backend} mock hit`);
      assert.equal(results[0].metadata.provider, backend);
      assert.deepEqual(seen[0].vector, [0.1, 0.2, 0.3]);
      assert.equal(seen[0].topK, 1);
    } finally {
      await mock.stop();
    }
  });

  test(`LangChain ${backend} vector store works against VectorMock`, async () => {
    const collection = collectionFor(backend);
    const seen = [];
    const mock = new VectorMock()
      .addCollection(collection, { dimension: 3 })
      .onQuery(collection, (query) => {
        seen.push(query);
        return [resultFor(backend)];
      });
    const url = await mock.start();

    const modules = await import(`agent-rt/@langchain/${backend}`);
    const classes = {
      chroma: modules.Chroma,
      milvus: modules.Milvus,
      pinecone: modules.PineconeVectorStore,
      qdrant: modules.QdrantVectorStore,
      weaviate: modules.WeaviateVectorStore,
    };

    try {
      const store = new classes[backend](
        {
          embedQuery: async (text) => {
            assert.equal(text, "agent runtime");
            return [0.1, 0.2, 0.3];
          },
          embedDocuments: async () => [],
        },
        {
          environment: {
            AGENT_RT_VECTOR_DB_URL: url,
            AGENT_RT_VECTOR_DB_COLLECTION: collection,
          },
        },
      );
      const documents = await store.similaritySearch("agent runtime", 1);
      assert.equal(documents.length, 1);
      assert.equal(documents[0].pageContent, `${backend} mock hit`);
      assert.equal(documents[0].metadata.provider, backend);
      assert.deepEqual(seen[0].vector, [0.1, 0.2, 0.3]);
      assert.equal(seen[0].topK, 1);
    } finally {
      await mock.stop();
    }
  });

  test(`LlamaIndex ${backend} vector store works against VectorMock`, async () => {
    const collection = collectionFor(backend);
    const seen = [];
    const mock = new VectorMock()
      .addCollection(collection, { dimension: 3 })
      .onQuery(collection, (query) => {
        seen.push(query);
        return [resultFor(backend)];
      });
    const url = await mock.start();

    const modules = await import(`agent-rt/@llamaindex/${backend}`);
    const classes = {
      chroma: modules.ChromaVectorStore,
      milvus: modules.MilvusVectorStore,
      pinecone: modules.PineconeVectorStore,
      qdrant: modules.QdrantVectorStore,
      weaviate: modules.WeaviateVectorStore,
    };

    try {
      const store = new classes[backend]({
        environment: {
          AGENT_RT_VECTOR_DB_URL: url,
          AGENT_RT_VECTOR_DB_COLLECTION: collection,
        },
      });
      const nodes = await store.query({
        queryEmbedding: [0.1, 0.2, 0.3],
        queryStr: "agent runtime",
        similarityTopK: 1,
      });
      assert.equal(nodes.length, 1);
      assert.equal(nodes[0].node.text, `${backend} mock hit`);
      assert.equal(nodes[0].node.metadata.provider, backend);
      assert.ok(Math.abs(nodes[0].score - 0.9) < 1e-9);
      assert.deepEqual(seen[0].vector, [0.1, 0.2, 0.3]);
      assert.equal(seen[0].topK, 1);
    } finally {
      await mock.stop();
    }
  });
}
