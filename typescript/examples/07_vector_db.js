const {
  VectorDBProviderRegistry,
  loadModel,
  registerPopularVectorDBBackends,
  vectorDBProviderFromEnvironment,
} = require("../dist/index.js");

async function main() {
  // OpenAI/OpenAI-compatible providers implement Agent RT's embedding contract.
  // Alternatively, pass a precomputed numeric vector in filters.vector.
  const embeddingProvider = await loadModel();
  if (typeof embeddingProvider.embed !== "function") {
    throw new Error(
      "This example needs an embedding-capable provider (for example OpenAI) " +
      "or a precomputed filters.vector.",
    );
  }

  const registry = new VectorDBProviderRegistry();
  registerPopularVectorDBBackends(registry, { embeddingProvider });

  // AGENT_RT_VECTOR_DB chooses chroma|milvus|pinecone|qdrant|weaviate.
  // Changing only the environment selects a different registered backend.
  const provider = vectorDBProviderFromEnvironment(registry);

  const text = process.env.AGENT_RT_VECTOR_DB_QUERY || "What is Agent RT?";
  const limit = Number(process.env.AGENT_RT_VECTOR_DB_LIMIT || "5");
  const results = await provider.search({ text, limit });

  console.log(`backend=${process.env.AGENT_RT_VECTOR_DB} results=${results.length}`);
  for (const result of results) {
    console.log(`- ${String(result.score ?? "").padStart(8)}  ${result.title}`);
    if (result.uri) console.log(`  ${result.uri}`);
    console.log(`  ${String(result.content)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
