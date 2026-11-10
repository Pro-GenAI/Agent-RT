const { vectorStoreRegistryFromConfig } = require("../dist/index.js");
const { exampleConfig, exampleProvider } = require("./_common.js");

async function main() {
  // Every vector_store_registry entry in agent-rt.yaml is registered under its
  // name and id; each store embeds queries with its configured embedding_model.
  const config = await exampleConfig();
  if (!config.vectorStores.length) {
    throw new Error("Add a vector_store_registry entry to your agent-rt.yaml.");
  }
  const registry = vectorStoreRegistryFromConfig(config, {
    embeddingProvider: await exampleProvider(),
  });

  const store = process.argv[2] || config.vectorStores[0].name;
  const text = process.argv[3] || "What is Agent RT?";
  const results = await registry.search(store, { text, limit: 5 });

  console.log(`vector_store=${store} results=${results.length}`);
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
