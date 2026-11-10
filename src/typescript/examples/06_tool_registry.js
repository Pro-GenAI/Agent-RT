const { ToolRegistry } = require("../dist/index.js");

async function main() {
  const registry = new ToolRegistry();

  registry.register(
    {
      name: "search",
      description: "Search a small demo catalog.",
      inputSchema: {
        type: "object",
        required: ["query"],
        properties: { query: { type: "string" } },
        additionalProperties: false,
      },
      sideEffect: "read",
    },
    {
      namespace: "catalog",
      handler: async (args) => ({
        query: args.query,
        matches: ["alpha", "beta"],
      }),
    },
  );

  console.log("visible tools:", registry.definitions().map((tool) => tool.name));

  const result = await registry.execute({
    id: "demo-1",
    name: "catalog.search",
    arguments: { query: "agent" },
  });
  console.log("result:", result);

  registry.disable("catalog.search");
  console.log("after disable:", registry.definitions().map((tool) => tool.name));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
