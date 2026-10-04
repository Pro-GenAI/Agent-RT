const {
  AgentLoop,
  loadModel,
} = require("../dist/index.js");
const { configuredModel, message } = require("./_common.js");

async function main() {
  const provider = await loadModel();
  const agent = {
    name: "extractor",
    instructions: "Extract the requested fields and return only valid JSON.",
    model: { model: configuredModel() },
    output: {
      format: "json",
      schema: {
        type: "object",
        required: ["language", "difficulty", "topics"],
        properties: {
          language: { type: "string" },
          difficulty: { type: "string" },
          topics: { type: "array", items: { type: "string" } },
        },
        additionalProperties: false,
      },
      maxRepairAttempts: 1,
    },
  };

  const result = await new AgentLoop(provider).run(
    agent,
    [message("user", "Create a beginner Python learning plan covering functions and classes.")],
  );

  console.log(JSON.stringify(result.structuredOutput, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
