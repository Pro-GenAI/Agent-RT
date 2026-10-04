const {
  AgentLoop,
  ToolRegistry,
  loadModel,
} = require("../dist/index.js");
const { makeAgent, message, responseText } = require("./_common.js");

async function main() {
  const registry = new ToolRegistry();

  registry.register(
    {
      name: "get_weather",
      description: "Get the current weather for a city.",
      inputSchema: {
        type: "object",
        required: ["city"],
        properties: { city: { type: "string" } },
        additionalProperties: false,
      },
      sideEffect: "read",
    },
    {
      handler: async (args) => ({
        city: args.city,
        temperature_c: 28,
        condition: "clear",
      }),
    },
  );

  const provider = await loadModel();
  const agent = makeAgent(
    "weather-assistant",
    "Use get_weather when the user asks about weather. Summarize the tool result.",
  );

  const result = await new AgentLoop(provider, undefined, registry).run(
    agent,
    [message("user", "What is the weather in Chennai?")],
  );

  console.log(responseText(result.finalResponse));
  console.log(`toolCalls=${result.toolCalls}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
