const {
  AgentLoop,
  loadModel,
} = require("../dist/index.js");
const { makeAgent, message } = require("./_common.js");

async function main() {
  const provider = await loadModel();
  const agent = makeAgent("streaming", "Answer in a short paragraph.");

  const result = await new AgentLoop(provider).runStreaming(
    agent,
    [message("user", "Give me three practical uses for an agent harness.")],
    (event) => {
      if (event.type === "text_delta" && event.text) {
        process.stdout.write(event.text);
      }
    },
  );

  process.stdout.write("\n");
  console.log(`termination=${result.terminationReason} tokens=${result.totalTokens}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
