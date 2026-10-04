const {
  AgentLoop,
  loadModel,
} = require("../dist/index.js");
const { makeAgent, message, responseText } = require("./_common.js");

async function main() {
  const provider = await loadModel();
  const agent = makeAgent(
    "basic",
    "Answer clearly and concisely. Do not use tools.",
  );

  const result = await new AgentLoop(provider).run(
    agent,
    [message("user", "Explain what an agent harness does in two sentences.")],
  );

  console.log(responseText(result.finalResponse));
  console.log(`termination=${result.terminationReason} turns=${result.turns}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
