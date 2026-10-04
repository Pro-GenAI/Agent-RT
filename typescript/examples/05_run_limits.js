const {
  AgentLoop,
  loadModel,
} = require("../dist/index.js");
const { makeAgent, message, responseText } = require("./_common.js");

async function main() {
  const provider = await loadModel();
  const agent = makeAgent("bounded", "Answer briefly.");

  const result = await new AgentLoop(provider).run(
    agent,
    [message("user", "Name two reasons to put explicit limits around an agent loop.")],
    {
      maxTurns: 1,
      maxToolCalls: 0,
      timeoutMs: 30_000,
      maxTotalTokens: 500,
    },
  );

  console.log(responseText(result.finalResponse));
  console.log(
    `termination=${result.terminationReason} turns=${result.turns} ` +
    `tools=${result.toolCalls} tokens=${result.totalTokens}`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
