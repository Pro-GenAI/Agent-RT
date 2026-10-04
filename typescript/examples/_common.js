/**
 * Return the configured model name used by the examples.
 */
function configuredModel() {
  const model = process.env.OPENAI_MODEL || process.env.ANTHROPIC_MODEL;
  if (!model) {
    throw new Error("Set OPENAI_MODEL or ANTHROPIC_MODEL before running this example.");
  }
  return model;
}

function makeAgent(name, instructions) {
  return {
    name,
    instructions,
    model: { model: configuredModel() },
  };
}

function message(role, text) {
  return {
    role,
    content: [{ type: "text", text }],
  };
}

function responseText(response) {
  if (!response) return "";
  return (response.message.content ?? [])
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("");
}

module.exports = {
  configuredModel,
  makeAgent,
  message,
  responseText,
};
