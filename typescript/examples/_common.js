const { loadConfig, loadModel } = require("../dist/index.js");

let configPromise;

/**
 * Load agent-rt.yaml from the current folder or ~/.agent-rt/config.yaml.
 * See agent-rt.example.yaml at the repository root for a starting point.
 */
function exampleConfig() {
  configPromise ??= loadConfig();
  return configPromise;
}

async function exampleProvider() {
  return loadModel(await exampleConfig());
}

/**
 * Return the default model name from the config used by the examples.
 */
async function configuredModel() {
  const config = await exampleConfig();
  const name = config.settings.default_model ?? config.models[0]?.modelName;
  if (!name) throw new Error("Add a model to model_list in your agent-rt.yaml.");
  return name;
}

async function makeAgent(name, instructions) {
  return {
    name,
    instructions,
    model: { model: await configuredModel() },
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
  exampleConfig,
  exampleProvider,
  makeAgent,
  message,
  responseText,
};
