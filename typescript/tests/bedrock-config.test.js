const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {
  parseConfig,
  vectorStoreRegistryFromConfig,
  loadConfig,
  loadModel,
} = require('../dist/index.js');

const YAML = `
model_list:
  - model_name: claude-sonnet-5
    litellm_params: {model: anthropic/claude-sonnet-5}
vector_store_registry:
  - vector_store_name: docs-kb
    litellm_params:
      custom_llm_provider: bedrock
      vector_store_id: T37J8R4WTM
      options: {region: us-west-2}
`;

test('Bedrock store supports name/id aliases and native Retrieve semantics', async () => {
  const config = await parseConfig(YAML, { environment: {} });
  const calls = [];
  let created = 0;
  const registry = vectorStoreRegistryFromConfig(config, {
    bedrockClientFactory: (store) => {
      assert.equal(store.options.region, 'us-west-2');
      created += 1;
      return {
        async retrieve(params) {
          calls.push(params);
          return { retrievalResults: [{
            content: { text: 'LiteLLM is a gateway.' },
            location: { webLocation: { url: 'https://www.litellm.com/docs' } },
            score: 0.8,
          }] };
        },
      };
    },
  });
  assert.equal(created, 0);
  assert.equal(registry.providers.get('docs-kb'), registry.providers.get('T37J8R4WTM'));
  const rows = await registry.search('T37J8R4WTM', { text: 'LiteLLM', limit: 3 });
  assert.equal(created, 1);
  assert.deepEqual(calls, [{
    knowledgeBaseId: 'T37J8R4WTM',
    retrievalQuery: { text: 'LiteLLM' },
    retrievalConfiguration: { vectorSearchConfiguration: { numberOfResults: 3 } },
  }]);
  assert.equal(rows[0].content, 'LiteLLM is a gateway.');
  assert.equal(rows[0].uri, 'https://www.litellm.com/docs');
});

test('Unused vector-store secrets are resolved only when the store is queried', async () => {
  const config = await parseConfig(YAML + `
  - vector_store_name: unused
    litellm_params:
      custom_llm_provider: qdrant
      vector_store_id: unused-id
      collection: docs
      api_base: os.environ/ABSENT_VECTOR_URL
`, { environment: {} });
  const registry = vectorStoreRegistryFromConfig(config, {
    bedrockClientFactory: () => ({ retrieve: async () => ({ retrievalResults: [] }) }),
  });
  await assert.rejects(registry.search('unused-id', { text: 'x' }), /ABSENT_VECTOR_URL/);
  assert.deepEqual(await registry.search('docs-kb', { text: 'x' }), []);
});

test('Validation uses injected provider without creating external SDK clients', async () => {
  const config = await parseConfig(YAML, { environment: {} });
  let validations = 0;
  const provider = await loadModel(config, {
    validate: true,
    providerFactory: () => ({
      complete: async () => ({}),
      validate: async () => { validations += 1; },
    }),
  });
  assert.equal(validations, 1);
  assert.equal(provider.modelNames[0], 'claude-sonnet-5');

  let queried = 0;
  await loadModel(config, {
    validate: true,
    validationClientFactory: () => ({
      models: { list: async () => {
        queried += 1;
        return { data: [{ id: 'claude-sonnet-5' }] };
      } },
    }),
  });
  assert.equal(queried, 1);
});

test('Configuration loading works without process.getBuiltinModule', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-rt-node20-'));
  const file = path.join(dir, 'agent-rt.yaml');
  fs.writeFileSync(file, YAML);
  const original = process.getBuiltinModule;
  try {
    process.getBuiltinModule = undefined;
    const loaded = await loadConfig(file, { environment: {} });
    assert.equal(loaded.vectorStores[0].provider, 'bedrock');
  } finally {
    process.getBuiltinModule = original;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
