const test = require('node:test');
const assert = require('node:assert/strict');

const { MemorySource, resolveAgentMemory } = require('../dist/index.js');
const {
	AIMessage,
	Document,
	HumanMessage,
	InMemoryStore,
} = require('../dist/ext/compat/langchain.js');
const {
	Memory,
	NodeWithScore,
	TextNode,
} = require('../dist/ext/compat/llamaindex.js');
const { Agent, run } = require('../dist/ext/compat/openai_agents.js');

const parts = (items) =>
	items.map((item) => item.content[0].text ?? item.content[0].data);

class FakeLangChainRetriever {
	async invoke(query) {
		return [
			new Document({ pageContent: 'doc for ' + query }),
			new Document({ pageContent: 'tagged', metadata: { source: 'wiki' } }),
		];
	}
}

class FakeLlamaIndexRetriever {
	async retrieve(query) {
		return [new NodeWithScore(new TextNode({ text: 'node for ' + query }), 0.9)];
	}
}

class FakeChatHistory {
	// @langchain/core BaseChatMessageHistory shape
	async getMessages() {
		return [new HumanMessage('hi'), new AIMessage('hello'), new HumanMessage('bye')];
	}
	async addMessage() {}
}

class FakeMem0 {
	calls = [];
	async add() {}
	async getAll() {}
	async search(query, options) {
		this.calls.push([query, options]);
		return { results: [{ id: '1', memory: 'likes tea' }] };
	}
}

class FakeSession {
	// @openai/agents Session shape
	async getItems(limit) {
		return [{ role: 'user', content: 'earlier' }].slice(-limit);
	}
	async addItems() {}
}

test('memory sources from other frameworks', async () => {
	const store = new InMemoryStore();
	store.put(['users', 'u1'], 'prefs', { units: 'metric' });
	store.put(['users', 'u2'], 'prefs', { units: 'imperial' });
	const llamaMemory = new Memory();
	llamaMemory.put({ role: 'user', content: 'remember blue' });
	const mem0 = new FakeMem0();

	const items = await resolveAgentMemory(
		[
			new FakeLangChainRetriever(),
			new FakeLlamaIndexRetriever(),
			new MemorySource(store, { searchOptions: { namespace: ['users', 'u1'] } }),
			new MemorySource(new FakeChatHistory(), { limit: 2 }),
			llamaMemory,
			new MemorySource(mem0, { searchOptions: { filters: { user_id: 'u1' } } }),
			new FakeSession(),
		],
		'colors',
	);

	assert.deepEqual(parts(items), [
		'doc for colors',
		{ content: 'tagged', metadata: { source: 'wiki' } },
		'node for colors',
		{ key: 'prefs', value: { units: 'metric' } },
		{ content: 'hello', role: 'assistant' },
		{ content: 'bye', role: 'user' },
		{ content: 'remember blue', role: 'user' },
		{ id: '1', memory: 'likes tea' },
		{ role: 'user', content: 'earlier' },
	]);
	assert.deepEqual(new Set(items.map((item) => item.trust)), new Set(['untrusted']));
	assert.deepEqual(mem0.calls, [['colors', { topK: 5, filters: { user_id: 'u1' } }]]);
});

test('upstream LangChain message shape uses its type as the role', async () => {
	const message = { type: 'ai', content: 'hello', lc_kwargs: {} };
	const items = await resolveAgentMemory([message]);
	assert.deepEqual(parts(items), [{ content: 'hello', role: 'ai' }]);
	assert.equal(items[0].trust, 'trusted');
});

test('OpenAI Agents compat agent passes memory to the run', async () => {
	const requests = [];
	const provider = {
		name: 'fake',
		complete: async (request) => {
			requests.push(request);
			return { message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } };
		},
	};
	const agent = new Agent({
		name: 'Helper',
		model: 'test-model',
		provider,
		memory: ['The user is called Ada.', new FakeSession()],
	});
	await run(agent, 'who am I?');

	const context = requests[0].messages[0].content[0].data.context;
	assert.equal(context.retrievedData[0].content[0].text, 'The user is called Ada.');
	const untrusted = requests[0].messages[1].content[0].data.untrustedContext;
	assert.deepEqual(untrusted.retrievedData[0].content[0].data, {
		role: 'user',
		content: 'earlier',
	});
});

const { loadMemory, MemoryBackend, parseConfig } = require('../dist/index.js');

const recorder = (result) => {
	const fn = async (...args) => {
		fn.calls.push(args);
		return result;
	};
	fn.calls = [];
	return fn;
};

test('loadMemory defaults to mem0 and works as a memory source', async () => {
	const client = {
		search: recorder({ results: [{ memory: 'likes tea' }] }),
		add: recorder([]),
	};
	const imported = [];
	const memory = await loadMemory(undefined, {
		options: { user_id: 'u1' },
		importer: async (name) => {
			imported.push(name);
			return { Memory: function () { return client; } };
		},
	});
	assert.ok(memory instanceof MemoryBackend);
	assert.deepEqual(imported, ['mem0ai/oss']);
	assert.deepEqual(await memory.recall('drinks?', 3), ['likes tea']);
	assert.deepEqual(client.search.calls, [
		['drinks?', { filters: { user_id: 'u1' }, topK: 3 }],
	]);
	await memory.remember([
		{ role: 'system', content: 'secret instructions' },
		{ role: 'user', content: [{ type: 'text', text: 'hi' }] },
		{ role: 'assistant', content: 'hello' },
	]);
	assert.deepEqual(client.add.calls, [
		[
			[
				{ role: 'user', content: 'hi' },
				{ role: 'assistant', content: 'hello' },
			],
			{ userId: 'u1' },
		],
	]);
	const items = await resolveAgentMemory([memory], 'drinks?');
	assert.deepEqual(parts(items), ['likes tea']);
	assert.equal(items[0].trust, 'untrusted');
});

test('loadMemory adapts Zep, Letta and Supermemory', async () => {
	const zepClient = {
		graph: { searchEdges: recorder({ data: [{ fact: 'F1' }] }) },
		thread: { addMessages: recorder() },
	};
	const lettaClient = {
		agents: {
			passages: {
				search: recorder({ results: [{ content: 'P1' }] }),
				create: recorder(),
			},
		},
	};
	const smClient = {
		search: recorder({ results: [{ chunk: 'C1' }] }),
		add: recorder(),
	};
	const modules = {
		'@getzep/zep-cloud': { ZepClient: function () { return zepClient; } },
		'@letta-ai/letta-client': { Letta: function () { return lettaClient; } },
		supermemory: { Supermemory: function () { return smClient; } },
	};
	const importer = async (name) => modules[name];

	const zep = await loadMemory(undefined, {
		backend: 'zep',
		options: { api_key: 'k', graph_id: 'g', thread_id: 't' },
		importer,
	});
	assert.deepEqual(await zep.recall('q', 2), ['F1']);
	assert.deepEqual(zepClient.graph.searchEdges.calls, [['g', { query: 'q', pageSize: 2 }]]);
	await zep.remember('hi');
	assert.deepEqual(zepClient.thread.addMessages.calls, [
		['t', { messages: [{ role: 'user', content: 'hi' }] }],
	]);

	const letta = await loadMemory(undefined, {
		backend: 'letta',
		options: { agent_id: 'a1' },
		importer,
	});
	assert.deepEqual(await letta.recall('q', 4), ['P1']);
	assert.deepEqual(lettaClient.agents.passages.search.calls, [['a1', { query: 'q', top_k: 4 }]]);
	await letta.remember([
		{ role: 'user', content: 'hi' },
		{ role: 'assistant', content: 'yo' },
	]);
	assert.deepEqual(lettaClient.agents.passages.create.calls, [
		['a1', { text: 'user: hi\nassistant: yo' }],
	]);

	const config = await parseConfig(
		'agent_rt_settings:\n  memory: {backend: supermemory, options: {namespace: u1}}\n',
		{ environment: {} },
	);
	const sm = await loadMemory(config, { importer });
	assert.deepEqual(await sm.recall('q', 1), ['C1']);
	assert.deepEqual(smClient.search.calls, [['u1', { query: 'q', limit: 1 }]]);
});

test('loadMemory reports bad backends, options and missing packages', async () => {
	await assert.rejects(loadMemory(undefined, { backend: 'nope' }), /unknown memory backend/);
	await assert.rejects(
		loadMemory(undefined, { backend: 'letta', options: { agent: 'a' } }),
		/unknown option/,
	);
	await assert.rejects(loadMemory(undefined, { backend: 'memu' }), /Python-only/);
	const missing = async () => {
		const error = new Error('Cannot find module');
		error.code = 'MODULE_NOT_FOUND';
		throw error;
	};
	await assert.rejects(
		loadMemory(undefined, { backend: 'zep', importer: missing }),
		/npm install @getzep\/zep-cloud/,
	);
	const deferred = await parseConfig(
		'agent_rt_settings:\n  memory:\n    backend: supermemory\n    options: {api_key: os.environ/SM_KEY}\n',
		{ environment: {} },
	);
	await assert.rejects(loadMemory(deferred), /SM_KEY/);
	await assert.rejects(
		parseConfig('agent_rt_settings:\n  memory: {backend: nope}\n', { environment: {} }),
		/memory.backend must be one of/,
	);
	await assert.rejects(
		parseConfig('agent_rt_settings:\n  memory: {options: {graph_id: g}}\n', { environment: {} }),
		/unknown key/,
	);
});
