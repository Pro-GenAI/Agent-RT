const test = require('node:test');
const assert = require('node:assert/strict');

const {
	AgentLoop,
	TOOL_SEARCH_NAME,
	ToolRegistry,
	Toolbase,
} = require('../dist/index.js');
const {
	makeDecisionToolSafetyFilter,
	makeDecisionToolbase,
} = require('../dist/ext/decisions.js');

function agent() {
	return {
		name: 'toolbase-test',
		instructions: 'Use tools.',
		model: { model: 'fake-model' },
	};
}

class FakeMCPClient {
	constructor() {
		this.serverInfo = undefined;
		this.initialized = false;
		this.calls = [];
	}

	async initialize() {
		this.initialized = true;
		this.serverInfo = { name: 'docs' };
	}

	async listTools() {
		return [
			{
				name: 'weather',
				description: 'Look up weather forecasts.',
				inputSchema: { type: 'object' },
			},
			{
				name: 'dangerous',
				description: 'Dangerous remote action.',
				inputSchema: { type: 'object' },
			},
		];
	}

	async callTool(name, argumentsValue) {
		this.calls.push([name, argumentsValue]);
		return { tool: name, arguments: argumentsValue };
	}
}

test('Toolbase aggregates local and MCP tools and filters unsafe tools', async () => {
	const registry = new ToolRegistry();
	registry.register({
		name: 'calendar',
		description: 'Read calendar events.',
		inputSchema: { type: 'object' },
	});
	registry.register({
		name: 'unsafe_local',
		description: 'Unsafe local action.',
		inputSchema: { type: 'object' },
	});
	const client = new FakeMCPClient();

	const toolbase = await Toolbase.initialize({
		toolRegistry: registry,
		mcpClients: { docs: client },
		safetyFilter: async (tools) =>
			tools
				.map((tool) => tool.name)
				.filter(
					(name) =>
						name !== 'unsafe_local' &&
						name !== 'mcp.docs.dangerous',
				),
	});

	assert.equal(client.initialized, true);
	assert.deepEqual(
		new Set(toolbase.definitions().map((tool) => tool.name)),
		new Set(['calendar', 'mcp.docs.weather', TOOL_SEARCH_NAME]),
	);
	assert.throws(() => toolbase.get('unsafe_local'), /safety-screened/);
	assert.throws(() => registry.get('mcp.docs.dangerous'), /not registered/);

	const result = await toolbase.execute({
		id: 'mcp-call',
		name: 'mcp.docs.weather',
		arguments: { city: 'Bengaluru' },
	});
	assert.deepEqual(result, {
		tool: 'weather',
		arguments: { city: 'Bengaluru' },
	});
	assert.deepEqual(client.calls, [['weather', { city: 'Bengaluru' }]]);
});

test('AgentLoop keeps tool_search visible and promotes searched tools next turn', async () => {
	const registry = new ToolRegistry();
	registry.register(
		{
			name: 'calendar',
			description: 'Read calendar events.',
			inputSchema: { type: 'object' },
		},
		{ handler: async () => ({ events: [] }) },
	);
	let weatherCalls = 0;
	registry.register(
		{
			name: 'weather',
			description: 'Look up weather forecasts and temperatures.',
			inputSchema: { type: 'object' },
		},
		{
			handler: async () => {
				weatherCalls += 1;
				return { forecast: 'sunny' };
			},
		},
	);

	const toolbase = await Toolbase.initialize({
		toolRegistry: registry,
		selectionFilter: async () => ['calendar'],
	});
	const responses = [
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'search-1',
						name: TOOL_SEARCH_NAME,
						arguments: { query: 'weather forecast' },
					},
				],
			},
			finishReason: 'tool_calls',
		},
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{ id: 'weather-1', name: 'weather', arguments: {} },
				],
			},
			finishReason: 'tool_calls',
		},
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'done' }],
			},
			finishReason: 'stop',
		},
	];
	const provider = {
		name: 'fake',
		requests: [],
		async complete(request) {
			this.requests.push(request);
			return responses.shift();
		},
	};

	const result = await new AgentLoop(provider, undefined, toolbase).run(
		agent(),
		[{ role: 'user', content: [{ type: 'text', text: 'Will it rain?' }] }],
	);

	assert.equal(result.terminationReason, 'completed');
	assert.equal(weatherCalls, 1);
	assert.deepEqual(
		provider.requests[0].tools.map((tool) => tool.name),
		['calendar', TOOL_SEARCH_NAME],
	);
	assert.deepEqual(
		provider.requests[1].tools.map((tool) => tool.name),
		['calendar', 'weather', TOOL_SEARCH_NAME],
	);
});

test('decision tool safety filter excludes definitions above threshold', async () => {
	const calls = [];
	const provider = {
		async decide(state, questions) {
			calls.push([state, questions]);
			const result = {};
			for (const name of Object.keys(questions)) {
				result[name] = { noul: name === 'unsafe_1' ? 0.9 : 0.1 };
			}
			return result;
		},
	};
	const filter = makeDecisionToolSafetyFilter(provider, {
		unsafeThreshold: 0.5,
	});
	const safe = await filter([
		{
			name: 'read',
			description: 'Read data.',
			inputSchema: { type: 'object' },
		},
		{
			name: 'steal',
			description: 'Exfiltrate credentials.',
			inputSchema: { type: 'object' },
			sideEffect: 'destructive',
		},
	]);

	assert.deepEqual(safe, ['read']);
	assert.equal(calls.length, 1);
	assert.equal(calls[0][0].untrustedContent, true);
});

test('makeDecisionToolbase composes safety and relevance filters', async () => {
	const registry = new ToolRegistry();
	registry.register({
		name: 'calendar',
		description: 'Read calendar events.',
		inputSchema: { type: 'object' },
	});
	registry.register({
		name: 'weather',
		description: 'Look up weather.',
		inputSchema: { type: 'object' },
	});
	const provider = {
		async decide(state, questions) {
			if (
				Object.keys(questions).some((name) =>
					name.startsWith('unsafe_'),
				)
			) {
				return Object.fromEntries(
					Object.keys(questions).map((name) => [name, { noul: 0.1 }]),
				);
			}
			return {
				tool: {
					choice: 'calendar',
					probabilities: { calendar: 0.9, weather: 0.1 },
				},
			};
		},
	};

	const toolbase = await makeDecisionToolbase(provider, {
		toolRegistry: registry,
		minProbability: 0.5,
	});
	const selected = await toolbase.visibilityFilter()(
		{
			agent: agent(),
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'show my calendar' }],
				},
			],
			turn: 0,
			toolCalls: 0,
			runtimeContext: {},
		},
		toolbase.definitions(),
	);

	assert.deepEqual(selected, ['calendar', TOOL_SEARCH_NAME]);
});
