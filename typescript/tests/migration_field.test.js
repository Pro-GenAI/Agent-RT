// Compatibility shapes found by migration field-testing third-party repositories.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const Anthropic = require('../dist/ext/compat/anthropic.js').default;
const OpenAI = require('../dist/ext/compat/openai.js').default;
const langchain = require('../dist/ext/compat/langchain.js');
const agents = require('../dist/ext/compat/openai_agents.js');

class ScriptedProvider {
	constructor(responses, error) {
		this.name = 'scripted';
		this.responses = [...responses];
		this.error = error;
		this.requests = [];
	}

	async complete(request) {
		this.requests.push(request);
		if (this.error) throw this.error;
		return this.responses.shift();
	}
}

function reply(text, toolCalls = [], finishReason = 'stop') {
	return {
		message: { role: 'assistant', content: text ? [{ type: 'text', text }] : [], toolCalls },
		model: 'm',
		usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
		finishReason,
	};
}

test('SDK error classes are exported, static, and vendor-specific', () => {
	const limited = new Anthropic.RateLimitError(429, { message: 'slow' }, undefined, {});
	assert.ok(limited instanceof Anthropic.APIError);
	assert.ok(limited instanceof Anthropic.AnthropicError);
	assert.equal(limited.status, 429);
	assert.equal(Anthropic.APIError, require('../dist/ext/compat/anthropic.js').APIError);
	assert.ok(Anthropic.APIError.generate(500, undefined, 'x') instanceof Anthropic.InternalServerError);
	assert.ok(new Anthropic.APIConnectionTimeoutError() instanceof Anthropic.APIConnectionError);
	assert.notEqual(OpenAI.APIError, Anthropic.APIError);
});

test('provider HTTP failures surface as the vendor error class', async () => {
	const failure = Object.assign(new Error('denied'), { status: 401 });
	const client = new Anthropic({ provider: new ScriptedProvider([], failure) });
	await assert.rejects(
		client.messages.create({ model: 'm', max_tokens: 5, messages: [{ role: 'user', content: 'hi' }] }),
		(error) => error instanceof Anthropic.AuthenticationError && error.cause === failure,
	);
	const plain = new Anthropic({ provider: new ScriptedProvider([], new TypeError('bad')) });
	await assert.rejects(
		plain.messages.create({ model: 'm', max_tokens: 5, messages: [{ role: 'user', content: 'hi' }] }),
		TypeError,
	);
});

test('Anthropic messages carry tool_use blocks, stop reasons, and tool history', async () => {
	const call = { id: 'call-1', name: 'lookup', arguments: { q: 'x' } };
	const provider = new ScriptedProvider([reply('ok', [call], 'tool_calls'), reply('{"a":1}')]);
	const client = new Anthropic({ provider });
	const history = [{ role: 'user', content: 'hi' }];
	const first = await client.messages.create({ model: 'm', max_tokens: 5, system: [{ type: 'text', text: 'sys' }], messages: history });
	assert.equal(first.stop_reason, 'tool_use');
	assert.deepEqual(first.content.at(-1), { type: 'tool_use', id: 'call-1', name: 'lookup', input: { q: 'x' } });

	history.push({ role: 'assistant', content: first.content });
	history.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'found' }] });
	const second = await client.beta.messages.parse({ model: 'm', max_tokens: 5, messages: history });
	assert.equal(second.stop_reason, 'end_turn');
	assert.deepEqual(second.parsed_output, { a: 1 });

	assert.equal(provider.requests[0].messages[0].content[0].text, 'sys');
	const [user, assistant, tool] = provider.requests[1].messages;
	assert.equal(user.role, 'user');
	assert.equal(assistant.toolCalls[0].name, 'lookup');
	assert.deepEqual([tool.role, tool.toolCallId, tool.content[0].text], ['tool', 'call-1', 'found']);
});

test('ES module entries default-export the client classes', async () => {
	const root = path.join(__dirname, '..', 'dist', 'ext', 'compat');
	const anthropicModule = await import(pathToFileURL(path.join(root, 'anthropic_esm.mjs')).href);
	const openaiModule = await import(pathToFileURL(path.join(root, 'openai_esm.mjs')).href);
	assert.equal(anthropicModule.default, Anthropic);
	assert.equal(anthropicModule.APIError, Anthropic.APIError);
	assert.equal(openaiModule.default, OpenAI);
});

test('LangChain structured tools validate input and report invocation errors', async () => {
	const schema = {
		safeParse: (value) =>
			typeof value.n === 'number' ? { success: true, data: value } : { success: false, error: 'n must be a number' },
		toJSONSchema: () => ({ type: 'object', properties: { n: { type: 'number' } }, required: ['n'] }),
	};
	const double = new langchain.DynamicStructuredTool({
		name: 'double',
		description: 'Double n',
		schema,
		func: async ({ n }) => n * 2,
	});
	assert.equal(await double.invoke({ n: 2 }), 4);
	await assert.rejects(double.invoke({ n: 'x' }), langchain.ToolInputParsingException);
	const message = await double.invoke({ type: 'tool_call', id: 't1', name: 'double', args: { n: 3 } });
	assert.ok(message instanceof langchain.ToolMessage);
	assert.equal(message.tool_call_id, 't1');

	const failing = new langchain.DynamicStructuredTool({
		name: 'fail',
		description: 'Always fails',
		schema: { type: 'object', properties: {} },
		func: async () => {
			throw new Error('boom');
		},
	});
	const seen = [];
	const middleware = langchain.toolErrorMiddleware({
		onError: (error) => {
			seen.push(langchain.ToolInvocationError.isInstance(error) ? error.toolCall.name : 'other');
			return 'handled';
		},
	});
	const started = [];
	const llmErrors = [];
	const callbacks = [
		{ handleToolStart: (_tool, input) => started.push(input) },
		{ handleLLMError: (error) => llmErrors.push(error.message) },
	];
	const model = new langchain.ChatOpenAI({
		model: 'm',
		provider: new ScriptedProvider([
			reply('', [{ id: 'c1', name: 'fail', arguments: {} }], 'tool_calls'),
			reply('', [{ id: 'c2', name: 'double', arguments: { n: 'x' } }], 'tool_calls'),
			reply('done'),
		]),
	});
	const result = await langchain
		.createAgent({ model, tools: [failing, double], middleware: [middleware] })
		.invoke('go', { callbacks });
	// As in LangChain: the tool's own error is passed on as is; only input the
	// schema rejects becomes a ToolInvocationError.
	assert.deepEqual(seen, ['other', 'double']);
	assert.ok(result.messages.some((item) => item.content === 'handled'));
	assert.deepEqual(started, ['{}', '{"n":"x"}']);

	const failingModel = new langchain.ChatOpenAI({ model: 'm', provider: new ScriptedProvider([], new Error('rate limited')) });
	await assert.rejects(langchain.createAgent({ model: failingModel }).invoke('go', { callbacks }), /rate limited/);
	assert.deepEqual(llmErrors, ['rate limited']);

	const zodLike = {
		safeParse: () => ({
			success: false,
			error: { issues: [{ message: 'Invalid input: expected number, received string', path: ['a'] }] },
		}),
	};
	const add = new langchain.DynamicStructuredTool({ name: 'add', description: 'Add', schema: zodLike, func: () => 0 });
	const parsing = await add.invoke({ a: 'one' }).catch((error) => error);
	assert.equal(
		String(parsing),
		'Error: Received tool input did not match expected schema\n\n✖ Invalid input: expected number, received string\n  → at a',
	);
	assert.match(new langchain.ToolInvocationError(new Error('bad'), { name: 'add', args: {} }).message, /\n\s+at /);
});

test('OpenAI Agents tools expose invoke, strict, RunContext, and Runner providers', async () => {
	const strictTool = agents.tool({
		name: 'echo',
		description: 'Echo',
		parameters: {
			type: 'object',
			properties: { text: { type: 'string' } },
			parse: (value) => {
				if (typeof value.text !== 'string') throw new Error('text required');
				return { text: value.text.toUpperCase() };
			},
		},
		execute: async ({ text }, context) => `${text}:${context instanceof agents.RunContext}`,
	});
	const loose = agents.tool({ name: 'loose', description: 'L', parameters: { type: 'object' }, strict: false, execute: () => 'ok' });
	assert.deepEqual([strictTool.strict, loose.strict], [true, false]);
	assert.equal(await strictTool.invoke(new agents.RunContext(), '{"text":"hi"}'), 'HI:true');
	assert.match(await strictTool.invoke(new agents.RunContext(), '{"text":1}'), /An error occurred while running the tool/);

	const provider = new ScriptedProvider([reply('', [{ id: 'c1', name: 'echo', arguments: { text: 'a' } }], 'tool_calls'), reply('final')]);
	const runner = new agents.Runner({ modelProvider: provider });
	const events = [];
	runner.on('agent_tool_start', (_context, _agent, item) => events.push(`start:${item.name}`));
	runner.on('agent_tool_end', (_context, _agent, item, output) => events.push(`end:${output}`));
	const result = await runner.run(new agents.Agent({ name: 'A', tools: [strictTool] }), 'go');
	assert.equal(result.finalOutput, 'final');
	assert.deepEqual(events, ['start:echo', 'end:A:true']);
	assert.throws(() => new agents.Runner({ modelProvider: {} }), TypeError);
	assert.equal(typeof new agents.OpenAIProvider({ apiKey: 'k' }).complete, 'function');
	agents.setTracingDisabled(true);
});

test('OpenAI Agents Agent exposes clone and modelSettings; LangChain agent state uses messages', async () => {
	const agent = new agents.Agent({ name: 'A', modelSettings: { parallelToolCalls: false } });
	assert.deepEqual(agent.modelSettings, { parallelToolCalls: false });
	const copy = agent.clone({ instructions: 'Be brief.' });
	assert.equal(copy.instructions, 'Be brief.');
	assert.deepEqual(copy.modelSettings, { parallelToolCalls: false });
	assert.throws(() => agent.clone({ model: 42 }), TypeError);

	const error = new langchain.ToolInvocationError(new Error('bad'), { name: 'add', args: {} });
	assert.deepEqual(error.toolCall, { name: 'add', args: {}, id: '', type: 'tool_call' });

	const model = new langchain.ChatOpenAI({ model: 'm', provider: new ScriptedProvider([reply('hi')]) });
	const state = await langchain.createAgent({ model }).invoke({ messages: [{ role: 'user', content: 'hello' }] });
	assert.ok(state.messages.every((item) => item instanceof langchain.BaseMessage));
});

test('OpenAI Agents runs drive SDK Model objects and stream SDK events', async () => {
	const requests = [];
	const sdkModel = {
		async getResponse(request) {
			requests.push(request);
			if (requests.length === 1)
				return {
					usage: new agents.Usage(),
					output: [{ type: 'function_call', callId: 'call-1', name: 'lookup', arguments: '{"q":"x"}' }],
				};
			return {
				usage: new agents.Usage(),
				output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: '{"answer":"found x"}' }] }],
			};
		},
	};
	const lookup = agents.tool({
		name: 'lookup',
		description: 'Look up',
		parameters: { type: 'object', properties: { q: { type: 'string' } } },
		execute: async ({ q }) => `found ${q}`,
	});
	const outputType = {
		type: 'json_schema',
		name: 'answer_output',
		strict: true,
		schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false },
	};
	const agent = new agents.Agent({ name: 'A', instructions: 'Be useful.', tools: [lookup], outputType }).clone({ model: sdkModel });
	const result = await new agents.Runner().run(agent, [{ role: 'user', content: 'find x' }]);
	assert.deepEqual(result.finalOutput, { answer: 'found x' });
	assert.equal(requests[0].outputType, outputType);
	assert.equal(requests[0].systemInstructions, 'Be useful.');
	assert.deepEqual(requests[0].input, [{ role: 'user', content: 'find x' }]);
	assert.equal(requests[1].input[1].type, 'function_call');
	assert.deepEqual(requests[1].input[2].output, { type: 'text', text: 'found x' });
	const output = result.newItems.find((item) => item.type === 'tool_call_output_item');
	assert.deepEqual([output.rawItem.callId, output.output], ['call-1', 'found x']);

	const streamingModel = {
		async getResponse() {
			throw new Error('streaming only');
		},
		async *getStreamedResponse() {
			yield { type: 'output_text_delta', delta: 'Hello' };
			yield { type: 'output_text_delta', delta: ', world' };
			yield {
				type: 'response_done',
				response: {
					output: [
						{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Hello, world' }] },
						{ type: 'function_call', callId: 'call_9', name: 'lookup', arguments: '{"q":"y"}' },
					],
				},
			};
		},
	};
	const streamed = await agents.run(
		new agents.Agent({ name: 'S', model: streamingModel, tools: [lookup], toolUseBehavior: { stopAtToolNames: ['lookup'] } }),
		'hi',
		{ stream: true },
	);
	const text = [];
	const toolCalls = [];
	for await (const event of streamed) {
		if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') text.push(event.data.delta);
		if (event.type === 'run_item_stream_event' && event.name === 'tool_called') toolCalls.push(event.item.rawItem.callId);
	}
	await streamed.completed;
	assert.equal(text.join(''), 'Hello, world');
	assert.deepEqual(toolCalls, ['call_9']);
	assert.equal(streamed.finalOutput, 'found y');
});

test('OpenAIChatCompletionsModel serves the SDK Model interface over an OpenAI-shaped client', async () => {
	const bodies = [];
	const client = {
		chat: {
			completions: {
				create: async (body) => {
					bodies.push(body);
					return {
						id: 'c',
						model: body.model,
						choices: [{ index: 0, message: { role: 'assistant', content: 'pong' }, finish_reason: 'stop' }],
						usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
					};
				},
			},
		},
	};
	const model = new agents.OpenAIChatCompletionsModel(client, 'route-model');
	const response = await model.getResponse({
		systemInstructions: 'sys',
		input: [{ role: 'user', content: 'ping' }],
		modelSettings: {},
		tools: [],
		outputType: 'text',
		handoffs: [],
		tracing: false,
	});
	assert.equal(response.output[0].content[0].text, 'pong');
	assert.equal(bodies[0].model, 'route-model');
	const provider = { getModel: () => model };
	const result = await new agents.Runner({ modelProvider: provider }).run(new agents.Agent({ name: 'A', model: 'route-model' }), 'ping');
	assert.equal(result.finalOutput, 'pong');
});

test('OpenAI Agents approvals pause the run and resume through RunState', async () => {
	const executed = [];
	const deleteFile = agents.tool({
		name: 'delete_file',
		description: 'Delete a file',
		parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
		needsApproval: true,
		execute: async ({ path }) => {
			executed.push(path);
			return `deleted ${path}`;
		},
	});
	const guarded = agents.tool({
		name: 'guarded',
		description: 'Guarded',
		parameters: { type: 'object', properties: {} },
		inputGuardrails: [{ name: 'deny', run: () => ({ behavior: { type: 'rejectContent', message: 'blocked by policy' } }) }],
		execute: async () => 'should not run',
	});
	const provider = new ScriptedProvider([
		reply('', [{ id: 'call-del', name: 'delete_file', arguments: { path: 'a.txt' } }], 'tool_calls'),
		reply('', [{ id: 'call-g', name: 'guarded', arguments: {} }], 'tool_calls'),
		reply('done'),
	]);
	const agent = new agents.Agent({ name: 'A', tools: [deleteFile, guarded] });
	const runner = new agents.Runner({ modelProvider: provider });

	const first = await runner.run(agent, 'clean up');
	assert.equal(first.interruptions.length, 1);
	assert.equal(first.interruptions[0].rawItem.callId, 'call-del');
	assert.deepEqual(executed, []);

	const state = await agents.RunState.fromString(agent, first.state.toString());
	state.approve(state.getInterruptions()[0]);
	const resumed = await runner.run(agent, state);
	assert.deepEqual(executed, ['a.txt']);
	assert.equal(resumed.finalOutput, 'done');
	const toolTexts = provider.requests[2].messages.filter((m) => m.role === 'tool').map((m) => m.content[0].text);
	assert.deepEqual(toolTexts, ['deleted a.txt', 'blocked by policy']);
	assert.ok(resumed.history.some((item) => item.type === 'function_call_result'));

	const rejecting = new ScriptedProvider([
		reply('', [{ id: 'call-2', name: 'delete_file', arguments: { path: 'b.txt' } }], 'tool_calls'),
		reply('ok'),
	]);
	const paused = await new agents.Runner({ modelProvider: rejecting }).run(agent, 'go');
	paused.state.reject(paused.interruptions[0], { message: 'not allowed' });
	await new agents.Runner({ modelProvider: rejecting }).run(agent, paused.state);
	assert.deepEqual(executed, ['a.txt']);
	assert.equal(rejecting.requests[1].messages.at(-1).content[0].text, 'not allowed');
});

test('OpenAIChatCompletionsModel builds SDK-shaped Chat Completions requests across a tool turn', async () => {
	const bodies = [];
	async function* chunks(list) {
		for (const chunk of list) yield chunk;
	}
	const client = {
		chat: {
			completions: {
				create: async (body) => {
					bodies.push(JSON.parse(JSON.stringify(body)));
					if (bodies.length === 1)
						return chunks([
							{ id: 'r1', choices: [{ index: 0, delta: { role: 'assistant', reasoning: 'Need the tool.' } }] },
							{
								id: 'r1',
								choices: [
									{
										index: 0,
										finish_reason: 'tool_calls',
										delta: { tool_calls: [{ index: 0, id: 'call-a', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } }] },
									},
								],
							},
						]);
					return chunks([
						{ id: 'r2', choices: [{ index: 0, delta: { content: 'Done.' } }] },
						{ id: 'r2', choices: [], usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 } },
					]);
				},
			},
		},
	};
	const lookup = agents.tool({
		name: 'lookup',
		description: 'Look up',
		parameters: { type: 'object', properties: { q: { type: 'string' } }, additionalProperties: true },
		strict: false,
		execute: async () => ({ found: true }),
	});
	const agent = new agents.Agent({
		name: 'A',
		instructions: 'Be useful.',
		model: new agents.OpenAIChatCompletionsModel(client, 'route-model'),
		modelSettings: { reasoning: { effort: 'high' }, maxTokens: 2000 },
		tools: [lookup],
	});
	const input = [
		{ role: 'user', content: 'Say hello.' },
		{ type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'Hello.' }] },
		{ role: 'user', content: [{ type: 'input_text', text: 'Look at this' }, { type: 'input_image', image: 'data:image/png;base64,AAAA' }] },
	];
	const streamed = await agents.run(agent, input, { stream: true });
	for await (const _event of streamed);
	await streamed.completed;
	assert.equal(streamed.error, undefined);
	assert.equal(streamed.finalOutput, 'Done.');

	const [first, second] = bodies;
	assert.equal(first.model, 'route-model');
	assert.equal(first.reasoning_effort, 'high');
	assert.equal(first.max_tokens, 2000);
	assert.equal(first.stream, true);
	assert.equal(first.tools[0].function.strict, false);
	assert.deepEqual(first.messages[0], { content: 'Be useful.', role: 'system' });
	assert.deepEqual(first.messages[2].content, [{ type: 'text', text: 'Hello.' }]);
	assert.equal(first.messages[3].content[1].image_url.url, 'data:image/png;base64,AAAA');
	const replay = second.messages.find((message) => message.role === 'assistant' && message.tool_calls);
	assert.equal(replay.reasoning, 'Need the tool.');
	assert.equal(replay.tool_calls[0].id, 'call-a');
	const toolMessage = second.messages.find((message) => message.role === 'tool');
	assert.deepEqual([toolMessage.tool_call_id, toolMessage.content], ['call-a', '{"found":true}']);
});

test('OpenAI Agents guardrails run before approvals and JSON output is parsed without validation', async () => {
	const checks = [];
	const write = agents.tool({
		name: 'write',
		description: 'Write',
		parameters: { type: 'object', properties: {} },
		needsApproval: true,
		inputGuardrails: [{ name: 'preflight', run: ({ toolCall }) => (checks.push(toolCall.callId), { behavior: { type: 'allow' } }) }],
		execute: async () => 'written',
	});
	const provider = new ScriptedProvider([reply('', [{ id: 'w1', name: 'write', arguments: {} }], 'tool_calls'), reply('ok')]);
	const runner = new agents.Runner({ modelProvider: provider });
	const agent = new agents.Agent({ name: 'A', tools: [write] });
	const paused = await runner.run(agent, 'go');
	assert.deepEqual(checks, ['w1']);
	paused.state.approve(paused.interruptions[0]);
	await runner.run(agent, paused.state);
	assert.deepEqual(checks, ['w1', 'w1']);

	const outputType = {
		type: 'json_schema',
		name: 'out',
		strict: true,
		schema: { type: 'object', properties: { a: { type: 'string' }, b: { type: 'string' } }, required: ['a', 'b'] },
	};
	const sparse = new agents.Agent({
		name: 'S',
		outputType,
		model: { getResponse: async () => ({ output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: '{"a":"x"}' }] }] }) },
	});
	assert.deepEqual((await agents.run(sparse, 'go')).finalOutput, { a: 'x' });
});

test('OpenAI Agents normalizes tool and handoff names like the SDK', async () => {
	// Field shape: a host tool named `notion-update-page` is called by the model
	// as `notion_update_page`, with a preflight guardrail and approval.
	const checks = [];
	const executed = [];
	const update = agents.tool({
		name: 'notion-update-page',
		description: 'Update one Notion page.',
		parameters: { type: 'object', properties: { page_id: { type: 'string' } }, additionalProperties: true },
		strict: false,
		needsApproval: true,
		inputGuardrails: [{ name: 'preflight', run: ({ toolCall }) => (checks.push(toolCall.callId), { behavior: { type: 'allow' } }) }],
		execute: async (args) => (executed.push(args), { updated: true }),
	});
	assert.equal(update.name, 'notion_update_page');
	assert.equal(agents.toFunctionToolName('my tool.v2'), 'my_tool_v2');
	assert.throws(() => agents.tool({ description: 'x', parameters: { type: 'object', properties: {} }, execute: [() => 'x'][0] }), /cannot be empty/);

	const provider = new ScriptedProvider([
		reply('', [{ id: 'call-n', name: 'notion_update_page', arguments: { page_id: 'roadmap' } }], 'tool_calls'),
		reply('updated'),
	]);
	const runner = new agents.Runner({ modelProvider: provider });
	const agent = new agents.Agent({ name: 'A', tools: [update] });
	const paused = await runner.run(agent, 'go');
	assert.deepEqual(checks, ['call-n']);
	assert.equal(paused.interruptions.length, 1);
	assert.deepEqual(provider.requests[0].tools.map((t) => t.name), ['notion_update_page']);
	paused.state.approve(paused.interruptions[0]);
	const resumed = await runner.run(agent, paused.state);
	assert.deepEqual(checks, ['call-n', 'call-n']);
	assert.deepEqual(executed, [{ page_id: 'roadmap' }]);
	assert.equal(resumed.finalOutput, 'updated');

	const billing = new agents.Agent({ name: 'Billing Agent-EU', instructions: 'b' });
	const triage = new agents.Agent({
		name: 'T',
		handoffs: [billing, agents.handoff(billing, { toolNameOverride: 'to-billing' })],
	});
	const recording = new ScriptedProvider([reply('hi')]);
	await new agents.Runner({ modelProvider: recording }).run(triage, 'go');
	assert.deepEqual(recording.requests[0].tools.map((t) => t.name), ['transfer_to_Billing_Agent_EU', 'to-billing']);
	assert.equal(billing.asTool().name, 'Billing_Agent_EU');
});

test('OpenAI Agents collects every approval interruption from one turn before pausing', async () => {
	// Field shape: two parallel calls to one approval-gated tool must both pause,
	// each bound to its own call id, while an ungated sibling still runs.
	const checks = [];
	const executed = [];
	const update = agents.tool({
		name: 'update',
		description: 'Update a page',
		parameters: { type: 'object', properties: { page: { type: 'string' } }, required: ['page'] },
		needsApproval: true,
		inputGuardrails: [{ run: ({ toolCall }) => (checks.push(toolCall.callId), { behavior: { type: 'allow' } }) }],
		execute: async ({ page }) => (executed.push(page), `updated ${page}`),
	});
	const lookup = agents.tool({
		name: 'lookup',
		description: 'Look up',
		parameters: { type: 'object', properties: {} },
		execute: async () => (executed.push('lookup'), 'found'),
	});
	const provider = new ScriptedProvider([
		reply('', [
			{ id: 'call-slow', name: 'update', arguments: { page: 'slow' } },
			{ id: 'call-fast', name: 'update', arguments: { page: 'fast' } },
			{ id: 'call-look', name: 'lookup', arguments: {} },
		], 'tool_calls'),
		reply('done'),
	]);
	const runner = new agents.Runner({ modelProvider: provider });
	const agent = new agents.Agent({ name: 'A', tools: [update, lookup] });
	const paused = await runner.run(agent, 'go');
	assert.deepEqual(paused.interruptions.map((item) => item.rawItem.callId), ['call-slow', 'call-fast']);
	assert.deepEqual(checks, ['call-slow', 'call-fast']);
	assert.deepEqual(executed, ['lookup']);
	assert.equal(provider.requests.length, 1);

	const state = await agents.RunState.fromString(agent, paused.state.toString());
	for (const item of state.getInterruptions()) state.approve(item);
	const resumed = await runner.run(agent, state);
	assert.deepEqual(executed, ['lookup', 'slow', 'fast']);
	assert.equal(resumed.finalOutput, 'done');
	const answered = provider.requests[1].messages.filter((m) => m.role === 'tool').map((m) => m.toolCallId).sort();
	assert.deepEqual(answered, ['call-fast', 'call-look', 'call-slow']);
});

test('vendor clients read SDK base URL and key from the environment at construction', () => {
	// Field shape: tests set process.env.OPENAI_BASE_URL to a local fake and
	// build `new OpenAI({ apiKey })` with no baseURL; the shim previously sent
	// those requests to the hosted API instead.
	const saved = { ...process.env };
	try {
		process.env.OPENAI_BASE_URL = 'http://127.0.0.1:9/v1';
		process.env.OPENAI_API_KEY = 'sk-env';
		process.env.ANTHROPIC_BASE_URL = 'http://127.0.0.1:9';
		assert.equal(new OpenAI({ apiKey: 'sk-app' }).provider.settings.baseUrl, 'http://127.0.0.1:9/v1');
		assert.equal(new OpenAI().provider.settings.apiKey, 'sk-env');
		assert.equal(new OpenAI({ baseURL: 'http://127.0.0.1:8/v1' }).provider.settings.baseUrl, 'http://127.0.0.1:8/v1');
		assert.equal(new Anthropic({ apiKey: 'k' }).provider.settings.baseUrl, 'http://127.0.0.1:9');
		assert.equal(new agents.OpenAIProvider().settings.baseUrl, 'http://127.0.0.1:9/v1');
		process.env.OPENAI_BASE_URL = '  ';
		assert.equal(new OpenAI({ apiKey: 'k' }).provider.settings.baseUrl, 'https://api.openai.com/v1');
	} finally {
		for (const key of Object.keys(process.env)) if (!(key in saved)) delete process.env[key];
		Object.assign(process.env, saved);
	}
});

test('OpenAI chat completions return tool_calls, stream chunks, and SDK error details', async () => {
	// Field shapes: tests read message.tool_calls, iterate `stream: true`
	// completions with for-await, and assert error.status/code/headers.get().
	const toolResponse = {
		message: {
			role: 'assistant',
			content: [],
			toolCalls: [{ id: 'call_1', name: 'get_weather', arguments: { city: 'Quito' } }],
		},
		model: 'm',
		finishReason: 'tool_calls',
	};
	const client = new OpenAI({ provider: new ScriptedProvider([toolResponse]) });
	const completion = await client.chat.completions.create({
		model: 'm',
		messages: [{ role: 'user', content: 'weather in Quito' }],
		tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }],
	});
	assert.deepEqual(completion.choices[0].message.tool_calls[0], {
		id: 'call_1',
		type: 'function',
		function: { name: 'get_weather', arguments: '{"city":"Quito"}' },
	});

	const textResponse = {
		message: { role: 'assistant', content: [{ type: 'text', text: 'desserts' }] },
		model: 'm',
		finishReason: 'stop',
	};
	const streamClient = new OpenAI({ provider: new ScriptedProvider([textResponse]) });
	const stream = await streamClient.chat.completions.create({ model: 'm', stream: true, messages: [{ role: 'user', content: 'x' }] });
	let text = '';
	let finish;
	for await (const chunk of stream) {
		text += chunk.choices[0]?.delta.content ?? '';
		finish = chunk.choices[0]?.finish_reason ?? finish;
	}
	assert.equal(text, 'desserts');
	assert.equal(finish, 'stop');

	const failure = Object.assign(
		new Error('OpenAI request failed (429): {"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded","param":null}}'),
		{ status: 429, responseHeaders: { 'retry-after': '0', 'x-request-id': 'req_1' } },
	);
	const failing = new OpenAI({ provider: new ScriptedProvider([], failure) });
	const error = await failing.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: 'x' }] }).catch((caught) => caught);
	assert.ok(error instanceof OpenAI.RateLimitError);
	assert.equal(error.status, 429);
	assert.equal(error.code, 'rate_limit_exceeded');
	assert.equal(error.headers.get('retry-after'), '0');
	assert.equal(error.requestID, 'req_1');
	assert.equal(error.message, '429 slow down');

	const streamFailure = await new OpenAI({ provider: new ScriptedProvider([], failure) }).chat.completions.create({ model: 'm', stream: true, messages: [] });
	await assert.rejects(async () => {
		for await (const _chunk of streamFailure);
	}, OpenAI.RateLimitError);
});

test('OpenAI JSON mode survives migration as response_format json_object', async () => {
	// Field shape: `response_format: { type: 'json_object' }` was dropped.
	const { JSON_OBJECT_OUTPUT } = require('../dist/index.js');
	const provider = new ScriptedProvider([
		{ message: { role: 'assistant', content: [{ type: 'text', text: '{"ok":1}' }] }, model: 'm', finishReason: 'stop' },
	]);
	const client = new OpenAI({ provider });
	await client.chat.completions.create({ model: 'm', response_format: { type: 'json_object' }, messages: [{ role: 'user', content: 'x' }] });
	assert.deepEqual(provider.requests[0].structuredOutput, JSON_OBJECT_OUTPUT);

	let sent;
	const sdkShaped = {
		chat: { completions: { create: async (params) => {
			sent = params;
			return { choices: [{ message: { role: 'assistant', content: '{}' }, finish_reason: 'stop' }], model: 'm' };
		} } },
	};
	const { OpenAIModelProvider } = require('../dist/index.js');
	const native = new OpenAIModelProvider({ apiKey: 'k', baseUrl: 'http://127.0.0.1:9/v1', websocket: false }, sdkShaped);
	await native.complete({ model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }], structuredOutput: JSON_OBJECT_OUTPUT });
	assert.deepEqual(sent.response_format, { type: 'json_object' });
});

test('Anthropic startup validation explains an SDK without the models API', async () => {
	// Field shape: migrated apps keep older @anthropic-ai/sdk versions (0.37
	// seen), which predate client.models; validation must say what to upgrade.
	const { validateAnthropicProviderSettings } = require('../dist/index.js');
	await assert.rejects(
		validateAnthropicProviderSettings(
			{ apiKey: 'k', baseUrl: 'http://127.0.0.1:9' },
			{ clientFactory: () => ({ messages: { create: async () => ({}) } }) },
		),
		/upgrade '@anthropic-ai\/sdk' to 0\.39\.0/,
	);
	const ids = await validateAnthropicProviderSettings(
		{ apiKey: 'k', baseUrl: 'http://127.0.0.1:9' },
		{ clientFactory: () => ({ models: { list: async () => ({ data: [{ id: 'claude-x' }] }) } }) },
	);
	assert.deepEqual(ids, ['claude-x']);
});

test('Anthropic messages.stream() and create({ stream: true }) follow the SDK stream shape', async () => {
	// Field shape: apps call `anthropic.messages.stream(params)` with
	// .on('text') listeners and `await stream.finalMessage()`, or iterate
	// `create({ stream: true })` events.
	const reply = () => ({
		message: { role: 'assistant', content: [{ type: 'text', text: 'hello there' }] },
		model: 'claude-x',
		finishReason: 'stop',
		usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
	});
	const client = new Anthropic({ provider: new ScriptedProvider([reply(), reply(), reply()]) });
	const params = { model: 'claude-x', max_tokens: 10, messages: [{ role: 'user', content: 'hi' }] };

	const texts = [];
	let ended = false;
	const stream = client.messages.stream(params).on('text', (delta, snapshot) => texts.push([delta, snapshot]));
	stream.on('end', () => {
		ended = true;
	});
	const message = await stream.finalMessage();
	assert.equal(message.content[0].text, 'hello there');
	assert.equal(await stream.finalText(), 'hello there');
	assert.deepEqual(texts, [['hello there', 'hello there']]);
	assert.ok(ended);

	const types = [];
	for await (const event of client.messages.stream(params)) types.push(event.type);
	assert.deepEqual(types, [
		'message_start',
		'content_block_start',
		'content_block_delta',
		'content_block_stop',
		'message_delta',
		'message_stop',
	]);

	const raw = await client.messages.create({ ...params, stream: true });
	let text = '';
	for await (const event of raw) {
		if (event.type === 'content_block_delta') text += event.delta.text;
	}
	assert.equal(text, 'hello there');

	const failing = new Anthropic({
		provider: new ScriptedProvider([], Object.assign(new Error('slow'), { status: 429 })),
	});
	const errors = [];
	const failed = failing.messages.stream(params).on('error', (error) => errors.push(error));
	await assert.rejects(failed.finalMessage(), Anthropic.RateLimitError);
	assert.ok(errors[0] instanceof Anthropic.RateLimitError);
	await assert.rejects(failing.messages.create({ ...params, stream: true }), Anthropic.RateLimitError);
});

test('vendor clients send requests through an injected fetch, including SSE streams', async () => {
	// Field shape: offline tests pass `new Anthropic({ fetch })` / `new
	// OpenAI({ fetch })` returning canned responses; the shims used to ignore
	// it and send the request to the configured endpoint instead.
	const events = [
		{ type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', model: 'claude-x', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 120, output_tokens: 0 } } },
		{ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
		{ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '{"summary":' } },
		{ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '"ok"}' } },
		{ type: 'content_block_stop', index: 0 },
		{ type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 240 } },
		{ type: 'message_stop' },
	];
	let anthropicCalls = 0;
	const anthropic = new Anthropic({
		apiKey: 'fixture-not-a-real-key',
		maxRetries: 0,
		fetch: async () => {
			anthropicCalls += 1;
			return new Response(events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join(''), {
				headers: { 'content-type': 'text/event-stream' },
			});
		},
	});
	const message = await anthropic.messages
		.stream({ model: 'claude-x', max_tokens: 300, messages: [{ role: 'user', content: 'brief' }] })
		.finalMessage();
	assert.equal(message.content[0].text, '{"summary":"ok"}');
	assert.equal(message.usage.output_tokens, 240);
	assert.equal(anthropicCalls, 1);

	const seen = [];
	const openai = new OpenAI({
		apiKey: 'k',
		baseURL: 'http://fixture.invalid/v1',
		fetch: async (url, init) => {
			seen.push([String(url), JSON.parse(init.body).model]);
			return new Response(
				JSON.stringify({ id: 'c1', model: 'm', choices: [{ index: 0, message: { role: 'assistant', content: 'fixture' }, finish_reason: 'stop' }] }),
				{ headers: { 'content-type': 'application/json' } },
			);
		},
	});
	const completion = await openai.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: 'x' }] });
	assert.equal(completion.choices[0].message.content, 'fixture');
	assert.deepEqual(seen, [['http://fixture.invalid/v1/chat/completions', 'm']]);
});
