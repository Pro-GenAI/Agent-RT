const test = require('node:test');
const assert = require('node:assert/strict');

const { createDeepAgent } = require('agent-rt/deepagents');
const { OpenRouter } = require('agent-rt/@openrouter/sdk');
const { ChatOpenAI } = require('../dist/ext/compat/langchain.js');

class QueueProvider {
	constructor(responses) {
		this.name = 'queue';
		this.requests = [];
		this.responses = [...responses];
	}

	async complete(request) {
		this.requests.push(request);
		if (!this.responses.length) throw new Error('unexpected model request');
		return this.responses.shift();
	}
}

function response(text, ...calls) {
	return {
		message: {
			role: 'assistant',
			content: text ? [{ type: 'text', text }] : [],
			toolCalls: calls.map(([name, args], i) => ({
				id: `call-${i}-${name}`,
				name,
				arguments: args,
			})),
		},
		model: 'test-model',
		finishReason: calls.length ? 'tool_calls' : 'stop',
		usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
	};
}

function toolOutputs(result) {
	return result.messages.filter((m) => m.role === 'tool').map((m) => m.content);
}

test('createDeepAgent plans, edits virtual files, and delegates to subagents', async () => {
	const provider = new QueueProvider([
		response(
			'',
			['write_todos', { todos: [{ content: 'draft', status: 'in_progress' }] }],
			['write_file', { file_path: '/notes/a.txt', content: 'hello world' }],
		),
		response('', ['task', { description: 'fix it', subagent_type: 'editor' }]),
		// The editor subagent shares the virtual filesystem.
		response('', [
			'edit_file',
			{ file_path: '/notes/a.txt', old_string: 'world', new_string: 'there' },
		]),
		response('edited'),
		response(
			'',
			['grep', { pattern: 'there', output_mode: 'content' }],
			['ls', { path: '/' }],
			['read_file', { file_path: '/in.md' }],
			['glob', { pattern: '*.txt' }],
			['glob', { pattern: '**/*.txt' }],
		),
		response('done'),
	]);
	const agent = createDeepAgent({
		model: new ChatOpenAI({ model: 'test-model', provider }),
		systemPrompt: 'Be brief.',
		subagents: [{ name: 'editor', description: 'Edits files.', systemPrompt: 'Edit.' }],
	});

	const result = await agent.invoke({
		messages: [{ role: 'user', content: 'go' }],
		files: { '/in.md': 'input' },
	});

	assert.equal(result.messages.at(-1).content, 'done');
	assert.equal(result.files['/notes/a.txt'].content, 'hello there');
	assert.equal(result.files['/in.md'], 'input');
	assert.deepEqual(result.todos, [{ content: 'draft', status: 'in_progress' }]);
	const outputs = toolOutputs(result);
	assert.ok(outputs.includes('edited'));
	assert.ok(outputs.includes('/notes/a.txt:1:hello there'));
	assert.ok(outputs.includes('["/in.md","/notes/"]'));
	assert.ok(outputs.includes('     1\tinput'));
	// `*` does not cross directories; `**/` does.
	assert.ok(outputs.includes('[]'));
	assert.ok(outputs.includes('["/notes/a.txt"]'));
	const names = provider.requests[0].tools.map((t) => t.name);
	for (const name of ['write_todos', 'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'task']) {
		assert.ok(names.includes(name), name);
	}
	assert.match(provider.requests[0].messages[0].content[0].text, /Be brief\./);
	// The subagent sees only the delegated task, not the parent conversation.
	assert.deepEqual(provider.requests[2].messages.map((m) => m.role), ['system', 'user']);
});

test('createDeepAgent file tools report errors instead of overwriting', async () => {
	const provider = new QueueProvider([
		response(
			'',
			['write_file', { file_path: '/a', content: 'new' }],
			['edit_file', { file_path: '/a', old_string: 'x', new_string: 'y' }],
			['task', { description: 'x', subagent_type: 'missing' }],
		),
		response('ok'),
	]);
	const agent = createDeepAgent({ model: new ChatOpenAI({ model: 'm', provider }) });
	const result = await agent.invoke({
		messages: [{ role: 'user', content: 'go' }],
		files: { '/a': 'x x' },
	});
	const outputs = toolOutputs(result);
	assert.match(outputs[0], /^Cannot write to \/a because it already exists/);
	assert.match(outputs[1], /appears 2 times/);
	assert.match(outputs[2], /unknown subagent_type 'missing'/);
	assert.deepEqual(result.files, { '/a': 'x x' });
});

test('createDeepAgent rejects unemulated options', () => {
	assert.throws(() => createDeepAgent(), /requires model/);
	const model = new ChatOpenAI({ model: 'm', provider: new QueueProvider([]) });
	assert.throws(() => createDeepAgent({ model, backend: {} }), /backend/);
});

test('createDeepAgent rejects unsupported subagent permissions and execution modes', () => {
	const model = new ChatOpenAI({ model: 'm', provider: new QueueProvider([]) });
	for (const key of ['permissions', 'mode', 'graph', 'graphId', 'runnable']) {
		assert.throws(
			() => createDeepAgent({ model, subagents: [{ name: 'worker', description: 'Work', [key]: { deny: true } }] }),
			new RegExp(key),
		);
	}
});

test('OpenRouter rejects request parameters not honored by Agent RT', async () => {
	const provider = new QueueProvider([]);
	const client = new OpenRouter({ provider });
	for (const key of ['toolChoice', 'topP', 'provider', 'plugins']) {
		await assert.rejects(
			client.chat.send({ chatRequest: { model: 'x', messages: [], [key]: { only: ['x'] } } }),
			new RegExp(key === 'toolChoice' ? 'tool_choice' : key === 'topP' ? 'top_p' : key),
		);
	}
	await assert.rejects(
		client.chat.send({ chatRequest: { model: 'x', messages: [] }, provider: { only: ['x'] } }),
		/wrapper options: provider/,
	);
	await assert.rejects(
		client.chat.send({ model: 'x', messages: [] }, { timeoutMs: 100 }),
		/per-call SDK options/,
	);
	assert.equal(provider.requests.length, 0);
});

test('OpenRouter chat.send converts camelCase requests and results', async () => {
	const provider = new QueueProvider([
		response('', ['lookup', { q: 'x' }]),
		response('streamed'),
	]);
	const client = new OpenRouter({ provider });
	const result = await client.chat.send({
		chatRequest: {
			model: 'openai/gpt-5',
			messages: [{ role: 'user', content: 'hi' }],
			maxTokens: 7,
			temperature: 0.4,
			reasoningEffort: 'low',
			tools: [
				{
					type: 'function',
					function: {
						name: 'lookup',
						parameters: { type: 'object', properties: { someField: { type: 'string' } } },
					},
				},
			],
		},
	});
	const request = provider.requests[0];
	assert.equal(request.model, 'openai/gpt-5');
	assert.equal(request.maxOutputTokens, 7);
	assert.equal(request.temperature, 0.4);
	assert.equal(request.reasoning.effort, 'low');
	// JSON Schema keys are the caller's own and are not snake_cased.
	assert.ok('someField' in request.tools[0].inputSchema.properties);
	assert.equal(result.choices[0].finishReason, 'tool_calls');
	assert.equal(result.choices[0].message.toolCalls[0].function.name, 'lookup');
	assert.equal(result.usage.promptTokens, 3);

	const stream = await client.chat.send({
		model: 'openai/gpt-5',
		messages: [{ role: 'user', content: 'hi' }],
		stream: true,
	});
	let text = '';
	for await (const chunk of stream) text += chunk.choices[0]?.delta?.content ?? '';
	assert.equal(text, 'streamed');
});

test('OpenRouter key never falls back to OPENAI_API_KEY', () => {
	const saved = { ...process.env };
	try {
		process.env.OPENAI_API_KEY = 'openai-secret';
		delete process.env.OPENROUTER_API_KEY;
		const settings = new OpenRouter().provider.settings;
		assert.equal(settings.apiKey, undefined);
		assert.match(settings.baseUrl, /^https:\/\/openrouter\.ai\/api\/v1/);
		process.env.OPENROUTER_API_KEY = 'router-secret';
		assert.equal(new OpenRouter().provider.settings.apiKey, 'router-secret');
	} finally {
		process.env = saved;
	}
});
