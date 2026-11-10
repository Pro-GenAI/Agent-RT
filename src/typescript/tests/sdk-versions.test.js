// Real vendor-SDK integration against a local, offline wire-format double.
//
// Most provider tests inject fake clients. These tests instead drive whichever
// `openai` / `@anthropic-ai/sdk` version is installed against a loopback HTTP
// server, so `make test-deps-matrix` can prove every supported SDK release
// (mid-2024 onward) still works. No real model endpoint is ever contacted.
const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');

const {
	AnthropicModelProvider,
	OpenAIModelProvider,
	validateAnthropicProviderSettings,
	validateOpenAIProviderSettings,
} = require('../dist/index.js');

const MODEL = 'sdk-matrix-model';
const WEATHER_TOOL = {
	name: 'get_weather',
	description: 'Look up the weather',
	inputSchema: {
		type: 'object',
		properties: { city: { type: 'string' } },
		required: ['city'],
	},
};

function installed(name) {
	try {
		require.resolve(name);
		return true;
	} catch {
		return false;
	}
}

function sse(events) {
	return (
		events
			.map(([event, data]) => {
				const payload =
					typeof data === 'string' ? data : JSON.stringify(data);
				return (
					(event ? `event: ${event}\n` : '') + `data: ${payload}\n`
				);
			})
			.join('\n') + '\n'
	);
}

function openAICompletion(body) {
	let message = { role: 'assistant', content: 'hello from openai' };
	let finish = 'stop';
	if (body.tools) {
		message = {
			role: 'assistant',
			content: null,
			tool_calls: [
				{
					id: 'call_1',
					type: 'function',
					function: {
						name: 'get_weather',
						arguments: '{"city": "Paris"}',
					},
				},
			],
		};
		finish = 'tool_calls';
	}
	return {
		id: 'chatcmpl-1',
		object: 'chat.completion',
		created: 1,
		model: MODEL,
		choices: [{ index: 0, message, finish_reason: finish }],
		usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
	};
}

function openAIStream() {
	const chunk = (delta, finish = null) => ({
		id: 'chatcmpl-1',
		object: 'chat.completion.chunk',
		created: 1,
		model: MODEL,
		choices: [{ index: 0, delta, finish_reason: finish }],
	});
	return sse([
		[null, chunk({ role: 'assistant', content: 'hel' })],
		[null, chunk({ content: 'lo' })],
		[null, chunk({}, 'stop')],
		[
			null,
			{
				id: 'chatcmpl-1',
				object: 'chat.completion.chunk',
				created: 1,
				model: MODEL,
				choices: [],
				usage: {
					prompt_tokens: 5,
					completion_tokens: 2,
					total_tokens: 7,
				},
			},
		],
		[null, '[DONE]'],
	]);
}

function anthropicMessage(body) {
	let content = [{ type: 'text', text: 'hello from claude' }];
	let stop = 'end_turn';
	if (body.tools) {
		content = [
			{
				type: 'tool_use',
				id: 'toolu_1',
				name: 'get_weather',
				input: { city: 'Paris' },
			},
		];
		stop = 'tool_use';
	}
	return {
		id: 'msg_1',
		type: 'message',
		role: 'assistant',
		model: MODEL,
		content,
		stop_reason: stop,
		stop_sequence: null,
		usage: { input_tokens: 5, output_tokens: 3 },
	};
}

function anthropicStream() {
	const delta = (text) => ({
		type: 'content_block_delta',
		index: 0,
		delta: { type: 'text_delta', text },
	});
	return sse([
		[
			'message_start',
			{
				type: 'message_start',
				message: {
					id: 'msg_1',
					type: 'message',
					role: 'assistant',
					model: MODEL,
					content: [],
					stop_reason: null,
					stop_sequence: null,
					usage: { input_tokens: 5, output_tokens: 1 },
				},
			},
		],
		[
			'content_block_start',
			{
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text', text: '' },
			},
		],
		['content_block_delta', delta('hel')],
		['content_block_delta', delta('lo')],
		['content_block_stop', { type: 'content_block_stop', index: 0 }],
		[
			'message_delta',
			{
				type: 'message_delta',
				delta: { stop_reason: 'end_turn', stop_sequence: null },
				usage: { output_tokens: 2 },
			},
		],
		['message_stop', { type: 'message_stop' }],
	]);
}

async function withWireDouble(run) {
	const recorded = [];
	const server = http.createServer((req, res) => {
		let raw = '';
		req.on('data', (chunk) => (raw += chunk));
		req.on('end', () => {
			const path = req.url.split('?', 1)[0];
			const body = raw ? JSON.parse(raw) : {};
			recorded.push({ method: req.method, path, body });
			const json = (value) => {
				res.writeHead(200, { 'content-type': 'application/json' });
				res.end(JSON.stringify(value));
			};
			const stream = (text) => {
				res.writeHead(200, { 'content-type': 'text/event-stream' });
				res.end(text);
			};
			if (req.method === 'GET' && path === '/v1/models') {
				if (req.headers['anthropic-version']) {
					json({
						data: [
							{
								type: 'model',
								id: MODEL,
								display_name: 'Matrix',
								created_at: '2024-06-01T00:00:00Z',
							},
						],
						has_more: false,
						first_id: MODEL,
						last_id: MODEL,
					});
				} else {
					json({
						object: 'list',
						data: [
							{
								id: MODEL,
								object: 'model',
								created: 1,
								owned_by: 'agent-rt',
							},
						],
					});
				}
			} else if (path === '/v1/chat/completions') {
				if (body.stream) stream(openAIStream());
				else json(openAICompletion(body));
			} else if (path === '/v1/messages') {
				if (body.stream) stream(anthropicStream());
				else json(anthropicMessage(body));
			} else if (path === '/v1/messages/count_tokens') {
				json({ input_tokens: 42 });
			} else {
				res.writeHead(404, { 'content-type': 'application/json' });
				res.end('{"error":"not found"}');
			}
		});
	});
	await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
	try {
		await run(`http://127.0.0.1:${server.address().port}`, recorded);
	} finally {
		server.closeAllConnections?.();
		await new Promise((resolve) => server.close(resolve));
	}
}

function request(extra = {}) {
	return {
		messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
		model: MODEL,
		...extra,
	};
}

function bodies(recorded, path) {
	return recorded.filter((entry) => entry.path === path).map((e) => e.body);
}

async function collect(iterable) {
	const events = [];
	for await (const event of iterable) events.push(event);
	return events;
}

test(
	'installed OpenAI SDK completes, streams, and validates',
	{ skip: !installed('openai') && 'openai is not installed' },
	async () => {
		await withWireDouble(async (baseUrl, recorded) => {
			const settings = {
				baseUrl: `${baseUrl}/v1`,
				apiKey: 'test-key',
				defaultModel: MODEL,
				transport: 'sdk',
				websocket: false,
			};
			assert.deepEqual(
				await validateOpenAIProviderSettings(settings, {
					checkDefaultModel: true,
				}),
				[MODEL],
			);
			const provider = new OpenAIModelProvider(settings);
			try {
				const response = await provider.complete(
					request({
						maxOutputTokens: 64,
						reasoning: { effort: 'low' },
					}),
				);
				assert.equal(
					response.message.content[0].text,
					'hello from openai',
				);
				assert.equal(response.usage?.inputTokens, 5);

				const toolResponse = await provider.complete(
					request({ tools: [WEATHER_TOOL] }),
				);
				assert.equal(toolResponse.finishReason, 'tool_calls');
				assert.deepEqual(toolResponse.message.toolCalls[0].arguments, {
					city: 'Paris',
				});

				const events = await collect(provider.stream(request()));
				assert.equal(
					events
						.filter((event) => event.type === 'text_delta')
						.map((event) => event.text)
						.join(''),
					'hello',
				);
				const completed = events.find(
					(event) => event.type === 'completed',
				);
				assert.equal(completed?.response?.usage?.outputTokens, 2);

				assert.deepEqual(
					(await provider.listModels()).map((entry) => entry.id),
					[MODEL],
				);
			} finally {
				await provider.close();
			}
			const [first, , streamBody] = bodies(
				recorded,
				'/v1/chat/completions',
			);
			assert.equal(first.reasoning_effort, 'low');
			assert.equal(first.max_completion_tokens, 64);
			assert.deepEqual(streamBody.stream_options, {
				include_usage: true,
			});
		});
	},
);

test(
	'installed Anthropic SDK completes, streams, validates, and counts tokens',
	{
		skip:
			!installed('@anthropic-ai/sdk') &&
			'@anthropic-ai/sdk is not installed',
	},
	async () => {
		await withWireDouble(async (baseUrl, recorded) => {
			const settings = {
				baseUrl,
				apiKey: 'test-key',
				defaultModel: MODEL,
			};
			assert.deepEqual(
				await validateAnthropicProviderSettings(settings, {
					checkDefaultModel: true,
				}),
				[MODEL],
			);
			const provider = new AnthropicModelProvider(settings);
			try {
				const response = await provider.complete(
					request({
						maxOutputTokens: 64,
						reasoning: {
							effort: 'low',
							thinking: 'enabled',
							budgetTokens: 1024,
						},
					}),
				);
				assert.equal(
					response.message.content[0].text,
					'hello from claude',
				);
				assert.equal(response.usage?.outputTokens, 3);

				const toolResponse = await provider.complete(
					request({ tools: [WEATHER_TOOL] }),
				);
				assert.equal(
					toolResponse.message.toolCalls[0].name,
					'get_weather',
				);
				assert.deepEqual(toolResponse.message.toolCalls[0].arguments, {
					city: 'Paris',
				});

				const events = await collect(provider.stream(request()));
				assert.equal(
					events
						.filter((event) => event.type === 'text_delta')
						.map((event) => event.text)
						.join(''),
					'hello',
				);

				assert.equal(await provider.countTokens(request()), 42);
				assert.deepEqual(
					(await provider.listModels()).map((entry) => entry.id),
					[MODEL],
				);
			} finally {
				await provider.close();
			}
			const first = bodies(recorded, '/v1/messages')[0];
			assert.deepEqual(first.thinking, {
				type: 'enabled',
				budget_tokens: 1024,
			});
			assert.deepEqual(first.output_config, { effort: 'low' });
			assert.equal(
				bodies(recorded, '/v1/messages/count_tokens')[0].model,
				MODEL,
			);
		});
	},
);
