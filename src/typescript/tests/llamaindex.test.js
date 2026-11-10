const assert = require('node:assert/strict');
const test = require('node:test');
const vectorMockModule = require('./vector-mock.js');

const { VectorMock } = vectorMockModule;

const {
	AgentInput,
	AgentOutput,
	AgentRTEmbedding,
	AgentRTRetriever,
	AgentStream,
	Context,
	Document,
	FunctionAgent,
	FunctionTool,
	Memory,
	OpenAI,
	PromptTemplate,
	ChatPromptTemplate,
	JSONOutputParser,
	CallbackManager,
	Settings,
	LlamaParseReader,
	LlamaCloudRetriever,
	LlamaCloudIndex,
	FaithfulnessEvaluator,
	RelevancyEvaluator,
	RetrieverQueryEngine,
	SentenceSplitter,
	SimpleDirectoryReader,
	ToolCall,
	ToolCallResult,
	VectorStoreIndex,
	Workflow,
	WorkflowEvent,
	StartEvent,
	StopEvent,
	InputRequiredEvent,
	HumanResponseEvent,
	step,
	ReActAgent,
	PlanningAgent,
	AgentWorkflow,
	agent,
	tool,
} = require('../dist/ext/compat/llamaindex.js');
const {
	InMemoryCheckpointStore,
	InMemoryFileSystem,
	RetrievalRegistry,
} = require('../dist/index.js');

class StreamingProvider {
	name = 'streaming';
	requests = [];

	async complete(request) {
		this.requests.push(request);
		return {
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'ok' }],
			},
			model: request.model,
			usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
			finishReason: 'stop',
		};
	}

	async *stream(request) {
		this.requests.push(request);
		yield { type: 'text_delta', text: 'o' };
		yield { type: 'text_delta', text: 'k' };
		yield {
			type: 'completed',
			response: {
				message: {
					role: 'assistant',
					content: [{ type: 'text', text: 'ok' }],
				},
				model: request.model,
				finishReason: 'stop',
			},
		};
	}
}

test('llamaindex LLM chat and completion support streaming', async () => {
	const provider = new StreamingProvider();
	const llm = new OpenAI({ model: 'test-model', provider });

	const chat = await llm.chat({
		messages: [{ role: 'user', content: 'hello' }],
		stream: true,
	});
	const chunks = [];
	for await (const chunk of chat) chunks.push(chunk);

	assert.deepEqual(
		chunks.map((chunk) => chunk.delta),
		['o', 'k', ''],
	);
	assert.equal(chunks.at(-1).message.content, 'ok');

	const completion = await llm.complete({ prompt: 'hello', stream: true });
	const completionChunks = [];
	for await (const chunk of completion) completionChunks.push(chunk);
	assert.equal(completionChunks.at(-1).text, 'ok');
});

class ToolProvider {
	name = 'tool';
	requests = [];

	async complete(request) {
		this.requests.push(request);
		return {
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'call-1',
						name: 'multiply',
						arguments: { a: 6, b: 7 },
					},
				],
			},
			model: request.model,
			finishReason: 'tool_calls',
		};
	}
}

test('llamaindex tool and FunctionTool translate into Agent RT tools', async () => {
	const multiply = tool(async ({ a, b }) => a * b, {
		name: 'multiply',
		description: 'Multiply two values',
		parameters: {
			type: 'object',
			properties: {
				a: { type: 'number' },
				b: { type: 'number' },
			},
			required: ['a', 'b'],
		},
	});
	const legacy = FunctionTool.fromDefaults(async ({ value }) => value, {
		name: 'identity',
		parameters: {
			type: 'object',
			properties: { value: { type: 'string' } },
		},
	});

	assert.equal((await multiply.call({ a: 2, b: 3 })).rawOutput, 6);
	assert.equal(legacy.toAgentRTTool().name, 'identity');

	const provider = new ToolProvider();
	const result = await new OpenAI({
		model: 'test-model',
		provider,
	}).predictAndCall([multiply], { userMsg: 'multiply' });

	assert.equal(result.toolCalls[0].rawOutput, 42);
	assert.equal(provider.requests[0].tools[0].name, 'multiply');
});

class StructuredProvider {
	name = 'structured';
	requests = [];

	async complete(request) {
		this.requests.push(request);
		return {
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: '{"answer":"yes"}' }],
			},
			model: request.model,
			finishReason: 'stop',
		};
	}
}

test('llamaindex structuredPredict and StructuredLLM use structured output', async () => {
	const schema = {
		type: 'object',
		properties: { answer: { type: 'string' } },
		required: ['answer'],
	};
	const provider = new StructuredProvider();
	const llm = new OpenAI({ model: 'test-model', provider });

	assert.deepEqual(await llm.structuredPredict(schema, 'answer'), {
		answer: 'yes',
	});

	const structured = llm.asStructuredLLM(schema);
	const response = await structured.complete({ prompt: 'answer' });
	assert.deepEqual(response.raw, { answer: 'yes' });
	assert.deepEqual(provider.requests[0].structuredOutput.schema, schema);
});

class AgentProvider {
	name = 'agent';
	requests = [];

	async complete(request) {
		this.requests.push(request);
		if (this.requests.length === 1) {
			return {
				message: {
					role: 'assistant',
					content: [],
					toolCalls: [
						{
							id: 'weather-1',
							name: 'weather',
							arguments: { city: 'SF' },
						},
					],
				},
				model: request.model,
				finishReason: 'tool_calls',
			};
		}
		return {
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'Sunny in SF.' }],
			},
			model: request.model,
			finishReason: 'stop',
		};
	}
}

test('llamaindex workflow agent is awaitable and async iterable with memory/context', async () => {
	const weather = tool(async ({ city }) => `Sunny in ${city}.`, {
		name: 'weather',
		description: 'Weather lookup',
		parameters: {
			type: 'object',
			properties: { city: { type: 'string' } },
			required: ['city'],
		},
	});
	const provider = new AgentProvider();
	const llm = new OpenAI({ model: 'test-model', provider });
	const memory = Memory.fromDefaults({ sessionId: 'thread-1' });
	const workflow = agent({
		llm,
		tools: [weather],
		systemPrompt: 'Be concise.',
		streaming: false,
	});
	const context = new Context(workflow);

	const run = workflow.run('weather?', { memory, ctx: context });
	const events = [];
	for await (const event of run) events.push(event);
	const result = await run;

	assert.equal(result.data, 'Sunny in SF.');
	assert.ok(events.some((event) => event instanceof AgentInput));
	assert.ok(events.some((event) => event instanceof ToolCall));
	assert.ok(events.some((event) => event instanceof ToolCallResult));
	assert.ok(events.some((event) => event instanceof AgentStream));
	assert.ok(events.some((event) => event instanceof AgentOutput));
	assert.equal(provider.requests[0].messages[0].role, 'system');
	assert.equal(provider.requests[1].messages.at(-1).role, 'tool');
	assert.equal(provider.requests[1].messages.at(-1).toolCallId, 'weather-1');
	assert.equal(memory.getAll().at(-1).content, 'Sunny in SF.');
	assert.equal(await context.store.get('memory'), memory);
});

test('llamaindex RAG migration covers readers, ingestion, embeddings, vector retrieval, and query synthesis', async () => {
	const encoder = new TextEncoder();
	const fileSystem = new InMemoryFileSystem();
	fileSystem.write('data/alpha.txt', encoder.encode('alpha apples are red'));
	fileSystem.write(
		'data/beta.txt',
		encoder.encode('beta bananas are yellow'),
	);

	const documents = new SimpleDirectoryReader({
		fileSystem,
		inputDir: 'data',
		requiredExts: ['.txt'],
	}).loadData();
	assert.equal(documents.length, 2);
	assert.ok(documents[0] instanceof Document);

	const embeddingProvider = {
		async embed(request) {
			const values = Array.isArray(request.input)
				? request.input
				: [request.input];
			return {
				data: values.map((value, index) => {
					const text = String(value).toLowerCase();
					return {
						index,
						embedding: [
							text.includes('alpha') || text.includes('apple')
								? 1
								: 0,
							text.includes('beta') || text.includes('banana')
								? 1
								: 0,
						],
					};
				}),
			};
		},
	};
	const embedding = new AgentRTEmbedding(
		embeddingProvider,
		'embedding-model',
	);
	const index = await VectorStoreIndex.fromDocuments(documents, {
		embedModel: embedding,
		transformations: [
			new SentenceSplitter({ chunkSize: 8, chunkOverlap: 0 }),
		],
	});

	const nodes = await index
		.asRetriever({ similarityTopK: 1 })
		.retrieve('alpha apple');
	assert.equal(nodes.length, 1);
	assert.match(nodes[0].node.text, /alpha apples/);

	const synthesizerLLM = {
		async complete({ prompt }) {
			assert.match(prompt, /alpha apples are red/);
			return { text: 'Apples are red.' };
		},
	};
	const response = await index
		.asQueryEngine({
			llm: synthesizerLLM,
			similarityTopK: 1,
		})
		.query('What color are apples?');
	assert.equal(response.toString(), 'Apples are red.');
	assert.equal(response.sourceNodes.length, 1);
});

test('llamaindex AgentRTRetriever bridges native Agent RT retrieval providers and registries', async () => {
	const provider = {
		kind: 'vector',
		async search(query) {
			assert.equal(query.limit, 1);
			return [
				{
					id: 'node-1',
					title: 'Guide',
					content: 'Agent RT migration guide',
					score: 0.95,
					metadata: { section: 'migration' },
				},
			];
		},
	};

	const direct = new AgentRTRetriever(provider, 1);
	const directResults = await direct.retrieve('migration');
	assert.equal(directResults[0].node.text, 'Agent RT migration guide');
	assert.equal(directResults[0].score, 0.95);

	const registry = new RetrievalRegistry();
	registry.register('docs', provider);
	const throughRegistry = new AgentRTRetriever(
		{
			registry,
			providerName: 'docs',
		},
		1,
	);
	const engine = new RetrieverQueryEngine({ retriever: throughRegistry });
	const response = await engine.query('migration');
	assert.match(response.toString(), /Agent RT migration guide/);
});

test('llamaindex tools support retries and return_error behavior', async () => {
	let attempts = 0;
	const retrying = tool(
		async () => {
			attempts += 1;
			if (attempts < 2) throw new Error('temporary');
			return 'ok';
		},
		{
			name: 'retrying',
			maxRetries: 1,
		},
	);
	const retried = await retrying.call({});
	assert.equal(retried.content, 'ok');
	assert.equal(attempts, 2);

	const recoverable = tool(
		async () => {
			throw new Error('recoverable failure');
		},
		{
			name: 'recoverable',
			errorBehavior: 'return_error',
		},
	);
	const failed = await recoverable.call({});
	assert.equal(failed.isError, true);
	assert.match(failed.content, /recoverable failure/);
});

test('llamaindex returnDirect tools stop the agent loop without another model call', async () => {
	const provider = new ToolProvider();
	const direct = tool(async ({ a, b }) => String(a * b), {
		name: 'multiply',
		returnDirect: true,
		parameters: {
			type: 'object',
			properties: {
				a: { type: 'number' },
				b: { type: 'number' },
			},
		},
	});
	const workflow = new FunctionAgent({
		llm: new OpenAI({ model: 'test-model', provider }),
		tools: [direct],
		streaming: false,
	});
	const result = await workflow.run('multiply');
	assert.equal(result.data, '42');
	assert.equal(provider.requests.length, 1);
});

test('llamaindex agent run cancel aborts active model requests', async () => {
	const provider = {
		name: 'abortable',
		async complete(request) {
			return await new Promise((resolve, reject) => {
				request.signal?.addEventListener(
					'abort',
					() => reject(new Error('provider aborted')),
					{ once: true },
				);
				if (request.signal?.aborted)
					reject(new Error('provider aborted'));
			});
		},
	};
	const workflow = new FunctionAgent({
		llm: new OpenAI({ model: 'test-model', provider }),
		streaming: false,
	});
	const run = workflow.run('cancel me');
	run.cancel();
	await assert.rejects(Promise.resolve(run), /aborted/);
});

test('llamaindex prompt templates, Settings, callbacks, and token counting preserve common migration shapes', async () => {
	Settings.reset();
	const prompt = new PromptTemplate('Hello {name} from {city}');
	assert.equal(
		prompt.format({ name: 'Ada', city: 'London' }),
		'Hello Ada from London',
	);
	assert.equal(
		prompt.partial({ name: 'Ada' }).format({ city: 'Paris' }),
		'Hello Ada from Paris',
	);

	const chatPrompt = new ChatPromptTemplate([
		{ role: 'system', content: 'You are {role}.' },
		{ role: 'user', content: 'Question: {question}' },
	]);
	assert.deepEqual(
		chatPrompt.formatMessages({ role: 'concise', question: 'Why?' }),
		[
			{ role: 'system', content: 'You are concise.' },
			{ role: 'user', content: 'Question: Why?' },
		],
	);

	const parser = new JSONOutputParser();
	assert.deepEqual(parser.parse('{"ok":true}'), { ok: true });

	const callbackManager = new CallbackManager();
	const events = [];
	callbackManager.on((event) => events.push(event.type));
	const provider = {
		name: 'counting',
		async complete(request) {
			return {
				message: {
					role: 'assistant',
					content: [{ type: 'text', text: 'ok' }],
				},
				model: request.model,
				finishReason: 'stop',
			};
		},
		async countTokens(request) {
			assert.equal(request.model, 'test-model');
			return 7;
		},
	};
	const llm = new OpenAI({ model: 'test-model', provider, callbackManager });
	await llm.complete({ prompt: 'hello' });
	assert.deepEqual(events, ['llm-start', 'llm-end']);
	assert.equal(await llm.countTokens('hello'), 7);

	Settings.tokenizer = (text) => text.split(/\s+/).filter(Boolean);
	const fallback = new OpenAI({
		model: 'test-model',
		provider: new StreamingProvider(),
	});
	assert.equal(await fallback.countTokens('one two three'), 3);
	Settings.reset();
});

test('llamaindex FunctionAgent and context/memory helpers preserve common shapes', async () => {
	const provider = new StreamingProvider();
	const llm = new OpenAI({ model: 'test-model', provider });
	const workflow = new FunctionAgent({ llm, streaming: false });
	const memory = Memory.fromDefaults({ tokenLimit: 100 });

	memory.put({ role: 'user', content: 'hello' });
	assert.equal(memory.get()[0].content, 'hello');
	assert.equal(memory.tokenLimit, 100);

	const context = new Context(workflow, { count: 1 });
	await context.store.set('name', 'Ada');
	const restored = Context.fromJSON(workflow, context.toJSON());
	assert.equal(restored.state.count, 1);
	assert.equal(await restored.store.get('name'), 'Ada');

	const storedMemory = Memory.fromDefaults({ sessionId: 'persisted' });
	storedMemory.put({ role: 'user', content: 'remember me' });
	await context.store.set('memory', storedMemory);
	const serialized = context.serialize();
	const deserialized = Context.deserialize(workflow, serialized);
	const recoveredMemory = await deserialized.store.get('memory');
	assert.ok(recoveredMemory instanceof Memory);
	assert.equal(recoveredMemory.sessionId, 'persisted');
	assert.equal(recoveredMemory.getAll()[0].content, 'remember me');
});

test('llamaindex hosted adapters require explicit Agent RT file/retrieval contracts', async () => {
	const fs = new InMemoryFileSystem();
	fs.write('docs/input.txt', new TextEncoder().encode('hello hosted parser'));
	const parsed = await new LlamaParseReader({
		fileSystem: fs,
		inputDir: 'docs',
		parser: async ({ path, data }) => ({
			id: `${path}#parsed`,
			text: new TextDecoder().decode(data).toUpperCase(),
			metadata: { parser: 'injected' },
		}),
	}).loadData();
	assert.equal(parsed[0].text, 'HELLO HOSTED PARSER');
	assert.equal(parsed[0].metadata.parser, 'injected');

	const provider = {
		kind: 'vector',
		async search() {
			return [
				{
					id: 'cloud-1',
					title: 'Cloud result',
					content: 'retrieved through Agent RT',
					score: 0.9,
				},
			];
		},
	};
	const retriever = new LlamaCloudRetriever(provider, 1);
	const index = new LlamaCloudIndex(retriever);
	const nodes = await index.asRetriever().retrieve('query');
	assert.equal(nodes[0].node.text, 'retrieved through Agent RT');
});

test('llamaindex Settings feed agents, RAG defaults, and evaluators', async () => {
	Settings.reset();
	const llm = {
		async complete({ prompt }) {
			if (prompt.includes('Score answer')) {
				return {
					text: '{"score":0.9,"passing":true,"feedback":"supported"}',
				};
			}
			return { text: 'synthesized' };
		},
		async chat() {
			return { message: { role: 'assistant', content: 'done' } };
		},
	};
	Settings.llm = llm;

	const embeddingProvider = {
		async embed(request) {
			const values = Array.isArray(request.input)
				? request.input
				: [request.input];
			return {
				data: values.map((_, index) => ({ index, embedding: [1, 0] })),
			};
		},
	};
	Settings.embedModel = new AgentRTEmbedding(embeddingProvider, 'embed');

	const index = await VectorStoreIndex.fromDocuments(
		[new Document({ text: 'settings document' })],
		{},
	);
	const response = await index
		.asQueryEngine({ similarityTopK: 1 })
		.query('settings');
	assert.equal(response.toString(), 'synthesized');

	const relevance = await new RelevancyEvaluator().evaluate({
		query: 'settings',
		response,
	});
	assert.equal(relevance.passing, true);
	assert.equal(relevance.score, 0.9);

	const faithfulness = await new FaithfulnessEvaluator().evaluate({
		response,
	});
	assert.equal(faithfulness.passing, true);

	const configuredAgent = new FunctionAgent({ streaming: false });
	assert.equal(configuredAgent.llm, llm);
	Settings.reset();
});

test('llamaindex Workflow supports fan-out fan-in HITL middleware and checkpoints', async () => {
	class NumberEvent extends WorkflowEvent {}
	const checkpoints = new InMemoryCheckpointStore();
	const hooks = [];
	const workflow = new Workflow({
		checkpointStore: checkpoints,
		checkpointId: 'workflow-1',
		middleware: [
			{
				beforeStep(ctx, event, name) {
					hooks.push('before:' + name);
					return event;
				},
				afterStep(ctx, event, result, name) {
					hooks.push('after:' + name);
					return result;
				},
			},
		],
		steps: [
			step(
				StartEvent,
				async () => [new NumberEvent(1), new NumberEvent(2)],
				{ name: 'start' },
			),
			step(
				NumberEvent,
				async (ctx, event) => {
					const pair = ctx.collectEvents(event, [
						NumberEvent,
						NumberEvent,
					]);
					if (!pair) return undefined;
					ctx.state.sum = pair.reduce(
						(total, item) => total + Number(item.data),
						0,
					);
					return new InputRequiredEvent('approve?');
				},
				{ name: 'collect' },
			),
			step(
				HumanResponseEvent,
				async (ctx, event) => {
					ctx.state.approved = event.response;
					return new StopEvent(ctx.state.sum);
				},
				{ name: 'human' },
			),
		],
	});
	const run = workflow.run();
	for await (const event of run)
		if (event instanceof InputRequiredEvent) run.respond('yes');
	assert.equal(await run, 3);
	assert.ok(hooks.includes('before:start'));
	const resumed = Context.loadCheckpoint(workflow, checkpoints, 'workflow-1');
	assert.equal(resumed.state.sum, 3);
	assert.equal(resumed.state.approved, 'yes');
});

test('llamaindex ReAct planning and multi-agent handoffs execute', async () => {
	const multiply = tool(async ({ a, b }) => a * b, {
		name: 'multiply',
		description: 'Multiply',
		parameters: {
			type: 'object',
			properties: { a: { type: 'number' }, b: { type: 'number' } },
			required: ['a', 'b'],
		},
	});
	class SequenceProvider {
		constructor(texts) {
			this.texts = [...texts];
			this.requests = [];
		}
		async complete(request) {
			this.requests.push(request);
			return {
				message: {
					role: 'assistant',
					content: [{ type: 'text', text: this.texts.shift() }],
				},
				model: request.model,
				finishReason: 'stop',
			};
		}
	}
	const reactProvider = new SequenceProvider([
		'Thought: calculate\nAction: multiply\nAction Input: {"a":6,"b":7}',
		'Thought: done\nAnswer: 42',
	]);
	const react = new ReActAgent({
		llm: new OpenAI({ model: 'test-model', provider: reactProvider }),
		tools: [multiply],
		streaming: false,
	});
	assert.equal((await react.run('6*7?')).data, '42');
	assert.match(
		reactProvider.requests[1].messages.at(-1).content[0].text,
		/Observation: 42/,
	);
	const planningProvider = new SequenceProvider(['one', 'two']);
	const planner = new PlanningAgent({
		llm: new OpenAI({ model: 'test-model', provider: planningProvider }),
		streaming: false,
		planner: async () => ['one task', 'two task'],
	});
	assert.equal((await planner.runPlan('work')).length, 2);
	const handoffProvider = {
		async complete(request) {
			return {
				message: {
					role: 'assistant',
					content: [],
					toolCalls: [
						{
							id: 'h1',
							name: 'handoff_to_agent',
							arguments: {
								agentName: 'writer',
								message: 'finish',
							},
						},
					],
				},
				model: request.model,
				finishReason: 'tool_calls',
			};
		},
	};
	const researcher = new FunctionAgent({
		name: 'researcher',
		llm: new OpenAI({ model: 'test-model', provider: handoffProvider }),
		streaming: false,
	});
	const writer = new FunctionAgent({
		name: 'writer',
		llm: new OpenAI({
			model: 'test-model',
			provider: new SequenceProvider(['final']),
		}),
		streaming: false,
	});
	const result = await AgentWorkflow.fromAgents([researcher, writer], {
		rootAgent: 'researcher',
	}).run('work');
	assert.equal(result.data, 'final');
	assert.equal(result.currentAgentName, 'writer');
});

test('llamaindex structured streams Zod-like schemas multimodal blocks and merged tool deltas', async () => {
	const zString = {
		_def: { typeName: 'ZodString' },
		safeParse(value) {
			return typeof value === 'string'
				? { success: true, data: value }
				: { success: false, error: new Error('string') };
		},
	};
	const zObject = {
		_def: { typeName: 'ZodObject', shape: () => ({ answer: zString }) },
		safeParse(value) {
			return value && typeof value.answer === 'string'
				? { success: true, data: value }
				: { success: false, error: new Error('answer') };
		},
	};
	const callbacks = new CallbackManager();
	const callbackEvents = [];
	callbacks.on((event) => callbackEvents.push(event.type));
	const provider = {
		requests: [],
		async complete(request) {
			this.requests.push(request);
			return {
				message: {
					role: 'assistant',
					content: [
						{
							type: 'image',
							data: { url: 'img' },
							mimeType: 'image/png',
						},
						{ type: 'text', text: '{"answer":"yes"}' },
					],
				},
				model: request.model,
				finishReason: 'stop',
			};
		},
		async *stream(request) {
			this.requests.push(request);
			yield { type: 'text_delta', text: '{"answer":' };
			yield { type: 'text_delta', text: '"yes"}' };
			yield {
				type: 'tool_call_delta',
				toolCallId: 'c1',
				toolName: 'lookup',
				argumentsDelta: '{"q":',
			};
			yield {
				type: 'tool_call_delta',
				toolCallId: 'c1',
				argumentsDelta: '"x"}',
			};
			yield {
				type: 'completed',
				response: {
					message: {
						role: 'assistant',
						content: [{ type: 'text', text: '{"answer":"yes"}' }],
					},
					model: request.model,
					finishReason: 'stop',
				},
			};
		},
	};
	const llm = new OpenAI({
		model: 'test-model',
		provider,
		callbackManager: callbacks,
	});
	const stream = await llm
		.asStructuredLLM(zObject)
		.complete({ prompt: 'answer', stream: true });
	const chunks = [];
	for await (const chunk of stream) chunks.push(chunk);
	assert.deepEqual(chunks.at(-1).raw, { answer: 'yes' });
	assert.deepEqual(provider.requests[0].structuredOutput.schema.required, [
		'answer',
	]);
	assert.ok(callbackEvents.includes('llm-stream'));
	const chat = await llm.chat({
		messages: [{ role: 'user', content: 'blocks' }],
	});
	assert.ok(Array.isArray(chat.message.content));
	assert.equal(chat.message.content[0].type, 'image');
	const toolStream = await llm.chat({
		messages: [{ role: 'user', content: 'tool' }],
		stream: true,
	});
	const toolChunks = [];
	for await (const chunk of toolStream)
		if (chunk.message.toolCalls?.length) toolChunks.push(chunk);
	assert.deepEqual(toolChunks.at(-1).message.toolCalls[0].input, { q: 'x' });
});


test('LlamaIndex vector DB package exports route through Agent RT retrieval', async () => {
	const {
		ChromaVectorStore,
		MilvusVectorStore,
		PineconeVectorStore,
		QdrantVectorStore,
		WeaviateVectorStore,
	} = await import('agent-rt/@llamaindex/qdrant');
	assert.ok(ChromaVectorStore);
	assert.ok(MilvusVectorStore);
	assert.ok(PineconeVectorStore);
	assert.ok(WeaviateVectorStore);

	const queries = [];
	const mock = new VectorMock()
		.addCollection('docs', { dimension: 2 })
		.onQuery('docs', (query) => {
			queries.push(query);
			return [{ id: 'q1', score: 0.88, metadata: { text: 'LlamaIndex compatibility over Agent RT', title: 'Migration' } }];
		});
	const url = await mock.start();
	const store = new QdrantVectorStore({
		environment: {
			AGENT_RT_VECTOR_DB_URL: url,
			AGENT_RT_VECTOR_DB_COLLECTION: 'docs',
		},
	});

	const nodes = await store.query({
		queryEmbedding: [0.1, 0.2],
		queryStr: 'migration',
		similarityTopK: 1,
		filters: { tenant: 'docs' },
	});
	assert.equal(nodes[0].node.text, 'LlamaIndex compatibility over Agent RT');
	assert.equal(nodes[0].node.metadata.title, 'Migration');
	assert.equal(nodes[0].score, 0.88);
	assert.deepEqual(queries[0].vector, [0.1, 0.2]);
	assert.deepEqual(queries[0].filter, { tenant: 'docs' });

	const { VectorStoreIndex } = await import('agent-rt/llamaindex');
	const index = VectorStoreIndex.fromVectorStore(store, {
		embedModel: {
			getQueryEmbedding: async (text) => {
				assert.equal(text, 'migration');
				return [0.1, 0.2];
			},
		},
	});
	const retrieved = await index.asRetriever({
		similarityTopK: 1,
		filters: { tenant: 'docs' },
	}).retrieve('migration');
	assert.equal(retrieved[0].node.text, 'LlamaIndex compatibility over Agent RT');
	await mock.stop();

	assert.throws(
		() => store.add([nodes[0].node]),
		/write\/upsert remains backend-native/,
	);
});
