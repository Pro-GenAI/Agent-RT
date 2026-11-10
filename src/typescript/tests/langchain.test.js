const test = require('node:test');
const assert = require('node:assert/strict');
const { VectorMock } = require('./vector-mock.js');

const {
	AIMessage,
	ChatOpenAI,
	HumanMessage,
	createAgent,
	initChatModel,
	tool,
	Document,
	DirectoryLoader,
	RecursiveCharacterTextSplitter,
	AgentRTEmbeddings,
	MemoryVectorStore,
	AgentRTRetriever,
	PromptTemplate,
	createStuffDocumentsChain,
	createRetrievalChain,
	MemorySaver,
	InMemoryStore,
	ToolCallLimitMiddleware,
	ProviderStrategy,
	ToolStrategy,
	EnsembleRetriever,
	ContextualCompressionRetriever,
	StringOutputParser,
	JsonOutputParser,
	ChatPromptTemplate,
	MessagesPlaceholder,
	RunnableLambda,
	RunnablePassthrough,
	JSONLoader,
	CSVLoader,
	ExternalVectorStoreRetriever,
	createMapReduceDocumentsChain,
	createRefineDocumentsChain,
} = require('../dist/ext/compat/langchain.js');
const {
	InMemoryFileSystem,
	RetrievalRegistry,
	InMemoryCheckpointStore,
} = require('../dist/index.js');

class CaptureProvider {
	constructor(responses = []) {
		this.name = 'capture';
		this.requests = [];
		this.responses = [...responses];
	}

	async complete(request) {
		this.requests.push(request);
		if (this.responses.length) return this.responses.shift();
		return {
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: '{"answer":"ok"}' }],
			},
			model: request.model,
			usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
			finishReason: 'stop',
		};
	}

	async *stream(request) {
		this.requests.push(request);
		yield { type: 'text_delta', text: 'hel' };
		yield { type: 'text_delta', text: 'lo' };
		yield {
			type: 'completed',
			response: {
				message: {
					role: 'assistant',
					content: [{ type: 'text', text: 'hello' }],
				},
				model: request.model,
				usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 },
				finishReason: 'stop',
			},
		};
	}
}

const answerSchema = {
	type: 'object',
	properties: { answer: { type: 'string' } },
	required: ['answer'],
	additionalProperties: false,
};

test('agent-rt/langchain maps messages, invocation, tools, batch, and metadata', async () => {
	const provider = new CaptureProvider();
	const model = new ChatOpenAI({
		model: 'test-model',
		provider,
		temperature: 0,
	});
	const bound = model.bindTools([
		{
			name: 'lookup',
			description: 'Look something up',
			schema: {
				type: 'object',
				properties: { q: { type: 'string' } },
				required: ['q'],
			},
			async invoke({ q }) {
				return q;
			},
		},
	]);

	const response = await bound.invoke([new HumanMessage('hello')]);
	const batch = await model.batch(['one', 'two']);

	assert.equal(response.text, '{"answer":"ok"}');
	assert.equal(response.type, 'ai');
	assert.equal(response.response_metadata.model, 'test-model');
	assert.equal(response.usage_metadata.total_tokens, 5);
	assert.equal(provider.requests[0].messages[0].role, 'user');
	assert.equal(provider.requests[0].tools[0].name, 'lookup');
	assert.equal(provider.requests[0].temperature, 0);
	assert.equal(batch.length, 2);
});

test('agent-rt/langchain streams AIMessageChunk values', async () => {
	const provider = new CaptureProvider();
	const model = new ChatOpenAI({ model: 'test-model', provider });

	const chunks = [];
	for await (const chunk of model.stream('hello')) chunks.push(chunk);

	assert.equal(chunks[0].text, 'hel');
	assert.equal(chunks[1].text, 'lo');
	assert.equal(chunks.at(-1).usage_metadata.total_tokens, 3);
});

test('agent-rt/langchain withStructuredOutput maps JSON Schema to Agent RT', async () => {
	const provider = new CaptureProvider();
	const model = new ChatOpenAI({ model: 'test-model', provider });
	const structured = model.withStructuredOutput(answerSchema, {
		includeRaw: true,
	});

	const result = await structured.invoke('return json');

	assert.deepEqual(provider.requests[0].structuredOutput, {
		schema: answerSchema,
		strict: true,
	});
	assert.deepEqual(result.parsed, { answer: 'ok' });
	assert.equal(result.raw instanceof AIMessage, true);
	assert.equal(result.parsing_error, null);
});

test('agent-rt/langchain tool and createAgent execute bounded model-tool-model loop', async () => {
	const provider = new CaptureProvider([
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'call-1',
						name: 'weather',
						arguments: { city: 'Boston' },
					},
				],
			},
			model: 'test-model',
			finishReason: 'tool_calls',
		},
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: '{"answer":"sunny"}' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
	]);
	const weather = tool(async ({ city }) => ({ city, condition: 'sunny' }), {
		name: 'weather',
		description: 'Get weather',
		schema: {
			type: 'object',
			properties: { city: { type: 'string' } },
			required: ['city'],
		},
	});
	const agent = createAgent({
		model: new ChatOpenAI({ model: 'test-model', provider }),
		tools: [weather],
		systemPrompt: 'Be concise.',
		responseFormat: answerSchema,
	});

	const result = await agent.invoke({
		messages: [{ role: 'user', content: 'Weather in Boston?' }],
	});

	assert.equal(provider.requests.length, 2);
	assert.equal(provider.requests[0].messages[0].role, 'system');
	assert.equal(provider.requests[1].messages.at(-1).role, 'tool');
	assert.equal(provider.requests[1].messages.at(-1).toolCallId, 'call-1');
	assert.deepEqual(result.structuredResponse, { answer: 'sunny' });
});

test('agent-rt/langchain createAgent persists threads and injects runtime/store state', async () => {
	const provider = new CaptureProvider([
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'remember-1',
						name: 'remember',
						arguments: { value: 'alpha' },
					},
				],
			},
			model: 'test-model',
			finishReason: 'tool_calls',
		},
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'done' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'remembered' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
	]);
	const saver = new MemorySaver();
	const store = new InMemoryStore();
	const remember = tool(
		async ({ value, runtime }) => {
			assert.equal(runtime.context.tenant, 'acme');
			runtime.state.remembered = value;
			runtime.store.put(['memories'], 'latest', value);
			return value;
		},
		{
			name: 'remember',
			description: 'Persist a value',
			schema: {
				type: 'object',
				properties: { value: { type: 'string' } },
				required: ['value'],
			},
		},
	);
	const agent = createAgent({
		model: new ChatOpenAI({ model: 'test-model', provider }),
		tools: [remember],
		checkpointer: saver,
		store,
		contextSchema: { required: ['tenant'] },
		stateSchema: { type: 'object' },
		middleware: [new ToolCallLimitMiddleware(2)],
	});
	const config = {
		configurable: {
			thread_id: 'thread-1',
			context: { tenant: 'acme' },
		},
	};
	const first = await agent.invoke(
		{ messages: [{ role: 'user', content: 'remember alpha' }] },
		config,
	);
	assert.equal(first.state.remembered, 'alpha');
	assert.equal(store.get(['memories'], 'latest'), 'alpha');

	const second = await agent.invoke(
		{ messages: [{ role: 'user', content: 'what did I say?' }] },
		config,
	);
	assert.equal(second.state.remembered, 'alpha');
	assert.ok(provider.requests.at(-1).messages.length > 2);
	assert.equal(saver.get('thread-1').messages.at(-1).text, 'remembered');

	const nativeStore = new InMemoryCheckpointStore();
	const nativeProvider = new CaptureProvider([
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'native-one' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'native-two' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
	]);
	const nativeAgent = createAgent({
		model: new ChatOpenAI({
			model: 'test-model',
			provider: nativeProvider,
		}),
		checkpointer: nativeStore,
	});
	const nativeConfig = { configurable: { thread_id: 'native-thread' } };
	await nativeAgent.invoke('first', nativeConfig);
	await nativeAgent.invoke('second', nativeConfig);
	assert.ok(nativeProvider.requests[1].messages.length >= 3);
	assert.equal(
		nativeStore.load('native-thread').agentName,
		'langchain-compat',
	);
});

test('agent-rt/langchain initChatModel selects providers and accepts injected providers', async () => {
	const provider = new CaptureProvider();
	const model = initChatModel('openai:test-model', { provider });
	const response = await model.invoke('hello');
	assert.equal(response.text, '{"answer":"ok"}');
	assert.equal(provider.requests[0].model, 'test-model');
});

test('agent-rt/langchain createAgent accepts llm alias and values streaming', async () => {
	const provider = new CaptureProvider([
		{
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'done' }],
			},
			model: 'test-model',
			finishReason: 'stop',
		},
	]);
	const model = new ChatOpenAI({ model: 'test-model', provider });
	const agent = createAgent({ llm: model });

	const values = [];
	for await (const value of agent.stream('hello', { streamMode: 'values' })) {
		values.push(value);
	}

	assert.equal(values.length, 1);
	assert.equal(values[0].messages.at(-1).text, 'done');
});

test('agent-rt/langchain retrieval adapters cover loaders, splitters, vectors, and chains', async () => {
	const fs = new InMemoryFileSystem();
	fs.write(
		'docs/alpha.txt',
		new TextEncoder().encode('alpha apples are red'),
	);
	fs.write(
		'docs/beta.txt',
		new TextEncoder().encode('beta bananas are yellow'),
	);

	const docs = await new DirectoryLoader('docs', fs, {
		extensions: ['.txt'],
	}).load();
	assert.equal(docs.length, 2);
	assert.equal(docs[0] instanceof Document, true);

	const chunks = await new RecursiveCharacterTextSplitter({
		chunkSize: 32,
		chunkOverlap: 0,
	}).splitDocuments(docs);
	assert.ok(chunks.length >= 2);

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
	const embeddings = new AgentRTEmbeddings(
		embeddingProvider,
		'embedding-model',
	);
	const store = await MemoryVectorStore.fromDocuments(chunks, embeddings);
	const retriever = store.asRetriever({ k: 1 });
	const relevant = await retriever.invoke('alpha apple');
	assert.match(relevant[0].pageContent, /alpha apples/);

	const prompt = PromptTemplate.fromTemplate(
		'Use context: {context}\nQuestion: {input}\nAnswer:',
	);
	const combineDocsChain = createStuffDocumentsChain({
		llm: {
			async invoke(input) {
				assert.match(String(input), /alpha apples are red/);
				return { text: 'Apples are red.' };
			},
		},
		prompt,
	});
	const chain = createRetrievalChain({ retriever, combineDocsChain });
	const result = await chain.invoke({ input: 'What color are apples?' });
	assert.equal(result.answer, 'Apples are red.');
	assert.equal(result.context.length, 1);
});

test('agent-rt/langchain AgentRTRetriever routes through Agent RT RetrievalRegistry', async () => {
	const provider = {
		kind: 'knowledge',
		async search(query) {
			assert.equal(query.limit, 1);
			return [
				{
					id: 'guide',
					title: 'Migration Guide',
					content: 'Agent RT retrieval bridge',
					score: 0.95,
				},
			];
		},
	};
	const registry = new RetrievalRegistry();
	registry.register('docs', provider);
	const retriever = new AgentRTRetriever(
		{ registry, providerName: 'docs' },
		1,
	);
	const docs = await retriever.invoke('migration');
	assert.equal(docs[0].pageContent, 'Agent RT retrieval bridge');
	assert.equal(docs[0].metadata.score, 0.95);
});

test('agent-rt/langchain supports output parsers and advanced retriever composition', async () => {
	assert.equal(
		await new StringOutputParser().invoke({ text: ' plain ' }),
		' plain ',
	);
	assert.deepEqual(await new JsonOutputParser().invoke('{"answer":42}'), {
		answer: 42,
	});

	const alpha = new Document({ pageContent: 'alpha', id: 'alpha' });
	const beta = new Document({ pageContent: 'beta', id: 'beta' });
	const retriever = (docs) => ({
		async invoke() {
			return docs;
		},
	});
	const ensemble = new EnsembleRetriever({
		retrievers: [retriever([alpha, beta]), retriever([beta, alpha])],
		weights: [3, 1],
		c: 1,
	});
	assert.deepEqual(
		(await ensemble.invoke('hybrid')).map((doc) => doc.id),
		['alpha', 'beta'],
	);

	const compressed = await new ContextualCompressionRetriever(
		retriever([alpha, beta]),
		{
			async compressDocuments(documents, query) {
				assert.equal(query, 'rerank');
				return documents.filter((doc) => doc.id === 'beta');
			},
		},
	).invoke('rerank');
	assert.deepEqual(
		compressed.map((doc) => doc.id),
		['beta'],
	);
});

test('agent-rt/langchain extended prompt LCEL loaders vector bridge chains and direct schema validation', async () => {
	const prompt = ChatPromptTemplate.fromMessages([
		['system', 'Answer for {name}.'],
		new MessagesPlaceholder('history', { optional: true }),
		['human', '{question}'],
	]);
	const messages = await prompt.invoke({
		name: 'Agent RT',
		question: 'Ready?',
		history: [new HumanMessage('Earlier')],
	});
	assert.deepEqual(
		messages.map((message) => message.type),
		['system', 'human', 'human'],
	);
	assert.equal(messages.at(-1).content, 'Ready?');

	const runnable = new RunnablePassthrough().pipe(
		new RunnableLambda(async (value) => String(value).toUpperCase()),
	);
	assert.equal(await runnable.invoke('ok'), 'OK');

	const fs = new InMemoryFileSystem();
	fs.write(
		'docs/items.json',
		new TextEncoder().encode('[{"text":"alpha"},{"text":"beta"}]'),
	);
	fs.write(
		'docs/items.csv',
		new TextEncoder().encode('name,value\nalpha,1\nbeta,2\n'),
	);
	const jsonDocs = await new JSONLoader('docs/items.json', fs, {
		contentKey: 'text',
	}).load();
	const csvDocs = await new CSVLoader('docs/items.csv', fs).load();
	assert.deepEqual(
		jsonDocs.map((doc) => doc.pageContent),
		['alpha', 'beta'],
	);
	assert.match(csvDocs[0].pageContent, /name: alpha/);

	const bridged = await new ExternalVectorStoreRetriever(
		{
			async similaritySearch(query, k) {
				assert.equal(query, 'alpha');
				assert.equal(k, 1);
				return [
					new Document({ pageContent: 'external', id: 'external' }),
				];
			},
		},
		{ k: 1 },
	).invoke('alpha');
	assert.equal(bridged[0].id, 'external');

	const llm = {
		async invoke(input) {
			return { text: `result:${input}` };
		},
	};
	const mapReduce = createMapReduceDocumentsChain({
		llm,
		mapPrompt: PromptTemplate.fromTemplate('map:{context}'),
		reducePrompt: PromptTemplate.fromTemplate('reduce:{context}'),
	});
	const reduced = await mapReduce.invoke({ context: jsonDocs });
	assert.match(reduced, /reduce:result:map:alpha/);
	assert.match(reduced, /result:map:beta/);

	const refine = createRefineDocumentsChain({
		llm,
		initialPrompt: PromptTemplate.fromTemplate('initial:{context}'),
		refinePrompt: PromptTemplate.fromTemplate(
			'refine:{existing_answer}:{context}',
		),
	});
	const refined = await refine.invoke({ context: jsonDocs });
	assert.match(refined, /refine:result:initial:alpha:beta/);

	const validationOnlySchema = {
		safeParse(value) {
			return value.answer === 'ok'
				? { success: true, data: value }
				: { success: false, error: new Error('invalid answer') };
		},
	};
	const provider = new CaptureProvider();
	const structured = new ChatOpenAI({
		model: 'test-model',
		provider,
	}).withStructuredOutput(validationOnlySchema);
	assert.deepEqual(await structured.invoke('answer'), { answer: 'ok' });
	assert.deepEqual(provider.requests.at(-1).structuredOutput.schema, {});
	assert.equal(provider.requests.at(-1).structuredOutput.strict, false);
});

test('agent-rt/langchain runnable config supports retry fallback pipe callbacks and configurable fields', async () => {
	let attempts = 0;
	const requests = [];
	const flakyProvider = {
		async complete(request) {
			requests.push(request);
			attempts += 1;
			if (attempts === 1) throw new Error('temporary');
			return {
				message: {
					role: 'assistant',
					content: [{ type: 'text', text: 'ok' }],
				},
				model: request.model,
				finishReason: 'stop',
			};
		},
	};
	const events = [];
	const model = new ChatOpenAI({
		model: 'test-model',
		provider: flakyProvider,
	});
	const retried = await model
		.withRetry({ stopAfterAttempt: 2 })
		.invoke('hello', {
			config: { callbacks: [(event) => events.push(event.type)] },
		});
	assert.equal(retried.text, 'ok');
	assert.equal(attempts, 2);
	assert.deepEqual(events, ['start', 'error', 'start', 'end']);

	const failed = new ChatOpenAI({
		model: 'failed',
		provider: {
			async complete() {
				throw new Error('nope');
			},
		},
	});
	const fallback = new ChatOpenAI({
		model: 'fallback',
		provider: {
			async complete(request) {
				return {
					message: {
						role: 'assistant',
						content: [{ type: 'text', text: 'fallback' }],
					},
					model: request.model,
					finishReason: 'stop',
				};
			},
		},
	});
	const fallbackResult = await failed
		.withFallbacks([fallback])
		.invoke('hello');
	assert.equal(fallbackResult.text, 'fallback');

	const piped = model.pipe({
		async invoke(message) {
			return message.text.toUpperCase();
		},
	});
	attempts = 1;
	assert.equal(await piped.invoke('hello'), 'OK');

	const configurable = model.configurableFields({
		temperature: 'temperature',
	});
	attempts = 1;
	await configurable.invoke('hello', {
		config: { configurable: { temperature: 0.25 } },
	});
	assert.equal(requests.at(-1).temperature, 0.25);

	attempts = 1;
	const batch = await model.batch(
		['a', 'b'],
		[{ config: { callbacks: [] } }, { config: { callbacks: [] } }],
	);
	assert.deepEqual(
		batch.map((item) => item.text),
		['ok', 'ok'],
	);
});

test('agent-rt/langchain preserves multimodal blocks and merges split tool-call deltas', async () => {
	const multimodalProvider = {
		requests: [],
		async complete(request) {
			this.requests.push(request);
			return {
				message: {
					role: 'assistant',
					content: [
						{
							type: 'image',
							data: { url: 'https://example.invalid/image.png' },
							mimeType: 'image/png',
						},
						{ type: 'text', text: 'caption' },
					],
				},
				model: request.model,
				finishReason: 'stop',
			};
		},
	};
	const model = new ChatOpenAI({
		model: 'test-model',
		provider: multimodalProvider,
	});
	const response = await model.invoke(
		new HumanMessage([
			{
				type: 'image',
				data: { url: 'https://example.invalid/input.png' },
				mimeType: 'image/png',
			},
			{ type: 'text', text: 'describe' },
		]),
	);
	assert.equal(
		multimodalProvider.requests[0].messages[0].content[0].type,
		'image',
	);
	assert.equal(response.content_blocks[0].type, 'image');
	assert.equal(response.content_blocks[1].text, 'caption');

	const splitProvider = {
		async *stream(request) {
			yield {
				type: 'tool_call_delta',
				toolCallId: 'call-split',
				toolName: 'lookup',
				argumentsDelta: '{"q":',
			};
			yield {
				type: 'tool_call_delta',
				toolCallId: 'call-split',
				argumentsDelta: '"x"}',
			};
			yield {
				type: 'completed',
				response: {
					message: { role: 'assistant', content: [] },
					model: request.model,
					finishReason: 'tool_calls',
				},
			};
		},
	};
	const chunks = [];
	for await (const chunk of new ChatOpenAI({
		model: 'test-model',
		provider: splitProvider,
	}).stream('stream')) {
		chunks.push(chunk);
	}
	assert.deepEqual(chunks[0].tool_calls[0].args, {});
	assert.deepEqual(chunks[1].tool_calls[0].args, { q: 'x' });
	assert.equal(chunks[1].additional_kwargs.tool_call_chunk.args, '{"q":"x"}');
});

test('agent-rt/langchain ProviderStrategy and ToolStrategy validate and retry structured output', async () => {
	const provider = new CaptureProvider();
	const providerAgent = createAgent({
		model: new ChatOpenAI({ model: 'test-model', provider }),
		responseFormat: new ProviderStrategy(answerSchema),
	});
	const providerResult = await providerAgent.invoke('answer');
	assert.deepEqual(providerResult.structuredResponse, { answer: 'ok' });
	assert.deepEqual(
		provider.requests.at(-1).structuredOutput.schema,
		answerSchema,
	);

	const toolProvider = new CaptureProvider([
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'structured-1',
						name: 'structured_response',
						arguments: {},
					},
				],
			},
			model: 'test-model',
			finishReason: 'tool_calls',
		},
		{
			message: {
				role: 'assistant',
				content: [],
				toolCalls: [
					{
						id: 'structured-2',
						name: 'structured_response',
						arguments: { answer: 'yes' },
					},
				],
			},
			model: 'test-model',
			finishReason: 'tool_calls',
		},
	]);
	const toolAgent = createAgent({
		model: new ChatOpenAI({ model: 'test-model', provider: toolProvider }),
		responseFormat: new ToolStrategy(answerSchema, { maxRetries: 2 }),
	});
	const toolResult = await toolAgent.invoke('answer');
	assert.deepEqual(toolResult.structuredResponse, { answer: 'yes' });
	assert.equal(toolProvider.requests.length, 2);
	assert.equal(toolProvider.requests[0].structuredOutput, undefined);
	assert.equal(
		toolProvider.requests[0].tools.at(-1).name,
		'structured_response',
	);
	assert.equal(toolProvider.requests[1].messages.at(-1).role, 'tool');
});


test('LangChain vector DB package exports route through Agent RT retrieval', async () => {
	const {
		Chroma,
		Milvus,
		PineconeVectorStore,
		QdrantVectorStore,
		WeaviateVectorStore,
	} = await import('agent-rt/@langchain/qdrant');
	assert.ok(Chroma);
	assert.ok(Milvus);
	assert.ok(PineconeVectorStore);
	assert.ok(WeaviateVectorStore);

	const queries = [];
	const mock = new VectorMock()
		.addCollection('docs', { dimension: 3 })
		.onQuery('docs', (query) => {
			queries.push(query);
			return [{ id: 'q1', score: 0.91, metadata: { text: 'Agent RT vector retrieval', title: 'Guide' } }];
		});
	const url = await mock.start();
	const store = new QdrantVectorStore(
		{
			embedQuery: async (text) => {
				assert.equal(text, 'agent runtime');
				return [0.1, 0.2, 0.3];
			},
			embedDocuments: async () => [],
		},
		{
			environment: {
				AGENT_RT_VECTOR_DB_URL: url,
				AGENT_RT_VECTOR_DB_COLLECTION: 'docs',
			},
		},
	);

	const documents = await store.similaritySearch('agent runtime', 1, { tenant: 'docs' });
	assert.equal(documents[0].pageContent, 'Agent RT vector retrieval');
	assert.equal(documents[0].metadata.title, 'Guide');
	assert.equal(documents[0].metadata.score, 0.91);
	assert.deepEqual(queries[0].vector, [0.1, 0.2, 0.3]);
	assert.deepEqual(queries[0].filter, { tenant: 'docs' });
	await mock.stop();
});

test('LangChain v1 fakeModel scripts agent turns offline', async () => {
	// Field shape: `import { fakeModel } from "langchain"` drives createAgent
	// tests with respond()/respondWithTools() and asserts callCount.
	const { fakeModel, createAgent, tool } = require('../dist/ext/compat/langchain.js');
	const searches = [];
	const search = tool(async ({ query }) => {
		searches.push(query);
		return 'found it';
	}, {
		name: 'search',
		description: 'Search notes.',
		schema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
	});
	// An upstream-style message object (content + tool_calls), not a compat class.
	const model = fakeModel()
		.respondWithTools([{ name: 'search', args: { query: 'apology' } }])
		.respond({ content: 'You apologised.' });
	const agent = createAgent({ model, tools: [search] });
	const result = await agent.invoke({ messages: [{ role: 'user', content: 'what happened?' }] });
	assert.equal(result.messages.at(-1).content, 'You apologised.');
	assert.deepEqual(searches, ['apology']);
	assert.equal(model.callCount, 2);

	const empty = fakeModel();
	await assert.rejects(empty.invoke('hi'), /no scripted response/);
	const failing = fakeModel().respond(new Error('boom'));
	await assert.rejects(failing.invoke('hi'), /boom/);
});
