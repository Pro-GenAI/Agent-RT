const test = require('node:test');
const assert = require('node:assert/strict');

const {
	Agent,
	Runner,
	handoff,
	run,
	tool,
} = require('../dist/ext/compat/openai_agents.js');

class QueueProvider {
	constructor(responses) {
		this.name = 'queue';
		this.responses = [...responses];
		this.requests = [];
	}

	async complete(request) {
		this.requests.push(request);
		if (this.responses.length === 0) {
			throw new Error('unexpected model request');
		}
		return this.responses.shift();
	}
}

function response(text = '', toolCalls = [], finishReason = 'stop') {
	return {
		message: {
			role: 'assistant',
			content: text ? [{ type: 'text', text }] : [],
			toolCalls,
		},
		model: 'test-model',
		finishReason,
	};
}

test('OpenAI Agents compatibility runs function tools through Agent RT', async () => {
	const provider = new QueueProvider([
		response(
			'',
			[
				{
					id: 'call-1',
					name: 'lookup',
					arguments: { topic: 'runtime' },
				},
			],
			'tool_calls',
		),
		response('Agent RT result'),
	]);
	const calls = [];
	const lookup = tool({
		name: 'lookup',
		description: 'Look up a topic',
		parameters: {
			type: 'object',
			properties: { topic: { type: 'string' } },
			required: ['topic'],
			additionalProperties: false,
		},
		execute: async ({ topic }) => {
			calls.push(topic);
			return 'found ' + topic;
		},
	});
	const agent = new Agent({
		name: 'Researcher',
		instructions: 'Use tools when useful.',
		model: 'test-model',
		provider,
		tools: [lookup],
	});

	const result = await run(agent, 'Research runtime');

	assert.equal(result.finalOutput, 'Agent RT result');
	assert.equal(result.lastAgent, agent);
	assert.deepEqual(calls, ['runtime']);
	assert.equal(provider.requests[0].tools[0].name, 'lookup');
});

test('OpenAI Agents compatibility exposes Runner, handoff, and package aliases', async () => {
	const specialistProvider = new QueueProvider([response('specialist')]);
	const parentProvider = new QueueProvider([response('parent')]);
	const specialist = new Agent({
		name: 'Specialist',
		model: 'test-model',
		provider: specialistProvider,
	});
	const parent = Agent.create({
		name: 'Parent',
		model: 'test-model',
		provider: parentProvider,
		handoffs: [handoff(specialist)],
	});

	const runner = new Runner();
	const result = await runner.run(parent, 'hello');

	assert.equal(result.finalOutput, 'parent');
	assert.equal(require('agent-rt/@openai/agents').Agent, Agent);
	assert.equal(require('agent-rt/openai-agents').Runner, Runner);
});

test('OpenAI Agents compatibility enforces agent input/output guardrails and maxTurns', async () => {
	const {
		InputGuardrailTripwireTriggered,
		OutputGuardrailTripwireTriggered,
		MaxTurnsExceededError,
	} = require('../dist/ext/compat/openai_agents.js');
	const seen = [];
	const travelOnly = {
		name: 'travel_only',
		execute: async ({ input, context, agent }) => {
			seen.push([input, context.context, agent.name]);
			return { outputInfo: { ok: false }, tripwireTriggered: !String(input).includes('trip') };
		},
	};
	const blocked = new QueueProvider([]);
	const planner = new Agent({ name: 'planner', provider: blocked, inputGuardrails: [travelOnly] });
	await assert.rejects(run(planner, 'write my essay', { context: 'ctx' }), (error) => {
		assert.ok(error instanceof InputGuardrailTripwireTriggered);
		assert.deepEqual(error.result.output.outputInfo, { ok: false });
		return true;
	});
	assert.equal(blocked.requests.length, 0); // tripwire fires before any model call
	assert.deepEqual(seen, [['write my essay', 'ctx', 'planner']]);

	const writer = new Agent({
		name: 'writer',
		provider: new QueueProvider([response('secret')]),
		outputGuardrails: [{ name: 'no_secret', execute: ({ agentOutput }) => ({ tripwireTriggered: agentOutput === 'secret' }) }],
	});
	await assert.rejects(run(writer, 'hi'), (error) => {
		assert.ok(error instanceof OutputGuardrailTripwireTriggered);
		assert.equal(error.result.agentOutput, 'secret');
		return true;
	});

	const lookup = tool({
		name: 'lookup',
		description: 'Look up.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		execute: async () => 'again',
	});
	const call = () => response('', [{ id: `c${Math.random()}`, name: 'lookup', arguments: {} }], 'tool_calls');
	const looper = new Agent({ name: 'looper', tools: [lookup], provider: new QueueProvider([call(), call(), call()]) });
	await assert.rejects(run(looper, 'go', { maxTurns: 2 }), MaxTurnsExceededError);
});
