const test = require('node:test');
const assert = require('node:assert/strict');
const {
  SubagentSwarm, CapabilityGrant, ExecutionBudget,
} = require('../dist/index.js');

const config = (name) => ({
  name, instructions: 'Finish assigned task', model: { model: 'mock' },
});
const tools = [
  { name: 'lookup', description: 'Read', inputSchema: { type: 'object' }, sideEffect: 'read' },
  { name: 'edit', description: 'Write', inputSchema: { type: 'object' }, sideEffect: 'write' },
];
const ok = (name) => ({
  messages: [], terminationReason: 'completed',
  turns: 1, toolCalls: 0, totalTokens: 0,
  structuredOutput: { agent: name },
});

test('subagents inherit chat, respect depth/fanout and keep parallel trace lineage', async () => {
  let active = 0;
  let peak = 0;
  const invoked = [];
  const invoker = {
    async invoke(agent, invocation) {
      invoked.push({ agent, invocation });
      if (agent.name === 'reviewer') {
        assert.deepEqual(agent.toolPolicy.allowed, []);
        assert.deepEqual(invocation.allowedTools, []);
        assert.equal(invocation.subagents, undefined);
        return ok('reviewed');
      }
      active++;
      peak = Math.max(peak, active);
      try {
        assert.equal(invocation.messages[0].content[0].text, 'all prior chat');
        if (agent.name === 'worker') {
          assert.equal(invocation.subagents.depth, 2);
          const grand = await invocation.subagents.spawn(
            'grandchild task', config('grandchild'),
            { allowedTools: { lookup: 'read' } },
          );
          assert.equal(grand.response, '{"agent":"grandchild"}');
          assert.ok(grand.review.response);
          await assert.rejects(
            invocation.subagents.spawn('second', config('grandchild'), { allowedTools: {} }),
            /only one child/,
          );
        } else if (agent.name === 'grandchild') {
          assert.equal(invocation.subagents.depth, 3);
          await assert.rejects(
            invocation.subagents.spawn('no more', config('worker'), { allowedTools: {} }),
            /depth limit/,
          );
        }
        await Promise.resolve();
        return ok(agent.name);
      } finally {
        active--;
      }
    },
  };
  const budget = new ExecutionBudget({ maxSubagents: 4 });
  const swarm = new SubagentSwarm(invoker, {
    chatHistory: [{ role: 'user', content: [{ type: 'text', text: 'all prior chat' }] }],
    toolDefinitions: tools,
    parentPermissions: { lookup: 'read', edit: 'write' },
    reviewer: config('reviewer'),
    budget,
  });
  const outcomes = await Promise.all([
    swarm.deploy('work A', config('worker'), { allowedTools: { lookup: 'read' } }),
    swarm.deploy('work B', config('worker'), { allowedTools: { lookup: 'read' } }),
  ]);
  assert.ok(outcomes.every((o) => o.response === '{"agent":"worker"}'));
  assert.ok(outcomes.every((o) => o.review.response === '{"agent":"reviewed"}'));
  assert.ok(peak >= 2);
  assert.equal(swarm.active, 0);
  assert.equal(budget.subagents, 4);
  const spans = swarm.traces.list({ traceId: swarm.traceId });
  assert.equal(spans.filter((s) => s.kind === 'subagent').length, 4);
  assert.equal(spans.filter((s) => s.attributes?.role === 'reviewer').length, 4);
  assert.ok(spans.every((s) => s.status === 'ok'));
  assert.equal(swarm.metrics.total('subagent.started'), 4);
  assert.equal(swarm.metrics.total('subagent.review.completed'), 4);
  swarm.close();
});

test('subagent permissions are narrowed and reviewer failures are monitored', async () => {
  const swarm = new SubagentSwarm({
    async invoke(agent) {
      if (agent.name === 'worker') throw new Error('worker failure');
      throw new Error('reviewer failure');
    },
  }, {
    chatHistory: [], toolDefinitions: tools,
    parentPermissions: { lookup: 'read', edit: 'write' },
    reviewer: config('reviewer'),
    capabilityGrant: new CapabilityGrant({ tools: ['lookup', 'edit'] }),
  });
  await assert.rejects(
    swarm.deploy('bad', config('worker'), { allowedTools: { edit: 'read' } }),
    /not read-only/,
  );
  await assert.rejects(
    swarm.deploy('bad', config('worker'), { allowedTools: { lookup: 'write' } }),
    /escalate/,
  );
  assert.equal(swarm.metrics.total('subagent.started'), 0);
  assert.equal(swarm.metrics.total('subagent.denied', { reason: 'permission' }), 2);
  const tool = swarm.root.asTool(config('worker'), { allowedTools: { lookup: 'read' } });
  assert.equal(tool.definition.sideEffect, 'read');
  const toolOutcome = await tool.handler({ task: 'delegated' }, { services: {}, requestContext: {}, limits: {} });
  assert.equal(toolOutcome.error, 'Error: worker failure');
  const out = await swarm.deploy('task', config('worker'), { allowedTools: {} });
  assert.equal(out.error, 'Error: worker failure');
  assert.equal(out.review.error, 'Error: reviewer failure');
  assert.equal(swarm.metrics.total('subagent.failed'), 2);
  assert.equal(swarm.metrics.total('subagent.review.failed'), 2);
  assert.equal(swarm.traces.list().filter((s) => s.status === 'error').length, 4);
  swarm.close();
  assert.throws(
    () => new SubagentSwarm(
      { invoke: async () => ok('x') },
      {
        chatHistory: [], toolDefinitions: tools, parentPermissions: { edit: 'write' },
        reviewer: config('reviewer'),
        capabilityGrant: new CapabilityGrant({ tools: ['lookup'] }),
      },
    ),
    /parent grant/,
  );
});
