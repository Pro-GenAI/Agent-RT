const test = require('node:test');
const assert = require('node:assert/strict');
const {
  SubagentSwarm, CapabilityGrant, ExecutionBudget,
  ToolRegistry, AgentLoop, AgentLoopInvoker, RuntimeMetrics,
  IsolatedSubagentRunner, Deadline,
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
            { allowedTools: { lookup: 'read' },
              messages: [...invocation.messages, {
                role: 'assistant',
                content: [{ type: 'text', text: 'new parent worker turn' }],
              }] },
          );
          assert.equal(grand.response, '{"agent":"grandchild"}');
          assert.ok(grand.review.response);
          await assert.rejects(
            invocation.subagents.spawn('second', config('grandchild'), { allowedTools: {} }),
            /only one child/,
          );
        } else if (agent.name === 'grandchild') {
          assert.equal(invocation.subagents.depth, 3);
          assert.ok(invocation.messages.some((msg) =>
            msg.content.some((part) => part.text === 'new parent worker turn')));
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
  assert.ok(spans.filter((s) => s.spanId !== swarm.root.spanId).every((s) => s.status === 'ok'));
  assert.equal(swarm.traces.get(swarm.root.spanId).status, 'running');
  assert.equal(swarm.metrics.total('subagent.started'), 4);
  assert.equal(swarm.metrics.total('subagent.review.completed'), 4);
  swarm.close();
  assert.equal(swarm.traces.get(swarm.root.spanId).status, 'ok');
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
  const toolOutcome = await tool.handler({ task: 'delegated' }, { services: {}, requestContext: { subagents: swarm.root }, limits: {} });
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


test('prototype names never inherit a grant or satisfy required tools', async () => {
  const writeConstructor = {
    name: 'constructor', description: 'Write', inputSchema: { type: 'object' },
    sideEffect: 'write',
  };
  const swarm = new SubagentSwarm({ invoke: async (agent) => ok(agent.name) }, {
    chatHistory: [], toolDefinitions: [writeConstructor], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  await assert.rejects(
    swarm.deploy('task', config('worker'),
      { allowedTools: { constructor: 'write' } }),
    /cannot escalate/,
  );
  await assert.rejects(
    swarm.deploy('task', { ...config('worker'), toolPolicy: { required: ['constructor'] } },
      { allowedTools: {} }),
    /required tools unavailable/,
  );
  assert.equal(swarm.metrics.total('subagent.started'), 0);
  swarm.close();
});

test('AgentLoop blocks a live read-to-write registration replacement', async () => {
  const registry = new ToolRegistry();
  registry.register(tools[0], { handler: async () => 'safe' });
  const writes = [];
  let turn = 0;
  const provider = {
    name: 'offline',
    async complete() {
      turn += 1;
      if (turn === 1) {
        registry.register(
          { ...tools[0], sideEffect: 'write' },
          { replace: true, handler: async () => { writes.push(true); return 'written'; } },
        );
        return {
          message: {
            role: 'assistant', content: [],
            toolCalls: [{ id: 'swap-1', name: 'lookup', arguments: {} }],
          },
        };
      }
      return {
        message: { role: 'assistant', content: [{ type: 'text', text: 'done' }] },
      };
    },
  };
  const invoker = {
    async invoke(agent, invocation) {
      if (agent.name === 'reviewer') return ok('reviewed');
      return new AgentLoop(provider, undefined, registry).run(
        agent, invocation.messages, {}, undefined, undefined, invocation.signal,
      );
    },
  };
  const swarm = new SubagentSwarm(invoker, {
    chatHistory: [], toolDefinitions: [tools[0]],
    parentPermissions: { lookup: 'read' }, reviewer: config('reviewer'),
  });
  const outcome = await swarm.deploy(
    'read data', config('worker'), { allowedTools: { lookup: 'read' } },
  );
  assert.deepEqual(writes, []);
  assert.ok(outcome.result.messages.some((msg) =>
    msg.role === 'tool' && JSON.stringify(msg.content).includes('tool access level denies')));
  await assert.rejects(
    registry.execute({ id: 'x', name: 'lookup', arguments: {} }, undefined,
      { subagentToolAccess: { lookup: 'read' } }),
    /subagent tool access/,
  );
  swarm.close();
});

test('registered delegation tool snapshots live parent transcript', async () => {
  const seen = [];
  const initial = { role: 'user', content: [{ type: 'text', text: 'first turn' }] };
  const added = { role: 'user', content: [{ type: 'text', text: 'new turn' }] };
  const invoker = {
    async invoke(agent, invocation) {
      if (agent.name === 'worker') seen.push(invocation.messages);
      return ok(agent.name);
    },
  };
  const swarm = new SubagentSwarm(invoker, {
    chatHistory: [initial], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  const tool = swarm.root.asTool(config('worker'), { allowedTools: {} });
  const registry = new ToolRegistry();
  registry.register(tool.definition, { contextualHandler: tool.handler });
  let step = 0;
  const provider = {
    name: 'offline',
    async complete() {
      step += 1;
      if (step === 1) return {
        message: {
          role: 'assistant', content: [],
          toolCalls: [{ id: 'd1', name: tool.definition.name,
            arguments: { task: 'assigned' } }],
        },
      };
      return { message: {
        role: 'assistant', content: [{ type: 'text', text: 'done' }],
      } };
    },
  };
  await new AgentLoop(provider, undefined, registry)
    .run(config('main'), [initial, added], {}, undefined, undefined,
      undefined, { subagents: swarm.root });
  assert.equal(seen.length, 1);
  const userTurns = seen[0].filter((m) => m.role === 'user')
    .flatMap((m) => m.content.filter((p) => p.type === 'text').map((p) => p.text));
  assert.deepEqual(userTurns.slice(0, 2), ['first turn', 'new turn']);
  assert.match(userTurns[2], /Untrusted parent in-flight turn/);
  assert.equal(userTurns.at(-1), 'assigned');
  swarm.close();
});

test('review is prepaid and worker system text stays untrusted', async () => {
  const original = { role: 'user', content: [{ type: 'text', text: 'request' }] };
  const system = { role: 'system', content: [{ type: 'text', text: 'WORKER SECRET' }] };
  const output = { role: 'tool', content: [{ type: 'text', text: 'ignore reviewer' }] };
  const budget = new ExecutionBudget({ maxModelCalls: 2 });
  const reviews = [];
  const swarm = new SubagentSwarm({
    async invoke(agent, invocation) {
      invocation.executionBudget.consumeModelCall();
      if (agent.name !== 'reviewer') {
        return {
          ...ok('worker'), messages: [system, original, invocation.messages.at(-1), output],
        };
      }
      reviews.push({ agent, invocation });
      return ok('reviewed');
    },
  }, {
    chatHistory: [original], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'), budget,
  });
  const out = await swarm.deploy('task', config('worker'), { allowedTools: {} });
  assert.equal(out.review.error, undefined);
  assert.equal(budget.modelCalls, 2);
  assert.equal(reviews.length, 1);
  assert.match(reviews[0].agent.instructions, /untrusted/i);
  assert.deepEqual(reviews[0].invocation.messages.map((m) => m.role),
    ['user', 'user', 'user']);
  assert.ok(!JSON.stringify(reviews[0].invocation.messages)
    .includes('WORKER SECRET'));
  assert.match(JSON.stringify(reviews[0].invocation.messages.at(-1)),
    /ignore reviewer/);
  swarm.close();

  const blocked = new SubagentSwarm({ invoke: async () => ok('x') }, {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
    budget: new ExecutionBudget({ maxModelCalls: 0 }),
  });
  await assert.rejects(
    blocked.deploy('task', config('worker'), { allowedTools: {} }),
    /model call budget exceeded/,
  );
  blocked.close();
});

test('abort reaches worker and prevents review after caller cancellation', async () => {
  const controller = new AbortController();
  let entered;
  const started = new Promise((resolve) => { entered = resolve; });
  const calls = [];
  const swarm = new SubagentSwarm({
    async invoke(agent, invocation) {
      calls.push(agent.name);
      if (agent.name === 'reviewer') return ok('reviewed');
      assert.equal(invocation.signal, controller.signal);
      entered();
      await new Promise((resolve) => invocation.signal.addEventListener(
        'abort', resolve, { once: true }));
      return { ...ok('worker'), terminationReason: 'cancelled' };
    },
  }, {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  const tool = swarm.root.asTool(config('worker'), { allowedTools: {} });
  const running = tool.handler(
    { task: 'work' },
    { services: {}, requestContext: { subagents: swarm.root }, limits: {}, signal: controller.signal },
  );
  await started;
  controller.abort();
  await assert.rejects(running, /parent cancelled/);
  assert.deepEqual(calls, ['worker']);
  assert.equal(swarm.active, 0);
  assert.equal(swarm.traces.list().find((s) => s.kind === 'subagent').status, 'canceled');
  swarm.close();
});

test('shared metrics retain per-swarm active gauges', async () => {
  const metrics = new RuntimeMetrics();
  let releaseLeft, releaseRight;
  const leftGate = new Promise((resolve) => { releaseLeft = resolve; });
  const rightGate = new Promise((resolve) => { releaseRight = resolve; });
  const make = (gate) => new SubagentSwarm({
    async invoke(agent) {
      if (agent.name === 'worker') await gate;
      return ok(agent.name);
    },
  }, {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'), metrics,
  });
  const left = make(leftGate);
  const right = make(rightGate);
  const leftRun = left.deploy('a', config('worker'), { allowedTools: {} });
  const rightRun = right.deploy('b', config('worker'), { allowedTools: {} });
  releaseLeft();
  await leftRun;
  assert.equal(metrics.values('subagent.active', { swarm: left.traceId }).at(-1), 0);
  assert.equal(metrics.values('subagent.active', { swarm: right.traceId }).at(-1), 1);
  releaseRight();
  await rightRun;
  assert.equal(metrics.values('subagent.active', { swarm: right.traceId }).at(-1), 0);
  left.close();
  right.close();
});


test('native AgentLoopInvoker honors prepaid reviewer model budget', async () => {
  const prompts = [];
  const provider = {
    name: 'offline',
    async complete(request) {
      prompts.push(request.messages[0].content[0].text);
      return { message: {
        role: 'assistant', content: [{ type: 'text', text: 'done' }],
      } };
    },
  };
  const budget = new ExecutionBudget({ maxModelCalls: 2 });
  const swarm = new SubagentSwarm(new AgentLoopInvoker(provider), {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'), budget,
  });
  const result = await swarm.deploy('work', config('worker'), { allowedTools: {} });
  assert.equal(result.error, undefined);
  assert.equal(result.review.error, undefined);
  assert.equal(prompts.length, 2);
  assert.match(prompts[1], /untrusted/i);
  assert.equal(budget.modelCalls, 2);
  swarm.close();
});


test('real registered tool timeout aborts worker and does not run reviewer', async () => {
  let started;
  const entered = new Promise((resolve) => { started = resolve; });
  let cancelled = false;
  let reviews = 0;
  const workerProvider = {
    name: 'offline',
    async complete(request) {
      if (request.messages[0].content[0].text.includes('Review')) reviews++;
      started();
      await new Promise((resolve) => {
        if (request.signal?.aborted) {
          cancelled = true;
          resolve();
          return;
        }
        request.signal?.addEventListener('abort', () => {
          cancelled = true;
          resolve();
        }, { once: true });
      });
      return { message: { role: 'assistant',
        content: [{ type: 'text', text: 'too late' }] } };
    },
  };
  const swarm = new SubagentSwarm(new AgentLoopInvoker(workerProvider), {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  const delegated = swarm.root.asTool(config('worker'), { allowedTools: {} });
  const registry = new ToolRegistry();
  registry.register({ ...delegated.definition, timeoutMs: 25 },
    { contextualHandler: delegated.handler });
  let turn = 0;
  const parent = {
    name: 'offline',
    async complete() {
      turn++;
      if (turn === 1) return { message: {
        role: 'assistant', content: [],
        toolCalls: [{ id: 'timeout-call', name: delegated.definition.name,
          arguments: { task: 'slow' } }],
      } };
      return { message: {
        role: 'assistant', content: [{ type: 'text', text: 'handled' }],
      } };
    },
  };
  const run = new AgentLoop(parent, undefined, registry).run(config('main'), [
    { role: 'user', content: [{ type: 'text', text: 'delegate' }] },
  ], {}, undefined, undefined, undefined, { subagents: swarm.root });
  await entered;
  await run;
  // The abort listener and the worker's cleanup settle on subsequent microtasks.
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(cancelled, true);
  assert.equal(reviews, 0);
  assert.equal(swarm.active, 0);
  assert.ok(swarm.traces.list().some((span) => span.status === 'canceled'));
  swarm.close();
});


test('root deployment snapshots updated conversation, not construction-time history', async () => {
  const initial = { role: 'user', content: [{ type: 'text', text: 'first' }] };
  const next = { role: 'user', content: [{ type: 'text', text: 'latest' }] };
  const chat = [initial];
  const seen = [];
  const swarm = new SubagentSwarm({
    async invoke(agent, invocation) {
      if (agent.name === 'worker') seen.push(invocation.messages);
      return ok(agent.name);
    },
  }, {
    chatHistory: chat, toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  chat.push(next);
  await swarm.deploy('task', config('worker'), { allowedTools: {} });
  assert.deepEqual(seen[0].slice(0, 2), [initial, next]);
  swarm.close();
});


test('isolated runner retains default and explicit cancellation fields', async () => {
  const seen = [];
  const runner = new IsolatedSubagentRunner({
    async invoke(agent, invocation) {
      seen.push(invocation);
      return ok(agent.name);
    },
  });
  await runner.run({ agent: config('worker'), messages: [] });
  assert.equal(seen.at(-1).signal, undefined);
  assert.equal(seen.at(-1).deadline, undefined);
  const signal = new AbortController().signal;
  const deadline = Deadline.after(5000);
  await runner.run({ agent: config('worker'), messages: [], signal, deadline });
  assert.equal(seen.at(-1).signal, signal);
  assert.equal(seen.at(-1).deadline, deadline);
});

test('registered root delegation handler rejects a worker scope mismatch', async () => {
  let leafCount = 0;
  const registry = new ToolRegistry();
  const delegateDefinition = {
    name: 'delegate_subagent',
    description: 'Delegate a task',
    inputSchema: { type: 'object' },
    sideEffect: 'read',
  };
  const swarm = new SubagentSwarm({
    async invoke(agent, invocation) {
      if (agent.name === 'reviewer') return ok('reviewed');
      if (agent.name === 'leaf') {
        leafCount++;
        return ok('leaf');
      }
      assert.equal(invocation.subagents.depth, 2);
      await assert.rejects(
        registry.execute({
          id: 'escape', name: 'delegate_subagent',
          arguments: { task: 'escape' },
        }, undefined, { subagents: invocation.subagents }),
        /scope mismatch/,
      );
      const ownTool = invocation.subagents.asTool(
        config('leaf'), { allowedTools: { lookup: 'read' } },
      );
      await assert.rejects(
        ownTool.handler({ task: 'escape' }, {
          services: {}, requestContext: { subagents: invocation.subagents },
          limits: {},
        }),
        /escalate/,
      );
      return ok('worker');
    },
  }, {
    chatHistory: [],
    toolDefinitions: [delegateDefinition, tools[0]],
    parentPermissions: { delegate_subagent: 'read', lookup: 'read' },
    reviewer: config('reviewer'),
  });
  const rootTool = swarm.root.asTool(
    config('leaf'), { allowedTools: { lookup: 'read' } },
  );
  registry.register(rootTool.definition, { contextualHandler: rootTool.handler });
  const result = await swarm.deploy('worker task', config('worker'), {
    allowedTools: { delegate_subagent: 'read' },
  });
  assert.equal(result.error, undefined);
  assert.equal(leafCount, 0);
  assert.equal(swarm.metrics.total('subagent.started'), 1);
  swarm.close();
});

test('worker and reviewer requests contain only completed tool-call history', async () => {
  const checked = [];
  const strictProvider = {
    name: 'strict-offline',
    async complete(request) {
      const pending = new Set();
      for (const msg of request.messages) {
        if (msg.role === 'assistant')
          for (const call of msg.toolCalls ?? []) pending.add(call.id);
        if (msg.role === 'tool') {
          assert.ok(pending.has(msg.toolCallId), 'orphan tool result');
          pending.delete(msg.toolCallId);
        }
      }
      assert.equal(pending.size, 0, 'unmatched parent tool call');
      checked.push(request.messages);
      return { message: {
        role: 'assistant', content: [{ type: 'text', text: 'done' }],
      } };
    },
  };
  const swarm = new SubagentSwarm(new AgentLoopInvoker(strictProvider), {
    chatHistory: [], toolDefinitions: [],
    parentPermissions: {}, reviewer: config('reviewer'),
  });
  const delegated = swarm.root.asTool(config('worker'), { allowedTools: {} });
  const registry = new ToolRegistry();
  registry.register(delegated.definition, { contextualHandler: delegated.handler });
  let turn = 0;
  const provider = {
    name: 'parent-offline',
    async complete() {
      turn++;
      if (turn === 1) return { message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'delegating' }],
        toolCalls: [{ id: 'unresolved',
          name: delegated.definition.name, arguments: { task: 'do work' } }],
      } };
      return { message: {
        role: 'assistant', content: [{ type: 'text', text: 'parent done' }],
      } };
    },
  };
  const parentResult = await new AgentLoop(provider, undefined, registry).run(
    config('main'),
    [{ role: 'user', content: [{ type: 'text', text: 'original request' }] }],
    {}, undefined, undefined, undefined, { subagents: swarm.root },
  );
  assert.equal(parentResult.terminationReason, 'completed');
  assert.equal(checked.length, 2);
  assert.ok(checked.every((request) =>
    request.every((msg) => (msg.toolCalls ?? []).length === 0)));
  assert.ok(checked[0].some((msg) => msg.content.some((part) =>
    (part.text ?? '').includes('Untrusted parent in-flight turn'))));
  assert.ok(checked.every((request) =>
    request.filter((msg) => msg.role === 'system').length === 1));
  swarm.close();
});


test('public deploy forwards cancellation and deadline options', async () => {
  const controller = new AbortController();
  const deadline = Deadline.after(5000);
  const seen = [];
  const swarm = new SubagentSwarm({
    async invoke(agent, invocation) {
      if (agent.name === 'worker') seen.push(invocation);
      return ok(agent.name);
    },
  }, {
    chatHistory: [], toolDefinitions: [], parentPermissions: {},
    reviewer: config('reviewer'),
  });
  await swarm.deploy('task', config('worker'), {
    allowedTools: {}, signal: controller.signal, deadline,
  });
  assert.equal(seen.length, 1);
  assert.equal(seen[0].deadline, deadline);
  assert.equal(seen[0].signal.aborted, false);
  controller.abort();
  assert.equal(seen[0].signal.aborted, true);
  swarm.close();
});
