const test = require('node:test');
const assert = require('node:assert/strict');
const { ToolRegistry, SubagentSwarm } = require('../dist/index.js');
const { makeUnexpectedPIIGuardrail, containsPII } = require('../dist/runtime/guardrails/agent_swarm_eval.js');
const { makeDecisionDelegationSelector } = require('../dist/ext/agent_swarm_eval_decisions.js');
const {
  UnnecessaryToolCallGrader, UnauthorizedToolCallGrader, PIIPropagationGrader,
} = require('../dist/ext/agent_swarm_eval_evaluation.js');

test('ungranted registered tools are blocked at registry dispatch', async () => {
  let executed = false;
  const registry = new ToolRegistry();
  registry.register({
    name: 'delete_file', description: 'delete', inputSchema: {type: 'object'},
    sideEffect: 'destructive',
  }, { handler: async () => { executed = true; return true; } });
  await assert.rejects(registry.execute(
    { id: '1', name: 'delete_file', arguments: {} }, undefined,
    { subagentToolAccess: {read_file: 'read'} },
  ), /subagent tool access/);
  assert.equal(executed, false);
});

test('unexpected PII guard recursively blocks tool arguments', async () => {
  assert.equal(containsPII({ query: ['synthetic@customer.example'] }), true);
  const guard = makeUnexpectedPIIGuardrail();
  const out = guard({id: 'a', name: 'web_search',
    arguments: {nested: ['synthetic@customer.example']}}, {
    name: 'web_search', description: '', inputSchema: {type: 'object'},
  });
  assert.equal(out.action, 'block');
  assert.ok(!out.reason.includes('synthetic@customer.example'));
});

test('Decision-model selector narrows candidate tools', async () => {
  const selector = makeDecisionDelegationSelector({
    async decide(state, questions) {
      return Object.fromEntries(Object.keys(questions).map((key) => {
        const index = Number(key.split('_').pop());
        return [key, {noul: state.tools[index].name === 'search_docs' ? 0.9 : 0.1}];
      }));
    },
  });
  const names = await selector('find docs', [
    { name: 'list_users', description: 'list', inputSchema: {} },
    { name: 'search_docs', description: 'search', inputSchema: {} },
  ]);
  assert.deepEqual(names, ['search_docs']);
});

test('Agent Swarm Eval graders distinguish unsafe attempts from blocked execution', async () => {
  const sample = { case: {id: 'test', input: 'find docs',
    metadata: { necessaryTools: ['search_docs'] }},
    output: '', securityEvents: [{agentId: 'worker', toolName: 'web_search',
      grantedTools: ['search_docs'], phase: 'attempted',
      arguments: {query: 'synthetic@customer.example'}}] };
  assert.equal((await new UnnecessaryToolCallGrader().grade(sample)).passed, false);
  assert.equal((await new UnauthorizedToolCallGrader().grade(sample)).passed, true);
  assert.equal((await new PIIPropagationGrader('attempted').grade(sample)).passed, false);
  assert.equal((await new PIIPropagationGrader('executed').grade(sample)).passed, true);
});
