const test = require('node:test');
const assert = require('node:assert/strict');
const { ToolRegistry, registerReaTermTool } = require('../dist/index.js');

test('ReaTerm registers a read-only tool with host-selected root and literal argv', async () => {
  const calls = [];
  const registry = new ToolRegistry();
  const registered = await registerReaTermTool(registry, {
    root: '/trusted/project',
    timeoutMs: 5000,
    client: { run: async (command, ...args) => {
      calls.push([command, args]);
      return 'contents';
    }},
  });
  assert.equal(registered.definition.sideEffect, 'read');
  assert.equal(registered.definition.timeoutMs, 6000);
  assert.equal(registered.definition.inputSchema.properties.root, undefined);
  assert.equal(await registry.execute({
    id: '1', name: 'reaterm', arguments: { command: 'cat', args: ['a b.txt'] },
  }), 'contents');
  assert.deepEqual(calls, [['cat', ['a b.txt']]]);
  await assert.rejects(registered.handler({ command: 'rm', args: [] }), /read-only/);
  await assert.rejects(registered.handler({ command: 'cat', args: [1] }), /array of strings/);
});

test('ReaTerm rejects invalid host settings', async () => {
  const registry = new ToolRegistry();
  await assert.rejects(registerReaTermTool(registry, { root: '' }), /root/);
  await assert.rejects(
    registerReaTermTool(registry, { root: '/safe', timeoutMs: Infinity }),
    /timeoutMs/,
  );
});
