const test = require('node:test');
const assert = require('node:assert/strict');
const { ToolRegistry, Toolbase, TOOL_SEARCH_NAME } = require('../dist/index.js');

test('access lists hide tools from discovery and reject invented calls', async () => {
  const registry = new ToolRegistry();
  for (const [name, sideEffect] of [['read_ok','none'], ['read_hidden','none'],
      ['write_ok','consequential'], ['write_hidden','consequential']]) {
    registry.register({name, description: name + ' searchable catalog entry',
      inputSchema: {type: 'object'}, sideEffect}, {handler: async () => 'executed'});
  }
  const base = await Toolbase.initialize({toolRegistry: registry,
    allowlist: ['read_ok','read_hidden','write_ok','write_hidden'],
    blocklist: ['read_hidden'], read_allowlist: ['read_ok','read_hidden'],
    write_allowlist: ['write_ok','write_hidden'], write_blocklist: ['write_hidden']});
  assert.deepEqual(new Set(base.definitions().map(x => x.name)),
    new Set([TOOL_SEARCH_NAME, 'read_ok', 'write_ok']));
  const search = await base.execute({id: 'search', name: TOOL_SEARCH_NAME,
    arguments: {query: 'searchable catalog entry'}});
  assert.ok(search.tools.every(x => ['read_ok','write_ok'].includes(x.name)));
  for (const name of ['read_hidden','write_hidden','invented']) {
    await assert.rejects(base.execute({id: name, name, arguments: {}}));
  }
});

test('blocked MCP tools never register', async () => {
  const registry = new ToolRegistry();
  const base = await Toolbase.initialize({toolRegistry: registry,
    mcpClients: {remote: {listTools: async () =>
      [{name: 'blocked', inputSchema: {type: 'object'}}]}},
    blocklist: ['mcp.remote.blocked']});
  assert.deepEqual(base.definitions().map(x => x.name), [TOOL_SEARCH_NAME]);
  assert.throws(() => registry.get('mcp.remote.blocked'));
});

