const test = require('node:test');
const assert = require('node:assert/strict');

const {
	ApprovalManager,
	ApprovalRequiredError,
	ApprovalDeniedError,
	ChatApprovals,
} = require('../dist/index.js');
const {
	describeRequest,
} = require('../dist/runtime/authorization/chat_approvals.js');

const definition = (name = 'write_file') => ({
	name,
	description: 'd',
	inputSchema: {},
	sideEffect: 'consequential',
});
const tick = () => new Promise((resolve) => setImmediate(resolve));
const describe = (name, args = {}) =>
	describeRequest({ id: 'a', call: { id: 'c', name, arguments: args } });

test('chat approvals notify once and approve by reply', async () => {
	const sent = [];
	const manager = new ApprovalManager();
	const chat = new ChatApprovals(
		manager,
		(text, code) => void sent.push([text, code]),
	);
	const call = {
		id: 'c1',
		name: 'write_file',
		arguments: { path: 'docs/README.md' },
	};
	for (let i = 0; i < 2; i++) {
		assert.throws(
			() => manager.check(call, definition()),
			ApprovalRequiredError,
		);
	}
	await tick();
	assert.deepEqual(sent, [
		[describe('write_file', { path: 'docs/README.md' }), '1'],
	]);
	assert.equal(
		chat.handleReply('Approve'),
		'Approved: Agent requests {"arguments":{"path":"docs/README.md"},"tool":"write_file"}',
	);
	manager.check(call, definition());
});

test('chat approvals pick by code among several and deny', async () => {
	const sent = [];
	const manager = new ApprovalManager();
	const chat = new ChatApprovals(
		manager,
		(text, code) => void sent.push([text, code]),
	);
	const calls = ['A', 'B'].map((n) => ({
		id: 'Call_' + n,
		name: 'write_file',
		arguments: { path: n + '.md' },
	}));
	for (const call of calls) {
		assert.throws(
			() => manager.check(call, definition()),
			ApprovalRequiredError,
		);
	}
	await tick();
	assert.deepEqual(
		sent.map(([, code]) => code),
		['1', '2'],
	);
	assert.match(chat.handleReply('maybe'), /^Reply/);
	assert.match(chat.handleReply('yes'), /^Several/);
	assert.equal(
		chat.handleReply('deny #2'),
		'Denied: Agent requests {"arguments":{"path":"B.md"},"tool":"write_file"}',
	);
	assert.equal(chat.handleReply('deny 2'), 'No such pending approval.');
	assert.throws(
		() => manager.check(calls[1], definition()),
		ApprovalDeniedError,
	);
	assert.equal(
		chat.handleReply('y'),
		'Approved: Agent requests {"arguments":{"path":"A.md"},"tool":"write_file"}',
	);
});

test('chat approvals survive a dead channel', async () => {
	const manager = new ApprovalManager();
	new ChatApprovals(manager, () => Promise.reject(new Error('offline')));
	const call = { id: '1', name: 'rm', arguments: {} };
	assert.throws(
		() => manager.check(call, definition('rm')),
		ApprovalRequiredError,
	);
	await tick();
});

test('chat approval describes the entire call with visible controls', () => {
	const args = {
		path: 'docs/README.md',
		command: 'echo safe\nrm -rf /srv/data\x1b[2K\u202e',
	};
	const prompt = describe('run_shell', args);
	assert.doesNotMatch(prompt, /[\n\x1b\u202e]/);
	assert.deepEqual(
		JSON.parse(
			prompt.slice('Agent requests '.length, -', approve?'.length),
		),
		{ tool: 'run_shell', arguments: args },
	);
	assert.notEqual(
		describe('edit', { path: 'a/README.md' }),
		describe('edit', { path: 'b/README.py' }),
	);
	assert.ok(
		describe('run_shell', {
			command: 'x'.repeat(5000) + 'hidden-tail',
		}).includes('hidden-tail'),
	);
});
