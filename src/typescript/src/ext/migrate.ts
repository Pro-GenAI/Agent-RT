#!/usr/bin/env node
/**
 * `npx agent-rt migrate`: rewrite framework imports to Agent RT, keeping the change only if the tests agree.
 *
 * The project's tests run before and after the rewrite. The rewrite is kept when
 * every test that fails afterwards also failed before, in the same test and at the
 * same project source line; otherwise every rewritten file is restored.
 */
import { spawnSync } from 'node:child_process';
import {
	existsSync,
	mkdtempSync,
	readFileSync,
	readdirSync,
	rmSync,
	statSync,
	writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import {
	dirname,
	extname,
	isAbsolute,
	join,
	relative,
	resolve,
	sep,
} from 'node:path';
import { pose } from './artie.js';

// Kept in step with TS_SPECIFIER_RULES in scripts/migration_test/migration_test.py.
export const TS_SPECIFIER_RULES: Record<string, Array<[string, string]>> = {
	LangChain: [
		['@langchain/', 'agent-rt/@langchain/'],
		['langchain', 'agent-rt/langchain'],
	],
	LlamaIndex: [
		['@llamaindex/', 'agent-rt/@llamaindex/'],
		['llamaindex', 'agent-rt/llamaindex'],
	],
	OpenAI: [['openai', 'agent-rt/openai']],
	Anthropic: [
		['@anthropic-ai/sdk', 'agent-rt/@anthropic-ai/sdk'],
		['anthropic', 'agent-rt/anthropic'],
	],
	'OpenAI Agents': [['@openai/agents', 'agent-rt/@openai/agents']],
};

const SOURCE_EXTENSIONS = new Set([
	'.js',
	'.jsx',
	'.ts',
	'.tsx',
	'.mjs',
	'.cjs',
	'.mts',
	'.cts',
]);
const SKIP_DIRS = new Set([
	'.git',
	'.hg',
	'.svn',
	'.venv',
	'venv',
	'env',
	'node_modules',
	'dist',
	'build',
	'coverage',
	'.next',
	'.turbo',
	'.cache',
]);
const WALL_ISSUE_URL = 'https://github.com/Pro-GenAI/Agent-RT/issues/new';

/** The `agent-rt/...` specifier for an upstream one, if the installed agent-rt exports it. */
export function rewriteSpecifier(
	specifier: string,
	exported: Set<string>,
): string | null {
	for (const rules of Object.values(TS_SPECIFIER_RULES)) {
		for (const [source, destination] of rules) {
			const migrated = source.endsWith('/')
				? specifier.startsWith(source)
					? destination + specifier.slice(source.length)
					: null
				: specifier === source
					? destination
					: null;
			if (migrated !== null)
				return exported.has(migrated) ? migrated : null;
		}
	}
	return null;
}

// Only module specifiers attached to import/export/require() and to test-runner
// module mocks (`vi.mock("x")`, `jest.mock("x")`), which must follow the import they replace.
const SPECIFIER_PATTERN =
	/(?<head>\b(?:from\s*|import\s*\(\s*|require\s*\(\s*|import\s+|export\s+[^;]*?\s+from\s*|(?:vi|vitest|jest)\.(?:mock|doMock|unmock|doUnmock|importActual|requireActual|importMock|requireMock)\s*\(\s*))(?<quote>['"])(?<specifier>[^'"]+)\k<quote>/g;

export function rewriteImports(
	text: string,
	exported: Set<string>,
): { text: string; changes: number; packages: Set<string> } {
	let changes = 0;
	const packages = new Set<string>();
	const rewritten = text.replace(
		SPECIFIER_PATTERN,
		(match, head: string, quote: string, specifier: string) => {
			const migrated = rewriteSpecifier(specifier, exported);
			if (migrated === null || migrated === specifier) return match;
			changes += 1;
			packages.add(
				specifier
					.split('/')
					.slice(0, specifier.startsWith('@') ? 2 : 1)
					.join('/'),
			);
			return `${head}${quote}${migrated}${quote}`;
		},
	);
	return { text: rewritten, changes, packages };
}

function* sourceFiles(root: string): Generator<string> {
	for (const entry of readdirSync(root, { withFileTypes: true })) {
		const path = join(root, entry.name);
		if (entry.isDirectory()) {
			if (!SKIP_DIRS.has(entry.name)) yield* sourceFiles(path);
		} else if (
			entry.isFile() &&
			SOURCE_EXTENSIONS.has(extname(entry.name).toLowerCase())
		) {
			yield path;
		}
	}
}

export type Rewrite = {
	originals: Map<string, string>;
	migrated: Map<string, string>;
	changes: number;
	packages: Set<string>;
};

export function planRewrite(project: string, exported: Set<string>): Rewrite {
	const plan: Rewrite = {
		originals: new Map(),
		migrated: new Map(),
		changes: 0,
		packages: new Set(),
	};
	for (const path of sourceFiles(project)) {
		let original: string;
		try {
			original = readFileSync(path, 'utf8');
		} catch {
			continue;
		}
		const result = rewriteImports(original, exported);
		if (!result.changes) continue;
		plan.originals.set(path, original);
		plan.migrated.set(path, result.text);
		plan.changes += result.changes;
		for (const name of result.packages) plan.packages.add(name);
	}
	return plan;
}

/** The nearest `node_modules/<name>` folder at or above the project. */
function installedPackage(project: string, name: string): string | null {
	for (let folder = project; ; folder = dirname(folder)) {
		const candidate = join(folder, 'node_modules', ...name.split('/'));
		if (existsSync(join(candidate, 'package.json'))) return candidate;
		if (dirname(folder) === folder) return null;
	}
}

export function agentRtExports(packageDir: string): Set<string> {
	const manifest = JSON.parse(
		readFileSync(join(packageDir, 'package.json'), 'utf8'),
	) as {
		exports?: Record<string, unknown>;
	};
	return new Set(
		Object.keys(manifest.exports ?? {}).map(
			(key) => 'agent-rt' + key.slice(1),
		),
	);
}

export type SuiteRun = {
	exitCode: number;
	seconds: number;
	total: number;
	/** test id -> innermost project source location ("test/app.test.ts:12"); null when no report was read. */
	failures: Map<string, string | null> | null;
	output: string;
};

const JS_LOCATION =
	/(?:^|[\s(]|file:\/\/)([^\s()'"]+?\.[cm]?[jt]sx?):(\d+):\d+/gm;

/** The innermost stack frame inside the project (JS stacks list innermost first). */
export function failureLocation(text: string, project: string): string | null {
	for (const match of text.matchAll(JS_LOCATION)) {
		const path = isAbsolute(match[1])
			? relative(project, match[1])
			: match[1];
		const parts = path.split(/[\\/]/);
		if (path.startsWith('..') || parts.some((part) => SKIP_DIRS.has(part)))
			continue;
		return `${parts.join('/')}:${match[2]}`;
	}
	return null;
}

function xmlText(value: string): string {
	return value
		.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
		.replace(/&lt;/g, '<')
		.replace(/&gt;/g, '>')
		.replace(/&quot;/g, '"')
		.replace(/&apos;/g, "'")
		.replace(/&#(\d+);/g, (_, code: string) =>
			String.fromCodePoint(Number(code)),
		)
		.replace(/&amp;/g, '&');
}

function xmlAttribute(attributes: string, name: string): string {
	const match = new RegExp(`\\b${name}="([^"]*)"`).exec(attributes);
	return match ? xmlText(match[1]) : '';
}

/** Vitest's JUnit report. */
export function parseJunit(
	xml: string,
	project: string,
): { total: number; failures: Map<string, string | null> } {
	const failures = new Map<string, string | null>();
	let total = 0;
	for (const testCase of xml.matchAll(
		/<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/g,
	)) {
		total += 1;
		const problem =
			/<(failure|error)\b([^>]*?)(?:\/>|>([\s\S]*?)<\/\1>)/.exec(
				testCase[2] ?? '',
			);
		if (!problem) continue;
		const id = `${xmlAttribute(testCase[1], 'classname')}::${xmlAttribute(testCase[1], 'name')}`;
		failures.set(
			id,
			failureLocation(
				`${xmlText(problem[3] ?? '')}\n${xmlAttribute(problem[2], 'message')}`,
				project,
			),
		);
	}
	return { total, failures };
}

type JestReport = {
	testResults?: Array<{
		name: string;
		status?: string;
		message?: string;
		assertionResults?: Array<{
			fullName: string;
			status: string;
			failureMessages?: string[];
		}>;
	}>;
};

/** Jest's `--json` report. */
export function parseJestJson(
	report: JestReport,
	project: string,
): { total: number; failures: Map<string, string | null> } {
	const failures = new Map<string, string | null>();
	let total = 0;
	for (const suite of report.testResults ?? []) {
		const file = relative(project, suite.name).split(sep).join('/');
		const tests = suite.assertionResults ?? [];
		if (!tests.length && suite.status === 'failed') {
			total += 1;
			failures.set(file, failureLocation(suite.message ?? '', project));
		}
		for (const test of tests) {
			total += 1;
			if (test.status === 'failed') {
				failures.set(
					`${file}::${test.fullName}`,
					failureLocation(
						(test.failureMessages ?? []).join('\n'),
						project,
					),
				);
			}
		}
	}
	return { total, failures };
}

type Runner = {
	command: string[];
	read:
		| ((path: string, project: string) => ReturnType<typeof parseJunit>)
		| null;
};

function testRunner(project: string, report: string): Runner | null {
	const manifest = JSON.parse(
		readFileSync(join(project, 'package.json'), 'utf8'),
	) as {
		scripts?: Record<string, string>;
	};
	const script = manifest.scripts?.test;
	if (!script || /no test specified/.test(script)) return null;
	if (/\bvitest\b/.test(script)) {
		return {
			command: [
				'npm',
				'test',
				'--',
				'--reporter=junit',
				`--outputFile=${report}`,
			],
			read: (path, root) => parseJunit(readFileSync(path, 'utf8'), root),
		};
	}
	if (/\bjest\b/.test(script)) {
		return {
			command: ['npm', 'test', '--', '--json', `--outputFile=${report}`],
			read: (path, root) =>
				parseJestJson(
					JSON.parse(readFileSync(path, 'utf8')) as JestReport,
					root,
				),
		};
	}
	// Other runners give only an exit code, so a run that fails both times cannot be compared.
	return { command: ['npm', 'test'], read: null };
}

function runTests(runner: Runner, project: string, report: string): SuiteRun {
	rmSync(report, { force: true });
	const started = performance.now();
	const [command, ...args] = runner.command;
	// A parent `node --test` would otherwise swallow the project's own node:test results.
	const env = { ...process.env };
	delete env.NODE_TEST_CONTEXT;
	const result = spawnSync(command, args, {
		cwd: project,
		env,
		encoding: 'utf8',
		stdio: ['ignore', 'pipe', 'pipe'],
		maxBuffer: 64 * 1024 * 1024,
		shell: process.platform === 'win32',
	});
	const seconds = (performance.now() - started) / 1000;
	let parsed: ReturnType<typeof parseJunit> | null = null;
	if (runner.read) {
		try {
			parsed = runner.read(report, project);
		} catch {
			parsed = null;
		}
	}
	return {
		exitCode: result.status ?? 1,
		seconds,
		total: parsed?.total ?? 0,
		failures: parsed?.failures ?? null,
		output: `${result.stdout ?? ''}${result.stderr ?? ''}${result.error ? String(result.error) : ''}`,
	};
}

/** Keep the migration only if nothing fails afterwards that did not fail the same way before. */
export function compare(
	before: SuiteRun,
	after: SuiteRun,
): { ok: boolean; problems: string[] } {
	if (after.exitCode === 0) return { ok: true, problems: [] };
	if (!after.failures || !before.failures) {
		return {
			ok: false,
			problems: [
				'the test report could not be read, so the failures cannot be compared',
			],
		};
	}
	const problems = [...after.failures]
		.filter(
			([id, location]) =>
				!before.failures!.has(id) ||
				before.failures!.get(id) !== location,
		)
		.map(([id, location]) => `${id} (${location ?? 'unknown location'})`);
	if (!problems.length && after.exitCode !== before.exitCode) {
		problems.push(
			`test run exit code changed from ${before.exitCode} to ${after.exitCode}`,
		);
	}
	return { ok: !problems.length, problems };
}

function folderSize(path: string | null): number {
	if (!path) return 0;
	let total = 0;
	for (const entry of readdirSync(path, { withFileTypes: true })) {
		const child = join(path, entry.name);
		// Nested node_modules belong to other packages.
		if (entry.isDirectory() && entry.name !== 'node_modules')
			total += folderSize(child);
		else if (entry.isFile()) total += statSync(child).size;
	}
	return total;
}

const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;
const change = (before: number, after: number) =>
	before
		? `  (${after >= before ? '+' : ''}${Math.round(((after - before) / before) * 100)}%)`
		: '';

function card(
	kept: boolean,
	plan: Rewrite,
	before: SuiteRun,
	after: SuiteRun,
	sizes: [number, number],
): string {
	const summary = (run: SuiteRun) =>
		run.failures
			? `${run.total - run.failures.size}/${run.total} passed`
			: run.exitCode === 0
				? 'passed'
				: 'failed';
	const rows: Array<[string, string]> = [
		['Result', kept ? 'MIGRATED' : 'REVERTED'],
		[
			'Rewrote',
			`${plan.changes} imports in ${plan.migrated.size} files (${[...plan.packages].sort().join(', ')})`,
		],
		['Tests', `${summary(before)} -> ${summary(after)}`],
		[
			'Test time',
			`${before.seconds.toFixed(1)} s -> ${after.seconds.toFixed(1)} s${change(before.seconds, after.seconds)}`,
		],
		// shortcut: Node cannot read a child's peak RSS, so memory is reported by the Python CLI only.
		[
			'Package size',
			`${mib(sizes[0])} -> ${mib(sizes[1])}${change(...sizes)}`,
		],
	];
	const width = Math.max(...rows.map(([label]) => label.length));
	const lines = rows.map(
		([label, value]) => `  ${label.padEnd(width)}  ${value}`,
	);
	const rule = '-'.repeat(Math.max(...lines.map((line) => line.length)));
	return ['Agent RT migration', rule, ...lines, rule].join('\n');
}

/** A pre-filled issue for the user to submit; nothing is sent automatically. */
function wallLink(project: string, plan: Rewrite): string | null {
	const git = (...args: string[]) =>
		spawnSync('git', args, {
			cwd: project,
			encoding: 'utf8',
			timeout: 10_000,
		}).stdout?.trim() ?? '';
	const match =
		/^(?:https:\/\/|git@)github\.com[/:]([\w.-]+\/[\w.-]+?)(?:\.git)?\/?$/.exec(
			git('remote', 'get-url', 'origin'),
		);
	if (!match) return null;
	const packages = Object.entries(TS_SPECIFIER_RULES)
		.filter(([, rules]) =>
			rules.some(([source]) =>
				[...plan.packages].some((name) =>
					source.endsWith('/')
						? name.startsWith(source)
						: name === source,
				),
			),
		)
		.map(([name]) => name);
	const query = new URLSearchParams({
		title: `Migration wall: ${match[1]}`,
		labels: 'migration-wall',
		body: `Repository: https://github.com/${match[1]}\nCommit: ${git('rev-parse', '--short', 'HEAD')}\nPackage: ${packages.join(', ')}\nLanguage: TS/JS\nResult: PASS (agent-rt migrate)\n`,
	});
	return `${WALL_ISSUE_URL}?${query}`;
}

const USAGE = `usage: npx agent-rt migrate [PATH]
       npx agent-rt demo   (offline tour on a mock model)

Rewrite LangChain, LlamaIndex, OpenAI, Anthropic and OpenAI Agents imports to Agent RT.
Runs \`npm test\` before and after, and restores every file unless no test fails
afterwards that did not fail at the same test and line before. Per-test comparison
needs Vitest or Jest; with other runners a suite that fails before cannot be verified.`;

export function main(
	argv: string[],
	log: (line: string) => void = console.log,
): number {
	if (
		argv[0] !== 'migrate' ||
		argv.length > 2 ||
		argv.includes('-h') ||
		argv.includes('--help')
	) {
		log(USAGE);
		return argv[0] === 'migrate' ? 0 : 2;
	}
	const project = resolve(argv[1] ?? '.');
	if (!existsSync(join(project, 'package.json'))) {
		log(`agent-rt migrate: no package.json in ${project}`);
		return 2;
	}
	const agentRt = installedPackage(project, 'agent-rt');
	if (!agentRt) {
		log(
			'agent-rt migrate: agent-rt is not installed in this project. Run `npm install agent-rt` and rerun.',
		);
		return 2;
	}
	const plan = planRewrite(project, agentRtExports(agentRt));
	if (!plan.migrated.size) {
		log(
			'agent-rt migrate: no supported framework imports found; nothing to do.',
		);
		return 0;
	}
	const scratch = mkdtempSync(join(tmpdir(), 'agent-rt-migrate-'));
	let kept = false;
	try {
		const report = join(scratch, 'report');
		const runner = testRunner(project, report);
		if (!runner) {
			log(
				'agent-rt migrate: no test script found, so the migration cannot be verified; nothing changed.',
			);
			return 1;
		}
		log('Running tests before migration...');
		const before = runTests(runner, project, report);
		if (runner.read && !before.failures) {
			log(
				`${before.output.slice(-4000)}\nagent-rt migrate: the test runner did not produce a report; nothing changed.`,
			);
			return 2;
		}
		if (before.failures && before.total === 0) {
			log(
				'agent-rt migrate: no tests found, so the migration cannot be verified; nothing changed.',
			);
			return 1;
		}
		let after: SuiteRun;
		let problems: string[];
		try {
			for (const [path, text] of plan.migrated)
				writeFileSync(path, text, 'utf8');
			log(
				`Rewrote ${plan.changes} imports in ${plan.migrated.size} files; running tests again...`,
			);
			after = runTests(runner, project, report);
			({ ok: kept, problems } = compare(before, after));
		} finally {
			// Also on a crash mid-run: never leave a half-verified rewrite.
			if (!kept)
				for (const [path, text] of plan.originals)
					writeFileSync(path, text, 'utf8');
		}
		const sizes: [number, number] = [
			[...plan.packages].reduce(
				(sum, name) =>
					sum + folderSize(installedPackage(project, name)),
				0,
			),
			folderSize(agentRt),
		];
		log('');
		log(card(kept, plan, before, after, sizes));
		if (!kept) {
			log('\nNew failures after migration (all files restored):');
			for (const problem of problems.slice(0, 20)) log(`  - ${problem}`);
			log(`\nLast test output:\n${after.output.slice(-4000)}`);
			return 1;
		}
		log(pose('celebrate', 'Migration passed! Your tests agree.'));
		for (const path of [...plan.migrated.keys()].sort())
			log(`  changed ${relative(project, path).split(sep).join('/')}`);
		const link = wallLink(project, plan);
		if (link)
			log(
				`\nAdd this repository to the migration wall (opens a pre-filled GitHub issue):\n  ${link}`,
			);
		return 0;
	} finally {
		rmSync(scratch, { recursive: true, force: true });
	}
}

if (require.main === module) {
	const argv = process.argv.slice(2);
	if (argv[0] === 'demo') {
		// Loaded lazily so `migrate` never pays for the runtime import.
		// eslint-disable-next-line @typescript-eslint/no-require-imports
		const demo: typeof import('./demo.js') = require('./demo.js');
		if (argv.length > 1) {
			console.log(demo.USAGE);
			process.exitCode = ['-h', '--help'].includes(argv[1]) ? 0 : 2;
		} else
			demo.run().catch((error: unknown) => {
				console.error(error);
				process.exitCode = 1;
			});
	} else process.exitCode = main(argv);
}
