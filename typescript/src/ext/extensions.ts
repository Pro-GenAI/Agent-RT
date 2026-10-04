import { lstatSync, readFileSync, readdirSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';

import {
	JevDecisionProvider,
	makeDecisionRegistrationGuard,
	type DecisionProvider,
} from './decisions.js';
import {
	enforceRegistrationSafety,
	enforceRegistrationSafetyAsync,
	type AsyncRegistrationSafetyGuard,
	type RegistrationSafetyGuard,
	type RegistrationSafetySubject,
} from './registration_safety.js';

export interface SkillPackage {
	name: string;
	version: string;
	instructions?: string;
	tools?: unknown[];
	schemas?: Record<string, unknown>;
	resources?: Record<string, unknown>;
	metadata?: Record<string, unknown>;
}

const SKILL_FILE = 'SKILL.md';
const MAX_SKILL_BYTES = 1_000_000;
const MAX_RESOURCE_FILES = 256;
const MAX_RESOURCE_BYTES = 16 * 1024 * 1024;

function frontmatterScalar(value: string): string {
	const trimmed = value.trim();
	if (
		trimmed.length >= 2 &&
		trimmed.startsWith('"') &&
		trimmed.endsWith('"')
	) {
		try {
			const parsed: unknown = JSON.parse(trimmed);
			return typeof parsed === 'string' ? parsed : String(parsed);
		} catch {
			return trimmed.slice(1, -1);
		}
	}
	if (
		trimmed.length >= 2 &&
		trimmed.startsWith("'") &&
		trimmed.endsWith("'")
	) {
		return trimmed.slice(1, -1).replace(/''/g, "'");
	}
	return trimmed;
}

function splitSkillMarkdown(text: string): {
	frontmatter: Record<string, string>;
	instructions: string;
	rawFrontmatter: string;
} {
	const normalized = text.replace(/\r\n/g, '\n');
	if (!normalized.startsWith('---\n')) {
		return { frontmatter: {}, instructions: text, rawFrontmatter: '' };
	}

	let end = normalized.indexOf('\n---\n', 4);
	let closingLength = 5;
	if (end < 0 && normalized.endsWith('\n---')) {
		// Closing delimiter at EOF without a trailing newline (no body).
		end = normalized.length - 4;
		closingLength = 4;
	}
	if (end < 0) {
		throw new Error(
			'SKILL.md frontmatter is missing its closing --- delimiter',
		);
	}

	const rawFrontmatter = normalized.slice(4, end);
	const frontmatter: Record<string, string> = {};
	for (const line of rawFrontmatter.split('\n')) {
		if (!line || /^\s/.test(line)) continue;
		const match = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
		if (match) frontmatter[match[1]] = frontmatterScalar(match[2]);
	}

	return {
		frontmatter,
		instructions: normalized.slice(end + closingLength).replace(/^\n+/, ''),
		rawFrontmatter,
	};
}

function collectResourcePaths(root: string, current = root): string[] {
	const resources: string[] = [];
	for (const entry of readdirSync(current, { withFileTypes: true })) {
		const fullPath = join(current, entry.name);
		if (entry.isSymbolicLink()) continue;
		if (entry.isDirectory()) {
			resources.push(...collectResourcePaths(root, fullPath));
		} else if (
			entry.isFile() &&
			resolve(fullPath) !== resolve(join(root, SKILL_FILE))
		) {
			resources.push(fullPath);
		}
	}
	return resources;
}

export function loadSkillPackage(
	skillPath: string,
	version?: string,
): SkillPackage {
	const requested = resolve(skillPath);
	let skillFile = requested;
	try {
		const requestedStat = lstatSync(requested);
		if (requestedStat.isSymbolicLink()) {
			throw new Error('skill paths must not be symlinks');
		}
		if (!requestedStat.isFile()) skillFile = join(requested, SKILL_FILE);
	} catch (error) {
		if (
			error instanceof Error &&
			error.message === 'skill paths must not be symlinks'
		) {
			throw error;
		}
		throw new Error(
			`skill path must be a ${SKILL_FILE} file or directory containing one`,
		);
	}

	if (basename(skillFile) !== SKILL_FILE) {
		throw new Error(
			`skill path must be a ${SKILL_FILE} file or directory containing one`,
		);
	}

	let skillStat;
	try {
		skillStat = lstatSync(skillFile);
	} catch {
		throw new Error(
			`skill path must be a ${SKILL_FILE} file or directory containing one`,
		);
	}
	if (skillStat.isSymbolicLink()) {
		throw new Error(`${SKILL_FILE} must not be a symlink`);
	}
	if (!skillStat.isFile())
		throw new Error(`skill path must contain ${SKILL_FILE}`);
	if (skillStat.size > MAX_SKILL_BYTES) {
		throw new Error(
			`${SKILL_FILE} exceeds the ${MAX_SKILL_BYTES}-byte limit`,
		);
	}

	skillFile = resolve(skillFile);
	const root = dirname(skillFile);
	const parsed = splitSkillMarkdown(readFileSync(skillFile, 'utf8'));
	const name = (parsed.frontmatter.name ?? basename(root)).trim();
	const selectedVersion = (
		version ??
		parsed.frontmatter.version ??
		'1'
	).trim();
	if (!name) throw new Error('skill name must not be empty');
	if (!selectedVersion) throw new Error('skill version must not be empty');

	const resources: Record<string, unknown> = {};
	let resourceBytes = 0;
	const resourcePaths = collectResourcePaths(root).sort();
	if (resourcePaths.length > MAX_RESOURCE_FILES) {
		throw new Error(
			`skill exceeds the ${MAX_RESOURCE_FILES}-resource-file limit`,
		);
	}

	const decoder = new TextDecoder('utf-8', { fatal: true });
	for (const resourcePath of resourcePaths) {
		// Check the size before reading so an oversized file is never loaded.
		if (resourceBytes + lstatSync(resourcePath).size > MAX_RESOURCE_BYTES) {
			throw new Error(
				`skill resources exceed the ${MAX_RESOURCE_BYTES}-byte limit`,
			);
		}
		const data = readFileSync(resourcePath);
		resourceBytes += data.byteLength;
		if (resourceBytes > MAX_RESOURCE_BYTES) {
			throw new Error(
				`skill resources exceed the ${MAX_RESOURCE_BYTES}-byte limit`,
			);
		}
		const relativePath = relative(root, resourcePath).split('\\').join('/');
		try {
			resources[relativePath] = decoder.decode(data);
		} catch {
			resources[relativePath] = new Uint8Array(data);
		}
	}

	const metadata: Record<string, unknown> = {
		source_format: 'skill-md',
		source_path: root,
		frontmatter: { ...parsed.frontmatter },
	};
	if (parsed.rawFrontmatter) metadata.raw_frontmatter = parsed.rawFrontmatter;
	if (parsed.frontmatter.description?.trim()) {
		metadata.description = parsed.frontmatter.description.trim();
	}

	return {
		name,
		version: selectedVersion,
		instructions: parsed.instructions,
		resources,
		metadata,
	};
}

function skillRegistrationSubject(
	skill: SkillPackage,
): RegistrationSafetySubject {
	const content: Record<string, string> = {
		instructions: skill.instructions ?? '',
	};
	const rawFrontmatter = skill.metadata?.raw_frontmatter;
	if (typeof rawFrontmatter === 'string')
		content.frontmatter = rawFrontmatter;
	for (const [index, tool] of (skill.tools ?? []).entries()) {
		if (tool && typeof tool === 'object') {
			const candidate = tool as Record<string, unknown>;
			content['tool:' + index + ':name'] = String(candidate.name ?? '');
			content['tool:' + index + ':description'] = String(
				candidate.description ?? '',
			);
			try {
				content['tool:' + index + ':metadata'] = JSON.stringify(
					candidate.metadata ?? {},
				);
			} catch {
				content['tool:' + index + ':metadata'] = String(
					candidate.metadata ?? '',
				);
			}
			const schema = candidate.inputSchema ?? candidate.input_schema;
			try {
				content['tool:' + index + ':input_schema'] = JSON.stringify(
					schema ?? {},
				);
			} catch {
				content['tool:' + index + ':input_schema'] = String(
					schema ?? '',
				);
			}
		}
	}
	for (const [path, resource] of Object.entries(skill.resources ?? {})) {
		if (typeof resource === 'string')
			content['resource:' + path] = resource;
	}
	const description = skill.metadata?.description;
	return {
		kind: 'skill',
		name: skill.name,
		description:
			typeof description === 'string'
				? description
				: String(description ?? ''),
		content,
	};
}

function compareVersions(left: string, right: string): number {
	const tokenize = (value: string): Array<number | string> =>
		value
			.match(/\d+|\D+/g)!
			.map((part) =>
				/^\d+$/.test(part) ? Number(part) : part.toLowerCase(),
			);

	const a = tokenize(left);
	const b = tokenize(right);
	const length = Math.max(a.length, b.length);
	for (let index = 0; index < length; index += 1) {
		if (index >= a.length) return -1;
		if (index >= b.length) return 1;
		const av = a[index];
		const bv = b[index];
		if (typeof av === 'number' && typeof bv === 'number') {
			if (av !== bv) return av - bv;
			continue;
		}
		if (typeof av === 'number') return 1;
		if (typeof bv === 'number') return -1;
		const compared = av.localeCompare(bv);
		if (compared !== 0) return compared;
	}
	return left.localeCompare(right);
}

export class SkillRegistry {
	private readonly skills = new Map<string, SkillPackage>();
	private readonly activeVersions = new Map<string, string>();

	constructor(readonly registrationGuard?: RegistrationSafetyGuard) {}

	install(skill: SkillPackage, activate = false): void {
		if (!skill.name.trim()) throw new Error('skill name must not be empty');
		if (!skill.version.trim())
			throw new Error('skill version must not be empty');
		enforceRegistrationSafety(
			skillRegistrationSubject(skill),
			this.registrationGuard,
		);
		this.skills.set(
			skill.name + '::' + skill.version,
			structuredClone(skill),
		);
		if (activate) this.activeVersions.set(skill.name, skill.version);
	}

	async installChecked(
		skill: SkillPackage,
		decisionGuard: AsyncRegistrationSafetyGuard,
		activate = false,
	): Promise<SkillPackage> {
		await enforceRegistrationSafetyAsync(
			skillRegistrationSubject(skill),
			decisionGuard,
		);
		this.install(skill, activate);
		return this.get(skill.name, skill.version);
	}

	async installFromPath(
		skillPath: string,
		options: {
			activate?: boolean;
			version?: string;
			decisionProvider?: DecisionProvider;
		} = {},
	): Promise<SkillPackage> {
		const skill = loadSkillPackage(skillPath, options.version);
		const decisionGuard = makeDecisionRegistrationGuard(
			options.decisionProvider ?? new JevDecisionProvider(),
		);
		return this.installChecked(
			skill,
			decisionGuard,
			options.activate ?? false,
		);
	}

	async installDirectory(
		directory: string,
		options:
			| boolean
			| {
					activate?: boolean;
					decisionProvider?: DecisionProvider;
			  } = {},
	): Promise<SkillPackage[]> {
		const resolvedOptions =
			typeof options === 'boolean' ? { activate: options } : options;
		const root = resolve(directory);
		let entries;
		let rootHasSkill = false;
		try {
			const rootStat = lstatSync(root);
			if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
				throw new Error('not a directory');
			}
			try {
				const skillStat = lstatSync(join(root, SKILL_FILE));
				if (skillStat.isSymbolicLink()) {
					throw new Error(`${SKILL_FILE} must not be a symlink`);
				}
				rootHasSkill = skillStat.isFile();
			} catch (error) {
				if (
					error instanceof Error &&
					error.message === `${SKILL_FILE} must not be a symlink`
				) {
					throw error;
				}
			}
			entries = readdirSync(root, { withFileTypes: true });
		} catch (error) {
			if (
				error instanceof Error &&
				error.message === `${SKILL_FILE} must not be a symlink`
			) {
				throw error;
			}
			throw new Error('skill directory does not exist');
		}
		if (rootHasSkill) {
			return [await this.installFromPath(root, resolvedOptions)];
		}

		const candidates = entries
			.filter((entry) => entry.isDirectory() && !entry.isSymbolicLink())
			.map((entry) => join(root, entry.name))
			.filter((candidate) => {
				try {
					const skillStat = lstatSync(join(candidate, SKILL_FILE));
					return !skillStat.isSymbolicLink() && skillStat.isFile();
				} catch {
					return false;
				}
			})
			.sort();

		if (candidates.length === 0) {
			throw new Error(
				`no child skill directories containing ${SKILL_FILE} were found`,
			);
		}
		const installed: SkillPackage[] = [];
		for (const candidate of candidates) {
			installed.push(
				await this.installFromPath(candidate, resolvedOptions),
			);
		}
		return installed;
	}

	activate(name: string, version: string): SkillPackage {
		const skill = this.get(name, version);
		this.activeVersions.set(name, version);
		return skill;
	}

	get(name: string, version?: string): SkillPackage {
		let selected = version ?? this.activeVersions.get(name);
		if (selected === undefined) {
			const versions = [...this.skills.keys()]
				.filter((key) => key.startsWith(name + '::'))
				.map((key) => key.slice(name.length + 2))
				.sort(compareVersions);
			if (versions.length === 0) throw new Error('unknown skill ' + name);
			selected = versions[versions.length - 1];
		}
		const skill = this.skills.get(name + '::' + selected);
		if (!skill)
			throw new Error('unknown skill version ' + name + '/' + selected);
		return structuredClone(skill);
	}

	active(): SkillPackage[] {
		return [...this.activeVersions.entries()]
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([name, version]) => this.get(name, version));
	}

	uninstall(name: string, version: string): void {
		const key = name + '::' + version;
		if (!this.skills.delete(key)) {
			throw new Error('unknown skill version ' + name + '/' + version);
		}
		if (this.activeVersions.get(name) === version)
			this.activeVersions.delete(name);
	}
}

export type MiddlewareHandler = (
	stage: string,
	value: unknown,
	next: (value: unknown) => Promise<unknown>,
) => Promise<unknown>;

export class MiddlewarePipeline {
	private readonly handlers = new Map<string, MiddlewareHandler[]>();

	use(stage: string, handler: MiddlewareHandler): void {
		if (!stage.trim())
			throw new Error('middleware stage must not be empty');
		const current = this.handlers.get(stage) ?? [];
		current.push(handler);
		this.handlers.set(stage, current);
	}

	async run(
		stage: string,
		value: unknown,
		terminal: (value: unknown) => Promise<unknown>,
	): Promise<unknown> {
		const handlers = [...(this.handlers.get(stage) ?? [])];
		const invoke = async (
			index: number,
			current: unknown,
		): Promise<unknown> => {
			if (index >= handlers.length) return terminal(current);
			return handlers[index](stage, current, (nextValue) =>
				invoke(index + 1, nextValue),
			);
		};
		return invoke(0, value);
	}
}

export type ExtensionKind =
	| 'model'
	| 'memory'
	| 'filesystem'
	| 'sandbox'
	| 'queue'
	| 'evaluator'
	| 'telemetry';

const SUPPORTED_EXTENSION_KINDS = new Set<string>([
	'model',
	'memory',
	'filesystem',
	'sandbox',
	'queue',
	'evaluator',
	'telemetry',
]);

function assertExtensionKind(kind: string): asserts kind is ExtensionKind {
	if (!SUPPORTED_EXTENSION_KINDS.has(kind)) {
		throw new Error('unsupported extension provider kind: ' + kind);
	}
}

export class ExtensionRegistry {
	private readonly providers = new Map<string, unknown>();

	register(
		kind: ExtensionKind,
		name: string,
		provider: unknown,
		replaceExisting = false,
	): void {
		assertExtensionKind(kind);
		if (!name.trim())
			throw new Error('extension provider name must not be empty');
		const key = kind + '::' + name;
		if (this.providers.has(key) && !replaceExisting) {
			throw new Error(
				'extension provider already registered: ' + kind + '/' + name,
			);
		}
		this.providers.set(key, provider);
	}

	get(kind: ExtensionKind, name: string): unknown {
		assertExtensionKind(kind);
		const key = kind + '::' + name;
		if (!this.providers.has(key)) {
			throw new Error('unknown extension provider ' + kind + '/' + name);
		}
		return this.providers.get(key);
	}

	list(kind?: ExtensionKind): Array<[ExtensionKind, string]> {
		if (kind !== undefined) assertExtensionKind(kind);
		return [...this.providers.keys()]
			.map((key) => key.split('::') as [ExtensionKind, string])
			.filter(
				([providerKind]) => kind === undefined || providerKind === kind,
			)
			.sort(
				([ak, an], [bk, bn]) =>
					ak.localeCompare(bk) || an.localeCompare(bn),
			);
	}
}
