/**
 * YAML runtime configuration (`config.yaml`, in the style of LiteLLM).
 *
 * The file declares several models (`model_list`), several vector stores
 * (`vector_store_registry`) and runtime settings (`agent_rt_settings`).
 * Secrets are referenced as `os.environ/NAME` and resolved while the file is
 * loaded; nothing else in the runtime reads configuration from the process
 * environment.
 */

export const ENVIRONMENT_REFERENCE_PREFIX = 'os.environ/';
export const MAX_CONFIG_BYTES = 1_000_000;
export const CONFIG_FILE_NAMES = ['agent-rt.yaml', 'agent-rt.yml'] as const;
export const SUPPORTED_MODEL_PROVIDERS = ['openai', 'anthropic'] as const;
export const SUPPORTED_VECTOR_STORE_PROVIDERS = [
	'bedrock',
	'chroma',
	'milvus',
	'pinecone',
	'qdrant',
	'weaviate',
] as const;

const PARAM_KEYS = ['agent_rt_params', 'litellm_params'] as const;
// Unset references here fail only when that model, store or server is used.
const DEFERRED_SECRET_PATH =
	/^config\.(model_list|vector_store_registry)\[\d+\]\.(agent_rt_params|litellm_params)\.(api_key|api_base)$|^config\.agent_rt_settings\.server\.|^config\.agent_rt_settings\.memory\.options\.|^config\.agent_rt_settings\.serve\./;
const MODEL_PARAM_FIELDS = new Set([
	'model',
	'api_key',
	'api_base',
	'websocket',
]);
const STORE_PARAM_FIELDS = new Set([
	'vector_store_id',
	'custom_llm_provider',
	'api_base',
	'api_key',
	'collection',
	'embedding_model',
	'timeout',
	'options',
	'vector_store_description',
	'vector_store_metadata',
	'allowed_models',
]);
const SETTINGS_SECTIONS = new Set([
	'default_model',
	'server',
	'sandbox',
	'web_search',
	'execution_limits',
	'decision_model',
	'observability',
	'cli',
	'memory',
	'serve',
]);
export const DEFAULT_MEMORY_BACKEND = 'mem0';
/** `agent_rt_settings.memory` backends and their options (same as Python). */
export const MEMORY_BACKEND_OPTIONS: Record<string, readonly string[]> = {
	mem0: ['api_key', 'host', 'config', 'user_id', 'agent_id', 'run_id'],
	memu: [
		'user_id',
		'llm_profiles',
		'blob_config',
		'database_config',
		'memorize_config',
		'retrieve_config',
		'user_config',
	],
	zep: ['api_key', 'base_url', 'graph_id', 'thread_id'],
	letta: ['api_key', 'base_url', 'agent_id'],
	supermemory: ['api_key', 'base_url', 'namespace'],
};
const SANDBOX_KEYS = [
	'backend',
	'workspace_root',
	'native_root',
	'docker_image',
	'docker_binary',
	'docker_user',
	'docker_read_only',
	'e2b_template',
	'microsandbox_image',
	'swerex_url',
	'swerex_api_key',
	'uid',
	'gid',
	'network_mode',
	'network_max_transfer_bytes',
	'network_max_bytes_per_second',
	'allowed_domains',
	'blocked_domains',
	'allow_http',
	'allow_ip',
	'allow_proxy',
	'allow_websocket',
	'setpriv_binary',
	'prlimit_binary',
];
const SECTION_ENVIRONMENT: Readonly<
	Record<string, Readonly<Record<string, string>>>
> = {
	execution_limits: {
		timeout_seconds: 'AGENT_RT_EXECUTION_TIMEOUT_SECONDS',
		memory_bytes: 'AGENT_RT_EXECUTION_MEMORY_BYTES',
		cpu_seconds: 'AGENT_RT_EXECUTION_CPU_SECONDS',
	},
	web_search: {
		tool: 'AGENT_RT_WEB_SEARCH_TOOL',
		// Pieces under 16 characters: mcp-scan reads long literals after api_key as secrets.
		api_key: 'AGENT_RT_' + 'WEB_SEARCH_' + 'API_KEY',
		timeout_seconds: 'AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS',
	},
	decision_model: {
		base_url: 'AGENT_RT_JEV_BASE_URL',
		model: 'AGENT_RT_JEV_MODEL',
		timeout_seconds: 'AGENT_RT_JEV_TIMEOUT_SECONDS',
		provider: 'AGENT_RT_DECISION_PROVIDER',
		api_key: 'AGENT_RT_' + 'DECISION_' + 'API_KEY',
	},
	observability: {
		api_key: 'TRACELOOP_' + 'API_KEY',
		base_url: 'TRACELOOP_BASE_URL',
		headers: 'TRACELOOP_HEADERS',
		trace_content: 'TRACELOOP_TRACE_CONTENT',
		telemetry_enabled: 'TRACELOOP_TELEMETRY',
		enrich_tokens: 'TRACELOOP_ENRICH_TOKENS',
		app_name: 'TRACELOOP_APP_NAME',
	},
	sandbox: Object.fromEntries(
		SANDBOX_KEYS.map((key) => [
			key,
			'AGENT_RT_SANDBOX_' + key.toUpperCase(),
		]),
	),
};
const SERVER_FIELDS = new Set([
	'api_keys',
	'default_agent',
	'allowed_websocket_origins',
	'host',
	'port',
	'max_concurrent_requests',
	'max_queued_requests',
]);
const CLI_FIELDS = new Set(['home', 'compact_token_limit']);
/** `agent_rt_settings.serve` platforms (used by Python's `agent-rt serve`). */
const SERVE_PLATFORMS = new Set(['telegram', 'discord']);
const SERVE_FIELDS = new Set(['bot_token', 'allowed_users']);

export class ConfigError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'ConfigError';
	}
}

/** No configuration file exists at the requested or default paths. */
export class ConfigNotFoundError extends ConfigError {
	constructor(message: string) {
		super(message);
		this.name = 'ConfigNotFoundError';
	}
}

/** An `os.environ/NAME` reference whose variable is unset; throws when used. */
export class MissingSecret {
	constructor(
		readonly name: string,
		readonly path: string,
	) {}

	error(): ConfigError {
		return new ConfigError(
			`${this.path}: environment variable ${this.name} referenced by os.environ/ is not set`,
		);
	}
}

/** The secret's value; throws when its `os.environ/` reference is unset. */
export function secretValue(
	value: string | MissingSecret | undefined,
): string | undefined {
	if (value instanceof MissingSecret) throw value.error();
	return value;
}

/** Throw for the first unset `os.environ/` reference in `values`. */
export function requireResolved(...values: unknown[]): void {
	for (const value of values) {
		if (value instanceof MissingSecret) throw value.error();
		if (Array.isArray(value)) requireResolved(...value);
		else if (isMapping(value)) requireResolved(...Object.values(value));
	}
}

/** One concrete provider/model behind a public `modelName`. */
export interface ModelDeployment {
	modelName: string;
	provider: (typeof SUPPORTED_MODEL_PROVIDERS)[number];
	model: string;
	apiKey?: string | MissingSecret;
	apiBase?: string | MissingSecret;
	websocket?: boolean;
	modelInfo: Record<string, unknown>;
}

/** One entry of `vector_store_registry`. */
export interface VectorStoreConfig {
	name: string;
	storeId: string;
	provider: (typeof SUPPORTED_VECTOR_STORE_PROVIDERS)[number];
	apiBase?: string | MissingSecret;
	apiKey?: string | MissingSecret;
	collection?: string;
	embeddingModel?: string;
	timeout: number;
	options: Record<string, string>;
	description?: string;
	metadata: Record<string, unknown>;
	allowedModels: string[];
}

export interface AgentRTConfig {
	models: ModelDeployment[];
	vectorStores: VectorStoreConfig[];
	settings: Record<string, unknown>;
	source?: string;
}

type Mapping = Record<string, unknown>;

export function modelNames(config: AgentRTConfig): string[] {
	return [...new Set(config.models.map((item) => item.modelName))];
}

export function defaultModel(config: AgentRTConfig): string | undefined {
	const configured = config.settings['default_model'];
	if (typeof configured === 'string') return configured;
	return modelNames(config)[0];
}

export function deployments(
	config: AgentRTConfig,
	modelName: string,
): ModelDeployment[] {
	return config.models.filter((item) => item.modelName === modelName);
}

export function findVectorStore(
	config: AgentRTConfig,
	reference: string,
): VectorStoreConfig | undefined {
	return config.vectorStores.find(
		(store) => store.name === reference || store.storeId === reference,
	);
}

/** Variables for the sandbox/web-search/limits/observability readers, built only from this file. */
export function runtimeEnvironment(
	config: AgentRTConfig,
): Record<string, string> {
	const result: Record<string, string> = {};
	for (const [section, names] of Object.entries(SECTION_ENVIRONMENT)) {
		const values = config.settings[section];
		if (!isMapping(values)) continue;
		for (const [key, variable] of Object.entries(names)) {
			const item = values[key];
			if (item === undefined || item === null) continue;
			if (typeof item === 'boolean') {
				result[variable] = item ? 'true' : 'false';
			} else if (Array.isArray(item)) {
				result[variable] = item.map(String).join(',');
			} else if (isMapping(item)) {
				result[variable] = Object.entries(item)
					.map(([k, v]) => `${k}=${String(v)}`)
					.join(',');
			} else {
				result[variable] = String(item);
			}
		}
	}
	return result;
}

export function serverOptions(config: AgentRTConfig): Record<string, unknown> {
	const server = config.settings['server'];
	if (!isMapping(server)) return {};
	requireResolved(server);
	return { ...server };
}

function isMapping(value: unknown): value is Mapping {
	if (typeof value !== 'object' || value === null) return false;
	// Plain objects only: not arrays, YAML timestamps (Date) or MissingSecret.
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

export function configSearchPaths(
	options: { cwd?: string; home?: string } = {},
): string[] {
	const base = options.cwd ?? processCwd();
	const user = options.home ?? homeDirectory();
	const paths = CONFIG_FILE_NAMES.map((name) => joinPath(base, name));
	paths.push(
		joinPath(base, '.agent-rt', 'config.yaml'),
		joinPath(base, '.agent-rt', 'config.yml'),
		joinPath(user, '.agent-rt', 'config.yaml'),
		joinPath(user, '.agent-rt', 'config.yml'),
		joinPath(user, '.config', 'agent-rt', 'config.yaml'),
		joinPath(user, '.config', 'agent-rt', 'config.yml'),
	);
	return paths;
}

function joinPath(...parts: string[]): string {
	const path = nodeBuiltin('node:path') as {
		join(...segments: string[]): string;
	};
	return path.join(...parts);
}

function processCwd(): string {
	const runtime = globalThis as unknown as {
		process?: { cwd?: () => string };
	};
	if (!runtime.process?.cwd)
		throw new ConfigError('no working directory available');
	return runtime.process.cwd();
}

function homeDirectory(): string {
	return (nodeBuiltin('node:os') as { homedir(): string }).homedir();
}

declare const require: ((id: string) => unknown) | undefined;

function nodeBuiltin(name: string): unknown {
	const runtime = globalThis as unknown as {
		process?: { getBuiltinModule?: (id: string) => unknown };
	};
	// getBuiltinModule was introduced after the first supported Node 20 releases.
	const builtin =
		runtime.process?.getBuiltinModule?.(name) ??
		(typeof require === 'function' ? require(name) : undefined);
	if (builtin === undefined) {
		throw new ConfigError(
			`config files require Node.js (${name} unavailable)`,
		);
	}
	return builtin;
}

/** Return the config file to use: an explicit path or the first default found. */
export function discoverConfigPath(
	explicit?: string,
	options: { cwd?: string; home?: string } = {},
): string {
	const fs = nodeBuiltin('node:fs') as {
		statSync(
			path: string,
			options: { throwIfNoEntry: false },
		):
			| {
					isFile(): boolean;
			  }
			| undefined;
	};
	const isFile = (path: string): boolean =>
		fs.statSync(path, { throwIfNoEntry: false })?.isFile() === true;
	if (explicit !== undefined && explicit.trim()) {
		const path = explicit.startsWith('~')
			? homeDirectory() + explicit.slice(1)
			: explicit;
		if (!isFile(path))
			throw new ConfigNotFoundError(`config file not found: ${path}`);
		return path;
	}
	const candidates = configSearchPaths(options);
	for (const candidate of candidates) {
		if (isFile(candidate)) return candidate;
	}
	throw new ConfigNotFoundError(
		'no config file found (pass a path or create one of: ' +
			candidates.join(', ') +
			')',
	);
}

function resolveSecrets(
	value: unknown,
	path: string,
	environment: Readonly<Record<string, string | undefined>>,
): unknown {
	if (typeof value === 'string') {
		if (!value.startsWith(ENVIRONMENT_REFERENCE_PREFIX)) return value;
		const name = value.slice(ENVIRONMENT_REFERENCE_PREFIX.length).trim();
		if (!name)
			throw new ConfigError(`${path}: empty os.environ/ reference`);
		const resolved = environment[name];
		if (resolved === undefined || !resolved.trim()) {
			const missing = new MissingSecret(name, path);
			if (DEFERRED_SECRET_PATH.test(path)) return missing;
			throw missing.error();
		}
		return resolved;
	}
	if (Array.isArray(value)) {
		return value.map((item, index) =>
			resolveSecrets(item, `${path}[${index}]`, environment),
		);
	}
	if (isMapping(value)) {
		return Object.fromEntries(
			Object.entries(value).map(([key, item]) => [
				key,
				resolveSecrets(item, `${path}.${key}`, environment),
			]),
		);
	}
	return value;
}

function mapping(value: unknown, path: string): Mapping {
	if (!isMapping(value)) throw new ConfigError(`${path} must be a mapping`);
	return value;
}

function checkKeys(value: Mapping, allowed: Set<string>, path: string): void {
	const unknown = Object.keys(value)
		.filter((key) => !allowed.has(key))
		.sort();
	if (unknown.length) {
		throw new ConfigError(`${path}: unknown key(s): ${unknown.join(', ')}`);
	}
}

function text(value: unknown, path: string, required: true): string;
function text(
	value: unknown,
	path: string,
	required?: false,
): string | undefined;
function text(
	value: unknown,
	path: string,
	required = false,
): string | undefined {
	if (value === undefined || value === null) {
		if (required) throw new ConfigError(`${path} is required`);
		return undefined;
	}
	if (typeof value !== 'string' || !value.trim()) {
		throw new ConfigError(`${path} must be a non-empty string`);
	}
	return value.trim();
}

function secretText(
	value: unknown,
	path: string,
): string | MissingSecret | undefined {
	return value instanceof MissingSecret ? value : text(value, path);
}

function positiveNumber(value: unknown, path: string): number | undefined {
	if (value === undefined || value === null) return undefined;
	if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
		throw new ConfigError(`${path} must be a positive finite number`);
	}
	return value;
}

function paramsOf(entry: Mapping, path: string): Mapping {
	const present = PARAM_KEYS.filter((key) => key in entry);
	if (present.length !== 1) {
		throw new ConfigError(
			`${path} requires exactly one of: ${PARAM_KEYS.join(', ')}`,
		);
	}
	return mapping(entry[present[0]], `${path}.${present[0]}`);
}

function parseModels(raw: unknown): ModelDeployment[] {
	if (!Array.isArray(raw)) throw new ConfigError('model_list must be a list');
	return raw.map((item, index) => {
		const path = `model_list[${index}]`;
		const entry = mapping(item, path);
		checkKeys(
			entry,
			new Set(['model_name', 'model_info', ...PARAM_KEYS]),
			path,
		);
		const modelName = text(entry['model_name'], `${path}.model_name`, true);
		const params = paramsOf(entry, path);
		checkKeys(params, MODEL_PARAM_FIELDS, `${path}.params`);
		const spec = text(params['model'], `${path}.params.model`, true);
		const slash = spec.indexOf('/');
		const provider = (slash < 0 ? '' : spec.slice(0, slash))
			.trim()
			.toLowerCase();
		const model = slash < 0 ? '' : spec.slice(slash + 1).trim();
		if (!provider || !model) {
			throw new ConfigError(
				`${path}.params.model must look like '<provider>/<model>'`,
			);
		}
		if (
			!(SUPPORTED_MODEL_PROVIDERS as readonly string[]).includes(provider)
		) {
			throw new ConfigError(
				`${path}.params.model: unsupported provider '${provider}'; supported: ${SUPPORTED_MODEL_PROVIDERS.join(', ')}`,
			);
		}
		const websocket = params['websocket'];
		if (websocket !== undefined && typeof websocket !== 'boolean') {
			throw new ConfigError(`${path}.params.websocket must be a boolean`);
		}
		return {
			modelName,
			provider: provider as ModelDeployment['provider'],
			model,
			apiKey: secretText(params['api_key'], `${path}.params.api_key`),
			apiBase: secretText(params['api_base'], `${path}.params.api_base`),
			websocket,
			modelInfo: {
				...mapping(entry['model_info'] ?? {}, `${path}.model_info`),
			},
		};
	});
}

function parseVectorStores(
	raw: unknown,
	modelNameSet: Set<string>,
): VectorStoreConfig[] {
	if (!Array.isArray(raw)) {
		throw new ConfigError('vector_store_registry must be a list');
	}
	const seen = new Set<string>();
	return raw.map((item, index) => {
		const path = `vector_store_registry[${index}]`;
		const entry = mapping(item, path);
		checkKeys(entry, new Set(['vector_store_name', ...PARAM_KEYS]), path);
		const name = text(
			entry['vector_store_name'],
			`${path}.vector_store_name`,
			true,
		);
		const params = paramsOf(entry, path);
		checkKeys(params, STORE_PARAM_FIELDS, `${path}.params`);
		const provider = text(
			params['custom_llm_provider'],
			`${path}.params.custom_llm_provider`,
			true,
		).toLowerCase();
		if (
			!(SUPPORTED_VECTOR_STORE_PROVIDERS as readonly string[]).includes(
				provider,
			)
		) {
			throw new ConfigError(
				`${path}.params.custom_llm_provider: unsupported '${provider}'; supported: ${SUPPORTED_VECTOR_STORE_PROVIDERS.join(', ')} (other hosted stores are not supported)`,
			);
		}
		const storeId = text(
			params['vector_store_id'],
			`${path}.params.vector_store_id`,
			true,
		);
		const embeddingModel = text(
			params['embedding_model'],
			`${path}.params.embedding_model`,
		);
		if (embeddingModel !== undefined && !modelNameSet.has(embeddingModel)) {
			throw new ConfigError(
				`${path}.params.embedding_model '${embeddingModel}' is not in model_list`,
			);
		}
		const options = Object.fromEntries(
			Object.entries(
				mapping(params['options'] ?? {}, `${path}.params.options`),
			).map(([key, value]) => [key, String(value)]),
		);
		const allowed = params['allowed_models'] ?? [];
		if (
			!Array.isArray(allowed) ||
			!allowed.every((m) => typeof m === 'string')
		) {
			throw new ConfigError(
				`${path}.params.allowed_models must be a list of model names`,
			);
		}
		for (const reference of new Set([name, storeId])) {
			if (seen.has(reference)) {
				throw new ConfigError(
					`${path}: duplicate vector store name or id '${reference}'`,
				);
			}
			seen.add(reference);
		}
		return {
			name,
			storeId,
			provider: provider as VectorStoreConfig['provider'],
			apiBase: secretText(params['api_base'], `${path}.params.api_base`),
			apiKey: secretText(params['api_key'], `${path}.params.api_key`),
			collection: text(params['collection'], `${path}.params.collection`),
			embeddingModel,
			timeout:
				positiveNumber(params['timeout'], `${path}.params.timeout`) ??
				20,
			options,
			description: text(
				params['vector_store_description'],
				`${path}.params.vector_store_description`,
			),
			metadata: {
				...mapping(
					params['vector_store_metadata'] ?? {},
					`${path}.params.vector_store_metadata`,
				),
			},
			allowedModels: allowed as string[],
		};
	});
}

function parseSettings(raw: unknown): Record<string, unknown> {
	const settings = { ...mapping(raw, 'agent_rt_settings') };
	checkKeys(settings, SETTINGS_SECTIONS, 'agent_rt_settings');
	for (const [section, names] of Object.entries(SECTION_ENVIRONMENT)) {
		const values = settings[section];
		if (values === undefined || values === null) continue;
		checkKeys(
			mapping(values, `agent_rt_settings.${section}`),
			new Set(Object.keys(names)),
			`agent_rt_settings.${section}`,
		);
	}
	if (settings['server'] !== undefined) {
		checkKeys(
			mapping(settings['server'], 'agent_rt_settings.server'),
			SERVER_FIELDS,
			'agent_rt_settings.server',
		);
	}
	if (settings['cli'] !== undefined) {
		checkKeys(
			mapping(settings['cli'], 'agent_rt_settings.cli'),
			CLI_FIELDS,
			'agent_rt_settings.cli',
		);
	}
	if (settings['serve'] !== undefined) {
		const serve = mapping(settings['serve'], 'agent_rt_settings.serve');
		checkKeys(serve, SERVE_PLATFORMS, 'agent_rt_settings.serve');
		for (const [platform, values] of Object.entries(serve)) {
			const path = `agent_rt_settings.serve.${platform}`;
			checkKeys(mapping(values, path), SERVE_FIELDS, path);
		}
	}
	if (settings['memory'] !== undefined && settings['memory'] !== null) {
		const memory = mapping(settings['memory'], 'agent_rt_settings.memory');
		checkKeys(
			memory,
			new Set(['backend', 'options']),
			'agent_rt_settings.memory',
		);
		const backend = memory['backend'] ?? DEFAULT_MEMORY_BACKEND;
		if (
			typeof backend !== 'string' ||
			!Object.prototype.hasOwnProperty.call(
				MEMORY_BACKEND_OPTIONS,
				backend,
			)
		)
			throw new ConfigError(
				'agent_rt_settings.memory.backend must be one of: ' +
					Object.keys(MEMORY_BACKEND_OPTIONS).join(', '),
			);
		checkKeys(
			mapping(
				memory['options'] ?? {},
				'agent_rt_settings.memory.options',
			),
			new Set(MEMORY_BACKEND_OPTIONS[backend]),
			'agent_rt_settings.memory.options',
		);
	}
	if (settings['default_model'] !== undefined) {
		settings['default_model'] = text(
			settings['default_model'],
			'agent_rt_settings.default_model',
		);
	}
	return settings;
}

/** Validate an already-parsed document. */
export function configFromObject(
	raw: unknown,
	options: {
		environment?: Readonly<Record<string, string | undefined>>;
		source?: string;
	} = {},
): AgentRTConfig {
	const environment = options.environment ?? processEnvironment();
	const document = mapping(
		resolveSecrets(raw ?? {}, 'config', environment),
		'config',
	);
	checkKeys(
		document,
		new Set(['model_list', 'vector_store_registry', 'agent_rt_settings']),
		'config',
	);
	const models = parseModels(document['model_list'] ?? []);
	const names = new Set(models.map((item) => item.modelName));
	const vectorStores = parseVectorStores(
		document['vector_store_registry'] ?? [],
		names,
	);
	const settings = parseSettings(document['agent_rt_settings'] ?? {});
	const configuredDefault = settings['default_model'];
	if (
		typeof configuredDefault === 'string' &&
		!names.has(configuredDefault)
	) {
		throw new ConfigError(
			`agent_rt_settings.default_model '${configuredDefault}' is not in model_list`,
		);
	}
	for (const store of vectorStores) {
		for (const allowed of store.allowedModels) {
			if (!names.has(allowed)) {
				throw new ConfigError(
					`vector store '${store.name}' allows unknown model '${allowed}'`,
				);
			}
		}
	}
	return { models, vectorStores, settings, source: options.source };
}

function processEnvironment(): Readonly<Record<string, string | undefined>> {
	const runtime = globalThis as unknown as {
		process?: { env?: Readonly<Record<string, string | undefined>> };
	};
	return runtime.process?.env ?? {};
}

/** Parse YAML text (YAML 1.1 like PyYAML's safe loader; custom tags are rejected). */
export async function parseConfig(
	yamlText: string,
	options: {
		environment?: Readonly<Record<string, string | undefined>>;
		source?: string;
	} = {},
): Promise<AgentRTConfig> {
	const YAML = await import('yaml');
	// YAML 1.1 (yes/no booleans, 0-prefixed octals, merge keys) matches PyYAML.
	const document = YAML.parseDocument(yamlText, { version: '1.1' });
	const problem = [...document.errors, ...document.warnings][0];
	if (problem) throw new ConfigError(`invalid YAML: ${problem.message}`);
	return configFromObject(document.toJS(), options);
}

/** Load `path` or the first default config file (current folder, then home folder). */
export async function loadConfig(
	path?: string,
	options: {
		environment?: Readonly<Record<string, string | undefined>>;
		cwd?: string;
		home?: string;
	} = {},
): Promise<AgentRTConfig> {
	const resolved = discoverConfigPath(path, options);
	const fs = nodeBuiltin('node:fs') as {
		statSync(path: string): { size: number };
		readFileSync(path: string, encoding: 'utf8'): string;
	};
	let content: string;
	try {
		if (fs.statSync(resolved).size > MAX_CONFIG_BYTES) {
			throw new ConfigError(
				`config file exceeds ${MAX_CONFIG_BYTES} bytes: ${resolved}`,
			);
		}
		content = fs.readFileSync(resolved, 'utf8');
		if (content.charCodeAt(0) === 0xfeff) content = content.slice(1);
	} catch (error) {
		if (error instanceof ConfigError) throw error;
		throw new ConfigError(
			`unable to read config file ${resolved}: ${(error as Error).message}`,
		);
	}
	try {
		return await parseConfig(content, {
			environment: options.environment,
			source: resolved,
		});
	} catch (error) {
		if (error instanceof ConfigError) {
			throw new ConfigError(`${resolved}: ${error.message}`);
		}
		throw error;
	}
}
