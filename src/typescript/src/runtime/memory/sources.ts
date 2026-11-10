/** Agent memory sources and memory backend adapters. */
import { type ContentPart, type MemoryScope } from '../types.js';
import { type LongTermMemoryStore } from './long_term.js';
import { type ContextItem, type TrustLevel } from '../context/items.js';

export interface MemorySourceOptions {
	name?: string;
	trust?: TrustLevel;
	limit?: number;
	scope?: MemoryScope;
	/**
	 * Passed to the source's search call, for example a mem0 `userId` or a
	 * LangGraph `namespace`.
	 */
	searchOptions?: Record<string, unknown>;
}

/**
 * Options for one `AgentConfig.memory` entry. `source` is a string, any custom
 * object, a memory store or retriever (Agent RT, LangChain, LangGraph,
 * LlamaIndex, OpenAI Agents sessions, mem0), or a function
 * `(query) => value(s)` that may be async. Stores, retrievers and functions are
 * searched with the latest user prompt on every run and default to untrusted;
 * static values default to trusted.
 */
export class MemorySource {
	readonly limit: number;

	constructor(
		readonly source: unknown,
		readonly options: MemorySourceOptions = {},
	) {
		this.limit = options.limit ?? 5;
		if (!Number.isInteger(this.limit) || this.limit < 1)
			throw new Error('memory limit must be at least 1');
		if (options.name !== undefined && !options.name.trim())
			throw new Error('memory source name must not be empty');
	}
}

function memoryJSONReplacer(_key: string, value: unknown): unknown {
	if (value instanceof Set)
		return [...value].sort((a, b) => String(a).localeCompare(String(b)));
	if (value instanceof Map) return Object.fromEntries(value);
	if (typeof value === 'bigint') return value.toString();
	return value;
}

type MemoryShape = Record<string, unknown>;

function hasMethods(value: unknown, ...names: string[]): boolean {
	return (
		typeof value === 'object' &&
		value !== null &&
		names.every(
			(name) => typeof (value as MemoryShape)[name] === 'function',
		)
	);
}

/**
 * Reduce records, documents, nodes and messages from any framework to their
 * content plus role/tags/metadata; other values pass through.
 */
function memoryItem(value: unknown): unknown {
	if (typeof value !== 'object' || value === null) return value;
	let item = value as MemoryShape;
	// LlamaIndex NodeWithScore.
	if (item.node !== undefined && 'score' in item)
		item = item.node as MemoryShape;
	let content: unknown;
	if (typeof item.pageContent === 'string') {
		content = item.pageContent; // LangChain Document
	} else if (hasMethods(item, 'getContent')) {
		content = (item.getContent as () => unknown).call(item); // LlamaIndex node
	} else if (
		'content' in item &&
		['kind', 'role', 'additional_kwargs', 'lc_kwargs'].some(
			(marker) => marker in item,
		)
	) {
		content = item.content; // Agent RT record, chat message
	} else if (['key', 'value', 'namespace'].every((name) => name in item)) {
		return { key: item.key, value: item.value }; // LangGraph store Item
	} else {
		return value;
	}
	const fields: MemoryShape = {};
	let role = item.role;
	if (
		role === undefined &&
		('additional_kwargs' in item || 'lc_kwargs' in item)
	) {
		role =
			typeof item._getType === 'function'
				? (item._getType as () => unknown).call(item)
				: item.type; // LangChain message
	}
	if (role !== undefined) fields.role = role;
	if (Array.isArray(item.tags) && item.tags.length) fields.tags = item.tags;
	if (
		item.metadata &&
		typeof item.metadata === 'object' &&
		Object.keys(item.metadata).length
	)
		fields.metadata = item.metadata;
	if (Object.keys(fields).length === 0 && typeof content === 'string')
		return content;
	if (item.kind !== undefined && fields.role === undefined)
		fields.kind = item.kind;
	return { content, ...fields };
}

function memoryPart(value: unknown): ContentPart {
	// Store records and documents carry embeddings and bookkeeping the model
	// does not need.
	value = memoryItem(value);
	if (typeof value === 'string') return { type: 'text', text: value };
	const text = JSON.stringify(value, memoryJSONReplacer);
	return {
		type: 'json',
		data: text === undefined ? String(value) : JSON.parse(text),
	};
}

async function callMemory(
	value: unknown,
	names: string[],
	...args: unknown[]
): Promise<unknown> {
	const target = value as MemoryShape;
	const name = names.find(
		(candidate) => typeof target[candidate] === 'function',
	);
	return await (target[name as string] as (...a: unknown[]) => unknown).apply(
		target,
		args,
	);
}

function asList(value: unknown): unknown[] {
	return Array.isArray(value) ? value : [value];
}

/** Search a dynamic memory source; `undefined` for static values. */
async function lookupMemory(
	value: unknown,
	query: string,
	source: MemorySource,
): Promise<unknown[] | undefined> {
	const options = { ...(source.options.searchOptions ?? {}) };
	const limit = source.limit;
	if (hasMethods(value, 'recall', 'remember')) {
		// loadMemory() backend.
		return asList(await callMemory(value, ['recall'], query, limit));
	}
	if (hasMethods(value, 'search', 'write', 'read')) {
		// Agent RT long-term store.
		const results = (value as LongTermMemoryStore).search({
			text: query || undefined,
			scope: source.options.scope,
			limit,
		});
		return results.map((result) => result.record ?? result);
	}
	if (hasMethods(value, 'search', 'put', 'get')) {
		// LangGraph BaseStore.
		const { namespace = [], ...rest } = options;
		return asList(
			await callMemory(value, ['search'], namespace, {
				query: query || undefined,
				limit,
				...rest,
			}),
		);
	}
	if (hasMethods(value, 'search', 'add', 'getAll')) {
		// mem0 Memory / MemoryClient (scope it with `filters`).
		const found = (await callMemory(value, ['search'], query, {
			topK: limit,
			...options,
		})) as unknown;
		return asList(
			Array.isArray(found)
				? found
				: ((found as MemoryShape)?.results ?? []),
		);
	}
	if (hasMethods(value, 'getItems', 'addItems')) {
		// OpenAI Agents Session.
		return asList(await callMemory(value, ['getItems'], limit));
	}
	if (
		hasMethods(value, 'put') &&
		(hasMethods(value, 'getAll') || hasMethods(value, 'getAllMessages'))
	) {
		// LlamaIndex Memory / ChatMemoryBuffer.
		const found = await callMemory(
			value,
			['getMessages', 'get'],
			query || undefined,
		);
		return asList(found).slice(-limit);
	}
	if (hasMethods(value, 'getMessages', 'addMessage')) {
		// LangChain chat message history.
		return asList(await callMemory(value, ['getMessages'])).slice(-limit);
	}
	if (hasMethods(value, 'retrieve')) {
		// LlamaIndex retriever.
		return asList(await callMemory(value, ['retrieve'], query));
	}
	if (
		hasMethods(value, 'invoke') &&
		/Retriever/.test((value as object).constructor?.name ?? '')
	) {
		// LangChain retriever.
		return asList(await callMemory(value, ['invoke'], query));
	}
	if (typeof value === 'function')
		return asList(await (value as (query: string) => unknown)(query));
	return undefined;
}

/** Turn `AgentConfig.memory` entries into context items for one run. */
export async function resolveAgentMemory(
	memory: readonly unknown[],
	query = '',
): Promise<ContextItem[]> {
	const items: ContextItem[] = [];
	for (const [index, entry] of memory.entries()) {
		const source =
			entry instanceof MemorySource ? entry : new MemorySource(entry);
		const name = source.options.name ?? 'memory-' + (index + 1);
		const found = await lookupMemory(source.source, query, source);
		const dynamic = found !== undefined;
		const values = dynamic
			? found.slice(0, source.limit)
			: asList(source.source);
		const trust =
			source.options.trust ?? (dynamic ? 'untrusted' : 'trusted');
		values.forEach((item, position) => {
			if (item === null || item === undefined) return;
			items.push({
				id: 'memory:' + name + ':' + (position + 1),
				kind: 'retrieved',
				content: [memoryPart(item)],
				metadata: { memory_source: name },
				trust,
			});
		});
	}
	return items;
}
