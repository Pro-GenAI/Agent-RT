/** Node platform helpers: lazy module loading, optional peers, and byte buffers (TypeScript only). */
import { runtimeSettingsMap } from './settings.js';

export function runtimeEnvironment(): Record<string, string | undefined> {
	return runtimeSettingsMap;
}

/** The host's real `PATH` (an OS lookup path, not Agent RT configuration). */
export function hostSearchPath(): string | undefined {
	return (
		globalThis as { process?: { env?: Record<string, string | undefined> } }
	).process?.env?.PATH;
}

declare const require: ((moduleName: string) => any) | undefined;

/**
 * Load a Node module lazily. `Function('return require(...)')` cannot work:
 * code built with the Function constructor runs in global scope, where
 * `require` does not exist in either CommonJS or ESM. Built-ins use
 * `process.getBuiltinModule` (works from any module system); optional peer
 * packages use the module's own `require`.
 */
export function nodeModule(name: string): any {
	const runtime = globalThis as unknown as {
		process?: { getBuiltinModule?: (id: string) => unknown };
	};
	if (name.startsWith('node:') && runtime.process?.getBuiltinModule) {
		const builtin = runtime.process.getBuiltinModule(name);
		if (builtin !== undefined) return builtin;
	}
	if (typeof require === 'function') return require(name);
	throw new Error('unable to load Node module ' + name);
}

/**
 * Import an optional peer package. The specifier goes through a variable so
 * bundlers (esbuild, webpack, ...) neither inline the package nor fail to
 * resolve it when the application has not installed it.
 */
export async function importOptionalPeer<T>(name: string): Promise<T> {
	const specifier = name;
	return (await import(specifier)) as T;
}

export function concatBytes(chunks: Uint8Array[]): Uint8Array {
	const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
	const result = new Uint8Array(length);
	let offset = 0;
	for (const chunk of chunks) {
		result.set(chunk, offset);
		offset += chunk.length;
	}
	return result;
}
