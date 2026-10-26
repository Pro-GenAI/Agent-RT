/** Virtual file systems, workspace files, and file-system backends. */

export interface FileStat {
	path: string;
	size: number;
	isDirectory: boolean;
}

export interface FileSystem {
	list(path?: string): FileStat[];
	read(path: string): Uint8Array;
	write(
		path: string,
		data: Uint8Array,
		options?: { overwrite?: boolean },
	): void;
	delete(path: string): boolean;
	move(
		source: string,
		destination: string,
		options?: { overwrite?: boolean },
	): void;
	copy(
		source: string,
		destination: string,
		options?: { overwrite?: boolean },
	): void;
	glob(pattern: string): string[];
	search(text: string, options?: { path?: string }): string[];
}

export class InMemoryFileSystem implements FileSystem {
	private readonly files = new Map<string, Uint8Array>();

	private normalize(path: string): string {
		const parts: string[] = [];
		for (const part of path.trim().replace(/^\/+/, '').split('/')) {
			if (!part || part === '.') continue;
			if (part === '..') {
				if (parts.length === 0)
					throw new Error('path escapes filesystem root');
				parts.pop();
			} else {
				parts.push(part);
			}
		}
		return parts.join('/');
	}

	list(path = ''): FileStat[] {
		const prefix = this.normalize(path);
		const prefixWithSep = prefix ? prefix + '/' : '';
		const children = new Map<string, FileStat>();
		for (const [filePath, data] of this.files.entries()) {
			if (!filePath.startsWith(prefixWithSep)) continue;
			const remainder = filePath.slice(prefixWithSep.length);
			if (!remainder) {
				children.set(filePath, {
					path: filePath,
					size: data.length,
					isDirectory: false,
				});
				continue;
			}
			const slash = remainder.indexOf('/');
			const first = slash === -1 ? remainder : remainder.slice(0, slash);
			const childPath = prefixWithSep + first;
			children.set(childPath, {
				path: childPath,
				size: slash === -1 ? data.length : 0,
				isDirectory: slash !== -1,
			});
		}
		return [...children.values()].sort((a, b) =>
			a.path.localeCompare(b.path),
		);
	}

	read(path: string): Uint8Array {
		const normalized = this.normalize(path);
		const data = this.files.get(normalized);
		if (!data) throw new Error('file not found: ' + normalized);
		return new Uint8Array(data);
	}

	write(
		path: string,
		data: Uint8Array,
		options: { overwrite?: boolean } = {},
	): void {
		const normalized = this.normalize(path);
		if (!normalized) throw new Error('cannot write filesystem root');
		if (options.overwrite === false && this.files.has(normalized)) {
			throw new Error('file already exists: ' + normalized);
		}
		this.files.set(normalized, new Uint8Array(data));
	}

	/** Remove every file. Deleting the root through `delete` is refused. */
	clear(): void {
		this.files.clear();
	}

	delete(path: string): boolean {
		const normalized = this.normalize(path);
		if (!normalized) {
			throw new Error('cannot delete filesystem root; use clear()');
		}
		if (this.files.delete(normalized)) return true;
		const prefix = normalized ? normalized + '/' : '';
		const matches = [...this.files.keys()].filter((key) =>
			key.startsWith(prefix),
		);
		for (const key of matches) this.files.delete(key);
		return matches.length > 0;
	}

	move(
		source: string,
		destination: string,
		options: { overwrite?: boolean } = {},
	): void {
		const data = this.read(source);
		if (this.normalize(source) === this.normalize(destination)) return;
		this.write(destination, data, options);
		this.delete(source);
	}

	copy(
		source: string,
		destination: string,
		options: { overwrite?: boolean } = {},
	): void {
		const data = this.read(source);
		if (this.normalize(source) === this.normalize(destination)) return;
		this.write(destination, data, options);
	}

	glob(pattern: string): string[] {
		// `*` and `?` stay within a path segment; `**` crosses segments.
		const normalized = this.normalize(pattern);
		let source = '';
		for (let index = 0; index < normalized.length; index += 1) {
			const character = normalized[index];
			if (normalized.startsWith('**', index)) {
				source += '.*';
				index += 1;
			} else if (character === '*') {
				source += '[^/]*';
			} else if (character === '?') {
				source += '[^/]';
			} else {
				source += character.replace(/[|\\{}()[\]^$+*?.]/g, '\\$&');
			}
		}
		const regex = new RegExp('^' + source + '$');
		return [...this.files.keys()].filter((path) => regex.test(path)).sort();
	}

	search(text: string, options: { path?: string } = {}): string[] {
		const prefix = this.normalize(options.path ?? '');
		const prefixWithSep = prefix ? prefix + '/' : '';
		const decoder = new TextDecoder();
		return [...this.files.entries()]
			.filter(
				([filePath, data]) =>
					(!prefix ||
						filePath === prefix ||
						filePath.startsWith(prefixWithSep)) &&
					decoder.decode(data).includes(text),
			)
			.map(([filePath]) => filePath)
			.sort();
	}
}

export class WorkspaceFiles {
	private readonly encoder = new TextEncoder();
	private readonly decoder = new TextDecoder();

	constructor(readonly filesystem: FileSystem) {}

	list(path = ''): FileStat[] {
		return this.filesystem.list(path);
	}

	readText(path: string): string {
		return this.decoder.decode(this.filesystem.read(path));
	}

	writeText(
		path: string,
		text: string,
		options: { overwrite?: boolean } = {},
	): void {
		this.filesystem.write(path, this.encoder.encode(text), options);
	}

	createText(path: string, text: string): void {
		this.writeText(path, text, { overwrite: false });
	}

	delete(path: string): boolean {
		return this.filesystem.delete(path);
	}

	/** Remove everything in the workspace (the root cannot be `delete`d). */
	clear(): void {
		const filesystem = this.filesystem as FileSystem & {
			clear?: () => void;
		};
		if (typeof filesystem.clear === 'function') {
			filesystem.clear();
			return;
		}
		for (const entry of this.filesystem.list('')) {
			this.filesystem.delete(entry.path);
		}
	}

	move(
		source: string,
		destination: string,
		options: { overwrite?: boolean } = {},
	): void {
		this.filesystem.move(source, destination, options);
	}

	copy(
		source: string,
		destination: string,
		options: { overwrite?: boolean } = {},
	): void {
		this.filesystem.copy(source, destination, options);
	}

	search(text: string, options: { path?: string } = {}): string[] {
		return this.filesystem.search(text, options);
	}

	glob(pattern: string): string[] {
		return this.filesystem.glob(pattern);
	}

	exactEdit(
		path: string,
		oldText: string,
		newText: string,
		options: { expectedOccurrences?: number } = {},
	): void {
		const expected = options.expectedOccurrences ?? 1;
		if (!Number.isInteger(expected) || expected < 1) {
			throw new Error(
				'expectedOccurrences must be an integer of at least 1',
			);
		}
		const current = this.readText(path);
		const count = current.split(oldText).length - 1;
		if (count !== expected) {
			throw new Error(
				'exact edit expected ' +
					expected +
					' occurrences, found ' +
					count,
			);
		}
		this.writeText(path, current.split(oldText).join(newText));
	}

	applyUnifiedPatch(path: string, patch: string): void {
		const original = patchLines(this.readText(path));
		const lines = patchLines(patch);
		const header = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,\d+)? @@/;
		const hunks: Array<{ oldStart: number; body: string[] }> = [];
		let index = 0;
		while (index < lines.length) {
			const match = lines[index].match(header);
			if (!match) {
				index += 1;
				continue;
			}
			const oldCount = match[2] === undefined ? 1 : Number(match[2]);
			// An empty old range (-N,0) names the line *before* the insertion
			// point, so it already is the zero-based insertion index.
			const oldStart =
				oldCount === 0 ? Number(match[1]) : Number(match[1]) - 1;
			index += 1;
			const body: string[] = [];
			while (index < lines.length && !lines[index].startsWith('@@ ')) {
				// Lines inside a hunk are never headers: a removed "-- x" line
				// looks like "--- x" and must not be skipped.
				body.push(lines[index]);
				index += 1;
			}
			hunks.push({ oldStart, body });
		}
		if (hunks.length === 0)
			throw new Error('unified patch contains no hunks');

		const output: string[] = [];
		let cursor = 0;
		for (const hunk of hunks) {
			if (hunk.oldStart < cursor || hunk.oldStart > original.length) {
				throw new Error('unified patch hunk is out of range');
			}
			output.push(...original.slice(cursor, hunk.oldStart));
			cursor = hunk.oldStart;
			for (let line of hunk.body) {
				// Editors often strip the leading space of blank context lines.
				if (line === '\n' || line === '\r\n') line = ' ' + line;
				const prefix = line[0] ?? '';
				const value = line.slice(1);
				if (prefix === ' ') {
					if (
						cursor >= original.length ||
						original[cursor] !== value
					) {
						throw new Error('unified patch context mismatch');
					}
					output.push(original[cursor]);
					cursor += 1;
				} else if (prefix === '-') {
					if (
						cursor >= original.length ||
						original[cursor] !== value
					) {
						throw new Error('unified patch removal mismatch');
					}
					cursor += 1;
				} else if (prefix === '+') {
					output.push(value);
				} else if (line.startsWith('\\ No newline at end of file')) {
					continue;
				} else {
					throw new Error('unsupported unified patch line');
				}
			}
		}
		output.push(...original.slice(cursor));
		this.writeText(path, output.join(''));
	}
}

export function patchLines(value: string): string[] {
	const matches = value.match(/.*(?:\n|$)/g) ?? [];
	return matches.filter((line) => line.length > 0);
}

export type FileSystemFactory = (workspaceId: string) => FileSystem;

export class FileSystemBackendRegistry {
	private readonly factories = new Map<string, FileSystemFactory>();

	register(
		backend: string,
		factory: FileSystemFactory,
		options: { replace?: boolean } = {},
	): void {
		if (!backend.trim())
			throw new Error('filesystem backend name must not be empty');
		if (this.factories.has(backend) && !options.replace) {
			throw new Error(
				'filesystem backend already registered: ' + backend,
			);
		}
		this.factories.set(backend, factory);
	}

	create(backend: string, workspaceId: string): FileSystem {
		const factory = this.factories.get(backend);
		if (!factory)
			throw new Error('filesystem backend not registered: ' + backend);
		return factory(workspaceId);
	}

	list(): string[] {
		return [...this.factories.keys()].sort();
	}
}
