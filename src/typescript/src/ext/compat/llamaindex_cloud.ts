import type {
	FileSystem,
	RetrievalProvider,
	RetrievalRegistry,
} from '../../index.js';
import { Document, NodeWithScore, TextNode } from './llamaindex_rag.js';

export type ParsedDocument = {
	text: string;
	metadata?: Record<string, unknown>;
	id?: string;
};

export type FileParser = (input: {
	path: string;
	data: Uint8Array;
}) =>
	| ParsedDocument
	| ParsedDocument[]
	| Promise<ParsedDocument | ParsedDocument[]>;

export class LlamaParseReader {
	constructor(
		readonly options: {
			fileSystem: FileSystem;
			parser: FileParser;
			paths?: string[];
			inputDir?: string;
			recursive?: boolean;
		},
	) {}

	async loadData(): Promise<Document[]> {
		const paths =
			this.options.paths ??
			(this.options.recursive
				? this.options.fileSystem.glob(
						this.options.inputDir
							? `${this.options.inputDir}/**/*`
							: '**/*',
					)
				: this.options.fileSystem
						.list(this.options.inputDir ?? '')
						.filter((entry) => !entry.isDirectory)
						.map((entry) => entry.path));
		const documents: Document[] = [];
		for (const path of paths) {
			const parsed = await this.options.parser({
				path,
				data: this.options.fileSystem.read(path),
			});
			for (const item of Array.isArray(parsed) ? parsed : [parsed]) {
				documents.push(
					new Document({
						id_: item.id ?? path,
						text: item.text,
						metadata: {
							filePath: path,
							...(item.metadata ?? {}),
						},
					}),
				);
			}
		}
		return documents;
	}
}

export class LlamaCloudRetriever {
	constructor(
		readonly source:
			| RetrievalProvider
			| { registry: RetrievalRegistry; providerName: string },
		readonly similarityTopK = 2,
		readonly filters?: Record<string, unknown>,
	) {}

	async retrieve(query: string): Promise<NodeWithScore[]> {
		const results =
			'search' in this.source
				? await this.source.search({
						text: query,
						limit: this.similarityTopK,
						filters: this.filters,
					})
				: await this.source.registry.search(this.source.providerName, {
						text: query,
						limit: this.similarityTopK,
						filters: this.filters,
					});
		return results.map(
			(result) =>
				new NodeWithScore(
					new TextNode({
						id_: result.id,
						text:
							typeof result.content === 'string'
								? result.content
								: JSON.stringify(result.content),
						metadata: {
							...(result.metadata ?? {}),
							title: result.title,
							...(result.uri ? { uri: result.uri } : {}),
						},
					}),
					result.score,
				),
		);
	}
}

export class LlamaCloudIndex {
	constructor(readonly retriever: LlamaCloudRetriever) {}

	asRetriever(): LlamaCloudRetriever {
		return this.retriever;
	}
}
