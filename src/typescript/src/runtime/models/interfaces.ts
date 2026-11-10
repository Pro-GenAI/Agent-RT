/** Provider-neutral model requests, responses, embeddings, and streaming events. */
import { type ModelMessage } from '../types.js';
import {
	type ToolDefinition,
	type ToolSelectionRequirement,
} from '../tools/definitions.js';
import { type PromptCacheHint } from '../context/items.js';

export interface ModelUsage {
	inputTokens?: number;
	outputTokens?: number;
	totalTokens?: number;
	cachedTokens?: number;
	reasoningTokens?: number;
}

export interface EmbeddingRequest {
	input: unknown;
	model?: string;
	dimensions?: number;
	encodingFormat?: string;
}

export interface EmbeddingItem {
	index: number;
	embedding: number[] | string;
}

export interface EmbeddingResponse {
	data: EmbeddingItem[];
	model?: string;
	usage?: ModelUsage;
	raw?: unknown;
}

export interface EmbeddingModelProvider {
	embed(request: EmbeddingRequest): Promise<EmbeddingResponse>;
}

export interface TokenCountingModelProvider {
	countTokens(request: ModelRequest): Promise<number>;
}

export interface ModelCatalogEntry {
	id: string;
	created?: number | string;
	ownedBy?: string;
	displayName?: string;
	raw?: unknown;
}

export interface ModelCatalogProvider {
	listModels(): Promise<ModelCatalogEntry[]>;
	retrieveModel(model: string): Promise<ModelCatalogEntry>;
}

export interface BatchJob {
	id: string;
	status?: string;
	raw?: unknown;
}

export interface BatchProvider {
	createBatch(params: Record<string, unknown>): Promise<BatchJob>;
	retrieveBatch(id: string): Promise<BatchJob>;
	listBatches(params?: Record<string, unknown>): Promise<BatchJob[]>;
	cancelBatch(id: string): Promise<BatchJob>;
	batchResults?(id: string): Promise<unknown>;
}

export type FinishReason =
	'stop' | 'tool_calls' | 'length' | 'content_filter' | 'error' | 'other';

export interface StructuredOutputRequirement {
	name?: string;
	schema: Record<string, unknown>;
	strict?: boolean;
}

/**
 * JSON mode (OpenAI `response_format: { type: 'json_object' }`): any JSON
 * object, without a schema. OpenAI-compatible providers send JSON mode for it.
 */
export const JSON_OBJECT_OUTPUT: StructuredOutputRequirement = Object.freeze({
	schema: Object.freeze({ type: 'object' }),
	strict: false,
});

export function isJsonObjectOutput(
	requirement: StructuredOutputRequirement | undefined,
): boolean {
	return (
		requirement !== undefined &&
		requirement.strict === false &&
		requirement.name === undefined &&
		Object.keys(requirement.schema).length === 1 &&
		requirement.schema.type === 'object'
	);
}

export interface ReasoningConfig {
	effort?: string;
	summary?: string;
	thinking?: 'adaptive' | 'enabled' | 'disabled';
	budgetTokens?: number;
}

export interface ModelRequest {
	messages: ModelMessage[];
	model?: string;
	tools?: ToolDefinition[];
	temperature?: number;
	maxOutputTokens?: number;
	structuredOutput?: StructuredOutputRequirement;
	reasoning?: ReasoningConfig;
	toolSelection?: ToolSelectionRequirement;
	metadata?: Record<string, unknown>;
	promptCache?: PromptCacheHint;
	signal?: AbortSignal;
}

export interface ModelResponse {
	message: ModelMessage;
	model?: string;
	usage?: ModelUsage;
	finishReason?: FinishReason;
	raw?: unknown;
}

export interface ModelProvider {
	readonly name: string;
	complete(request: ModelRequest): Promise<ModelResponse>;
}

export type StreamEventType =
	| 'text_delta'
	| 'reasoning_delta'
	| 'status'
	| 'tool_call_delta'
	| 'completed';

export interface ModelStreamEvent {
	type: StreamEventType;
	text?: string;
	status?: string;
	toolCallId?: string;
	toolName?: string;
	argumentsDelta?: string;
	response?: ModelResponse;
	raw?: unknown;
}

export interface StreamingModelProvider extends ModelProvider {
	stream(request: ModelRequest): AsyncIterable<ModelStreamEvent>;
}
