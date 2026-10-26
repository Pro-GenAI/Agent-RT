/** Agent configuration, model settings, and compiled execution plans. */
import { type ToolCall } from '../types.js';
import {
	type ToolDefinition,
	type ToolSelectionPolicy,
	type ToolSelectionRequirement,
} from '../tools/definitions.js';
import {
	type ModelProvider,
	type ModelStreamEvent,
	type StreamingModelProvider,
	type StructuredOutputRequirement,
} from '../models/interfaces.js';
import { type ModelTarget } from '../models/routing.js';

export type StreamEventHandler = (
	event: ModelStreamEvent,
) => void | Promise<void>;

export function isStreamingModelProvider(
	provider: ModelProvider,
): provider is StreamingModelProvider {
	return (
		typeof (provider as Partial<StreamingModelProvider>).stream ===
		'function'
	);
}

export interface ModelSettings {
	model: string;
	provider?: string;
	fallbackModels?: ModelTarget[];
	temperature?: number;
	maxOutputTokens?: number;
	metadata?: Record<string, unknown>;
}

export interface AgentOutputRequirements {
	format?: 'text' | 'json';
	schema?: Record<string, unknown>;
	maxRepairAttempts?: number;
}

export interface AgentConfig {
	name: string;
	role?: string;
	instructions: string;
	description?: string;
	model: ModelSettings;
	capabilities?: string[];
	output?: AgentOutputRequirements;
	toolPolicy?: ToolSelectionPolicy;
	metadata?: Record<string, unknown>;
	/** Strings, custom objects, stores or functions; see `MemorySource`. */
	memory?: readonly unknown[];
}

export interface CompiledExecutionPlan {
	modelSettings: ModelSettings;
	visibleTools: readonly ToolDefinition[];
	toolPolicy?: ToolSelectionPolicy;
	structuredOutput?: StructuredOutputRequirement;
	toolSelection?: ToolSelectionRequirement;
	visibleToolNames: ReadonlySet<string>;
	toolRegistryVersion?: number;
	dynamicToolFilter: boolean;
	simpleTextFastPath: boolean;
	activeStages: readonly string[];
}

export interface ToolExecutor {
	execute(call: ToolCall, signal?: AbortSignal): Promise<unknown>;
}
