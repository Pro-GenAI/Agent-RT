/** Agent checkpoints and checkpoint stores. */
import {
	type JSONValue,
	type ModelMessage,
	type WorkflowState,
	validateJSONValue,
} from '../types.js';
import { type AgentConfig } from '../agent/config.js';
import { type AgentRunResult } from '../agent/limits.js';

export interface AgentCheckpoint {
	checkpointId: string;
	agentName: string;
	messages: ModelMessage[];
	workflowState?: WorkflowState;
	turns: number;
	toolCalls: number;
	totalTokens: number;
	metadata?: Record<string, JSONValue>;
	createdAtMs: number;
}

function validateAgentCheckpoint(checkpoint: AgentCheckpoint): void {
	if (!checkpoint.checkpointId.trim()) {
		throw new Error('checkpointId must not be empty');
	}
	if (!checkpoint.agentName.trim()) {
		throw new Error('checkpoint agentName must not be empty');
	}
	if (
		checkpoint.turns < 0 ||
		checkpoint.toolCalls < 0 ||
		checkpoint.totalTokens < 0
	) {
		throw new Error('checkpoint counters must be non-negative');
	}
	validateJSONValue(checkpoint.metadata ?? {});
}

export interface CheckpointStore {
	save(checkpoint: AgentCheckpoint): AgentCheckpoint;
	load(checkpointId: string): AgentCheckpoint | undefined;
	delete(checkpointId: string): boolean;
	list(options?: { agentName?: string }): AgentCheckpoint[];
}

export class InMemoryCheckpointStore implements CheckpointStore {
	private readonly checkpoints = new Map<string, AgentCheckpoint>();

	save(checkpoint: AgentCheckpoint): AgentCheckpoint {
		validateAgentCheckpoint(checkpoint);
		this.checkpoints.set(checkpoint.checkpointId, checkpoint);
		return checkpoint;
	}

	load(checkpointId: string): AgentCheckpoint | undefined {
		return this.checkpoints.get(checkpointId);
	}

	delete(checkpointId: string): boolean {
		return this.checkpoints.delete(checkpointId);
	}

	list(options: { agentName?: string } = {}): AgentCheckpoint[] {
		return [...this.checkpoints.values()]
			.filter(
				(checkpoint) =>
					options.agentName === undefined ||
					checkpoint.agentName === options.agentName,
			)
			.sort(
				(a, b) =>
					b.createdAtMs - a.createdAtMs ||
					b.checkpointId.localeCompare(a.checkpointId),
			);
	}
}

export function checkpointFromResult(
	checkpointId: string,
	agent: AgentConfig,
	result: AgentRunResult,
	options: {
		workflowState?: WorkflowState;
		metadata?: Record<string, JSONValue>;
		createdAtMs?: number;
	} = {},
): AgentCheckpoint {
	const checkpoint: AgentCheckpoint = {
		checkpointId,
		agentName: agent.name,
		messages: [...result.messages],
		workflowState: options.workflowState,
		turns: result.turns,
		toolCalls: result.toolCalls,
		totalTokens: result.totalTokens,
		metadata: options.metadata ?? {},
		createdAtMs: options.createdAtMs ?? Date.now(),
	};
	validateAgentCheckpoint(checkpoint);
	return checkpoint;
}
