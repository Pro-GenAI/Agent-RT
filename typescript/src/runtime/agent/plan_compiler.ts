/** Compiling an agent's execution plan: visible tools, policies, and active stages. */
import {
	type ToolSelectionRequirement,
	normalizeToolSelectionPolicy,
	toolPolicyPermits,
} from '../tools/definitions.js';
import {
	type ContextItem,
	type ContextSelectionPolicy,
} from '../context/items.js';
import { type AgentConfig, type CompiledExecutionPlan } from './config.js';
import { ContextAssembler } from '../context/assembler.js';

import { type AgentLoopInternals } from './run_steps.js';

export function compilePlan(
	this: AgentLoopInternals,
	agent: AgentConfig,
	contextItems: ContextItem[] = [],
	contextPolicy?: ContextSelectionPolicy,
): CompiledExecutionPlan {
	const stageStarted = this.metrics ? performance.now() : undefined;
	const toolPolicy = normalizeToolSelectionPolicy(agent.toolPolicy);
	let visibleTools = this.toolRegistry?.definitions() ?? [];
	if (this.capabilityGrant)
		visibleTools = this.capabilityGrant.filterTools(visibleTools);
	visibleTools = this.promptInjectionDefense.filterTools(
		contextItems,
		visibleTools,
	);
	if (toolPolicy) {
		visibleTools = visibleTools.filter((tool) =>
			toolPolicyPermits(toolPolicy, tool.name),
		);
	}
	if (contextPolicy?.toolNames) {
		const isolatedToolNames = new Set(contextPolicy.toolNames);
		visibleTools = visibleTools.filter((tool) =>
			isolatedToolNames.has(tool.name),
		);
	}

	const visibleToolNames = new Set(visibleTools.map((tool) => tool.name));
	let toolSelection: ToolSelectionRequirement | undefined;
	if (!this.toolFilter && toolPolicy) {
		const missingRequired = (toolPolicy.required ?? []).filter(
			(name) => !visibleToolNames.has(name),
		);
		if (missingRequired.length > 0) {
			throw new Error(
				`required tools are unavailable: ${missingRequired.sort().join(', ')}`,
			);
		}
		toolSelection = {
			required: [...(toolPolicy.required ?? [])].sort(),
			preferred: (toolPolicy.preferred ?? []).filter((name) =>
				visibleToolNames.has(name),
			),
		};
	}

	const structuredOutput =
		agent.output?.schema !== undefined
			? {
					name: agent.name,
					schema: agent.output.schema,
					strict: true,
				}
			: undefined;
	const activeStages = [
		visibleTools.length > 0 ? 'tools' : undefined,
		this.toolFilter ? 'dynamic_tool_filter' : undefined,
		structuredOutput ? 'structured_output' : undefined,
		this.inputGuardrails.length > 0 ? 'input_guardrails' : undefined,
		this.outputGuardrails.length > 0 ? 'output_guardrails' : undefined,
	].filter((stage): stage is string => stage !== undefined);

	const simpleTextFastPath =
		activeStages.length === 0 &&
		toolPolicy === undefined &&
		(agent.model.fallbackModels?.length ?? 0) === 0 &&
		this.contextAssembler.constructor === ContextAssembler &&
		Object.keys(this.contextAssembler.options).length === 0;
	const plan: CompiledExecutionPlan = {
		modelSettings: agent.model,
		visibleTools,
		toolPolicy,
		structuredOutput,
		toolSelection,
		visibleToolNames,
		toolRegistryVersion: this.toolRegistry?.version,
		dynamicToolFilter: this.toolFilter !== undefined,
		simpleTextFastPath,
		activeStages,
	};
	if (stageStarted !== undefined) {
		this.metrics?.record(
			'agent_loop.stage.duration_ms',
			performance.now() - stageStarted,
			{ kind: 'histogram', labels: { stage: 'compile_plan' } },
		);
	}
	return plan;
}
