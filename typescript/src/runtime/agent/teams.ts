/** Agent routing, round-robin, speaker-selected, swarm, hierarchical, and speculative teams. */
import { type AgentConfig } from './config.js';
import {
	type AgentInvocation,
	type AgentInvoker,
	type IsolatedSubagentRunner,
	type SubagentSpec,
} from './delegation.js';
import { type AgentRunResult } from './limits.js';

export interface AgentRoute {
	name: string;
	agent: AgentConfig;
	invoker: AgentInvoker;
	intents?: string[];
	capabilities?: string[];
}

export class AgentRouter {
	constructor(readonly routes: AgentRoute[]) {}

	select(
		options: {
			intent?: string;
			requiredCapabilities?: string[];
		} = {},
	): AgentRoute {
		const required = new Set(options.requiredCapabilities ?? []);
		const candidates = this.routes
			.filter((route) =>
				[...required].every((capability) =>
					(route.capabilities ?? []).includes(capability),
				),
			)
			.map((route) => ({
				route,
				score:
					[...required].filter((capability) =>
						(route.capabilities ?? []).includes(capability),
					).length +
					(options.intent &&
					(route.intents ?? []).includes(options.intent)
						? 1000
						: 0),
			}))
			.sort(
				(left, right) =>
					right.score - left.score ||
					left.route.name.localeCompare(right.route.name),
			);
		if (!candidates.length) throw new Error('no matching agent route');
		return candidates[0].route;
	}

	async route(
		invocation: AgentInvocation,
		options: { intent?: string; requiredCapabilities?: string[] } = {},
	): Promise<{ route: string; result: AgentRunResult }> {
		const selected = this.select(options);
		return {
			route: selected.name,
			result: await selected.invoker.invoke(selected.agent, invocation),
		};
	}
}

export interface TeamMember {
	name: string;
	agent: AgentConfig;
	invoker: AgentInvoker;
}

export interface TeamTurn {
	speaker: string;
	result: AgentRunResult;
}

export class RoundRobinTeam {
	private index = 0;

	constructor(readonly members: TeamMember[]) {
		if (!members.length)
			throw new Error('round-robin team requires at least one member');
	}

	async step(invocation: AgentInvocation): Promise<TeamTurn> {
		const member = this.members[this.index];
		this.index = (this.index + 1) % this.members.length;
		return {
			speaker: member.name,
			result: await member.invoker.invoke(member.agent, invocation),
		};
	}

	async run(invocations: AgentInvocation[]): Promise<TeamTurn[]> {
		const turns: TeamTurn[] = [];
		for (const invocation of invocations)
			turns.push(await this.step(invocation));
		return turns;
	}
}

export type SpeakerSelector = (
	members: TeamMember[],
	invocation: AgentInvocation,
	history: TeamTurn[],
) => Promise<string>;

export class ModelSelectedSpeakerTeam {
	readonly history: TeamTurn[] = [];

	constructor(
		readonly members: TeamMember[],
		readonly selector: SpeakerSelector,
	) {
		if (!members.length)
			throw new Error('speaker team requires at least one member');
	}

	async step(invocation: AgentInvocation): Promise<TeamTurn> {
		const selectedName = await this.selector(this.members, invocation, [
			...this.history,
		]);
		const member = this.members.find(
			(candidate) => candidate.name === selectedName,
		);
		if (!member)
			throw new Error('selector chose unknown speaker: ' + selectedName);
		const turn = {
			speaker: member.name,
			result: await member.invoker.invoke(member.agent, invocation),
		};
		this.history.push(turn);
		return turn;
	}
}

export interface SwarmDecision {
	nextAgent?: string;
	reason?: string;
}

export type SwarmHandoffPolicy = (
	currentAgent: string,
	result: AgentRunResult,
) => Promise<SwarmDecision>;

export class SwarmTeam {
	constructor(
		readonly members: Record<
			string,
			{ agent: AgentConfig; invoker: AgentInvoker }
		>,
		readonly handoffPolicy: SwarmHandoffPolicy,
	) {}

	async run(
		start: string,
		invocation: AgentInvocation,
		maxHandoffs = 8,
	): Promise<TeamTurn[]> {
		if (!this.members[start])
			throw new Error('unknown swarm member: ' + start);
		const turns: TeamTurn[] = [];
		let current = start;
		for (let i = 0; i <= maxHandoffs; i += 1) {
			const member = this.members[current];
			const result = await member.invoker.invoke(
				member.agent,
				invocation,
			);
			turns.push({ speaker: current, result });
			const decision = await this.handoffPolicy(current, result);
			if (!decision.nextAgent) return turns;
			if (!this.members[decision.nextAgent]) {
				throw new Error('unknown swarm member: ' + decision.nextAgent);
			}
			current = decision.nextAgent;
		}
		throw new Error('swarm handoff limit exceeded');
	}
}

export interface TeamNode {
	execute(invocation: AgentInvocation): Promise<unknown>;
}

export class AgentTeamNode implements TeamNode {
	constructor(
		readonly agent: AgentConfig,
		readonly invoker: AgentInvoker,
	) {}

	execute(invocation: AgentInvocation): Promise<AgentRunResult> {
		return this.invoker.invoke(this.agent, invocation);
	}
}

export class HierarchicalTeam implements TeamNode {
	constructor(
		readonly children: Record<string, TeamNode>,
		readonly selector: (invocation: AgentInvocation) => Promise<string>,
	) {}

	async execute(invocation: AgentInvocation): Promise<unknown> {
		const childName = await this.selector(invocation);
		const child = this.children[childName];
		if (!child) throw new Error('unknown team child: ' + childName);
		return child.execute(invocation);
	}
}

export class ParallelSubagentExecutor {
	constructor(readonly runner: IsolatedSubagentRunner) {}

	async run(specs: SubagentSpec[]): Promise<AgentRunResult[]> {
		return Promise.all(specs.map((spec) => this.runner.run(spec)));
	}
}

export interface MapReduceResult {
	mapped: AgentRunResult[];
	reduced: unknown;
}

export class MapReduceOrchestrator {
	constructor(
		readonly runner: IsolatedSubagentRunner,
		readonly reducer: (results: AgentRunResult[]) => Promise<unknown>,
	) {}

	async run(specs: SubagentSpec[]): Promise<MapReduceResult> {
		const mapped = await Promise.all(
			specs.map((spec) => this.runner.run(spec)),
		);
		return {
			mapped,
			reduced: await this.reducer(mapped),
		};
	}
}

export interface SpeculativeBranch {
	name: string;
	spec: SubagentSpec;
}

export interface SpeculativeBranchResult {
	name: string;
	result: AgentRunResult;
	score: number;
}

export interface SpeculativeResult {
	branches: SpeculativeBranchResult[];
	selected?: SpeculativeBranchResult;
	merged?: unknown;
}

export class SpeculativeOrchestrator {
	constructor(
		readonly runner: IsolatedSubagentRunner,
		readonly scorer: (
			name: string,
			result: AgentRunResult,
		) => Promise<number>,
		readonly merger?: (
			branches: SpeculativeBranchResult[],
		) => Promise<unknown>,
	) {}

	async run(
		branches: SpeculativeBranch[],
		mode: 'select' | 'merge' = 'select',
	): Promise<SpeculativeResult> {
		if (!branches.length) {
			throw new Error(
				'speculative execution requires at least one branch',
			);
		}
		if (mode === 'merge' && !this.merger) {
			throw new Error('merge mode requires a merger');
		}
		if (mode !== 'select' && mode !== 'merge') {
			throw new Error('unsupported speculative mode');
		}
		const results = await Promise.all(
			branches.map((branch) => this.runner.run(branch.spec)),
		);
		const scored: SpeculativeBranchResult[] = [];
		for (let index = 0; index < branches.length; index += 1) {
			scored.push({
				name: branches[index].name,
				result: results[index],
				score: await this.scorer(branches[index].name, results[index]),
			});
		}
		if (mode === 'merge') {
			return {
				branches: scored,
				merged: await this.merger!(scored),
			};
		}
		const selected = [...scored].sort(
			(left, right) =>
				right.score - left.score || left.name.localeCompare(right.name),
		)[0];
		return { branches: scored, selected };
	}
}
