/** Plans, plan tracking, verification, and planner execution. */

export type PlanStepStatus =
	'pending' | 'ready' | 'in_progress' | 'completed' | 'blocked' | 'failed';

export interface PlanStep {
	id: string;
	title: string;
	description?: string;
	dependencies?: string[];
	milestone?: string;
	acceptanceCriteria?: string[];
	status?: PlanStepStatus;
	result?: unknown;
	notes?: string[];
}

export interface Plan {
	id: string;
	goal: string;
	steps: PlanStep[];
	completionCriteria?: string[];
	version?: number;
}

function validatePlan(plan: Plan): void {
	const ids = plan.steps.map((step) => step.id);
	if (new Set(ids).size !== ids.length) {
		throw new Error('plan step ids must be unique');
	}
	const known = new Set(ids);
	for (const step of plan.steps) {
		for (const dependency of step.dependencies ?? []) {
			if (!known.has(dependency)) {
				throw new Error(
					'plan step ' +
						step.id +
						' has unknown dependency: ' +
						dependency,
				);
			}
			if (dependency === step.id) {
				throw new Error(
					'plan step ' + step.id + ' cannot depend on itself',
				);
			}
		}
	}
}

export interface PlanProgress {
	total: number;
	completed: number;
	failed: number;
	blocked: number;
	fractionComplete: number;
}

export class PlanTracker {
	plan: Plan;

	constructor(plan: Plan) {
		validatePlan(plan);
		this.plan = {
			...plan,
			version: plan.version ?? 1,
			steps: plan.steps.map((step) => ({
				...step,
				dependencies: [...(step.dependencies ?? [])],
				acceptanceCriteria: [...(step.acceptanceCriteria ?? [])],
				notes: [...(step.notes ?? [])],
				status: step.status ?? 'pending',
			})),
			completionCriteria: [...(plan.completionCriteria ?? [])],
		};
	}

	get(stepId: string): PlanStep {
		const step = this.plan.steps.find(
			(candidate) => candidate.id === stepId,
		);
		if (!step) throw new Error('unknown plan step: ' + stepId);
		return step;
	}

	private replaceStep(stepId: string, changes: Partial<PlanStep>): PlanStep {
		let found = false;
		this.plan = {
			...this.plan,
			version: (this.plan.version ?? 1) + 1,
			steps: this.plan.steps.map((step) => {
				if (step.id !== stepId) return step;
				found = true;
				return { ...step, ...changes };
			}),
		};
		if (!found) throw new Error('unknown plan step: ' + stepId);
		return this.get(stepId);
	}

	start(stepId: string): PlanStep {
		const step = this.get(stepId);
		const incomplete = (step.dependencies ?? []).filter(
			(dependency) =>
				(this.get(dependency).status ?? 'pending') !== 'completed',
		);
		if (incomplete.length) {
			throw new Error(
				'step ' +
					stepId +
					' has incomplete dependencies: ' +
					incomplete.join(','),
			);
		}
		return this.replaceStep(stepId, { status: 'in_progress' });
	}

	complete(stepId: string, result?: unknown, note?: string): PlanStep {
		const step = this.get(stepId);
		return this.replaceStep(stepId, {
			status: 'completed',
			result,
			notes: [...(step.notes ?? []), ...(note ? [note] : [])],
		});
	}

	fail(stepId: string, note: string): PlanStep {
		const step = this.get(stepId);
		return this.replaceStep(stepId, {
			status: 'failed',
			notes: [...(step.notes ?? []), note],
		});
	}

	block(stepId: string, note: string): PlanStep {
		const step = this.get(stepId);
		return this.replaceStep(stepId, {
			status: 'blocked',
			notes: [...(step.notes ?? []), note],
		});
	}

	replan(
		steps: PlanStep[],
		options: { reason: string; completionCriteria?: string[] },
	): Plan {
		const completed = new Map(
			this.plan.steps
				.filter((step) => (step.status ?? 'pending') === 'completed')
				.map((step) => [step.id, step]),
		);
		const merged = steps.map((step) => completed.get(step.id) ?? step);
		const next: Plan = {
			id: this.plan.id,
			goal: this.plan.goal,
			steps: merged,
			completionCriteria:
				options.completionCriteria ??
				this.plan.completionCriteria ??
				[],
			version: (this.plan.version ?? 1) + 1,
		};
		validatePlan(next);
		this.plan = next;
		if (this.plan.steps.length) {
			const first = this.plan.steps[0];
			this.replaceStep(first.id, {
				notes: [...(first.notes ?? []), 'replanned: ' + options.reason],
			});
		}
		return this.plan;
	}

	get progress(): PlanProgress {
		const statuses = this.plan.steps.map(
			(step) => step.status ?? 'pending',
		);
		const completed = statuses.filter(
			(status) => status === 'completed',
		).length;
		const total = statuses.length;
		return {
			total,
			completed,
			failed: statuses.filter((status) => status === 'failed').length,
			blocked: statuses.filter((status) => status === 'blocked').length,
			fractionComplete: total === 0 ? 1 : completed / total,
		};
	}
}

export interface VerificationIssue {
	criterion: string;
	message: string;
	stepId?: string;
}

export interface VerificationReport {
	passed: boolean;
	issues?: VerificationIssue[];
	reviewer?: string;
}

export class PlanVerifier {
	verify(plan: Plan): VerificationReport {
		const issues: VerificationIssue[] = [];
		for (const step of plan.steps) {
			if ((step.status ?? 'pending') !== 'completed') {
				issues.push({
					criterion: 'step_completed',
					message:
						'step ' + step.id + ' is ' + (step.status ?? 'pending'),
					stepId: step.id,
				});
				for (const criterion of step.acceptanceCriteria ?? []) {
					issues.push({
						criterion,
						message:
							'acceptance criterion cannot be satisfied before completion',
						stepId: step.id,
					});
				}
			}
		}
		if (
			(plan.completionCriteria ?? []).length > 0 &&
			plan.steps.length === 0
		) {
			issues.push({
				criterion: 'plan_nonempty',
				message: 'completion criteria require at least one plan step',
			});
		}
		return { passed: issues.length === 0, issues };
	}
}

export interface Planner {
	createPlan(goal: string): Promise<Plan>;
}

export interface StepExecutor {
	executeStep(step: PlanStep): Promise<unknown>;
}

export interface PlanExecutionResult {
	plan: Plan;
	verification: VerificationReport;
}

export class PlannerExecutor {
	constructor(
		readonly planner: Planner,
		readonly executor: StepExecutor,
		readonly verifier: PlanVerifier = new PlanVerifier(),
	) {}

	async run(goal: string): Promise<PlanExecutionResult> {
		const tracker = new PlanTracker(await this.planner.createPlan(goal));
		const pending = new Set(tracker.plan.steps.map((step) => step.id));
		while (pending.size) {
			let progressed = false;
			for (const step of [...tracker.plan.steps]) {
				if (!pending.has(step.id)) continue;
				if (
					(step.dependencies ?? []).every(
						(dependency) =>
							(tracker.get(dependency).status ?? 'pending') ===
							'completed',
					)
				) {
					tracker.start(step.id);
					try {
						const result = await this.executor.executeStep(
							tracker.get(step.id),
						);
						tracker.complete(step.id, result);
					} catch (error) {
						tracker.fail(
							step.id,
							error instanceof Error
								? error.message
								: String(error),
						);
					}
					pending.delete(step.id);
					progressed = true;
				}
			}
			if (!progressed && pending.size) {
				for (const stepId of [...pending]) {
					tracker.block(
						stepId,
						'dependencies did not become satisfiable',
					);
					pending.delete(stepId);
				}
			}
		}
		return {
			plan: tracker.plan,
			verification: this.verifier.verify(tracker.plan),
		};
	}
}
