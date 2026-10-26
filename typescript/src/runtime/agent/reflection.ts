/** Review findings, reflection passes, and critic panels. */
import { type VerificationReport } from './planning.js';

export interface ReviewFinding {
	severity: 'info' | 'warning' | 'error';
	message: string;
	repairable?: boolean;
}

export interface ReviewResult {
	findings: ReviewFinding[];
}

export class ReflectionPass {
	constructor(
		readonly reviewer: (work: unknown) => Promise<ReviewResult>,
		readonly repair: (
			work: unknown,
			review: ReviewResult,
		) => Promise<unknown>,
		readonly maxRepairs = 1,
	) {}

	async run(work: unknown): Promise<{ work: unknown; review: ReviewResult }> {
		let current = work;
		for (let attempt = 0; attempt <= this.maxRepairs; attempt += 1) {
			const review = await this.reviewer(current);
			const passed = !review.findings.some(
				(finding) => finding.severity === 'error',
			);
			if (passed || attempt >= this.maxRepairs)
				return { work: current, review };
			if (
				!review.findings.some((finding) => finding.repairable !== false)
			) {
				return { work: current, review };
			}
			current = await this.repair(current, review);
		}
		throw new Error('unreachable');
	}
}

export interface CriticAgent {
	name: string;
	review(work: unknown): Promise<VerificationReport>;
}

export class CriticPanel {
	constructor(readonly critics: CriticAgent[]) {}

	async review(work: unknown): Promise<VerificationReport[]> {
		const reports = await Promise.all(
			this.critics.map((critic) => critic.review(work)),
		);
		return reports.map((report, index) => ({
			...report,
			reviewer: report.reviewer ?? this.critics[index].name,
		}));
	}
}
