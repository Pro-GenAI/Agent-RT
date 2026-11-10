/** Attribute-based policy rules and the policy engine. */
import { permissionGlobMatches } from './permissions.js';

export type PolicyEffect = 'allow' | 'deny';

export interface PolicyRequest {
	domain: string;
	action: string;
	subject?: string;
	resource?: string;
	attributes?: Record<string, string>;
}

export interface PolicyRule {
	effect: PolicyEffect;
	domains?: string[];
	actions?: string[];
	subjects?: string[];
	resources?: string[];
	attributes?: Record<string, string>;
}

export interface PolicyDecision {
	allowed: boolean;
	reason: string;
	matchedRule?: PolicyRule;
}

export function policyGlobMatches(
	value: string | undefined,
	patterns: string[] | undefined,
): boolean {
	return permissionGlobMatches(value, patterns);
}

export class PolicyEngine {
	constructor(
		readonly rules: PolicyRule[] = [],
		readonly defaultEffect: PolicyEffect = 'deny',
	) {
		if (defaultEffect !== 'allow' && defaultEffect !== 'deny') {
			throw new Error('default policy effect must be allow or deny');
		}
	}

	evaluate(request: PolicyRequest): PolicyDecision {
		const matching = this.rules.filter(
			(rule) =>
				policyGlobMatches(request.domain, rule.domains) &&
				policyGlobMatches(request.action, rule.actions) &&
				policyGlobMatches(request.subject, rule.subjects) &&
				policyGlobMatches(request.resource, rule.resources) &&
				Object.entries(rule.attributes ?? {}).every(
					([key, value]) => request.attributes?.[key] === value,
				),
		);
		const deny = matching.find((rule) => rule.effect === 'deny');
		if (deny)
			return {
				allowed: false,
				reason: 'policy denied by matching rule',
				matchedRule: deny,
			};
		const allow = matching.find((rule) => rule.effect === 'allow');
		if (allow)
			return {
				allowed: true,
				reason: 'policy allowed by matching rule',
				matchedRule: allow,
			};
		const allowed = this.defaultEffect === 'allow';
		return {
			allowed,
			reason: `policy ${allowed ? 'allowed' : 'denied'} by default`,
		};
	}

	check(request: PolicyRequest): PolicyDecision {
		const decision = this.evaluate(request);
		if (!decision.allowed) throw new Error(decision.reason);
		return decision;
	}
}
