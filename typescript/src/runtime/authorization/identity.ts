/** Principals, credentials, authentication context, and authorization rules. */

export type IdentityKind = 'user' | 'service' | 'delegated';

export type AuthenticationMethod =
	'session' | 'oauth' | 'api_key' | 'service' | 'delegation';

export interface Principal {
	id: string;
	kind: IdentityKind;
	tenantId?: string;
	roles?: string[];
	scopes?: string[];
	attributes?: Record<string, string>;
	onBehalfOf?: string;
}

export interface CredentialReference {
	id: string;
	kind: AuthenticationMethod;
	expiresAt?: number;
	scopes?: string[];
}

export interface AuthenticationContext {
	principal: Principal;
	method: AuthenticationMethod;
	credential?: CredentialReference;
}

export interface AuthorizationRequirement {
	scopes?: string[];
	roles?: string[];
	attributes?: Record<string, string>;
	resourceOwnerId?: string;
	tenantId?: string;
}

export interface AuthorizationDecision {
	allowed: boolean;
	reason: string;
}

export class AuthorizationEngine {
	evaluate(
		authentication: AuthenticationContext,
		requirement: AuthorizationRequirement,
	): AuthorizationDecision {
		const expiresAt = authentication.credential?.expiresAt;
		if (expiresAt !== undefined && Date.now() >= expiresAt) {
			return {
				allowed: false,
				reason: 'authentication credential is expired',
			};
		}
		const principal = authentication.principal;
		const scopes = new Set([
			...(principal.scopes ?? []),
			...(authentication.credential?.scopes ?? []),
		]);
		if (
			requirement.tenantId !== undefined &&
			principal.tenantId !== requirement.tenantId
		) {
			return { allowed: false, reason: 'tenant does not match' };
		}
		if ((requirement.scopes ?? []).some((scope) => !scopes.has(scope))) {
			return { allowed: false, reason: 'required scopes are missing' };
		}
		const roles = new Set(principal.roles ?? []);
		if ((requirement.roles ?? []).some((role) => !roles.has(role))) {
			return { allowed: false, reason: 'required roles are missing' };
		}
		for (const [key, value] of Object.entries(
			requirement.attributes ?? {},
		)) {
			if (principal.attributes?.[key] !== value) {
				return {
					allowed: false,
					reason: `required attribute does not match: ${key}`,
				};
			}
		}
		if (
			requirement.resourceOwnerId !== undefined &&
			requirement.resourceOwnerId !== principal.id &&
			requirement.resourceOwnerId !== principal.onBehalfOf
		) {
			return {
				allowed: false,
				reason: 'resource ownership does not match',
			};
		}
		return {
			allowed: true,
			reason: 'authorization requirements satisfied',
		};
	}

	check(
		authentication: AuthenticationContext,
		requirement: AuthorizationRequirement,
	): AuthorizationDecision {
		const decision = this.evaluate(authentication, requirement);
		if (!decision.allowed) throw new Error(decision.reason);
		return decision;
	}
}
