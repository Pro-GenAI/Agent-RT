/** Deny-overrides permission rules with path-level authorization. */
import {
	type ToolDefinition,
	type ToolSideEffect,
} from '../tools/definitions.js';

export type PermissionEffect = 'allow' | 'deny';

export interface PermissionRequest {
	operation: string;
	agent?: string;
	tool?: string;
	path?: string;
	network?: string;
	data?: string;
	sideEffect?: ToolSideEffect;
}

export interface PermissionRule {
	effect: PermissionEffect;
	operations?: string[];
	agents?: string[];
	tools?: string[];
	paths?: string[];
	networks?: string[];
	data?: string[];
	sideEffects?: ToolSideEffect[];
}

export interface PermissionDecision {
	allowed: boolean;
	reason: string;
	matchedRule?: PermissionRule;
}

export function normalizePermissionPath(path: string): string {
	const parts: string[] = [];
	for (const part of path
		.trim()
		.replace(/\\/g, '/')
		.replace(/^\/+/, '')
		.split('/')) {
		if (!part || part === '.') continue;
		if (part === '..') {
			if (parts.length === 0)
				throw new Error('path escapes permission root');
			parts.pop();
		} else {
			parts.push(part);
		}
	}
	return parts.join('/');
}

const globRegexCache = new Map<string, RegExp>();

/**
 * Translate a shell-style glob into a RegExp with the same semantics as
 * Python's `fnmatch.fnmatchcase`: `*` and `?` match any character including
 * newlines and `/`, `[seq]` / `[!seq]` are character classes, and an
 * unterminated `[` is a literal. Keeping the two runtimes identical matters
 * because these patterns back deny rules.
 */
function globToRegExp(pattern: string): RegExp {
	const cached = globRegexCache.get(pattern);
	if (cached) return cached;
	let source = '';
	let index = 0;
	while (index < pattern.length) {
		const character = pattern[index];
		index += 1;
		if (character === '*') {
			source += '[\\s\\S]*';
		} else if (character === '?') {
			source += '[\\s\\S]';
		} else if (character === '[') {
			let end = index;
			if (pattern[end] === '!') end += 1;
			if (pattern[end] === ']') end += 1;
			while (end < pattern.length && pattern[end] !== ']') end += 1;
			if (end >= pattern.length) {
				source += '\\[';
			} else {
				let body = pattern
					.slice(index, end)
					.replace(/\\/g, '\\\\')
					.replace(/\]/g, '\\]');
				index = end + 1;
				if (body[0] === '!') body = '^' + body.slice(1);
				else if (body[0] === '^') body = '\\' + body;
				source += '[' + body + ']';
			}
		} else {
			source += character.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
		}
	}
	let regex: RegExp;
	try {
		regex = new RegExp('^' + source + '$');
	} catch {
		// A malformed class (for example a reversed range) must not throw
		// from inside a permission check; fall back to literal brackets.
		regex = new RegExp(
			'^' +
				pattern
					.split('*')
					.map((chunk) =>
						chunk
							.split('?')
							.map((part) =>
								part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
							)
							.join('[\\s\\S]'),
					)
					.join('[\\s\\S]*') +
				'$',
		);
	}
	if (globRegexCache.size >= 1024) globRegexCache.clear();
	globRegexCache.set(pattern, regex);
	return regex;
}

export function permissionGlobMatches(
	value: string | undefined,
	patterns: string[] | undefined,
): boolean {
	if (!patterns || patterns.length === 0) return true;
	if (value === undefined) return false;
	return patterns.some((pattern) => globToRegExp(pattern).test(value));
}

function permissionRuleMatches(
	rule: PermissionRule,
	request: PermissionRequest,
): boolean {
	const normalizedPath =
		request.path === undefined
			? undefined
			: normalizePermissionPath(request.path);
	const paths = rule.paths?.map((pattern) =>
		/[*?[]/.test(pattern)
			? pattern.trim().replace(/\\/g, '/').replace(/^\/+/, '')
			: normalizePermissionPath(pattern),
	);
	return (
		permissionGlobMatches(request.operation, rule.operations) &&
		permissionGlobMatches(request.agent, rule.agents) &&
		permissionGlobMatches(request.tool, rule.tools) &&
		permissionGlobMatches(normalizedPath, paths) &&
		permissionGlobMatches(request.network, rule.networks) &&
		permissionGlobMatches(request.data, rule.data) &&
		(!rule.sideEffects ||
			rule.sideEffects.length === 0 ||
			(request.sideEffect !== undefined &&
				rule.sideEffects.includes(request.sideEffect)))
	);
}

export class PermissionDeniedError extends Error {
	constructor(
		readonly request: PermissionRequest,
		readonly decision: PermissionDecision,
	) {
		super(decision.reason);
		this.name = 'PermissionDeniedError';
	}
}

export class PermissionEngine {
	readonly rules: PermissionRule[];

	constructor(
		rules: PermissionRule[] = [],
		readonly defaultEffect: PermissionEffect = 'deny',
	) {
		if (defaultEffect !== 'allow' && defaultEffect !== 'deny') {
			throw new Error('default permission effect must be allow or deny');
		}
		this.rules = [...rules];
	}

	evaluate(request: PermissionRequest): PermissionDecision {
		const matching = this.rules.filter((rule) =>
			permissionRuleMatches(rule, request),
		);
		const deny = matching.find((rule) => rule.effect === 'deny');
		if (deny)
			return {
				allowed: false,
				reason: 'permission denied by matching rule',
				matchedRule: deny,
			};
		const allow = matching.find((rule) => rule.effect === 'allow');
		if (allow)
			return {
				allowed: true,
				reason: 'permission allowed by matching rule',
				matchedRule: allow,
			};
		const allowed = this.defaultEffect === 'allow';
		return {
			allowed,
			reason: `permission ${allowed ? 'allowed' : 'denied'} by default policy`,
		};
	}

	check(request: PermissionRequest): PermissionDecision {
		const decision = this.evaluate(request);
		if (!decision.allowed)
			throw new PermissionDeniedError(request, decision);
		return decision;
	}

	checkTool(
		tool: string,
		definition: ToolDefinition,
		agent?: string,
	): PermissionDecision {
		return this.check({
			operation: 'execute',
			agent,
			tool,
			sideEffect: definition.sideEffect ?? 'none',
		});
	}

	checkPath(
		path: string,
		operation: 'read' | 'write' | 'execute',
		agent?: string,
	): PermissionDecision {
		return this.check({
			operation,
			agent,
			path: normalizePermissionPath(path),
		});
	}
}
