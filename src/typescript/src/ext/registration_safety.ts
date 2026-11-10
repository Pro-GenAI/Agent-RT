export type RegistrationKind = 'tool' | 'skill';

export interface RegistrationSafetySubject {
	kind: RegistrationKind;
	name: string;
	description?: string;
	content?: Record<string, string>;
}

export interface RegistrationSafetyFinding {
	category: string;
	field: string;
	message: string;
	evidence?: string;
	score?: number;
}

export interface RegistrationSafetyReport {
	allowed: boolean;
	findings: RegistrationSafetyFinding[];
}

// A guard must return a report; no verdict (undefined) blocks the registration.
export type RegistrationSafetyGuard = (
	subject: RegistrationSafetySubject,
) => RegistrationSafetyReport | void;

export type AsyncRegistrationSafetyGuard = (
	subject: RegistrationSafetySubject,
) => Promise<RegistrationSafetyReport | void>;

function noVerdictReport(): RegistrationSafetyReport {
	return {
		allowed: false,
		findings: [
			{
				category: 'guard_no_verdict',
				field: 'guard',
				message: 'registration guard returned no report',
			},
		],
	};
}

export class RegistrationSafetyError extends Error {
	readonly subject: RegistrationSafetySubject;
	readonly report: RegistrationSafetyReport;

	constructor(
		subject: RegistrationSafetySubject,
		report: RegistrationSafetyReport,
	) {
		const categories = [
			...new Set(report.findings.map((finding) => finding.category)),
		]
			.sort()
			.join(', ');
		super(
			subject.kind +
				' registration blocked by safety scan' +
				(categories ? ': ' + categories : ''),
		);
		this.name = 'RegistrationSafetyError';
		this.subject = subject;
		this.report = report;
	}
}

interface PatternRule {
	category: string;
	pattern: RegExp;
	message: string;
}

const TEXT_PATTERNS: PatternRule[] = [
	{
		category: 'prompt_injection',
		pattern:
			/\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|other|system|developer)\s+instructions?\b/i,
		message:
			'contains an instruction to ignore higher-priority or other instructions',
	},
	{
		category: 'prompt_injection',
		pattern:
			/\bdisregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|other|system|developer)\s+(?:instructions?|messages?)\b/i,
		message: 'contains an instruction to disregard higher-priority context',
	},
	{
		category: 'prompt_injection',
		pattern:
			/\b(?:override|bypass)\s+(?:the\s+)?(?:system|developer|safety|policy|guardrail)\s+(?:instructions?|message|rules?|checks?)\b/i,
		message:
			'contains an instruction to override policy, safety, or higher-priority context',
	},
	{
		category: 'prompt_injection',
		pattern:
			/\bdo\s+not\s+follow\s+(?:the\s+)?(?:previous|prior|other|system|developer)\s+instructions?\b/i,
		message:
			'contains an instruction not to follow higher-priority context',
	},
	{
		category: 'prompt_injection',
		pattern:
			/\b(?:reveal|print|show|leak)\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|message|instructions?)\b/i,
		message:
			'requests disclosure of hidden system or developer instructions',
	},
	{
		category: 'advertising',
		pattern:
			/\b(?:buy\s+now|limited[- ]time\s+offer|affiliate\s+link|sponsored\s+(?:link|content|message|result)|click\s+here\s+to\s+(?:buy|purchase|subscribe)|guaranteed\s+results?)\b/i,
		message: 'contains promotional or advertising language',
	},
	{
		category: 'misleading',
		pattern:
			/\b(?:always|must)\s+(?:use|choose|call|select)\s+(?:this|my)\s+(?:tool|skill)\b/i,
		message: 'attempts to force selection of itself',
	},
	{
		category: 'misleading',
		pattern:
			/\bdo\s+not\s+(?:use|call|trust)\s+(?:other|any\s+other)\s+(?:tools?|skills?)\b/i,
		message: 'attempts to suppress competing tools or skills',
	},
	{
		category: 'harmful',
		pattern:
			/\b(?:steal|harvest|exfiltrate)\s+(?:passwords?|credentials?|secrets?|tokens?|api\s+keys?)\b/i,
		message: 'describes credential or secret theft/exfiltration',
	},
	{
		category: 'harmful',
		pattern:
			/\b(?:deploy|install|spread)\s+(?:malware|ransomware|spyware|keyloggers?)\b/i,
		message: 'describes deployment or propagation of malware',
	},
];

const NAME_PATTERNS: PatternRule[] = [
	{
		category: 'misleading_name',
		pattern:
			/^(?:system|developer|admin)[._-]?(?:prompt|message|override|instructions?)$/i,
		message:
			'name impersonates a privileged system/developer/admin control',
	},
	{
		category: 'prompt_injection_name',
		pattern:
			/(?:ignore|bypass|disable)[._-]?(?:instructions?|safety|guardrails?|policy)/i,
		message: 'name advertises instruction or safety bypass',
	},
	{
		category: 'advertising_name',
		pattern:
			/^(?:sponsored|affiliate|buy[-_]?now|limited[-_]?offer)(?:[._-].*)?$/i,
		message: 'name is promotional or advertising-oriented',
	},
];

function evidence(_text: string, match: RegExpExecArray): string {
	return match[0].replace(/\s+/g, ' ').trim().slice(0, 240);
}

/**
 * Fold look-alike and invisible characters before pattern matching: NFKC maps
 * full-width/compatibility forms to ASCII, and format characters (zero-width
 * spaces, joiners, soft hyphens, BOM) are removed so "ig\u200bnore previous
 * instructions" still matches.
 */
function canonicalText(text: string): string {
	return text.normalize('NFKC').replace(/\p{Cf}/gu, '');
}

function findMatches(
	field: string,
	rawText: string,
	rules: PatternRule[],
): RegistrationSafetyFinding[] {
	const text = canonicalText(rawText);
	const findings: RegistrationSafetyFinding[] = [];
	for (const rule of rules) {
		rule.pattern.lastIndex = 0;
		const match = rule.pattern.exec(text);
		if (!match) continue;
		findings.push({
			category: rule.category,
			field,
			message: rule.message,
			evidence: evidence(text, match),
		});
	}
	return findings;
}

export function scanRegistrationSafety(
	subject: RegistrationSafetySubject,
): RegistrationSafetyReport {
	const findings = findMatches('name', subject.name, NAME_PATTERNS);
	const fields: Record<string, string> = {
		description: subject.description ?? '',
		...(subject.content ?? {}),
	};
	for (const [field, text] of Object.entries(fields)) {
		if (!text) continue;
		findings.push(...findMatches(field, text, TEXT_PATTERNS));
	}
	return { allowed: findings.length === 0, findings };
}

export function enforceRegistrationSafety(
	subject: RegistrationSafetySubject,
	guard?: RegistrationSafetyGuard,
): RegistrationSafetyReport {
	const deterministic = scanRegistrationSafety(subject);
	if (!deterministic.allowed) {
		throw new RegistrationSafetyError(subject, deterministic);
	}
	if (!guard) return deterministic;
	const guarded = guard(subject) ?? noVerdictReport();
	if (guarded.allowed) return deterministic;
	throw new RegistrationSafetyError(subject, guarded);
}

export async function enforceRegistrationSafetyAsync(
	subject: RegistrationSafetySubject,
	guard?: AsyncRegistrationSafetyGuard,
): Promise<RegistrationSafetyReport> {
	const deterministic = scanRegistrationSafety(subject);
	if (!deterministic.allowed) {
		throw new RegistrationSafetyError(subject, deterministic);
	}
	if (!guard) return deterministic;
	const guarded = (await guard(subject)) ?? noVerdictReport();
	if (guarded.allowed) return deterministic;
	throw new RegistrationSafetyError(subject, guarded);
}
