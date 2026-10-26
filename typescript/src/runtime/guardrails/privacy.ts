/** PII morphing and privacy redaction. */
import { parseEnvBoolean } from '../observability/openllmetry.js';
import { nodeModule, runtimeEnvironment } from '../platform.js';

export const PII_MORPHER_DISABLE_ENV = 'AGENT_RT_DISABLE_PII_MORPHER';

const PII_SHORT_HASH_CHARS_PER_DIGEST = 10;

export interface PIIEntity {
	kind: string;
	start: number;
	end: number;
}

const PII_PATTERN_SPECS: ReadonlyArray<readonly [string, string]> = [
	[
		'email',
		String.raw`(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_\`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![A-Za-z0-9._%+-])`,
	],
	['ssn', String.raw`(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)`],
	['credit_card', String.raw`(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)`],
	[
		'ipv4',
		String.raw`(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)`,
	],
	[
		'phone',
		String.raw`(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)`,
	],
];

const PII_HINT_SUFFIXES: ReadonlyArray<readonly [string, readonly string[]]> = [
	['email', ['email', 'emailaddress']],
	['phone', ['phone', 'phonenumber', 'mobile', 'mobilenumber']],
	['ssn', ['ssn', 'socialsecuritynumber']],
	['credit_card', ['creditcard', 'cardnumber']],
	['ipv4', ['ipaddress', 'ipv4']],
	['ipv6', ['ipv6']],
	['first_name', ['firstname']],
	['last_name', ['lastname']],
	['person', ['fullname', 'personname']],
	['address', ['streetaddress', 'postaladdress', 'mailingaddress']],
	['date_of_birth', ['dateofbirth', 'birthdate', 'dob']],
];

const PII_FIRST_NAMES = [
	'Avery',
	'Blake',
	'Casey',
	'Dakota',
	'Emerson',
	'Finley',
	'Harper',
	'Jordan',
	'Kai',
	'Logan',
	'Morgan',
	'Parker',
	'Quinn',
	'Riley',
	'Rowan',
	'Taylor',
] as const;

const PII_LAST_NAMES = [
	'Bennett',
	'Brooks',
	'Carter',
	'Ellis',
	'Hayes',
	'Jordan',
	'Lane',
	'Morgan',
	'Parker',
	'Reed',
	'Rivera',
	'Sawyer',
	'Taylor',
	'Walker',
	'Ward',
	'Young',
] as const;

function normalizePIIKey(key: unknown): string {
	return String(key)
		.toLowerCase()
		.replace(/[^a-z0-9]/g, '');
}

function inferPIIKind(key: unknown): string | undefined {
	const normalized = normalizePIIKey(key);
	for (const [kind, suffixes] of PII_HINT_SUFFIXES) {
		if (suffixes.some((suffix) => normalized.endsWith(suffix))) return kind;
	}
	return undefined;
}

function looksLikeCreditCard(value: string): boolean {
	const digits = [...value].filter((char) => /\d/.test(char)).join('');
	if (digits.length < 13 || digits.length > 19) return false;
	let total = 0;
	const parity = digits.length % 2;
	for (let index = 0; index < digits.length; index += 1) {
		let digit = Number(digits[index]);
		if (index % 2 === parity) {
			digit *= 2;
			if (digit > 9) digit -= 9;
		}
		total += digit;
	}
	return total % 10 === 0;
}

function piiDigests(value: string): { sha256: string; md5: string } {
	const crypto = nodeModule('node:crypto') as {
		createHash: (algorithm: string) => {
			update: (
				data: string,
				encoding?: string,
			) => {
				digest: (encoding: 'hex') => string;
			};
		};
	};
	return {
		sha256: crypto.createHash('sha256').update(value, 'utf8').digest('hex'),
		// MD5 is used only as a user-requested deterministic fingerprint component,
		// never as a security boundary or password/credential primitive.
		md5: crypto.createHash('md5').update(value, 'utf8').digest('hex'),
	};
}

export interface PIIMorpherOptions {
	enabled?: boolean;
	env?: Record<string, string | undefined>;
}

export class PIIMorpher {
	readonly enabled: boolean;
	private readonly replacements = new Map<string, string>();

	constructor(options: PIIMorpherOptions = {}) {
		const env = options.env ?? runtimeEnvironment();
		let disabled: boolean | undefined;
		try {
			disabled = parseEnvBoolean(env[PII_MORPHER_DISABLE_ENV]);
		} catch {
			throw new Error(
				PII_MORPHER_DISABLE_ENV + ' must be a boolean value',
			);
		}
		this.enabled = (options.enabled ?? true) && disabled !== true;
	}

	static shortHash(value: string): string {
		const digests = piiDigests(value);
		const width = PII_SHORT_HASH_CHARS_PER_DIGEST;
		return digests.sha256.slice(0, width) + digests.md5.slice(0, width);
	}

	get mapping(): Readonly<Record<string, string>> {
		return Object.fromEntries(this.replacements);
	}

	private static seedParts(shortHash: string): [number, number, number] {
		return [
			Number.parseInt(shortHash.slice(0, 8), 16),
			Number.parseInt(shortHash.slice(8, 16), 16),
			Number.parseInt(shortHash.slice(12, 20), 16),
		];
	}

	private fakeValue(kind: string, shortHash: string): string {
		const [first, second, third] = PIIMorpher.seedParts(shortHash);
		if (kind === 'email') {
			const firstName =
				PII_FIRST_NAMES[first % PII_FIRST_NAMES.length].toLowerCase();
			const lastName =
				PII_LAST_NAMES[second % PII_LAST_NAMES.length].toLowerCase();
			return `${firstName}.${lastName}.${String(third % 10_000).padStart(4, '0')}@example.test`;
		}
		if (kind === 'phone') {
			return `+1-000-${String(first % 1_000).padStart(3, '0')}-${String(second % 10_000).padStart(4, '0')}`;
		}
		if (kind === 'ssn') {
			return `000-${String(first % 100).padStart(2, '0')}-${String(second % 10_000).padStart(4, '0')}`;
		}
		if (kind === 'credit_card') {
			return `0000 0000 ${String(first % 10_000).padStart(4, '0')} ${String(second % 10_000).padStart(4, '0')}`;
		}
		if (kind === 'ipv4') {
			const networks = ['192.0.2', '198.51.100', '203.0.113'] as const;
			return `${networks[first % networks.length]}.${1 + (second % 254)}`;
		}
		if (kind === 'ipv6') {
			const groups = [0, 4, 8, 12].map((index) =>
				shortHash.slice(index, index + 4),
			);
			return '2001:db8:' + groups.join(':') + '::1';
		}
		if (kind === 'first_name') {
			return PII_FIRST_NAMES[first % PII_FIRST_NAMES.length];
		}
		if (kind === 'last_name') {
			return PII_LAST_NAMES[second % PII_LAST_NAMES.length];
		}
		if (kind === 'person') {
			const firstName = PII_FIRST_NAMES[first % PII_FIRST_NAMES.length];
			const lastName = PII_LAST_NAMES[second % PII_LAST_NAMES.length];
			return firstName + ' ' + lastName;
		}
		if (kind === 'address') {
			return `${100 + (first % 9_900)} Example Avenue, Testville, ZZ 00000`;
		}
		if (kind === 'date_of_birth') {
			const year = 1970 + (first % 31);
			const month = 1 + (second % 12);
			const day = 1 + (third % 28);
			return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
		}
		const normalizedKind = kind
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, '-')
			.replace(/^-|-$/g, '');
		const label = normalizedKind || 'pii';
		return `fake-${label}-${String(first % 1_000_000).padStart(6, '0')}`;
	}

	morphPII(original: string, kind: string): string {
		if (!this.enabled || !original) return original;
		const key = PIIMorpher.shortHash(original);
		const existing = this.replacements.get(key);
		if (existing !== undefined) return existing;
		const fake = this.fakeValue(kind, key);
		this.replacements.set(key, fake);
		return fake;
	}

	private detectedEntities(text: string): PIIEntity[] {
		const entities: PIIEntity[] = [];
		for (const [kind, source] of PII_PATTERN_SPECS) {
			const pattern = new RegExp(source, 'g');
			for (const match of text.matchAll(pattern)) {
				const original = match[0];
				if (kind === 'credit_card' && !looksLikeCreditCard(original))
					continue;
				const start = match.index ?? 0;
				entities.push({ kind, start, end: start + original.length });
			}
		}
		return entities;
	}

	private static selectEntities(
		text: string,
		explicit: readonly PIIEntity[],
		detected: readonly PIIEntity[],
	): PIIEntity[] {
		const candidates: Array<{
			start: number;
			priority: number;
			negativeLength: number;
			entity: PIIEntity;
		}> = [];
		for (const [priority, entities] of [
			[0, explicit],
			[1, detected],
		] as const) {
			for (const entity of entities) {
				if (
					!entity.kind.trim() ||
					!Number.isInteger(entity.start) ||
					!Number.isInteger(entity.end) ||
					entity.start < 0 ||
					entity.end <= entity.start ||
					entity.end > text.length
				) {
					throw new Error(
						'PII entity spans must be valid and non-empty',
					);
				}
				candidates.push({
					start: entity.start,
					priority,
					negativeLength: -(entity.end - entity.start),
					entity,
				});
			}
		}
		candidates.sort(
			(left, right) =>
				left.start - right.start ||
				left.priority - right.priority ||
				left.negativeLength - right.negativeLength,
		);
		const selected: PIIEntity[] = [];
		let occupiedUntil = -1;
		for (const candidate of candidates) {
			if (candidate.entity.start < occupiedUntil) continue;
			selected.push(candidate.entity);
			occupiedUntil = candidate.entity.end;
		}
		return selected;
	}

	morphText(
		text: string,
		options: { kind?: string; entities?: readonly PIIEntity[] } = {},
	): string {
		if (!this.enabled || !text) return text;
		if (options.kind !== undefined) {
			const leading = text.length - text.trimStart().length;
			const trailing = text.length - text.trimEnd().length;
			const end = trailing ? text.length - trailing : text.length;
			const core = text.slice(leading, end);
			if (!core) return text;
			return (
				text.slice(0, leading) +
				this.morphPII(core, options.kind) +
				text.slice(end)
			);
		}
		const selected = PIIMorpher.selectEntities(
			text,
			options.entities ?? [],
			this.detectedEntities(text),
		);
		let result = text;
		for (let index = selected.length - 1; index >= 0; index -= 1) {
			const entity = selected[index];
			const original = text.slice(entity.start, entity.end);
			const replacement = this.morphPII(original, entity.kind);
			result =
				result.slice(0, entity.start) +
				replacement +
				result.slice(entity.end);
		}
		return result;
	}

	morph(value: unknown, keyHint?: unknown): unknown {
		if (!this.enabled) return value;
		if (Array.isArray(value)) return value.map((item) => this.morph(item));
		if (value && typeof value === 'object') {
			const output: Record<string, unknown> = {};
			for (const [key, item] of Object.entries(value)) {
				output[key] = this.morph(item, key);
			}
			return output;
		}
		if (typeof value === 'string') {
			return this.morphText(value, { kind: inferPIIKind(keyHint) });
		}
		return value;
	}
}

export interface PrivacyRedactionPolicy {
	sensitiveKeys?: string[];
	replacement?: string;
	textPatterns?: RegExp[];
}

/** Lowercase and drop punctuation so `apiKey`, `x-api-key`, `api_key` agree. */
function normalizeRedactionKey(key: string): string {
	return key.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export class PrivacyRedactor {
	readonly sensitiveKeys: Set<string>;
	private readonly normalizedSensitiveKeys: string[];
	readonly replacement: string;
	readonly textPatterns: RegExp[];
	readonly piiMorpher: PIIMorpher;

	constructor(
		policy: PrivacyRedactionPolicy = {},
		piiMorpher: PIIMorpher = new PIIMorpher(),
	) {
		this.sensitiveKeys = new Set(
			(
				policy.sensitiveKeys ?? [
					'password',
					'secret',
					'token',
					'authorization',
					'passwd',
					'api_key',
					'access_token',
					'refresh_token',
					'private_key',
					'credential',
					'credentials',
					'cookie',
				]
			).map((key) => key.toLowerCase()),
		);
		// Keys match after normalisation and as suffixes: `client_secret` and
		// `refresh_token` are covered while `token_count` is not.
		this.normalizedSensitiveKeys = [...this.sensitiveKeys]
			.map(normalizeRedactionKey)
			.filter(Boolean);
		this.replacement = policy.replacement ?? '[REDACTED]';
		this.textPatterns = [...(policy.textPatterns ?? [])];
		this.piiMorpher = piiMorpher;
	}

	private redactValue(value: unknown, keyHint?: unknown): unknown {
		if (Array.isArray(value))
			return value.map((item) => this.redactValue(item));
		if (value && typeof value === 'object') {
			const output: Record<string, unknown> = {};
			for (const [key, item] of Object.entries(value)) {
				const normalized = normalizeRedactionKey(key);
				output[key] = this.normalizedSensitiveKeys.some((item) =>
					normalized.endsWith(item),
				)
					? this.replacement
					: this.redactValue(item, key);
			}
			return output;
		}
		if (typeof value === 'string') {
			const redacted = this.textPatterns.reduce(
				(result, pattern) => result.replace(pattern, this.replacement),
				value,
			);
			return this.piiMorpher.morphText(redacted, {
				kind: inferPIIKind(keyHint),
			});
		}
		return value;
	}

	redact(value: unknown): unknown {
		return this.redactValue(value);
	}
}
