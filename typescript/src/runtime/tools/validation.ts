/** Structured-output and tool-argument validation against JSON schemas. */
import { isProviderState } from '../../internal/media.js';
import { type ModelMessage, type ToolCall } from '../types.js';
import { type ToolDefinition } from './definitions.js';
import { type AgentOutputRequirements } from '../agent/config.js';

export class StructuredOutputValidationError extends Error {
	readonly issues: string[];
	readonly rawOutput: unknown;

	constructor(issues: string[], rawOutput?: unknown) {
		super(`structured output validation failed: ${issues.join('; ')}`);
		this.name = 'StructuredOutputValidationError';
		this.issues = [...issues];
		this.rawOutput = rawOutput;
	}
}

function messageText(message: ModelMessage): string {
	return message.content
		.filter((part) => part.type === 'text')
		.map((part) => part.text ?? '')
		.join('');
}

function extractStructuredOutput(message: ModelMessage): unknown {
	for (const part of message.content) {
		if (
			part.type === 'json' &&
			part.data !== undefined &&
			!isProviderState(part)
		) {
			return part.data;
		}
	}
	const text = messageText(message).trim();
	if (!text) {
		throw new StructuredOutputValidationError(
			['response did not contain JSON'],
			text,
		);
	}
	try {
		return JSON.parse(text);
	} catch (error) {
		const detail = error instanceof Error ? error.message : 'invalid JSON';
		throw new StructuredOutputValidationError(
			[`invalid JSON: ${detail}`],
			text,
		);
	}
}

/** Own-property check; `in` would also match Object.prototype members. */
function hasOwnKey(target: object, key: string): boolean {
	return Object.prototype.hasOwnProperty.call(target, key);
}

function matchesSchemaType(value: unknown, expected: string): boolean {
	if (expected === 'object')
		return (
			typeof value === 'object' && value !== null && !Array.isArray(value)
		);
	if (expected === 'array') return Array.isArray(value);
	if (expected === 'string') return typeof value === 'string';
	if (expected === 'boolean') return typeof value === 'boolean';
	if (expected === 'null') return value === null;
	if (expected === 'integer')
		return typeof value === 'number' && Number.isInteger(value);
	if (expected === 'number')
		return typeof value === 'number' && Number.isFinite(value);
	return true;
}

/** JSON equality: containers compare recursively, `true` is not `1`. */
function jsonEqual(left: unknown, right: unknown): boolean {
	if (Array.isArray(left) || Array.isArray(right)) {
		return (
			Array.isArray(left) &&
			Array.isArray(right) &&
			left.length === right.length &&
			left.every((item, index) => jsonEqual(item, right[index]))
		);
	}
	if (
		typeof left === 'object' &&
		left !== null &&
		typeof right === 'object' &&
		right !== null
	) {
		const leftKeys = Object.keys(left);
		const rightKeys = Object.keys(right);
		return (
			leftKeys.length === rightKeys.length &&
			leftKeys.every(
				(key) =>
					hasOwnKey(right, key) &&
					jsonEqual(
						(left as Record<string, unknown>)[key],
						(right as Record<string, unknown>)[key],
					),
			)
		);
	}
	return (
		Object.is(left, right) || (left === right && typeof left !== 'object')
	);
}

const MAX_SCHEMA_DEPTH = 32;

function isSchemaObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveSchemaRef(
	root: Record<string, unknown>,
	reference: string,
): Record<string, unknown> {
	if (!reference.startsWith('#')) {
		throw new Error(
			`unsupported $ref (only local '#/...' refs): ${reference}`,
		);
	}
	let node: unknown = root;
	const tokens =
		reference === '#' ? [] : reference.slice(1).split('/').slice(1);
	for (const token of tokens) {
		const key = token.replace(/~1/g, '/').replace(/~0/g, '~');
		if (isSchemaObject(node) && hasOwnKey(node, key)) {
			node = node[key];
		} else {
			throw new Error(`unresolvable $ref: ${reference}`);
		}
	}
	if (!isSchemaObject(node)) {
		throw new Error(`$ref does not point to a schema: ${reference}`);
	}
	return node;
}

/**
 * Validate against a practical JSON Schema subset: `type` (string or list),
 * `enum`, `const`, `properties`, `required`, `additionalProperties` (bool or
 * schema), `items`, numeric bounds, `minLength`/`maxLength`/`pattern`,
 * `minItems`/`maxItems`/`uniqueItems`, `minProperties`/`maxProperties`,
 * `allOf`/`anyOf`/`oneOf`/`not`, and local `$ref`. Unknown keywords are ignored.
 */
function validateStructuredValue(
	value: unknown,
	schema: Record<string, unknown>,
	path = '$',
	root: Record<string, unknown> = schema,
	depth = 0,
): string[] {
	if (depth > MAX_SCHEMA_DEPTH)
		return [`${path}: schema nesting is too deep`];
	const issues: string[] = [];
	const sub = (
		childValue: unknown,
		childSchema: Record<string, unknown>,
		childPath: string,
	): string[] =>
		validateStructuredValue(
			childValue,
			childSchema,
			childPath,
			root,
			depth + 1,
		);

	if (typeof schema.$ref === 'string') {
		try {
			issues.push(
				...sub(value, resolveSchemaRef(root, schema.$ref), path),
			);
		} catch (error) {
			return [
				`${path}: ${error instanceof Error ? error.message : String(error)}`,
			];
		}
	}

	const expected = schema.type;
	const types =
		typeof expected === 'string'
			? [expected]
			: Array.isArray(expected)
				? expected.filter(
						(item): item is string => typeof item === 'string',
					)
				: [];
	if (
		types.length > 0 &&
		!types.some((item) => matchesSchemaType(value, item))
	) {
		return [`${path}: expected ${types.join(' or ')}`];
	}

	if ('const' in schema && !jsonEqual(value, schema.const)) {
		issues.push(`${path}: value does not equal the required constant`);
	}
	if (
		Array.isArray(schema.enum) &&
		!schema.enum.some((candidate) => jsonEqual(value, candidate))
	) {
		issues.push(`${path}: value is not in enum`);
	}

	const subschemas = (keyword: string): Record<string, unknown>[] => {
		const list = schema[keyword];
		return Array.isArray(list) ? list.filter(isSchemaObject) : [];
	};
	for (const child of subschemas('allOf'))
		issues.push(...sub(value, child, path));
	const anyOf = subschemas('anyOf');
	if (
		anyOf.length > 0 &&
		!anyOf.some((child) => sub(value, child, path).length === 0)
	) {
		issues.push(`${path}: value does not match any allowed schema`);
	}
	const oneOf = subschemas('oneOf');
	if (
		oneOf.length > 0 &&
		oneOf.filter((child) => sub(value, child, path).length === 0).length !==
			1
	) {
		issues.push(`${path}: value must match exactly one allowed schema`);
	}
	if (
		isSchemaObject(schema.not) &&
		sub(value, schema.not, path).length === 0
	) {
		issues.push(`${path}: value matches a disallowed schema`);
	}

	if (typeof value === 'number') {
		const bounds: [string, (limit: number) => boolean][] = [
			['minimum', (limit) => value < limit],
			['maximum', (limit) => value > limit],
			['exclusiveMinimum', (limit) => value <= limit],
			['exclusiveMaximum', (limit) => value >= limit],
		];
		for (const [keyword, failed] of bounds) {
			const limit = schema[keyword];
			if (typeof limit === 'number' && failed(limit)) {
				issues.push(`${path}: violates ${keyword} ${limit}`);
			}
		}
	}

	if (typeof value === 'string') {
		if (
			typeof schema.minLength === 'number' &&
			value.length < schema.minLength
		) {
			issues.push(`${path}: shorter than minLength ${schema.minLength}`);
		}
		if (
			typeof schema.maxLength === 'number' &&
			value.length > schema.maxLength
		) {
			issues.push(`${path}: longer than maxLength ${schema.maxLength}`);
		}
		if (typeof schema.pattern === 'string') {
			try {
				if (!new RegExp(schema.pattern, 'u').test(value)) {
					issues.push(`${path}: does not match pattern`);
				}
			} catch {
				issues.push(`${path}: schema pattern is invalid`);
			}
		}
	}

	if (isSchemaObject(value)) {
		const required = schema.required;
		if (Array.isArray(required)) {
			for (const propertyName of required) {
				if (
					typeof propertyName === 'string' &&
					!hasOwnKey(value, propertyName)
				) {
					issues.push(
						`${path}.${propertyName}: required property is missing`,
					);
				}
			}
		}
		const properties = isSchemaObject(schema.properties)
			? schema.properties
			: {};
		for (const [key, childSchema] of Object.entries(properties)) {
			if (hasOwnKey(value, key) && isSchemaObject(childSchema)) {
				issues.push(...sub(value[key], childSchema, `${path}.${key}`));
			}
		}
		const extras = Object.keys(value).filter(
			(key) => !hasOwnKey(properties, key),
		);
		if (schema.additionalProperties === false) {
			for (const key of extras) {
				issues.push(
					`${path}.${key}: additional property is not allowed`,
				);
			}
		} else if (isSchemaObject(schema.additionalProperties)) {
			for (const key of extras) {
				issues.push(
					...sub(
						value[key],
						schema.additionalProperties,
						`${path}.${key}`,
					),
				);
			}
		}
		const count = Object.keys(value).length;
		if (
			typeof schema.minProperties === 'number' &&
			count < schema.minProperties
		) {
			issues.push(
				`${path}: fewer than minProperties ${schema.minProperties}`,
			);
		}
		if (
			typeof schema.maxProperties === 'number' &&
			count > schema.maxProperties
		) {
			issues.push(
				`${path}: more than maxProperties ${schema.maxProperties}`,
			);
		}
	}

	if (Array.isArray(value)) {
		if (isSchemaObject(schema.items)) {
			const itemSchema = schema.items;
			value.forEach((item, index) => {
				issues.push(...sub(item, itemSchema, `${path}[${index}]`));
			});
		}
		if (
			typeof schema.minItems === 'number' &&
			value.length < schema.minItems
		) {
			issues.push(`${path}: fewer than minItems ${schema.minItems}`);
		}
		if (
			typeof schema.maxItems === 'number' &&
			value.length > schema.maxItems
		) {
			issues.push(`${path}: more than maxItems ${schema.maxItems}`);
		}
		if (schema.uniqueItems === true) {
			const duplicate = value.findIndex((item, index) =>
				value
					.slice(0, index)
					.some((earlier) => jsonEqual(item, earlier)),
			);
			if (duplicate >= 0)
				issues.push(`${path}[${duplicate}]: duplicate item`);
		}
	}

	return issues;
}

export class ToolArgumentValidationError extends Error {
	readonly toolName: string;
	readonly issues: string[];

	constructor(toolName: string, issues: string[]) {
		super(`invalid arguments for tool ${toolName}: ${issues.join('; ')}`);
		this.name = 'ToolArgumentValidationError';
		this.toolName = toolName;
		this.issues = [...issues];
	}
}

export function validateToolArguments(
	definition: ToolDefinition,
	argumentsValue: Record<string, unknown>,
): void {
	const issues = validateStructuredValue(
		argumentsValue,
		definition.inputSchema,
	);
	if (issues.length > 0) {
		throw new ToolArgumentValidationError(definition.name, issues);
	}
}

export function toolValidationErrorPayload(
	call: ToolCall,
	issues: string[],
): Record<string, unknown> {
	return {
		error: {
			type: 'tool_argument_validation',
			tool: call.name,
			issues: [...issues],
		},
	};
}

export function validateStructuredMessage(
	message: ModelMessage,
	requirements: AgentOutputRequirements,
): unknown {
	const value = extractStructuredOutput(message);
	if (requirements.schema) {
		const issues = validateStructuredValue(value, requirements.schema);
		if (issues.length > 0)
			throw new StructuredOutputValidationError(issues, value);
	}
	return value;
}
