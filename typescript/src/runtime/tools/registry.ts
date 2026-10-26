/** The tool registry: registration, authorization, guardrails, and execution. */
import {
	type ActionBlockerLike,
	evaluateActionBlocker,
} from '../../ext/action_blockers.js';
import {
	type RegistrationSafetyGuard,
	type RegistrationSafetySubject,
	enforceRegistrationSafety,
	type AsyncRegistrationSafetyGuard,
	enforceRegistrationSafetyAsync,
} from '../../ext/registration_safety.js';
import { type ToolCall } from '../types.js';
import {
	type ContextualToolHandler,
	type ToolDefinition,
	type ToolHandler,
	effectiveToolExecutionLimits,
	validateToolDefinition,
} from './definitions.js';
import { type ToolLifecycleHooks } from '../observability/audit.js';
import { type ApprovalManager } from '../authorization/approvals.js';
import { type DryRunResult } from '../resilience/transactions.js';
import { type RateLimiter } from '../resilience/rate_limits.js';
import { type PermissionEngine } from '../authorization/permissions.js';
import {
	ActionBlockedError,
	applyGuardrailResult,
} from '../guardrails/results.js';
import {
	type ModelToolInputGuardrailOptions,
	type ToolInputGuardrail,
	type ToolOutputGuardrail,
	makeDefaultToolInputGuardrail,
} from '../guardrails/tool_guardrails.js';
import { type CapabilityDescriptor } from './capabilities.js';
import {
	ToolArgumentValidationError,
	validateToolArguments,
} from './validation.js';
import { ToolTimeoutError } from './programs.js';

export interface DeferredToolRegistration {
	name: string;
	namespace?: string;
	enabled: boolean;
	description?: string;
	metadata?: Record<string, unknown>;
	loader: () => ToolDefinition;
	handler?: ToolHandler;
	contextualHandler?: ContextualToolHandler;
}

export interface RegisteredTool {
	definition: ToolDefinition;
	namespace?: string;
	enabled: boolean;
	handler?: ToolHandler;
	contextualHandler?: ContextualToolHandler;
}

export class ToolRegistry {
	private readonly tools = new Map<string, RegisteredTool>();
	private readonly deferredTools = new Map<
		string,
		DeferredToolRegistration
	>();
	private revision = 0;
	private readonly modelToolInputGuardrail?: ToolInputGuardrail;

	constructor(
		readonly hooks: ToolLifecycleHooks = {},
		readonly services: Record<string, unknown> = {},
		readonly permissionEngine?: PermissionEngine,
		readonly toolInputGuardrails: ToolInputGuardrail[] = [],
		readonly toolOutputGuardrails: ToolOutputGuardrail[] = [],
		readonly approvalManager?: ApprovalManager,
		readonly rateLimiter?: RateLimiter,
		modelToolInputGuardrail: ModelToolInputGuardrailOptions = {},
		readonly registrationGuard?: RegistrationSafetyGuard,
		readonly actionBlockers: ActionBlockerLike[] = [],
	) {
		if (modelToolInputGuardrail.enabled) {
			this.modelToolInputGuardrail = makeDefaultToolInputGuardrail(
				modelToolInputGuardrail.model,
				modelToolInputGuardrail.classify,
			);
		}
	}

	get version(): number {
		return this.revision;
	}

	static qualifiedName(name: string, namespace?: string): string {
		if (!name.trim()) throw new Error('tool name must not be empty');
		if (namespace !== undefined && !namespace.trim()) {
			throw new Error('tool namespace must not be empty');
		}
		return namespace ? `${namespace}.${name}` : name;
	}

	private static registrationSubject(
		definition: ToolDefinition,
		qualifiedName: string,
	): RegistrationSafetySubject {
		return {
			kind: 'tool',
			name: qualifiedName,
			description: definition.description,
			content: {
				input_schema: JSON.stringify(definition.inputSchema),
				output_schema: JSON.stringify(definition.outputSchema ?? {}),
				metadata: JSON.stringify(definition.metadata ?? {}),
			},
		};
	}

	register(
		definition: ToolDefinition,
		options: {
			namespace?: string;
			enabled?: boolean;
			replace?: boolean;
			handler?: ToolHandler;
			contextualHandler?: ContextualToolHandler;
		} = {},
	): RegisteredTool {
		validateToolDefinition(definition);
		const name = ToolRegistry.qualifiedName(
			definition.name,
			options.namespace,
		);
		if (options.handler && options.contextualHandler) {
			throw new Error(
				'register either handler or contextualHandler, not both',
			);
		}
		enforceRegistrationSafety(
			ToolRegistry.registrationSubject(definition, name),
			this.registrationGuard,
		);
		if (
			(this.tools.has(name) || this.deferredTools.has(name)) &&
			!options.replace
		) {
			throw new Error(`tool already registered: ${name}`);
		}
		this.deferredTools.delete(name);
		const registered: RegisteredTool = {
			definition,
			namespace: options.namespace,
			enabled: options.enabled ?? true,
			handler: options.handler,
			contextualHandler: options.contextualHandler,
		};
		this.tools.set(name, registered);
		this.revision += 1;
		return registered;
	}

	async registerChecked(
		definition: ToolDefinition,
		decisionGuard: AsyncRegistrationSafetyGuard,
		options: {
			namespace?: string;
			enabled?: boolean;
			replace?: boolean;
			handler?: ToolHandler;
			contextualHandler?: ContextualToolHandler;
		} = {},
	): Promise<RegisteredTool> {
		validateToolDefinition(definition);
		const name = ToolRegistry.qualifiedName(
			definition.name,
			options.namespace,
		);
		await enforceRegistrationSafetyAsync(
			ToolRegistry.registrationSubject(definition, name),
			decisionGuard,
		);
		return this.register(definition, options);
	}

	registerDeferred(
		name: string,
		loader: () => ToolDefinition,
		options: {
			namespace?: string;
			enabled?: boolean;
			replace?: boolean;
			description?: string;
			metadata?: Record<string, unknown>;
			handler?: ToolHandler;
			contextualHandler?: ContextualToolHandler;
		} = {},
	): DeferredToolRegistration {
		const qualified = ToolRegistry.qualifiedName(name, options.namespace);
		if (options.handler && options.contextualHandler) {
			throw new Error(
				'register either handler or contextualHandler, not both',
			);
		}
		enforceRegistrationSafety(
			{
				kind: 'tool',
				name: qualified,
				description: options.description ?? '',
				content: {
					metadata: JSON.stringify(options.metadata ?? {}),
				},
			},
			this.registrationGuard,
		);
		if (
			(this.tools.has(qualified) || this.deferredTools.has(qualified)) &&
			!options.replace
		) {
			throw new Error(`tool already registered: ${qualified}`);
		}
		this.tools.delete(qualified);
		const deferred: DeferredToolRegistration = {
			name,
			namespace: options.namespace,
			enabled: options.enabled ?? true,
			description: options.description,
			metadata: options.metadata,
			loader,
			handler: options.handler,
			contextualHandler: options.contextualHandler,
		};
		this.deferredTools.set(qualified, deferred);
		this.revision += 1;
		return deferred;
	}

	deferredNames(options: { includeDisabled?: boolean } = {}): string[] {
		return [...this.deferredTools.entries()]
			.filter(([, tool]) => options.includeDisabled || tool.enabled)
			.map(([name]) => name);
	}

	load(name: string): RegisteredTool {
		const existing = this.tools.get(name);
		if (existing) return existing;

		const deferred = this.deferredTools.get(name);
		if (!deferred) throw new Error(`deferred tool not registered: ${name}`);
		if (!deferred.enabled) {
			throw new Error(`deferred tool is disabled: ${name}`);
		}

		const definition = deferred.loader();
		if (definition.name !== deferred.name) {
			throw new Error(
				`deferred tool loader for ${name} returned mismatched name ${definition.name}`,
			);
		}
		// register() removes the deferred entry only after its safety scan
		// passes, so a rejected definition leaves the registration intact.
		return this.register(definition, {
			namespace: deferred.namespace,
			enabled: true,
			replace: true,
			handler: deferred.handler,
			contextualHandler: deferred.contextualHandler,
		});
	}

	get(name: string): RegisteredTool {
		const tool = this.tools.get(name);
		if (!tool) throw new Error(`tool not registered: ${name}`);
		return tool;
	}

	enable(name: string): void {
		const tool = this.get(name);
		this.tools.set(name, { ...tool, enabled: true });
		this.revision += 1;
	}

	disable(name: string): void {
		const tool = this.get(name);
		this.tools.set(name, { ...tool, enabled: false });
		this.revision += 1;
	}

	unregister(name: string): RegisteredTool {
		const tool = this.get(name);
		this.tools.delete(name);
		this.revision += 1;
		return tool;
	}

	list(options: { includeDisabled?: boolean } = {}): RegisteredTool[] {
		return [...this.tools.values()].filter(
			(tool) => options.includeDisabled || tool.enabled,
		);
	}

	definitions(): ToolDefinition[] {
		return this.list().map((tool) => {
			if (!tool.namespace) return tool.definition;
			return {
				...tool.definition,
				name: ToolRegistry.qualifiedName(
					tool.definition.name,
					tool.namespace,
				),
			};
		});
	}

	capabilityDescriptors(
		options: {
			includeDisabled?: boolean;
			includeDeferred?: boolean;
		} = {},
	): CapabilityDescriptor[] {
		const capabilities: CapabilityDescriptor[] = this.list({
			includeDisabled: options.includeDisabled,
		}).map((tool) => ({
			id: `tool:${ToolRegistry.qualifiedName(
				tool.definition.name,
				tool.namespace,
			)}`,
			kind: 'tool',
			name: ToolRegistry.qualifiedName(
				tool.definition.name,
				tool.namespace,
			),
			description: tool.definition.description,
			namespace: tool.namespace,
			metadata: tool.definition.metadata,
		}));

		if (options.includeDeferred !== false) {
			for (const [name, tool] of this.deferredTools) {
				if (!options.includeDisabled && !tool.enabled) continue;
				capabilities.push({
					id: `tool:${name}`,
					kind: 'tool',
					name,
					description: tool.description,
					namespace: tool.namespace,
					metadata: tool.metadata,
				});
			}
		}
		return capabilities;
	}

	namespaces(options: { includeDeferred?: boolean } = {}): string[] {
		const values = new Set<string>();
		for (const tool of this.tools.values()) {
			if (tool.namespace) values.add(tool.namespace);
		}
		if (options.includeDeferred !== false) {
			for (const tool of this.deferredTools.values()) {
				if (tool.namespace) values.add(tool.namespace);
			}
		}
		return [...values].sort();
	}

	definitionsInNamespace(namespace: string): ToolDefinition[] {
		if (!namespace.trim())
			throw new Error('tool namespace must not be empty');
		return this.list()
			.filter((tool) => tool.namespace === namespace)
			.map((tool) => ({
				...tool.definition,
				name: ToolRegistry.qualifiedName(
					tool.definition.name,
					tool.namespace,
				),
			}));
	}

	async execute(
		call: ToolCall,
		signal?: AbortSignal,
		requestContext: Record<string, unknown> = {},
	): Promise<unknown> {
		const registered = this.get(call.name);
		if (!registered.enabled) {
			throw new Error(`tool is disabled: ${call.name}`);
		}

		if (this.permissionEngine) {
			const agent =
				typeof requestContext.agent === 'string'
					? requestContext.agent
					: undefined;
			this.permissionEngine.checkTool(
				call.name,
				registered.definition,
				agent,
			);
		}

		let effectiveCall = call;
		if (this.hooks.preCall) {
			const transformed = await this.hooks.preCall(
				call,
				registered.definition,
			);
			if (transformed) {
				if (
					transformed.id !== call.id ||
					transformed.name !== call.name
				) {
					throw new Error(
						'pre-call hook cannot change tool call identity or name',
					);
				}
				effectiveCall = transformed;
			}
		}

		for (const blocker of this.actionBlockers) {
			const decision = await evaluateActionBlocker(
				blocker,
				effectiveCall,
				registered.definition,
				requestContext,
			);
			if (decision.blocked) {
				throw new ActionBlockedError(
					decision.reason ?? 'action blocker rejected tool call',
					[...(decision.classifications ?? [])],
				);
			}
		}

		const inputGuardrails = this.modelToolInputGuardrail
			? [this.modelToolInputGuardrail, ...this.toolInputGuardrails]
			: this.toolInputGuardrails;
		for (const guardrail of inputGuardrails) {
			const guarded = applyGuardrailResult(
				effectiveCall,
				await guardrail(effectiveCall, registered.definition),
			);
			if (guarded.id !== call.id || guarded.name !== call.name) {
				throw new Error(
					'tool input guardrail cannot change tool call identity or name',
				);
			}
			effectiveCall = guarded;
		}

		if (effectiveCall.argumentError !== undefined) {
			throw new ToolArgumentValidationError(call.name, [
				effectiveCall.argumentError,
			]);
		}
		validateToolArguments(registered.definition, effectiveCall.arguments);
		if (this.rateLimiter) {
			const keys = ['tool:' + effectiveCall.name];
			if (
				typeof requestContext.userId === 'string' &&
				requestContext.userId
			) {
				keys.push('user:' + requestContext.userId);
			}
			if (
				typeof requestContext.tenantId === 'string' &&
				requestContext.tenantId
			) {
				keys.push('tenant:' + requestContext.tenantId);
			}
			this.rateLimiter.checkMany(keys);
		}
		if (
			requestContext.dryRun === true &&
			!['none', 'read'].includes(
				registered.definition.sideEffect ?? 'none',
			)
		) {
			return {
				tool: effectiveCall.name,
				arguments: { ...effectiveCall.arguments },
				sideEffect: registered.definition.sideEffect ?? 'none',
				wouldExecute: true,
			} satisfies DryRunResult;
		}
		if (this.approvalManager) {
			const sessionId =
				typeof requestContext.sessionId === 'string'
					? requestContext.sessionId
					: undefined;
			this.approvalManager.check(
				effectiveCall,
				registered.definition,
				sessionId,
			);
		}
		if (!registered.handler && !registered.contextualHandler) {
			throw new Error(`tool has no registered handler: ${call.name}`);
		}

		if (this.hooks.audit) {
			await this.hooks.audit({
				phase: 'start',
				call: effectiveCall,
				definition: registered.definition,
				requestContext,
			});
		}

		const limits = effectiveToolExecutionLimits(registered.definition);
		let value: unknown;
		try {
			if (limits.timeoutMs !== undefined && limits.timeoutMs <= 0) {
				throw new ToolTimeoutError(call.name, limits.timeoutMs);
			}
			const controller =
				limits.timeoutMs === undefined
					? undefined
					: new AbortController();
			let parentAbort: (() => void) | undefined;
			if (controller && signal) {
				parentAbort = () => controller.abort(signal.reason);
				signal.addEventListener('abort', parentAbort, { once: true });
			}
			const effectiveSignal = controller?.signal ?? signal;
			const operation = registered.contextualHandler
				? registered.contextualHandler(effectiveCall.arguments, {
						services: this.services,
						requestContext,
						signal: effectiveSignal,
						limits,
					})
				: registered.handler!(effectiveCall.arguments, effectiveSignal);
			if (limits.timeoutMs === undefined) {
				value = await operation;
			} else {
				let timer: ReturnType<typeof setTimeout> | undefined;
				try {
					value = await Promise.race([
						operation,
						new Promise<never>((_, reject) => {
							timer = setTimeout(() => {
								const error = new ToolTimeoutError(
									call.name,
									limits.timeoutMs!,
								);
								controller!.abort(error);
								reject(error);
							}, limits.timeoutMs);
						}),
					]);
				} finally {
					if (timer !== undefined) clearTimeout(timer);
					if (signal && parentAbort)
						signal.removeEventListener('abort', parentAbort);
				}
			}
		} catch (error) {
			if (this.hooks.onError) {
				await this.hooks.onError(
					effectiveCall,
					registered.definition,
					error,
				);
			}
			if (this.hooks.audit) {
				await this.hooks.audit({
					phase: 'error',
					call: effectiveCall,
					definition: registered.definition,
					error,
					requestContext,
				});
			}
			if (signal?.aborted) throw error;
			if (registered.definition.errorBehavior === 'return_error') {
				return {
					error: {
						type: 'tool_execution_error',
						tool: call.name,
						message:
							error instanceof Error
								? error.message
								: String(error),
					},
				};
			}
			throw error;
		}

		try {
			if (this.hooks.postCall) {
				value = await this.hooks.postCall(
					effectiveCall,
					registered.definition,
					value,
				);
			}
			for (const guardrail of this.toolOutputGuardrails) {
				value = applyGuardrailResult(
					value,
					await guardrail(
						value,
						effectiveCall,
						registered.definition,
						requestContext,
					),
				);
			}
		} catch (error) {
			// The handler already ran; record a terminal audit event so the
			// trail never ends at "start" for a call with side effects.
			if (this.hooks.onError) {
				await this.hooks.onError(
					effectiveCall,
					registered.definition,
					error,
				);
			}
			if (this.hooks.audit) {
				await this.hooks.audit({
					phase: 'error',
					call: effectiveCall,
					definition: registered.definition,
					error,
					requestContext,
				});
			}
			throw error;
		}
		if (this.hooks.audit) {
			await this.hooks.audit({
				phase: 'success',
				call: effectiveCall,
				definition: registered.definition,
				value,
				requestContext,
			});
		}
		return value;
	}
}
