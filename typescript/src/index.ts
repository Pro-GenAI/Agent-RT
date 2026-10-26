/**
 * Agent RT public entrypoint.
 *
 * The runtime lives in themed modules under `src/runtime/` (see
 * docs/DEVELOPMENT.md); this barrel only re-exports their public names.
 */

export {
	ConfigError,
	ConfigNotFoundError,
	MissingSecret,
	configFromObject,
	configSearchPaths,
	discoverConfigPath,
	loadConfig,
	parseConfig,
	serverOptions as configServerOptions,
	type AgentRTConfig,
	type ModelDeployment,
	type VectorStoreConfig,
} from './ext/config.js';

export {
	MemoryBackend,
	loadMemory,
	type LoadMemoryOptions,
} from './ext/memory_backends.js';

export {
	DEFAULT_BLOCKED_COMMANDS,
	DEFAULT_COMMAND_ARGUMENT_NAMES,
	DEFAULT_DECISION_ACTION_BLOCKER_INSTRUCTIONS,
	RuleBasedActionBlocker,
	makeAgentActionGuardActionBlocker,
	makeDecisionActionBlocker,
	makeRuleBasedActionBlocker,
	type ActionBlocker,
	type ActionBlockerCall,
	type ActionBlockerDecision,
	type ActionBlockerFunction,
	type ActionBlockerLike,
	type ActionBlockerResult,
	type ActionBlockerToolDefinition,
	type DecisionActionBlockerProvider,
} from './ext/action_blockers.js';

export {
	RegistrationSafetyError,
	enforceRegistrationSafety,
	enforceRegistrationSafetyAsync,
	scanRegistrationSafety,
	type AsyncRegistrationSafetyGuard,
	type RegistrationKind,
	type RegistrationSafetyFinding,
	type RegistrationSafetyGuard,
	type RegistrationSafetyReport,
	type RegistrationSafetySubject,
} from './ext/registration_safety.js';

export {
	type OpenLLMetryConfig,
	openLLMetryConfigFromEnv,
	initializeOpenLLMetryFromEnv,
} from './runtime/observability/openllmetry.js';

export {
	runtimeSettings,
	configureRuntime,
	hello,
} from './runtime/settings.js';

export {
	type JSONScalar,
	type JSONValue,
	type WorkflowStateValue,
	WorkflowState,
	type SessionRef,
	type MemoryScope,
	type MessageRole,
	type ContentPartType,
	type ContentPart,
	type ToolCall,
	type ModelMessage,
} from './runtime/types.js';

export {
	type SessionSnapshot,
	ShortTermSessionMemory,
	TenantSessionMemory,
} from './runtime/memory/sessions.js';

export {
	type MemoryKind,
	type MemoryRecord,
	type MemorySearchQuery,
	type MemorySearchResult,
	type LongTermMemoryStore,
	InMemoryLongTermMemoryStore,
} from './runtime/memory/long_term.js';

export {
	type EmbeddingProvider,
	SemanticMemory,
	type Episode,
	EpisodicMemory,
} from './runtime/memory/semantic.js';

export {
	type MemoryMigration,
	type MemoryLifecyclePolicy,
	LifecycleMemoryStore,
	type Procedure,
	ProceduralMemory,
} from './runtime/memory/lifecycle.js';

export {
	type MemoryWriteCandidate,
	type MemoryWriteDecision,
	MemoryWritePolicy,
	MemoryRetrievalPolicy,
	ScopedMemoryStore,
} from './runtime/memory/policies.js';

export {
	type ToolSideEffect,
	type ToolErrorBehavior,
	type ToolExecutionMode,
	type ToolHandler,
	EXECUTION_TIMEOUT_ENV,
	EXECUTION_MEMORY_ENV,
	EXECUTION_CPU_ENV,
	type ToolExecutionLimits,
	executionLimitsFromEnv,
	type ToolExecutionContext,
	type ContextualToolHandler,
	type ToolDefinition,
	validateToolDefinition,
	type ToolSelectionPolicy,
	type ToolSelectionRequirement,
} from './runtime/tools/definitions.js';

export {
	type AuditAction,
	type AuditOutcome,
	type AuditRecord,
	type AuditTrail,
	InMemoryAuditTrail,
	type ToolAuditPhase,
	type ToolAuditEvent,
	type ToolLifecycleHooks,
	makeAuditTrailHook,
} from './runtime/observability/audit.js';

export {
	PII_MORPHER_DISABLE_ENV,
	type PIIEntity,
	type PIIMorpherOptions,
	PIIMorpher,
	type PrivacyRedactionPolicy,
	PrivacyRedactor,
} from './runtime/guardrails/privacy.js';

export {
	type ApprovalDecision,
	type ApprovalScope,
	type ApprovalRequest,
	type ApprovalGrant,
	ApprovalRequiredError,
	ApprovalDeniedError,
	type ApprovalStore,
	InMemoryApprovalStore,
	ApprovalManager,
} from './runtime/authorization/approvals.js';

export {
	type DryRunResult,
	type TransactionStep,
	type TransactionResult,
	SideEffectTransaction,
} from './runtime/resilience/transactions.js';

export {
	type FailureKind,
	type FailureDisposition,
	classifyFailure,
	type RetryPolicy,
	RetryExecutor,
	type RecoveryAction,
	type RecoveryContext,
	RecoveryRouter,
} from './runtime/resilience/retry.js';

export {
	type CircuitState,
	CircuitOpenError,
	CircuitBreaker,
	type RateLimit,
	RateLimitExceededError,
	RateLimiter,
} from './runtime/resilience/rate_limits.js';

export {
	type PermissionEffect,
	type PermissionRequest,
	type PermissionRule,
	type PermissionDecision,
	PermissionDeniedError,
	PermissionEngine,
} from './runtime/authorization/permissions.js';

export {
	type IdentityKind,
	type AuthenticationMethod,
	type Principal,
	type CredentialReference,
	type AuthenticationContext,
	type AuthorizationRequirement,
	type AuthorizationDecision,
	AuthorizationEngine,
} from './runtime/authorization/identity.js';

export {
	type CapabilityGrantOptions,
	CapabilityGrant,
	TenantContext,
} from './runtime/authorization/grants.js';

export {
	type SecretMetadata,
	SecretValue,
	type SecretStore,
	InMemorySecretStore,
	ScopedSecretStore,
} from './runtime/authorization/secrets.js';

export {
	type GuardrailAction,
	type GuardrailResult,
	GuardrailViolationError,
	ActionBlockedError,
} from './runtime/guardrails/results.js';

export {
	type ToolInputGuardrail,
	type ToolOutputGuardrail,
	DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
	type AgentActionGuardClassifier,
	makeAgentActionGuardToolInputGuardrail,
	makeDefaultToolInputGuardrail,
	type ModelToolInputGuardrailOptions,
} from './runtime/guardrails/tool_guardrails.js';

export {
	type PolicyEffect,
	type PolicyRequest,
	type PolicyRule,
	type PolicyDecision,
	PolicyEngine,
} from './runtime/authorization/policy.js';

export {
	type DataClassification,
	type DataEgressRequest,
	DataExfiltrationPolicy,
	makeToolInputExfiltrationGuardrail,
	makeToolOutputExfiltrationGuardrail,
} from './runtime/guardrails/exfiltration.js';

export {
	type CapabilityKind,
	type CapabilityDescriptor,
	type CapabilitySearchResult,
	type CapabilityScorer,
	LexicalCapabilityScorer,
	CapabilityCatalog,
} from './runtime/tools/capabilities.js';

export {
	type DeferredToolRegistration,
	type RegisteredTool,
	ToolRegistry,
} from './runtime/tools/registry.js';

export {
	TODO_STATUSES,
	type TodoStatus,
	TODO_MAX_ITEMS,
	TODO_MAX_CONTENT_CHARS,
	type TodoItem,
	TodoList,
	marshalToolResult,
} from './runtime/tools/todos.js';

export {
	type TrustLevel,
	type ContextItemKind,
	type ContextItem,
	PromptInjectionDefense,
	type ContextSelectionPolicy,
	type ContextCompactionPolicy,
	type ContextCompactor,
	DeterministicContextCompactor,
	type ArtifactReference,
	type ArtifactStore,
	InMemoryArtifactStore,
	type ContextOffloadPolicy,
	type PromptCacheHint,
	type PromptCachePolicy,
	type ContextAssembly,
} from './runtime/context/items.js';

export {
	type InputGuardrail,
	type OutputGuardrail,
	type BoundaryGuardrailPolicy,
	DEFAULT_BOUNDARY_GUARDRAIL_POLICY,
	makeDefaultInputGuardrail,
	makeDefaultOutputGuardrail,
} from './runtime/guardrails/boundary.js';

export {
	type ModelUsage,
	type EmbeddingRequest,
	type EmbeddingItem,
	type EmbeddingResponse,
	type EmbeddingModelProvider,
	type TokenCountingModelProvider,
	type ModelCatalogEntry,
	type ModelCatalogProvider,
	type BatchJob,
	type BatchProvider,
	type FinishReason,
	type StructuredOutputRequirement,
	JSON_OBJECT_OUTPUT,
	isJsonObjectOutput,
	type ReasoningConfig,
	type ModelRequest,
	type ModelResponse,
	type ModelProvider,
	type StreamEventType,
	type ModelStreamEvent,
	type StreamingModelProvider,
} from './runtime/models/interfaces.js';

export {
	DEFAULT_OPENAI_BASE_URL,
	OPENAI_BASE_URL_ENV,
	OPENAI_API_KEY_ENV,
	OPENAI_MODEL_ENV,
	DEFAULT_ANTHROPIC_BASE_URL,
	ANTHROPIC_BASE_URL_ENV,
	ANTHROPIC_API_KEY_ENV,
	ANTHROPIC_MODEL_ENV,
	type ModelListClientFactory,
} from './runtime/models/sdk.js';

export {
	type OpenAIProviderSettings,
	type ResolvedOpenAIProviderSettings,
	type AnthropicProviderSettings,
	type ResolvedAnthropicProviderSettings,
	resolveOpenAIProviderSettings,
	resolveAnthropicProviderSettings,
	validateOpenAIProviderSettings,
	validateAnthropicProviderSettings,
	resolveOpenAIModel,
	resolveAnthropicModel,
} from './runtime/models/provider_settings.js';

export {
	MAX_RATE_LIMIT_BLOCK_MS,
	OpenAIRateLimitGate,
} from './runtime/models/openai_http.js';

export { OpenAIModelProvider } from './runtime/models/openai.js';

export { AnthropicModelProvider } from './runtime/models/anthropic.js';

export {
	ConfiguredModelProvider,
	loadModel,
} from './runtime/models/configured.js';

export {
	type StreamEventHandler,
	type ModelSettings,
	type AgentOutputRequirements,
	type AgentConfig,
	type CompiledExecutionPlan,
	type ToolExecutor,
} from './runtime/agent/config.js';

export {
	type MemorySourceOptions,
	MemorySource,
	resolveAgentMemory,
} from './runtime/memory/sources.js';

export {
	type PlanStepStatus,
	type PlanStep,
	type Plan,
	type PlanProgress,
	PlanTracker,
	type VerificationIssue,
	type VerificationReport,
	PlanVerifier,
	type Planner,
	type StepExecutor,
	type PlanExecutionResult,
	PlannerExecutor,
} from './runtime/agent/planning.js';

export {
	type ReviewFinding,
	type ReviewResult,
	ReflectionPass,
	type CriticAgent,
	CriticPanel,
} from './runtime/agent/reflection.js';

export {
	type AgentInvocation,
	type AgentInvoker,
	type AgentTool,
	agentAsTool,
	type HandoffRequest,
	type HandoffResult,
	HandoffManager,
	type SubagentSpec,
	IsolatedSubagentRunner,
	type WorkerAssignment,
	type WorkerResult,
	SupervisorWorkerTeam,
} from './runtime/agent/delegation.js';

export {
	type AgentRoute,
	AgentRouter,
	type TeamMember,
	type TeamTurn,
	RoundRobinTeam,
	type SpeakerSelector,
	ModelSelectedSpeakerTeam,
	type SwarmDecision,
	type SwarmHandoffPolicy,
	SwarmTeam,
	type TeamNode,
	AgentTeamNode,
	HierarchicalTeam,
	ParallelSubagentExecutor,
	type MapReduceResult,
	MapReduceOrchestrator,
	type SpeculativeBranch,
	type SpeculativeBranchResult,
	type SpeculativeResult,
	SpeculativeOrchestrator,
} from './runtime/agent/teams.js';

export {
	type MCPCapabilityKind,
	type MCPTransport,
	type MCPServerInfo,
	type MCPTool,
	type MCPResource,
	type MCPPrompt,
	MCPClient,
} from './runtime/protocols/mcp.js';

export {
	type MCPCapabilityFilterOptions,
	MCPCapabilityFilter,
	type MCPAccessPolicyOptions,
	MCPAccessPolicy,
	AuthorizedMCPClient,
} from './runtime/protocols/mcp_access.js';

export {
	type ProtocolKind,
	type ProtocolAdapter,
	ProtocolAdapterRegistry,
	type ConnectorDefinition,
	ConnectorRegistry,
} from './runtime/protocols/adapters.js';

export {
	type RemoteAgentCard,
	type RemoteAgentDirectory,
	RemoteAgentRegistry,
	type RemoteMessage,
	type RemoteTaskStatus,
	type RemoteTask,
	RemoteTaskStore,
	type RemoteArtifact,
	RemoteArtifactStore,
	type RemoteEvent,
	type RemoteAgentTransport,
	CapabilityNegotiation,
	RemoteAgentClient,
} from './runtime/protocols/remote.js';

export {
	type BrowserAction,
	type BrowserState,
	type BrowserBackend,
	BrowserSession,
	type ComputerAction,
	type ComputerState,
	type ComputerBackend,
	ComputerSession,
} from './runtime/interfaces/browser.js';

export {
	type RetrievalKind,
	type RetrievalQuery,
	type RetrievalResult,
	type RetrievalProvider,
	type RetrievalReranker,
	RetrievalRegistry,
} from './runtime/retrieval/registry.js';

export {
	AGENT_RT_VECTOR_DB_ENV,
	AGENT_RT_VECTOR_DB_COLLECTION_ENV,
	AGENT_RT_VECTOR_DB_URL_ENV,
	AGENT_RT_VECTOR_DB_OPTION_PREFIX,
	type VectorDBConfig,
	type VectorDBProviderFactory,
	VectorDBProviderRegistry,
	vectorDBConfigFromEnvironment,
	vectorDBProviderFromEnvironment,
	AGENT_RT_VECTOR_DB_API_KEY_ENV,
	AGENT_RT_VECTOR_DB_TIMEOUT_ENV,
	POPULAR_VECTOR_DB_BACKENDS,
	EnvironmentVectorDBProvider,
} from './runtime/retrieval/vector_db.js';

export {
	registerPopularVectorDBBackends,
	vectorStoreRegistryFromConfig,
	resolveVectorStores,
} from './runtime/retrieval/vector_stores.js';

export {
	FILE_SEARCH_TOOL_NAME,
	FILE_SEARCH_MAX_RESULTS,
	registerFileSearchTool,
} from './runtime/retrieval/file_search.js';

export {
	AGENT_RT_WEB_SEARCH_TOOL_ENV,
	AGENT_RT_WEB_SEARCH_API_KEY_ENV,
	AGENT_RT_WEB_SEARCH_TOKEN_ENV,
	AGENT_RT_WEB_SEARCH_TIMEOUT_ENV,
	EnvironmentWebSearchProvider,
} from './runtime/retrieval/web_search.js';

export {
	type MultimodalMessage,
	multimodalToModelMessage,
	type RealtimeEventType,
	type RealtimeEvent,
	type RealtimeTransport,
	RealtimeSession,
} from './runtime/interfaces/realtime.js';

export {
	type ProgressStatus,
	type ProgressEvent,
	ProgressReporter,
	type RuntimeRequest,
	type RuntimeResponse,
	type RuntimeExecutor,
	CLIInterface,
	type APIOperation,
	type APIRequest,
	APIInterface,
	type IDEDiagnostic,
	type IDEContext,
	type IDEProgress,
	IDEIntegration,
	type ChatEnvelope,
	type ChatReply,
	type ChatAdapter,
	ChatSessionBridge,
	type ApprovalPresentation,
	approvalPresentation,
} from './runtime/interfaces/surfaces.js';

export {
	type SpanKind,
	type SpanStatus,
	type TraceSpan,
	TraceRecorder,
	type LogSeverity,
	type StructuredLogRecord,
	StructuredLogger,
} from './runtime/observability/tracing.js';

export {
	type MetricKind,
	type MetricPoint,
	RuntimeMetrics,
} from './runtime/observability/metrics.js';

export {
	type TokenUsageRecord,
	TokenLedger,
	type CostCategory,
	type CostRecord,
	CostLedger,
} from './runtime/observability/usage.js';

export {
	type ReplayExchange,
	type ReplayBundle,
	DebugReplayStore,
	DeterministicReplay,
	ReplayModelProvider,
} from './runtime/observability/replay.js';

export { ContextAssembler } from './runtime/context/assembler.js';

export {
	type ToolFilterContext,
	type ToolVisibilityFilter,
	type ToolSafetyFilter,
	TOOL_SEARCH_NAME,
	type ToolbaseMCPClient,
	Toolbase,
} from './runtime/tools/toolbase.js';

export {
	type ModelTarget,
	type ModelDescriptor,
	type RoutingRequirements,
	type ModelRoutingPolicy,
	FirstMatchRoutingPolicy,
	ModelRegistry,
} from './runtime/models/routing.js';

export {
	StructuredOutputValidationError,
	ToolArgumentValidationError,
	validateToolArguments,
} from './runtime/tools/validation.js';

export {
	type TerminationReason,
	BudgetExceededError,
	type ExecutionBudgetLimits,
	ExecutionBudget,
	Deadline,
	LoopDetector,
	type Finalizer,
	runFinalizers,
} from './runtime/resilience/budgets.js';

export {
	type AgentRunLimits,
	SECURE_RUN_LIMITS,
	secureToolRegistry,
	type AgentRunResult,
} from './runtime/agent/limits.js';

export {
	type AgentCheckpoint,
	type CheckpointStore,
	InMemoryCheckpointStore,
	checkpointFromResult,
} from './runtime/checkpoints/store.js';

export {
	type DurableEventType,
	type DurableEvent,
	type EventStore,
	InMemoryEventStore,
	type EventRetentionPolicy,
	RetainedEventStore,
	TenantEventStore,
} from './runtime/checkpoints/events.js';

export {
	type IdempotencyRecord,
	type IdempotencyStore,
	InMemoryIdempotencyStore,
} from './runtime/resilience/idempotency.js';

export {
	type TaskStatus,
	type TaskLifecycleState,
	TaskLifecycle,
	type BackgroundTaskRunner,
	type BackgroundTaskSnapshot,
	BackgroundTaskManager,
} from './runtime/tasks/background.js';

export {
	type WorkQueueItem,
	InMemoryWorkQueue,
} from './runtime/tasks/queue.js';

export {
	type ScheduledTask,
	InMemoryScheduler,
	type ExternalEvent,
	type EventTriggerRule,
	EventTriggerDispatcher,
} from './runtime/tasks/scheduler.js';

export {
	type FileStat,
	type FileSystem,
	InMemoryFileSystem,
	WorkspaceFiles,
	type FileSystemFactory,
	FileSystemBackendRegistry,
} from './runtime/workspace/filesystem.js';

export {
	type WorkspaceRecord,
	PersistentWorkspaceStore,
	TenantWorkspaceStore,
} from './runtime/workspace/stores.js';

export {
	type ArtifactKind,
	type ArtifactStatus,
	type ProvenanceRecord,
	type Artifact,
	type ArtifactVersion,
	type ArtifactRepository,
	InMemoryArtifactRepository,
} from './runtime/workspace/artifacts.js';

export {
	type SandboxResourceLimits,
	sandboxResourceLimitsFromEnv,
	type SandboxNetworkMode,
	SANDBOX_NETWORK_MODE_ENV,
	SANDBOX_ALLOWED_DOMAINS_ENV,
	SANDBOX_BLOCKED_DOMAINS_ENV,
	SANDBOX_ALLOW_HTTP_ENV,
	SANDBOX_ALLOW_WEBSOCKET_ENV,
	SANDBOX_ALLOW_IP_ENV,
	SANDBOX_ALLOW_PROXY_ENV,
	SANDBOX_NETWORK_BPS_ENV,
	SANDBOX_NETWORK_BYTES_ENV,
	type SandboxNetworkPolicy,
	sandboxNetworkPolicyFromEnv,
	sandboxNetworkAllows,
} from './runtime/sandbox/policy.js';

export {
	type SandboxCommand,
	type SandboxCommandResult,
	type SandboxSnapshot,
	type SandboxBackend,
	type SandboxCommandRunner,
	CallbackSandboxBackend,
	type SandboxBackendName,
	SANDBOX_BACKEND_ENV,
} from './runtime/sandbox/backends.js';

export {
	DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES,
	NativeSandboxBackend,
} from './runtime/sandbox/native.js';

export { DockerSandboxBackend } from './runtime/sandbox/docker.js';

export { E2BSandboxBackend } from './runtime/sandbox/e2b.js';

export { MicrosandboxBackend } from './runtime/sandbox/microsandbox.js';

export { SWEReXSandboxBackend } from './runtime/sandbox/swe_rex.js';

export {
	sandboxBackendFromEnv,
	type SandboxPackageManager,
	type SandboxPackageInstaller,
	CallbackSandboxPackageManager,
	type CodeInterpreterRunner,
	CodeInterpreterRegistry,
	SandboxSession,
	sandboxShellTool,
} from './runtime/sandbox/session.js';

export {
	ToolTimeoutError,
	type ToolProgramResult,
	ToolProgramExecutor,
} from './runtime/tools/programs.js';

export { AgentLoop } from './runtime/agent/loop.js';
