"""Agent RT public entrypoint.

The runtime core lives in themed modules under ``ext.runtime`` (see
docs/DEVELOPMENT.md); this module only re-exports it. Lazily loaded
features resolve through ``__getattr__`` from ``ext.runtime.optional``.
"""

from __future__ import annotations

# Standard-library names stay importable here because ``__all__`` has always
# listed every public module global.
import copy
import fnmatch
import hashlib
import inspect
import json
import math
import os
import random
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ext.action_blockers import (
    DEFAULT_BLOCKED_COMMANDS,
    DEFAULT_COMMAND_ARGUMENT_NAMES,
    ActionBlocker,
    ActionBlockerDecision,
    ActionBlockerFunction,
    ActionBlockerLike,
    ActionBlockerResult,
    RuleBasedActionBlocker,
    make_agent_action_guard_action_blocker,
    make_decision_action_blocker,
    make_rule_based_action_blocker,
)
from ext.config import AgentRTConfig, ConfigError, ConfigNotFoundError, load_config
from ext.memory_backends import MemoryBackend, load_memory
from ext.registration_safety import (
    RegistrationSafetyGuard,
    RegistrationSafetySubject,
    enforce_registration_safety,
)
from ext.runtime.agent.config import (
    AgentConfig,
    AgentOutputRequirements,
    CompiledExecutionPlan,
    ModelSettings,
    StreamEventHandler,
    TokenUsageRecorder,
    ToolExecutor,
)
from ext.runtime.agent.limits import (
    SECURE_RUN_LIMITS,
    AgentRunLimits,
    AgentRunResult,
    secure_tool_registry,
)
from ext.runtime.agent.loop import AgentLoop
from ext.runtime.authorization.approvals import (
    ApprovalDecision,
    ApprovalDeniedError,
    ApprovalGrant,
    ApprovalManager,
    ApprovalRequest,
    ApprovalRequiredError,
    ApprovalScope,
    ApprovalStore,
    InMemoryApprovalStore,
)
from ext.runtime.authorization.grants import CapabilityGrant, TenantContext
from ext.runtime.authorization.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionEffect,
    PermissionEngine,
    PermissionRequest,
    PermissionRule,
)
from ext.runtime.checkpoints.events import (
    DurableEvent,
    DurableEventType,
    EventRetentionPolicy,
    EventStore,
    InMemoryEventStore,
    RetainedEventStore,
    TenantEventStore,
)
from ext.runtime.checkpoints.store import (
    AgentCheckpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
    checkpoint_from_result,
)
from ext.runtime.context.assembler import ContextAssembler
from ext.runtime.context.items import (
    ArtifactReference,
    ArtifactStore,
    ContextAssembly,
    ContextCompactionPolicy,
    ContextCompactor,
    ContextItem,
    ContextItemKind,
    ContextOffloadPolicy,
    ContextSelectionPolicy,
    DeterministicContextCompactor,
    InMemoryArtifactStore,
    PromptCacheHint,
    PromptCachePolicy,
    PromptInjectionDefense,
    TrustLevel,
)
from ext.runtime.guardrails.boundary import (
    BoundaryGuardrailPolicy,
    InputGuardrail,
    OutputGuardrail,
    make_default_input_guardrail,
    make_default_output_guardrail,
)
from ext.runtime.guardrails.privacy import (
    PII_MORPHER_DISABLE_ENV,
    PIIEntity,
    PIIMorpher,
    PrivacyRedactionPolicy,
    PrivacyRedactor,
)
from ext.runtime.guardrails.results import (
    ActionBlockedError,
    GuardrailAction,
    GuardrailResult,
    GuardrailViolationError,
    _apply_guardrail_result,
)
from ext.runtime.guardrails.tool_guardrails import (
    DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
    AgentActionGuardClassifier,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    make_agent_action_guard_tool_input_guardrail,
    make_default_tool_input_guardrail,
)
from ext.runtime.memory.sources import MemorySource, resolve_agent_memory
from ext.runtime.models.anthropic import AnthropicModelProvider
from ext.runtime.models.anthropic_payloads import (
    _anthropic_message,
    _anthropic_response,
)
from ext.runtime.models.configured import ConfiguredModelProvider, load_model
from ext.runtime.models.conversion import (
    PROVIDER_STATE_MIME,
    _data_url,
    _media_reference,
    _message_text,
    _provider_tool_call,
)
from ext.runtime.models.interfaces import (
    JSON_OBJECT_OUTPUT,
    EmbeddingItem,
    EmbeddingModelProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningConfig,
    StreamEventType,
    StreamingModelProvider,
    StructuredOutputRequirement,
    TokenCountingModelProvider,
)
from ext.runtime.models.openai import OpenAIModelProvider
from ext.runtime.models.openai_http import (
    MAX_RATE_LIMIT_BLOCK_SECONDS,
    _compatible_proxy_url,
    _OpenAICompatibleHTTPClient,
    _OpenAIRateLimitGate,
)
from ext.runtime.models.openai_payloads import (
    _openai_message,
    _openai_response,
    _OpenAICompatibleHTTPError,
)
from ext.runtime.models.provider_settings import (
    AnthropicProviderSettings,
    OpenAIProviderSettings,
)
from ext.runtime.models.routing import (
    FirstMatchRoutingPolicy,
    ModelDescriptor,
    ModelRegistry,
    ModelRoutingPolicy,
    ModelTarget,
    RoutingRequirements,
)
from ext.runtime.models.sdk import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_BASE_URL_ENV,
    ANTHROPIC_MODEL_ENV,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
    BatchJob,
    ModelCatalogEntry,
)
from ext.runtime.observability.audit import (
    AuditAction,
    AuditOutcome,
    AuditRecord,
    AuditTrail,
    InMemoryAuditTrail,
    ToolAuditEvent,
    ToolAuditHook,
    ToolAuditPhase,
    ToolErrorHook,
    ToolLifecycleHooks,
    ToolPostCallHook,
    ToolPreCallHook,
    make_audit_trail_hook,
)
from ext.runtime.observability.metrics import MetricPoint, RuntimeMetrics
from ext.runtime.observability.openllmetry import (
    OpenLLMetryConfig,
    initialize_openllmetry_from_env,
    openllmetry_config_from_env,
)
from ext.runtime.resilience.budgets import (
    BudgetExceededError,
    Deadline,
    ExecutionBudget,
    ExecutionBudgetLimits,
    Finalizer,
    LoopDetector,
    TerminationReason,
    run_finalizers,
)
from ext.runtime.resilience.idempotency import (
    IdempotencyRecord,
    IdempotencyStore,
    InMemoryIdempotencyStore,
)
from ext.runtime.resilience.rate_limits import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RateLimit,
    RateLimiter,
    RateLimitExceededError,
)
from ext.runtime.resilience.retry import (
    FailureDisposition,
    FailureKind,
    RecoveryAction,
    RecoveryContext,
    RecoveryRouter,
    RetryExecutor,
    RetryPolicy,
    classify_failure,
)
from ext.runtime.resilience.transactions import (
    DryRunResult,
    SideEffectTransaction,
    TransactionResult,
    TransactionStep,
)
from ext.runtime.settings import _RUNTIME_SETTINGS, configure_runtime, runtime_settings
from ext.runtime.tasks.background import (
    BackgroundTaskManager,
    BackgroundTaskRunner,
    BackgroundTaskSnapshot,
    TaskLifecycle,
    TaskLifecycleState,
    TaskStatus,
)
from ext.runtime.tasks.queue import InMemoryWorkQueue, WorkQueueItem
from ext.runtime.tasks.scheduler import (
    EventTriggerDispatcher,
    EventTriggerRule,
    ExternalEvent,
    InMemoryScheduler,
    ScheduledTask,
)
from ext.runtime.tools.capabilities import (
    CapabilityCatalog,
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityScorer,
    CapabilitySearchResult,
    LexicalCapabilityScorer,
)
from ext.runtime.tools.definitions import (
    EXECUTION_CPU_ENV,
    EXECUTION_MEMORY_ENV,
    EXECUTION_TIMEOUT_ENV,
    ContextualToolHandler,
    ToolDefinition,
    ToolErrorBehavior,
    ToolExecutionContext,
    ToolExecutionLimits,
    ToolExecutionMode,
    ToolHandler,
    ToolSelectionPolicy,
    ToolSelectionRequirement,
    ToolSideEffect,
    tool_execution_limits_from_env,
    validate_tool_definition,
)
from ext.runtime.tools.programs import (
    ToolProgramExecutor,
    ToolProgramResult,
    ToolTimeoutError,
)
from ext.runtime.tools.registry import (
    DeferredToolRegistration,
    RegisteredTool,
    ToolRegistry,
)
from ext.runtime.tools.todos import (
    TODO_MAX_CONTENT_CHARS,
    TODO_MAX_ITEMS,
    TODO_STATUSES,
    TodoList,
    marshal_tool_result,
)
from ext.runtime.tools.toolbase import (
    TOOL_SEARCH_NAME,
    Toolbase,
    ToolFilterContext,
    ToolSafetyFilter,
    ToolVisibilityFilter,
)
from ext.runtime.tools.validation import (
    StructuredOutputValidationError,
    ToolArgumentValidationError,
    _extract_structured_output,
    _validate_structured_value,
    validate_tool_arguments,
)
from ext.runtime.types import (
    CancellationToken,
    ContentPart,
    ContentPartType,
    FinishReason,
    JSONScalar,
    JSONValue,
    MemoryScope,
    MessageRole,
    ModelMessage,
    RunCancelled,
    SessionRef,
    ToolCall,
    WorkflowState,
    _validate_json_value,
    asyncio,
)
from ext.runtime.workspace.filesystem import (
    ArtifactStatus,
    FileStat,
    FileSystem,
    FileSystemBackendRegistry,
    FileSystemFactory,
    InMemoryFileSystem,
    WorkspaceFiles,
)

_LAZY_FEATURE_EXPORTS = frozenset(
    {
        "APIInterface",
        "APIOperationHandler",
        "APIRequest",
        "AgentInvocation",
        "AgentInvoker",
        "AgentRoute",
        "AgentRouter",
        "AgentTeamNode",
        "AgentTool",
        "AuthorizedMCPClient",
        "BrowserAction",
        "BrowserBackend",
        "BrowserSession",
        "BrowserState",
        "CLIInterface",
        "CapabilityNegotiation",
        "CapabilitySet",
        "ComputerAction",
        "ComputerBackend",
        "ComputerSession",
        "ComputerState",
        "ConnectorDefinition",
        "ConnectorRegistry",
        "CriticAgent",
        "CriticPanel",
        "HandoffManager",
        "HandoffRequest",
        "HandoffResult",
        "HierarchicalTeam",
        "IDEContext",
        "IDEDiagnostic",
        "IDEIntegration",
        "IDEProgress",
        "IsolatedSubagentRunner",
        "MCPAccessPolicy",
        "MCPCapabilityFilter",
        "MCPCapabilityKind",
        "MCPClient",
        "MCPPrompt",
        "MCPResource",
        "MCPServerInfo",
        "MCPTool",
        "MCPTransport",
        "MapReduceOrchestrator",
        "MapReduceResult",
        "ModelSelectedSpeakerTeam",
        "MultimodalMessage",
        "ParallelSubagentExecutor",
        "Plan",
        "PlanExecutionResult",
        "PlanProgress",
        "PlanStep",
        "PlanStepStatus",
        "PlanTracker",
        "PlanVerifier",
        "Planner",
        "PlannerExecutor",
        "ProgressEvent",
        "ProgressListener",
        "ProgressReporter",
        "ProgressStatus",
        "ProtocolAdapter",
        "ProtocolAdapterRegistry",
        "ProtocolKind",
        "RealtimeEvent",
        "RealtimeEventType",
        "RealtimeSession",
        "RealtimeTransport",
        "ReflectionPass",
        "FILE_SEARCH_TOOL_NAME",
        "RemoteAgentCard",
        "register_file_search_tool",
        "resolve_vector_stores",
        "vector_store_registry_from_config",
        "RemoteAgentClient",
        "RemoteAgentDirectory",
        "RemoteAgentRegistry",
        "RemoteAgentTransport",
        "RemoteArtifact",
        "RemoteArtifactStore",
        "RemoteEvent",
        "RemoteMessage",
        "RemoteTask",
        "RemoteTaskStatus",
        "RemoteTaskStore",
        "AGENT_RT_WEB_SEARCH_API_KEY_ENV",
        "AGENT_RT_WEB_SEARCH_TIMEOUT_ENV",
        "AGENT_RT_WEB_SEARCH_TOKEN_ENV",
        "AGENT_RT_WEB_SEARCH_TOOL_ENV",
        "EnvironmentWebSearchProvider",
        "EnvironmentVectorDBProvider",
        "POPULAR_VECTOR_DB_BACKENDS",
        "AGENT_RT_VECTOR_DB_API_KEY_ENV",
        "AGENT_RT_VECTOR_DB_TIMEOUT_ENV",
        "register_popular_vector_db_backends",
        "AGENT_RT_VECTOR_DB_COLLECTION_ENV",
        "AGENT_RT_VECTOR_DB_ENV",
        "AGENT_RT_VECTOR_DB_OPTION_PREFIX",
        "AGENT_RT_VECTOR_DB_URL_ENV",
        "RetrievalKind",
        "RetrievalProvider",
        "RetrievalReranker",
        "RetrievalQuery",
        "RetrievalRegistry",
        "RetrievalResult",
        "VectorDBConfig",
        "VectorDBProviderRegistry",
        "vector_db_config_from_environment",
        "vector_db_provider_from_environment",
        "ReviewFinding",
        "ReviewResult",
        "RoundRobinTeam",
        "RuntimeExecutor",
        "RuntimeRequest",
        "RuntimeResponse",
        "SpeakerSelector",
        "SpeculativeBranch",
        "SpeculativeBranchResult",
        "SpeculativeOrchestrator",
        "SpeculativeResult",
        "StepExecutor",
        "SubagentSpec",
        "SupervisorWorkerTeam",
        "SwarmDecision",
        "SwarmHandoffPolicy",
        "SwarmTeam",
        "TeamMember",
        "TeamNode",
        "TeamTurn",
        "VerificationIssue",
        "VerificationReport",
        "WorkerAssignment",
        "WorkerResult",
        "agent_as_tool",
        "WorkspaceRecord",
        "PersistentWorkspaceStore",
        "TenantWorkspaceStore",
        "ArtifactKind",
        "ProvenanceRecord",
        "Artifact",
        "ArtifactVersion",
        "ArtifactRepository",
        "InMemoryArtifactRepository",
        "SandboxResourceLimits",
        "sandbox_resource_limits_from_env",
        "SandboxNetworkMode",
        "SandboxNetworkPolicy",
        "sandbox_network_policy_from_env",
        "SandboxSnapshot",
        "SandboxCommand",
        "SandboxCommandResult",
        "SandboxBackend",
        "SandboxCommandRunner",
        "CallbackSandboxBackend",
        "SandboxBackendName",
        "SANDBOX_BACKEND_ENV",
        "NativeSandboxBackend",
        "DockerSandboxBackend",
        "E2BSandboxBackend",
        "MicrosandboxBackend",
        "SWEReXSandboxBackend",
        "sandbox_backend_from_env",
        "SandboxPackageManager",
        "SandboxPackageInstaller",
        "CallbackSandboxPackageManager",
        "CodeInterpreterRunner",
        "CodeInterpreterRegistry",
        "SandboxSession",
        "sandbox_shell_tool",
        "ApprovalPresentation",
        "ChatAdapter",
        "ChatEnvelope",
        "ChatReply",
        "ChatSessionBridge",
        "CostCategory",
        "CostLedger",
        "CostRecord",
        "DebugReplayStore",
        "DeterministicReplay",
        "LogSeverity",
        "ReplayBundle",
        "ReplayExchange",
        "ReplayModelProvider",
        "SpanKind",
        "SpanStatus",
        "StructuredLogRecord",
        "StructuredLogger",
        "TokenLedger",
        "TokenUsageRecord",
        "TraceRecorder",
        "TraceSpan",
        "approval_presentation",
        "MemoryKind",
        "SessionSnapshot",
        "ShortTermSessionMemory",
        "TenantSessionMemory",
        "MemoryRecord",
        "MemorySearchQuery",
        "MemorySearchResult",
        "LongTermMemoryStore",
        "InMemoryLongTermMemoryStore",
        "EmbeddingProvider",
        "SemanticMemory",
        "Episode",
        "EpisodicMemory",
        "MemoryMigration",
        "MemoryLifecyclePolicy",
        "LifecycleMemoryStore",
        "Procedure",
        "ProceduralMemory",
        "MemoryWriteCandidate",
        "MemoryWriteDecision",
        "MemoryWritePolicy",
        "MemoryRetrievalPolicy",
        "ScopedMemoryStore",
        "IdentityKind",
        "AuthenticationMethod",
        "Principal",
        "CredentialReference",
        "AuthenticationContext",
        "AuthorizationRequirement",
        "AuthorizationDecision",
        "AuthorizationEngine",
        "SecretMetadata",
        "SecretValue",
        "SecretStore",
        "InMemorySecretStore",
        "ScopedSecretStore",
        "PolicyEffect",
        "PolicyRequest",
        "PolicyRule",
        "PolicyDecision",
        "PolicyEngine",
        "DataClassification",
        "DataEgressRequest",
        "DataExfiltrationPolicy",
        "make_tool_input_exfiltration_guardrail",
        "make_tool_output_exfiltration_guardrail",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_FEATURE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ext.runtime import optional

    for export in _LAZY_FEATURE_EXPORTS:
        value = getattr(optional, export)
        globals()[export] = value
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_FEATURE_EXPORTS))


# Preserve the historical public identity of the moved core symbols so
# introspection and pickle lookup continue through agent_rt.<Name>.
for _value in tuple(globals().values()):
    if (
        isinstance(_value, type) or callable(_value) and hasattr(_value, "__code__")
    ) and getattr(_value, "__module__", "").startswith("ext.runtime."):
        _value.__module__ = __name__


def hello() -> str:
    return "agent-rt"


from ext.compat.base import (
    install_compat_submodules as _install_compat_submodules,
)

_install_compat_submodules(__import__("sys").modules[__name__])


__all__ = tuple(
    sorted(
        {name for name in globals() if not name.startswith("_")}
        | set(_LAZY_FEATURE_EXPORTS)
    )
)
