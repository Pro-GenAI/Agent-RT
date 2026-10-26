"""Lazily loaded feature surface.

The implementations live in themed modules under ``ext.runtime`` (see
docs/DEVELOPMENT.md); this module re-exports them so
``from ext.runtime.optional import X`` and the root ``agent_rt.__getattr__``
keep working. Nothing here may be imported by the eager core.
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib
import math
import os
import posixpath
import re
import shlex
import signal
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

import httpx

# Lazily loaded feature surface extracted from agent_rt.
# This module is imported only when one of the names in _LAZY_FEATURE_EXPORTS
# is first accessed through the root module.
import agent_rt as _core
from agent_rt import (
    AgentConfig,
    AgentRunResult,
    ApprovalDecision,
    ApprovalRequest,
    ArtifactStatus,
    CancellationToken,
    CapabilityGrant,
    ContentPart,
    ContextItem,
    ContextualToolHandler,
    EmbeddingModelProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    ExecutionBudget,
    FileSystem,
    FileSystemBackendRegistry,
    GuardrailResult,
    GuardrailViolationError,
    InMemoryFileSystem,
    JSONValue,
    MemoryScope,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PrivacyRedactor,
    SessionRef,
    TenantContext,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolHandler,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    WorkflowState,
    WorkspaceFiles,
)
from ext.runtime.agent.delegation import (
    AgentInvocation,
    AgentInvoker,
    AgentTool,
    HandoffManager,
    HandoffRequest,
    HandoffResult,
    IsolatedSubagentRunner,
    SubagentSpec,
    SupervisorWorkerTeam,
    WorkerAssignment,
    WorkerResult,
    agent_as_tool,
)
from ext.runtime.agent.planning import (
    Plan,
    PlanExecutionResult,
    Planner,
    PlannerExecutor,
    PlanProgress,
    PlanStep,
    PlanStepStatus,
    PlanTracker,
    PlanVerifier,
    StepExecutor,
    VerificationIssue,
    VerificationReport,
)
from ext.runtime.agent.reflection import (
    CriticAgent,
    CriticPanel,
    ReflectionPass,
    ReviewFinding,
    ReviewResult,
)
from ext.runtime.agent.teams import (
    AgentRoute,
    AgentRouter,
    AgentTeamNode,
    HierarchicalTeam,
    MapReduceOrchestrator,
    MapReduceResult,
    ModelSelectedSpeakerTeam,
    ParallelSubagentExecutor,
    RoundRobinTeam,
    SpeakerSelector,
    SpeculativeBranch,
    SpeculativeBranchResult,
    SpeculativeOrchestrator,
    SpeculativeResult,
    SwarmDecision,
    SwarmHandoffPolicy,
    SwarmTeam,
    TeamMember,
    TeamNode,
    TeamTurn,
)
from ext.runtime.authorization.identity import (
    AuthenticationContext,
    AuthenticationMethod,
    AuthorizationDecision,
    AuthorizationEngine,
    AuthorizationRequirement,
    CredentialReference,
    IdentityKind,
    Principal,
)
from ext.runtime.authorization.policy import (
    PolicyDecision,
    PolicyEffect,
    PolicyEngine,
    PolicyRequest,
    PolicyRule,
)
from ext.runtime.authorization.secrets import (
    InMemorySecretStore,
    ScopedSecretStore,
    SecretMetadata,
    SecretStore,
    SecretValue,
)
from ext.runtime.guardrails.exfiltration import (
    DataClassification,
    DataEgressRequest,
    DataExfiltrationPolicy,
    make_tool_input_exfiltration_guardrail,
    make_tool_output_exfiltration_guardrail,
)
from ext.runtime.interfaces.browser import (
    BrowserAction,
    BrowserBackend,
    BrowserSession,
    BrowserState,
    ComputerAction,
    ComputerBackend,
    ComputerSession,
    ComputerState,
)
from ext.runtime.interfaces.realtime import (
    MultimodalMessage,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
    RealtimeTransport,
)
from ext.runtime.interfaces.surfaces import (
    APIInterface,
    APIOperation,
    APIOperationHandler,
    APIRequest,
    ApprovalPresentation,
    ChatAdapter,
    ChatEnvelope,
    ChatReply,
    ChatSessionBridge,
    CLIInterface,
    IDEContext,
    IDEDiagnostic,
    IDEIntegration,
    IDEProgress,
    ProgressEvent,
    ProgressListener,
    ProgressReporter,
    ProgressStatus,
    RuntimeExecutor,
    RuntimeRequest,
    RuntimeResponse,
    approval_presentation,
)
from ext.runtime.memory.lifecycle import (
    LifecycleMemoryStore,
    MemoryLifecyclePolicy,
    MemoryMigration,
    ProceduralMemory,
    Procedure,
)
from ext.runtime.memory.long_term import (
    InMemoryLongTermMemoryStore,
    LongTermMemoryStore,
    MemoryKind,
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from ext.runtime.memory.policies import (
    MemoryRetrievalPolicy,
    MemoryWriteCandidate,
    MemoryWriteDecision,
    MemoryWritePolicy,
    ScopedMemoryStore,
)
from ext.runtime.memory.semantic import (
    EmbeddingProvider,
    Episode,
    EpisodicMemory,
    SemanticMemory,
)
from ext.runtime.memory.sessions import (
    SessionSnapshot,
    ShortTermSessionMemory,
    TenantSessionMemory,
)
from ext.runtime.observability.replay import (
    DebugReplayStore,
    DeterministicReplay,
    ReplayBundle,
    ReplayExchange,
    ReplayModelProvider,
)
from ext.runtime.observability.tracing import (
    LogSeverity,
    SpanKind,
    SpanStatus,
    StructuredLogger,
    StructuredLogRecord,
    TraceRecorder,
    TraceSpan,
)
from ext.runtime.observability.usage import (
    CostCategory,
    CostLedger,
    CostRecord,
    TokenLedger,
    TokenUsageRecord,
)
from ext.runtime.protocols.adapters import (
    ConnectorDefinition,
    ConnectorRegistry,
    ProtocolAdapter,
    ProtocolAdapterRegistry,
    ProtocolKind,
)
from ext.runtime.protocols.mcp import (
    MCPCapabilityKind,
    MCPClient,
    MCPPrompt,
    MCPResource,
    MCPServerInfo,
    MCPTool,
    MCPTransport,
)
from ext.runtime.protocols.mcp_access import (
    AuthorizedMCPClient,
    MCPAccessPolicy,
    MCPCapabilityFilter,
)
from ext.runtime.protocols.remote import (
    CapabilityNegotiation,
    CapabilitySet,
    RemoteAgentCard,
    RemoteAgentClient,
    RemoteAgentDirectory,
    RemoteAgentRegistry,
    RemoteAgentTransport,
    RemoteArtifact,
    RemoteArtifactStore,
    RemoteEvent,
    RemoteMessage,
    RemoteTask,
    RemoteTaskStatus,
    RemoteTaskStore,
)
from ext.runtime.retrieval.file_search import (
    FILE_SEARCH_MAX_RESULTS,
    FILE_SEARCH_TOOL_NAME,
    register_file_search_tool,
)
from ext.runtime.retrieval.registry import (
    RetrievalKind,
    RetrievalProvider,
    RetrievalQuery,
    RetrievalRegistry,
    RetrievalReranker,
    RetrievalResult,
)
from ext.runtime.retrieval.vector_db import (
    AGENT_RT_VECTOR_DB_API_KEY_ENV,
    AGENT_RT_VECTOR_DB_COLLECTION_ENV,
    AGENT_RT_VECTOR_DB_ENV,
    AGENT_RT_VECTOR_DB_OPTION_PREFIX,
    AGENT_RT_VECTOR_DB_TIMEOUT_ENV,
    AGENT_RT_VECTOR_DB_URL_ENV,
    POPULAR_VECTOR_DB_BACKENDS,
    EnvironmentVectorDBProvider,
    VectorDBConfig,
    VectorDBProviderRegistry,
    vector_db_config_from_environment,
    vector_db_provider_from_environment,
)
from ext.runtime.retrieval.vector_stores import (
    register_popular_vector_db_backends,
    resolve_vector_stores,
    vector_store_registry_from_config,
)
from ext.runtime.retrieval.web_search import (
    AGENT_RT_WEB_SEARCH_API_KEY_ENV,
    AGENT_RT_WEB_SEARCH_TIMEOUT_ENV,
    AGENT_RT_WEB_SEARCH_TOKEN_ENV,
    AGENT_RT_WEB_SEARCH_TOOL_ENV,
    EnvironmentWebSearchProvider,
)

# Preserve the historical public identity of lazily extracted symbols so
# introspection and pickle lookup continue through agent_rt.<Name>.
for _name, _value in tuple(globals().items()):
    if (
        not _name.startswith("_")
        and _name not in vars(_core)
        and getattr(_value, "__module__", "").startswith("ext.runtime.")
    ):
        _value.__module__ = "agent_rt"


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in vars(_core)
    )
)

from ext.runtime.sandbox.backends import (
    SANDBOX_BACKEND_ENV,
    CallbackSandboxBackend,
    SandboxBackend,
    SandboxBackendName,
    SandboxCommand,
    SandboxCommandResult,
    SandboxCommandRunner,
    SandboxSnapshot,
)
from ext.runtime.sandbox.docker import DockerSandboxBackend
from ext.runtime.sandbox.e2b import E2BSandboxBackend
from ext.runtime.sandbox.microsandbox import MicrosandboxBackend
from ext.runtime.sandbox.native import (
    DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES,
    NativeSandboxBackend,
    _collect_process_result,
    _native_preexec,
)
from ext.runtime.sandbox.policy import (
    SANDBOX_ALLOW_HTTP_ENV,
    SANDBOX_ALLOW_IP_ENV,
    SANDBOX_ALLOW_PROXY_ENV,
    SANDBOX_ALLOW_WEBSOCKET_ENV,
    SANDBOX_ALLOWED_DOMAINS_ENV,
    SANDBOX_BLOCKED_DOMAINS_ENV,
    SANDBOX_NETWORK_BPS_ENV,
    SANDBOX_NETWORK_BYTES_ENV,
    SANDBOX_NETWORK_MODE_ENV,
    SandboxNetworkMode,
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    sandbox_network_policy_from_env,
    sandbox_resource_limits_from_env,
)
from ext.runtime.sandbox.session import (
    CallbackSandboxPackageManager,
    CodeInterpreterRegistry,
    CodeInterpreterRunner,
    SandboxPackageInstaller,
    SandboxPackageManager,
    SandboxSession,
    sandbox_backend_from_env,
    sandbox_shell_tool,
)
from ext.runtime.sandbox.swe_rex import SWEReXSandboxBackend
from ext.runtime.workspace.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRepository,
    ArtifactVersion,
    InMemoryArtifactRepository,
    ProvenanceRecord,
)
from ext.runtime.workspace.stores import (
    PersistentWorkspaceStore,
    TenantWorkspaceStore,
    WorkspaceRecord,
)

# Names defined after the public surface above have always reported this
# module as their home.
for _name, _value in tuple(globals().items()):
    if (
        not _name.startswith("_")
        and (isinstance(_value, type) or hasattr(_value, "__code__"))
        and getattr(_value, "__module__", "").startswith("ext.runtime.")
    ):
        _value.__module__ = __name__
