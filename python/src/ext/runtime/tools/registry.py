"""The tool registry: registration, authorization, guardrails, and execution."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ext.action_blockers import (
    ActionBlockerLike,
)
from ext.action_blockers import (
    evaluate_action_blocker as _evaluate_action_blocker,
)
from ext.registration_safety import (
    RegistrationSafetyGuard,
    RegistrationSafetySubject,
    enforce_registration_safety,
)
from ext.runtime.authorization.approvals import ApprovalManager
from ext.runtime.authorization.permissions import PermissionEngine
from ext.runtime.guardrails.results import ActionBlockedError, _apply_guardrail_result
from ext.runtime.guardrails.tool_guardrails import (
    DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
    AgentActionGuardClassifier,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    _call_maybe_async,
    _call_tool_output_guardrail,
    make_default_tool_input_guardrail,
)
from ext.runtime.observability.audit import ToolAuditEvent, ToolLifecycleHooks
from ext.runtime.resilience.rate_limits import RateLimiter
from ext.runtime.resilience.transactions import DryRunResult
from ext.runtime.tools.capabilities import CapabilityDescriptor
from ext.runtime.tools.definitions import (
    ContextualToolHandler,
    ToolDefinition,
    ToolExecutionContext,
    ToolHandler,
    _effective_tool_execution_limits,
)
from ext.runtime.tools.programs import ToolTimeoutError
from ext.runtime.tools.validation import (
    ToolArgumentValidationError,
    validate_tool_arguments,
)
from ext.runtime.types import CancellationToken, ToolCall

if TYPE_CHECKING:
    import asyncio

    from ext.runtime.models.interfaces import ModelResponse
    from ext.runtime.resilience.budgets import (
        Deadline,
        ExecutionBudget,
        Finalizer,
        LoopDetector,
    )
else:
    from ext.runtime.types import asyncio


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    namespace: str | None = None
    enabled: bool = True
    handler: ToolHandler | None = None
    contextual_handler: ContextualToolHandler | None = None

    @property
    def name(self) -> str:
        if self.namespace:
            return f"{self.namespace}.{self.definition.name}"
        return self.definition.name

    def model_definition(self) -> ToolDefinition:
        if not self.namespace:
            return self.definition
        return ToolDefinition(
            name=self.name,
            description=self.definition.description,
            input_schema=self.definition.input_schema,
            output_schema=self.definition.output_schema,
            metadata=self.definition.metadata,
            side_effect=self.definition.side_effect,
            error_behavior=self.definition.error_behavior,
            timeout_seconds=self.definition.timeout_seconds,
            memory_bytes=self.definition.memory_bytes,
            cpu_seconds=self.definition.cpu_seconds,
            execution_mode=self.definition.execution_mode,
        )


@dataclass(frozen=True)
class DeferredToolRegistration:
    name: str
    loader: Callable[[], ToolDefinition]
    namespace: str | None = None
    enabled: bool = True
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    handler: ToolHandler | None = None
    contextual_handler: ContextualToolHandler | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}.{self.name}" if self.namespace else self.name


def _scan_json(value: Any) -> str:
    """Serialise for the registration safety scan without escaping non-ASCII.

    ``json.dumps`` defaults to ``ensure_ascii=True``, which turns invisible and
    full-width characters into literal ``\\uXXXX`` text and defeats the
    scanner's Unicode canonicalisation.
    """
    return json.dumps(value, default=str, ensure_ascii=False)


class ToolRegistry:
    def __init__(
        self,
        hooks: ToolLifecycleHooks | None = None,
        *,
        services: Mapping[str, Any] | None = None,
        permission_engine: PermissionEngine | None = None,
        action_blockers: Sequence[ActionBlockerLike] = (),
        tool_input_guardrails: Sequence[ToolInputGuardrail] = (),
        enable_model_tool_input_guardrail: bool = False,
        tool_input_guardrail_model: str = DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
        tool_input_guardrail_classifier: AgentActionGuardClassifier | None = None,
        tool_output_guardrails: Sequence[ToolOutputGuardrail] = (),
        approval_manager: ApprovalManager | None = None,
        rate_limiter: RateLimiter | None = None,
        execution_budget: ExecutionBudget | None = None,
        deadline: Deadline | None = None,
        loop_detector: LoopDetector | None = None,
        cost_estimator: Callable[[ModelResponse], float] | None = None,
        finalizers: Sequence[Finalizer] = (),
        registration_guard: RegistrationSafetyGuard | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._deferred: dict[str, DeferredToolRegistration] = {}
        self._version = 0
        self.hooks = hooks or ToolLifecycleHooks()
        self.services = services or {}
        self.permission_engine = permission_engine
        self.action_blockers = tuple(action_blockers)
        resolved_input_guardrails = list(tool_input_guardrails)
        if enable_model_tool_input_guardrail:
            resolved_input_guardrails.insert(
                0,
                make_default_tool_input_guardrail(
                    model=tool_input_guardrail_model,
                    classify=tool_input_guardrail_classifier,
                ),
            )
        self.tool_input_guardrails = tuple(resolved_input_guardrails)
        self.tool_output_guardrails = tuple(tool_output_guardrails)
        self.approval_manager = approval_manager
        self.rate_limiter = rate_limiter
        self.execution_budget = execution_budget
        self.deadline = deadline
        self.loop_detector = loop_detector
        self.cost_estimator = cost_estimator
        self.finalizers = tuple(finalizers)
        self.registration_guard = registration_guard

    @property
    def version(self) -> int:
        return self._version

    def fork(self) -> ToolRegistry:
        """Shallow copy with independent tool tables, for per-request additions."""
        clone = copy.copy(self)
        clone._tools = dict(self._tools)
        clone._deferred = dict(self._deferred)
        return clone

    @staticmethod
    def qualified_name(name: str, namespace: str | None = None) -> str:
        if not name.strip():
            raise ValueError("tool name must not be empty")
        if namespace is not None and not namespace.strip():
            raise ValueError("tool namespace must not be empty")
        return f"{namespace}.{name}" if namespace else name

    def register(
        self,
        definition: ToolDefinition,
        *,
        namespace: str | None = None,
        enabled: bool = True,
        replace: bool = False,
        handler: ToolHandler | None = None,
        contextual_handler: ContextualToolHandler | None = None,
    ) -> RegisteredTool:
        name = self.qualified_name(definition.name, namespace)
        if handler is not None and contextual_handler is not None:
            raise ValueError("register either handler or contextual_handler, not both")
        enforce_registration_safety(
            RegistrationSafetySubject(
                kind="tool",
                name=name,
                description=definition.description,
                content={
                    "input_schema": _scan_json(definition.input_schema),
                    "output_schema": _scan_json(definition.output_schema),
                    "metadata": _scan_json(definition.metadata),
                },
            ),
            guard=self.registration_guard,
        )
        if (name in self._tools or name in self._deferred) and not replace:
            raise ValueError(f"tool already registered: {name}")
        self._deferred.pop(name, None)
        registered = RegisteredTool(
            definition,
            namespace,
            enabled,
            handler,
            contextual_handler,
        )
        self._tools[name] = registered
        self._version += 1
        return registered

    def register_deferred(
        self,
        name: str,
        loader: Callable[[], ToolDefinition],
        *,
        namespace: str | None = None,
        enabled: bool = True,
        replace: bool = False,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        handler: ToolHandler | None = None,
        contextual_handler: ContextualToolHandler | None = None,
    ) -> DeferredToolRegistration:
        qualified = self.qualified_name(name, namespace)
        if handler is not None and contextual_handler is not None:
            raise ValueError("register either handler or contextual_handler, not both")
        enforce_registration_safety(
            RegistrationSafetySubject(
                kind="tool",
                name=qualified,
                description=description,
                content={"metadata": _scan_json(metadata or {})},
            ),
            guard=self.registration_guard,
        )
        if (qualified in self._tools or qualified in self._deferred) and not replace:
            raise ValueError(f"tool already registered: {qualified}")
        self._tools.pop(qualified, None)
        deferred = DeferredToolRegistration(
            name,
            loader,
            namespace,
            enabled,
            description,
            metadata or {},
            handler,
            contextual_handler,
        )
        self._deferred[qualified] = deferred
        self._version += 1
        return deferred

    def deferred_names(self, *, include_disabled: bool = False) -> tuple[str, ...]:
        return tuple(
            name
            for name, tool in self._deferred.items()
            if include_disabled or tool.enabled
        )

    def load(self, name: str) -> RegisteredTool:
        if name in self._tools:
            return self._tools[name]
        try:
            deferred = self._deferred[name]
        except KeyError as exc:
            raise KeyError(f"deferred tool not registered: {name}") from exc
        if not deferred.enabled:
            raise RuntimeError(f"deferred tool is disabled: {name}")

        definition = deferred.loader()
        if definition.name != deferred.name:
            raise ValueError(
                f"deferred tool loader for {name} returned mismatched name "
                f"{definition.name}"
            )
        # register() removes the deferred entry only after its safety scan
        # passes, so a rejected definition leaves the registration intact.
        return self.register(
            definition,
            namespace=deferred.namespace,
            enabled=True,
            replace=True,
            handler=deferred.handler,
            contextual_handler=deferred.contextual_handler,
        )

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not registered: {name}") from exc

    def enable(self, name: str) -> None:
        tool = self.get(name)
        self._tools[name] = RegisteredTool(
            tool.definition,
            tool.namespace,
            True,
            tool.handler,
            tool.contextual_handler,
        )
        self._version += 1

    def disable(self, name: str) -> None:
        tool = self.get(name)
        self._tools[name] = RegisteredTool(
            tool.definition,
            tool.namespace,
            False,
            tool.handler,
            tool.contextual_handler,
        )
        self._version += 1

    def unregister(self, name: str) -> RegisteredTool:
        try:
            tool = self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"tool not registered: {name}") from exc
        self._version += 1
        return tool

    def list(self, *, include_disabled: bool = False) -> tuple[RegisteredTool, ...]:
        return tuple(
            tool for tool in self._tools.values() if include_disabled or tool.enabled
        )

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.model_definition() for tool in self.list())

    def capability_descriptors(
        self,
        *,
        include_disabled: bool = False,
        include_deferred: bool = True,
    ) -> tuple[CapabilityDescriptor, ...]:
        capabilities: list[CapabilityDescriptor] = []
        for tool in self.list(include_disabled=include_disabled):
            capabilities.append(
                CapabilityDescriptor(
                    id=f"tool:{tool.name}",
                    kind="tool",
                    name=tool.name,
                    description=tool.definition.description,
                    namespace=tool.namespace,
                    metadata=tool.definition.metadata,
                )
            )
        if include_deferred:
            for name, deferred in self._deferred.items():
                if not include_disabled and not deferred.enabled:
                    continue
                capabilities.append(
                    CapabilityDescriptor(
                        id=f"tool:{name}",
                        kind="tool",
                        name=name,
                        description=deferred.description,
                        namespace=deferred.namespace,
                        metadata=deferred.metadata,
                    )
                )
        return tuple(capabilities)

    def namespaces(self, *, include_deferred: bool = True) -> tuple[str, ...]:
        values = {
            tool.namespace
            for tool in self._tools.values()
            if tool.namespace is not None
        }
        if include_deferred:
            values.update(
                tool.namespace
                for tool in self._deferred.values()
                if tool.namespace is not None
            )
        return tuple(sorted(values))

    def definitions_in_namespace(
        self,
        namespace: str,
    ) -> tuple[ToolDefinition, ...]:
        if not namespace.strip():
            raise ValueError("tool namespace must not be empty")
        return tuple(
            tool.model_definition()
            for tool in self.list()
            if tool.namespace == namespace
        )

    async def execute(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> Any:
        registered = self.get(call.name)
        if not registered.enabled:
            raise RuntimeError(f"tool is disabled: {call.name}")

        if self.permission_engine is not None:
            agent = (request_context or {}).get("agent")
            self.permission_engine.check_tool(
                call.name,
                registered.definition,
                agent=agent if isinstance(agent, str) else None,
            )

        effective_call = call
        if self.hooks.pre_call is not None:
            transformed = await self.hooks.pre_call(call, registered.definition)
            if transformed is not None:
                if transformed.id != call.id or transformed.name != call.name:
                    raise ValueError(
                        "pre-call hook cannot change tool call identity or name"
                    )
                effective_call = transformed

        for blocker in self.action_blockers:
            decision = await _evaluate_action_blocker(
                blocker,
                effective_call,
                registered.definition,
                request_context,
            )
            if decision.blocked:
                raise ActionBlockedError(
                    decision.reason or "action blocker rejected tool call",
                    decision.classifications,
                )

        for guardrail in self.tool_input_guardrails:
            guarded = _apply_guardrail_result(
                effective_call,
                await _call_maybe_async(
                    guardrail, effective_call, registered.definition
                ),
            )
            if not isinstance(guarded, ToolCall):
                raise TypeError("tool input guardrail transform must return ToolCall")
            if guarded.id != call.id or guarded.name != call.name:
                raise ValueError(
                    "tool input guardrail cannot change tool call identity or name"
                )
            effective_call = guarded

        if effective_call.argument_error is not None:
            raise ToolArgumentValidationError(
                call.name, (effective_call.argument_error,)
            )
        validate_tool_arguments(registered.definition, effective_call.arguments)
        if self.rate_limiter is not None:
            context = request_context or {}
            keys = [f"tool:{effective_call.name}"]
            user_id = context.get("user_id")
            tenant_id = context.get("tenant_id")
            if isinstance(user_id, str) and user_id:
                keys.append(f"user:{user_id}")
            if isinstance(tenant_id, str) and tenant_id:
                keys.append(f"tenant:{tenant_id}")
            self.rate_limiter.check_many(keys)
        if (request_context or {}).get(
            "dry_run"
        ) is True and registered.definition.side_effect not in ("none", "read"):
            return DryRunResult(
                tool=effective_call.name,
                arguments=dict(effective_call.arguments),
                side_effect=registered.definition.side_effect,
            )
        if self.approval_manager is not None:
            session_id = (request_context or {}).get("session_id")
            self.approval_manager.check(
                effective_call,
                registered.definition,
                session_id=session_id if isinstance(session_id, str) else None,
            )
        if registered.handler is None and registered.contextual_handler is None:
            raise RuntimeError(f"tool has no registered handler: {call.name}")

        if self.hooks.audit is not None:
            await self.hooks.audit(
                ToolAuditEvent(
                    phase="start",
                    call=effective_call,
                    definition=registered.definition,
                    request_context=request_context or {},
                )
            )

        limits = _effective_tool_execution_limits(registered.definition)

        async def invoke_handler() -> Any:
            if registered.contextual_handler is not None:
                return await registered.contextual_handler(
                    effective_call.arguments,
                    ToolExecutionContext(
                        services=self.services,
                        request_context=request_context or {},
                        cancellation_token=cancellation_token,
                        limits=limits,
                    ),
                )
            assert registered.handler is not None
            return await registered.handler(
                effective_call.arguments,
                cancellation_token,
            )

        try:
            if limits.timeout_seconds is not None and limits.timeout_seconds <= 0:
                raise ToolTimeoutError(call.name, limits.timeout_seconds)
            if limits.timeout_seconds is None:
                value = await invoke_handler()
            else:
                try:
                    value = await asyncio.wait_for(
                        invoke_handler(),
                        timeout=limits.timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    raise ToolTimeoutError(call.name, limits.timeout_seconds) from exc
        except asyncio.CancelledError as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(
                    effective_call,
                    registered.definition,
                    exc,
                )
            if self.hooks.audit is not None:
                await self.hooks.audit(
                    ToolAuditEvent(
                        phase="error",
                        call=effective_call,
                        definition=registered.definition,
                        error=exc,
                        request_context=request_context or {},
                    )
                )
            raise
        except Exception as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(
                    effective_call,
                    registered.definition,
                    exc,
                )
            if self.hooks.audit is not None:
                await self.hooks.audit(
                    ToolAuditEvent(
                        phase="error",
                        call=effective_call,
                        definition=registered.definition,
                        error=exc,
                        request_context=request_context or {},
                    )
                )
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise
            if registered.definition.error_behavior == "return_error":
                return {
                    "error": {
                        "type": "tool_execution_error",
                        "tool": call.name,
                        "message": str(exc),
                    }
                }
            raise

        try:
            if self.hooks.post_call is not None:
                value = await self.hooks.post_call(
                    effective_call,
                    registered.definition,
                    value,
                )
            for guardrail in self.tool_output_guardrails:
                value = _apply_guardrail_result(
                    value,
                    await _call_tool_output_guardrail(
                        guardrail,
                        value,
                        effective_call,
                        registered.definition,
                        request_context or {},
                    ),
                )
        except Exception as exc:
            # The handler already ran; record a terminal audit event so the
            # trail never ends at "requested" for a call with side effects.
            if self.hooks.on_error is not None:
                await self.hooks.on_error(effective_call, registered.definition, exc)
            if self.hooks.audit is not None:
                await self.hooks.audit(
                    ToolAuditEvent(
                        phase="error",
                        call=effective_call,
                        definition=registered.definition,
                        error=exc,
                        request_context=request_context or {},
                    )
                )
            raise
        if self.hooks.audit is not None:
            await self.hooks.audit(
                ToolAuditEvent(
                    phase="success",
                    call=effective_call,
                    definition=registered.definition,
                    value=value,
                    request_context=request_context or {},
                )
            )
        return value
