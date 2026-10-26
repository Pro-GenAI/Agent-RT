"""Compiling an agent's execution plan: visible tools, policies, and active stages."""

from __future__ import annotations

from collections.abc import Sequence
from time import monotonic
from typing import TYPE_CHECKING

from ext.runtime.agent.config import (
    AgentConfig,
    CompiledExecutionPlan,
)
from ext.runtime.context.assembler import ContextAssembler
from ext.runtime.context.items import (
    ContextItem,
    ContextSelectionPolicy,
)
from ext.runtime.models.interfaces import (
    StructuredOutputRequirement,
)
from ext.runtime.tools.definitions import (
    ToolSelectionRequirement,
)

if TYPE_CHECKING:

    from ext.runtime.agent.loop import AgentLoop
else:
    pass


def compile_plan(
    self: AgentLoop,
    agent: AgentConfig,
    context_items: Sequence[ContextItem] = (),
    context_policy: ContextSelectionPolicy | None = None,
) -> CompiledExecutionPlan:
    stage_started = monotonic() if self.metrics is not None else None
    tool_policy = agent.tool_policy
    visible_tools = (
        self.tool_registry.definitions() if self.tool_registry is not None else ()
    )
    if self.capability_grant is not None:
        visible_tools = self.capability_grant.filter_tools(visible_tools)
    visible_tools = self.prompt_injection_defense.filter_tools(
        context_items,
        visible_tools,
    )
    if tool_policy is not None:
        visible_tools = tool_policy.filter_definitions(visible_tools)
    if context_policy is not None and context_policy.tool_names is not None:
        visible_tools = tuple(
            tool for tool in visible_tools if tool.name in context_policy.tool_names
        )

    visible_names = frozenset(tool.name for tool in visible_tools)
    tool_selection: ToolSelectionRequirement | None = None
    if self.tool_filter is None and tool_policy is not None:
        missing_required = tool_policy.required - visible_names
        if missing_required:
            raise ValueError(
                "required tools are unavailable: " + ", ".join(sorted(missing_required))
            )
        tool_selection = ToolSelectionRequirement(
            required=tuple(sorted(tool_policy.required)),
            preferred=tuple(
                name for name in tool_policy.preferred if name in visible_names
            ),
        )

    structured_output = (
        StructuredOutputRequirement(
            schema=agent.output.schema,
            name=agent.name,
            strict=True,
        )
        if agent.output is not None and agent.output.schema is not None
        else None
    )
    active_stages = tuple(
        stage
        for stage, enabled in (
            ("tools", bool(visible_tools)),
            ("dynamic_tool_filter", self.tool_filter is not None),
            ("structured_output", structured_output is not None),
            (
                "input_guardrails",
                len(self.input_guardrails) > self._default_input_guardrail_count,
            ),
            (
                "output_guardrails",
                len(self.output_guardrails) > self._default_output_guardrail_count,
            ),
        )
        if enabled
    )
    simple_text_fast_path = (
        not active_stages
        and tool_policy is None
        and not agent.model.fallback_models
        and self.context_assembler.__class__ is ContextAssembler
        and self.context_assembler.compaction_policy is None
        and self.context_assembler.artifact_store is None
        and self.context_assembler.offload_policy is None
        and self.context_assembler.prompt_cache_policy is None
    )
    plan = CompiledExecutionPlan(
        model_settings=agent.model,
        visible_tools=tuple(visible_tools),
        tool_policy=tool_policy,
        structured_output=structured_output,
        tool_selection=tool_selection,
        visible_tool_names=visible_names,
        tool_registry_version=(
            self.tool_registry.version if self.tool_registry is not None else None
        ),
        dynamic_tool_filter=self.tool_filter is not None,
        simple_text_fast_path=simple_text_fast_path,
        active_stages=active_stages,
    )
    if stage_started is not None and self.metrics is not None:
        self.metrics.record(
            "agent_loop.stage.duration_ms",
            (monotonic() - stage_started) * 1000,
            kind="histogram",
            labels={"stage": "compile_plan"},
        )
    return plan
