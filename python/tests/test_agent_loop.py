import asyncio
import os

import pytest

from agent_rt import (
    AgentCheckpoint,
    AgentConfig,
    AgentLoop,
    AgentOutputRequirements,
    AgentRunLimits,
    BackgroundTaskManager,
    CancellationToken,
    ContentPart,
    ContextAssembler,
    ContextCompactionPolicy,
    ContextItem,
    ContextOffloadPolicy,
    ContextSelectionPolicy,
    InMemoryArtifactStore,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    InMemoryIdempotencyStore,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ModelUsage,
    OpenLLMetryConfig,
    PromptCachePolicy,
    RuntimeMetrics,
    StructuredOutputValidationError,
    ToolArgumentValidationError,
    ToolCall,
    ToolDefinition,
    ToolLifecycleHooks,
    ToolProgramExecutor,
    ToolRegistry,
    ToolSelectionPolicy,
    WorkflowState,
    checkpoint_from_result,
    initialize_openllmetry_from_env,
    openllmetry_config_from_env,
    validate_tool_arguments,
)


class FakeProvider:
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class StreamingFakeProvider:
    name = "streaming-fake"

    def __init__(self, streams):
        self.streams = [list(events) for events in streams]
        self.requests = []

    async def complete(self, request):
        raise AssertionError("complete() should not be used by run_streaming")

    async def stream(self, request):
        self.requests.append(request)
        if not self.streams:
            raise AssertionError("unexpected extra stream call")
        for event in self.streams.pop(0):
            yield event


class FakeTools:
    def __init__(self):
        self.calls = []

    async def execute(self, call):
        self.calls.append(call)
        return {"ok": call.arguments}


def text_message(role, text, *, tool_calls=()):
    return ModelMessage(
        role=role,
        content=(ContentPart(type="text", text=text),),
        tool_calls=tool_calls,
    )


class TestAgentLoop:
    def agent(self):
        return AgentConfig(
            name="test",
            instructions="Be concise.",
            model=ModelSettings(model="fake-model"),
        )

    def test_openllmetry_activation_requires_valid_export_configuration(self):
        assert openllmetry_config_from_env({}) is None
        assert openllmetry_config_from_env({"TRACELOOP_API_KEY": ""}) is None
        assert (
            openllmetry_config_from_env({"TRACELOOP_BASE_URL": "not a valid endpoint"})
            is None
        )
        assert (
            openllmetry_config_from_env(
                {"TRACELOOP_API_KEY": "key", "TRACELOOP_TRACE_CONTENT": "maybe"}
            )
            is None
        )
        assert not initialize_openllmetry_from_env(
            {}, initializer=lambda **_options: pytest.fail("initializer should not run")
        )

    def test_openllmetry_valid_environment_initializes_with_validated_options(self):
        calls = []
        env = {
            "TRACELOOP_API_KEY": " test-key ",
            "TRACELOOP_BASE_URL": "https://collector.example.com",
            "TRACELOOP_HEADERS": "x-tenant=tenant-a,x-env=prod",
            "TRACELOOP_TRACE_CONTENT": "false",
            "TRACELOOP_TELEMETRY": "0",
            "TRACELOOP_ENRICH_TOKENS": "true",
            "TRACELOOP_APP_NAME": "agent-rt-tests",
        }

        config = openllmetry_config_from_env(env)
        assert config == OpenLLMetryConfig(
            api_key="test-key",
            base_url="https://collector.example.com",
            headers={"x-tenant": "tenant-a", "x-env": "prod"},
            trace_content=False,
            telemetry_enabled=False,
            enrich_tokens=True,
            app_name="agent-rt-tests",
        )
        assert initialize_openllmetry_from_env(
            env, initializer=lambda **options: calls.append(options)
        )
        assert calls == [
            {
                "app_name": "agent-rt-tests",
                "api_key": "test-key",
                "api_endpoint": "https://collector.example.com",
                "headers": {"x-tenant": "tenant-a", "x-env": "prod"},
                "telemetry_enabled": False,
                "should_enrich_metrics": True,
            }
        ]

    def test_compiled_plan_caches_static_tool_metadata_and_tracks_registry_version(
        self,
    ):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Lookup",
                input_schema={"type": "object"},
            )
        )
        loop = AgentLoop(FakeProvider([]), tool_registry=registry)
        first = loop.compile_plan(self.agent())
        assert tuple(tool.name for tool in first.visible_tools) == ("lookup",)
        assert first.tool_registry_version == registry.version

        registry.disable("lookup")
        second = loop.compile_plan(self.agent())
        assert second.visible_tools == ()
        assert first.tool_registry_version != second.tool_registry_version

    def test_compiled_plan_keeps_dynamic_tool_filter_dynamic(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Lookup",
                input_schema={"type": "object"},
            )
        )
        loop = AgentLoop(
            FakeProvider([]),
            tool_registry=registry,
            tool_filter=lambda _context, tools: (tool.name for tool in tools),
        )
        plan = loop.compile_plan(self.agent())
        assert plan.dynamic_tool_filter
        assert "dynamic_tool_filter" in plan.active_stages

    def test_simple_text_fast_path_has_conservative_boundaries(self):
        loop = AgentLoop(FakeProvider([]))
        assert loop.compile_plan(self.agent()).simple_text_fast_path

        structured_agent = AgentConfig(
            name="test",
            instructions="Be concise.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={"type": "object"},
            ),
        )
        assert not loop.compile_plan(structured_agent).simple_text_fast_path

        class CustomAssembler(ContextAssembler):
            pass

        custom = AgentLoop(FakeProvider([]), context_assembler=CustomAssembler())
        assert not custom.compile_plan(self.agent()).simple_text_fast_path

    async def test_stage_timing_diagnostics_cover_fast_and_five_turn_paths(self):
        fast_metrics = RuntimeMetrics()
        fast_provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        await AgentLoop(fast_provider, metrics=fast_metrics).run(
            self.agent(),
            [text_message("user", "go")],
        )
        assert (
            len(
                fast_metrics.values(
                    "agent_loop.stage.duration_ms", labels={"stage": "compile_plan"}
                )
            )
            == 1
        )
        assert (
            len(
                fast_metrics.values(
                    "agent_loop.stage.duration_ms",
                    labels={"stage": "request_assembly", "path": "fast"},
                )
            )
            == 1
        )
        assert (
            len(
                fast_metrics.values(
                    "agent_loop.stage.duration_ms", labels={"stage": "model_call"}
                )
            )
            == 1
        )

        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            return {"ok": True}

        registry.register(
            ToolDefinition(
                name="step",
                description="Advance one turn.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        calls = [
            ToolCall(id=f"step-{index}", name="step", arguments={})
            for index in range(4)
        ]
        provider = FakeProvider(
            [
                *[
                    ModelResponse(
                        message=text_message("assistant", "", tool_calls=(call,))
                    )
                    for call in calls
                ],
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        metrics = RuntimeMetrics()
        result = await AgentLoop(
            provider,
            tool_registry=registry,
            metrics=metrics,
        ).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert result.turns == 5
        assert (
            len(
                metrics.values(
                    "agent_loop.stage.duration_ms", labels={"stage": "compile_plan"}
                )
            )
            == 1
        )
        assert (
            len(
                metrics.values(
                    "agent_loop.stage.duration_ms",
                    labels={"stage": "request_assembly", "path": "general"},
                )
            )
            == 5
        )
        assert (
            len(
                metrics.values(
                    "agent_loop.stage.duration_ms", labels={"stage": "model_call"}
                )
            )
            == 5
        )

    async def test_tool_round_trip_continues_to_completion(self):
        call = ToolCall(id="1", name="lookup", arguments={"q": "x"})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,)),
                    finish_reason="tool_calls",
                    usage=ModelUsage(total_tokens=5),
                ),
                ModelResponse(
                    message=text_message("assistant", "done"),
                    finish_reason="stop",
                    usage=ModelUsage(total_tokens=7),
                ),
            ]
        )
        tools = FakeTools()

        result = await AgentLoop(provider, tools).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert result.termination_reason == "completed"
        assert result.turns == 2
        assert result.tool_calls == 1
        assert result.total_tokens == 12
        assert provider.requests[1].messages[-1].role == "tool"
        assert provider.requests[1].messages[-1].tool_call_id == "1"

    async def test_agent_loop_records_model_usage_with_attribution(self):
        recorded = []

        class Recorder:
            def record_model_usage(self, usage, **attribution):
                recorded.append((usage, attribution))

        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "done"),
                    model="resolved-model",
                    usage=ModelUsage(
                        input_tokens=11,
                        output_tokens=7,
                        cached_tokens=3,
                        reasoning_tokens=2,
                        total_tokens=18,
                    ),
                )
            ]
        )
        result = await AgentLoop(provider, token_ledger=Recorder()).run(
            self.agent(),
            [text_message("user", "go")],
            tool_context={"user_id": "user-1", "tenant_id": "tenant-1"},
            task_id="task-1",
        )

        assert result.total_tokens == 18
        assert len(recorded) == 1
        usage, attribution = recorded[0]
        assert usage.input_tokens == 11
        assert usage.output_tokens == 7
        assert attribution["user_id"] == "user-1"
        assert attribution["tenant_id"] == "tenant-1"
        assert attribution["task_id"] == "task-1"
        assert attribution["agent_id"] == self.agent().name
        assert attribution["model"] == "resolved-model"

    async def test_max_turns_stops_repeated_tool_loop(self):
        call = ToolCall(id="1", name="lookup", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
            ]
        )
        result = await AgentLoop(provider, FakeTools()).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(max_turns=1),
        )
        assert result.termination_reason == "max_turns"
        assert result.turns == 1

    async def test_tool_call_limit_is_checked_before_execution(self):
        call = ToolCall(id="1", name="lookup", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )
        tools = FakeTools()
        result = await AgentLoop(provider, tools).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(max_tool_calls=0),
        )
        assert result.termination_reason == "max_tool_calls"
        assert tools.calls == []

    async def test_budget_and_stop_conditions(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "done"),
                    usage=ModelUsage(total_tokens=10),
                )
            ]
        )
        budgeted = await AgentLoop(provider).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(max_total_tokens=10),
        )
        assert budgeted.termination_reason == "budget_exhausted"

        stopped = await AgentLoop(FakeProvider([])).run(
            self.agent(),
            [text_message("user", "go")],
            stop_requested=lambda: True,
        )
        assert stopped.termination_reason == "stop_requested"

    async def test_output_schema_is_forwarded(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={"type": "object", "required": ["value"]},
            ),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", '{"value":1}'))]
        )
        await AgentLoop(provider).run(agent, [text_message("user", "go")])
        requirement = provider.requests[0].structured_output
        assert requirement is not None
        assert requirement.schema["type"] == "object"
        assert requirement.strict

    async def test_structured_output_is_parsed_and_returned(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": "integer"}},
                    "additionalProperties": False,
                },
            ),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", '{"value": 3}'))]
        )
        result = await AgentLoop(provider).run(agent, [text_message("user", "go")])
        assert result.structured_output == {"value": 3}

    async def test_invalid_structured_output_is_repaired_once(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": "integer"}},
                },
                max_repair_attempts=1,
            ),
        )
        provider = FakeProvider(
            [
                ModelResponse(message=text_message("assistant", '{"value": "bad"}')),
                ModelResponse(message=text_message("assistant", '{"value": 7}')),
            ]
        )
        result = await AgentLoop(provider).run(agent, [text_message("user", "go")])
        assert result.termination_reason == "completed"
        assert result.turns == 2
        assert result.structured_output == {"value": 7}
        repair_message = provider.requests[1].messages[-1]
        assert repair_message.role == "user"
        assert "$.value: expected integer" in repair_message.content[0].text

    async def test_invalid_structured_output_raises_typed_error_after_repairs(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={"type": "object", "required": ["value"]},
                max_repair_attempts=0,
            ),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", '{"other": 1}'))]
        )
        with pytest.raises(StructuredOutputValidationError) as caught:
            await AgentLoop(provider).run(agent, [text_message("user", "go")])
        assert "$.value: required property is missing" in caught.value.issues

    async def test_invalid_json_raises_typed_error_when_repair_disabled(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(format="json", max_repair_attempts=0),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "not-json"))]
        )
        with pytest.raises(StructuredOutputValidationError) as caught:
            await AgentLoop(provider).run(agent, [text_message("user", "go")])
        assert "invalid JSON" in str(caught.value)

    async def test_json_content_part_is_validated_without_reparsing_text(self):
        agent = AgentConfig(
            name="structured",
            instructions="Return JSON.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(
                format="json",
                schema={"type": "array", "items": {"type": "string"}},
            ),
        )
        response = ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="json", data=["a", "b"]),),
            )
        )
        result = await AgentLoop(FakeProvider([response])).run(
            agent,
            [text_message("user", "go")],
        )
        assert result.structured_output == ["a", "b"]

    async def test_streaming_forwards_model_events_and_returns_final_result(self):
        final = ModelResponse(
            message=text_message("assistant", "Hello"),
            usage=ModelUsage(total_tokens=4),
            finish_reason="stop",
        )
        provider = StreamingFakeProvider(
            [
                [
                    ModelStreamEvent(type="reasoning_delta", text="thinking"),
                    ModelStreamEvent(type="text_delta", text="Hel"),
                    ModelStreamEvent(type="text_delta", text="lo"),
                    ModelStreamEvent(type="status", status="finalizing"),
                    ModelStreamEvent(type="completed", response=final),
                ]
            ]
        )
        seen = []

        async def on_event(event):
            seen.append(event)

        result = await AgentLoop(provider).run_streaming(
            self.agent(),
            [text_message("user", "go")],
            on_event,
        )

        assert [event.type for event in seen] == [
            "reasoning_delta",
            "text_delta",
            "text_delta",
            "status",
            "completed",
        ]
        assert result.termination_reason == "completed"
        assert result.final_response == final
        assert result.total_tokens == 4

    async def test_streaming_preserves_tool_continuation_across_model_turns(self):
        call = ToolCall(id="1", name="lookup", arguments={"q": "x"})
        first = ModelResponse(
            message=text_message("assistant", "", tool_calls=(call,)),
            finish_reason="tool_calls",
        )
        second = ModelResponse(
            message=text_message("assistant", "done"),
            finish_reason="stop",
        )
        provider = StreamingFakeProvider(
            [
                [
                    ModelStreamEvent(
                        type="tool_call_delta",
                        tool_call_id="1",
                        tool_name="lookup",
                        arguments_delta='{"q":"x"}',
                    ),
                    ModelStreamEvent(type="completed", response=first),
                ],
                [
                    ModelStreamEvent(type="text_delta", text="done"),
                    ModelStreamEvent(type="completed", response=second),
                ],
            ]
        )
        events = []

        async def on_event(event):
            events.append(event)

        tools = FakeTools()
        result = await AgentLoop(provider, tools).run_streaming(
            self.agent(),
            [text_message("user", "go")],
            on_event,
        )

        assert result.turns == 2
        assert result.tool_calls == 1
        assert [event.type for event in events] == [
            "tool_call_delta",
            "completed",
            "text_delta",
            "completed",
        ]
        assert provider.requests[1].messages[-1].role == "tool"

    async def test_streaming_requires_completed_response_event(self):
        provider = StreamingFakeProvider(
            [[ModelStreamEvent(type="text_delta", text="partial")]]
        )

        async def on_event(event):
            pass

        with pytest.raises(RuntimeError, match="without a completed response"):
            await AgentLoop(provider).run_streaming(
                self.agent(),
                [text_message("user", "go")],
                on_event,
            )

    async def test_streaming_rejects_non_streaming_provider(self):
        async def on_event(event):
            pass

        with pytest.raises(TypeError, match="provider that implements stream"):
            await AgentLoop(
                FakeProvider([ModelResponse(message=text_message("assistant", "done"))])
            ).run_streaming(
                self.agent(),
                [text_message("user", "go")],
                on_event,
            )

    async def test_pre_cancelled_run_never_calls_provider(self):
        token = CancellationToken()
        token.cancel()
        provider = FakeProvider([])
        result = await AgentLoop(provider).run(
            self.agent(),
            [text_message("user", "go")],
            cancellation_token=token,
        )
        assert result.termination_reason == "cancelled"
        assert provider.requests == []

    async def test_cancellation_interrupts_active_model_call_and_is_propagated(self):
        token = CancellationToken()

        class SlowProvider:
            name = "slow"

            def __init__(self):
                self.requests = []
                self.cancelled = False

            async def complete(self, request):
                self.requests.append(request)
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                raise AssertionError("unreachable")

        provider = SlowProvider()
        run_task = asyncio.create_task(
            AgentLoop(provider).run(
                self.agent(),
                [text_message("user", "go")],
                cancellation_token=token,
            )
        )
        await asyncio.sleep(0)
        token.cancel()
        result = await run_task

        assert result.termination_reason == "cancelled"
        assert provider.requests[0].cancellation_token is token
        assert provider.cancelled

    async def test_outer_task_cancellation_cancels_active_model_and_marks_token(self):
        token = CancellationToken()

        class SlowProvider:
            name = "slow"

            def __init__(self):
                self.started = asyncio.Event()
                self.cancelled = False

            async def complete(self, request):
                self.started.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                raise AssertionError("unreachable")

        provider = SlowProvider()
        finalized = []
        run_task = asyncio.create_task(
            AgentLoop(provider, finalizers=(lambda: finalized.append(True),)).run(
                self.agent(),
                [text_message("user", "go")],
                cancellation_token=token,
            )
        )
        await provider.started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert token.is_cancelled
        assert provider.cancelled
        assert finalized == [True]

    async def test_outer_task_cancellation_cancels_active_tool(self):
        token = CancellationToken()
        call = ToolCall(id="1", name="slow", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )

        class SlowTools:
            def __init__(self):
                self.started = asyncio.Event()
                self.cancelled = False

            async def execute(self, call, cancellation_token=None):
                self.started.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                raise AssertionError("unreachable")

        tools = SlowTools()
        run_task = asyncio.create_task(
            AgentLoop(provider, tools).run(
                self.agent(),
                [text_message("user", "go")],
                cancellation_token=token,
            )
        )
        await tools.started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert token.is_cancelled
        assert tools.cancelled

    async def test_cancellation_interrupts_active_tool_call_and_is_propagated(self):
        token = CancellationToken()
        call = ToolCall(id="1", name="slow", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )

        class CancellableTools:
            def __init__(self):
                self.token = None
                self.cancelled = False

            async def execute(self, call, cancellation_token=None):
                self.token = cancellation_token
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                raise AssertionError("unreachable")

        tools = CancellableTools()
        run_task = asyncio.create_task(
            AgentLoop(provider, tools).run(
                self.agent(),
                [text_message("user", "go")],
                cancellation_token=token,
            )
        )
        while tools.token is None:
            await asyncio.sleep(0)
        token.cancel()
        result = await run_task

        assert result.termination_reason == "cancelled"
        assert tools.token is token
        assert tools.cancelled

    async def test_cancellation_interrupts_stream_iteration(self):
        token = CancellationToken()

        class SlowStreamingProvider:
            name = "slow-stream"

            async def complete(self, request):
                raise AssertionError("complete should not be used")

            async def stream(self, request):
                yield ModelStreamEvent(type="text_delta", text="start")
                await asyncio.sleep(10)

        seen = []

        async def on_event(event):
            seen.append(event)

        run_task = asyncio.create_task(
            AgentLoop(SlowStreamingProvider()).run_streaming(
                self.agent(),
                [text_message("user", "go")],
                on_event,
                cancellation_token=token,
            )
        )
        while not seen:
            await asyncio.sleep(0)
        token.cancel()
        result = await run_task
        assert result.termination_reason == "cancelled"
        assert seen[0].text == "start"

    async def test_registry_forwards_only_enabled_tool_definitions(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={"type": "object"},
            ),
            namespace="crm",
        )
        registry.register(
            ToolDefinition(
                name="hidden",
                description="Hidden.",
                input_schema={"type": "object"},
            ),
            enabled=False,
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )
        assert [tool.name for tool in provider.requests[0].tools] == ["crm.lookup"]

    def test_validate_tool_arguments_surfaces_typed_schema_issues(self):
        definition = ToolDefinition(
            name="lookup",
            description="Look up.",
            input_schema={
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
        with pytest.raises(ToolArgumentValidationError) as caught:
            validate_tool_arguments(definition, {"id": "bad", "extra": True})
        assert "$.id: expected integer" in caught.value.issues
        assert "$.extra: additional property is not allowed" in caught.value.issues

    async def test_invalid_tool_arguments_return_actionable_error_without_execution(
        self,
    ):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "integer"}},
                },
            )
        )
        bad_call = ToolCall(id="bad", name="lookup", arguments={"id": "x"})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message(
                        "assistant",
                        "",
                        tool_calls=(bad_call,),
                    )
                ),
                ModelResponse(message=text_message("assistant", "recovered")),
            ]
        )
        tools = FakeTools()
        result = await AgentLoop(provider, tools, registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert tools.calls == []
        assert result.termination_reason == "completed"
        assert result.tool_calls == 1
        tool_message = provider.requests[1].messages[-1]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == "bad"
        payload = tool_message.content[0].data
        assert payload["error"]["type"] == "tool_argument_validation"
        assert "$.id: expected integer" in payload["error"]["issues"]

    async def test_unknown_registered_tool_call_returns_error_without_execution(self):
        registry = ToolRegistry()
        call = ToolCall(id="missing", name="missing", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        tools = FakeTools()
        await AgentLoop(provider, tools, registry).run(
            self.agent(),
            [text_message("user", "go")],
        )
        assert tools.calls == []
        payload = provider.requests[1].messages[-1].content[0].data
        assert "not registered" in payload["error"]["issues"][0]

    async def test_registry_handler_dispatches_without_external_executor_and_marshals_text(
        self,
    ):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            return f"value:{arguments['id']}"

        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "integer"}},
                },
            ),
            handler=handler,
        )
        call = ToolCall(id="1", name="lookup", arguments={"id": 3})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )

        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert result.termination_reason == "completed"
        tool_message = provider.requests[1].messages[-1]
        assert tool_message.content[0].type == "text"
        assert tool_message.content[0].text == "value:3"

    async def test_registry_handler_json_result_is_marshaled_as_json(self):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            return {"value": arguments["id"]}

        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        call = ToolCall(id="1", name="lookup", arguments={"id": 4})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )
        tool_message = provider.requests[1].messages[-1]
        assert tool_message.content[0].type == "json"
        assert tool_message.content[0].data == {"value": 4}

    async def test_registry_return_error_marshals_handler_failure(self):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            raise RuntimeError("boom")

        registry.register(
            ToolDefinition(
                name="fragile",
                description="Fails.",
                input_schema={"type": "object"},
                error_behavior="return_error",
            ),
            handler=handler,
        )
        call = ToolCall(id="1", name="fragile", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )
        payload = provider.requests[1].messages[-1].content[0].data
        assert payload["error"]["type"] == "tool_execution_error"
        assert payload["error"]["message"] == "boom"

    async def test_registry_raise_error_behavior_propagates_handler_failure(self):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            raise RuntimeError("boom")

        registry.register(
            ToolDefinition(
                name="fragile",
                description="Fails.",
                input_schema={"type": "object"},
                error_behavior="raise",
            ),
            handler=handler,
        )
        call = ToolCall(id="1", name="fragile", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )
        with pytest.raises(RuntimeError, match="boom"):
            await AgentLoop(provider, tool_registry=registry).run(
                self.agent(),
                [text_message("user", "go")],
            )

    async def test_registered_tool_timeout_returns_recoverable_tool_error(self):
        registry = ToolRegistry()
        started = False

        async def handler(arguments, cancellation_token):
            nonlocal started
            started = True
            await asyncio.sleep(10)

        registry.register(
            ToolDefinition(
                name="slow",
                description="Slow.",
                input_schema={"type": "object"},
                timeout_seconds=0.01,
            ),
            handler=handler,
        )
        call = ToolCall(id="slow-1", name="slow", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "recovered")),
            ]
        )

        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert started
        assert result.termination_reason == "completed"
        payload = provider.requests[1].messages[-1].content[0].data
        assert payload["error"]["type"] == "tool_timeout"
        assert payload["error"]["tool"] == "slow"

    async def test_execution_env_defaults_apply_to_tool_calls(self):
        names = (
            "AGENT_RT_EXECUTION_TIMEOUT_SECONDS",
            "AGENT_RT_EXECUTION_MEMORY_BYTES",
            "AGENT_RT_EXECUTION_CPU_SECONDS",
        )
        previous = {name: os.environ.get(name) for name in names}
        os.environ[names[0]] = "0"
        os.environ[names[1]] = "4096"
        os.environ[names[2]] = "2.5"
        seen_limits = []
        side_effects = []
        try:
            registry = ToolRegistry()

            async def contextual(arguments, context):
                seen_limits.append(context.limits)
                return arguments.get("value")

            registry.register(
                ToolDefinition(
                    name="inspect_limits",
                    description="Inspect limits.",
                    input_schema={"type": "object"},
                    timeout_seconds=1,
                ),
                contextual_handler=contextual,
            )
            value = await registry.execute(
                ToolCall(
                    id="ctx-limits", name="inspect_limits", arguments={"value": "ok"}
                )
            )
            assert value == "ok"
            assert seen_limits[0].timeout_seconds == 1
            assert seen_limits[0].memory_bytes == 4096
            assert seen_limits[0].cpu_seconds == 2.5

            async def slow_handler(arguments, cancellation_token):
                side_effects.append(arguments)
                return "unexpected"

            timeout_registry = ToolRegistry()
            timeout_registry.register(
                ToolDefinition(
                    name="slow_env",
                    description="Slow via env default.",
                    input_schema={"type": "object"},
                ),
                handler=slow_handler,
            )
            call = ToolCall(id="env-timeout", name="slow_env", arguments={})
            provider = FakeProvider(
                [
                    ModelResponse(
                        message=text_message("assistant", "", tool_calls=(call,))
                    ),
                    ModelResponse(message=text_message("assistant", "done")),
                ]
            )
            await AgentLoop(provider, tool_registry=timeout_registry).run(
                self.agent(),
                [text_message("user", "go")],
            )
            assert side_effects == []
            payload = provider.requests[1].messages[-1].content[0].data
            assert payload["error"]["type"] == "tool_timeout"
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    async def test_zero_tool_timeout_prevents_handler_side_effect(self):
        registry = ToolRegistry()
        calls = []

        async def handler(arguments, cancellation_token):
            calls.append(arguments)
            return "unexpected"

        registry.register(
            ToolDefinition(
                name="slow",
                description="Slow.",
                input_schema={"type": "object"},
                timeout_seconds=0,
            ),
            handler=handler,
        )
        call = ToolCall(id="slow-0", name="slow", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )
        assert calls == []

    async def test_shorter_run_timeout_wins_over_tool_timeout(self):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            await asyncio.sleep(10)

        registry.register(
            ToolDefinition(
                name="slow",
                description="Slow.",
                input_schema={"type": "object"},
                timeout_seconds=1,
            ),
            handler=handler,
        )
        call = ToolCall(id="slow-run", name="slow", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )
        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(timeout_seconds=0.01),
        )
        assert result.termination_reason == "timeout"

    async def test_concurrent_tool_calls_overlap_and_preserve_result_order(self):
        registry = ToolRegistry()
        started = []
        both_started = asyncio.Event()

        async def handler(arguments, cancellation_token):
            value = arguments["value"]
            started.append(value)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            if value == "first":
                await asyncio.sleep(0.02)
            return {"value": value}

        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        calls = (
            ToolCall(id="a", name="lookup", arguments={"value": "first"}),
            ToolCall(id="b", name="lookup", arguments={"value": "second"}),
        )
        provider = FakeProvider(
            [
                ModelResponse(message=text_message("assistant", "", tool_calls=calls)),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )

        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(concurrent_tool_calls=True),
        )

        assert sorted(started) == sorted(["first", "second"])
        assert result.tool_calls == 2
        second_request = provider.requests[1]
        tool_messages = [m for m in second_request.messages if m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["a", "b"]
        assert [m.content[0].data["value"] for m in tool_messages] == [
            "first",
            "second",
        ]

    async def test_sequential_tool_is_barrier_inside_concurrent_batch(self):
        registry = ToolRegistry()
        active = 0
        max_active = 0
        first_pair_started = asyncio.Event()
        parallel_started = 0
        sequential_active_counts = []

        async def parallel_handler(arguments, cancellation_token):
            nonlocal active, max_active, parallel_started
            active += 1
            max_active = max(max_active, active)
            parallel_started += 1
            if parallel_started == 2:
                first_pair_started.set()
            await asyncio.wait_for(first_pair_started.wait(), timeout=0.2)
            await asyncio.sleep(0.01)
            active -= 1
            return {"value": arguments["value"]}

        async def sequential_handler(arguments, cancellation_token):
            sequential_active_counts.append(active)
            return {"value": "sequential"}

        registry.register(
            ToolDefinition(
                name="parallel",
                description="Parallel.",
                input_schema={"type": "object"},
                execution_mode="parallel",
            ),
            handler=parallel_handler,
        )
        registry.register(
            ToolDefinition(
                name="serial",
                description="Serial.",
                input_schema={"type": "object"},
                execution_mode="sequential",
            ),
            handler=sequential_handler,
        )

        calls = (
            ToolCall(id="p1", name="parallel", arguments={"value": "one"}),
            ToolCall(id="p2", name="parallel", arguments={"value": "two"}),
            ToolCall(id="s1", name="serial", arguments={}),
        )
        provider = FakeProvider(
            [
                ModelResponse(message=text_message("assistant", "", tool_calls=calls)),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )

        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(concurrent_tool_calls=True),
        )

        assert max_active == 2
        assert sequential_active_counts == [0]
        tool_messages = [m for m in provider.requests[1].messages if m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["p1", "p2", "s1"]

    async def test_tool_selection_policy_filters_visibility_and_forwards_hints(self):
        registry = ToolRegistry()
        for name in ("search", "write", "admin"):
            registry.register(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                )
            )
        agent = AgentConfig(
            name="policy",
            instructions="Use tools.",
            model=ModelSettings(model="fake-model"),
            tool_policy=ToolSelectionPolicy(
                allowed=frozenset({"search", "write"}),
                denied=frozenset({"admin"}),
                required=frozenset({"search"}),
                preferred=("write", "admin"),
            ),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )

        await AgentLoop(provider, tool_registry=registry).run(
            agent,
            [text_message("user", "go")],
        )

        request = provider.requests[0]
        assert [tool.name for tool in request.tools] == ["search", "write"]
        assert request.tool_selection.required == ("search",)
        assert request.tool_selection.preferred == ("write",)

    async def test_selection_policy_blocks_hidden_tool_before_execution(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="admin",
                description="Admin.",
                input_schema={"type": "object"},
            )
        )
        call = ToolCall(id="admin-1", name="admin", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        tools = FakeTools()
        agent = AgentConfig(
            name="policy",
            instructions="Use tools.",
            model=ModelSettings(model="fake-model"),
            tool_policy=ToolSelectionPolicy(denied=frozenset({"admin"})),
        )

        await AgentLoop(provider, tools, registry).run(
            agent,
            [text_message("user", "go")],
        )

        assert tools.calls == []
        payload = provider.requests[1].messages[-1].content[0].data
        assert "not permitted" in payload["error"]["issues"][0]

    async def test_compiled_plan_is_inspectable_and_filters_static_stages(self):
        registry = ToolRegistry()
        for name in ("search", "write"):
            registry.register(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                )
            )
        agent = AgentConfig(
            name="compiled",
            instructions="Use search.",
            model=ModelSettings(model="fake-model"),
            output=AgentOutputRequirements(schema={"type": "object"}),
            tool_policy=ToolSelectionPolicy(
                allowed=frozenset({"search"}),
                required=frozenset({"search"}),
            ),
        )
        loop = AgentLoop(FakeProvider([]), tool_registry=registry)

        plan = loop.compile_plan(
            agent,
            context_policy=ContextSelectionPolicy(
                tool_names=frozenset({"search"}),
            ),
        )

        assert [tool.name for tool in plan.visible_tools] == ["search"]
        assert plan.visible_tool_names == frozenset({"search"})
        assert plan.tool_selection.required == ("search",)
        assert plan.tool_registry_version == registry.version
        assert "tools" in plan.active_stages
        assert "structured_output" in plan.active_stages
        assert not plan.dynamic_tool_filter

    async def test_compiled_plan_invalidates_when_tool_registry_changes_mid_run(self):
        registry = ToolRegistry()

        async def mutate_registry(arguments, cancellation_token):
            registry.register(
                ToolDefinition(
                    name="later",
                    description="Registered during the run.",
                    input_schema={"type": "object"},
                )
            )
            return {"ok": True}

        registry.register(
            ToolDefinition(
                name="mutate",
                description="Mutate registry.",
                input_schema={"type": "object"},
            ),
            handler=mutate_registry,
        )
        initial_version = registry.version
        call = ToolCall(id="mutate-1", name="mutate", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )

        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert result.termination_reason == "completed"
        assert registry.version > initial_version
        assert [tool.name for tool in provider.requests[0].tools] == ["mutate"]
        assert [tool.name for tool in provider.requests[1].tools] == ["mutate", "later"]

    async def test_dynamic_tool_filter_changes_visibility_by_turn_and_context(self):
        registry = ToolRegistry()

        async def handler(arguments, cancellation_token):
            return {"ok": True}

        for name in ("search", "write"):
            registry.register(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                ),
                handler=handler,
            )

        seen_context = []

        def tool_filter(context, tools):
            seen_context.append((context.turn, context.runtime_context["phase"]))
            return ["search"] if context.turn == 0 else ["write"]

        call = ToolCall(id="search-1", name="search", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        loop = AgentLoop(provider, tool_registry=registry, tool_filter=tool_filter)
        await loop.run(
            self.agent(),
            [text_message("user", "go")],
            tool_context={"phase": "runtime"},
        )

        assert [tool.name for tool in provider.requests[0].tools] == ["search"]
        assert [tool.name for tool in provider.requests[1].tools] == ["write"]
        assert seen_context == [(0, "runtime"), (1, "runtime")]

    async def test_deferred_tool_schema_is_not_loaded_until_explicitly_requested(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="eager",
                description="Eager.",
                input_schema={"type": "object"},
            )
        )
        loads = []

        def load_lazy():
            loads.append("lazy")
            return ToolDefinition(
                name="lazy",
                description="Lazy.",
                input_schema={"type": "object"},
            )

        registry.register_deferred("lazy", load_lazy)
        first_provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        await AgentLoop(first_provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert loads == []
        assert [tool.name for tool in first_provider.requests[0].tools] == ["eager"]
        assert registry.deferred_names() == ("lazy",)

        registry.load("lazy")
        second_provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        await AgentLoop(second_provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
        )

        assert loads == ["lazy"]
        assert [tool.name for tool in second_provider.requests[0].tools] == [
            "eager",
            "lazy",
        ]
        assert registry.deferred_names() == ()

    async def test_tool_program_executor_runs_multiple_calls_without_model_round_trip(
        self,
    ):
        registry = ToolRegistry()
        executed = []

        async def handler(arguments, cancellation_token):
            executed.append(arguments["value"])
            return {"value": arguments["value"]}

        registry.register(
            ToolDefinition(
                name="work",
                description="Work.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        calls = (
            ToolCall(id="a", name="work", arguments={"value": 1}),
            ToolCall(id="b", name="work", arguments={"value": 2}),
        )

        results = await ToolProgramExecutor(registry).execute(calls)

        assert executed == [1, 2]
        assert [result.call.id for result in results] == ["a", "b"]
        assert [result.value["value"] for result in results] == [1, 2]

    async def test_tool_program_executor_can_run_calls_concurrently(self):
        registry = ToolRegistry()
        started = []
        release = asyncio.Event()

        async def handler(arguments, cancellation_token):
            started.append(arguments["value"])
            if len(started) == 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=0.2)
            return arguments["value"]

        registry.register(
            ToolDefinition(
                name="work",
                description="Work.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        calls = (
            ToolCall(id="a", name="work", arguments={"value": "first"}),
            ToolCall(id="b", name="work", arguments={"value": "second"}),
        )

        results = await ToolProgramExecutor(registry).execute(
            calls,
            concurrent=True,
        )

        assert sorted(started) == sorted(["first", "second"])
        assert [result.call.id for result in results] == ["a", "b"]
        assert [result.value for result in results] == ["first", "second"]

    async def test_tool_lifecycle_hooks_transform_and_audit_execution(self):
        events = []

        async def pre_call(call, definition):
            events.append(("pre", call.arguments["value"]))
            return ToolCall(
                id=call.id,
                name=call.name,
                arguments={"value": call.arguments["value"] + 1},
            )

        async def post_call(call, definition, value):
            events.append(("post", value))
            return value * 10

        async def audit(event):
            events.append(("audit", event.phase, event.value))

        registry = ToolRegistry(
            ToolLifecycleHooks(
                pre_call=pre_call,
                post_call=post_call,
                audit=audit,
            )
        )

        async def handler(arguments, cancellation_token):
            events.append(("handler", arguments["value"]))
            return arguments["value"]

        registry.register(
            ToolDefinition(
                name="work",
                description="Work.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            ),
            handler=handler,
        )

        value = await registry.execute(
            ToolCall(id="1", name="work", arguments={"value": 2})
        )

        assert value == 30
        assert events == [
            ("pre", 2),
            ("audit", "start", None),
            ("handler", 3),
            ("post", 3),
            ("audit", "success", 30),
        ]

    async def test_tool_lifecycle_error_hook_and_audit_observe_failure(self):
        errors = []
        audits = []

        async def on_error(call, definition, error):
            errors.append((call.id, str(error)))

        async def audit(event):
            audits.append(
                (event.phase, None if event.error is None else str(event.error))
            )

        registry = ToolRegistry(
            ToolLifecycleHooks(
                on_error=on_error,
                audit=audit,
            )
        )

        async def handler(arguments, cancellation_token):
            raise RuntimeError("boom")

        registry.register(
            ToolDefinition(
                name="fragile",
                description="Fails.",
                input_schema={"type": "object"},
                error_behavior="raise",
            ),
            handler=handler,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await registry.execute(ToolCall(id="err", name="fragile", arguments={}))

        assert errors == [("err", "boom")]
        assert audits == [("start", None), ("error", "boom")]

    async def test_contextual_tool_handler_receives_services_and_request_context(self):
        registry = ToolRegistry(services={"client": "svc"})
        seen = []
        token = CancellationToken()

        async def contextual_handler(arguments, context):
            seen.append(
                (
                    context.services["client"],
                    context.request_context["request_id"],
                    context.cancellation_token,
                )
            )
            return (
                f"{context.services['client']}:{context.request_context['request_id']}"
            )

        registry.register(
            ToolDefinition(
                name="contextual",
                description="Uses injected services.",
                input_schema={"type": "object"},
            ),
            contextual_handler=contextual_handler,
        )

        value = await registry.execute(
            ToolCall(id="ctx", name="contextual", arguments={}),
            token,
            {"request_id": "r1"},
        )

        assert value == "svc:r1"
        assert seen == [("svc", "r1", token)]

    async def test_agent_loop_propagates_tool_context_to_contextual_handler(self):
        registry = ToolRegistry(services={"client": "svc"})
        seen = []

        async def contextual_handler(arguments, context):
            seen.append(
                (
                    context.services["client"],
                    context.request_context["request_id"],
                )
            )
            return {"ok": True}

        registry.register(
            ToolDefinition(
                name="contextual",
                description="Uses injected services.",
                input_schema={"type": "object"},
            ),
            contextual_handler=contextual_handler,
        )
        call = ToolCall(id="ctx-loop", name="contextual", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "", tool_calls=(call,))
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )

        await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            [text_message("user", "go")],
            tool_context={"request_id": "loop"},
        )

        assert seen == [("svc", "loop")]

    async def test_context_assembler_isolates_selected_subwork_context(self):
        assembler = ContextAssembler()
        tools = (
            ToolDefinition(
                name="search",
                description="Search.",
                input_schema={"type": "object"},
            ),
            ToolDefinition(
                name="write",
                description="Write.",
                input_schema={"type": "object"},
            ),
        )
        state = WorkflowState(
            state_type="job",
            version=1,
            data={"step": "research"},
        )
        items = (
            ContextItem(
                id="r1",
                kind="retrieved",
                content=(ContentPart(type="text", text="old retrieval"),),
            ),
            ContextItem(
                id="r2",
                kind="retrieved",
                content=(ContentPart(type="text", text="selected retrieval"),),
            ),
            ContextItem(
                id="f1",
                kind="file",
                content=(ContentPart(type="text", text="file excerpt"),),
            ),
            ContextItem(
                id="o1",
                kind="observation",
                content=(ContentPart(type="text", text="observation"),),
            ),
        )
        assembly = assembler.isolate(
            messages=(
                text_message("user", "old"),
                text_message("assistant", "middle"),
                text_message("user", "recent"),
            ),
            tools=tools,
            workflow_state=state,
            context_items=items,
            runtime_metadata={"trace": "abc"},
            policy=ContextSelectionPolicy(
                max_messages=1,
                retrieved_ids=frozenset({"r2"}),
                file_ids=frozenset({"f1"}),
                include_observations=False,
                tool_names=frozenset({"search"}),
            ),
        )

        assert assembly.messages[0].content[0].text == "recent"
        assert [tool.name for tool in assembly.tools] == ["search"]
        assert [item.id for item in assembly.retrieved_data] == ["r2"]
        assert [item.id for item in assembly.files] == ["f1"]
        assert assembly.observations == ()
        assert assembly.workflow_state == state
        assert assembly.metadata == {"trace": "abc"}

    async def test_agent_loop_assembles_only_selected_model_context(self):
        registry = ToolRegistry()
        for name in ("search", "write"):
            registry.register(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                )
            )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        state = WorkflowState(
            state_type="job",
            version=1,
            data={"step": "research"},
        )
        items = (
            ContextItem(
                id="r1",
                kind="retrieved",
                content=(ContentPart(type="text", text="keep me"),),
            ),
            ContextItem(
                id="o1",
                kind="observation",
                content=(ContentPart(type="text", text="hide me"),),
            ),
        )

        result = await AgentLoop(provider, tool_registry=registry).run(
            self.agent(),
            (
                text_message("user", "old"),
                text_message("user", "recent"),
            ),
            workflow_state=state,
            context_items=items,
            context_policy=ContextSelectionPolicy(
                max_messages=1,
                retrieved_ids=frozenset({"r1"}),
                include_observations=False,
                tool_names=frozenset({"search"}),
            ),
            context_metadata={"request_id": "req-1"},
        )

        request = provider.requests[0]
        assert [tool.name for tool in request.tools] == ["search"]
        assert request.metadata["request_id"] == "req-1"
        assert request.messages[0].role == "system"
        assert request.messages[0].content[0].text == "Be concise."
        context_payload = request.messages[1].content[0].data["context"]
        assert context_payload["workflow_state"]["data"]["step"] == "research"
        # Retrieved items default to untrusted: user-role data, not system.
        assert request.messages[2].role == "user"
        untrusted = request.messages[2].content[0].data["untrusted_context"]
        assert untrusted["retrieved_data"][0]["id"] == "r1"
        assert "observations" not in untrusted
        assert request.messages[-1].content[0].text == "recent"
        assert len(result.messages) > len(request.messages) - 1

    async def test_context_compaction_preserves_recent_messages(self):
        assembler = ContextAssembler(
            compaction_policy=ContextCompactionPolicy(
                max_messages=3,
                keep_recent_messages=2,
            )
        )
        messages = tuple(text_message("user", f"message-{index}") for index in range(5))

        assembly = assembler.isolate(messages=messages)

        assert len(assembly.messages) == 3
        # Summaries derive from untrusted content: user-role data, never system.
        assert assembly.messages[0].role == "user"
        assert assembly.messages[0].content[0].text.startswith("[compacted context]")
        assert [message.content[0].text for message in assembly.messages[-2:]] == [
            "message-3",
            "message-4",
        ]

    async def test_context_offloading_replaces_large_inline_payload(self):
        store = InMemoryArtifactStore()
        assembler = ContextAssembler(
            artifact_store=store,
            offload_policy=ContextOffloadPolicy(
                max_inline_characters=80,
            ),
        )
        item = ContextItem(
            id="large",
            kind="retrieved",
            content=(ContentPart(type="text", text="x" * 500),),
        )

        assembly = assembler.isolate(
            messages=(),
            context_items=(item,),
        )

        offloaded = assembly.retrieved_data[0]
        assert offloaded.content[0].type == "file"
        assert offloaded.metadata["offloaded"]
        artifact_id = offloaded.metadata["artifact_id"]
        stored = store.get(artifact_id)
        assert stored["id"] == "large"
        assert stored["content"][0]["text"] == "x" * 500

    async def test_prompt_cache_hint_tracks_stable_prefix(self):
        assembler = ContextAssembler(
            prompt_cache_policy=PromptCachePolicy(namespace="tests")
        )
        tool = ToolDefinition(
            name="search",
            description="Search.",
            input_schema={"type": "object"},
        )
        first = assembler.assemble_request(
            self.agent(),
            (text_message("user", "one"),),
            tools=(tool,),
        )
        second = assembler.assemble_request(
            self.agent(),
            (text_message("user", "different dynamic turn"),),
            tools=(tool,),
        )
        changed_agent = AgentConfig(
            name="test",
            instructions="Different instructions.",
            model=ModelSettings(model="fake-model"),
        )
        changed = assembler.assemble_request(
            changed_agent,
            (text_message("user", "one"),),
            tools=(tool,),
        )

        assert first.prompt_cache is not None
        assert first.prompt_cache.key == second.prompt_cache.key
        assert first.prompt_cache.stable_message_count == 1
        assert first.prompt_cache.key != changed.prompt_cache.key

    async def test_checkpoint_store_persists_and_lists_agent_state(self):
        store = InMemoryCheckpointStore()
        checkpoint = AgentCheckpoint(
            checkpoint_id="cp-1",
            agent_name="test",
            messages=(text_message("user", "saved"),),
            turns=2,
            tool_calls=1,
            total_tokens=9,
            metadata={"reason": "boundary"},
            created_at_ms=100,
        )
        store.save(checkpoint)

        assert store.load("cp-1") == checkpoint
        assert store.list(agent_name="test") == (checkpoint,)
        assert store.delete("cp-1")
        assert store.load("cp-1") is None

    async def test_agent_loop_resumes_checkpoint_history_and_budgets(self):
        first_provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "first"),
                    usage=ModelUsage(total_tokens=5),
                )
            ]
        )
        agent = self.agent()
        first_result = await AgentLoop(first_provider).run(
            agent,
            [text_message("user", "start")],
        )
        state = WorkflowState(
            state_type="job",
            version=1,
            data={"step": "resume"},
        )
        checkpoint = checkpoint_from_result(
            "cp-resume",
            agent,
            first_result,
            workflow_state=state,
        )

        second_provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "continued"),
                    usage=ModelUsage(total_tokens=3),
                )
            ]
        )
        resumed = await AgentLoop(second_provider).run(
            agent,
            [],
            limits=AgentRunLimits(max_turns=4, max_total_tokens=20),
            resume_checkpoint=checkpoint,
        )

        assert resumed.turns == first_result.turns + 1
        assert resumed.total_tokens == first_result.total_tokens + 3
        request = second_provider.requests[0]
        assert request.messages[-1].content[0].text == "first"
        assert "start" in [
            part.text
            for message in request.messages
            for part in message.content
            if part.text is not None
        ]

    async def test_agent_loop_emits_model_tool_and_lifecycle_events(self):
        registry = ToolRegistry()

        async def lookup_handler(arguments, cancellation_token=None):
            return "value"

        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up.",
                input_schema={"type": "object"},
            ),
            handler=lookup_handler,
        )
        call = ToolCall(id="tool-1", name="lookup", arguments={})
        provider = FakeProvider(
            [
                ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=(),
                        tool_calls=(call,),
                    )
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        events = InMemoryEventStore()

        result = await AgentLoop(
            provider,
            tool_registry=registry,
            event_store=events,
        ).run(
            self.agent(),
            [text_message("user", "go")],
            task_id="task-events",
        )

        assert result.termination_reason == "completed"
        types = [event.type for event in events.list("task-events")]
        assert types[0] == "lifecycle_transition"
        assert "model_requested" in types
        assert "model_completed" in types
        assert "tool_requested" in types
        assert "tool_completed" in types
        assert types[-1] == "lifecycle_transition"
        assert events.list("task-events")[-1].payload["to"] == "completed"

    async def test_idempotency_replays_tool_result_without_second_side_effect(self):
        calls = {"count": 0}

        async def handler(arguments, cancellation_token=None):
            calls["count"] += 1
            return {"count": calls["count"]}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="write",
                description="Write.",
                input_schema={"type": "object"},
            ),
            handler=handler,
        )
        idempotency = InMemoryIdempotencyStore()
        call = ToolCall(id="same-call", name="write", arguments={})

        first_provider = FakeProvider(
            [
                ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=(),
                        tool_calls=(call,),
                    )
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        await AgentLoop(
            first_provider,
            tool_registry=registry,
            idempotency_store=idempotency,
        ).run(
            self.agent(),
            [text_message("user", "go")],
            task_id="task-idem",
        )

        second_provider = FakeProvider(
            [
                ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=(),
                        tool_calls=(call,),
                    )
                ),
                ModelResponse(message=text_message("assistant", "done")),
            ]
        )
        await AgentLoop(
            second_provider,
            tool_registry=registry,
            idempotency_store=idempotency,
        ).run(
            self.agent(),
            [text_message("user", "go again")],
            task_id="task-idem",
        )

        assert calls["count"] == 1
        replayed = second_provider.requests[1].messages[-1].content[0].data
        assert replayed == {"count": 1}

    async def test_background_task_reports_progress_and_result(self):
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def runner(report, token):
            report(0.25, "started")
            started.set()
            await asyncio.sleep(0)
            report(0.75, "almost")
            return {"ok": True}

        initial = manager.submit("bg-1", runner)
        assert initial.status == "queued"
        await started.wait()
        current = manager.get("bg-1")
        assert current.status == "running"
        assert current.progress >= 0.25

        final = await manager.wait("bg-1")
        assert final.status == "completed"
        assert final.progress == 1.0
        assert final.result == {"ok": True}

    async def test_background_task_can_be_cancelled(self):
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def runner(report, token):
            report(0.4, "working")
            started.set()
            while not token.is_cancelled:
                await asyncio.sleep(0.01)
            return "ignored"

        manager.submit("bg-cancel", runner)
        await started.wait()
        assert manager.cancel("bg-cancel")
        final = await manager.wait("bg-cancel")
        assert final.status == "canceled"
        assert final.progress < 1.0

    async def test_model_settings_are_forwarded(self):
        agent = AgentConfig(
            name="settings",
            instructions="Use settings.",
            model=ModelSettings(
                model="fake-model",
                temperature=0.25,
                max_output_tokens=321,
                metadata={"trace": "yes"},
            ),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "done"))]
        )
        await AgentLoop(provider).run(agent, [text_message("user", "go")])
        request = provider.requests[0]
        assert request.model == "fake-model"
        assert request.temperature == 0.25
        assert request.max_output_tokens == 321
        assert request.metadata == {"trace": "yes"}

    async def test_usage_falls_back_to_input_plus_output(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    message=text_message("assistant", "done"),
                    usage=ModelUsage(input_tokens=3, output_tokens=4),
                )
            ]
        )
        result = await AgentLoop(provider).run(
            self.agent(),
            [text_message("user", "go")],
        )
        assert result.total_tokens == 7

    async def test_missing_tool_executor_raises(self):
        call = ToolCall(id="1", name="lookup", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )
        with pytest.raises(RuntimeError, match="no tool executor"):
            await AgentLoop(provider).run(
                self.agent(),
                [text_message("user", "go")],
            )

    async def test_stop_between_multiple_tool_calls_prevents_later_execution(self):
        calls = (
            ToolCall(id="1", name="first", arguments={}),
            ToolCall(id="2", name="second", arguments={}),
        )
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=calls))]
        )

        class StopAfterOne(FakeTools):
            async def execute(self, call):
                value = await super().execute(call)
                return value

        tools = StopAfterOne()
        result = await AgentLoop(provider, tools).run(
            self.agent(),
            [text_message("user", "go")],
            stop_requested=lambda: len(tools.calls) >= 1,
        )
        assert result.termination_reason == "stop_requested"
        assert [call.id for call in tools.calls] == ["1"]

    async def test_timeout_during_tool_execution(self):
        call = ToolCall(id="1", name="slow", arguments={})
        provider = FakeProvider(
            [ModelResponse(message=text_message("assistant", "", tool_calls=(call,)))]
        )

        class SlowTools:
            async def execute(self, call):
                await asyncio.sleep(0.05)
                return "late"

        result = await AgentLoop(provider, SlowTools()).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(timeout_seconds=0.001),
        )
        assert result.termination_reason == "timeout"
        assert result.tool_calls == 0

    async def test_timeout(self):
        class SlowProvider:
            name = "slow"

            async def complete(self, request):
                await asyncio.sleep(0.05)
                raise AssertionError("unreachable")

        result = await AgentLoop(SlowProvider()).run(
            self.agent(),
            [text_message("user", "go")],
            limits=AgentRunLimits(timeout_seconds=0.001),
        )
        assert result.termination_reason == "timeout"
