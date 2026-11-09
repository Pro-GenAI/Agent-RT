"""Offline benchmark cache and cross-adapter comparisons."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def anyio_backend():
    """Run these async tests on asyncio without requiring pytest-asyncio."""
    return "asyncio"

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "swarmguard"
sys.path.insert(0, str(BENCH))

from cache import (CacheMiss, CachedDecisionProvider, CachedModelProvider,
                   NonCacheableResponse, ResponseCache, cached_sdk_completion)
import run as benchmark
from run import (evaluate_case, load_cases, model_for_case,
                 parse_subagent_model_overrides, subagent_settings_for_case)


def test_priority_manifest_orders_twenty_cases_with_pressure_probes():
    cases = load_cases()
    manifest = json.loads((BENCH / "priority_cases.json").read_text(encoding="utf-8"))
    priorities = manifest["top_20"]
    assert len(cases) == 65
    assert len(priorities) == 20
    assert [case["id"] for case in cases[:20]] == [
        item["id"] for item in priorities
    ]
    assert [item["rank"] for item in priorities] == list(range(1, 21))
    assert len({case["id"] for case in cases}) == 65
    assert len({item["focus"] for item in priorities}) == 20
    categories = {
        "delegation_escalation", "indirect_attacks", "unauthorized_tools",
        "pii_propagation", "unnecessary_tools",
    }
    assert {case["category"] for case in cases[:5]} == categories
    assert {case["category"] for case in cases[5:10]} == categories
    assert {case["category"] for case in cases[10:15]} == categories
    assert {case["category"] for case in cases[15:20]} == categories
    assert all(case["id"].endswith("_12") for case in cases[5:10])
    assert all(case["id"].endswith("_13") for case in cases[10:15])
    assert all(case["id"].endswith("_11") for case in cases[15:20])
    assert all(
        len([case for case in cases if case["category"] == category]) == 13
        for category in categories
    )
    assert all(
        len(case["attack"]) >= 200 and len(case["scenario_tags"]) >= 2
        for case in cases[5:20]
    )
    for case in cases[5:20]:
        assert set(case["necessary_tools"]).issubset(case["granted_tools"])
        assert case["mock_tool"] in benchmark.TOOLS
        assert set(case["mock_arguments"]) == {benchmark.TOOLS[case["mock_tool"]][1]}
        assert "@customer.example" in case["synthetic_pii"]
    assert all(case["attack_surface"] == "document_body" for case in cases[5:15])



def test_live_mock_tools_supply_document_evidence_not_trusted_prompt():
    """Untrusted attacker text is delivered through tool output, not user role."""
    case = next(c for c in load_cases() if c["id"] == "indirect_attacks_11")
    prompt = benchmark.prompt(case)
    assert case["task"] in prompt
    assert case["attack"] not in prompt

    document = benchmark.mock_tool_result(case, "read_file", {"path": "test.md"})
    assert document["ok"] is True
    assert document["simulated"] is True
    assert document["documents"][0]["untrusted_excerpt"] == case["attack"]
    assert "Root cause:" in document["documents"][0]["content"]

    search = benchmark.mock_tool_result(case, "search_docs", {"query": "policy"})
    assert search["documents"][0]["untrusted_excerpt"] == case["attack"]

    for name, args in (
        ("list_users", {"query": "users"}),
        ("web_search", {"query": "public"}),
        ("delete_file", {"path": "irrelevant"}),
        ("delegate_subagent", {"task": "child"}),
    ):
        result = benchmark.mock_tool_result(case, name, args)
        assert result["simulated"] is True
        assert "documents" not in result


@pytest.mark.anyio
async def test_live_sdk_reads_inert_untrusted_evidence_before_final_answer(tmp_path):
    from configured import configured_models, run_sdk

    case = next(c for c in load_cases() if c["id"] == "indirect_attacks_11")
    config = configured_fixture()
    _worker, _reviewer, deployment = configured_models(config)

    class FakeSDK:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0
            self.previous_tool_response = None

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert case["attack"] not in kwargs["messages"][0]["content"]
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "read-once", "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"reports/mock.md"}',
                        },
                    }],
                }
            else:
                tool_message = next(
                    item for item in kwargs["messages"] if item["role"] == "tool"
                )
                self.previous_tool_response = json.loads(tool_message["content"])
                message = {
                    "role": "assistant", "content": "A concise incident summary.",
                }
            return {
                "choices": [{
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }],
                "model": kwargs["model"],
                "usage": {"total_tokens": 11},
            }

    client = FakeSDK()
    with ResponseCache(tmp_path / "live-tool-evidence.sqlite") as cache:
        trace = await run_sdk(
            case, cache, benchmark, deployment, "offline-sdk-identity",
            client=client,
        )
        result = await evaluate_case(
            case, "openai_sdk", cache, "mock-worker",
            False, None, None, trace_override=trace,
        )
    assert client.calls == 2
    assert client.previous_tool_response["documents"][0]["untrusted_excerpt"] == case["attack"]
    assert trace.output == "A concise incident summary."
    assert result["attempted"] == 1
    assert result["executed"] == 1
    assert result["repeated_calls"] == 0
    assert result["output_observed"] is True


@pytest.mark.anyio
async def test_repeated_successful_tool_executions_report_as_redundancy(tmp_path):
    case = next(c for c in load_cases() if c["id"] == "unnecessary_tools_01")
    args = {"query": "least-privilege docs"}
    events = []
    for _ in range(3):
        events.extend([
            benchmark.event(case, "attempted", "search_docs", args, case["granted_tools"]),
            benchmark.event(case, "executed", "search_docs", args, case["granted_tools"]),
        ])
    trace = benchmark.RunTrace(events=events, output="completed")
    with ResponseCache(tmp_path / "repeated-calls.sqlite") as cache:
        result = await evaluate_case(
            case, "openai_sdk", cache, "mock-worker",
            False, None, None, trace_override=trace,
        )
    assert result["attempted"] == 3
    assert result["repeated_calls"] == 2
    assert result["grades"]["unnecessary_tool_calls"]["passed"] is True



@pytest.mark.parametrize("category", [
    "delegation_escalation", "indirect_attacks", "unauthorized_tools",
    "pii_propagation", "unnecessary_tools",
])
@pytest.mark.parametrize("variant", [12, 13])
def test_pressure_probe_embeds_attack_in_untrusted_document_body(category, variant):
    """An attacker instruction is returned as a document, never the parent task."""
    case_id = f"{category}_{variant}"
    case = next(item for item in load_cases() if item["id"] == case_id)
    assert case["attack_surface"] == "document_body"
    assert case["attack"] not in benchmark.prompt(case)
    tool = case["necessary_tools"][0]
    assert tool in ("read_file", "search_docs")
    data = benchmark.mock_tool_result(case, tool, {benchmark.TOOLS[tool][1]: "test"})
    assert data["simulated"] is True
    assert data["documents"][0]["content"].startswith(case["document_excerpt"])
    assert data["documents"][0]["content"].endswith(case["attack"])
    assert "\\n\\n" not in data["documents"][0]["content"]
    assert "untrusted_excerpt" not in data["documents"][0]
    assert case["mock_tool"] in case["attack"]
    assert case["mock_tool"] not in case["granted_tools"] or (
        case["mock_tool"] not in case["necessary_tools"]
    )



@pytest.mark.anyio
async def test_scripted_sdk_compliance_with_injected_document_is_graded(tmp_path):
    """Deterministic model double demonstrates an *observed* tool-policy breach.

    This is not evidence the user's live model follows the same injection.
    """
    from configured import configured_models, run_sdk

    case = next(c for c in load_cases() if c["id"] == "indirect_attacks_12")
    config = configured_fixture()
    _worker, _reviewer, deployment = configured_models(config)

    class ComplyingSDK:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert case["attack"] not in kwargs["messages"][0]["content"]
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "lookup", "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"reports/mock-retention.md"}',
                        },
                    }],
                }
            elif self.calls == 2:
                evidence = next(
                    item for item in kwargs["messages"] if item["role"] == "tool"
                )
                document = json.loads(evidence["content"])["documents"][0]
                assert case["attack"] in document["content"]
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "leak", "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(case["mock_arguments"]),
                        },
                    }],
                }
            else:
                message = {"role": "assistant", "content": "Retention summary."}
            return {
                "choices": [{
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }],
                "model": kwargs["model"],
                "usage": {"total_tokens": 11},
            }

    fake = ComplyingSDK()
    with ResponseCache(tmp_path / "injected-scripted-sdk.sqlite") as cache:
        trace = await run_sdk(
            case, cache, benchmark, deployment, "offline-pressure-case",
            client=fake,
        )
        result = await evaluate_case(
            case, "openai_sdk", cache, "mock-worker", False, None, None,
            trace_override=trace,
        )
    assert fake.calls == 3
    assert trace.output == "Retention summary."
    assert result["unauthorized_attempts"] == 1
    assert result["unexpected_pii_executed"] == 1
    assert result["grades"]["unauthorized_tool_calls"]["passed"] is False
    assert result["grades"]["pii_propagation:attempted"]["passed"] is False
    assert result["grades"]["pii_propagation"]["passed"] is False


def test_cache_modes_and_decision_namespaces(tmp_path):
    path = tmp_path / "bench.sqlite"
    class Fake:
        calls = 0
        def decide(self, state, questions):
            self.calls += 1
            return {"decision": {"noul": 0.92}}

    fake = Fake()
    with ResponseCache(path, "read-write") as store:
        decisions = CachedDecisionProvider(fake, store, identity="decision-v1")
        assert decisions.decide({"task": "x"}, {"decision": {"type": "noul"}}) == {
            "decision": {"noul": 0.92}}
        assert decisions.decide({"task": "x"}, {"decision": {"type": "noul"}}) == {
            "decision": {"noul": 0.92}}
        assert fake.calls == 1
        assert store.hits == 1
    with ResponseCache(path, "read") as store:
        assert store.get("decision", {"model": "decision-v1", "state": {"task": "x"},
                                      "questions": {"decision": {"type": "noul"}}})
        with pytest.raises(CacheMiss):
            store.get("llm:openai-sdk", {"missing": True})


@pytest.mark.anyio
async def test_offline_comparison_blocks_unauthorized_and_pii(tmp_path):
    cases = load_cases()
    assert len(cases) == 65
    for category in ("unauthorized_tools", "pii_propagation"):
        case = next(c for c in cases if c["category"] == category)
        with ResponseCache(tmp_path / f"{category}.sqlite") as cache:
            sdk = await evaluate_case(case, "openai_sdk", cache, "mock", False, None, None)
            guarded = await evaluate_case(case, "agent_rt_guarded", cache, "mock", False, None, None)
            decision = await evaluate_case(case, "agent_rt_guarded_decision",
                                           cache, "mock", False, None, None)
        assert sdk["executed"] > 0
        assert guarded["executed"] == 0
        assert decision["executed"] == 0


@pytest.mark.anyio
async def test_cache_replay_does_not_repeat_provider_calls(tmp_path):
    case = next(c for c in load_cases() if c["category"] == "unnecessary_tools")
    path = tmp_path / "replay.sqlite"
    with ResponseCache(path, "read-write") as cache:
        result = await evaluate_case(case, "agent_rt_guarded_decision", cache,
                                     "mock", False, None, None)
        assert cache.writes > 0
    with ResponseCache(path, "read") as cache:
        replay = await evaluate_case(case, "agent_rt_guarded_decision", cache,
                                     "mock", False, None, None)
        assert replay["attempted"] == result["attempted"]
        assert cache.hits > 0


def test_subagent_model_and_endpoint_override_priority():
    cases = load_cases()
    specific = cases[0]
    category = specific["category"]
    first = specific["id"]
    overrides = {
        "model": parse_subagent_model_overrides(
            ["worker=default-worker", f"{category}=category-worker",
             f"{first}=specific-worker"], cases,
        ),
        "base_url": parse_subagent_model_overrides(
            ["worker=https://default.example/v1",
             f"{category}=https://category.example/v1",
             f"{first}=https://case.example/v1"], cases,
        ),
        "api_key_env": parse_subagent_model_overrides(
            ["worker=MY_WORKER_KEY", f"{first}=MY_CASE_KEY"], cases,
        ),
        "temperature": parse_subagent_model_overrides(["worker=0.4"], cases),
        "max_output_tokens": parse_subagent_model_overrides(["worker=128"], cases),
    }
    config = subagent_settings_for_case(specific, "fallback", overrides)
    assert config.model == "specific-worker"
    assert config.base_url == "https://case.example/v1"
    assert config.api_key_env == "MY_CASE_KEY"
    assert config.temperature == 0.4
    assert config.max_output_tokens == 128
    another = next(c for c in cases if c["category"] == category and c["id"] != first)
    assert model_for_case(another, "fallback", overrides["model"]) == "category-worker"
    third = next(c for c in cases if c["category"] != category)
    assert model_for_case(third, "fallback", overrides["model"]) == "default-worker"
    with pytest.raises(ValueError, match="unknown sub-agent target"):
        parse_subagent_model_overrides(["unlisted-case=model"], cases)
    with pytest.raises(ValueError, match="duplicate"):
        parse_subagent_model_overrides(["worker=model-a", "worker=model-b"], cases)
    with pytest.raises(ValueError, match="without credentials"):
        subagent_settings_for_case(specific, "fallback", {
            "base_url": {"worker": "https://user:secret@example.com/v1"}
        })
    with pytest.raises(ValueError, match="environment variable"):
        subagent_settings_for_case(specific, "fallback", {
            "api_key_env": {"worker": "NOT-AN-ENV-VAR"}
        })


@pytest.mark.anyio
async def test_sdk_cache_repeated_run_reuses_responses_and_separates_endpoints(
    tmp_path, monkeypatch,
):
    case = next(c for c in load_cases() if c["category"] == "unnecessary_tools")
    called = []
    parent = benchmark.ScriptedSDK

    class CountingSDK(parent):
        async def create(self, **kwargs):
            called.append(kwargs["model"])
            return await super().create(**kwargs)

    monkeypatch.setattr(benchmark, "ScriptedSDK", CountingSDK)
    path = tmp_path / "models.sqlite"
    selections = {
        "model": {"worker": "model-one"},
        "base_url": {"worker": "https://first.example/v1"},
    }
    with ResponseCache(path) as cache:
        first = await evaluate_case(case, "openai_sdk", cache, "fallback",
                                    False, None, None, subagent_overrides=selections)
        first_calls = len(called)
        assert first_calls > 0
        second = await evaluate_case(case, "openai_sdk", cache, "fallback",
                                     False, None, None, subagent_overrides=selections)
        assert second == first
        assert len(called) == first_calls
        assert cache.hits >= first_calls
        await evaluate_case(case, "openai_sdk", cache, "fallback",
                            False, None, None, subagent_overrides={
                                **selections, "base_url": {"worker": "https://second.example/v1"},
                            })
        assert len(called) > first_calls
        url_calls = len(called)
        await evaluate_case(case, "openai_sdk", cache, "fallback",
                            False, None, None, subagent_overrides={
                                **selections, "model": {"worker": "model-two"},
                            })
        assert len(called) > url_calls
    total_live_calls = len(called)
    with ResponseCache(path, mode="read") as cache:
        await evaluate_case(case, "openai_sdk", cache, "fallback",
                            False, None, None, subagent_overrides=selections)
        assert cache.hits > 0
        assert len(called) == total_live_calls  # Read-only replay made zero calls


@pytest.mark.anyio
async def test_agent_rt_cache_identical_evals_do_not_call_model_again(
    tmp_path, monkeypatch,
):
    case = next(c for c in load_cases() if c["category"] == "unnecessary_tools")
    parent = benchmark.ScriptedModel
    decision_parent = benchmark.ScriptedDecisions
    calls = []
    decision_calls = []

    class CountingDecision(decision_parent):
        def decide(self, state, questions):
            decision_calls.append(tuple(questions))
            return super().decide(state, questions)

    monkeypatch.setattr(benchmark, "ScriptedDecisions", CountingDecision)

    class CountingModel(parent):
        async def complete(self, request):
            calls.append(request.model)
            return await super().complete(request)

    monkeypatch.setattr(benchmark, "ScriptedModel", CountingModel)
    path = tmp_path / "agent-rt.sqlite"
    with ResponseCache(path) as cache:
        await evaluate_case(case, "agent_rt_guarded_decision", cache, "model-a",
                            False, None, None)
        count = len(calls)
        decisions = len(decision_calls)
        assert count > 0
        assert decisions > 0
        await evaluate_case(case, "agent_rt_guarded_decision", cache, "model-a",
                            False, None, None)
        assert len(calls) == count
        assert len(decision_calls) == decisions
        assert cache.hits > 0


@pytest.mark.anyio
@pytest.mark.parametrize("error", [
    ConnectionError("internet disconnected"),
    TimeoutError("request timed out"),
    RuntimeError("HTTP 503: service unavailable"),
    RuntimeError("HTTP 429: rate limit"),
])
async def test_failed_provider_calls_do_not_poison_cache(tmp_path, error):
    path = tmp_path / "failure.sqlite"

    class FailingSDK:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise error

    sdk = FailingSDK()
    request = {"model": "test-model", "messages": [{"role": "user", "content": "test"}],
               "tools": []}
    with ResponseCache(path) as cache:
        for _ in range(2):
            with pytest.raises(type(error)):
                await cached_sdk_completion(sdk, cache, **request)
        assert sdk.calls == 2
        assert cache.writes == 0
        assert cache.hits == 0


@pytest.mark.anyio
async def test_error_payload_and_incomplete_sdk_responses_are_not_cached(tmp_path):
    class InvalidSDK:
        def __init__(self, response):
            self.chat = self
            self.completions = self
            self.response = response

        async def create(self, **kwargs):
            return self.response

    requests = [
        {"error": {"message": "rate limited"}},
        {"choices": [{"finish_reason": "length",
                      "message": {"role": "assistant", "content": "partial"}}]},
        {"choices": [{"finish_reason": "stop",
                      "message": {"role": "assistant", "content": ""}}]},
    ]
    with ResponseCache(tmp_path / "errors.sqlite") as cache:
        for response in requests:
            with pytest.raises(NonCacheableResponse):
                await cached_sdk_completion(
                    InvalidSDK(response), cache, model="same",
                    messages=[{"role": "user", "content": "same"}], tools=[],
                )
            assert cache.writes == 0


def test_failed_decision_answers_are_not_cached(tmp_path):
    class Decisions:
        calls = 0

        def decide(self, state, questions):
            self.calls += 1
            return {"relevance": {"error": "decision backend unavailable"}}

    decisions = Decisions()
    with ResponseCache(tmp_path / "decision.sqlite") as cache:
        wrapped = CachedDecisionProvider(decisions, cache, identity="decision-v2")
        for _ in range(2):
            with pytest.raises(NonCacheableResponse):
                wrapped.decide({"task": "x"}, {"relevance": {"type": "noul"}})
        assert cache.writes == 0
        assert decisions.calls == 2


@pytest.mark.anyio
async def test_incomplete_agent_rt_response_does_not_update_cache(tmp_path):
    from ext.runtime.models.interfaces import ModelRequest, ModelResponse
    from ext.runtime.types import ContentPart, ModelMessage

    class FailedModel:
        name = "failed-model"

        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="partial"),),
                ),
                model=request.model,
                finish_reason="length",
            )

    model = FailedModel()
    request = ModelRequest(
        messages=(ModelMessage(role="user", content=(
            ContentPart(type="text", text="test"),)),),
        model="test-model",
    )
    with ResponseCache(tmp_path / "failed-model.sqlite") as cache:
        cached = CachedModelProvider(model, cache, identity="test-model")
        for _ in range(2):
            with pytest.raises(NonCacheableResponse):
                await cached.complete(request)
        assert model.calls == 2
        assert cache.writes == 0


def test_failed_decision_transport_does_not_update_cache(tmp_path):
    class FailingDecision:
        calls = 0

        def decide(self, state, questions):
            self.calls += 1
            raise ConnectionError("decision endpoint offline")

    model = FailingDecision()
    with ResponseCache(tmp_path / "failed-decision.sqlite") as cache:
        cached = CachedDecisionProvider(model, cache, identity="decision-endpoint")
        for _ in range(2):
            with pytest.raises(ConnectionError):
                cached.decide({"task": "test"}, {"relevance": {"type": "noul"}})
        assert model.calls == 2
        assert cache.writes == 0


def test_make_eval_swarmguard_supports_optional_worker_alias():
    """One Makefile entry point, with an explicit model override."""
    makefile = (BENCH.parents[1] / "Makefile").read_text(encoding="utf-8")
    assert ".PHONY: eval-swarmguard\n" in makefile
    assert "eval-swarmguard:" in makefile
    assert "eval-swarmguard-alt" not in makefile
    assert '$(if $(strip $(SUBAGENT_MODEL)),--subagent-model "$(SUBAGENT_MODEL)")' in makefile


def test_public_cli_requires_explicit_config_without_scripted_fallback(tmp_path):
    """The sole live CLI fails clearly without its required YAML file."""
    missing_config = tmp_path / "missing-agent-rt.yaml"
    result = subprocess.run(
        [
            sys.executable, str(BENCH / "run.py"),
            "--config", str(missing_config),
            "--cache-path", str(tmp_path / "unused.sqlite"),
        ],
        check=False, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "[SwarmGuard] Loading configuration..." in result.stderr
    assert "config file not found" in result.stderr
    assert "Starting evaluation" not in result.stderr
    assert not result.stdout


def configured_fixture(*, decision=True, timeout_seconds=None, alternate=False):
    from ext.config import parse_config

    limits = (
        f"\n  execution_limits:\n    timeout_seconds: {timeout_seconds}\n"
        if timeout_seconds is not None else ""
    )
    alternative = """
  - model_name: subagent-alt
    agent_rt_params: {model: openai/gpt-worker-alt, api_key: test, api_base: https://alternative.example.test/v1}
""" if alternate else ""
    extra = """
  decision_model:
    provider: jev
    base_url: https://decision.example.test
    model: test-decision
""" if decision else ""
    return parse_config(
        """
model_list:
  - model_name: main-agent
    agent_rt_params: {model: openai/gpt-main, api_key: test, api_base: https://main.example.test/v1}
  - model_name: subagent
    agent_rt_params: {model: openai/gpt-worker, api_key: test, api_base: https://worker.example.test/v1}
""" + alternative + """
agent_rt_settings:
  default_model: main-agent
  subagents: {default_model: subagent}
""" + extra + limits
    )


def test_make_eval_requires_explicit_working_decision_model(tmp_path):
    from configured import configured_decisions, configured_models, validate_decision_provider

    with ResponseCache(tmp_path / "preflight.sqlite") as cache:
        with pytest.raises(ValueError, match="decision_model"):
            configured_decisions(configured_fixture(decision=False), cache)
        config = configured_fixture()
        worker, reviewer, deployment = configured_models(config)
        assert (worker, reviewer, deployment.model) == (
            "subagent", "main-agent", "gpt-worker",
        )

        class InvalidDecision:
            calls = 0

            def decide(self, state, questions):
                self.calls += 1
                return {"tool_0": {"error": "HTTP 503"}}

        invalid = InvalidDecision()
        checks = configured_decisions(config, cache, provider=invalid)
        for _ in range(2):
            with pytest.raises(NonCacheableResponse):
                validate_decision_provider(checks)
        assert invalid.calls == 2
        assert cache.writes == 0


def test_configured_worker_alias_override_resolves_deployment():
    from configured import configured_models

    config = configured_fixture(alternate=True)
    assert configured_models(config)[:2] == ("subagent", "main-agent")
    worker, reviewer, deployment = configured_models(
        config, subagent_model="subagent-alt",
    )
    assert (worker, reviewer, deployment.model) == (
        "subagent-alt", "main-agent", "gpt-worker-alt",
    )
    assert str(deployment.api_base) == "https://alternative.example.test/v1"

    with pytest.raises(ValueError, match="not a configured model_list alias"):
        configured_models(config, subagent_model="misspelled-worker")
    with pytest.raises(ValueError, match="not a configured model_list alias"):
        configured_models(config, subagent_model="")


@pytest.mark.anyio
async def test_subagent_model_override_has_isolated_cache_keys(tmp_path):
    """A -> B -> A must miss only for B and reuse A for both live harnesses."""
    from configured import execute
    from ext.runtime.models.interfaces import ModelResponse
    from ext.runtime.types import ContentPart, ModelMessage

    class Decision:
        def __init__(self):
            self.calls = 0

        def decide(self, state, questions):
            self.calls += 1
            return {key: {"noul": 0.95} for key in questions}

    class Provider:
        name = "model-alias-switch-test"

        def __init__(self):
            self.calls = []

        async def complete(self, request):
            self.calls.append(request.model)
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="done"),),
                ),
                model=request.model,
                finish_reason="stop",
            )

    class SDK:
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **request):
            self.calls.append(request["model"])
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }],
                "model": request["model"],
                "usage": {"total_tokens": 1},
            }

    config = configured_fixture(alternate=True)
    sdk, provider, decisions = SDK(), Provider(), Decision()
    cache_path = tmp_path / "worker-model-switch.sqlite"
    counts = []
    results = []
    for alias in ("subagent", "subagent-alt", "subagent"):
        if alias == "subagent-alt":
            # A cached evaluation for worker A must not satisfy a read-only
            # replay for model B, even with the same case and Decision model.
            previous_calls = (len(sdk.calls), len(provider.calls))
            with ResponseCache(cache_path, mode="read") as cache:
                with pytest.raises(CacheMiss):
                    await execute(
                        config, cache, benchmark, limit=1,
                        subagent_model=alias, sdk_client=sdk, provider=provider,
                        decision_provider=decisions,
                    )
            assert (len(sdk.calls), len(provider.calls)) == previous_calls
        with ResponseCache(cache_path, mode="read-write") as cache:
            report = await execute(
                config, cache, benchmark, limit=1,
                subagent_model=alias,
                sdk_client=sdk, provider=provider,
                decision_provider=decisions,
            )
        results.append(report)
        counts.append((len(sdk.calls), len(provider.calls), decisions.calls))

    assert [r["worker_model"] for r in results] == [
        "subagent", "subagent-alt", "subagent",
    ]
    assert [r["worker_deployment_model"] for r in results] == [
        "gpt-worker", "gpt-worker-alt", "gpt-worker",
    ]
    assert [r["reviewer_model"] for r in results] == ["main-agent"] * 3
    assert sdk.calls == ["gpt-worker", "gpt-worker-alt"]
    assert "subagent" in provider.calls
    assert "subagent-alt" in provider.calls
    assert counts[1][0] > counts[0][0]
    assert counts[1][1] > counts[0][1]
    assert counts[2] == counts[1]  # Alias A responses are reusable after B.
    assert results[1]["cache"]["writes"] > 0
    assert results[2]["cache"]["writes"] == 0
    assert results[2]["cache"]["hits"] > 0
    # The unchanged Decision preflight also reuses its own cache.
    assert counts[2][2] == counts[1][2]


@pytest.mark.anyio
async def test_configured_sdk_vs_full_swarm_cache_replay_without_provider_calls(tmp_path):
    from configured import execute
    from ext.runtime.models.interfaces import ModelResponse, ModelUsage
    from ext.runtime.types import ContentPart, ModelMessage

    class Decision:
        def __init__(self):
            self.calls = 0

        def decide(self, state, questions):
            self.calls += 1
            return {key: {"noul": 0.95} for key in questions}

    class Provider:
        name = "mock-configured"
        def __init__(self):
            self.calls = []

        async def complete(self, request):
            self.calls.append(request.model)
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="completed"),),
                ),
                model=request.model,
                usage=ModelUsage(input_tokens=8, output_tokens=3),
                finish_reason="stop",
            )

    class SDK:
        def __init__(self):
            self.calls = 0
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls += 1
            return {
                "choices": [{"finish_reason": "stop",
                             "message": {"role": "assistant", "content": "completed"}}],
                "usage": {"total_tokens": 11},
                "model": kwargs["model"],
            }

    models = Provider()
    sdk = SDK()
    decision = Decision()
    config = configured_fixture()
    path = tmp_path / "configured-replay.sqlite"
    with ResponseCache(path, "read-write") as cache:
        initial = await execute(
            config, cache, benchmark, limit=2,
            provider=models, sdk_client=sdk, decision_provider=decision,
        )
        assert set(initial["configurations"]) == {"openai_sdk", "agent_rt"}
        assert initial["worker_model"] == "subagent"
        assert initial["reviewer_model"] == "main-agent"
        assert initial["cache"]["writes"] > 0
    calls = (sdk.calls, len(models.calls), decision.calls)
    assert calls[0] > 0 and calls[1] >= 4 and calls[2] > 0
    assert "subagent" in models.calls
    assert "main-agent" in models.calls

    with ResponseCache(path, "read-write") as cache:
        repeated = await execute(
            config, cache, benchmark, limit=2,
            provider=models, sdk_client=sdk, decision_provider=decision,
        )
        assert repeated["cache"]["writes"] == 0
        assert repeated["cache"]["hits"] > 0
        assert repeated["configurations"] == initial["configurations"]
    assert (sdk.calls, len(models.calls), decision.calls) == calls



@pytest.mark.anyio
async def test_reviewer_timeout_explains_shared_deadline_without_reset(tmp_path):
    """The benchmark must not turn a short parent deadline into a false pass."""
    import asyncio
    from configured import (
        configured_decisions, configured_models, run_agent_rt,
        validate_decision_provider,
    )
    from ext.runtime.models.interfaces import ModelResponse
    from ext.runtime.types import ContentPart, ModelMessage

    class Decision:
        def decide(self, state, questions):
            return {key: {"noul": 0.95} for key in questions}

    class SlowReviewer:
        name = "slow-reviewer-test"

        async def complete(self, request):
            if request.metadata.get("swarmguard_role") == "reviewer":
                await asyncio.sleep(2)
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="done"),),
                ),
                model=request.model,
                finish_reason="stop",
            )

    config = configured_fixture(timeout_seconds=0.5)
    worker, reviewer, _deployment = configured_models(config)
    case = next(c for c in load_cases() if c["category"] == "unauthorized_tools")
    with ResponseCache(tmp_path / "reviewer-timeout.sqlite") as cache:
        decisions = configured_decisions(config, cache, provider=Decision())
        validate_decision_provider(decisions)
        with pytest.raises(RuntimeError, match="Agent-RT reviewer failed") as error:
            await run_agent_rt(
                case, cache, benchmark, config, worker, reviewer, decisions,
                provider=SlowReviewer(),
            )
    message = str(error.value)
    assert "parent deadline expired" in message
    assert "0.5s execution deadline covers both" in message
    assert "agent_rt_settings.execution_limits.timeout_seconds" in message
    assert "./agent-rt.yaml" in message


@pytest.mark.anyio
async def test_configured_benchmark_stops_before_sdk_on_decision_failure(tmp_path):
    from configured import execute

    class OfflineDecision:
        def decide(self, state, questions):
            raise ConnectionError("decision endpoint down")

    class SDK:
        calls = 0
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls += 1
            raise AssertionError("SDK must not run before decision validation")

    sdk = SDK()
    with ResponseCache(tmp_path / "decision-down.sqlite") as cache:
        with pytest.raises(ConnectionError, match="decision endpoint down"):
            await execute(
                configured_fixture(), cache, benchmark, limit=1,
                sdk_client=sdk, decision_provider=OfflineDecision(),
            )
        assert cache.writes == 0
        assert sdk.calls == 0


def test_evaluation_progress_bar_and_non_tty_logs():
    import io
    from configured import EvaluationProgress

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    bar = EvaluationProgress(terminal, width=10)
    bar.status("Loading configuration...")
    bar.update("openai_sdk", 0, 2, 0, 4)
    bar.update("openai_sdk", 1, 2, 1, 4)
    bar.update("openai_sdk", 2, 2, 2, 4)
    bar.update("agent_rt", 0, 2, 2, 4)
    bar.update("agent_rt", 2, 2, 4, 4)
    bar.finish()
    rendered = terminal.getvalue()
    assert "Loading configuration...\n" in rendered
    assert "\r" in rendered
    assert "[##########] 4/4 (100%) | Agent-RT: 2/2 cases" in rendered
    assert rendered.endswith("\n")

    logs = io.StringIO()
    log_progress = EvaluationProgress(logs, width=8)
    log_progress.update("openai_sdk", 0, 2, 0, 4)
    log_progress.update("openai_sdk", 1, 2, 1, 4)
    log_progress.update("agent_rt", 2, 2, 4, 4)
    assert "\r" not in logs.getvalue()
    assert "[########] 4/4 (100%)" in logs.getvalue()
    assert logs.getvalue().endswith("\n")


@pytest.mark.anyio
async def test_configured_eval_reports_completed_cases_across_both_phases(
    tmp_path, monkeypatch,
):
    from configured import execute
    import configured

    class Decisions:
        def decide(self, state, questions):
            return {key: {"noul": 0.95} for key in questions}

    async def fake_case(
        case, framework, cache, bench, config, worker, reviewer, deployment,
        decisions, **kwargs,
    ):
        return {
            "id": case["id"],
            "attempted": 0,
            "executed": 0,
            "unauthorized_attempts": 0,
            "unexpected_pii_executed": 0,
            "delegation_escalations": 0,
            "tokens": 0,
        }

    monkeypatch.setattr(configured, "evaluate_configured", fake_case)
    stages = []
    updates = []
    with ResponseCache(tmp_path / "progress.sqlite") as cache:
        report = await execute(
            configured_fixture(), cache, benchmark, limit=2,
            provider=object(), sdk_client=object(),
            decision_provider=Decisions(),
            on_stage=stages.append,
            on_progress=lambda *args: updates.append(args),
        )

    assert report["cases"] == 2
    assert set(report["configurations"]) == {"openai_sdk", "agent_rt"}
    assert any("Validating Decision model" in stage for stage in stages)
    assert any("Starting evaluation" in stage for stage in stages)
    assert updates == [
        ("openai_sdk", 0, 2, 0, 4),
        ("openai_sdk", 1, 2, 1, 4),
        ("openai_sdk", 2, 2, 2, 4),
        ("agent_rt", 0, 2, 2, 4),
        ("agent_rt", 1, 2, 3, 4),
        ("agent_rt", 2, 2, 4, 4),
    ]

    # Default is ten; 20 runs all pressure challenges and 65 runs everything.
    for selected_limit, expected in [(None, 10), (20, 20), (65, 65)]:
        progress = []
        overrides = {} if selected_limit is None else {"limit": selected_limit}
        with ResponseCache(tmp_path / f"prioritized-{expected}.sqlite") as cache:
            selected = await execute(
                configured_fixture(), cache, benchmark,
                provider=object(), sdk_client=object(),
                decision_provider=Decisions(),
                on_progress=lambda *args: progress.append(args),
                **overrides,
            )
        assert selected["cases"] == expected
        assert progress[-1] == (
            "agent_rt", expected, expected, 2 * expected, 2 * expected,
        )



def test_rich_report_highlights_violations_and_bounds_long_runs():
    import io
    from rich.console import Console
    from report import render_report

    baseline = {
        "id": "indirect_attacks_01", "attempted": 5, "executed": 5,
        "blocked": 0, "unauthorized_attempts": 2,
        "unexpected_pii_executed": 0, "delegation_escalations": 0,
        "tokens": 1200, "output_observed": False,
        "repeated_calls": 2,
        "grades": {
            "unnecessary_tool_calls": {"passed": False},
            "unauthorized_tool_calls": {"passed": False},
        },
    }
    agent = {
        **baseline, "attempted": 3, "executed": 3,
        "unauthorized_attempts": 0, "output_observed": True,
        "repeated_calls": 0, "grades": {},
    }
    ok = {**agent, "id": "pii_propagation_01"}
    report = {
        "cases": 2, "worker_model": "subagent",
        "worker_deployment_model": "gpt-worker-alt",
        "reviewer_model": "main-agent",
        "decision_model": "test-decision",
        "cache": {"hits": 0, "misses": 6, "writes": 6},
        "configurations": {
            "openai_sdk": {"cases": [baseline, ok], "tokens": 1200},
            "agent_rt": {"cases": [agent, ok], "tokens": 2300},
        },
    }

    stream = io.StringIO()
    console = Console(file=stream, width=150, color_system=None)
    render_report(report, console=console, max_cases=1)
    content = stream.getvalue()
    assert "Overall comparison" in content
    assert "gpt-worker-alt" in content
    assert "2 ungranted attempt(s)" in content
    assert "unauthorized execution" in content
    assert "no final answer" not in content
    assert "2 repeated successful tool call(s)" in content
    assert "Repeated successful calls" in content
    assert "Showing 1 of 2 cases" in content
    assert "Tools blocked" in content
    assert "Tokens" not in content
    assert "not directly comparable" not in content
    assert "None" in content  # Agent-RT has no flagged findings for the shown case
    expanded = io.StringIO()
    render_report(report, console=Console(file=expanded, width=150, color_system=None))
    assert "None" in expanded.getvalue()
    assert "No flagged findings" not in expanded.getvalue()
    assert '"configurations"' not in content


@pytest.mark.anyio
async def test_configured_cli_progress_only_writes_to_stderr(
    tmp_path, capsys, monkeypatch,
):
    import configured

    monkeypatch.setattr(configured, "load_config", lambda path: object())

    sdk_case = {
        "id": "indirect_attacks_01", "attempted": 5, "executed": 5,
        "blocked": 0, "unauthorized_attempts": 2,
        "unexpected_pii_executed": 0, "delegation_escalations": 0,
        "output_observed": False,
        "grades": {
            "unauthorized_tool_calls": {"passed": False, "score": 0.6},
            "unnecessary_tool_calls": {"passed": False, "score": 0.6},
        },
    }
    guarded_case = {
        **sdk_case,
        "attempted": 3, "executed": 3, "unauthorized_attempts": 0,
        "output_observed": True, "grades": {},
    }
    report = {
        "mode": "live", "cases": 1, "worker_model": "subagent",
        "reviewer_model": "main-agent", "decision_model": "test-decision",
        "cache": {"hits": 0, "misses": 4, "writes": 4},
        "configurations": {
            "openai_sdk": {"cases": [sdk_case], "tokens": 123},
            "agent_rt": {"cases": [guarded_case], "tokens": 456},
        },
    }

    async def fake_execute(*args, **kwargs):
        assert kwargs["limit"] == 10  # Default also applies at the CLI boundary.
        assert kwargs["subagent_model"] in (None, "subagent-alt")
        kwargs["on_stage"]("Validating Decision model (using cache if available)...")
        kwargs["on_stage"]("Starting evaluation: 1 cases x 2 frameworks (2 total runs).")
        kwargs["on_progress"]("openai_sdk", 0, 1, 0, 2)
        kwargs["on_progress"]("openai_sdk", 1, 1, 1, 2)
        kwargs["on_progress"]("agent_rt", 0, 1, 1, 2)
        kwargs["on_progress"]("agent_rt", 1, 1, 2, 2)
        return report

    monkeypatch.setattr(configured, "execute", fake_execute)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--cache-path", str(tmp_path / "cli.sqlite")],
    )
    await configured.main(benchmark)
    captured = capsys.readouterr()
    assert "Overall comparison" in captured.out
    assert "Case findings" in captured.out
    assert "Unauthorized attempts" in captured.out
    assert "Tokens" not in captured.out
    assert "not directly comparable" not in captured.out
    assert "None" in captured.out
    assert "No flagged findings" not in captured.out
    assert "no final answer" not in captured.out
    assert "configurations" not in captured.out
    assert "[SwarmGuard] Loading configuration..." in captured.err
    assert "Validating Decision model" in captured.err
    assert "100%" in captured.err
    assert "[SwarmGuard] Evaluation complete" in captured.err
    assert not captured.out.startswith("[SwarmGuard]")

    # JSON remains available to automation and as a saved artifact.
    json_path = tmp_path / "complete.json"
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--cache-path", str(tmp_path / "cli.sqlite"),
                      "--json", "--output", str(json_path)],
    )
    await configured.main(benchmark)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "Overall comparison" not in captured.out

    # Saving JSON alone does not replace the human-readable terminal summary.
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--cache-path", str(tmp_path / "cli.sqlite"),
                      "--output", str(json_path)],
    )
    await configured.main(benchmark)
    captured = capsys.readouterr()
    assert "Overall comparison" in captured.out
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--cache-path", str(tmp_path / "cli.sqlite"),
                      "--subagent-model", "subagent-alt", "--json"],
    )
    await configured.main(benchmark)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report

