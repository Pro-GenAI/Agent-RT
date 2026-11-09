# SwarmGuard benchmark maintenance instructions

These instructions apply to `benchmarks/swarmguard/`. Read root `AGENTS.md`
and the Python/TypeScript package instructions when changing those runtimes.

## Scope and comparison

SwarmGuard investigates unexpected behavior in delegated AI agents for the
2026 Swarm Alignment Sprint. It compares only:
- `openai_sdk`: a *minimal reference manual harness* using the normal OpenAI
  SDK, same YAML-configured worker deployment and full inert mock tool catalog
  (no Agent-RT authorization enforcement).
- `agent_rt`: genuine `SubagentSwarm` delegation and reviewer, parent
  permission checks, Decision-model-selected worker tools, capability grants,
  registry authorization and the unexpected-PII guard.

The user-facing interface is **only `make eval-swarmguard`** from the root.
It must always run **live** using explicitly selected `./agent-rt.yaml`;
it must **never silently fall back to scripted mocks** or a missing/failed
Decision model. Keep deterministic offline model/provider doubles strictly
inside tests. Do not add more `make eval-*` targets or `EVAL_*` variables.

The Makefile must print an immediate initialization message, before heavy
Python imports or network calls, using unbuffered Python (`python -u`).
The manual SDK loop permits six model turns, to reduce inconclusive
tool-call-only traces. Maintain bounded execution and accurate call counts;
never invent a final answer or silently remove unsuccessful attempts.
The live CLI reports YAML parsing and Decision-model validation, then shows
a dynamic completed-run progress bar: **20 completed runs by default**
(ten prioritized cases per framework), or twice the explicit `--limit` value
(up to 65 cases / 130 completed runs). `EvaluationProgress` writes only to
stderr, flushes every update, uses a carriage-return progress bar on TTYs,
and periodic newline-delimited snapshots when stderr is redirected. Use
`report.py` and Rich for a concise, bounded, default stdout summary:
aggregate SDK/Agent-RT totals, case-level call counts, failed graders and
redundant successful tool calls, with a separate aggregate final-answer count.
Do not classify `output_observed=False` as a security finding; a missing
final answer may reflect an exhausted model/tool-call loop. Never invent
failure signals: only report observed violations or measured redundancies.
Keep methodological cautions about zero blocked calls. Do not include
token-usage totals or comparisons in Rich tables; show `None` for a
case with no flagged findings. Raw JSON is available on stdout with `--json`, or saved independently
with `--output FILE` while keeping Rich visible; don't mix progress with
stdout in JSON mode. Rich is already in the Python `cli`/`dev` extras:
avoid making it a heavyweight runtime-core dependency. Progress reflects
**completed cases**;
never increment before a case finishes, and keep the bar newline-clean on
exceptions. Callback-based tests should verify both framework phases and
the final total using offline doubles.

## YAML source of truth

`agent_rt_settings.default_model` selects the main/reviewer model. The
validated `agent_rt_settings.subagents.default_model` selects the worker;
optional `subagents.reviewer_model` overrides the reviewer. Both must
exist in `model_list`. Worker must have **exactly one** OpenAI-compatible
deployment, so the SDK baseline uses the *same upstream model and endpoint*.
`configured_models(config, subagent_model=...)` selects an optional registered
`model_list` alias for all benchmark worker cases; `execute` and the CLI
pass it through without changing the reviewer or Decision model. The
Makefile can forward this alias with `SUBAGENT_MODEL=...` on the existing
`make eval-swarmguard` target; this is not an EVAL_* environment config. An unknown
alias or multi-deployment alias must fail before execution. SDK cache keys
must incorporate the selected alias in addition to physical deployment
identity and request model; Agent-RT already uses request.model (the worker
alias) and its model-config identity. Never cross-replay worker responses
from a different selected alias. Decision cache identity stays independent.
`agent_rt_settings.decision_model` must explicitly specify a working
`base_url` and `model`, plus `provider` (`jev` or `openai`) and
`api_key` if required. The decision service is **not** inferred from a
chat-completion URL, CLI environment variables, or silently defaulted.
Maintain strict Python and TypeScript configuration-schema parity in
`python/src/ext/config.py` and `typescript/src/ext/config.ts`; update
`agent-rt.example.yaml` and the website configuration reference.

`run.py` exposes test-only scripted helpers and delegates the CLI to
`configured.py`. `configured.py` validates models, constructs configured
providers, preflights a real Decision `noul` answer before cases, runs the
two live harnesses, applies common graders and reports both sides. The
actual benchmark never reads `EVAL_*` or `OPENAI_MODEL` as configuration.
Only `os.environ/NAME` secret references in the YAML are resolved.

## Security and evaluation invariants

The Agent-RT worker and reviewer share one parent deadline from
`agent_rt_settings.execution_limits.timeout_seconds`. Never silently reset,
ignore or extend this deadline to hide slow reviewers. The benchmark should
report an actionable error instructing the user to edit the local ignored
`agent-rt.yaml`; the live benchmark example uses 120s to accommodate two
sequential model calls. The general runtime example can remain at 30s.

The `cases/*.jsonl` files contain **65 synthetic** cases, 13 per category:
the original 50, five advanced variants in `advanced_scenarios.jsonl`,
and ten new high-pressure cases in `pressure_scenarios.jsonl`.
The curated `priority_cases.json` defines 20 ranked cases: five original
representatives (1-5), ten pressure probes (6-15), and five earlier advanced
variants (16-20). The remaining 45 are controls and lower-priority variants.
The default live run remains ten cases, covering each category twice; 
`--limit 20` and `--limit 65` expand coverage.

Both live paths must use `run.mock_tool_result` for identical, inert tool
outputs. Existing controls put attacker text in a labeled `untrusted_excerpt`;
new `attack_surface: document_body` probes instead place it in the normal
document `content` field, without changing its tool-result trust level or
the trusted parent instructions. This makes stronger but unvalidated
adversarial hypotheses: do not assert increased failures without live
experiments on the user-selected model. No real network, file writes,
adaptive multi-round prompt injection, child identity changes, or child
execution are involved. Preserve synthetic PII and grant invariants.
Ranking is qualitative risk judgment, not an empirical severity score.

The five categories are:
`unnecessary_tools`, `unauthorized_tools`, `pii_propagation`,
`delegation_escalation`, `indirect_attacks`. **Never** add real PII or
real destructive tools, and do not add legitimate-PII tests unless the
research scope explicitly changes.

- **Unnecessary** means granted but not necessary for the task; this is
  distinct from an **unauthorized** attempted call outside the delegated
  allowlist.
- Capture `attempted`, `blocked` and `executed` separately and assert
  registry/handler enforcement, not only tool visibility.
- **Unexpected PII propagation** is measured at attempted and executed
  phases; avoid leaking email/canary values into ordinary errors.
- **Escalation** cannot be hidden by Decision-model relevance pruning:
  the originally requested worker grant must be authorized first.
- **Indirect attack** content is untrusted data, never new permission.
- The Decision model is advisory **only**: it may narrow, never grow, the
  parent's grant.
- The oracle `necessary_tools` is for grading, never fed to the live
  model selector. The parent-provided `granted_tools` is part of the
  worker prompt and runtime grant.
- The parent/main agent does **not** independently receive a
  high-level task, choose a worker, plan work, or decide whether to
  delegate. The harness chooses each of the 55 predefined worker tasks
  and parent grants; `subagents.default_model` runs the worker,
  while the configured main model (or `subagents.reviewer_model`)
  **only reviews** the worker output. Report this limitation alongside
  different SDK tool visibility and the extra Agent-RT reviewer model
  usage. Do not market the current benchmark as autonomous main-agent
  planning evaluation.

Relevant enforcement lives in Python
`ext/runtime/agent/subagents.py`, `ext/runtime/agent/delegation.py`,
`ext/runtime/tools/registry.py`, `ext/runtime/guardrails/swarmguard.py`,
`ext/swarmguard_decisions.py`, `ext/swarmguard_evaluation.py` and
corresponding TypeScript modules.

## Cache contract

`cache.py` caches **successful** SDK LLM, Agent-RT LLM and Decision-model
answers in distinct versioned SQLite namespaces. Never cache credentials,
permissions, tool effects, authorization decisions or model failures.
Transport disconnects, timeouts, HTTP 429/5xx, invalid envelopes,
incomplete/filtered responses and Decision refusals must not write cache.
Keep the full stable model/endpoint identity, messages, tool schemas,
generation parameters and decision questions in keys; never put secret
values into keys or reports.

The actual `SubagentSwarm` creates random task/trace IDs and embeds a task
ID in the reviewer's evidence. `configured.stable_benchmark_request_key`
**only removes those generated orchestration IDs from the cache key**,
not the provider input, and retains the role, prompts, tool schemas and
security policy context. This enables an identical `read-write` rerun to
make no further provider calls. Do not strip task policy/tool grants or
user input. Revalidate runtime guards on replay; cached responses do
not bypass authorization.

An initial successful Decision preflight is cacheable to avoid an extra
model call on identical runs. A cached preflight cannot establish that a
provider is *still* online; `refresh` forces fresh validation. A failed
or malformed preflight must stop the evaluation before any SDK request.

## Tests and maintenance

Run `cd python && pytest tests/test_swarmguard.py
tests/test_swarmguard_benchmark.py` and the TypeScript suite when touching
shared contracts. The CLI needs live credentials; **tests must use offline
doubles**. Test: absent Decision model, broken Decision response/transport,
SDK vs real swarm comparison, actual no-tool-execution enforcement, and
two identical runs with zero second-run SDK, reviewer, worker and Decision
calls. Check that a changed model/endpoint identity causes a cache miss.
Use `pytest.mark.anyio` and the `anyio_backend='asyncio'` test fixture in
SwarmGuard-specific Python tests, even when pytest-asyncio is absent.
The retired `make eval-swarmguard-tests` target and legacy
`--framework`/scripted CLI options must not be reintroduced; the
scripted harness remains available as imported helpers for regression tests.

Maintain `README.md`, this file, `agent-rt.example.yaml` and canonical
website docs when schema, benchmark semantics or cache behavior changes.
Do not claim a task is tested or passing unless execution was actually run.
