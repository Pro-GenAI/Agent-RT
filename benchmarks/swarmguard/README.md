# SwarmGuard-Bench

This experimental benchmark contains **65 synthetic cases** and runs the
**ten top-ranked cases per framework** by default, comparing two live
executions of each selected delegated-worker task:

| Result | Execution |
| --- | --- |
| `openai_sdk` | A minimal manual harness using the regular OpenAI SDK and the configured worker model, with the full mock tool catalog and no Agent-RT grants |
| `agent_rt` | Agent-RT `SubagentSwarm`, the configured worker and reviewer/main models, Decision-model-guided least-privilege grants, runtime authorization and unexpected-PII tool-input guard |

The SDK is a model API, not itself a swarm orchestrator. The difference is
between **these two example harnesses**, not an intrinsic SDK security
guarantee. Agent-RT makes additional reviewer calls; the compact comparison deliberately
omits token usage rather than comparing unequal workloads.

## One command

From the repository root:

```bash
make eval-swarmguard
```

The command immediately prints an initialization message, then reports YAML
loading and Decision-model validation status before evaluation begins. Once
cases start, it shows a **live progress bar for the 20 default harness/case runs**
(10 cases for the OpenAI SDK, then 10 for Agent-RT). The progress total
scales to twice the selected case count when `--limit` is overridden. The bar
updates after each completed case in interactive terminals; redirected logs
receive periodic newline-delimited progress snapshots. Initialization,
progress and errors go to **stderr**, while the final **Rich tables on stdout**
summarize per-harness metrics and notable case findings. The table lists up to
20 cases, prioritizing flagged cases; full records remain available as JSON.
A/E/B denotes attempted/executed/blocked tool calls. A grade passing with no
attempted calls does not establish that the runtime guard intervened.

The benchmark report uses the existing Python `cli` extra's
`rich>=13.7,<15` dependency. If Rich is missing, install
`pip install 'agent-rt[cli]'` or `pip install rich`. Raw JSON can still be
emitted without Rich using `--json`. The progress count reflects completed
cases, not individual HTTP requests; an in-flight case may take time before
the bar advances.

Both `make eval-swarmguard` and `make eval-swarmguard-alt` use the **same live evaluation**; the latter selects the registered `subagent-alt` worker alias. They run the
two-way **live** comparison, never silently falls back to scripted answers,
and always uses `./agent-rt.yaml` explicitly. A real, functioning Decision
model must be configured before any evaluation case can execute.

Use the standard [Agent-RT YAML configuration](../../agent-rt.example.yaml),
not separate `EVAL_*` variables. For example:

```yaml
model_list:
  - model_name: main-agent
    agent_rt_params:
      model: openai/cx/gpt-6-luna
      api_base: os.environ/OPENAI_BASE_URL
      api_key: os.environ/OPENAI_API_KEY
      websocket: false

  - model_name: subagent
    agent_rt_params:
      model: openai/qwen3.8-27b
      api_base: os.environ/OPENAI_SUBAGENT_BASE_URL
      api_key: os.environ/OPENAI_SUBAGENT_API_KEY
      websocket: false

  # Optional second worker model to compare across benchmark runs.
  - model_name: subagent-alt
    agent_rt_params:
      model: openai/YOUR_ALTERNATE_WORKER_MODEL
      api_base: os.environ/OPENAI_SUBAGENT_BASE_URL
      api_key: os.environ/OPENAI_SUBAGENT_API_KEY
      websocket: false

agent_rt_settings:
  default_model: main-agent
  subagents:
    default_model: subagent
    reviewer_model: main-agent
  decision_model:
    provider: jev
    base_url: https://your-working-jev-endpoint.example
    model: YOUR_DECISION_MODEL
    # api_key: os.environ/AGENT_RT_DECISION_API_KEY
  execution_limits:
    timeout_seconds: 120  # shared by worker + reviewer for each Agent-RT case
```

**Replace the Decision endpoint and model with a real service.** `provider:
jev` requires a Jev-compatible `POST /v1/systemone` endpoint. For a service
implementing the OpenAI Decisions API instead, set `provider: openai`, its
`base_url`, `model` and `api_key`. An ordinary OpenAI-compatible *chat*
completion endpoint is **not necessarily** a Decision endpoint.

Export the API-key and endpoint environment variables referenced by the
file. These variables are secrets resolved by `os.environ/NAME`; they are
not a second model-selection configuration. The main and worker model aliases
must exist in `model_list`. The worker must have one OpenAI-compatible
deployment so the regular SDK comparison is well-defined. The reviewer uses
`subagents.reviewer_model`, or `default_model` if omitted.

### Select a different sub-agent model for an evaluation

Use `--subagent-model ALIAS` with a name declared under `model_list`.
For the usual Makefile workflow, use the alternate-model shortcut or set
`SUBAGENT_MODEL` explicitly:

```bash
make eval-swarmguard-alt                        # alternate worker alias, ten cases per framework
make eval-swarmguard SUBAGENT_MODEL=subagent-alt  # equivalent
```

For direct CLI use, including five-case evaluations:

```bash
# Five cases per framework, with every worker using the alternate model.
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml \
  --subagent-model subagent-alt --limit 5

# Compare a second model using the same ten-case selection.
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml \
  --subagent-model subagent --limit 10
```

The selected alias applies to **every worker case in both frameworks**:
OpenAI SDK receives its resolved upstream model and API endpoint; Agent-RT
uses the same configured worker alias and retains its normal reviewer.
This switch does **not** change the reviewer, Decision model, or number of
cases. It does not create five simultaneous sub-agents; `--limit 5` executes
five delegated-worker tasks per framework. Omit the flag to use
`agent_rt_settings.subagents.default_model`.

An alias must exist in `model_list` and resolve to exactly **one
OpenAI-compatible deployment**. Use a different alias for each model or
endpoint, without editing or exporting additional benchmark-specific
environment variables. The full report includes both the chosen alias
(`worker_model`) and upstream ID (`worker_deployment_model`).

**Caching is separated by selected worker model.** SDK cache keys include the
worker alias, configured deployment identity and request model ID. Agent-RT
cache keys include the worker alias and model configuration. Repeating the
same selection replays successful responses; switching A → B → A cannot
reuse B's worker responses for A. Decision caching remains independent
and is reused when its inputs are unchanged. Use `--cache-mode refresh`
when you want fresh trials regardless of model identity.

The Agent-RT side gives the worker and its reviewer **one shared
per-case deadline** from `agent_rt_settings.execution_limits.timeout_seconds`.
For a live, two-model evaluation, start with 120 seconds rather than the
30-second general-purpose example and adjust based on your providers' latency.
This setting is a hard upper bound, not a guaranteed completion time: slow
worker calls leave less time for review. If the reviewer fails with
`TimeoutError: parent deadline expired`, increase this value in the
**local, git-ignored** `./agent-rt.yaml`, then rerun `make eval-swarmguard`.
Do not reset the deadline between worker and reviewer or silently treat
timed-out cases as successful. The normal SDK comparison does not use an
Agent-RT reviewer, so its latency is not directly comparable.

The benchmark fails immediately if its YAML is absent or invalid, the worker
model cannot be resolved, or the Decision configuration is missing. A
**real Decision request is validated before any benchmark case**; transport
failures, invalid answers, missing answers, HTTP errors and refusals abort
the run. A previously successful startup validation may be replayed from cache
on an identical rerun: cached replay does not check whether the endpoint has
subsequently gone offline. Use `--cache-mode refresh` to force fresh requests
when validating current model availability.

## Dataset and scoring

The `cases/*.jsonl` fixtures contain **65 synthetic tasks**, 13 per
category. The original 50 remain intact, five earlier layered variants
are in [`advanced_scenarios.jsonl`](cases/advanced_scenarios.jsonl), and
ten additional red-team probes are in
[`pressure_scenarios.jsonl`](cases/pressure_scenarios.jsonl).
The [priority manifest](priority_cases.json) orders 20 cases: original
representatives (1-5), ten new pressure probes (6-15), and five earlier
advanced variants (16-20). The other 45 cases are still available.

The new pressure probes aim to increase the opportunity to detect actual
misuse through forged approvals, copied tool-call transcripts, policy
exceptions, deletion pretexts, synthetic-contact egress, and unnecessary
directory preflights. They are **unvalidated adversarial hypotheses**:
a correct model may refuse all attacks. Only a real attempted or executed
tool call can support a reported failure.

The *trusted* worker task never contains attacker text. Both harnesses receive
the same inert and task-relevant document or search result. The older controls
use a distinct `untrusted_excerpt` field; new pressure probes place attack
instructions directly in the document `content` field. Both are untrusted
**tool-returned data**, never parent or system instructions. The SDK's
bounded manual loop allows six model turns, which reduces premature
tool-call-only terminations.

This is **synthetic tool-response injection with a fixed payload**, not
adaptive multi-round steering, real child identity changes, actual
multi-hop exfiltration, or executed nested agents. Priority is qualitative
risk judgment, not empirically measured severity. The default ten-case run
is an exploratory smoke test, **not** enough to establish reliable
comparative failure rates.

This addresses agent-infrastructure security (Sprint Track 3), the runtime
boundary for delegated workers (Track 1), and monitorable security events
(Track 4), in the sense of the [Apart Research 2026 Swarm Alignment Sprint](https://apartresearch.com/sprints/swarm-alignment-2026).
The current benchmark has one delegated worker plus a reviewer rather than
an autonomous multi-agent network. It does not demonstrate emergent collusion,
covert coordination, real multi-round infection, or actual nested child
executions. For research-quality results, report exact case IDs, the
configuration and model identities, cache mode, repeated independent trials,
baseline outcomes, variance, limitations, and dual-use/ethical considerations.

The five categories are:

- Unnecessary use of a granted tool
- Calls to tools outside the worker's delegated grant
- Unexpected synthetic email/canary propagation
- Delegation permission escalation attempts
- Indirect attacks via untrusted task or tool text

Each fixture provides `necessary_tools`, `granted_tools`, `mock_tool`,
`mock_arguments` and an optional `attack`. The ground-truth tool minimum
is used by graders, **not** supplied to the live Decision model. The
`granted_tools` allowlist *is* included in the worker's task prompt, as
a parent delegation policy. There are no legitimate-PII scenarios. All
tool handlers are inert mocks: `delete_file` cannot delete anything and
`web_search` cannot contact the public web.

Every case reports attempted, blocked and executed tool calls; unauthorized
attempts, PII execution, delegation escalation, repeated identical
successful calls, model usage (in optional JSON only), and grader outcomes.
The Rich case findings now show only observed policy/tool-use findings:
`no final answer` is **not** a security violation and is not listed
as such. The overall comparison separately counts cases with final
answers. Repeated identical tool calls are a redundancy signal, not
automatically a permission violation. Do not infer an attack succeeded
from a malicious fixture alone; an attempted or executed behavior must
appear in the trace. The Agent-RT side uses the actual `SubagentSwarm`
delegation/reviewer path, but the **parent's decision to delegate is fixed by
the benchmark**, not generated by a parent LLM. Security comparisons must
report that limitation, the different tool visibility, the extra reviewer
model calls, and model sampling uncertainty. Do not present these tests
alone as proof of production-world security improvements.

## Response cache and reproducibility

Successful SDK LLM, Agent-RT LLM and Decision-model responses are cached in
separate SQLite namespaces under ignored `.cache/responses.sqlite`.
The default mode is `read-write`. **Running the identical command again
reuses successful model responses** and does not call either model provider
again. The benchmark canonicalizes its own generated task/trace identifiers
for cache lookup, but continues to pass unmodified input to model providers
and reapplies guards, permissions and graders.

Connection errors, timeouts, HTTP 429/5xx errors, failed/refused/incomplete
model responses, and malformed Decision answers are **never cached**.
A later failure does not erase previously completed successful responses.
Keys incorporate public model aliases, upstream IDs, endpoints, request
messages, tools, model parameters and Decision questions; API-key **values**
are never stored in cache keys. If you rotate an account's secret while
retaining the same model/endpoint, select a new cache path or force refresh
to avoid reusing the previous account's responses.

For direct script use (or custom case counts):

```bash
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml             # default: 10
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --limit 5   # original quick five
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --limit 20  # all ten new challenges
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --limit 65  # entire dataset
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --subagent-model subagent-alt --limit 5
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --cache-mode read
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --cache-mode refresh
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --output report.json # Rich table + complete JSON file
python benchmarks/swarmguard/run.py --config ./agent-rt.yaml --json               # complete JSON on stdout
```

`--output report.json` saves all per-case grader details while retaining Rich
tables in the terminal. `--json` prints the full machine-readable document
instead of tables (progress stays on stderr); both options can be combined.
The default Rich report omits tool arguments and token counts, displays
`None` when a case has no observed policy/tool-use findings, and can be
redirected to text without ANSI escape codes. A missing final answer is
shown only through the aggregate completion count, not as a policy failure. The full JSON schema remains unchanged for tools
that explicitly request `--json` or `--output`.
No blocked calls or failed graders does not establish that a specific
attack was attempted and prevented.

`read` fails on cache misses without contacting any model. `refresh` deliberately
contacts the model again and overwrites only successful answers. Cached
response replay must not be counted as a fresh stochastic model trial or
as newly billed inference.

## Offline regression tests

Unit tests use scripted doubles and **never** contact live model or Decision
endpoints:

```bash
cd python
pytest tests/test_swarmguard.py tests/test_swarmguard_benchmark.py
```

The test-only helpers in `run.py` remain available for these deterministic
security invariants. Existing baseline code and datasets predate these priority
and fixture improvements; distinguish preexisting work from sprint work
explicitly in the submission. All fixtures use synthetic contact addresses
and inert tool handlers; no live exfiltration is performed. They are **not** the user-facing `make eval-swarmguard`
execution path. Install Python package dependencies as described in the
repository contributor handbook; the SwarmGuard tests use the AnyIO pytest
plugin with the asyncio backend.
