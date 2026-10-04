# Python comparison

Reproducible Python benchmarks comparing Agent RT with LangChain and LlamaIndex. Results are summarized in the [top-level comparison](../README.md); this page explains how to reproduce them. Run everything from `comparison/python` in the isolated environment created by `uv sync`.

## Candidates

- Agent RT
- LangChain
- LlamaIndex

## Import benchmark

`benchmark_imports.py` runs each framework import in a fresh Python subprocess and records import wall time, end-to-end subprocess time, CPU time, RSS before/after import, RSS delta, and peak RSS where the platform exposes it. It writes raw samples plus CSV/JSON summaries under `results/`.

Use the comparison-local environment only. The benchmark project pins Python 3.13 so the full third-party dependency set has compatible wheels. With `uv`:

```bash
cd comparison/python
uv sync
uv run python benchmark_imports.py --runs 30
```

This creates `comparison/python/.venv`; it does not install LangChain or LlamaIndex into `python/.venv`.

LlamaIndex's NLTK dependency is pinned to a patched upstream commit in this project; see `../AGENTS.md` for the reason and when to remove the pin.

Equivalent isolated `pip` workflow:

```bash
cd comparison/python
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e ../../python langchain langchain-openai langchain-anthropic llama-index llama-index-llms-openai llama-index-llms-anthropic tiktoken
python benchmark_imports.py --runs 30
```

The local Agent RT source is referenced from `../../python`. Missing comparison packages are recorded as unavailable instead of aborting the full run.

## Fresh-install footprint benchmark

`benchmark_install_size.py` compares disk footprint from scratch instead of measuring the shared comparison environment. It creates one fresh `uv venv` per framework, records that empty environment as the baseline, installs only the packages needed for that framework, then reports the additional logical bytes. uv installs use copy mode so cache hardlinks do not make one environment appear artificially cheap.

To create the empty environments only:

```bash
uv run python benchmark_install_size.py --prepare-only
```

To recreate the environments, install every framework, compare footprint, and retain the environments for inspection:

```bash
uv run python benchmark_install_size.py --publish --keep-venvs
cat install-size-measured-results.csv
```

Omit `--keep-venvs` to remove the per-framework environments after writing the results. `--publish` refreshes the checked-in `install-size-measured-results.csv` and requires an unfiltered all-framework install. Use `--framework "Agent RT"` (or another framework label) for a targeted diagnostic without publication. Agent RT is installed from the local `../../python` source; LangChain installs `langchain` plus both `langchain-openai` and `langchain-anthropic`, while LlamaIndex installs `llama-index` plus both `llama-index-llms-openai` and `llama-index-llms-anthropic`. LlamaIndex also uses the patched NLTK commit pinned by this comparison project. This benchmark measures the complete fresh environment delta including transitive dependencies, not just the top-level package directory.

## Runtime/API benchmark

`benchmark_runtime.py` compares framework overhead against deterministic localhost OpenAI-compatible mocks implemented by `mock_llm_api.py`. Agent RT uses its production-default persistent Responses WebSocket path (`/v1/responses`), including `previous_response_id` continuation and `stream_id` multiplexing; LangChain and LlamaIndex use their configured HTTP/SSE paths. No external model or API is contacted.

Each framework runs in its own worker process. The benchmark separates worker module load, first and subsequent provider construction, first and warm completion, first and warm streaming, 5/20/100-turn conversations, a fixed 100-request warm loop, and 1/4/16-way concurrency. It records RSS at each initialization/runtime boundary and writes raw samples, `runtime-summary.csv`/JSON, and `runtime-memory-stages.csv` under `results/`.

Run it from the isolated comparison environment:

```bash
cd comparison/python
uv sync
uv run python benchmark_runtime.py --runs 30
cat results/runtime-summary.csv
cat results/runtime-memory-stages.csv
```

For targeted profiling without overwriting the durable all-framework aggregate:

```bash
uv run python benchmark_runtime.py --framework Agent RT --runs 5 --profile
python -m pstats results/profiles/agent-rt.prof
```

`--profile` enables standard-library `cProfile` and `tracemalloc`; it is diagnostic and changes timings. The mock API advertises `gpt-4o-mini` only because some SDKs validate model names locally; the response is still generated entirely by the local mock server.

## Tool-call and structured-output benchmark

`benchmark_features.py` measures equivalent wire-level feature work across Agent RT, LangChain, and LlamaIndex: one tool round trip, five repeated tool round trips, valid JSON Schema output, an invalid-first/repair sequence, and 20 repeated requests using the same schema. The tool continuation includes a deterministic serialized JSON tool result. This feature benchmark explicitly disables Agent RT WebSocket mode so every framework stays on the same HTTP/SSE transport contract; schema validation is performed by the benchmark runner so library-specific Pydantic/schema-validation costs do not distort the comparison.

```bash
uv run python benchmark_features.py --runs 30
cat results/features/feature-summary.csv
cat results/features/feature-memory-stages.csv
```

Unfiltered runs with at least 30 samples refresh `feature-measured-results.csv`. Raw samples and post-timing stage-memory probes remain under `results/features/`.

## Tool-schema token benchmark

`benchmark_tool_tokens.py` estimates the main-model prompt-token cost of exposing a full tool catalog versus using Agent RT's Decision-model tool filter before the main LLM request. It uses `tiktoken` with `o200k_base` over compact OpenAI-style tool JSON. These are reproducible tokenizer estimates rather than provider-reported billed-token counts; provider-specific internal tool framing can differ. The full-catalog baseline represents configurations where LangChain, LlamaIndex, or another client forwards every registered tool to the main model; it is not a claim that those libraries cannot implement application-defined filtering.

Agent RT uses the real `make_decision_tool_visibility_filter` path with a deterministic Decision-model double. The double supplies repeatable relevance probabilities and records the exact `state` + `questions` payload that a Jev-compatible Decision model receives, so no external model or API call is required. Main-model tool tokens and Decision-model input tokens are reported separately. The "combined" estimate uses the same tokenizer for both only as a text-volume diagnostic; it should not be interpreted as provider billing when the Decision model is local or has different tokenization/pricing.

```bash
uv sync
uv run python benchmark_tool_tokens.py --publish
cat tool-token-measured-results.csv
```

The default contract measures 8, 16, 32, 64, and 128 registered tools, four user scenarios, 16 candidates per Decision-model question, and at most 8 selected tools. `--publish` is accepted only with those defaults, preventing custom diagnostics from overwriting the checked-in aggregate.

| Registered tools | Full catalog → main LLM | Agent RT → main LLM | Main tokens saved | Main reduction | Decision-model input | Same-tokenizer combined |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 861 | 112 | 749 | 87.0% | 1,056.5 | 1,168.5 |
| 16 | 1,717 | 112 | 1,605 | 93.5% | 2,040.5 | 2,152.5 |
| 32 | 3,429 | 219 | 3,210 | 93.6% | 4,081 | 4,300 |
| 64 | 6,853 | 433 | 6,420 | 93.7% | 8,162 | 8,595 |
| 128 | 13,765 | 865 | 12,900 | 93.7% | 16,388 | 17,253 |

Checked-in aggregate: [`tool-token-measured-results.csv`](tool-token-measured-results.csv). Detailed per-scenario CSV/JSON output is written under `results/tool-tokens/`.

## Metrics

- Wall-clock time in milliseconds.
- Median and p95 iteration latency.
- Derived CPU percentage (process CPU time divided by import wall time).
- Peak resident memory (RSS) in MiB.
- Startup/import time.
- Optional installed dependency size.
- Fresh-install environment footprint above an empty `uv venv` baseline.
- Estimated tool-schema prompt tokens for full-catalog versus Decision-model-selected tool exposure.

## Measured results

Published 30-run benchmark updates automatically regenerate the Python SVG charts under `../assets/` from the durable CSVs via `generate_charts.py`; those charts are embedded in the top-level comparison README.

The checked-in cold-import aggregate is [`measured-results.csv`](measured-results.csv), produced from 30 cold-process runs per framework.

The checked-in fresh-install footprint aggregate is [`install-size-measured-results.csv`](install-size-measured-results.csv), produced from clean per-framework `uv venv` installs after subtracting each empty-venv baseline.

The checked-in runtime aggregate is [`runtime-measured-results.csv`](runtime-measured-results.csv). Any **unfiltered** runtime benchmark run with `--runs 30` or more automatically refreshes that durable aggregate after writing the detailed output under `results/`; `--framework` diagnostics never overwrite it.

The checked-in feature aggregate is [`feature-measured-results.csv`](feature-measured-results.csv), produced from 30 unfiltered tool/structured-output runs per framework.

The checked-in deterministic tool-token aggregate is [`tool-token-measured-results.csv`](tool-token-measured-results.csv), produced with `tiktoken`/`o200k_base` under the default 8–128-tool contract.

### Current comparison snapshot

| Framework | Fresh install | Cold import | Warm completion | Completion p95 | Stream first text | Five-turn | Tool round trip | Structured valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent RT | 46.43 MiB | 141.4 ms | 1.56 ms | 1.77 ms | 1.98 ms | 8.12 ms | 3.34 ms | 1.54 ms |
| LangChain + OpenAI + Anthropic addons | 69.87 MiB | 0.3 ms | 9.03 ms | 11.36 ms | 7.65 ms | 40.12 ms | 9.81 ms | 5.35 ms |
| LlamaIndex + OpenAI + Anthropic addons | 219.91 MiB | 2019.1 ms | 4.26 ms | 5.35 ms | 5.13 ms | 29.31 ms | 7.31 ms | 3.49 ms |

Runtime/import/feature values are the checked-in 30-run snapshot from 2026-09-28. Fresh-install footprint is the 2026-10-03 Python 3.13.13 snapshot from clean `uv venv` environments using copy mode and reports logical installed bytes above the empty-venv baseline. The LangChain cold-import number is only the lazy top-level package entry point and is not equivalent to constructing a full agent stack. Runtime values use a persistent-connection localhost mock and 10 untimed steady-state warmups.

Raw samples, stage-memory data, profiles, and detailed JSON/CSV summaries remain under the ignored `results/` directory so they can be regenerated without adding benchmark noise to normal commits.

## Diagnostics (maintainers)

These runs are diagnostic-only: they never refresh the checked-in aggregates and are not headline scores.

### Scalability stress benchmark

`benchmark_scalability.py` is a diagnostic stress runner for long-lived provider reuse. By default it performs 5,000 sequential completion calls followed by 5,000 completion calls at up to 256 in-flight requests per framework, with each framework isolated in its own worker process. Agent RT uses the local Responses WebSocket mock; the other Python frameworks use the deterministic HTTP mock. No external model/API calls are made.

It records total wall time, process CPU time and derived CPU utilization, requests/second, median/p95/p99 per-call latency, starting/ending/post-GC RSS, sampled peak RSS, and RSS growth. Memory is sampled throughout each phase and written separately so leaks or stepwise growth can be inspected over time.

```bash
cd comparison/python
uv sync
uv run python benchmark_scalability.py
cat results/scalability/scalability-summary.csv
cat results/scalability/scalability-memory-samples.csv
```

Use `--continuous-calls`, `--concurrent-calls`, and `--concurrency` to scale the workload. `--concurrent-calls` is the total request count and `--concurrency` is the maximum number actively in flight; set them equal to deliberately launch the full batch at once. `--framework "Agent RT"` limits a diagnostic to one framework. These stress runs are diagnostic-only and never refresh the checked-in runtime aggregates. Worker safety limits are enabled by default: each worker is pinned to at most 2 CPU cores and is terminated if RSS exceeds 1,024 MiB. Override with `--max-cpu-cores N` and `--max-memory-mib N`. On platforms without Linux CPU affinity and `/proc` memory accounting the runner fails closed; `--no-resource-limits` is the explicit opt-out.

For a larger but still bounded local run, for example 4 CPU cores and 2 GiB RSS per worker:

```bash
uv run python benchmark_scalability.py \
  --max-cpu-cores 4 \
  --max-memory-mib 2048
```

### Import and transport diagnostics

Two diagnostic helpers support the second optimization milestone without changing headline benchmark semantics:

```bash
uv run python profile_import_dataclasses.py --top 20
uv run python benchmark_transport_coldstart.py --runs 30 \
  --output results/perf073-transport-coldstart.json
```

`profile_import_dataclasses.py` instruments dataclass transformation during root import and reports cost by source region plus the slowest individual classes. It is diagnostic only and does not replace the fresh-process cold-import benchmark.

`benchmark_transport_coldstart.py` compares primitive HTTPX and HTTPCore cold startup in fresh subprocesses against the deterministic localhost mock. It measures library import, client/pool construction, first and second POST, and validates HTTPCore SSE framing. It does **not** establish production parity for proxy/TLS environment handling; use it to decide whether a transport implementation is worth pursuing, not as a headline framework score.
