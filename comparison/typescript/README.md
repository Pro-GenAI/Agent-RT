# TypeScript comparison

Reproducible Node.js benchmarks comparing Agent RT with LangChain.js and LlamaIndex.TS. Results are summarized in the [top-level comparison](../README.md); this page explains how to reproduce them. Build Agent RT first (`cd typescript && npm run build`), then run everything from `comparison/typescript`.

## Candidates

- Agent RT
- LangChain.js
- LlamaIndex.TS

## Module-load benchmark

`benchmark-imports.mjs` runs each framework module load in a fresh Node process and records module-load wall time, end-to-end subprocess time, CPU time, RSS before/after import, RSS delta, and maximum observed RSS. It writes raw samples plus CSV/JSON summaries under `results/`.

Build Agent RT, then install comparison dependencies only in the comparison folder:

```bash
cd typescript
npm run build
cd ../comparison/typescript
npm ci
node benchmark-imports.mjs --runs 30
```

External comparison packages live in `comparison/typescript/node_modules`; the command does not add them to the main `typescript/node_modules` dependency set. The local Agent RT benchmark imports `../../typescript/dist/index.js` directly. Missing comparison packages are recorded as unavailable instead of aborting the full run.

## Runtime/API benchmark

`benchmark-runtime.mjs` compares framework overhead against deterministic localhost OpenAI-compatible mocks implemented by `mock-llm-api.mjs`. Agent RT uses its production-default persistent Responses WebSocket path (`/v1/responses`), including `previous_response_id` continuation and `stream_id` multiplexing; LangChain.js and LlamaIndex.TS use their configured HTTP/SSE paths. No external model or API is contacted.

Each framework runs in its own worker process. The benchmark separates worker module load, first and subsequent provider construction, first and warm completion, first and warm streaming, 5/20/100-turn conversations, a fixed 100-request warm loop, and 1/4/16-way concurrency. It records RSS at each boundary and writes raw samples, `runtime-summary.csv`/JSON, and `runtime-memory-stages.csv` under `results/`.

Build the local harness, then run from the isolated comparison environment:

```bash
cd typescript
npm run build
cd ../comparison/typescript
npm ci
node benchmark-runtime.mjs --runs 30
cat results/runtime-summary.csv
cat results/runtime-memory-stages.csv
```

For targeted profiling without overwriting the durable all-framework aggregate:

```bash
node benchmark-runtime.mjs --framework Agent RT --runs 5 --profile
```

`--profile` runs the worker with Node CPU profiling, `--expose-gc`, detailed `process.memoryUsage()` capture, and a heap snapshot; it is diagnostic and changes timings. The comparison environment includes both OpenAI and Anthropic provider addons for each alternative framework: `@langchain/openai` + `@langchain/anthropic` and `@llamaindex/openai` + `@llamaindex/anthropic`. The runtime benchmark currently exercises the OpenAI-compatible adapters.

## Tool-call and structured-output benchmark

`benchmark-features.mjs` measures equivalent wire-level feature work across Agent RT, LangChain.js, and LlamaIndex.TS: one tool round trip, five repeated tool round trips, valid JSON Schema output, an invalid-first/repair sequence, and 20 repeated requests using the same schema. The tool continuation includes a deterministic serialized JSON tool result. This feature benchmark explicitly disables Agent RT WebSocket mode so every framework stays on the same HTTP/SSE transport contract; structured validation is performed by the benchmark runner so framework-specific Zod/schema-validation costs are not mixed into adapter overhead.

```bash
node benchmark-features.mjs --runs 30
cat results/features/feature-summary.csv
cat results/features/feature-memory-stages.csv
```

Unfiltered runs with at least 30 samples refresh `feature-measured-results.csv`. Raw samples and post-timing RSS/heap/external-memory probes remain under `results/features/`.

## Metrics

- Wall-clock time in milliseconds.
- Median and p95 iteration latency.
- Derived CPU percentage (process CPU time divided by module-load wall time).
- Peak resident memory (RSS) in MiB.
- Startup/module-load time.
- Fresh-install `node_modules` footprint from an isolated npm install.

## Measured results

Published 30-run benchmark updates automatically regenerate the TypeScript SVG charts under `../assets/` from the durable CSVs via `generate-charts.mjs`; those charts are embedded in the top-level comparison README.

The checked-in module-load aggregate is [`measured-results.csv`](measured-results.csv), produced from 30 cold-process runs per framework.

The checked-in runtime aggregate is [`runtime-measured-results.csv`](runtime-measured-results.csv). Any **unfiltered** runtime benchmark run with `--runs 30` or more automatically refreshes that durable aggregate after writing the detailed output under `results/`; `--framework` diagnostics never overwrite it.

The checked-in feature aggregate is [`feature-measured-results.csv`](feature-measured-results.csv), produced from 30 unfiltered tool/structured-output runs per framework.

The checked-in fresh-install aggregate is [`install-size-measured-results.csv`](install-size-measured-results.csv). Reproduce it after building Agent RT with:

```bash
node benchmark-install-size.mjs --publish
```

The benchmark creates a clean temporary npm project per framework, installs with scripts/audit/funding/package-lock disabled, and measures logical bytes under `node_modules`. Agent RT is first packed with `npm pack` so the measurement reflects its installable package rather than the repository checkout. LangChain.js includes both `@langchain/openai` and `@langchain/anthropic`, and LlamaIndex.TS includes both `@llamaindex/openai` and `@llamaindex/anthropic`; the footprint therefore represents both provider addon surfaces even though the runtime benchmark currently exercises the OpenAI-compatible adapters. The published 2026-10-03 snapshot used Node.js 22.22.2 and npm 10.9.7.

### Current comparison snapshot

| Framework | Fresh install | Cold module load | Warm completion | Completion p95 | Stream first text | Five-turn | Tool round trip | Structured valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent RT | 1.24 MiB | 64.4 ms | 1.85 ms | 2.31 ms | 1.48 ms | 6.68 ms | 2.70 ms | 1.25 ms |
| LangChain.js + OpenAI + Anthropic addons | 82.44 MiB | 496.5 ms | 2.60 ms | 3.67 ms | 2.52 ms | 12.38 ms | 3.33 ms | 1.58 ms |
| LlamaIndex.TS + OpenAI + Anthropic addons | 60.38 MiB | 983.6 ms | 2.23 ms | 2.64 ms | 1.80 ms | 8.98 ms | 2.96 ms | 1.55 ms |

Runtime/import/feature values are the checked-in 30-run snapshot from 2026-09-28. Fresh-install footprint is the 2026-10-03 clean npm-install snapshot described above. Runtime values use a persistent-connection localhost mock and 10 untimed steady-state warmups.

Raw samples, stage-memory data, profiles, and detailed JSON/CSV summaries remain under the ignored `results/` directory so they can be regenerated without adding benchmark noise to normal commits.

## Diagnostics (maintainers)

These runs are diagnostic-only: they never refresh the checked-in aggregates and are not headline scores.

### Scalability stress benchmark

`benchmark-scalability.mjs` is a diagnostic stress runner for long-lived provider reuse. By default it performs 5,000 sequential completion calls followed by 5,000 completion calls at up to 256 in-flight requests per framework, with each framework isolated in its own worker process. Agent RT uses the local Responses WebSocket mock; the other TypeScript frameworks use the deterministic HTTP mock. No external model/API calls are made.

It records total wall time, process CPU time and derived CPU utilization, requests/second, median/p95/p99 per-call latency, starting/ending/post-GC RSS and V8 heap usage, sampled peak RSS/heap/external/array-buffer memory, and memory growth. Workers run with `--expose-gc` so the post-GC measurement distinguishes retained memory from reclaimable heap. Memory is sampled throughout each phase and written separately.

```bash
cd typescript
npm run build
cd ../comparison/typescript
npm ci
node benchmark-scalability.mjs
cat results/scalability/scalability-summary.csv
cat results/scalability/scalability-memory-samples.csv
```

Use `--continuous-calls`, `--concurrent-calls`, and `--concurrency` to scale the workload. `--concurrent-calls` is the total request count and `--concurrency` is the maximum number actively in flight; set them equal to deliberately launch the full batch at once. `--framework "Agent RT"` limits a diagnostic to one framework. These stress runs are diagnostic-only and never refresh the checked-in runtime aggregates. Worker safety limits are enabled by default: each worker is pinned to at most 2 CPU cores with Linux `taskset` and is terminated if RSS exceeds 1,024 MiB. Override with `--max-cpu-cores N` and `--max-memory-mib N`. If Linux `taskset` or `/proc` accounting is unavailable the runner fails closed; `--no-resource-limits` is the explicit opt-out.

For a larger but still bounded local run, for example 4 CPU cores and 2 GiB RSS per worker:

```bash
node benchmark-scalability.mjs \
  --max-cpu-cores 4 \
  --max-memory-mib 2048
```
