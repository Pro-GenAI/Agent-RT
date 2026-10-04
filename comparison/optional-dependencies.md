# Optional dependency split design

Date: 2026-09-27

This document is the required design/impact record for `PERF-052` before changing package manifests.

## Current dependency graph

### Python

The current core install declares:

- `httpcore[asyncio]>=1,<2`
- `openai>=1,<3`
- `anthropic>=0.40,<1`
- `agent-action-guard[harmactionseval]>=1.2.1,<2`
- `onnxruntime<1.24` on Python < 3.11

The provider SDKs and Agent Action Guard are already imported lazily at the feature call sites. The default OpenAI-compatible provider transport uses the core `httpcore[asyncio]` dependency and does not require the OpenAI SDK. Provider startup validation and explicit SDK transport do require the matching SDK. The model-backed tool-input guardrail requires `agent-action-guard`; callers that inject a classifier do not.

A depth-2 `uv tree` before the split shows the guard package pulling NumPy, ONNX Runtime, OpenAI, tokenizers, and evaluation extras. The OpenAI and Anthropic SDKs also add Pydantic, Jiter, distro, and other client dependencies.

Measured installed directories in the isolated comparison environment before the split:

| Python package directory | Installed size |
| --- | ---: |
| `openai` | ~19 MiB |
| `anthropic` | ~7.5 MiB |
| `agent_action_guard` | ~0.24 MiB |
| `onnxruntime` | ~49 MiB |

These numbers exclude many transitive package directories, so they are a lower bound on removable install footprint.

### TypeScript

The current runtime dependencies are:

- `openai`
- `@anthropic-ai/sdk`
- `agent-action-guard`

All three are dynamically imported only when the matching SDK/guard feature is used. The compatible OpenAI transport uses Node's built-in `fetch` and Web Streams.

A depth-1 `npm ls` before the split shows:

- OpenAI pulling AWS/Smithy helpers, Undici, WebSocket, and Zod-related dependencies.
- Anthropic pulling JSON Schema tooling, webhooks support, and Zod.
- Agent Action Guard pulling Hugging Face tokenizers, ONNX Runtime Node, and OpenAI.

Measured direct package directories before the split:

| TypeScript package directory | Installed size |
| --- | ---: |
| `openai` | ~30 MiB |
| `@anthropic-ai/sdk` | ~14 MiB |
| `agent-action-guard` | ~0.20 MiB |

Again, these values exclude transitive dependencies and therefore understate removable install footprint.

## Proposed dependency graph

### Python

Keep only `httpcore[asyncio]` in the required dependency set.

Add explicit extras:

- `agent-rt[openai]` — OpenAI SDK support, including startup validation and explicit SDK transport.
- `agent-rt[anthropic]` — Anthropic SDK support.
- `agent-rt[providers]` — both provider SDKs.
- `agent-rt[guardrails]` — Agent Action Guard and its existing HarmActionsEval/ONNX requirements.
- `agent-rt[api]` — FastAPI and Uvicorn for hosting the optional OpenAI-compatible inbound API; these server dependencies remain outside the minimal runtime install.
- `agent-rt[cli]` — prompt-toolkit and Rich for the optional interactive terminal client; terminal UI dependencies remain outside the minimal runtime install.
- `agent-rt[all]` — all optional Python integrations, including the API server and terminal client.

Provider-neutral contracts, request/response types, compatible OpenAI transport, custom provider implementations, classifier-injected guardrails, and Jev-compatible System-1 decision clients remain available in the core install. Laya model weights and ONNX runtimes are not direct Agent RT dependencies; `AGENT_RT_JEV_BASE_URL` may target a local or hosted Jev-compatible `/v1/systemone` service, while `AGENT_RT_JEV_MODEL` selects the model served by that endpoint.

### TypeScript

Move OpenAI, Anthropic, Agent Action Guard, and OpenLLMetry/Traceloop from required `dependencies` to optional `peerDependencies`, marked optional through `peerDependenciesMeta`. Provider/guard packages may remain in `devDependencies` for integration coverage; Traceloop is loaded dynamically only when monitoring is configured so its large OpenTelemetry graph is not part of the required production install.

This keeps one npm package and the existing root export surface. Consumers install only integrations they use:

- `npm install agent-rt openai`
- `npm install agent-rt @anthropic-ai/sdk`
- `npm install agent-rt agent-action-guard`
- `npm install agent-rt @traceloop/node-server-sdk`

No new package or subpath is required for this first split because the integration code is already lazily imported and the root module does not eagerly evaluate those SDKs.

## Compatibility impact

The public provider and guardrail classes/functions remain exported from the same modules.

Behavioral change: users who previously relied on a transitive installation of an SDK/guard package must now install that integration explicitly. When an optional dependency is missing, Agent RT must throw an actionable error naming the package and install command rather than exposing a raw `ModuleNotFoundError` / `ERR_MODULE_NOT_FOUND`.

The compatible OpenAI transport remains usable from the minimal core install. Tests that inject mock clients/classifiers remain dependency-independent.

No provider-specific dependency may be imported at module import time after the split.

## Publishing implications

Python wheels/sdists publish PEP 621 optional extras in package metadata. The base wheel stays the same code artifact; only dependency metadata changes.

The npm package publishes optional peer dependency declarations. Integration packages remain development dependencies in this repository but are not bundled into the published package tarball because only `dist` and `README.md` are published.

Lockfiles must be refreshed after manifest changes. Package build/tarball inspection must confirm that exports remain intact and no integration SDK source is bundled.

## Migration plan

1. Add actionable optional-import helpers/errors around OpenAI SDK, Anthropic SDK, and Agent Action Guard dynamic imports.
2. Move Python provider/guard dependencies into extras while retaining `httpcore[asyncio]` as core.
3. Move TypeScript integrations to optional peer dependencies and keep them in dev dependencies.
4. Update installation documentation with core, provider, guardrail, and all-integration commands.
5. Run full language test suites in the repository environments.
6. Create stripped temporary installs without optional integrations and verify core import, compatible OpenAI provider construction, classifier-injected guardrails, and actionable failures for missing SDK/guard packages.
7. Inspect built Python and npm package metadata/tarballs.
8. Measure minimal-install dependency/package footprint and rerun cold import/runtime checks where relevant.

## Expected benchmark impact

Cold module import should not materially regress because these integrations were already lazy. The primary expected improvement is install size and clean-environment dependency count. Baseline runtime memory before first integration use may improve in environments where dependency import hooks or package metadata were previously touched, but no such improvement is assumed without measurement.

The existing cold-import, runtime, feature, and regression-check runners remain the sources of performance evidence. Any headline improvement must come from measured post-change results.
