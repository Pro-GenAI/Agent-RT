# Architecture and Extension Contracts

Agent RT separates the agent loop from evaluation, control-plane management, extension loading, deployment, optimization, and operations. The goal is to keep application-level agent logic portable while giving infrastructure integrations explicit contracts that can be tested and replaced independently.

## Module map

The core Python runtime is `agent_rt.py`; the core TypeScript runtime is `index.ts`. Higher-level capabilities are intentionally split into companion modules. In Python, companion modules live in the `ext` package (framework adapters under `ext/compat/`); only `agent_rt.py`, `agent_rt_api.py`, and `agent_rt_cli.py` are top-level modules. `ext/runtime` holds the lazily loaded advanced feature family and `ext/transports` holds provider transports.

- Evaluation: Python `ext/evaluation.py`; TypeScript `evaluation.ts`.
- Prompt/configuration/feature flags: Python `ext/management.py`; TypeScript `management.ts`.
- Skills, middleware, provider registration: Python `ext/extensions.py`; TypeScript `extensions.ts`.
- Decision-model adapters: Python `ext/decisions.py`; TypeScript `decisions.ts`.
- Deployment, distributed execution, scaling, quotas, persistence schemas, migrations: Python `ext/deployment.py`; TypeScript `ext/deployment.ts`.
- Optimization: Python `ext/optimization.py`; TypeScript `optimization_runtime.ts`, `latency_policy.ts`, and `cost_policy.ts`.
- Benchmarks, conformance checks, and health checks: Python `ext/operations.py`; TypeScript `operations.ts`.
- OpenAI- and Anthropic-compatible inbound API hosting: Python `agent_rt_api.py`, installed with the optional `agent-rt[api]` extra.
- End-user terminal chat: Python `agent_rt_cli.py`, with Rich/prompt-toolkit installed through the optional `agent-rt[cli]` extra.
- Registration safety scanner: Python `ext/registration_safety.py`; TypeScript `registration_safety.ts`.
- Framework compatibility adapters (LangChain, LlamaIndex, OpenAI Agents, AutoGen, CrewAI, OpenAI SDK, Anthropic SDK): Python `ext/compat/`; TypeScript `langchain.ts`, `llamaindex.ts`, `openai_agents.ts`, `openai.ts`, and `anthropic.ts`. See `MIGRATION.md`.

## Extension contract principles

Extensions should depend on the smallest stable interface that solves their problem. Model adapters should implement provider-neutral request/response semantics. Memory, filesystem, sandbox, queue, evaluator, and telemetry integrations should be registered behind named provider contracts rather than imported directly into agent logic.

The core package remains provider-neutral and keeps vendor integrations optional. In Python, the root module also keeps the common agent/runtime/provider path eager while lazily loading a zero-outward-reference advanced feature family (planning/multi-agent, MCP/protocol, remote-agent, browser/computer, retrieval/realtime, CLI/API, IDE, workspace/artifact, and sandbox/code-interpreter integrations) from `ext.runtime.optional` on first public-symbol access; documented root imports remain unchanged. Python implementation-only transport and migration helpers live under the packaged `ext` namespace rather than as top-level modules; user-facing imports remain `agent_rt` and the documented compatibility aliases. TypeScript follows the same layout: the core lives in `src/index.ts`, companion contracts live under `src/ext/` (`ext/decisions.ts`, `ext/deployment.ts`, `ext/evaluation.ts`, `ext/extensions.ts`, `ext/management.ts`, `ext/operations.ts`, `ext/registration_safety.ts`, `ext/optimization/`), framework adapters and their helpers under `src/ext/compat/`, transports under `src/ext/transports/`, and shared internals (media helpers, ambient peer types) under `src/internal/`. The package `exports` map exposes the public `ext` modules; implementation-only helpers are not exported.

Provider registrations should have stable names, explicit replacement behavior, runtime validation of provider kinds, and conformance coverage. A replacement provider must preserve the semantic contract of the provider kind even if its storage engine, transport, process boundary, or vendor changes.

Middleware should operate at named runtime stages and should not depend on undocumented call ordering. Middleware that transforms values should preserve the declared type/shape expected by the next handler or terminal operation.

## Provider selection and transports

OpenAI requests default to a lightweight Responses WebSocket transport (`websockets` in Python and `ws` in TypeScript) that keeps one persistent `/v1/responses` connection and uses `previous_response_id` for incremental tool-loop continuation. `OPENAI_WEBSOCKET=0` or the language-specific `websocket=False` / `websocket: false` setting selects the existing OpenAI-compatible HTTP/SSE transport (`httpcore` in Python and native `fetch`/Web Streams in TypeScript); custom compatible endpoints that do not implement Responses WebSocket mode must opt out. The official OpenAI SDK is loaded only for SDK transport or SDK-backed startup validation. Anthropic's built-in client uses the optional `anthropic` Python package or `@anthropic-ai/sdk` TypeScript peer. Model-backed Agent Action Guard is optional as well. Missing integration use must fail with an actionable install instruction rather than a raw module-resolution error.

Environment startup selects providers from base URLs: `OPENAI_BASE_URL` selects OpenAI and takes precedence when both base URLs are configured; otherwise `ANTHROPIC_BASE_URL` selects Anthropic. If neither base URL is configured, `OPENAI_MODEL` alone is accepted as an OpenAI hint. API keys alone and `ANTHROPIC_MODEL` alone do not choose a provider, and startup fails when none of the accepted selectors is present. Startup rejects malformed non-HTTP(S) base URLs; when validation is enabled it validates connectivity and credentials through the selected SDK's model-list operation and verifies an environment-provided default model appears in that list. Callers using a minimal OpenAI-compatible install may explicitly disable SDK validation.

The concrete adapters normalize both non-streaming responses and streaming text/tool-call deltas into the harness `ModelResponse` / `ModelStreamEvent` contracts. Credentials belong at the provider/client boundary rather than in `AgentConfig` or `ModelSettings`; request-level model selection overrides the provider default.

## Inbound API and terminal client

Both surfaces ship only in the Python package; the TypeScript package provides the core runtime contracts and does not bundle an HTTP server or terminal client.

The Python inbound API layer is intentionally separate from the core runtime. `agent_rt_api` adapts OpenAI-compatible HTTP, SSE, and Responses WebSocket requests plus Anthropic Messages HTTP/SSE requests into provider-neutral `ModelMessage` inputs and delegates execution to an existing `AgentLoop`. Exposed agent IDs become API `model` IDs, so server routing does not rewrite the provider model configured inside `AgentConfig`. The Anthropic adapter maps `system`, text blocks, `tool_use`, and `tool_result` history into Agent RT messages and maps Agent RT text/tool-call outputs back into Anthropic content blocks and stream events. Request-scoped generation settings may override temperature/token limits only inside a server-owned generation cap; tools, policies, guardrails, permissions, and execution limits remain server-owned. Request-defined Anthropic tools are rejected rather than silently executed; tool capabilities must be configured in the Agent RT runtime. The inbound boundary is fail-closed for public exposure: unauthenticated binds default to loopback, request/WebSocket payloads are bounded, rate and concurrency limits are enabled, API runs have finite time/token budgets, credentials use constant-time matching, browser WebSocket origins can be allowlisted, raw internal exceptions are not returned to clients, and documentation endpoints are disabled unless explicitly enabled. FastAPI and Uvicorn are optional dependencies in `agent-rt[api]`; importing the core `agent_rt` module does not require them.

The Python terminal client follows the same boundary. `agent_rt_cli` owns terminal concerns—prompt editing/history, streaming display, transcript persistence, session/model slash commands, and one-shot/stdin operation—but delegates every model/tool turn to `AgentLoop`. Session files intentionally persist normalized Agent RT messages rather than provider-specific payloads. Rich and prompt-toolkit are isolated in `agent-rt[cli]`; the base `agent_rt` import remains independent of terminal UI libraries. The console entry point is packaged with Agent RT, but interactive startup gives actionable `agent-rt[cli]` guidance when its UI dependencies are unavailable.

Inbound rate limiting is keyed by verified credential or client address, never by unverified header values. Request validation (body shape, generation parameters, model, content-part types) completes before a streaming response starts; unsupported content parts are rejected instead of dropped; interrupted runs are reported as truncations and never leak unexecuted server-side tool calls; and unauthenticated callers learn nothing beyond liveness from `/health`. The terminal client treats Git as inspection-only in practice: writes to `.git/` and Git attribute/submodule files are refused, the Git tools refuse to run while repository config defines filter drivers, approval previews state what they hide, and recoverable tool errors go back to the model so the turn and its history are preserved.

## Vector database providers

Vector database selection follows the same provider boundary. Applications register vendor-specific `RetrievalProvider` factories in `VectorDBProviderRegistry` and select one at startup with `AGENT_RT_VECTOR_DB`. `AGENT_RT_VECTOR_DB_COLLECTION` and `AGENT_RT_VECTOR_DB_URL` carry common settings, while `AGENT_RT_VECTOR_DB_OPTION_*` exposes explicit non-secret adapter options. Vector database SDKs are intentionally not core dependencies; credentials stay in the adapter/client or secret boundary, and changing the selected backend does not change retrieval call sites.

`RetrievalRegistry` can attach a provider-neutral `RetrievalReranker` to any registered provider. Retrieval executes first, then the reranker receives the retrieved candidates. Python `reranker_k` and TypeScript `rerankerK` are optional top-k controls: when supplied, the runtime passes `k` to the reranker and enforces that cap on the reranked sequence; when omitted, the reranker has no reranker-specific result cap. The ordinary `RetrievalQuery.limit` remains the final registry-level result limit. `DecisionRetrievalReranker` supplies the built-in Decision-model implementation, while cross-encoders, hosted reranking services, or application-specific rankers can implement the same contract without adding a core dependency.

The built-in service adapters intentionally target Chroma, Milvus, Pinecone, Qdrant, and Weaviate over HTTP using the provider-neutral `EmbeddingModelProvider` contract. This avoids adding vendor SDKs to the runtime while still allowing a single `AGENT_RT_VECTOR_DB` switch. FAISS and pgvector are left as registry extensions because their normal integration path is local/native or database-driver based rather than a portable HTTP contract.

## Skills and registration scanning

Skills/plugins are versioned packages of instructions, tools, schemas, and resources. Agent RT can import the `SKILL.md` directory format used by Claude Code, Codex, and other agentic coding tools: the Markdown body becomes the skill instructions, top-level frontmatter supplies fields such as `name`, `version`, and `description`, and companion files are retained as skill resources. A missing `version` defaults to `1`, and callers may override the imported version explicitly. `SkillRegistry` accepts either one skill directory/`SKILL.md` file or a collection root whose direct child directories each contain `SKILL.md`, so locations such as `.claude/skills`, `.codex/skills`, and `.agents/skills` can be loaded without converting each skill by hand. Local skill loading is bounded by file-count/byte limits and never follows symlinked resource paths; it does not itself fetch remote content.

Before registry mutation, a deterministic registration scanner checks names, descriptions, tool schemas/metadata, complete skill frontmatter, skill instructions, embedded tool declarations, and string resources for prompt-injection language, misleading authority/selection claims, advertising/promotional steering, harmful credential/malware behavior, and suspicious privileged-looking names. Filesystem skill installation additionally requires a Decision-model registration scan and fails closed if that scan cannot complete, exceeds its configured content bound, or crosses a category threshold. External TypeScript tools use `ToolRegistry.registerChecked(...)`; Python external tools configure `ToolRegistry(registration_guard=make_decision_registration_guard(...))`. Lower-level in-process registration remains available for trusted application code but still runs the deterministic scanner.

The Python coding CLI discovers global `$AGENT_RT_HOME/skills` and project `.claude/skills`, `.codex/skills`, `.agents/skills`, and `.agent-rt/skills` roots at launch, then exposes read-only catalog/instruction/resource tools so skills are loaded progressively rather than appended wholesale to the system prompt. Project `.agent-rt/skills` has final precedence for duplicate name/version pairs. Activating a new version should be an explicit deployment choice so rollback remains possible. When no version is active or explicitly requested, registries use natural numeric version ordering so `10` is newer than `2`.

## Decision models

Decision providers are separate from generative `ModelProvider` implementations. They accept state plus typed `choice`/`score`/`noul` questions and return calibrated decisions suitable for control-plane gates rather than text generation. The built-in Jev-compatible clients target `/v1/systemone` without embedding model weights or ONNX runtimes in Agent RT. The local default remains `http://127.0.0.1:8000`, while `AGENT_RT_JEV_BASE_URL` can point to any hosted Jev-compatible service and `AGENT_RT_JEV_MODEL` selects the remote Laya or other Jev-compatible Decision model. `AGENT_RT_JEV_TIMEOUT_SECONDS` controls the request timeout. Explicit constructor configuration takes precedence over environment defaults, and the configured model is included as a top-level `model` field in Jev requests.

Decision-backed adapters are intentionally composed with existing deterministic contracts: tool-output decisions feed `GuardrailResult` and receive the latest user prompt, tool-call arguments, and tool output as Decision-model state; tool relevance feeds runtime visibility filtering while required tools remain preserved; memory relevance/sensitivity/confidence feeds the existing write policy thresholds; `DecisionMemoryGuard` treats memory as untrusted data and scores both harmful-action influence and misleading/instruction-hijacking influence before persistence or retrieval exposure; `DecisionRetrievalReranker` scores retrieved candidates for relevance and trust before prompt assembly, with an optional top-k cap; retrieval/context decisions can otherwise filter or downgrade trust; and model-response failure decisions map into the existing `FailureDisposition` taxonomy. `DecisionMemoryWritePolicy`/`DecisionMemoryWriteGate` combine the safety questions with their existing memory-quality questions in one Decision call and reject unsafe memories before store mutation. Security and persistence thresholds remain explicit runtime policy rather than being delegated entirely to model output.

`Toolbase` is the initialization boundary for large tool inventories. It combines registered and deferred local tools with tools discovered from configured MCP clients, safety-screens the complete candidate set before model exposure, then exposes only the relevance-selected subset on each turn plus the built-in `tool_search` fallback. A successful `tool_search` result promotes matching safe tools into subsequent turns without bypassing `CapabilityGrant`, prompt-injection filtering, `ToolSelectionPolicy`, context policy, or execution-time `ToolRegistry` checks. `make_decision_toolbase` / `makeDecisionToolbase` compose Decision-model safety screening with the existing relevance selector; a tool rejected by the safety screen is absent from search as well as ordinary visibility.

Model-response classification is opt-in on `AgentLoop`. A classified refusal or reported error terminates with `model_response_failure` and retains the typed failure disposition on the run result. This classification describes what the model response reports; exception-based `classify_failure` remains the source for raised runtime failures.

## Deployment and persistence

Deployment targets use a common execution abstraction so local processes, containers, Kubernetes, serverless, and managed runtimes do not require changes to agent logic.

Horizontally scaled runtime nodes should keep durable state in shared stores. Per-process memory is appropriate only for ephemeral caches and local coordination. Inbound API admission should be bounded per exposed model rather than by one global queue so an exhausted or slow model does not create head-of-line blocking for unrelated models. Provider rate-limit cooldowns should also be keyed by the concrete provider model and should fail fast until the advertised reset deadline; workers should not sleep while holding execution capacity. Deployment kinds are validated at runtime as well as by static types so dynamic-language and JavaScript callers cannot silently register unsupported targets.

Persisted payloads should carry or otherwise be associated with a non-empty schema name and a positive integer version. Forward migrations must use positive integer versions, be deterministic, and be ordered. Validated migration execution should check the source payload and each migrated version against its registered schema before proceeding. Deployments that introduce a new persistence version should test upgrade paths against representative stored data before rollout.

Tenant accounting should never permit negative aggregate usage; count-based quotas and deltas must be integers; monetary values must be finite; configured quota limits must be non-negative; and release operations should be explicit for reversible reservations such as concurrency and storage.

## Security expectations

Extensions must respect the harness permission, capability, tenant-isolation, secret-handling, guardrail, and exfiltration boundaries. A provider or plugin must not treat registration as authorization. Retrieved/file/observation context is untrusted by default in both runtimes; applications must opt data into trusted status rather than relying on an omitted trust label.

Sandbox backend selection is fail-closed. `AGENT_RT_SANDBOX_BACKEND` names a concrete backend (`native`, `docker`, or `e2b`); disabling aliases such as `none`, `off`, `false`, `0`, blank, or `null` are invalid, and `AGENT_RT_DISABLE_SANDBOX` is explicitly rejected. An unset selector defaults to `native`, but native startup requires a configured restricted UID. Native execution must not claim controls it cannot enforce: the built-in native backend changes identity and applies host resource limits, but rejects policies that require network isolation; Docker is the built-in local backend that can enforce no-network execution, while E2B remains an optional managed sandbox integration. Backends must reject unsupported network/resource policies rather than silently weakening them.

Sandbox execution must also own the lifetime of what it starts. A timeout or cancellation terminates the command (the whole process group for native execution, the named container for Docker) rather than abandoning it; captured output is bounded while it is read, not after it is buffered; privilege dropping clears supplementary groups and never inherits the caller's primary group; container backends drop all capabilities, forbid privilege escalation, and cap swap together with memory; and a resource limit of `0` is rejected because container runtimes read it as "unlimited" while rlimits read it as "nothing may run". Backends that hold remote state (E2B, microsandbox) create at most one sandbox per session and expose `close_session`/`closeSession`, surfaced as `SandboxSession.close()`. URL allowlists reject targets containing backslashes, whitespace, or control characters because URL parsers disagree on them. Snapshots, restore, and clones cover only the in-process workspace files and session settings; they are not a disk snapshot of the backend.

### Action blockers and model-backed tool input guardrails

Action blockers are a deny-only tool-execution boundary that is separate from transform-capable tool-input guardrails. A `ToolRegistry` can configure multiple blockers; they run in configuration order after permission checks and the pre-call transform, and the first blocker that rejects the effective call stops execution with `ActionBlockedError` (a `GuardrailViolationError` subtype). Blockers run before tool-input guardrails, argument validation, rate limiting, dry-run/approval handling, and the tool handler, so a blocked action never reaches approval or side effects.

The built-in `RuleBasedActionBlocker` matches command prefixes at shell-command segment boundaries in common command arguments (`command`, `cmd`, `script`, `shell`, and `argv`) and can also match normalized tool names such as `git_push`. Its default deny list is `sudo`, `rm -rf`, `git add`, `git commit`, `git push`, `git reset --hard`, `git merge`, and `git rebase`. Rules are configurable; matching token prefixes rather than arbitrary substrings avoids blocking benign strings such as `echo git push`.

Agent Action Guard is available both through the legacy opt-in model-backed tool-input guardrail and through `make_agent_action_guard_action_blocker` / `makeAgentActionGuardActionBlocker`, allowing it to compose with rule-based and custom blockers. The harness normalizes each candidate call to a function action containing the tool name and arguments. Python uses the synchronous `is_action_harmful` API; TypeScript awaits the asynchronous `isActionHarmful` API. Classifier injection remains available for deterministic testing or alternative implementations.

Decision models, including Laya/Jev-compatible providers, can be adapted with `make_decision_action_blocker` / `makeDecisionActionBlocker`. The adapter asks a typed `noul` blocking question and denies the call when the returned probability meets the configured threshold. Custom blockers can be plain functions/callables or objects exposing `check`; they may be synchronous or asynchronous in Python and synchronous or promise-returning in TypeScript.

Agent Action Guard is a third-party dependency licensed under CC BY 4.0. Its default runtime uses local ONNX inference and may download/cache embedding model assets on first classification; deployments should account for startup/network/cache behavior or configure the upstream embedding backend explicitly.

## Agent-loop invariants

The loop must leave a transcript that a provider will accept at every exit. Every tool call the model produced receives a tool message, so a run that stops for approval, timeout, cancellation, or the tool-call limit records `tool_not_executed` results for the calls it did not run, and a concurrent batch keeps the results of the siblings that completed. Run-level deadline expiry is the only thing that ends a run as `timeout`; a timeout raised by a tool's or provider's own I/O is an ordinary failure. Trajectory state (`LoopDetector`) is observed per run, and idempotency replay is keyed by call id and arguments. Token budgets and cost ledgers depend on providers reporting usage, so streaming transports must request it.

## Validation, trust, and provider-state contracts

Tool-argument and structured-output validation implements a documented JSON Schema subset in both runtimes: `type` (string or list), `enum`, `const`, `properties`, `required`, `additionalProperties` (boolean or schema), `items`, numeric bounds, `minLength`/`maxLength`/`pattern`, `minItems`/`maxItems`/`uniqueItems`, `minProperties`/`maxProperties`, `allOf`/`anyOf`/`oneOf`/`not`, and local `$ref`. Unknown keywords are ignored, so handlers still own business rules. Property checks use own-property lookups, never prototype membership.

Context assembly honours the trust label. Workflow state and `trusted` items go to a system message; `untrusted` items (the default) go to a clearly labelled user-role data message, so retrieved or tool-derived text never carries system authority. The same rule applies to model-written summaries such as compacted conversation history. MCP clients only use features the server advertised during initialization, and `MCPAccessPolicy` fails closed: configured requirements without an authenticated principal are an error, and `CapabilityGrant.tools` (when non-empty) constrains MCP tool names. Registration safety folds look-alike and invisible characters (NFKC, format characters) before pattern matching, and a registration guard that returns no verdict blocks. Tenant ids and tenant-scoped resource ids must not contain the `::` namespace separator. Privacy redaction matches secret-like keys by normalized suffix (`apiKey`, `x-api-key`, `client_secret`, `refresh_token`) rather than exact name. `PIIMorpher` then pseudonymizes detected PII into deterministic synthetic values while preserving repeated references: the internal lookup key is a 20-hex fingerprint formed from the first 10 hex characters of SHA-256 plus the first 10 of MD5, and only fingerprint → fake-data mappings are retained. Built-in detection is deliberately conservative (email, North American phone numbers, SSNs, Luhn-valid payment-card numbers, and IPv4), structured field-name hints cover common person/address/date fields, and callers can supply explicit `PIIEntity` spans from a stronger external recognizer. `AGENT_RT_DISABLE_PII_MORPHER=true` disables morphing. The unkeyed fingerprint is a deterministic lookup mechanism, not cryptographic tokenization; deployments that need resistance to dictionary attacks or reversible pseudonymization should place keyed HMAC/AES-SIV tokenization at the appropriate data boundary.

Provider adapters convert media content parts instead of dropping them: images, PDFs/files, and (OpenAI) audio are sent as the provider's native blocks, and parts a provider cannot carry (video, media in system/tool roles where unsupported) raise rather than vanish. Provider-private state, currently Anthropic `thinking`/`redacted_thinking` blocks, is preserved as a JSON content part with a vendor MIME type, replayed verbatim ahead of `tool_use` on the next turn, and ignored by other providers and by structured-output extraction. A truncated Responses WebSocket response (`response.incomplete`) is a normal result with `finish_reason: "length"` and usage, not a failure that tears down the connection. Server-supplied rate-limit waits are capped.

## Runtime state expectations

Work queues dead-letter items whose every lease attempt expired; schedulers catch up arithmetically; event dispatch marks an event delivered only after every rule was enqueued; background-task cancellation before start is recorded, and cancelling a waiter does not cancel the job. Event ids skip ids hidden by retention. Transaction commit failures run every remaining compensation, then re-raise the original error annotated with `compensated`/`compensation_errors` (`compensationErrors`). Filesystems refuse to delete their root (use `clear()`), treat moving a file onto itself as a no-op, and use `*`/`?` within one segment and `**` across segments in globs; unified patches treat in-hunk `---`/`+++` lines as content and honour zero-length hunk ranges. Deferred tool registrations can carry their handlers, a rejected registration leaves the deferred entry intact, and a blocked or failed output guardrail still emits a terminal audit event. Blocking Decision-model hooks run off the event loop (Python: `acall` twins), response caches are size-bounded, and single-flight entries live as long as the shared task rather than any one waiter.

## Caching, single-flight, and cost-policy contracts

Caches must only reuse results when the operation is safe to reuse, and tenant or security boundaries must be reflected in validated cache namespaces/keys. Callers that need to distinguish a miss from a cached null/undefined value should use the explicit lookup result rather than the convenience getter. Single-flight request collapsing must only combine semantically identical operations, and failed operations must be evicted so later calls can retry. Side-effecting tool calls should not be cached or coalesced unless the application has an explicit idempotency contract. Cost policies must reject non-finite budgets/costs, malformed model tiers, invalid confidence values, and fractional token/context limits before making routing or early-exit decisions. Cross-language deterministic controls, including percentage feature-flag rollout, must operate on the same UTF-8 byte representation so identity decisions do not diverge by runtime.

## Conformance testing

Use `ConformanceSuite` to define stable behavioral checks for provider adapters, persistence implementations, protocol adapters, tool schemas, and extension APIs. Conformance suites should assert externally observable behavior rather than internal implementation details.

Typical provider checks include required callable methods, supported request/response shapes, error behavior, cancellation behavior when applicable, and deterministic serialization boundaries.

Conformance checks are not a replacement for integration tests against real infrastructure. They are a shared minimum contract that keeps adapters substitutable.

## Harness safety evaluation

Repository-level harness evaluation uses Pro-GenAI HarmActionsEval as the default safety benchmark for harmful model/tool behavior. It evaluates manipulated harmful/unethical prompts against matching function tools and reports upstream `SafeActions@k` and `HarmActions@k` based on whether the model response actually emits the harmful call; a refusal in response text does not override an emitted harmful tool action. Python invokes the upstream package directly, while the TypeScript repository runner delegates to the sibling Python adapter so both runtimes share one benchmark implementation. The older `ToolRegistry` dispatch benchmark remains available explicitly as the `performance` benchmark. HarmActionsEval requires `OPENAI_MODEL` and OpenAI-compatible credentials; both runners retain JSON output for automation.

## Performance benchmarking

Use `BenchmarkRunner` for repeatable microbenchmarks around agent-loop overhead, tool dispatch, serialization, checkpointing, memory operations, concurrency, streaming, caching, batching, and provider adapters.

Benchmarks should include warmups, enough iterations to reduce noise, a non-empty named category, integer iteration counts, and an explicit workload. Latency critical-path analysis must reject cyclic dependency graphs rather than treating them as valid execution plans. Compare results only under comparable runtime, hardware, interpreter/Node version, and dependency conditions.

Benchmark thresholds should normally live in CI or release policy rather than the generic benchmark runner so projects can choose appropriate budgets for their environment.

## Operational health

Use `HealthRegistry` for dependency checks across model providers, stores, queues, sandboxes, connectors, and runtime dependencies. Health-check kinds are runtime-validated, and duplicate `(kind, name)` registrations require an explicit replacement choice so accidental probe shadowing is visible.

Liveness answers whether the process can continue serving health traffic. Readiness answers whether the runtime can safely accept work. A dependency failure may make readiness false while leaving liveness true so an orchestrator can stop routing traffic without restarting a healthy process unnecessarily.

Health checks should be bounded and lightweight. Registry runs may apply a timeout so a stalled dependency becomes not-ready rather than blocking the health surface indefinitely. Expensive end-to-end probes belong in separate synthetic monitoring.

Health endpoints should avoid exposing secrets, credentials, raw prompts, user content, or sensitive dependency responses. Health detail should be diagnostic but bounded.

## Environment variable reference

Variables named elsewhere in the docs, collected in one place. Credentials belong at the provider/client or secret boundary and are never part of `AgentConfig`.

| Variable | Purpose |
| --- | --- |
| `OPENAI_BASE_URL` | Selects the OpenAI provider (takes precedence over Anthropic) and sets its endpoint. |
| `OPENAI_API_KEY` | OpenAI credential; required for the default `https://api.openai.com/v1` endpoint. |
| `OPENAI_MODEL` | Default OpenAI model; alone, it is accepted as an OpenAI provider hint. |
| `OPENAI_WEBSOCKET` | `0` selects the HTTP/SSE transport instead of the Responses WebSocket default (native environment startup only). |
| `ANTHROPIC_BASE_URL` | Selects the Anthropic provider when `OPENAI_BASE_URL` is not set. |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Anthropic credential and default model; neither alone selects a provider. |
| `AGENT_RT_SANDBOX_BACKEND` | Concrete sandbox backend (`native`, `docker`, `e2b`); disabling aliases are invalid. |
| `AGENT_RT_DISABLE_SANDBOX` | Explicitly rejected. |
| `AGENT_RT_VECTOR_DB` | Selects the registered vector database provider. |
| `AGENT_RT_VECTOR_DB_COLLECTION`, `AGENT_RT_VECTOR_DB_URL` | Common vector database settings. |
| `AGENT_RT_VECTOR_DB_OPTION_*` | Explicit non-secret adapter options. |
| `AGENT_RT_JEV_BASE_URL`, `AGENT_RT_JEV_MODEL`, `AGENT_RT_JEV_TIMEOUT_SECONDS` | Decision-model service endpoint (default `http://127.0.0.1:8000`), model, and request timeout. |
| `AGENT_RT_WEB_SEARCH_TOOL` | Web-search backend for compatibility clients (`tavily`, `brave`, `serper`). |
| `TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY`, `SERPER_API_KEY` | Backend-specific web-search credentials. |
| `AGENT_RT_WEB_SEARCH_API_KEY`, `AGENT_RT_WEB_SEARCH_TOKEN` | Fallback web-search credentials, in that precedence order. |
| `AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS` | Web-search HTTP timeout (default `20`). |
| `AGENT_RT_HOME` | Global Agent RT home; the Python coding CLI reads `$AGENT_RT_HOME/skills`. |
| `PYTHON` | Interpreter the TypeScript benchmark runner uses for HarmActionsEval. |

## Contract evolution and migration

Framework migration (LangChain, LlamaIndex, OpenAI/Anthropic SDKs, and others) is covered in `MIGRATION.md`. This section covers evolving Agent RT's own extension contracts.

When adding a new extension contract, keep Python and TypeScript semantics aligned and add focused tests in both languages. Update package exports, publish/test commands, README capability lists, and AGENTS guidance.

When changing an existing contract, prefer additive changes. If a persisted shape changes, add a new schema version and forward migration rather than mutating stored data assumptions in place. If an extension API must change incompatibly, document the old and new forms and provide a transition path before removing the old contract.

## Example lifecycle

A typical production integration follows this sequence:

1. Implement a provider, skill, deployment adapter, or persistence backend against the stable interface.
2. Run its conformance suite and focused unit tests.
3. Register it through configuration or the extension registry.
4. Add health checks for external dependencies.
5. Benchmark critical paths under representative load.
6. Roll out behind environment configuration or a feature flag when appropriate.
7. Observe traces, metrics, evaluation results, quota usage, and health.
8. Roll back prompt/plugin/config versions or deployment selection if the change regresses behavior.
