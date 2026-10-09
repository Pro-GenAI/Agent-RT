# Introduction to Agent Harnesses

An **agent harness** is the runtime and control layer around an AI model that turns a model from a text generator into a system that can reliably perform multi-step work.

The model supplies reasoning and generation. The harness supplies everything required to make that reasoning useful, controllable, persistent, observable, secure, and executable.

A useful mental model is:

```text
Agent Harness
=
Model interface
+ agent loop
+ context engineering
+ tools
+ state
+ workspace
+ execution isolation
+ orchestration
+ persistence
+ permissions
+ human control
+ failure recovery
+ observability
+ evaluation
+ interoperability
+ deployment
```

A harness is therefore much broader than prompting. It determines what an agent can see, what it can do, how long it can work, how it recovers from failure, how it collaborates with other agents, and how humans can inspect and control its behavior.

## Why a harness is needed

A raw language model can generate text, code, structured data, and tool-call requests. By itself, however, it does not provide a complete execution system. A production agent usually needs state, tools, files, controlled code execution, recovery, approvals, resumption, delegation, security boundaries, traces, cost accounting, and evaluation. The harness provides these capabilities.

## High-level architecture

A complete harness can be viewed as several cooperating planes:

```text
                         ┌────────────────────────────┐
                         │      User / Application    │
                         │ Chat · CLI · IDE · API     │
                         └─────────────┬──────────────┘
                                       │
                          auth / session / request
                                       │
                 ┌─────────────────────▼─────────────────────┐
                 │              CONTROL PLANE                │
                 │ Agent definition · Planner · Routing      │
                 │ Policy · Budgets · Human approvals        │
                 └──────────┬──────────────────┬─────────────┘
                            │                  │
                 ┌──────────▼───────┐  ┌──────▼──────────────┐
                 │   MODEL PLANE    │  │   EXECUTION PLANE   │
                 │ Providers        │  │ Tools · MCP          │
                 │ Context builder  │  │ Shell · Filesystem   │
                 │ Structured I/O   │  │ Browser · Sandbox    │
                 │ Compaction       │  │ Remote agents        │
                 └──────────┬───────┘  └─────────┬───────────┘
                            │                    │
                      ┌─────▼────────────────────▼─────┐
                      │           STATE PLANE          │
                      │ Messages · Sessions · Memory   │
                      │ Checkpoints · Files · Artifacts│
                      │ Task state · Event log         │
                      └──────────────┬─────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     RELIABILITY / GOVERNANCE    │
                    │ Permissions · Guardrails        │
                    │ Retry · Idempotency · Secrets   │
                    │ Audit · Isolation · Rollback    │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │       OPERATIONS / QUALITY      │
                    │ Tracing · Logs · Metrics        │
                    │ Evals · Regression · Cost       │
                    └─────────────────────────────────┘
```

## Harness components

### 1. Agent definition

Defines identity, role, instructions, dynamic instructions, model settings, enabled tools, permissions, output schemas, memory scope, and termination rules.

### 2. Model abstraction and routing

Provides provider-neutral model interfaces, multiple providers, aliases, per-agent models, fallback chains, capability metadata, reasoning settings, and routing based on capability, cost, latency, or context size.

### 3. Core agent loop

Runs the model → tool calls → tool results → model continuation loop until completion. The loop also needs max turns, explicit termination, cancellation, time and cost budgets, and loop detection.

### 4. Planning and reasoning orchestration

Supports plans, todo lists, task decomposition, dependencies, milestones, explicit completion and acceptance criteria, progress tracking, and replanning that preserves completed work when constraints change. Planner/executor separation keeps decomposition distinct from bounded execution. Verification should evaluate whether planned steps and criteria are actually satisfied before success is declared. Reflection passes can inspect and repair defects in intermediate or final work, while dedicated critic/verifier agents should review independently rather than owning primary execution.

### 5. Tool system

Includes typed function tools, schemas, registration, namespaces, argument validation, dispatch, result marshaling, concurrency, sequential constraints, timeouts, lifecycle hooks, and dependency injection. A tool runtime should validate arguments before side effects, associate every result with the originating call, enforce tool-local deadlines independently from run deadlines, preserve deterministic result ordering for concurrent batches, and allow sequential tools to act as execution barriers when ordering or side effects require it.

### 6. Tool discovery and policy

Supports allowed and denied tools, required or preferred tools, dynamic visibility, deferred loading, semantic capability search, and context-sensitive filtering. Capability discovery should use a provider-neutral catalog that can index tools, skills, connectors, files, and agents while delegating ranking to a pluggable scorer, so simple deterministic matching and embedding-backed semantic ranking can share one contract. A harness should apply visibility controls both when deciding which schemas are visible to the model and again at dispatch so hidden capabilities cannot be invoked by a stale or malformed model call. Runtime filters can incorporate agent state, permissions, environment, task phase, and other request context. Deferred registrations keep expensive or rarely used schemas out of model context until explicitly materialized. Large harnesses should avoid exposing every tool on every turn.

### 7. Programmatic tool composition

Allows generated code or an executor to orchestrate several tools without an LLM round trip between every call, reducing latency and token usage for mechanical workflows. Programmatic execution should preserve call/result association, cancellation and deadline semantics, lifecycle hooks, and the same dependency-injection boundary used by model-requested tool calls.

### 8. Context engineering

Builds each model request from instructions, messages, selected workflow state, retrieved knowledge, files, tool descriptions, and runtime metadata. Context selection should be explicit and composable: each call may receive a narrowed view of history, state, tools, files, observations, and metadata while durable state remains separate from model-visible context.

### 9. Context isolation, compaction, and offloading

Isolates subwork by deriving a smaller context view rather than copying the entire parent context. Selection boundaries should preserve least privilege for tools and data. Compaction should retain a useful recent tail while transforming older material through an explicit policy, and offloading should replace oversized inline material with durable references that can be retrieved when needed. Stable instructions and tool schemas should be separated from volatile turn context so provider adapters can exploit prompt-prefix caching without changing runtime semantics.

### 10. Memory

May include short-term session memory, long-term factual memory, semantic/retrieval memory, episodic task history, and procedural memory such as reusable skills.

- Short-term memory should be scoped by stable session and thread identities so concurrent conversations do not leak into one another, and retention limits should be explicit.
- Persistent memory should expose explicit read, write, update, delete, and search semantics over scoped records rather than relying on opaque provider behavior.
- Semantic memory should separate embedding generation from storage and ranking, while episodic memory should preserve useful task decisions, actions, outcomes, and traces in a form that can be searched or summarized later.
- Procedural memory can preserve reusable instructions, scripts, templates, or learned workflows.
- Memory writes should pass explicit relevance, sensitivity, confidence, duplication, and user-policy checks; retrieval should rank candidates using relevance, recency, confidence, scope, and task context.
- Scope boundaries should be enforceable at the storage view itself so callers cannot accidentally cross user, tenant, agent, project, workspace, or task boundaries.
- Lifecycle controls should make retention, expiration, deletion, compaction provenance, and schema migration explicit rather than leaving stale records to provider-specific behavior.

### 11. State management

Typed workflow state can hold task variables, intermediate outputs, plan state, counters, approvals, tool results, artifacts, and execution metadata without reconstructing everything from chat history.

- Durable state should use an explicit type identifier and schema version, serialize through a constrained durable format, reject values that cannot round-trip safely, and remain distinct from conversational message history.
- Checkpoints should capture enough execution state to resume safely after interruption, including message history, durable workflow state, and already-consumed execution budgets so resumed work does not silently reset limits.
- Durable execution should also keep an append-only event history for important state changes, model/tool activity, approvals, checkpoints, and lifecycle transitions.
- Replayed operations with external side effects should use stable idempotency keys so retries or resumed runs can reuse the original result rather than duplicating the side effect.
- Task lifecycle transitions should be explicit and validated instead of inferred from arbitrary status strings.
- Long-running work should expose a queryable task handle with progress, cancellation, result, and failure state rather than depending on one synchronous request.
- Queue abstractions should make priority, worker leases, retry attempts, visibility delays, and concurrency limits explicit.
- Scheduled execution should distinguish one-shot deadlines from recurring cadence and advance recurring schedules deterministically after each due run.

### 12. Checkpointing and durable execution

Persists execution state so tasks can resume after failure, restart, disconnection, or long waits. It can also support sessions, background jobs, queues, scheduling, and event-triggered execution.

### 13. Idempotency

Prevents retries or resumed execution from duplicating side effects such as messages, payments, deployments, or database records.

### 14. Filesystem and workspace

Provides list, read, write, edit, patch, search, glob, move, copy, and delete operations through a backend-neutral filesystem contract so the same workspace APIs can target local, remote, memory-backed, object-backed, container-backed, or sandbox-backed storage. Persistent workspaces should use stable workspace identities so files survive turns and resumed executions, while backend selection remains pluggable. Paths should be normalized within an explicit root, creation should distinguish overwrite from create-only behavior, and structured edits should verify expected context or occurrence counts before mutating files so agents do not silently patch the wrong revision.

### 15. Artifacts

Treats reports, source code, datasets, documents, images, archives, and other outputs as first-class objects with media type, metadata, versioning, lineage, parentage, provenance, retention, location, and lifecycle state. Artifact repositories should distinguish mutable draft revisions from finalized outputs, preserve revision ancestry and useful diffs, enforce retention before deletion, and attach structured provenance identifying relevant sources, models, tools, agents, or transformations.

### 16. Sandboxed execution

Runs generated commands and code behind an explicit sandbox backend rather than directly on the host.

- A sandbox session should own its workspace, working directory, environment, runtime/package state, network policy, and resource policy, while the backend is responsible for hard isolation and enforcement of CPU, memory, disk, process, and network restrictions.
- Backend selection may be environment-driven, but it must fail closed: the backend variable names a concrete isolation mechanism and must not accept `none`, `off`, or similar aliases as a way to bypass isolation.
- If native execution is offered through a restricted OS identity, unsupported isolation dimensions (for example, network namespaces) must be rejected rather than silently treated as enforced.
- Container and managed-sandbox backends should likewise reject policy modes they cannot implement.
- Untrusted retrieved or tool-derived content, including summaries a model writes of it, should reach the model as labelled data rather than with system authority.
- A sandbox owns the lifetime of what it starts: timeouts and cancellation must terminate the command rather than abandon it, output must be bounded as it is read, and privilege reduction must remove inherited groups and primary-group membership, not only the user id.
- The harness can additionally enforce execution deadlines and output caps before results are returned.
- Snapshots should capture workspace files plus session/runtime state so restore and clone/branch operations are deterministic and isolated from later mutations.

### 17. Shell and code execution

May expose shell commands, Git, package managers, build/test tools, Python, JavaScript/TypeScript, notebooks, and other runtimes through the sandbox boundary. Language runtimes should be registered behind a common interpreter contract so implementations can support persistent per-session state. Environment management should make runtime versions, installed dependencies, variables, and working directories explicit rather than relying on process-global state. Network access should also be explicit: no-network should be the safe default, allowlists should match domains and subdomains deterministically, blocked domains should override broader access, and proxy settings should be propagated to the sandbox backend rather than implemented as hidden global process state.

### 18. Browser, search, and computer use

Can include web search, browser navigation, extraction, clicking, form entry, downloads, screenshots, mouse/keyboard control, and general GUI interaction.

### 19. Multimodality and realtime interaction

Can support text, images, PDFs, documents, audio, video, mixed-content messages, audio streaming, speech input/output, interruption handling, and turn detection.

### 20. Human-in-the-loop controls

Allows execution to pause for missing information, review, edits, or approval. Approval scopes can include allow once, deny once, session grants, durable grants, and revocation.

### 21. Permissions and least privilege

Controls tools, paths, network access, agents, data, operations, and side effects. Permission checks should run outside model text, normalize paths before matching read/write/execute rules, and let explicit deny rules override broader grants. The harness should expose only the capabilities needed for the current task.

### 22. Authentication, authorization, and secrets

Represents user identity, service identity, OAuth delegation, API/session credential references, scopes, roles, attributes, resource ownership, and tenant boundaries. Authentication state should carry identity and credential metadata without embedding secret material; authorization should evaluate scopes, roles, attributes, ownership, delegation, and tenant constraints independently of model text. Secrets should stay in dedicated stores, require explicit retrieval/reveal, support expiration and narrow credential grants, and expose metadata-only references to model-visible context. Tenant isolation should namespace sessions, credentials, workspaces, task/event streams, quota keys, and memory so identical application-level identifiers cannot collide across tenants.

### 23. Guardrails and policy

Guardrails can inspect user input, model output, tool input, and tool output. Each guardrail may allow a value unchanged, classify it, transform/redact it, or block it. Input guardrails should run before application messages enter model history; output guardrails before model responses become final/history state; tool-input guardrails after tool-call transforms but before validation and side effects; tool-output guardrails before results are audited or exposed back to the model. A policy engine can enforce business, safety, security, and compliance rules outside the model.

### 24. Prompt-injection and exfiltration defenses

External data should remain explicitly labeled untrusted data instead of silently becoming instructions. Context assembly should preserve that trust boundary in model-visible payloads, and untrusted context should not automatically gain access to write, consequential, or destructive capabilities. Authorization must remain independent of model-generated text. Sensitive data should carry an explicit classification and be checked against allowed egress channels and targets so restricted information cannot leave through tools, connectors, network calls, logs, or model-visible paths.

### 25. Side-effect management

Classifies tools as read-only, reversible, consequential, or destructive. Consequential workflows can use explicit prepare → review → approve → commit flows. Approval checkpoints should stop before side effects, persist grants at appropriate once/session/durable scopes, support revocation, and expose a distinct waiting state rather than treating review as an execution error. Dry runs should produce intended-action previews without consuming approval or committing side effects. Multi-step transactions should separate preparation from commit and execute compensating actions in reverse order when a later committed step fails. Governance around those actions should preserve actor-attributed audit records for requests, authorization decisions, execution, changes, and cancellation. Sensitive values should be redacted before persistence or export rather than only masked at display time. Durable event storage should support retention TTLs, deletion/purge, archival, ephemeral runs, and selective logging suppression.

### 26. Failure handling

Distinguishes transient dependency failures, model-correctable errors, user-correctable conditions, authorization/policy failures, and terminal system errors. Recovery policy should be explicit about whether to retry, repair arguments, choose an alternate tool, switch models, request user input, or escalate. Retry behavior should be bounded and operation-specific, with exponential backoff and optional jitter. Repeated dependency failures should trip circuit breakers that recover through cooldown or health checks instead of continuing to hammer degraded services.

### 27. Runtime controls

Includes timeouts, retries, circuit breakers, rate limits, execution budgets, token and cost limits, deadlines, loop detection, and deterministic cleanup.

- Rate limits should be enforceable at relevant boundaries such as user, tenant, model, tool, and provider scopes, with rejected multi-scope checks applied atomically so one failed quota does not consume another.
- Execution budgets should be shareable across nested work so model calls, retries, subagent slots, elapsed time, tokens, tool calls, and estimated monetary cost draw from a common envelope.
- Whole-task deadlines should propagate into model and tool operations rather than being re-created independently at each layer.
- Argument and output validation should enforce the schema constraints it advertises (types, bounds, patterns, composition) rather than silently ignoring them.
- Repeated equivalent trajectories should terminate as loops, with detector state scoped to a single run, and registered finalizers should execute deterministically in reverse order on success, failure, timeout, or cancellation.
- Only the run's own deadline ends a run as a timeout; a timeout inside a tool is a tool failure.
- Whatever way a run stops, its transcript should remain valid for the next model call: every tool call gets a result, even if that result says it was not executed.
- Budgets that count tokens or cost require usage to be reported on every transport, including streaming.

### 28. Multi-agent orchestration

Can include agents-as-tools, handoffs, isolated subagents, supervisor/worker teams, routers, round-robin teams, model-selected speakers, swarms, hierarchical teams, parallel workers, map/reduce, and speculative branches.

- Agent-as-tool composition should keep the parent in control of the conversation after bounded specialist work, while a handoff should explicitly transfer ownership and the context being transferred.
- Isolated subagents should receive only declared messages, context, tools, state, permissions, and budgets, and parent execution budgets should account for spawned workers.
- Supervisor/worker teams should preserve assignment identity and worker scope, while routers should select specialists from explicit task intent or capability requirements.
- Round-robin teams should make speaker order deterministic; coordinator-selected speaker teams should validate the chosen member; peer swarms should bound handoff depth; and hierarchical teams should compose child teams recursively through a uniform execution contract.
- Parallel fan-out should preserve mapper result order even when workers overlap, and map/reduce orchestration should make the reduction stage explicit rather than implicitly merging worker outputs.
- Speculative branches should remain isolated until evaluation, use explicit scoring or selection criteria, resolve ties deterministically, and require an explicit merger when alternatives are combined rather than selected.
- Multiple agents are most useful when they provide specialization, parallelism, independent verification, permission separation, or context isolation.

### 29. MCP

The Model Context Protocol connects a harness to external tools, resources, prompts, and capabilities. A harness-side MCP client should negotiate server capabilities before discovery or use, keep transport concerns separate from policy, and consume tools, resources, and prompts through explicit protocol operations. Authentication scopes, delegated credential grants, and local policy should be checked both when capabilities are exposed and when they are invoked or read. Per-agent or per-task capability filters should narrow remote tools, resources, and prompts before model exposure rather than relying only on downstream denial.

### 30. Agent-to-agent interoperability

Agent-to-agent protocols allow independent agent systems to discover each other, exchange messages, create and track remote tasks, stream status, exchange artifacts, and receive asynchronous callbacks. Protocol adapters should isolate REST/OpenAPI/GraphQL/WebSocket/gRPC mechanics behind a consistent request contract, while connector definitions map product-specific operations onto those adapters. Remote agent discovery should return explicit agent cards containing endpoint, modalities, and capabilities. Remote task and artifact records should preserve their wire-facing status/version semantics separately from local lifecycle models, with streamed events updating durable remote state. Optional features should be negotiated from the intersection of local and remote capabilities, and required features should fail explicitly when unavailable.

```text
MCP:
agent/runtime ↔ tools, resources, external capabilities

Agent-to-agent protocol:
agent system ↔ another agent system
```

### 31. Protocol adapters and connectors

External capabilities may be exposed through REST, OpenAPI, GraphQL, WebSocket, gRPC, MCP, agent-to-agent protocols, or custom connectors for source control, email, chat, drives, databases, and enterprise applications. Browser automation and computer-use providers should expose controlled, stateful action contracts rather than leaking vendor-specific driver APIs into the agent loop. Search and retrieval providers should normalize web, enterprise, file, knowledge-base, and database sources behind common query/result contracts. Multimodal messages should preserve typed text, image, PDF, audio, video, document, file, and structured JSON parts with media metadata.

### 32. Structured outputs

Schema-constrained outputs provide stable contracts for downstream code. The harness can support typed schemas, parsing, validation, repair, and explicit validation errors.

### 33. Streaming and progress

Can stream text, tool calls, tool results, task-state transitions, subagent progress, artifacts, and approval requests as machine-readable events. Realtime and voice runtimes additionally need low-latency audio input/output events, speech-start and speech-stop turn markers, interruption handling, response completion, and deterministic transport cleanup. User-facing surfaces should translate runtime state into understandable status events that expose active work, completed steps, pending approvals, waiting conditions, and resumable task/session identity. CLI, API, IDE, and chat adapters should share stable runtime contracts rather than reimplement execution semantics; messaging adapters should preserve user and thread identity, and approval surfaces should present consequences, relevant diffs, and explicit allow/deny choices before consequential actions.

### 34. Observability

Includes traces, hierarchical spans, structured logs, metrics, correlation IDs, latency, errors, retries, token accounting, and cost accounting. Trace structure should preserve task → agent → turn → model/tool/subagent relationships while still allowing guardrail, queue, and remote-service spans to participate in the same trace. Structured logs should carry severity and correlation/task/session identifiers and should pass through privacy protection before persistence, including secret redaction and deterministic synthetic substitution for detected PII when referential consistency is useful. Metrics should support labeled counters, gauges, and histograms for latency, throughput, errors, retries, tool success, queue time, and completion. Token and cost ledgers should attribute usage and spend to relevant user, tenant, task, agent, model, tool, storage, compute, and external-service scopes.

```text
task
 └─ agent
     └─ turn
         ├─ model call
         ├─ tool call
         └─ subagent
```

### 35. Replay and debugging

Captured state, checkpoints, events, model/tool inputs, and recorded external outputs can support execution replay and deterministic debugging of nondeterministic workflows. Debug replay should preserve the original state/event/checkpoint bundle for inspection, while deterministic replay should substitute recorded model, tool, or external outputs in stable sequence and reject mismatched inputs when reproducibility requires exact correspondence.

### 36. Evaluation

Can evaluate final outputs, schemas, tool selection, tool arguments, execution trajectories, safety behavior, policy compliance, and regressions. Datasets can contain golden successes, known failures, adversarial cases, production incidents, and reviewed feedback.

### 37. Prompt and configuration management

Prompts and runtime configuration can be versioned, deployed, compared, rolled back, separated by environment, and controlled through feature flags.

### 38. Skills, plugins, middleware, and extensions

Reusable packages may contain instructions, tools, schemas, templates, scripts, and resources. Extension APIs can support custom model adapters, memory stores, filesystems, sandboxes, queues, policies, telemetry exporters, and evaluators.

### 39. Deployment and scaling

A harness may run locally, in containers, on Kubernetes, in serverless infrastructure, or on managed platforms. Scaling concerns include durable shared state, queues, distributed workers, task leases, horizontal scaling, multi-tenancy, quotas, and schema migrations.

### 40. Caching and optimization

May include prompt caching, safe response caching, retrieval caching, tool-result caching, deduplication, batching, parallelism, prefetching, early exits, model tiering, context reduction, and critical-path optimization.

Optimization must preserve correctness, permissions, freshness requirements, and side-effect semantics.

## Foundational feature set

A practical implementation should prioritize the capabilities that other features depend on:

1. agent loop and typed tools;
2. explicit workflow state;
3. context construction and compaction;
4. filesystem and artifact abstractions;
5. persistent checkpoints and resume;
6. sandboxed shell/code execution;
7. tool permissions and human approval;
8. retries, timeouts, idempotency, and termination;
9. tracing, usage, and cost accounting;
10. structured output validation;
11. MCP integration;
12. isolated subagents;
13. evaluation and regression testing;
14. authentication and secrets boundaries;
15. streaming and cancellation.

More elaborate swarm, debate, and hierarchical-team patterns should generally come later because they depend on most of these lower-level capabilities.

## Design principles

### State and context are different

Persist complete durable task state separately. Construct each model context from only the information relevant to the current step.

### Tools are security boundaries

A model requesting a tool call does not mean the action is authorized. Validation, permissions, guardrails, and approvals must execute outside the model.

### Checkpointing requires idempotency

Resume support is not enough. Side effects must be safe to retry or replay without creating duplicate real-world actions.

### Context quality drives long-horizon reliability

Compaction, retrieval, offloading, and subagent isolation are core runtime capabilities because irrelevant accumulated history can degrade agent performance.

### Multi-agent systems should solve a concrete problem

Additional agents add communication overhead, latency, cost, synchronization, and new failure modes. Use them when the benefits are concrete.

### Observability should exist from the beginning

Without traces, state transitions, tool inputs, and execution metadata, nondeterministic agent failures are difficult to diagnose or reproduce.

### Evaluation should grade behavior, not only final text

A useful evaluation layer separates execution from grading so the same scenarios can compare prompts, models, tools, policies, and orchestration strategies. Deterministic assertions should cover exact outcomes and schemas, while model-based judges should use explicit rubrics. Tool selection, arguments, trajectories, safety boundaries, production failures, and versioned golden cases should remain first-class evaluation inputs rather than being reduced to final-answer similarity.

Repository-level harmful-action benchmarking, including the default external safety benchmark and its metrics, is described in `Architecture-and-Extensions.md` and `DEVELOPMENT.md`; it is a separate measurement from local dispatch performance benchmarks.

### Operational configuration should stay outside agent logic

Prompt versions, environment-specific configuration, feature flags, reusable skills, middleware, and provider adapters should be controlled through explicit registries and deployment state rather than hard-coded into agent behavior. This keeps experiments reversible, lets environments diverge without source changes, and gives extensions stable contracts that can be tested independently of the core loop. Provider connection settings loaded from environment should be validated at startup before agent work begins: validate endpoint syntax, use the selected provider SDK to verify authenticated model-list connectivity, and confirm any configured default model is available. Provider selection should also be deterministic rather than silently changing vendors when no configuration is present.

### Deployment should preserve agent semantics

Local processes, containers, Kubernetes, serverless functions, and managed runtimes should implement the same execution contract so agent logic does not encode infrastructure assumptions. Horizontal scaling requires durable state outside individual runtime nodes, explicit distributed-worker routing, tenant-aware accounting, and versioned persistence schemas with forward migrations so state remains portable across releases.

### Optimization needs explicit safety and budget contracts

Caching, request coalescing, batching, latency shortcuts, and cost routing should be opt-in policy decisions rather than hidden runtime behavior. Cache namespaces and TTLs must preserve isolation and freshness; single-flight reuse should only collapse operations that are semantically identical; batching must preserve input/output correspondence; latency policies should make parallelism and speculation explicit; and cost policies should enforce quality floors and budgets while reducing context or exiting early only under declared rules.

### Interoperability belongs at the boundary

Protocols such as MCP and agent-to-agent standards should be adapters around stable internal contracts rather than defining the harness's internal architecture.

## Agent RT design goals

Agent RT (Agent Runtime) is intended to provide a fast, lightweight runtime for AI agents that remains provider-neutral and explicit about runtime behavior. The concrete module map, extension contracts, security expectations, conformance guidance, operational health semantics, environment variable reference, and extension-contract evolution notes are maintained in `Architecture-and-Extensions.md`. Contributor workflows live in `DEVELOPMENT.md`, and moving existing LangChain, LlamaIndex, OpenAI, Anthropic, and other framework code onto Agent RT is covered in `MIGRATION.md`.

Its internal contracts should separate agent configuration, model interaction, tool execution, routing, state, and termination concerns so provider-specific details do not leak into application-level agent code. Higher-level capabilities should build on these stable boundaries rather than coupling the harness to a particular model vendor or orchestration style.

The harness should prefer small composable primitives, deterministic control surfaces, clear failure semantics, and observable execution over opaque framework behavior. Python and TypeScript implementations should follow the same conceptual model even when language-specific APIs differ.

### Decision models

Agent RT keeps generative models separate from *Decision models* (for example Laya and other Jev-compatible models), which answer typed choice and score questions with calibrated outputs. Decision models serve as optional control-plane gates, such as tool safety screening, relevance pruning, memory screening, and tool-output checks, that feed deterministic runtime policy rather than replacing it. For large tool inventories, `Toolbase` combines local/deferred and MCP-discovered tools, applies safety filtering before relevance selection, and exposes `tool_search` so the agent can recover safe tools omitted from the initial subset. Provider contracts, endpoints, and security thresholds are described in the Decision models section of `Architecture-and-Extensions.md`.
