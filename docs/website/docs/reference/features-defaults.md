> **Canonical HTML page:** [Agent RT Features, Defaults & Usage Reference](https://agent-rt.pro-genai.com/docs/reference/features-defaults)

> Complete Agent RT reference for defaults and usage: loop, tools, guardrails, permissions, memory, orchestration, sandboxing, retrieval, observability, and APIs.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Reference/Features & defaults
 Reference

# Features, defaults
& usage

A self-contained map of Agent RT. Use this page to understand what the runtime does, what happens by default, what requires explicit configuration, and which object or setting controls each capability.

 **Default does not always mean enabled.**Agent RT distinguishes built-in safe defaults from optional integrations. For example, conservative conversation-boundary input/output guardrails are enabled by default, while the model-backed tool-input guardrail is opt-in. When enabled, its default model is `agent-action-guard`.

## Default behavior at a glance

| Area | Default | How to change it |
| --- | --- | --- |
| Run turns | `16` maximum model turns | Set `AgentRunLimits.max_turns` / `maxTurns`. |
| Tool calls | `64` maximum tool calls per run | Set `AgentRunLimits.max_tool_calls` / `maxToolCalls`. |
| Concurrent tool calls | Off | Enable in run limits. Sequential tools still act as barriers. |
| Run timeout | No run-level timeout unless configured | Set run limits, a `Deadline`, and/or tool-local timeouts. |
| Total-token cap | No explicit run cap unless configured | Set `max_total_tokens` / `maxTotalTokens`. |
| Conversation-boundary input guardrail | Enabled | Customize `BoundaryGuardrailPolicy`; pass `None`/`null` to disable the built-in layer. |
| Input character limit | `1,000,000` characters across user content | Change `max_input_characters` / equivalent TS policy field. |
| Output character limit | `1,000,000` characters | Change `max_output_characters`. |
| Content-part limit | `1024` | Change `max_content_parts`. |
| Control characters | Sanitized at model input/output boundaries | Set `sanitize_control_characters=False` if intentionally required. |
| Assistant output role | Required | Change `require_assistant_output` in the boundary policy. |
| Model-backed tool-input guardrail | Off unless enabled | Enable it on `ToolRegistry`. The default model/backend is `agent-action-guard`. |
| Prompt-injection defense object | Created by default by `AgentLoop` | Inject a custom `PromptInjectionDefense`. |
| Sandbox network mode | `none` | Set `agent_rt_settings.sandbox.network_mode` to `allowlist` or `unrestricted` where the backend can enforce it. |
| Sandbox backend selection | `native` when the config does not select one | Use `agent_rt_settings.sandbox.backend`; supported backends include Docker and optional managed integrations. |
| Memory, persistence, schedulers, queues | Explicitly configured | Inject the relevant store/manager; Agent RT does not silently persist application data. |
| Permissions and approvals | Explicitly configured where your application needs them | Use `PermissionEngine`, capability grants, `ApprovalManager`, and policy rules. |
| OpenAI-compatible base URL | `https://api.openai.com/v1` | Configure the provider/base URL or environment. |
| Anthropic base URL | `https://api.anthropic.com` | Configure the provider/base URL or environment. |

## Tutorial: build an agent one capability at a time

The fastest way to learn Agent RT is to start with a minimal agent and add one runtime capability at a time. The snippets below use the same public API shapes as the repository examples, but they are embedded here so you do not need to leave the documentation.

### Step 1 — run a basic agent in Python

```python
import asyncio

from agent_rt import (
    AgentConfig,
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelSettings,
    load_config,
    load_model,
)


async def main():
    config = load_config()  # agent-rt.yaml, ~/.agent-rt/config.yaml, or an explicit path
    model = config.default_model
    provider = load_model(config)

    agent = AgentConfig(
        name="basic",
        instructions="Answer clearly and concisely. Do not use tools.",
        model=ModelSettings(model=model),
    )

    message = ModelMessage(
        role="user",
        content=(
            ContentPart(
                type="text",
                text="Explain what an agent runtime does in two sentences.",
            ),
        ),
    )

    result = await AgentLoop(provider).run(agent, [message])

    text = "".join(
        part.text or ""
        for part in result.final_response.message.content
        if part.type == "text"
    )
    print(text)
    print("termination:", result.termination_reason)
    print("turns:", result.turns)


asyncio.run(main())
```

### Step 1 — the same agent in TypeScript

```javascript
const { AgentLoop, loadConfig, loadModel } = require("agent-rt");

async function main() {
  const config = await loadConfig(); // agent-rt.yaml, ~/.agent-rt/config.yaml, or an explicit path
  const model = config.settings.default_model ?? config.models[0].modelName;

  const provider = await loadModel(config);

  const agent = {
    name: "basic",
    instructions: "Answer clearly and concisely. Do not use tools.",
    model: { model },
  };

  const messages = [{
    role: "user",
    content: [{
      type: "text",
      text: "Explain what an agent runtime does in two sentences.",
    }],
  }];

  const result = await new AgentLoop(provider).run(agent, messages);

  const text = result.finalResponse.message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text || "")
    .join("");

  console.log(text);
  console.log("termination:", result.terminationReason);
  console.log("turns:", result.turns);
}

main();
```

### Step 2 — add a tool

Tools are schemas plus handlers. The model only proposes the call; Agent RT validates and dispatches it.

```python
from agent_rt import AgentLoop, ToolDefinition, ToolRegistry, load_config, load_model

registry = ToolRegistry()

async def weather(arguments, cancellation_token):
    city = arguments["city"]
    return {"city": city, "temperature_c": 28, "condition": "clear"}

registry.register(
    ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}},
            "additionalProperties": False,
        },
        side_effect="read",
    ),
    handler=weather,
)

loop = AgentLoop(load_model(load_config()), tool_registry=registry)
```



```javascript
const { AgentLoop, ToolRegistry, loadConfig, loadModel } = require("agent-rt");

const registry = new ToolRegistry();

registry.register(
  {
    name: "get_weather",
    description: "Get the current weather for a city.",
    inputSchema: {
      type: "object",
      required: ["city"],
      properties: { city: { type: "string" } },
      additionalProperties: false,
    },
    sideEffect: "read",
  },
  {
    handler: async (args) => ({
      city: args.city,
      temperature_c: 28,
      condition: "clear",
    }),
  },
);

const provider = await loadModel(config);
const loop = new AgentLoop(provider, undefined, registry);
```

### Step 3 — stream output

```python
async def on_event(event):
    if event.type == "text_delta" and event.text:
        print(event.text, end="", flush=True)

result = await AgentLoop(provider).run_streaming(
    agent,
    [user_message("Give me three practical uses for an agent runtime.")],
    on_event,
)
```



```javascript
const result = await new AgentLoop(provider).runStreaming(
  agent,
  [message("user", "Give me three practical uses for an agent runtime.")],
  (event) => {
    if (event.type === "text_delta" && event.text) {
      process.stdout.write(event.text);
    }
  },
);
```

### Step 4 — require structured JSON

```python
from agent_rt import AgentConfig, AgentOutputRequirements, ModelSettings

agent = AgentConfig(
    name="extractor",
    instructions="Return only valid JSON.",
    model=ModelSettings(model=configured_model()),
    output=AgentOutputRequirements(
        format="json",
        schema={
            "type": "object",
            "required": ["language", "difficulty", "topics"],
            "properties": {
                "language": {"type": "string"},
                "difficulty": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        max_repair_attempts=1,
    ),
)
```



```javascript
const agent = {
  name: "extractor",
  instructions: "Return only valid JSON.",
  model: { model: configuredModel() },
  output: {
    format: "json",
    schema: {
      type: "object",
      required: ["language", "difficulty", "topics"],
      properties: {
        language: { type: "string" },
        difficulty: { type: "string" },
        topics: { type: "array", items: { type: "string" } },
      },
      additionalProperties: false,
    },
    maxRepairAttempts: 1,
  },
};
```

### Step 5 — bound the run

```python
from agent_rt import AgentRunLimits

limits = AgentRunLimits(
    max_turns=4,
    max_tool_calls=8,
    timeout_seconds=30,
    max_total_tokens=4_000,
)

result = await AgentLoop(provider).run(
    agent,
    messages,
    limits=limits,
)
```



```javascript
const result = await new AgentLoop(provider).run(
  agent,
  messages,
  {
    maxTurns: 4,
    maxToolCalls: 8,
    timeoutMs: 30_000,
    maxTotalTokens: 4_000,
  },
);
```

### Step 6 — use the registry directly, without a model

```python
from agent_rt import ToolCall, ToolDefinition, ToolRegistry

registry = ToolRegistry()

async def search(arguments, cancellation_token):
    return {"query": arguments["query"], "matches": ["alpha", "beta"]}

registry.register(
    ToolDefinition(
        name="search",
        description="Search a demo catalog.",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
        side_effect="read",
    ),
    namespace="catalog",
    handler=search,
)

result = await registry.execute(
    ToolCall(
        id="demo-1",
        name="catalog.search",
        arguments={"query": "agent"},
    )
)
```

**What this tutorial establishes**The basic model call, tools, streaming, structured output, run limits, and registry behavior are all separate pieces. You can adopt only the pieces you need, then add policy, persistence, orchestration, sandboxing, and observability incrementally.

## 1. Agent definition and provider layer

An agent combines instructions, model/provider configuration, tools, output expectations, and runtime policy. The provider layer is intentionally separate from the loop, so the same agent runtime can use OpenAI-compatible, Anthropic-compatible, custom, routed, aliased, or fallback providers without moving policy into provider-specific application code.

**Use it when:** you need to switch model vendors, expose the same agent through several interfaces, apply routing, or keep tool/state semantics stable while models change.

```python
# Python: conceptual minimal setup
agent = AgentConfig(name="assistant", instructions="Help the user.")
loop = AgentLoop(provider)
result = loop.run(agent, messages)

# Add AgentRunLimits(...) when you need custom execution bounds.
```

## 2. Core agent loop

The loop performs model request → tool-call dispatch → tool results → model continuation until the model completes or a runtime termination condition is reached. Termination is explicit: turn limits, tool-call limits, cancellation, timeout/deadline, token budget, loop detection, model failure classification, or normal completion.

The default limits are intentionally bounded for turns and tool calls: 16 turns and 64 tool calls. Concurrency is off by default so tool execution is deterministic unless the application explicitly opts into concurrent batches.

## 3. Structured output and streaming

Agent RT supports provider-neutral streaming events and structured output validation/repair. Structured output is useful when downstream code needs a schema rather than free-form text. Streaming is useful when interfaces need incremental model text, tool activity, or status events without changing the runtime’s execution semantics.

**How to use:** define the output schema/validator on the agent or request path, and use the streaming run API when your UI or service consumes events incrementally.

## 4. Tools and tool registry

A tool is a runtime capability with a name, description, argument schema, optional output schema, side-effect classification, timeout/resource information, and a handler. `ToolRegistry` provides registration, lookup, namespaces, enable/disable, replacement, deferred loading, contextual/async handlers, validation, guardrails, lifecycle hooks, and dependency-injected execution context.

**Execution order:** visibility → model call → tool lookup/enabled check → permissions/capability checks → pre-call transform → action blockers → tool-input guardrails → argument validation → rate limits/dry-run → approvals/policy → handler execution → tool-output guardrails → result marshaling → model continuation.

 **Why visibility and dispatch are separate**Hiding a tool from the model reduces capability exposure, but dispatch still re-checks execution policy. A stale or malformed call cannot rely on having seen an older schema.

## 5. Tool selection, visibility, and discovery

`ToolSelectionPolicy` can allow, deny, require, or prefer tools. Runtime visibility filters can narrow the visible set per turn. `Toolbase` combines enabled local tools, materialized deferred tools, and discovered MCP tools into a safety-screened internal catalog. The main model receives the relevance-selected subset plus `tool_search`; search results can promote additional safe tools on later turns without bypassing capability grants, prompt-injection filtering, tool-selection policy, context isolation, or execution-time registry checks. `make_decision_toolbase` / `makeDecisionToolbase` use Decision models for safety screening and relevance selection, while `CapabilityCatalog` provides the bounded search index.

**Use it when:** your application has many tools, tenant-specific capabilities, task phases, specialist agents, or expensive schemas that should not appear on every turn.

## 6. Guardrails: four different boundaries

“Guardrail” is a family of hooks, not one feature. Agent RT distinguishes four execution boundaries:

| Guardrail | Runs when | Built-in default | Typical use |
| --- | --- | --- | --- |
| User/model-boundary input | Before user messages enter the model request/history path | Yes: size/content-part checks and control-character sanitization | Input size limits, sanitization, application-specific input policy |
| Model output | Before assistant output becomes final/history state | Yes: role, size, part-count, and control-character checks | Output policy, schema/content checks, redaction |
| Tool input | After call transforms, before argument validation and side effects | No model-backed classifier unless enabled. If enabled, default = `agent-action-guard` | Harmful-action detection, exfiltration prevention, argument transformation |
| Tool output | Before tool results are exposed back to the model | Custom/optional | Prompt-injection filtering, secret redaction, restricted-data egress checks |

### Default conversation-boundary guardrails

`AgentLoop` creates a default `BoundaryGuardrailPolicy`. It permits up to 1,000,000 input characters, 1,000,000 output characters, and 1024 content parts; sanitizes disallowed control characters; and requires final model output to use the assistant role. Application guardrails are added on top of this built-in layer.

```python
# Python: customize the built-in input/output boundary policy
policy = BoundaryGuardrailPolicy(
    max_input_characters=200_000,
    max_output_characters=100_000,
    max_content_parts=256,
    sanitize_control_characters=True,
    require_assistant_output=True,
)
loop = AgentLoop(provider, boundary_guardrail_policy=policy)

# Disable only the built-in boundary layer if your application deliberately
# replaces it with its own input/output guardrails:
loop = AgentLoop(provider, boundary_guardrail_policy=None)
```

### Composable action blockers

`ToolRegistry` can run multiple deny-only action blockers before tool-input guardrails, validation, approvals, and handler execution. The first rejection raises `ActionBlockedError`. Built-ins include `RuleBasedActionBlocker`, Agent Action Guard adapters, and Decision-model adapters for Jev-compatible providers; custom functions/callables and objects exposing `check` are also supported.

The default rule set blocks `sudo`, `rm -rf`, `git add`, `git commit`, `git push`, `git reset --hard`, `git merge`, and `git rebase`. Rules are configurable and match command prefixes at shell-command segment boundaries rather than arbitrary substrings.

```python
from agent_rt import RuleBasedActionBlocker, ToolRegistry

registry = ToolRegistry(
    action_blockers=(
        RuleBasedActionBlocker(["sudo", "rm -rf", "git push"]),
        my_custom_blocker,
    ),
)
```

### Agent Action Guard: the default model-backed tool-input model

Agent Action Guard protects *tool calls*, not raw user messages. The package is installed with Agent RT by default. When model-backed tool-input screening is enabled, `DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL` is `agent-action-guard`. Screening happens before argument validation and before the tool handler can cause side effects.

```python
# Python
registry = ToolRegistry(enable_model_tool_input_guardrail=True)

# Tests/private deployments can inject a classifier instead of downloading
# the default model:
registry = ToolRegistry(
    enable_model_tool_input_guardrail=True,
    tool_input_guardrail_classifier=my_classifier,
)
```

Custom tool-input and tool-output guardrails can return allow, transform, or block results. Blocking raises a guardrail violation; transforming changes the value passed to the next runtime stage while preserving tool-call identity.

## 7. Permissions, authorization, credentials, and tenancy

`PermissionEngine` and `PermissionRule` implement deny-overrides permission checks for agents, tools, filesystem paths, network targets, operations, data classes, and side-effect classes. `CapabilityGrant` narrows the capabilities visible/usable by a run. Authentication and authorization objects carry identity, scopes, roles, tenant membership, delegated access, credential metadata, and expiry without embedding secret values into model-visible text.

**Rule:** authorization is runtime state, not prompt text. A model saying “the user approved this” does not grant a capability.

```python
from agent_rt import PermissionEngine, PermissionRule, ToolRegistry

permissions = PermissionEngine((
    PermissionRule(
        effect="allow",
        operations=("execute",),
        tools=("safe.*",),
    ),
    PermissionRule(
        effect="deny",
        operations=("execute",),
        side_effects=("destructive",),
    ),
))

registry = ToolRegistry(permission_engine=permissions)
```

Deny rules override broad allows. The same engine can enforce path-level rules:

```python
permissions = PermissionEngine((
    PermissionRule(
        effect="allow",
        operations=("read",),
        paths=("workspace/**",),
    ),
    PermissionRule(
        effect="deny",
        operations=("read", "write"),
        paths=("workspace/secrets/**",),
    ),
    PermissionRule(
        effect="allow",
        operations=("write",),
        paths=("workspace/output/**",),
    ),
))
```

## 8. Human approvals and consequential actions

`ApprovalManager` can pause consequential or destructive tool calls before side effects. Approval stores can represent allow-once, deny-once, session, durable grants, and revocation. The agent loop can surface a waiting-for-approval state rather than pretending the run completed.

Dry-run request context can preview side effects without executing a handler or consuming an approval. `SideEffectTransaction` separates prepare from commit and supports compensation after partial failure.

```python
from agent_rt import ApprovalManager, ToolDefinition, ToolRegistry

approvals = ApprovalManager()
registry = ToolRegistry(approval_manager=approvals)

registry.register(
    ToolDefinition(
        name="publish",
        description="Publish a release.",
        input_schema={"type": "object"},
        side_effect="consequential",
    ),
    handler=publish_release,
)

# Executing a consequential tool can raise ApprovalRequiredError.
# Present that request to a reviewer, then resolve it:
approvals.resolve(
    approval_request,
    "allow",
    actor_id="reviewer-2",
)
```

## 9. Prompt injection and data exfiltration controls

`ContextItem` is untrusted unless explicitly marked trusted. `PromptInjectionDefense` preserves that trust distinction and can remove write/consequential capabilities when untrusted external context is present. `DataExfiltrationPolicy` classifies data and can be attached to tool-input/tool-output guardrails to block restricted information from disallowed model, tool, network, or logging channels.

This is separate from Agent Action Guard: one controls trust and egress boundaries; the other can classify proposed tool actions.

## 10. Context engineering

`ContextAssembler`, `ContextItem`, and `ContextSelectionPolicy` build provider requests from instructions, selected history, state, retrieved knowledge, files, observations, tools, and metadata. Context selection can narrow subwork without deleting durable history. Compaction can summarize older turns, oversized context can be offloaded to artifact storage, and prompt-cache hints separate stable prefixes from volatile turn content.

## 11. Workflow state and short-term memory

`WorkflowState` is a versioned JSON-safe durable state envelope. It is separate from conversational messages so applications can store counters, decisions, plan state, approval state, intermediate outputs, and other structured execution data without reconstructing them from chat.

`SessionRef` identifies stable session/thread scope. `ShortTermSessionMemory` stores bounded recent messages and workflow state per thread.

```python
from agent_rt import SessionRef, ShortTermSessionMemory, WorkflowState

session = SessionRef("session-42", "main")
memory = ShortTermSessionMemory(max_messages=20)

state = WorkflowState(
    state_type="order_workflow",
    version=1,
    data={
        "step": "review",
        "approved": False,
        "attempts": 1,
    },
)

memory.set_workflow_state(session, state)
memory.append_messages(session, messages)

snapshot = memory.snapshot(session)
print(snapshot.workflow_state.data)
print(len(snapshot.messages))
```

## 12. Long-term, semantic, episodic, and procedural memory

Memory stores use provider-neutral read/write/update/delete/search contracts. `SemanticMemory` adds embedding-based ranking; `EpisodicMemory` records task actions and outcomes; `ProceduralMemory` stores reusable instructions, scripts, templates, and workflows. Write/retrieval policies can filter by relevance, sensitivity, confidence, duplication, scope, recency, and kind. Scoped/lifecycle wrappers enforce isolation, expiry, purge, compaction provenance, and schema migrations.

`load_memory()` / `loadMemory()` connect a dedicated memory package (mem0 by default; Zep, Letta, Supermemory, or memU in Python) as an `AgentConfig.memory` source and store finished conversations with `remember()`. See

[Agent memory](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html#agent-memory)

.

**Default:** persistent memory is not silently enabled. Choose and inject the memory layer appropriate for your application.

## 13. Checkpoints, events, idempotency, and durable execution

`AgentCheckpoint` captures messages, workflow state, and already-consumed run counters so a resumed run does not reset execution limits. `EventStore` provides append-only task events. Idempotency stores let repeated tool-call identifiers reuse prior results within a task/tool scope instead of duplicating external side effects.

`TaskLifecycle` validates submitted, queued, running, waiting, completed, failed, and canceled transitions. Background task, queue, scheduler, and trigger abstractions support longer-running execution patterns when the application needs them.

## 14. Planning, verification, reflection, and multi-agent orchestration

Agent RT includes explicit plans, dependency-aware plan steps, acceptance criteria, progress tracking, versioned replanning, verification, reflection/repair, critic panels, specialist agents as tools, handoffs, isolated subagents, supervisor/worker patterns, teams, parallel fan-out, map-reduce, and speculative alternatives.

Child work can receive a narrowed context, tool scope, capability grant, and budget so delegation does not automatically inherit every parent capability.

`TodoList` gives a model a task list through one `todo` tool (add, update, remove, clear, list; statuses `pending`, `in_progress`, `completed`). Each instance holds one session's tasks; the terminal CLI and web UI register it by default and save each session's list with its transcript.

## 15. Filesystems, workspaces, and artifacts

The filesystem layer is backend-neutral and supports list/read/write/delete/move/copy/glob/search. `WorkspaceFiles` adds safe text creation, exact replacements, and context-checked patching. Persistent workspace stores retain workspace identity across reopen/resume flows.

Artifacts are versioned first-class outputs for reports, code, datasets, documents, images, and archives. Revisions can track parentage, metadata, diffs, provenance, retention, lifecycle state, and locations.

## 16. Sandboxed shell and code execution

`SandboxBackend` and `SandboxSession` keep generated execution behind an injected isolation boundary. Backends can expose shell tools and language runtimes while carrying explicit workspace, environment, package state, resource limits, network policy, snapshots, and restore/clone behavior.

Config-driven sandbox selection defaults to `native` when `agent_rt_settings.sandbox.backend` is unset; supported deployments can select Docker or optional managed backends. Values that attempt to disable sandboxing through the backend selector are rejected. The native backend is POSIX-oriented and requires an explicitly restricted identity; use a backend capable of enforcing the isolation dimensions you require.

Network mode defaults to `none`. Allowlists and unrestricted access require explicit configuration, and backends that cannot enforce a requested network policy fail closed rather than silently weakening it.

```yaml
agent_rt_settings:
  sandbox:
    backend: native      # use a restricted OS identity
    uid: 1234
    gid: 1235
    network_mode: none   # network access is disabled by default
```



```python
from agent_rt import (
    SandboxCommand,
    SandboxResourceLimits,
    SandboxSession,
    sandbox_backend_from_env,
)

backend = sandbox_backend_from_env()
session = SandboxSession(
    "build-session",
    backend,
    limits=SandboxResourceLimits(memory_bytes=512 * 1024 * 1024),
)

result = await session.execute(
    SandboxCommand(argv=("python", "--version"))
)

print(result.exit_code)
print(result.stdout.decode())
```

## 17. Retrieval, browser, computer, multimodal, and realtime interfaces

`RetrievalRegistry` normalizes web, enterprise, file, knowledge, and database search behind one result contract and can run an injected `RetrievalReranker` after provider retrieval. Python `reranker_k` and TypeScript `rerankerK` optionally cap the reranked sequence; if omitted, the reranker itself has no top-k cap. The normal retrieval query limit is still applied to the final registry result. `DecisionRetrievalReranker` provides the built-in Decision-model implementation. Browser and computer sessions provide controlled navigation/search/form/download and screenshot/pointer/keyboard capabilities. Multimodal messages support text, image, PDF, audio, video, document, file, and JSON parts. Realtime sessions cover low-latency audio/text events, speech turn markers, completion, interruption, and cleanup.

```python
from agent_rt import RetrievalQuery, RetrievalRegistry, RetrievalResult

class KnowledgeBase:
    kind = "knowledge"

    async def search(self, query):
        return (
            RetrievalResult("1", "Agent runtime", "Runtime overview", score=0.95),
            RetrievalResult("2", "Policies", "Policy overview", score=0.82),
        )

retrieval = RetrievalRegistry()
retrieval.register("docs", KnowledgeBase())

results = await retrieval.search(
    "docs",
    RetrievalQuery("guardrails", limit=2),
)
```

## 18. Vector databases

The project supports Chroma, Milvus, Pinecone, Qdrant, and Weaviate HTTP adapters and hosted Bedrock Knowledge Bases retrieval (optional AWS SDK), with YAML configuration, lazy initialization, and LangChain/LlamaIndex-compatible imports. See the

[vector database guide](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html)

for concrete setup.

## 19. Provider routing and Decision models

Model aliases, fallback chains, routing policies, cost/latency-aware selection, and provider-neutral metadata let applications select a generation model without rewriting the loop. Decision models—including Jev-compatible models, which you can self-host with the open-source

[Ollaya](https://github.com/ollaya-dev/ollaya)

server—can be used for bounded routing/selection tasks such as capability choice. Every Decision helper also works with the OpenAI-compatible Decisions API through `decision_model: {provider: openai}` or `OpenAIDecisionProvider`.

## 20. MCP, connectors, and remote agents

`MCPClient` negotiates capabilities before consuming remote tools, resources, and prompts. Authorized MCP access can combine authentication scopes, credential grants, deny-overrides policy rules, and per-kind filters. Protocol and connector registries separate protocol mechanics from service-specific operations. Remote-agent support covers discovery, capability negotiation, task state, artifacts, streaming events, cancellation, and callbacks.

## 21. Reliability: retries, recovery, rate limits, budgets, deadlines, loops

Failure classification separates transient dependency failures, model-correctable failures, user-correctable issues, policy failures, and terminal system failures. Retry executors provide bounded backoff; recovery routing can choose repair, alternate tools, model fallback, user input, or escalation. Circuit breakers provide closed/open/half-open behavior.

`RateLimiter`, `ExecutionBudget`, `Deadline`, and `LoopDetector` provide independent controls for quotas, model/retry/subagent/cost budgets, nested deadlines, and repeated trajectories.

```python
from agent_rt import ExecutionBudget, ExecutionBudgetLimits

budget = ExecutionBudget(
    ExecutionBudgetLimits(
        max_model_calls=12,
        max_retries=3,
        max_subagents=4,
        max_cost=2.50,
    )
)

loop = AgentLoop(
    provider,
    execution_budget=budget,
)
```

## 22. Observability, auditing, privacy, tokens, and cost

Tracing records task/agent/turn/model/tool/subagent/guardrail/queue/remote spans. Structured logging supports privacy protection before persistence: secret-like keys are redacted, while `PIIMorpher` deterministically substitutes detected PII with synthetic values so repeated values remain correlated without retaining plaintext. The lookup map stores only a short SHA-256+MD5 fingerprint to fake-data mapping; set `AGENT_RT_DISABLE_PII_MORPHER=true` to disable morphing. Runtime metrics cover latency, throughput, errors, retries, queue time, and completion. Token and cost ledgers can attribute usage to users, tenants, tasks, agents, models, tools, storage, compute, and external services.

Audit trails record actor-attributed request/authorization/execution outcomes. Retention-aware event storage supports TTL, purge, archival, ephemeral runs, and logging suppression where policy requires it.

```python
from agent_rt import TraceRecorder

traces = TraceRecorder()

task_span = traces.start(
    span_id="span-task",
    trace_id="trace-42",
    name="task",
    kind="task",
    task_id="task-42",
)

agent_span = traces.start(
    span_id="span-agent",
    trace_id="trace-42",
    name="agent",
    kind="agent",
    parent_span_id=task_span.span_id,
    task_id="task-42",
)
```

## 23. Evaluation, replay, and testing

Evaluation APIs support deterministic harness assessment and regression workflows. Debug/replay primitives reconstruct captured state, events, and checkpoints while substituting recorded model/tool/external outputs. Repository security scans and adversarial runtime tests are designed to run offline against deterministic malicious-provider doubles rather than real external models.

## 24. API server, CLI, IDE, and chat surfaces

The runtime can be exposed through a CLI, a stable application API, or an optional OpenAI/Anthropic-compatible server surface. Interface layers are adapters over the same runtime rather than separate execution engines, so tools, guardrails, routing, run limits, state, and policy remain consistent.

IDE integration can pass current-file, selection, diagnostics, and diff context into workspace operations. The terminal CLI and `agent-rt web` share saved sessions, so a session started in one resumes in the other. The CLI reads Codex and Claude Code project setup as is: `AGENTS.md`/`CLAUDE.md` instructions, skills in `.agents/`, `.claude/`, and `.codex/` (trusted by file hash after a full scan, and scanned again when they change), and MCP servers from `.mcp.json`, `~/.claude.json`, and `~/.codex/config.toml` (see the

[runtime guide](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html#project-instructions-skills-mcp)

). `agent-rt serve telegram,discord` (Python, beta) answers allowed users as a Telegram or Discord bot without workspace tools. Chat bridges preserve channel identity and thread state. Approval presentation provides standardized consequence/diff/allow-or-deny UI data.

## 25. Compatibility and migration

Compatibility modules support common LangChain and LlamaIndex integration patterns, while OpenAI- and Anthropic-compatible HTTP surfaces ease application migration. Use compatibility layers to keep surrounding application code familiar while moving policy, tools, limits, state, and execution control into Agent RT. Migration field testing is ongoing; 65+ third-party repositories currently pass their test suites both before and after migration.

## 26. Production deployment checklist

| Before production | Recommended action |
| --- | --- |
| Tools | Expose the minimum schemas; classify side effects; set per-tool timeouts/resource needs. |
| Guardrails | Keep default boundary guardrails unless you deliberately replace them; add application-specific input/output and tool guardrails. |
| Tool action screening | Enable Agent Action Guard or another classifier when model-backed tool-action screening is part of your threat model. |
| Permissions | Use deny-overrides rules and scoped capability grants outside the prompt. |
| Approvals | Gate consequential/destructive operations before side effects. |
| Untrusted data | Mark external context untrusted; restrict consequential capabilities and data egress. |
| Execution | Use a sandbox backend that can actually enforce your filesystem, process, resource, and network requirements. |
| Network | Keep no-network or explicit allowlists where possible. |
| Durability | Use checkpoints, events, idempotency, queues/schedulers only where your workload needs them. |
| Limits | Set run, token, time, retry, rate, and monetary budgets appropriate to the workload. |
| Observability | Enable traces/metrics/audit while redacting sensitive data before persistence. |
| Testing | Use deterministic provider/tool doubles for failure, security, replay, and regression tests. |

**Mental model**The model proposes. Agent RT controls context, capabilities, validation, policy, execution, persistence, budgets, and observability. Optional integrations extend those boundaries; they do not replace them.

[RelatedArchitecture](https://agent-rt.pro-genai.com/docs/reference/architecture.html)



[NextProduction patterns →](https://agent-rt.pro-genai.com/docs/guides/production.html)
