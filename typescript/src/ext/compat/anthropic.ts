import {
  AnthropicModelProvider,
  type BatchJob,
  type BatchProvider,
  type ModelCatalogEntry,
  type ModelCatalogProvider,
  type ModelMessage,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type ReasoningConfig,
  type RetrievalProvider,
  type SandboxSession,
  type StructuredOutputRequirement,
  type TokenCountingModelProvider,
} from "../../index.js";
import {
  completeWithMigrationTools,
  expandMigrationTools,
  type CodeExecutionRecord,
  type MCPExecutionRecord,
  type MCPMigrationClient,
  type WebSearchExecutionRecord,
  validateAnthropicMCPServers,
} from "./migration_tools.js";

type JSONObject = Record<string, unknown>;

type AnthropicClientOptions = {
  apiKey?: string;
  baseURL?: string;
  provider?: ModelProvider;
  defaultModel?: string;
  sandboxSession?: SandboxSession;
  maxCodeToolRounds?: number;
  mcpClients?: Record<string, MCPMigrationClient>;
  webSearchProvider?: RetrievalProvider;
};

function isObject(value: unknown): value is JSONObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function structuredOutput(value: unknown): StructuredOutputRequirement | undefined {
  if (!isObject(value)) return undefined;
  let payload: JSONObject = value;
  if (isObject(payload.format)) payload = payload.format;
  if (isObject(payload.json_schema)) payload = payload.json_schema;
  const schema = isObject(payload.schema) ? payload.schema : undefined;
  if (!schema) return undefined;
  return {
    schema,
    ...(typeof payload.name === "string" && payload.name ? { name: payload.name } : {}),
    strict: typeof payload.strict === "boolean" ? payload.strict : true,
  };
}

function anthropicReasoning(
  thinking: unknown,
  outputConfig: unknown,
): ReasoningConfig | undefined {
  let mode: ReasoningConfig["thinking"];
  let budgetTokens: number | undefined;
  let effort: string | undefined;
  if (thinking !== undefined) {
    if (!isObject(thinking)) throw new TypeError("thinking must be an object");
    if (thinking.type !== undefined) {
      if (thinking.type !== "adaptive" && thinking.type !== "enabled" && thinking.type !== "disabled") {
        throw new TypeError("thinking.type must be adaptive, enabled, or disabled");
      }
      mode = thinking.type;
    }
    if (thinking.budget_tokens !== undefined) {
      if (!Number.isInteger(thinking.budget_tokens) || Number(thinking.budget_tokens) < 1) {
        throw new TypeError("thinking.budget_tokens must be a positive integer");
      }
      if (mode !== "enabled") {
        throw new TypeError("thinking.budget_tokens requires thinking.type='enabled'");
      }
      budgetTokens = Number(thinking.budget_tokens);
    }
  }
  if (isObject(outputConfig) && outputConfig.effort !== undefined) {
    if (typeof outputConfig.effort !== "string" || !outputConfig.effort.trim()) {
      throw new TypeError("output_config.effort must be a non-empty string");
    }
    effort = outputConfig.effort;
  }
  return mode !== undefined || budgetTokens !== undefined || effort !== undefined
    ? { thinking: mode, budgetTokens, effort }
    : undefined;
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (!isObject(part)) return "";
    return typeof part.text === "string" ? part.text : "";
  }).join("");
}

function messagesFromAnthropic(messages: unknown, system?: unknown): ModelMessage[] {
  const result: ModelMessage[] = [];
  if (typeof system === "string" && system) {
    result.push({ role: "system", content: [{ type: "text", text: system }] });
  }
  if (!Array.isArray(messages)) throw new TypeError("messages must be an array");
  for (const item of messages) {
    if (!isObject(item) || typeof item.role !== "string") {
      throw new TypeError("each message must include a role");
    }
    if (item.role !== "user" && item.role !== "assistant") {
      throw new TypeError(`unsupported Anthropic message role ${JSON.stringify(item.role)}`);
    }
    result.push({
      role: item.role,
      content: [{ type: "text", text: textFromContent(item.content) }],
    });
  }
  return result;
}



function anthropicBatchShape(job: BatchJob): unknown {
  if (job.raw && typeof job.raw === "object" && job.raw !== null) return job.raw;
  return {
    id: job.id,
    type: "message_batch",
    processing_status: job.status,
  };
}

function asyncBatchResults(value: unknown): AsyncIterable<unknown> {
  if (value && typeof value === "object" && Symbol.asyncIterator in value) {
    return value as AsyncIterable<unknown>;
  }
  if (value && typeof value === "object" && Symbol.iterator in value) {
    const iterable = value as Iterable<unknown>;
    return {
      async *[Symbol.asyncIterator]() {
        for (const item of iterable) yield item;
      },
    };
  }
  if (Array.isArray(value)) {
    return {
      async *[Symbol.asyncIterator]() {
        for (const item of value) yield item;
      },
    };
  }
  throw new TypeError("Anthropic batch results must be iterable");
}

function anthropicModelShape(entry: ModelCatalogEntry) {
  if (entry.raw && typeof entry.raw === "object" && entry.raw !== null) return entry.raw;
  return {
    id: entry.id,
    type: "model",
    display_name: entry.displayName ?? entry.id,
    created_at: typeof entry.created === "string" ? entry.created : undefined,
  };
}

function responseText(response: ModelResponse): string {
  return response.message.content.map((part) => part.text ?? "").join("");
}

function anthropicShape(
  response: ModelResponse,
  executions: CodeExecutionRecord[] = [],
  mcpCalls: MCPExecutionRecord[] = [],
  webSearchCalls: WebSearchExecutionRecord[] = [],
) {
  const text = responseText(response);
  const mcpContent = mcpCalls.flatMap((call) => [
    {
      type: "mcp_tool_use" as const,
      id: call.id,
      server_name: call.serverLabel,
      name: call.toolName,
      input: call.arguments,
    },
    {
      type: "mcp_tool_result" as const,
      tool_use_id: call.id,
      content: JSON.stringify(call.output),
      is_error: false,
    },
  ]);
  const webSearchContent = webSearchCalls.flatMap((call) => [
    {
      type: "server_tool_use" as const,
      id: call.id,
      name: "web_search" as const,
      input: { query: call.query },
    },
    {
      type: "web_search_tool_result" as const,
      tool_use_id: call.id,
      content: call.results
        .filter((result) => result.url)
        .map((result) => ({
          type: "web_search_result" as const,
          title: result.title,
          url: result.url as string,
          encrypted_content: null,
          page_age: null,
        })),
    },
  ]);
  const executionContent = executions.flatMap((execution) => [
    {
      type: "server_tool_use" as const,
      id: execution.id,
      name: execution.kind === "code" ? "code_execution" : "bash",
      input: execution.arguments,
    },
    {
      type: "code_execution_tool_result" as const,
      tool_use_id: execution.id,
      content: {
        type: "code_execution_result" as const,
        stdout: execution.output.stdout,
        stderr: execution.output.stderr,
        return_code: execution.output.exit_code,
      },
    },
  ]);
  return {
    id: undefined,
    type: "message",
    role: "assistant",
    model: response.model,
    content: [
      ...mcpContent,
      ...webSearchContent,
      ...executionContent,
      ...(text ? [{ type: "text" as const, text }] : []),
    ],
    stop_reason: response.finishReason,
    usage: response.usage
      ? {
          input_tokens: response.usage.inputTokens,
          output_tokens: response.usage.outputTokens,
        }
      : undefined,
  };
}

export class Anthropic {
  readonly provider: ModelProvider;
  readonly sandboxSession?: SandboxSession;
  readonly maxCodeToolRounds: number;
  readonly mcpClients: Record<string, MCPMigrationClient>;
  readonly webSearchProvider?: RetrievalProvider;
  readonly messages: {
    create: (params: JSONObject) => Promise<ReturnType<typeof anthropicShape>>;
    countTokens: (params: JSONObject) => Promise<{ input_tokens: number }>;
    batches: {
      create: (params: JSONObject) => Promise<unknown>;
      retrieve: (id: string) => Promise<unknown>;
      list: (params?: JSONObject) => Promise<{
        data: unknown[];
        has_more: false;
        first_id?: string;
        last_id?: string;
      }>;
      cancel: (id: string) => Promise<unknown>;
      results: (id: string) => Promise<AsyncIterable<unknown>>;
    };
  };
  readonly beta: {
    messages: {
      create: (params: JSONObject) => Promise<ReturnType<typeof anthropicShape>>;
    };
  };
  readonly models: {
    list: () => Promise<{
      data: unknown[];
      has_more: false;
      first_id?: string;
      last_id?: string;
    }>;
    retrieve: (model: string) => Promise<unknown>;
  };

  constructor(options: AnthropicClientOptions = {}) {
    this.provider = options.provider ?? new AnthropicModelProvider({
      apiKey: options.apiKey,
      baseUrl: options.baseURL,
      defaultModel: options.defaultModel,
    });
    this.sandboxSession = options.sandboxSession;
    this.maxCodeToolRounds = options.maxCodeToolRounds ?? 8;
    this.mcpClients = { ...(options.mcpClients ?? {}) };
    this.webSearchProvider = options.webSearchProvider;
    const createMessage = async (params: JSONObject) => {
      validateAnthropicMCPServers(params.mcp_servers, params.tools);
      const format = structuredOutput(params.output_config ?? params.output_format);
      const reasoning = anthropicReasoning(params.thinking, params.output_config);
      const expanded = await expandMigrationTools(
        params.tools,
        this.sandboxSession,
        this.mcpClients,
        undefined,
        this.webSearchProvider,
      );
      const request: ModelRequest = {
        model: typeof params.model === "string" ? params.model : undefined,
        messages: messagesFromAnthropic(params.messages, params.system),
        ...(expanded.tools ? { tools: expanded.tools } : {}),
        ...(typeof params.temperature === "number" ? { temperature: params.temperature } : {}),
        ...(typeof params.max_tokens === "number" ? { maxOutputTokens: params.max_tokens } : {}),
        ...(format ? { structuredOutput: format } : {}),
        ...(reasoning ? { reasoning } : {}),
      };
      const completed = await completeWithMigrationTools(this.provider, request, {
        sandboxSession: this.sandboxSession,
        mcpBindings: expanded.mcpBindings,
        webSearchBindings: expanded.webSearchBindings,
        maxRounds: this.maxCodeToolRounds,
      });
      return anthropicShape(
        completed.response,
        completed.executions,
        completed.mcpCalls,
        completed.webSearchCalls,
      );
    };
    const messageBatches = {
      create: async (params: JSONObject) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.createBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch creation");
        }
        if (!Array.isArray(params.requests)) {
          throw new TypeError("Anthropic messages.batches.create requires requests array");
        }
        return anthropicBatchShape(await provider.createBatch({ ...params }));
      },
      retrieve: async (id: string) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.retrieveBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch retrieval");
        }
        return anthropicBatchShape(await provider.retrieveBatch(id));
      },
      list: async (params: JSONObject = {}) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.listBatches !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch listing");
        }
        const data = (await provider.listBatches({ ...params })).map(anthropicBatchShape);
        const ids = data.map((item) =>
          item && typeof item === "object" && typeof (item as Record<string, unknown>).id === "string"
            ? (item as Record<string, unknown>).id as string
            : undefined,
        ).filter((id): id is string => id !== undefined);
        return {
          data,
          has_more: false as const,
          first_id: ids[0],
          last_id: ids.length ? ids[ids.length - 1] : undefined,
        };
      },
      cancel: async (id: string) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.cancelBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch cancellation");
        }
        return anthropicBatchShape(await provider.cancelBatch(id));
      },
      results: async (id: string) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.batchResults !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch results");
        }
        return asyncBatchResults(await provider.batchResults(id));
      },
    };
    this.messages = {
      create: createMessage,
      countTokens: async (params) => {
        const provider = this.provider as ModelProvider & Partial<TokenCountingModelProvider>;
        if (typeof provider.countTokens !== "function") {
          throw new TypeError("configured Agent RT provider does not support token counting");
        }
        const request: ModelRequest = {
          model: typeof params.model === "string" ? params.model : undefined,
          messages: messagesFromAnthropic(params.messages, params.system),
        };
        return { input_tokens: await provider.countTokens(request) };
      },
      batches: messageBatches,
    };
    this.beta = { messages: { create: createMessage } };
    this.models = {
      list: async () => {
        const provider = this.provider as ModelProvider & Partial<ModelCatalogProvider>;
        if (typeof provider.listModels !== "function") {
          throw new TypeError("configured Agent RT provider does not support model listing");
        }
        const data = (await provider.listModels()).map(anthropicModelShape);
        return {
          data,
          has_more: false,
          first_id: data.length && typeof (data[0] as Record<string, unknown>).id === "string"
            ? (data[0] as Record<string, unknown>).id as string
            : undefined,
          last_id: data.length && typeof (data[data.length - 1] as Record<string, unknown>).id === "string"
            ? (data[data.length - 1] as Record<string, unknown>).id as string
            : undefined,
        };
      },
      retrieve: async (model) => {
        const provider = this.provider as ModelProvider & Partial<ModelCatalogProvider>;
        if (typeof provider.retrieveModel !== "function") {
          throw new TypeError("configured Agent RT provider does not support model retrieval");
        }
        return anthropicModelShape(await provider.retrieveModel(model));
      },
    };
  }
}

export default Anthropic;
