import {
  OpenAIModelProvider,
  type BatchJob,
  type BatchProvider,
  type EmbeddingModelProvider,
  type EmbeddingRequest,
  type ModelCatalogEntry,
  type ModelCatalogProvider,
  type ModelMessage,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type ReasoningConfig,
  type RetrievalProvider,
  type RetrievalRegistry,
  type SandboxSession,
  type StructuredOutputRequirement,
} from "../../index.js";
import {
  completeWithMigrationTools,
  expandMigrationTools,
  type CodeExecutionRecord,
  type FileSearchExecutionRecord,
  type MCPExecutionRecord,
  type MCPMigrationClient,
  type WebSearchExecutionRecord,
} from "./migration_tools.js";

type JSONObject = Record<string, unknown>;

type OpenAIClientOptions = {
  apiKey?: string;
  baseURL?: string;
  provider?: ModelProvider;
  defaultModel?: string;
  sandboxSession?: SandboxSession;
  maxCodeToolRounds?: number;
  mcpClients?: Record<string, MCPMigrationClient>;
  maxResponseStates?: number;
  retrievalRegistry?: RetrievalRegistry;
  webSearchProvider?: RetrievalProvider;
};

function isObject(value: unknown): value is JSONObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function schemaObject(value: unknown): JSONObject | undefined {
  if (isObject(value)) return value;
  return undefined;
}

function structuredOutput(value: unknown): StructuredOutputRequirement | undefined {
  if (!isObject(value)) return undefined;
  let payload: JSONObject = value;
  if (isObject(payload.format)) payload = payload.format;
  if (isObject(payload.json_schema)) payload = payload.json_schema;
  const schema = schemaObject(payload.schema);
  if (!schema) return undefined;
  return {
    schema,
    ...(typeof payload.name === "string" && payload.name ? { name: payload.name } : {}),
    strict: typeof payload.strict === "boolean" ? payload.strict : true,
  };
}

function openAIReasoning(
  reasoning: unknown,
  reasoningEffort?: unknown,
): ReasoningConfig | undefined {
  let effort: string | undefined;
  let summary: string | undefined;
  if (reasoningEffort !== undefined) {
    if (typeof reasoningEffort !== "string" || !reasoningEffort.trim()) {
      throw new TypeError("reasoning_effort must be a non-empty string");
    }
    effort = reasoningEffort;
  }
  if (reasoning !== undefined) {
    if (!isObject(reasoning)) throw new TypeError("reasoning must be an object");
    if (reasoning.effort !== undefined) {
      if (typeof reasoning.effort !== "string" || !reasoning.effort.trim()) {
        throw new TypeError("reasoning.effort must be a non-empty string");
      }
      effort = reasoning.effort;
    }
    if (reasoning.summary !== undefined) {
      if (typeof reasoning.summary !== "string" || !reasoning.summary.trim()) {
        throw new TypeError("reasoning.summary must be a non-empty string");
      }
      summary = reasoning.summary;
    }
  }
  return effort !== undefined || summary !== undefined ? { effort, summary } : undefined;
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (!isObject(part)) return "";
    if (typeof part.text === "string") return part.text;
    if (typeof part.content === "string") return part.content;
    return "";
  }).join("");
}

function messagesFromOpenAI(messages: unknown): ModelMessage[] {
  if (!Array.isArray(messages)) throw new TypeError("messages must be an array");
  return messages.map((item) => {
    if (!isObject(item) || typeof item.role !== "string") {
      throw new TypeError("each message must include a role");
    }
    const role = item.role;
    if (!["system", "user", "assistant", "tool"].includes(role)) {
      throw new TypeError(`unsupported message role ${JSON.stringify(role)}`);
    }
    return {
      role: role as ModelMessage["role"],
      content: [{ type: "text", text: textFromContent(item.content) }],
      ...(typeof item.tool_call_id === "string" ? { toolCallId: item.tool_call_id } : {}),
    };
  });
}

function responsesMessages(input: unknown, instructions?: unknown): ModelMessage[] {
  const result: ModelMessage[] = [];
  if (typeof instructions === "string" && instructions) {
    result.push({ role: "system", content: [{ type: "text", text: instructions }] });
  }
  if (typeof input === "string") {
    result.push({ role: "user", content: [{ type: "text", text: input }] });
    return result;
  }
  if (Array.isArray(input)) return result.concat(messagesFromOpenAI(input));
  throw new TypeError("Responses input must be a string or message array");
}



function openAIBatchShape(job: BatchJob): unknown {
  if (job.raw && typeof job.raw === "object" && job.raw !== null) return job.raw;
  return {
    id: job.id,
    object: "batch",
    status: job.status,
  };
}

function openAIModelShape(entry: ModelCatalogEntry) {
  if (entry.raw && typeof entry.raw === "object" && entry.raw !== null) return entry.raw;
  return {
    id: entry.id,
    object: "model",
    created: typeof entry.created === "number" ? entry.created : undefined,
    owned_by: entry.ownedBy ?? "",
  };
}

function usageShape(response: ModelResponse) {
  return response.usage
    ? {
        prompt_tokens: response.usage.inputTokens,
        completion_tokens: response.usage.outputTokens,
        total_tokens: response.usage.totalTokens,
      }
    : undefined;
}

function responseText(response: ModelResponse): string {
  return response.message.content.map((part) => part.text ?? "").join("");
}

function chatShape(response: ModelResponse) {
  return {
    id: undefined,
    object: "chat.completion",
    model: response.model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: responseText(response) },
      finish_reason: response.finishReason,
    }],
    usage: usageShape(response),
  };
}

function responsesShape(
  response: ModelResponse,
  executions: CodeExecutionRecord[] = [],
  mcpCalls: MCPExecutionRecord[] = [],
  fileSearchCalls: FileSearchExecutionRecord[] = [],
  webSearchCalls: WebSearchExecutionRecord[] = [],
  id?: string,
) {
  const text = responseText(response);
  const webSearchOutput = webSearchCalls.map((call) => ({
    id: call.id,
    type: "web_search_call" as const,
    status: "completed" as const,
    action: {
      type: "search" as const,
      query: call.query,
      sources: call.results
        .filter((result) => result.url)
        .map((result) => ({
          type: "url" as const,
          url: result.url as string,
          title: result.title,
        })),
    },
    results: call.results.map((result) => ({
      title: result.title,
      url: result.url,
      text: result.text,
      score: result.score,
    })),
  }));
  const fileSearchOutput = fileSearchCalls.map((call) => ({
    id: call.id,
    type: "file_search_call" as const,
    status: "completed" as const,
    queries: [call.query],
    results: call.results.map((result) => ({
      file_id: result.file_id,
      filename: result.filename,
      score: result.score,
      text: result.text,
      attributes: result.attributes,
    })),
  }));
  const mcpOutput = mcpCalls.map((call) => ({
    id: call.id,
    type: "mcp_call" as const,
    status: "completed" as const,
    server_label: call.serverLabel,
    name: call.toolName,
    arguments: JSON.stringify(call.arguments),
    output: JSON.stringify(call.output),
    error: null,
  }));
  const toolOutput = executions.map((execution) =>
    execution.kind === "code"
      ? {
          id: execution.id,
          type: "code_interpreter_call" as const,
          status: "completed" as const,
          code: typeof execution.arguments.code === "string" ? execution.arguments.code : "",
          outputs: [
            {
              type: "logs" as const,
              logs: [execution.output.stdout, execution.output.stderr]
                .filter(Boolean)
                .join(""),
            },
          ],
        }
      : {
          id: execution.id,
          type: "shell_call" as const,
          status: "completed" as const,
          action: execution.arguments,
          output: execution.output,
        },
  );
  return {
    id,
    object: "response",
    status: "completed",
    model: response.model,
    output_text: text,
    output: [
      ...webSearchOutput,
      ...fileSearchOutput,
      ...mcpOutput,
      ...toolOutput,
      ...(text
        ? [{
            type: "message" as const,
            role: "assistant" as const,
            status: "completed" as const,
            content: [{ type: "output_text" as const, text, annotations: [] }],
          }]
        : []),
    ],
    usage: response.usage
      ? {
          input_tokens: response.usage.inputTokens,
          output_tokens: response.usage.outputTokens,
          total_tokens: response.usage.totalTokens,
        }
      : undefined,
    error: null,
  };
}

export class OpenAI {
  readonly provider: ModelProvider;
  readonly sandboxSession?: SandboxSession;
  readonly maxCodeToolRounds: number;
  readonly mcpClients: Record<string, MCPMigrationClient>;
  readonly maxResponseStates: number;
  readonly retrievalRegistry?: RetrievalRegistry;
  readonly webSearchProvider?: RetrievalProvider;
  private readonly responseStates = new Map<string, ModelMessage[]>();
  private responseSequence = 0;
  readonly chat: {
    completions: {
      create: (params: JSONObject) => Promise<ReturnType<typeof chatShape>>;
    };
  };
  readonly responses: {
    create: (params: JSONObject) => Promise<ReturnType<typeof responsesShape>>;
  };
  readonly embeddings: {
    create: (params: JSONObject) => Promise<{
      object: "list";
      model?: string;
      data: Array<{ object: "embedding"; index: number; embedding: number[] | string }>;
      usage?: { prompt_tokens?: number; total_tokens?: number };
    }>;
  };
  readonly models: {
    list: () => Promise<{ object: "list"; data: unknown[] }>;
    retrieve: (model: string) => Promise<unknown>;
  };
  readonly batches: {
    create: (params: JSONObject) => Promise<unknown>;
    retrieve: (id: string) => Promise<unknown>;
    list: (params?: JSONObject) => Promise<{ object: "list"; data: unknown[] }>;
    cancel: (id: string) => Promise<unknown>;
  };

  constructor(options: OpenAIClientOptions = {}) {
    this.provider = options.provider ?? new OpenAIModelProvider({
      apiKey: options.apiKey,
      baseUrl: options.baseURL,
      defaultModel: options.defaultModel,
    });
    this.sandboxSession = options.sandboxSession;
    this.maxCodeToolRounds = options.maxCodeToolRounds ?? 8;
    this.mcpClients = { ...(options.mcpClients ?? {}) };
    this.maxResponseStates = options.maxResponseStates ?? 128;
    this.retrievalRegistry = options.retrievalRegistry;
    this.webSearchProvider = options.webSearchProvider;
    if (!Number.isInteger(this.maxResponseStates) || this.maxResponseStates < 1) {
      throw new RangeError("maxResponseStates must be a positive integer");
    }
    this.chat = {
      completions: {
        create: async (params) => {
          const model = typeof params.model === "string" ? params.model : undefined;
          const expanded = await expandMigrationTools(
            params.tools,
            this.sandboxSession,
            this.mcpClients,
            this.retrievalRegistry,
            this.webSearchProvider,
          );
          const reasoning = openAIReasoning(undefined, params.reasoning_effort);
          const request: ModelRequest = {
            model,
            messages: messagesFromOpenAI(params.messages),
            ...(expanded.tools ? { tools: expanded.tools } : {}),
            ...(typeof params.temperature === "number" ? { temperature: params.temperature } : {}),
            ...(typeof params.max_completion_tokens === "number"
              ? { maxOutputTokens: params.max_completion_tokens }
              : typeof params.max_tokens === "number"
                ? { maxOutputTokens: params.max_tokens }
                : {}),
            ...(structuredOutput(params.response_format)
              ? { structuredOutput: structuredOutput(params.response_format) }
              : {}),
            ...(reasoning ? { reasoning } : {}),
          };
          const completed = await completeWithMigrationTools(this.provider, request, {
            sandboxSession: this.sandboxSession,
            mcpBindings: expanded.mcpBindings,
            fileSearchBindings: expanded.fileSearchBindings,
            webSearchBindings: expanded.webSearchBindings,
            maxRounds: this.maxCodeToolRounds,
          });
          return chatShape(completed.response);
        },
      },
    };
    this.responses = {
      create: async (params) => {
        const model = typeof params.model === "string" ? params.model : undefined;
        const format = isObject(params.text) ? structuredOutput(params.text) : undefined;
        const reasoning = openAIReasoning(params.reasoning);
        let priorHistory: ModelMessage[] = [];
        if (params.previous_response_id !== undefined) {
          if (typeof params.previous_response_id !== "string" || !params.previous_response_id) {
            throw new TypeError("previous_response_id must be a non-empty string");
          }
          const stored = this.responseStates.get(params.previous_response_id);
          if (!stored) {
            throw new Error(`unknown or evicted previous_response_id ${JSON.stringify(params.previous_response_id)}`);
          }
          priorHistory = stored;
        }
        const expanded = await expandMigrationTools(
          params.tools,
          this.sandboxSession,
          this.mcpClients,
          this.retrievalRegistry,
          this.webSearchProvider,
        );
        const request: ModelRequest = {
          model,
          messages: [
            ...priorHistory,
            ...responsesMessages(params.input, params.instructions),
          ],
          ...(expanded.tools ? { tools: expanded.tools } : {}),
          ...(typeof params.temperature === "number" ? { temperature: params.temperature } : {}),
          ...(typeof params.max_output_tokens === "number"
            ? { maxOutputTokens: params.max_output_tokens }
            : {}),
          ...(format ? { structuredOutput: format } : {}),
          ...(reasoning ? { reasoning } : {}),
        };
        const completed = await completeWithMigrationTools(this.provider, request, {
          sandboxSession: this.sandboxSession,
          mcpBindings: expanded.mcpBindings,
          fileSearchBindings: expanded.fileSearchBindings,
          webSearchBindings: expanded.webSearchBindings,
          maxRounds: this.maxCodeToolRounds,
        });
        const responseId = `resp_agent_rt_${++this.responseSequence}`;
        this.responseStates.set(responseId, completed.history);
        while (this.responseStates.size > this.maxResponseStates) {
          const oldest = this.responseStates.keys().next().value as string | undefined;
          if (oldest === undefined) break;
          this.responseStates.delete(oldest);
        }
        return responsesShape(
          completed.response,
          completed.executions,
          completed.mcpCalls,
          completed.fileSearchCalls,
          completed.webSearchCalls,
          responseId,
        );
      },
    };
    this.embeddings = {
      create: async (params) => {
        const provider = this.provider as ModelProvider & Partial<EmbeddingModelProvider>;
        if (typeof provider.embed !== "function") {
          throw new TypeError("configured Agent RT provider does not support embeddings");
        }
        const request: EmbeddingRequest = {
          input: params.input,
          ...(typeof params.model === "string" ? { model: params.model } : {}),
          ...(typeof params.dimensions === "number" ? { dimensions: params.dimensions } : {}),
          ...(typeof params.encoding_format === "string"
            ? { encodingFormat: params.encoding_format }
            : {}),
        };
        const response = await provider.embed(request);
        return {
          object: "list",
          model: response.model,
          data: response.data.map((item) => ({
            object: "embedding" as const,
            index: item.index,
            embedding: item.embedding,
          })),
          usage: response.usage
            ? {
                prompt_tokens: response.usage.inputTokens,
                total_tokens: response.usage.totalTokens,
              }
            : undefined,
        };
      },
    };
    this.batches = {
      create: async (params) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.createBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch creation");
        }
        if (typeof params.input_file_id !== "string" || !params.input_file_id) {
          throw new TypeError("OpenAI batches.create requires input_file_id");
        }
        return openAIBatchShape(await provider.createBatch({ ...params }));
      },
      retrieve: async (id) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.retrieveBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch retrieval");
        }
        return openAIBatchShape(await provider.retrieveBatch(id));
      },
      list: async (params = {}) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.listBatches !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch listing");
        }
        const data = (await provider.listBatches({ ...params })).map(openAIBatchShape);
        return { object: "list", data };
      },
      cancel: async (id) => {
        const provider = this.provider as ModelProvider & Partial<BatchProvider>;
        if (typeof provider.cancelBatch !== "function") {
          throw new TypeError("configured Agent RT provider does not support batch cancellation");
        }
        return openAIBatchShape(await provider.cancelBatch(id));
      },
    };
    this.models = {
      list: async () => {
        const provider = this.provider as ModelProvider & Partial<ModelCatalogProvider>;
        if (typeof provider.listModels !== "function") {
          throw new TypeError("configured Agent RT provider does not support model listing");
        }
        const entries = await provider.listModels();
        return { object: "list", data: entries.map(openAIModelShape) };
      },
      retrieve: async (model) => {
        const provider = this.provider as ModelProvider & Partial<ModelCatalogProvider>;
        if (typeof provider.retrieveModel !== "function") {
          throw new TypeError("configured Agent RT provider does not support model retrieval");
        }
        return openAIModelShape(await provider.retrieveModel(model));
      },
    };
  }
}

export default OpenAI;
