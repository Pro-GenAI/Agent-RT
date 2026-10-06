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
  type StreamingModelProvider,
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
  sdkEnv,
} from "./migration_tools.js";
import type * as AnthropicTypes from "./anthropic_types.js";
import { createSDKErrors, withSDKErrors } from "./sdk_errors.js";
import { MessageStream, rawMessageStream } from "./anthropic_stream.js";

export { MessageStream } from "./anthropic_stream.js";

type JSONObject = Record<string, unknown>;

const errors = createSDKErrors("Anthropic");
export const AnthropicError = errors.BaseError;
export const APIError = errors.APIError;
export const APIUserAbortError = errors.APIUserAbortError;
export const APIConnectionError = errors.APIConnectionError;
export const APIConnectionTimeoutError = errors.APIConnectionTimeoutError;
export const BadRequestError = errors.BadRequestError;
export const AuthenticationError = errors.AuthenticationError;
export const PermissionDeniedError = errors.PermissionDeniedError;
export const NotFoundError = errors.NotFoundError;
export const ConflictError = errors.ConflictError;
export const UnprocessableEntityError = errors.UnprocessableEntityError;
export const RateLimitError = errors.RateLimitError;
export const InternalServerError = errors.InternalServerError;
export type AnthropicError = InstanceType<typeof AnthropicError>;
export type APIError = InstanceType<typeof APIError>;
export type APIConnectionError = InstanceType<typeof APIConnectionError>;
export type RateLimitError = InstanceType<typeof RateLimitError>;
export type * from "./anthropic_types.js";

let messageSequence = 0;

type AnthropicClientOptions = {
  apiKey?: string | undefined;
  baseURL?: string | undefined;
  provider?: ModelProvider | undefined;
  defaultModel?: string | undefined;
  sandboxSession?: SandboxSession | undefined;
  maxCodeToolRounds?: number | undefined;
  mcpClients?: Record<string, MCPMigrationClient> | undefined;
  webSearchProvider?: RetrievalProvider | undefined;
  // Vendor SDK client options accepted so migrated constructors type-check;
  // retries, timeouts, and transport come from the Agent RT provider.
  timeout?: number | undefined;
  maxRetries?: number | undefined;
  defaultHeaders?: Record<string, string | null | undefined> | undefined;
  defaultQuery?: Record<string, string | undefined> | undefined;
  fetch?: unknown | undefined;
  dangerouslyAllowBrowser?: boolean | undefined;
  logLevel?: string | undefined;
  logger?: unknown | undefined;
  authToken?: string | null | undefined;
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

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  return textFromContent(content);
}

function messagesFromAnthropic(messages: unknown, system?: unknown): ModelMessage[] {
  const result: ModelMessage[] = [];
  const systemText = textFromContent(system);
  if (systemText) {
    result.push({ role: "system", content: [{ type: "text", text: systemText }] });
  }
  if (!Array.isArray(messages)) throw new TypeError("messages must be an array");
  for (const item of messages) {
    if (!isObject(item) || typeof item.role !== "string") {
      throw new TypeError("each message must include a role");
    }
    if (item.role !== "user" && item.role !== "assistant") {
      throw new TypeError(`unsupported Anthropic message role ${JSON.stringify(item.role)}`);
    }
    if (!Array.isArray(item.content)) {
      result.push({
        role: item.role,
        content: [{ type: "text", text: textFromContent(item.content) }],
      });
      continue;
    }
    // Block content: tool_use blocks become assistant tool calls and
    // tool_result blocks become tool messages, as the provider expects.
    const text: string[] = [];
    const toolCalls: NonNullable<ModelMessage["toolCalls"]> = [];
    const toolResults: ModelMessage[] = [];
    for (const block of item.content) {
      if (typeof block === "string") {
        text.push(block);
      } else if (isObject(block) && block.type === "tool_use") {
        toolCalls.push({
          id: String(block.id ?? ""),
          name: String(block.name ?? ""),
          arguments: isObject(block.input) ? block.input : {},
        });
      } else if (isObject(block) && block.type === "tool_result") {
        toolResults.push({
          role: "tool",
          toolCallId: String(block.tool_use_id ?? ""),
          content: [{ type: "text", text: toolResultText(block.content) }],
        });
      } else if (isObject(block) && typeof block.text === "string") {
        text.push(block.text);
      }
    }
    if (text.length || toolCalls.length || !toolResults.length) {
      result.push({
        role: item.role,
        content: text.length ? [{ type: "text", text: text.join("") }] : [],
        ...(toolCalls.length ? { toolCalls } : {}),
      });
    }
    result.push(...toolResults);
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

/** Map an Agent RT finish reason back to Anthropic's `stop_reason`. */
function anthropicStopReason(response: ModelResponse): AnthropicTypes.StopReason {
  if (response.message.toolCalls?.length) return "tool_use";
  switch (response.finishReason) {
    case "tool_calls":
      return "tool_use";
    case "length":
      return "max_tokens";
    case "content_filter":
      return "refusal";
    default:
      return "end_turn";
  }
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
  messageSequence += 1;
  return {
    id: `msg_agent_rt_${messageSequence}`,
    type: "message",
    role: "assistant",
    model: response.model,
    content: [
      ...mcpContent,
      ...webSearchContent,
      ...executionContent,
      ...(text ? [{ type: "text" as const, text }] : []),
      ...(response.message.toolCalls ?? []).map((call) => ({
        type: "tool_use" as const,
        id: call.id,
        name: call.name,
        input: call.arguments,
      })),
    ],
    stop_reason: anthropicStopReason(response),
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.inputTokens ?? 0,
      output_tokens: response.usage?.outputTokens ?? 0,
    },
  };
}

type CreateParams = AnthropicTypes.MessageCreateParamsNonStreaming | JSONObject;

type MessagesCreate = {
  (
    params: AnthropicTypes.MessageCreateParamsStreaming | (JSONObject & { stream: true }),
  ): Promise<AsyncIterable<AnthropicTypes.RawMessageStreamEvent>>;
  (params: CreateParams): Promise<AnthropicTypes.Message>;
};

function parsedOutput(message: { content: unknown[] }, params: JSONObject): unknown {
  const text = message.content
    .map((block) =>
      isObject(block) && block.type === "text" && typeof block.text === "string" ? block.text : "",
    )
    .join("");
  if (!text) return null;
  let format: unknown = isObject(params.output_config) ? params.output_config.format : undefined;
  format ??= params.output_format;
  if (isObject(format) && typeof format.parse === "function") {
    return (format.parse as (content: string) => unknown)(text);
  }
  return JSON.parse(text);
}

export class Anthropic {
  static AnthropicError = AnthropicError;
  static APIError = APIError;
  static APIUserAbortError = APIUserAbortError;
  static APIConnectionError = APIConnectionError;
  static APIConnectionTimeoutError = APIConnectionTimeoutError;
  static BadRequestError = BadRequestError;
  static AuthenticationError = AuthenticationError;
  static PermissionDeniedError = PermissionDeniedError;
  static NotFoundError = NotFoundError;
  static ConflictError = ConflictError;
  static UnprocessableEntityError = UnprocessableEntityError;
  static RateLimitError = RateLimitError;
  static InternalServerError = InternalServerError;

  readonly provider: ModelProvider;
  readonly sandboxSession?: SandboxSession;
  readonly maxCodeToolRounds: number;
  readonly mcpClients: Record<string, MCPMigrationClient>;
  readonly webSearchProvider?: RetrievalProvider;
  readonly messages: {
    create: MessagesCreate;
    stream: (params: CreateParams) => MessageStream;
    parse: <T = unknown>(params: CreateParams) => Promise<AnthropicTypes.ParsedMessage<T>>;
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
      create: MessagesCreate;
      stream: (params: CreateParams) => MessageStream;
      parse: <T = unknown>(params: CreateParams) => Promise<AnthropicTypes.ParsedMessage<T>>;
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
    // Like the SDK, an unset baseURL/apiKey falls back to the environment.
    this.provider = options.provider ?? new AnthropicModelProvider({
      // The SDKs send every request through an injected `fetch`; tests use
      // it as an offline transport.
      ...(typeof options.fetch === "function" ? { fetch: options.fetch as typeof fetch } : {}),
      apiKey: options.apiKey ?? sdkEnv("ANTHROPIC_API_KEY"),
      baseUrl: options.baseURL ?? sdkEnv("ANTHROPIC_BASE_URL"),
      defaultModel: options.defaultModel,
    });
    this.sandboxSession = options.sandboxSession;
    this.maxCodeToolRounds = options.maxCodeToolRounds ?? 8;
    this.mcpClients = { ...(options.mcpClients ?? {}) };
    this.webSearchProvider = options.webSearchProvider;
    const createMessage = (params: CreateParams): Promise<AnthropicTypes.Message> =>
      withSDKErrors(errors, () => {
        const { stream: _stream, ...rest } = params as JSONObject;
        return createShape(rest);
      }) as Promise<AnthropicTypes.Message>;
    // `stream: true` returns the raw event stream; failures still surface
    // from create() as SDK errors, as in the SDK.
    const createOrStream = (async (params: CreateParams) => {
      if ((params as JSONObject).stream === true) {
        const message = await createMessage(params);
        return rawMessageStream(Promise.resolve(message));
      }
      return createMessage(params);
    }) as MessagesCreate;
    // messages.stream() sends a streaming request, as the SDK does, when no
    // local migration tool needs the bounded tool loop.
    const streamShape = async (params: JSONObject) => {
      const { stream: _stream, ...rest } = params;
      const { request, expanded } = await buildRequest(rest);
      const streaming = this.provider as ModelProvider & Partial<StreamingModelProvider>;
      const localTools =
        expanded.mcpBindings?.size || expanded.webSearchBindings?.size || this.sandboxSession;
      if (localTools || typeof streaming.stream !== "function") return createShape(rest);
      let completed: ModelResponse | undefined;
      for await (const event of streaming.stream(request)) {
        if (event.type === "completed" && event.response) completed = event.response;
      }
      if (!completed) throw new Error("Anthropic stream ended without a completed message");
      return anthropicShape(completed);
    };
    const streamMessage = (params: CreateParams): MessageStream =>
      new MessageStream(
        () =>
          withSDKErrors(errors, () => streamShape(params as JSONObject)) as Promise<
            AnthropicTypes.Message
          >,
      );
    const parseMessage = async <T = unknown>(
      params: CreateParams,
    ): Promise<AnthropicTypes.ParsedMessage<T>> => {
      const message = await createMessage(params);
      return { ...message, parsed_output: parsedOutput(message, params as JSONObject) as T | null };
    };
    const buildRequest = async (params: JSONObject) => {
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
      return { request, expanded };
    };
    const createShape = async (params: JSONObject) => {
      const { request, expanded } = await buildRequest(params);
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
      create: createOrStream,
      stream: streamMessage,
      parse: parseMessage,
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
    this.beta = {
      messages: { create: createOrStream, stream: streamMessage, parse: parseMessage },
    };
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

/** SDK type names reachable as `Anthropic.MessageParam`, `Anthropic.Messages.Message`, ... */
// eslint-disable-next-line @typescript-eslint/no-namespace
export declare namespace Anthropic {
  export type Model = AnthropicTypes.Model;
  export type MessageParam = AnthropicTypes.MessageParam;
  export type Message = AnthropicTypes.Message;
  export type ContentBlock = AnthropicTypes.ContentBlock;
  export type ContentBlockParam = AnthropicTypes.ContentBlockParam;
  export type TextBlock = AnthropicTypes.TextBlock;
  export type TextBlockParam = AnthropicTypes.TextBlockParam;
  export type ImageBlockParam = AnthropicTypes.ImageBlockParam;
  export type DocumentBlockParam = AnthropicTypes.DocumentBlockParam;
  export type ToolUseBlock = AnthropicTypes.ToolUseBlock;
  export type ToolUseBlockParam = AnthropicTypes.ToolUseBlockParam;
  export type ToolResultBlockParam = AnthropicTypes.ToolResultBlockParam;
  export type ThinkingBlock = AnthropicTypes.ThinkingBlock;
  export type ThinkingBlockParam = AnthropicTypes.ThinkingBlockParam;
  export type Tool = AnthropicTypes.Tool;
  export type ToolChoice = AnthropicTypes.ToolChoice;
  export type Usage = AnthropicTypes.Usage;
  export type StopReason = AnthropicTypes.StopReason;
  export type CacheControlEphemeral = AnthropicTypes.CacheControlEphemeral;
  export type MessageCreateParams = AnthropicTypes.MessageCreateParams;
  export type MessageCreateParamsStreaming = AnthropicTypes.MessageCreateParamsStreaming;
  export type MessageCreateParamsNonStreaming = AnthropicTypes.MessageCreateParamsNonStreaming;
  export type RawMessageStreamEvent = AnthropicTypes.RawMessageStreamEvent;
  export type MessageStreamEvent = AnthropicTypes.MessageStreamEvent;
  export type AnthropicError = InstanceType<typeof errors.BaseError>;
  export type APIError = InstanceType<typeof errors.APIError>;
  export type APIConnectionError = InstanceType<typeof errors.APIConnectionError>;
  export type RateLimitError = InstanceType<typeof errors.RateLimitError>;
  // eslint-disable-next-line @typescript-eslint/no-namespace
  export namespace Messages {
    export type MessageParam = AnthropicTypes.MessageParam;
    export type Message = AnthropicTypes.Message;
    export type ContentBlock = AnthropicTypes.ContentBlock;
    export type ContentBlockParam = AnthropicTypes.ContentBlockParam;
    export type TextBlock = AnthropicTypes.TextBlock;
    export type TextBlockParam = AnthropicTypes.TextBlockParam;
    export type ImageBlockParam = AnthropicTypes.ImageBlockParam;
    export type DocumentBlockParam = AnthropicTypes.DocumentBlockParam;
    export type ToolUseBlock = AnthropicTypes.ToolUseBlock;
    export type ToolUseBlockParam = AnthropicTypes.ToolUseBlockParam;
    export type ToolResultBlockParam = AnthropicTypes.ToolResultBlockParam;
    export type Tool = AnthropicTypes.Tool;
    export type Usage = AnthropicTypes.Usage;
    export type MessageCreateParams = AnthropicTypes.MessageCreateParams;
    export type MessageCreateParamsStreaming = AnthropicTypes.MessageCreateParamsStreaming;
    export type MessageCreateParamsNonStreaming = AnthropicTypes.MessageCreateParamsNonStreaming;
    export type RawMessageStreamEvent = AnthropicTypes.RawMessageStreamEvent;
    export type MessageStreamEvent = AnthropicTypes.MessageStreamEvent;
  }
}

export default Anthropic;
