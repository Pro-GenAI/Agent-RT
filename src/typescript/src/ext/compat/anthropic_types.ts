/**
 * Anthropic Messages API types for the Agent RT Anthropic shim, so code typed
 * against `Anthropic.MessageParam` or `Anthropic.Messages.Message` still
 * type-checks after migration. Type-only: nothing here exists at runtime.
 */

export type Model = string;

export type CacheControlEphemeral = { type: 'ephemeral'; ttl?: '5m' | '1h' };

export type CitationParam = Record<string, unknown>;

export type TextBlockParam = {
	type: 'text';
	text: string;
	cache_control?: CacheControlEphemeral | null;
	citations?: CitationParam[] | null;
};

export type Base64ImageSource = {
	type: 'base64';
	media_type: 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp';
	data: string;
};

export type URLImageSource = { type: 'url'; url: string };

export type ImageBlockParam = {
	type: 'image';
	source: Base64ImageSource | URLImageSource;
	cache_control?: CacheControlEphemeral | null;
};

export type DocumentBlockParam = {
	type: 'document';
	source: Record<string, unknown>;
	title?: string | null;
	context?: string | null;
	citations?: { enabled?: boolean } | null;
	cache_control?: CacheControlEphemeral | null;
};

export type ToolUseBlockParam = {
	type: 'tool_use';
	id: string;
	name: string;
	input: unknown;
	cache_control?: CacheControlEphemeral | null;
};

export type ToolResultBlockParam = {
	type: 'tool_result';
	tool_use_id: string;
	content?:
		string | Array<TextBlockParam | ImageBlockParam | DocumentBlockParam>;
	is_error?: boolean;
	cache_control?: CacheControlEphemeral | null;
};

export type ThinkingBlockParam = {
	type: 'thinking';
	thinking: string;
	signature: string;
};

export type RedactedThinkingBlockParam = {
	type: 'redacted_thinking';
	data: string;
};

export type ContentBlockParam =
	| TextBlockParam
	| ImageBlockParam
	| DocumentBlockParam
	| ToolUseBlockParam
	| ToolResultBlockParam
	| ThinkingBlockParam
	| RedactedThinkingBlockParam;

export type TextBlock = {
	type: 'text';
	text: string;
	citations?: unknown[] | null;
};

export type ToolUseBlock = {
	type: 'tool_use';
	id: string;
	name: string;
	input: unknown;
};

export type ThinkingBlock = {
	type: 'thinking';
	thinking: string;
	signature: string;
};

export type RedactedThinkingBlock = { type: 'redacted_thinking'; data: string };

export type ContentBlock =
	TextBlock | ToolUseBlock | ThinkingBlock | RedactedThinkingBlock;

export type MessageParam = {
	role: 'user' | 'assistant';
	content: string | Array<ContentBlockParam | ContentBlock>;
};

export type StopReason =
	| 'end_turn'
	| 'max_tokens'
	| 'stop_sequence'
	| 'tool_use'
	| 'pause_turn'
	| 'refusal';

export type Usage = {
	input_tokens: number;
	output_tokens: number;
	cache_creation_input_tokens?: number | null;
	cache_read_input_tokens?: number | null;
};

export type Message = {
	id: string;
	type: 'message';
	role: 'assistant';
	model: Model;
	content: ContentBlock[];
	stop_reason: StopReason | null;
	stop_sequence: string | null;
	usage: Usage;
};

export type Tool = {
	name: string;
	description?: string;
	input_schema: {
		type: 'object';
		properties?: unknown;
		required?: string[];
		[key: string]: unknown;
	};
	cache_control?: CacheControlEphemeral | null;
	type?: 'custom' | null;
};

export type ToolUnion =
	Tool | { type: string; name: string; [key: string]: unknown };

export type ToolChoice =
	| { type: 'auto'; disable_parallel_tool_use?: boolean }
	| { type: 'any'; disable_parallel_tool_use?: boolean }
	| { type: 'tool'; name: string; disable_parallel_tool_use?: boolean }
	| { type: 'none' };

export type ThinkingConfigParam =
	| { type: 'enabled'; budget_tokens: number }
	| { type: 'disabled' }
	| { type: 'adaptive' };

export type MessageCreateParamsBase = {
	model: Model;
	messages: MessageParam[];
	max_tokens: number;
	system?: string | TextBlockParam[];
	temperature?: number;
	top_p?: number;
	top_k?: number;
	stop_sequences?: string[];
	tools?: ToolUnion[];
	tool_choice?: ToolChoice;
	thinking?: ThinkingConfigParam;
	metadata?: { user_id?: string | null };
	stream?: boolean;
	[key: string]: unknown;
};

export type MessageCreateParamsNonStreaming = MessageCreateParamsBase & {
	stream?: false;
};

export type MessageCreateParamsStreaming = MessageCreateParamsBase & {
	stream: true;
};

export type MessageCreateParams =
	MessageCreateParamsNonStreaming | MessageCreateParamsStreaming;

export type TextDelta = { type: 'text_delta'; text: string };
export type InputJSONDelta = { type: 'input_json_delta'; partial_json: string };
export type ThinkingDelta = { type: 'thinking_delta'; thinking: string };
export type SignatureDelta = { type: 'signature_delta'; signature: string };

export type MessageDeltaUsage = {
	output_tokens: number;
	input_tokens?: number | null;
	cache_creation_input_tokens?: number | null;
	cache_read_input_tokens?: number | null;
};

export type RawMessageStartEvent = { type: 'message_start'; message: Message };
export type RawMessageDeltaEvent = {
	type: 'message_delta';
	delta: { stop_reason: StopReason | null; stop_sequence: string | null };
	usage: MessageDeltaUsage;
};
export type RawMessageStopEvent = { type: 'message_stop' };
export type RawContentBlockStartEvent = {
	type: 'content_block_start';
	index: number;
	content_block: ContentBlock;
};
export type RawContentBlockDeltaEvent = {
	type: 'content_block_delta';
	index: number;
	delta: TextDelta | InputJSONDelta | ThinkingDelta | SignatureDelta;
};
export type RawContentBlockStopEvent = {
	type: 'content_block_stop';
	index: number;
};

export type RawMessageStreamEvent =
	| RawMessageStartEvent
	| RawMessageDeltaEvent
	| RawMessageStopEvent
	| RawContentBlockStartEvent
	| RawContentBlockDeltaEvent
	| RawContentBlockStopEvent;

export type MessageStreamEvent = RawMessageStreamEvent;

/** A `messages.parse()` result: the message plus its parsed structured output. */
export type ParsedMessage<T = unknown> = Message & {
	parsed_output: T | null;
	stop_details?: { category?: string; [key: string]: unknown } | null;
};
