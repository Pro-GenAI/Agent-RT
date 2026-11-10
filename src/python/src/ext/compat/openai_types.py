"""``openai.types`` shapes for the Agent RT OpenAI shim.

Response objects are attribute namespaces with the SDK class names, so
``isinstance(response, ChatCompletion)``, keyword construction in test
doubles, and ``model_dump()`` keep working after migration. Request params
are ``TypedDict`` definitions, which are plain dicts at runtime, matching the
SDK. ``MODULES`` maps each upstream submodule path to the names it exports.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from ext.compat.anthropic_types import BaseModel as _NamespaceModel


class BaseModel(_NamespaceModel):
    def __class_getitem__(cls, item: Any) -> Any:
        # SDK parsed types are generics (`ParsedResponse[Model]`,
        # `ParsedResponseOutputText[Model]`); the parameter is erased here.
        return cls

    @classmethod
    def model_validate(cls, value: Any, **_: Any) -> Any:
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    @classmethod
    def model_construct(cls, **values: Any) -> Any:
        return cls(**values)


# Chat Completions
class CompletionUsage(BaseModel):
    pass


class Function(BaseModel):
    pass


class ChatCompletionMessageToolCall(BaseModel):
    pass


class ChatCompletionMessage(BaseModel):
    pass


class Choice(BaseModel):
    pass


class ChatCompletion(BaseModel):
    pass


class ChoiceDeltaToolCallFunction(BaseModel):
    pass


class ChoiceDeltaToolCall(BaseModel):
    pass


class ChoiceDelta(BaseModel):
    pass


class ChunkChoice(BaseModel):
    pass


class ChatCompletionChunk(BaseModel):
    pass


class ChatCompletionMessageParam(TypedDict, total=False):
    role: str
    content: Any
    name: str
    tool_calls: list[Any]
    tool_call_id: str


ChatCompletionSystemMessageParam = ChatCompletionMessageParam
ChatCompletionUserMessageParam = ChatCompletionMessageParam
ChatCompletionAssistantMessageParam = ChatCompletionMessageParam
ChatCompletionToolMessageParam = ChatCompletionMessageParam
ChatCompletionDeveloperMessageParam = ChatCompletionMessageParam


class ChatCompletionToolParam(TypedDict, total=False):
    type: Literal["function"]
    function: dict[str, Any]


ChatCompletionToolChoiceOptionParam = Any
ChatModel = str


# Embeddings
class Embedding(BaseModel):
    pass


class EmbeddingUsage(BaseModel):
    pass


class CreateEmbeddingResponse(BaseModel):
    pass


# Responses
class ResponseUsage(BaseModel):
    pass


class ResponseOutputText(BaseModel):
    pass


class ResponseOutputMessage(BaseModel):
    pass


class ResponseFunctionToolCall(BaseModel):
    pass


class Response(BaseModel):
    def __getattr__(self, name: str) -> Any:
        # The SDK computes `output_text` from `output`; test doubles that
        # build a Response without it still read it.
        if name == "output_text":
            return "".join(
                getattr(part, "text", "") or ""
                for item in self.__dict__.get("output") or ()
                if getattr(item, "type", None) == "message"
                for part in getattr(item, "content", None) or ()
                if getattr(part, "type", None) == "output_text"
            )
        raise AttributeError(name)


class ParsedResponseOutputText(ResponseOutputText):
    pass


class ParsedResponseOutputMessage(ResponseOutputMessage):
    pass


class ParsedResponse(Response):
    def __getattr__(self, name: str) -> Any:
        # Like the SDK, `output_parsed` is the first output text's `parsed`.
        if name == "output_parsed":
            for item in self.__dict__.get("output") or ():
                if getattr(item, "type", None) != "message":
                    continue
                for part in getattr(item, "content", None) or ():
                    if getattr(part, "type", None) == "output_text":
                        return getattr(part, "parsed", None)
            return None
        return super().__getattr__(name)


class ResponseTextDeltaEvent(BaseModel):
    pass


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    pass


class ResponseCompletedEvent(BaseModel):
    pass


ResponseStreamEvent = (
    ResponseTextDeltaEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseCompletedEvent
)
ResponseInputParam = list[Any]
ResponseInputItemParam = dict[str, Any]
EasyInputMessageParam = dict[str, Any]


_CHAT = {
    "ChatCompletion": ChatCompletion,
    "ChatCompletionChunk": ChatCompletionChunk,
    "ChatCompletionMessage": ChatCompletionMessage,
    "ChatCompletionMessageToolCall": ChatCompletionMessageToolCall,
    "ChatCompletionMessageParam": ChatCompletionMessageParam,
    "ChatCompletionSystemMessageParam": ChatCompletionSystemMessageParam,
    "ChatCompletionUserMessageParam": ChatCompletionUserMessageParam,
    "ChatCompletionAssistantMessageParam": ChatCompletionAssistantMessageParam,
    "ChatCompletionToolMessageParam": ChatCompletionToolMessageParam,
    "ChatCompletionDeveloperMessageParam": ChatCompletionDeveloperMessageParam,
    "ChatCompletionToolParam": ChatCompletionToolParam,
    "ChatCompletionToolChoiceOptionParam": ChatCompletionToolChoiceOptionParam,
}
_RESPONSES = {
    "EasyInputMessageParam": EasyInputMessageParam,
    "ParsedResponse": ParsedResponse,
    "ParsedResponseOutputMessage": ParsedResponseOutputMessage,
    "ParsedResponseOutputText": ParsedResponseOutputText,
    "Response": Response,
    "ResponseCompletedEvent": ResponseCompletedEvent,
    "ResponseFunctionCallArgumentsDeltaEvent": ResponseFunctionCallArgumentsDeltaEvent,
    "ResponseFunctionToolCall": ResponseFunctionToolCall,
    "ResponseInputItemParam": ResponseInputItemParam,
    "ResponseInputParam": ResponseInputParam,
    "ResponseOutputMessage": ResponseOutputMessage,
    "ResponseOutputText": ResponseOutputText,
    "ResponseStreamEvent": ResponseStreamEvent,
    "ResponseTextDeltaEvent": ResponseTextDeltaEvent,
    "ResponseUsage": ResponseUsage,
}

# Upstream submodule path (relative to ``openai``) -> exported names.
MODULES: dict[str, dict[str, Any]] = {
    "types": {
        "ChatModel": ChatModel,
        "CompletionUsage": CompletionUsage,
        "CreateEmbeddingResponse": CreateEmbeddingResponse,
        "Embedding": Embedding,
    },
    "types.chat": _CHAT,
    "types.chat.chat_completion": {"ChatCompletion": ChatCompletion, "Choice": Choice},
    "types.chat.chat_completion_chunk": {
        "ChatCompletionChunk": ChatCompletionChunk,
        "Choice": ChunkChoice,
        "ChoiceDelta": ChoiceDelta,
        "ChoiceDeltaToolCall": ChoiceDeltaToolCall,
        "ChoiceDeltaToolCallFunction": ChoiceDeltaToolCallFunction,
    },
    "types.chat.chat_completion_message": {
        "ChatCompletionMessage": ChatCompletionMessage
    },
    "types.chat.chat_completion_message_tool_call": {
        "ChatCompletionMessageToolCall": ChatCompletionMessageToolCall,
        "Function": Function,
    },
    "types.chat.chat_completion_message_param": {
        "ChatCompletionMessageParam": ChatCompletionMessageParam
    },
    "types.completion_usage": {"CompletionUsage": CompletionUsage},
    "types.create_embedding_response": {
        "CreateEmbeddingResponse": CreateEmbeddingResponse,
        "Usage": EmbeddingUsage,
    },
    "types.embedding": {"Embedding": Embedding},
    "types.responses": _RESPONSES,
}
