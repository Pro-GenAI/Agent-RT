"""``anthropic.types`` shapes for the Agent RT Anthropic shim.

Response objects are attribute namespaces with the SDK class names, so
``isinstance(response, Message)`` and ``model_dump()`` keep working after
migration. Request params are ``TypedDict`` definitions, which are plain
dicts at runtime, matching the SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Literal, TypedDict


def _dump(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _dump(item) for key, item in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


class BaseModel(SimpleNamespace):
    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict[str, Any]:
        data = _dump(self)
        if exclude_none:
            data = {key: item for key, item in data.items() if item is not None}
        return data

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        import json

        return json.dumps(self.model_dump(**kwargs))


class Usage(BaseModel):
    pass


class TextBlock(BaseModel):
    pass


class ToolUseBlock(BaseModel):
    pass


class ThinkingBlock(BaseModel):
    pass


class Message(BaseModel):
    pass


ContentBlock = TextBlock | ToolUseBlock | ThinkingBlock
StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]
Model = str


class CacheControlEphemeralParam(TypedDict, total=False):
    type: Literal["ephemeral"]
    ttl: str


class TextBlockParam(TypedDict, total=False):
    type: Literal["text"]
    text: str
    cache_control: CacheControlEphemeralParam | None
    citations: Iterable[Any] | None


class ImageBlockParam(TypedDict, total=False):
    type: Literal["image"]
    source: dict[str, Any]
    cache_control: CacheControlEphemeralParam | None


class DocumentBlockParam(TypedDict, total=False):
    type: Literal["document"]
    source: dict[str, Any]
    title: str | None
    context: str | None
    cache_control: CacheControlEphemeralParam | None


class ToolUseBlockParam(TypedDict, total=False):
    type: Literal["tool_use"]
    id: str
    name: str
    input: object
    cache_control: CacheControlEphemeralParam | None


class ToolResultBlockParam(TypedDict, total=False):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | Iterable[TextBlockParam | ImageBlockParam]
    is_error: bool
    cache_control: CacheControlEphemeralParam | None


class ThinkingBlockParam(TypedDict, total=False):
    type: Literal["thinking"]
    thinking: str
    signature: str


ContentBlockParam = (
    TextBlockParam
    | ImageBlockParam
    | DocumentBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ThinkingBlockParam
)


class MessageParam(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str | Iterable[ContentBlockParam | ContentBlock]


class ToolParam(TypedDict, total=False):
    name: str
    description: str
    input_schema: dict[str, Any]
    cache_control: CacheControlEphemeralParam | None


class ToolChoiceParam(TypedDict, total=False):
    type: Literal["auto", "any", "tool", "none"]
    name: str
    disable_parallel_tool_use: bool


class ThinkingConfigParam(TypedDict, total=False):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int


TYPES_EXPORTS: dict[str, Any] = {
    name: value
    for name, value in dict(globals()).items()
    if name[:1].isupper()
    and name != "TYPES_EXPORTS"
    and name not in {"Any", "Iterable", "Literal", "SimpleNamespace", "TypedDict"}
}
