from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ext.compat._format import safe_format


class PromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = template

    def format(self, **kwargs: Any) -> str:
        return safe_format(self.template, kwargs)

    def partial_format(self, **kwargs: Any) -> PromptTemplate:
        return PromptTemplate(self.template.format_map(_PartialDict(kwargs)))

    def __str__(self) -> str:
        return self.template


class _PartialDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class ChatPromptMessage:
    role: Any
    content: str


class ChatPromptTemplate:
    def __init__(
        self, message_templates: Sequence[ChatPromptMessage | Mapping[str, Any]]
    ) -> None:
        self.message_templates = list(message_templates)

    def format_messages(self, **kwargs: Any) -> list[Any]:
        from ext.compat.llamaindex import ChatMessage

        result = []
        for message in self.message_templates:
            if isinstance(message, Mapping):
                role = message["role"]
                content = str(message["content"])
            else:
                role = message.role
                content = message.content
            result.append(ChatMessage(role=role, content=safe_format(content, kwargs)))
        return result


class JSONOutputParser:
    def __init__(self, schema_hint: str | None = None) -> None:
        self.schema_hint = schema_hint

    def parse(self, output: str) -> Any:
        return json.loads(output)

    def format(self, prompt: str) -> str:
        if not self.schema_hint:
            return prompt
        return (
            f"{prompt}\n\nReturn valid JSON matching this schema:\n{self.schema_hint}"
        )


class CallbackManager:
    def __init__(self) -> None:
        self._handlers: list[Callable[[Mapping[str, Any]], Any]] = []

    def on(self, handler: Callable[[Mapping[str, Any]], Any]) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def emit(self, event: Mapping[str, Any]) -> None:
        import inspect

        for handler in tuple(self._handlers):
            value = handler(event)
            if inspect.isawaitable(value):
                await value


class _Settings:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.llm: Any = None
        self.embed_model: Any = None
        self.callback_manager = CallbackManager()
        self.tokenizer: Callable[[str], Sequence[Any]] | None = None


Settings = _Settings()
