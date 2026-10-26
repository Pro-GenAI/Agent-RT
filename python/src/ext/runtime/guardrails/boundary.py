"""Input/output boundary guardrails and the default guardrail set."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ext.runtime.guardrails.results import GuardrailResult
from ext.runtime.types import ContentPart, ModelMessage

InputGuardrail = Callable[[Sequence[ModelMessage]], GuardrailResult]


OutputGuardrail = Callable[[ModelMessage], GuardrailResult]


@dataclass(frozen=True)
class BoundaryGuardrailPolicy:
    max_input_characters: int = 1_000_000
    max_output_characters: int = 1_000_000
    max_content_parts: int = 1024
    sanitize_control_characters: bool = True
    require_assistant_output: bool = True

    def __post_init__(self) -> None:
        if self.max_input_characters < 1:
            raise ValueError("max_input_characters must be positive")
        if self.max_output_characters < 1:
            raise ValueError("max_output_characters must be positive")
        if self.max_content_parts < 1:
            raise ValueError("max_content_parts must be positive")


def _guardrail_content_size(parts: Sequence[ContentPart]) -> int:
    total = 0
    for part in parts:
        if part.text is not None:
            total += len(part.text)
        if part.data is not None:
            try:
                total += len(
                    json.dumps(part.data, separators=(",", ":"), ensure_ascii=False)
                )
            except (TypeError, ValueError):
                total += len(str(part.data))
    return total


def _sanitize_guardrail_text(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in "\t\n\r" or ord(character) >= 32 and ord(character) != 127
    )


def make_default_input_guardrail(
    policy: BoundaryGuardrailPolicy = BoundaryGuardrailPolicy(),
) -> InputGuardrail:
    def guard(messages: Sequence[ModelMessage]) -> GuardrailResult:
        user_messages = [message for message in messages if message.role == "user"]
        part_count = sum(len(message.content) for message in user_messages)
        if part_count > policy.max_content_parts:
            return GuardrailResult(
                action="block",
                reason="user input contains too many content parts",
                classifications=("input:content_parts_exceeded",),
            )
        character_count = sum(
            _guardrail_content_size(message.content) for message in user_messages
        )
        if character_count > policy.max_input_characters:
            return GuardrailResult(
                action="block",
                reason="user input exceeds the configured character limit",
                classifications=("input:size_exceeded",),
            )
        if not policy.sanitize_control_characters:
            return GuardrailResult()
        transformed: list[ModelMessage] = []
        changed = False
        for message in messages:
            if message.role != "user":
                transformed.append(message)
                continue
            parts: list[ContentPart] = []
            for part in message.content:
                if part.text is None:
                    parts.append(part)
                    continue
                text = _sanitize_guardrail_text(part.text)
                changed = changed or text != part.text
                parts.append(replace(part, text=text))
            transformed.append(replace(message, content=tuple(parts)))
        return GuardrailResult(
            action="transform" if changed else "allow",
            value=tuple(transformed) if changed else None,
            classifications=("input:control_characters_sanitized",) if changed else (),
        )

    return guard


def make_default_output_guardrail(
    policy: BoundaryGuardrailPolicy = BoundaryGuardrailPolicy(),
) -> OutputGuardrail:
    def guard(message: ModelMessage) -> GuardrailResult:
        if policy.require_assistant_output and message.role != "assistant":
            return GuardrailResult(
                action="block",
                reason="model output must use the assistant role",
                classifications=("output:invalid_role",),
            )
        if len(message.content) > policy.max_content_parts:
            return GuardrailResult(
                action="block",
                reason="model output contains too many content parts",
                classifications=("output:content_parts_exceeded",),
            )
        if _guardrail_content_size(message.content) > policy.max_output_characters:
            return GuardrailResult(
                action="block",
                reason="model output exceeds the configured character limit",
                classifications=("output:size_exceeded",),
            )
        if not policy.sanitize_control_characters:
            return GuardrailResult()
        parts: list[ContentPart] = []
        changed = False
        for part in message.content:
            if part.text is None:
                parts.append(part)
                continue
            text = _sanitize_guardrail_text(part.text)
            changed = changed or text != part.text
            parts.append(replace(part, text=text))
        return GuardrailResult(
            action="transform" if changed else "allow",
            value=replace(message, content=tuple(parts)) if changed else None,
            classifications=("output:control_characters_sanitized",) if changed else (),
        )

    return guard
