"""Structured-output and tool-argument validation against JSON schemas."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ext.runtime.models.conversion import _is_provider_state
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import ModelMessage, ToolCall

if TYPE_CHECKING:
    from ext.runtime.agent.config import AgentOutputRequirements


class StructuredOutputValidationError(ValueError):
    def __init__(self, issues: Sequence[str], raw_output: Any = None) -> None:
        self.issues = tuple(issues)
        self.raw_output = raw_output
        super().__init__(
            "structured output validation failed: " + "; ".join(self.issues)
        )


def _plain_message_text(message: ModelMessage) -> str:
    """Text-only view of a message; JSON parts are intentionally excluded."""
    return "".join(part.text or "" for part in message.content if part.type == "text")


def _extract_structured_output(message: ModelMessage) -> Any:
    for part in message.content:
        if (
            part.type == "json"
            and part.data is not None
            and not _is_provider_state(part)
        ):
            return part.data
    text = _plain_message_text(message).strip()
    if not text:
        raise StructuredOutputValidationError(("response did not contain JSON",), text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputValidationError(
            (f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",),
            text,
        ) from exc


def _matches_schema_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (
            isinstance(value, float) and value.is_integer()
        )
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality: ``True`` is not ``1`` and containers compare recursively."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return type(left) is type(right) and left == right


_MAX_SCHEMA_DEPTH = 32


def _resolve_schema_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#"):
        raise ValueError(f"unsupported $ref (only local '#/...' refs): {reference}")
    node: Any = root
    for token in reference[1:].split("/")[1:] if reference != "#" else ():
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and key in node:
            node = node[key]
        else:
            raise ValueError(f"unresolvable $ref: {reference}")
    if not isinstance(node, Mapping):
        raise ValueError(f"$ref does not point to a schema: {reference}")
    return node


def _validate_structured_value(
    value: Any,
    schema: Mapping[str, Any],
    path: str = "$",
    *,
    root: Mapping[str, Any] | None = None,
    depth: int = 0,
) -> tuple[str, ...]:
    """Validate ``value`` against a practical JSON Schema subset.

    Supported: ``type`` (string or list), ``enum``, ``const``, ``properties``,
    ``required``, ``additionalProperties`` (bool or schema), ``items``,
    ``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``,
    ``minLength``/``maxLength``/``pattern``, ``minItems``/``maxItems``/
    ``uniqueItems``, ``minProperties``/``maxProperties``, ``allOf``/``anyOf``/
    ``oneOf``/``not``, and local ``$ref``. Unknown keywords are ignored.
    """
    root = root if root is not None else schema
    if depth > _MAX_SCHEMA_DEPTH:
        return (f"{path}: schema nesting is too deep",)
    issues: list[str] = []

    def sub(
        child_value: Any, child_schema: Mapping[str, Any], child_path: str
    ) -> tuple[str, ...]:
        return _validate_structured_value(
            child_value, child_schema, child_path, root=root, depth=depth + 1
        )

    reference = schema.get("$ref")
    if isinstance(reference, str):
        try:
            issues.extend(sub(value, _resolve_schema_ref(root, reference), path))
        except ValueError as exc:
            return (f"{path}: {exc}",)

    expected = schema.get("type")
    if isinstance(expected, str):
        types: list[str] = [expected]
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        types = [item for item in expected if isinstance(item, str)]
    else:
        types = []
    if types and not any(_matches_schema_type(value, item) for item in types):
        return (f"{path}: expected {' or '.join(types)}",)

    if "const" in schema and not _json_equal(value, schema["const"]):
        issues.append(f"{path}: value does not equal the required constant")
    enum_values = schema.get("enum")
    if (
        isinstance(enum_values, Sequence)
        and not isinstance(enum_values, (str, bytes))
        and not any(_json_equal(value, candidate) for candidate in enum_values)
    ):
        issues.append(f"{path}: value is not in enum")

    all_of = schema.get("allOf")
    if isinstance(all_of, Sequence) and not isinstance(all_of, (str, bytes)):
        for child in all_of:
            if isinstance(child, Mapping):
                issues.extend(sub(value, child, path))
    any_of = schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, (str, bytes)):
        children = [child for child in any_of if isinstance(child, Mapping)]
        if children and not any(not sub(value, child, path) for child in children):
            issues.append(f"{path}: value does not match any allowed schema")
    one_of = schema.get("oneOf")
    if isinstance(one_of, Sequence) and not isinstance(one_of, (str, bytes)):
        children = [child for child in one_of if isinstance(child, Mapping)]
        matches = sum(1 for child in children if not sub(value, child, path))
        if children and matches != 1:
            issues.append(f"{path}: value must match exactly one allowed schema")
    negated = schema.get("not")
    if isinstance(negated, Mapping) and not sub(value, negated, path):
        issues.append(f"{path}: value matches a disallowed schema")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for keyword, failed in (
            ("minimum", lambda limit: value < limit),
            ("maximum", lambda limit: value > limit),
            ("exclusiveMinimum", lambda limit: value <= limit),
            ("exclusiveMaximum", lambda limit: value >= limit),
        ):
            limit = schema.get(keyword)
            if (
                isinstance(limit, (int, float))
                and not isinstance(limit, bool)
                and failed(limit)
            ):
                issues.append(f"{path}: violates {keyword} {limit}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            issues.append(f"{path}: shorter than minLength {min_length}")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            issues.append(f"{path}: longer than maxLength {max_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    issues.append(f"{path}: does not match pattern")
            except re.error:
                issues.append(f"{path}: schema pattern is invalid")

    if isinstance(value, dict):
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    issues.append(f"{path}.{key}: required property is missing")

        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                issues.extend(sub(value[key], child_schema, f"{path}.{key}"))

        additional = schema.get("additionalProperties")
        extras = [key for key in value if key not in properties]
        if additional is False:
            for key in extras:
                issues.append(f"{path}.{key}: additional property is not allowed")
        elif isinstance(additional, Mapping):
            for key in extras:
                issues.extend(sub(value[key], additional, f"{path}.{key}"))

        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            issues.append(f"{path}: fewer than minProperties {min_properties}")
        max_properties = schema.get("maxProperties")
        if isinstance(max_properties, int) and len(value) > max_properties:
            issues.append(f"{path}: more than maxProperties {max_properties}")

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                issues.extend(sub(item, item_schema, f"{path}[{index}]"))
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            issues.append(f"{path}: fewer than minItems {min_items}")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            issues.append(f"{path}: more than maxItems {max_items}")
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(value):
                if any(_json_equal(item, earlier) for earlier in value[:index]):
                    issues.append(f"{path}[{index}]: duplicate item")
                    break

    return tuple(issues)


class ToolArgumentValidationError(ValueError):
    def __init__(self, tool_name: str, issues: Sequence[str]) -> None:
        self.tool_name = tool_name
        self.issues = tuple(issues)
        super().__init__(
            f"invalid arguments for tool {tool_name}: " + "; ".join(self.issues)
        )


def validate_tool_arguments(
    definition: ToolDefinition,
    arguments: Mapping[str, Any],
) -> None:
    issues = _validate_structured_value(arguments, definition.input_schema)
    if issues:
        raise ToolArgumentValidationError(definition.name, issues)


def _tool_validation_error_payload(
    call: ToolCall,
    issues: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "error": {
            "type": "tool_argument_validation",
            "tool": call.name,
            "issues": list(issues),
        }
    }


def _validate_structured_message(
    message: ModelMessage,
    requirements: AgentOutputRequirements,
) -> Any:
    value = _extract_structured_output(message)
    if requirements.schema is not None:
        issues = _validate_structured_value(value, requirements.schema)
        if issues:
            raise StructuredOutputValidationError(issues, value)
    return value
