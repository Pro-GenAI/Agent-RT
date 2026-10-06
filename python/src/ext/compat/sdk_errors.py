"""Vendor-SDK-shaped exception hierarchies for the OpenAI and Anthropic shims.

Migrated code catches ``openai.APIError`` / ``anthropic.RateLimitError`` and
constructs them in tests with the SDK signatures, so each vendor gets its own
hierarchy with the upstream class names, constructor signatures, and
attributes. Nothing here imports httpx or the vendor SDKs.
"""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

_STATUS_CLASSES = (
    (400, "BadRequestError"),
    (401, "AuthenticationError"),
    (403, "PermissionDeniedError"),
    (404, "NotFoundError"),
    (409, "ConflictError"),
    (422, "UnprocessableEntityError"),
    (429, "RateLimitError"),
)


def _headers_get(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    return getter(name)


def build_error_classes(
    module_name: str, base_name: str, *, extra_server_errors: Mapping[int, str] = {}
) -> dict[str, type]:
    """Create one vendor's exception hierarchy, keyed by class name."""

    def make(name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
        namespace.setdefault("__module__", module_name)
        return types.new_class(name, bases, {}, lambda ns: ns.update(namespace))

    def base_init(self: Any, *args: Any, **kwargs: Any) -> None:
        Exception.__init__(self, *args)

    base = make(base_name, (Exception,), {"__init__": base_init})

    def api_error_init(
        self: Any, message: str, request: Any = None, *, body: object | None = None
    ) -> None:
        Exception.__init__(self, message)
        self.message = message
        self.request = request
        self.body = body
        details = body.get("error", body) if isinstance(body, Mapping) else None
        details = details if isinstance(details, Mapping) else {}
        self.code = details.get("code")
        self.param = details.get("param")
        self.type = details.get("type")

    api_error = make("APIError", (base,), {"__init__": api_error_init})

    def validation_init(
        self: Any, response: Any, body: object | None, *, message: str | None = None
    ) -> None:
        api_error_init(
            self,
            message or "Data returned by API invalid for expected schema.",
            getattr(response, "request", None),
            body=body,
        )
        self.response = response
        self.status_code = getattr(response, "status_code", None)

    validation_error = make(
        "APIResponseValidationError", (api_error,), {"__init__": validation_init}
    )

    def status_init(
        self: Any, message: str, *, response: Any, body: object | None = None
    ) -> None:
        api_error_init(self, message, getattr(response, "request", None), body=body)
        self.response = response
        self.status_code = getattr(response, "status_code", None)
        headers = getattr(response, "headers", None)
        self.request_id = _headers_get(headers, "request-id") or _headers_get(
            headers, "x-request-id"
        )

    status_error = make("APIStatusError", (api_error,), {"__init__": status_init})

    def connection_init(
        self: Any, *, message: str = "Connection error.", request: Any = None
    ) -> None:
        api_error_init(self, message, request, body=None)

    connection_error = make(
        "APIConnectionError", (api_error,), {"__init__": connection_init}
    )

    def timeout_init(self: Any, request: Any = None) -> None:
        connection_init(self, message="Request timed out.", request=request)

    timeout_error = make(
        "APITimeoutError", (connection_error,), {"__init__": timeout_init}
    )

    classes: dict[str, type] = {
        base_name: base,
        "APIError": api_error,
        "APIResponseValidationError": validation_error,
        "APIStatusError": status_error,
        "APIConnectionError": connection_error,
        "APITimeoutError": timeout_error,
    }
    by_status: dict[int, type] = {}
    for status, name in _STATUS_CLASSES:
        classes[name] = by_status[status] = make(name, (status_error,), {})
    classes["InternalServerError"] = make("InternalServerError", (status_error,), {})
    for status, name in extra_server_errors.items():
        classes[name] = by_status[status] = make(
            name, (classes["InternalServerError"],), {}
        )
    classes["_by_status"] = by_status  # type: ignore[assignment]
    return classes


def _json_body(message: str) -> Any:
    """The JSON error body embedded in a transport failure message, if any."""
    start = message.find("{")
    if start < 0:
        return None
    try:
        return json.loads(message[start:])
    except ValueError:
        return None


def translate_error(classes: Mapping[str, Any], exc: BaseException) -> BaseException:
    """Map an Agent RT provider failure onto the vendor hierarchy.

    Errors that already belong to the hierarchy pass through unchanged; errors
    without an HTTP status or a connection/timeout shape pass through too, so
    programming errors are never disguised as API errors.
    """
    if isinstance(exc, classes["APIError"]):
        return exc
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    cause = exc.__cause__
    if status is None and isinstance(getattr(cause, "status_code", None), int):
        # Agent RT wraps an HTTP 429 in RateLimitExceededError; the SDK raises
        # RateLimitError with the response's status, headers, and body.
        return translate_error(classes, cause)
    message = getattr(exc, "message", None) or str(exc)
    if isinstance(status, int):
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) != status:
            response = SimpleNamespace(
                status_code=status,
                headers=dict(getattr(exc, "headers", None) or {}),
                request=None,
            )
        if status >= 500:
            cls = classes["_by_status"].get(status, classes["InternalServerError"])
        else:
            cls = classes["_by_status"].get(status, classes["APIStatusError"])
        body = getattr(exc, "body", None)
        if body is None:
            body = _json_body(message)
        return cls(message, response=response, body=body)
    if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__:
        return classes["APITimeoutError"](request=getattr(exc, "request", None))
    if isinstance(exc, ConnectionError) or "ConnectionError" in type(exc).__name__:
        return classes["APIConnectionError"](
            message=message, request=getattr(exc, "request", None)
        )
    return exc


def public_error_classes(classes: Mapping[str, Any]) -> dict[str, type]:
    return {name: value for name, value in classes.items() if not name.startswith("_")}


OPENAI_ERRORS = build_error_classes("agent_rt.openai", "OpenAIError")
ANTHROPIC_ERRORS = build_error_classes(
    "agent_rt.anthropic",
    "AnthropicError",
    extra_server_errors={503: "ServiceUnavailableError", 529: "OverloadedError"},
)
