"""Persistent, versioned benchmark cache for main LLM and Decision-model responses.

Only use synthetic benchmark data. Caches hold model outputs, never tool side effects.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Awaitable, Callable

CACHE_VERSION = 2
CACHE_MODES = ("off", "read", "read-write", "refresh")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    raise TypeError(f"cache request contains non-serializable {type(value).__name__}")


def cache_key(request: Any) -> str:
    payload = json.dumps({"version": CACHE_VERSION, "request": _jsonable(request)},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheMiss(RuntimeError):
    """Read-only replay must never silently reach a live model."""


class NonCacheableResponse(RuntimeError):
    """A failed or incomplete provider response must never be persisted."""


class ModelCapabilityIssue(RuntimeError):
    """A completed but unusable model turn, cached as a typed benchmark outcome."""

    def __init__(self, reason: str = "empty_completion") -> None:
        if reason != "empty_completion":
            raise ValueError("unsupported model capability issue")
        self.reason = reason
        super().__init__(reason)


_CAPABILITY_MARKER = "_swarmguard_model_capability"
_CAPABILITY_REASONS = frozenset({"empty_completion"})


def _capability_record(reason: str) -> dict[str, str]:
    return {_CAPABILITY_MARKER: reason}


def _capability_reason(value: Any) -> str | None:
    if not isinstance(value, Mapping) or _CAPABILITY_MARKER not in value:
        return None
    reason = value[_CAPABILITY_MARKER]
    if (
        set(value) != {_CAPABILITY_MARKER}
        or not isinstance(reason, str)
        or reason not in _CAPABILITY_REASONS
    ):
        raise NonCacheableResponse("invalid cached model capability marker")
    return reason


def require_successful_decision(answers: Any, questions: Mapping[str, Any]) -> None:
    """Reject error envelopes and malformed Decision-model results before caching."""
    if not isinstance(answers, Mapping) or answers.get("error") or answers.get("status") in (
        "failed", "error", "cancelled", "incomplete",
    ):
        raise NonCacheableResponse("Decision model returned an error or invalid payload")
    for name, question in questions.items():
        value = answers.get(name)
        if not isinstance(value, Mapping) or value.get("error"):
            raise NonCacheableResponse("Decision model returned a missing or failed answer")
        kind = question.get("type") if isinstance(question, Mapping) else None
        if kind == "noul":
            probability = value.get("noul")
            if (isinstance(probability, bool) or
                    not isinstance(probability, (int, float)) or
                    not 0 <= probability <= 1):
                raise NonCacheableResponse("Decision model returned an invalid probability")
        elif kind == "choice" and not isinstance(value.get("choice"), str):
            raise NonCacheableResponse("Decision model returned an invalid choice")


def require_successful_model_response(response: Any) -> None:
    """Do not store provider-level error, truncated or empty model responses."""
    from ext.runtime.models.interfaces import ModelResponse

    if not isinstance(response, ModelResponse):
        raise NonCacheableResponse("Agent-RT provider returned an invalid response")
    if response.finish_reason in ("error", "length", "content_filter"):
        raise NonCacheableResponse(
            f"Agent-RT provider returned incomplete or failed finish reason: {response.finish_reason}"
        )
    if response.message.role != "assistant":
        raise NonCacheableResponse("Agent-RT provider returned a non-assistant message")
    if not response.message.tool_calls and not any(
        part.type == "text" and bool(part.text and part.text.strip())
        or part.type != "text" and part.data is not None
        for part in response.message.content
    ):
        raise ModelCapabilityIssue("empty_completion")


def require_successful_sdk_response(response: Any) -> None:
    """Validate the SDK's successful completion envelope before writing it."""
    if not isinstance(response, Mapping):
        raise NonCacheableResponse("OpenAI SDK returned an invalid response")
    if response.get("error") or response.get("status") in (
        "failed", "error", "cancelled", "incomplete",
    ):
        raise NonCacheableResponse("OpenAI SDK returned an error response")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NonCacheableResponse("OpenAI SDK completion contains no choices")
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise NonCacheableResponse("OpenAI SDK completion contains an invalid choice")
        if choice.get("finish_reason") not in ("stop", "tool_calls", "function_call"):
            raise NonCacheableResponse("OpenAI SDK completion failed, is incomplete or was truncated")
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("error"):
            raise NonCacheableResponse("OpenAI SDK completion contains an invalid message")
        if not message.get("content") and not message.get("tool_calls") and not message.get("refusal"):
            raise ModelCapabilityIssue("empty_completion")


def _old_first_trial_key(namespace: str, request: Any) -> Any | None:
    """Find the equivalent trial-0 request from older k>1 benchmark runs.

    Before first-slot normalization, k=1 used no suffix while k>1 used
    ':trial=0'. A legacy entry is safe to replay because all other model,
    prompt, grant and request fields remain in the cache key. This fallback
    is restricted to SwarmGuard's versioned model/Decision namespaces.
    """
    if not isinstance(request, Mapping):
        return None
    source = {
        "llm:openai-sdk": ("provider", "sdk-configured-v3:"),
        "llm:agent-rt": ("provider", "agent-rt-configured-v3:"),
        "decision": ("model", "swarmguard-decision-v3:"),
    }.get(namespace)
    if source is None:
        return None
    field, prefix = source
    identity = request.get(field)
    if (not isinstance(identity, str) or
            not identity.startswith(prefix) or ":trial=" in identity):
        return None
    return {**request, field: identity + ":trial=0"}


class ResponseCache:
    def __init__(self, path: str | Path, mode: str = "read-write") -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}")
        self.path = Path(path)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        if mode != "off":
            if mode == "read" and not self.path.exists():
                raise CacheMiss(f"read-only cache does not exist: {self.path}")
            if mode != "read":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(
                f"file:{self.path}?mode=ro" if mode == "read" else str(self.path),
                uri=(mode == "read"), check_same_thread=False,
            )
            if mode != "read":
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS responses "
                    "(namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                    "PRIMARY KEY(namespace, key))"
                )
                self._db.commit()

    def get(self, namespace: str, request: Any) -> Any | None:
        if self.mode == "off" or self.mode == "refresh":
            return None
        assert self._db is not None
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM responses WHERE namespace=? AND key=?",
                (namespace, cache_key(request)),
            ).fetchone()
            if row is None:
                old = _old_first_trial_key(namespace, request)
                if old is not None:
                    row = self._db.execute(
                        "SELECT value FROM responses WHERE namespace=? AND key=?",
                        (namespace, cache_key(old)),
                    ).fetchone()
            if row is not None:
                self.hits += 1
                return json.loads(row[0])
            self.misses += 1
        if self.mode == "read":
            raise CacheMiss(f"missing cached {namespace} response (key={cache_key(request)})")
        return None

    def put(self, namespace: str, request: Any, response: Any) -> None:
        if self.mode not in ("read-write", "refresh"):
            return
        assert self._db is not None
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses(namespace,key,value) VALUES(?,?,?)",
                (namespace, cache_key(request),
                 json.dumps(_jsonable(response), sort_keys=True, ensure_ascii=False)),
            )
            self._db.commit()
            self.writes += 1

    async def aget_or_call(
        self, namespace: str, request: Any, call: Callable[[], Awaitable[Any]],
        *, validate: Callable[[Any], None] | None = None,
    ) -> Any:
        found = self.get(namespace, request)
        if found is not None:
            if validate is not None:
                validate(found)
            return found
        # Network errors, HTTP failures, rate limits and timeouts propagate
        # without cache writes. Validate semantic failures before persistence.
        response = await call()
        if validate is not None:
            validate(response)
        self.put(namespace, request, response)
        return response

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class CachedDecisionProvider:
    def __init__(self, provider: Any, cache: ResponseCache, *, identity: str) -> None:
        self.provider = provider
        self.cache = cache
        self.identity = identity

    def decide(self, state: Any, questions: Any) -> Any:
        request = {"model": self.identity, "state": state, "questions": questions}
        cached = self.cache.get("decision", request)
        if cached is not None:
            require_successful_decision(cached, questions)
            return cached
        # A failed call or invalid decision answer is not a cacheable result.
        answers = self.provider.decide(state, questions)
        require_successful_decision(answers, questions)
        self.cache.put("decision", request, answers)
        return answers


class CachedModelProvider:
    """Caches Agent-RT ModelResponse, reconstructing typed messages on replay."""

    def __init__(
        self, provider: Any, cache: ResponseCache, *, identity: str,
        key_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.identity = identity
        self.key_transform = key_transform

    @property
    def name(self) -> str:
        return getattr(self.provider, "name", "cached-provider")

    async def complete(self, request: Any) -> Any:
        from ext.runtime.models.interfaces import ModelResponse, ModelUsage
        from ext.runtime.types import ContentPart, ModelMessage, ToolCall

        identity = {
            "provider": self.identity, "model": request.model,
            "messages": request.messages, "tools": request.tools,
            "temperature": request.temperature, "max_output_tokens": request.max_output_tokens,
            "structured_output": request.structured_output, "reasoning": request.reasoning,
            "tool_selection": request.tool_selection, "metadata": request.metadata,
            "prompt_cache": request.prompt_cache,
        }

        if self.key_transform is not None:
            identity = self.key_transform(identity)

        async def call() -> Any:
            response = await self.provider.complete(request)
            try:
                require_successful_model_response(response)
            except ModelCapabilityIssue as exc:
                # Successful HTTP / structured model response, but no usable
                # content or tool calls. Cache the *limitation*, not an answer.
                return _capability_record(exc.reason)
            return {
                "message": dataclasses.asdict(response.message),
                "model": response.model,
                "usage": dataclasses.asdict(response.usage) if response.usage else None,
                "finish_reason": response.finish_reason,
            }

        def validate(value: Any) -> None:
            if _capability_reason(value) is not None:
                return
            if not isinstance(value, Mapping) or not isinstance(value.get("message"), Mapping):
                raise NonCacheableResponse("Agent-RT cached response has an invalid envelope")
            if value.get("finish_reason") in ("error", "length", "content_filter"):
                raise NonCacheableResponse("Agent-RT cached response has a failure finish reason")
            message = value["message"]
            if message.get("role") != "assistant":
                raise NonCacheableResponse("Agent-RT cached response has invalid role")
            parts = message.get("content") or ()
            if not isinstance(parts, (tuple, list)):
                raise NonCacheableResponse("Agent-RT cached response has invalid content")
            usable = bool(message.get("tool_calls")) or any(
                isinstance(part, Mapping) and (
                    (part.get("type") == "text" and
                     isinstance(part.get("text"), str) and
                     bool(part["text"].strip()))
                    or (part.get("type") != "text" and part.get("data") is not None)
                )
                for part in parts
            )
            if not usable:
                # Older cache entries with empty/whitespace text parts are
                # also model limitations, not successful answers.
                raise ModelCapabilityIssue("empty_completion")

        result = await self.cache.aget_or_call(
            "llm:agent-rt", identity, call, validate=validate,
        )
        issue = _capability_reason(result)
        if issue is not None:
            raise ModelCapabilityIssue(issue)
        message = result["message"]
        return ModelResponse(
            message=ModelMessage(
                role=message["role"],
                content=tuple(ContentPart(**part) for part in message["content"]),
                tool_calls=tuple(ToolCall(**tool) for tool in message.get("tool_calls", ())),
                tool_call_id=message.get("tool_call_id"),
            ),
            model=result.get("model"),
            usage=ModelUsage(**result["usage"]) if result.get("usage") else None,
            finish_reason=result.get("finish_reason"),
        )


async def cached_sdk_completion(
    client: Any, cache: ResponseCache, *, model: str,
    messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    provider_identity: str = "openai-sdk",
) -> dict[str, Any]:
    """OpenAI SDK requests are cached independently from Agent-RT's model provider."""
    # Assistant responses may contain response-only fields (e.g. annotations).
    # Strip them before both cache keying and API submission, without mutating
    # the caller's conversation history.
    request_messages = [
        {key: value for key, value in message.items() if key != "annotations"}
        for message in messages
    ]
    request: dict[str, Any] = {
        "model": model, "messages": request_messages, "tools": tools,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if max_output_tokens is not None:
        request["max_completion_tokens"] = max_output_tokens

    async def call() -> dict[str, Any]:
        response = _jsonable(await client.chat.completions.create(**request))
        try:
            require_successful_sdk_response(response)
        except ModelCapabilityIssue as exc:
            return _capability_record(exc.reason)
        return response

    def validate(response: Any) -> None:
        if _capability_reason(response) is not None:
            return
        require_successful_sdk_response(response)

    result = await cache.aget_or_call(
        "llm:openai-sdk", {"provider": provider_identity, "request": request},
        call, validate=validate,
    )
    issue = _capability_reason(result)
    if issue is not None:
        raise ModelCapabilityIssue(issue)
    return result
