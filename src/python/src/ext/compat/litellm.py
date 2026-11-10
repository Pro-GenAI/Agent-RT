"""``litellm`` completion/embedding functions over Agent RT providers.

``completion`` / ``acompletion`` return OpenAI-shaped responses that also
support LiteLLM's dict-style access (``response["choices"][0]["message"]``).
Models route by LiteLLM's provider prefix: ``openai/`` (or a bare OpenAI
model, plus any OpenAI-compatible server through ``api_base``),
``anthropic/`` (or a bare ``claude-*`` model), and ``openrouter/``. Other
LiteLLM providers raise instead of silently calling the wrong endpoint.
Exceptions are the ``agent_rt.openai`` classes, which LiteLLM's exceptions
subclass upstream.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from ext.compat.base import OpenAI, _anthropic_provider, _openai_provider
from ext.compat.openrouter import _chat_kwargs, openrouter_provider
from ext.compat.sdk_errors import OPENAI_ERRORS, public_error_classes

_PROVIDERS = ("openai", "anthropic", "openrouter")


def _client(model, kwargs):
    """Pop LiteLLM routing options from ``kwargs``; return (client, model)."""
    provider = kwargs.pop("provider", None)
    api_key = kwargs.pop("api_" + "key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    route = kwargs.pop("custom_llm_provider", None)
    kwargs.pop(
        "drop_params", None
    )  # completion() applies it; embeddings have no allowlist
    # Agent RT providers make one attempt with no client deadline, so only
    # values that keep that behavior pass; others raise instead of being ignored.
    retries = kwargs.pop("num_retries", None)
    unsupported = sorted(
        [o for o in ("timeout", "request_timeout") if kwargs.pop(o, None) is not None]
        + (["num_retries"] if retries else [])
    )
    if unsupported:
        raise NotImplementedError(
            "litellm compatibility does not support request options: "
            + ", ".join(unsupported)
        )
    prefix, sep, rest = str(model).partition("/")
    if route is None and sep and prefix in _PROVIDERS:
        route, model = prefix, rest
    elif route is None and sep:
        raise ValueError(
            f"litellm compatibility does not support provider {prefix!r}; use "
            f"{', '.join(p + '/' for p in _PROVIDERS)} or openai/<model> with api_base "
            "for an OpenAI-compatible server"
        )
    route = route or ("anthropic" if str(model).startswith("claude") else "openai")
    if provider is None and route == "anthropic":
        provider = _anthropic_provider(
            model=model, credential=api_key, base_url=api_base
        )
    elif provider is None and route == "openrouter":
        provider = openrouter_provider(api_key=api_key, base_url=api_base)
    elif provider is None and route == "openai":
        provider = _openai_provider(model=model, credential=api_key, base_url=api_base)
    elif provider is None:
        raise ValueError(f"litellm compatibility does not support provider {route!r}")
    return OpenAI(provider=provider), model


def _chat(model, messages, kwargs):
    drop = bool(kwargs.pop("drop_params", None))
    client, model = _client(model, kwargs)
    request = {"model": model, "messages": messages or [], **kwargs}
    return client, _chat_kwargs(request, "litellm completion", drop=drop)


async def acompletion(model, messages=None, **kwargs):
    client, request = _chat(model, messages, kwargs)
    return await client._async.chat.completions.create(**request)


def completion(model, messages=None, **kwargs):
    client, request = _chat(model, messages, kwargs)
    return client.chat.completions.create(**request)


async def aembedding(model, input=None, **kwargs):
    client, model = _client(model, kwargs)
    return await client._async.embeddings.create(model=model, input=input, **kwargs)


def embedding(model, input=None, **kwargs):
    client, model = _client(model, kwargs)
    return client.embeddings.create(model=model, input=input, **kwargs)


def install_litellm_compat(parent: Any) -> None:
    module: Any = types.ModuleType(f"{parent.__name__}.litellm")
    errors = public_error_classes(OPENAI_ERRORS)
    exports = {
        "acompletion": acompletion,
        "aembedding": aembedding,
        "completion": completion,
        "embedding": embedding,
        **errors,
        "Timeout": errors["APITimeoutError"],
    }
    exceptions = types.ModuleType(f"{module.__name__}.exceptions")
    exceptions.__dict__.update({k: v for k, v in exports.items() if k[0].isupper()})
    module.__dict__.update(exports, exceptions=exceptions)
    module.__all__ = tuple(sorted(exports))
    module.__path__ = []
    sys.modules[module.__name__] = module
    sys.modules[exceptions.__name__] = exceptions
    parent.litellm = module


__all__ = (
    "acompletion",
    "aembedding",
    "completion",
    "embedding",
    "install_litellm_compat",
)
