"""``openrouter.OpenRouter`` (the official Python SDK) over the Agent RT OpenAI shim.

``chat.send`` / ``chat.send_async`` take the SDK's snake_case keyword
arguments and return OpenAI-shaped chat completions (``choices[0].message``),
which is also what the SDK returns. Requests go to the OpenRouter
OpenAI-compatible endpoint through an Agent RT provider; the key comes from
``api_key=`` or ``OPENROUTER_API_KEY`` and never falls back to OpenAI's.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

from ext.compat.base import OpenAI, _rt

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"  # pragma: allowlist secret


_CHAT_OPTIONS = frozenset({
    "model", "messages", "temperature", "max_tokens", "max_completion_tokens",
    "tools", "response_format", "reasoning_effort", "stream",
})


def _chat_kwargs(kwargs, owner="OpenRouter chat.send", drop=False):
    unsupported = sorted(set(kwargs) - _CHAT_OPTIONS)
    if unsupported and drop:
        # LiteLLM's drop_params=True asks for unsupported options to be dropped.
        return {k: v for k, v in kwargs.items() if k in _CHAT_OPTIONS}
    if unsupported:
        raise NotImplementedError(
            f"{owner} cannot honor request options: " + ", ".join(unsupported)
        )
    return kwargs


def openrouter_provider(*, api_key=None, base_url=None):
    """An Agent RT provider for OpenRouter's OpenAI-compatible endpoint."""
    rt = _rt()
    key = api_key() if callable(api_key) else api_key
    settings = {
        "base_url": base_url or OPENROUTER_BASE_URL,
        "api_" + "key": key
        or os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
        or None,
        "websocket": False,
    }
    return rt.OpenAIModelProvider(rt.OpenAIProviderSettings(**settings))


class _Chat:
    def __init__(self, client):
        self._client = client

    async def send_async(self, **kwargs):
        return await self._client._async.chat.completions.create(**_chat_kwargs(kwargs))

    def send(self, **kwargs):
        return self._client.chat.completions.create(**_chat_kwargs(kwargs))


class _Embeddings:
    def __init__(self, client):
        self._client = client

    async def generate_async(self, **kwargs):
        return await self._client._async.embeddings.create(**kwargs)

    def generate(self, **kwargs):
        return self._client.embeddings.create(**kwargs)


class OpenRouter:
    def __init__(self, api_key=None, *, server_url=None, provider=None, **_):
        # App attribution headers, retry config, and custom HTTP clients are
        # SDK transport options; Agent RT's provider owns the transport.
        self.provider = provider or openrouter_provider(
            api_key=api_key, base_url=server_url
        )
        client = OpenAI(provider=self.provider)
        self.chat = _Chat(client)
        self.embeddings = _Embeddings(client)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


def install_openrouter_compat(parent: Any) -> None:
    module = types.ModuleType(f"{parent.__name__}.openrouter")
    exports = {"OpenRouter": OpenRouter}
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    sys.modules[module.__name__] = module
    parent.openrouter = module


__all__ = ("OpenRouter", "install_openrouter_compat", "openrouter_provider")
