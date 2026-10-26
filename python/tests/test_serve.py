import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from agent_rt import ContentPart, ModelMessage
from agent_rt_cli import main
from ext import serve
from ext.config import ConfigError, parse_config

TOKEN = "123:secret-token"


class EchoSession:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str, *, stream: bool = True) -> SimpleNamespace:
        self.sent.append(text)
        message = ModelMessage(
            role="assistant", content=(ContentPart(type="text", text=f"echo {len(self.sent)}: {text}"),)
        )
        return SimpleNamespace(
            final_response=SimpleNamespace(message=message), termination_reason="completed"
        )


def bridge(platform: str = "telegram", allowed=(7,)) -> serve.ChatBridge:
    return serve.ChatBridge(
        platform, {"bot_token": TOKEN, "allowed_users": list(allowed)}, lambda _: EchoSession()
    )


def test_parse_platforms():
    assert serve.parse_platforms(["telegram,discord"]) == ("telegram", "discord")
    assert serve.parse_platforms(["Discord", "discord,telegram"]) == ("discord", "telegram")
    with pytest.raises(ValueError, match="at least one platform"):
        serve.parse_platforms([" , "])
    with pytest.raises(ValueError, match="unsupported platform: slack"):
        serve.parse_platforms(["telegram,slack"])


def test_serve_fails_without_platforms(capsys):
    assert main(["serve"]) == 2
    assert main(["serve", "irc"]) == 2
    err = capsys.readouterr().err
    assert "at least one platform" in err and "unsupported platform: irc" in err


def test_serve_config_schema_and_deferred_tokens():
    config = parse_config(
        "agent_rt_settings:\n  serve:\n"
        "    telegram: {bot_token: os.environ/NOPE, allowed_users: [7]}\n",
        environ={},
    )
    with pytest.raises(ConfigError, match="NOPE"):  # unset token fails only when served
        bridge_settings = config.settings["serve"]["telegram"]
        serve.ChatBridge("telegram", bridge_settings, lambda _: EchoSession())
    with pytest.raises(ConfigError, match="serve"):
        parse_config("agent_rt_settings:\n  serve: {slack: {bot_token: x}}\n", environ={})
    with pytest.raises(ConfigError, match="serve.discord"):
        parse_config("agent_rt_settings:\n  serve: {discord: {token: x}}\n", environ={})
    with pytest.raises(ValueError, match="bot_token is required"):
        serve.ChatBridge("discord", {}, lambda _: EchoSession())


async def test_bridge_keeps_one_conversation_per_chat_and_checks_users():
    bot = bridge()
    assert await bot.reply("c1", 7, "hi") == "echo 1: hi"
    assert await bot.reply("c1", 7, "again") == "echo 2: again"
    assert await bot.reply("c2", 7, "other") == "echo 1: other"
    assert "user id 8" in await bot.reply("c1", 8, "let me in")
    assert await bot.reply("c1", 7, "/new") == "Started a new conversation."
    assert await bot.reply("c1", 7, "fresh") == "echo 1: fresh"


def test_chunks():
    assert serve.chunks("abcde", 2) == ["ab", "cd", "e"]


async def test_telegram_polls_and_replies(capsys):
    bot = bridge()
    sent: list[dict] = []
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "artie_bot"}})
        if method == "sendMessage":
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})
        polls += 1
        if polls > 1:
            raise httpx.ConnectError(f"boom {request.url}")
        message = {"text": "hello", "from": {"id": 7}, "chat": {"id": 42}}
        return httpx.Response(200, json={"ok": True, "result": [{"update_id": 5, "message": message}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(serve.run_telegram(bot, client))
        while not sent:
            await asyncio.sleep(0.01)
        task.cancel()
    assert sent == [{"chat_id": 42, "text": "echo 1: hello"}]
    out = capsys.readouterr()
    assert "@artie_bot" in out.out
    assert TOKEN not in out.err  # the token is in every Telegram URL


async def test_telegram_rejected_token():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401, json={"ok": False}))
    ) as client:
        with pytest.raises(serve.ServeError, match="token rejected"):
            await serve.run_telegram(bridge(), client)


class FakeGateway:
    def __init__(self, events: list[dict], close_code: int | None = None) -> None:
        self.events = events
        self.close_code = close_code
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def recv(self) -> str:
        return json.dumps({"op": 10, "d": {"heartbeat_interval": 60_000}})

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield json.dumps(event)
        if self.close_code:
            raise ConnectionClosedError(Close(self.close_code, "Authentication failed."), None)
        await asyncio.Event().wait()


async def test_discord_answers_dms_and_mentions_only():
    bot = bridge("discord", allowed=("7",))
    posts: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bot {TOKEN}"
        posts.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})

    def message(content, **extra):
        data = {"content": content, "channel_id": "c1", "author": {"id": "7"}, **extra}
        return {"op": 0, "t": "MESSAGE_CREATE", "s": 2, "d": data}

    gateway = FakeGateway(
        [
            {"op": 0, "t": "READY", "s": 1, "d": {"user": {"id": "99", "username": "artie"}}},
            message("dm hello"),
            message("not for me", guild_id="g"),
            message("<@99> ping", guild_id="g", mentions=[{"id": "99"}]),
            message("bot talk", author={"id": "7", "bot": True}),
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(serve.run_discord(bot, client, connect=lambda _: gateway))
        while len(posts) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
    assert gateway.sent[0]["op"] == 2 and gateway.sent[0]["d"]["intents"] == serve.DISCORD_INTENTS
    contents = sorted(body["content"] for _, body in posts)
    assert contents == ["echo 1: dm hello", "echo 2: ping"]
    assert all(body["allowed_mentions"] == {"parse": []} for _, body in posts)


async def test_discord_fatal_close_stops():
    async with httpx.AsyncClient() as client:
        with pytest.raises(serve.ServeError, match="4004"):
            await serve.run_discord(
                bridge("discord"), client, connect=lambda _: FakeGateway([], close_code=4004)
            )
