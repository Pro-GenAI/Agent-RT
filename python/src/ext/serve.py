"""``agent-rt serve telegram,discord``: chat with Agent RT from Telegram and Discord bots.

Each chat keeps its own in-memory conversation. Bots answer only the user ids listed in
``agent_rt_settings.serve.<platform>.allowed_users`` and have no workspace tools, because
anyone can message a public bot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from agent_rt import AgentLoop, configure_runtime, load_config, load_model
from agent_rt_cli import (
    _CLI_SETTINGS,
    ChatSession,
    _message_text,
    _startup_model,
    discover_builtin_models,
    with_builtin_models,
)
from ext.config import require_resolved

PLATFORMS = ("telegram", "discord")
TELEGRAM_API = "https://api.telegram.org"
TELEGRAM_LIMIT = 4096
DISCORD_API = "https://discord.com/api/v10"
DISCORD_GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
DISCORD_LIMIT = 2000
# GUILD_MESSAGES | DIRECT_MESSAGES. DMs and mentions carry their text without the
# privileged MESSAGE_CONTENT intent, and the bot answers nothing else.
DISCORD_INTENTS = (1 << 9) | (1 << 12)
# Gateway close codes that reconnecting cannot fix (bad token, intents, version, sharding).
DISCORD_FATAL_CLOSE_CODES = {4004, 4010, 4011, 4012, 4013, 4014}
POLL_SECONDS = 30
RETRY_SECONDS = 5.0
SERVE_INSTRUCTIONS = (
    "You are Agent RT, a concise and helpful assistant chatting with people on {platform}. "
    "You have no access to files, commands, or other tools. Keep replies short enough to "
    "read in a chat app."
)


class ServeError(RuntimeError):
    """A platform failure that retrying cannot fix."""


def parse_platforms(values: Sequence[str]) -> tuple[str, ...]:
    """``["telegram,discord"]`` (or space-separated names) -> unique platform names."""
    names = [
        name.strip().lower() for value in values for name in value.split(",") if name.strip()
    ]
    if not names:
        raise ValueError(
            "pass at least one platform to serve on: telegram, discord "
            "(for example: agent-rt serve telegram,discord)"
        )
    unknown = sorted(set(names) - set(PLATFORMS))
    if unknown:
        raise ValueError(
            f"unsupported platform: {', '.join(unknown)} (choose from {', '.join(PLATFORMS)})"
        )
    return tuple(dict.fromkeys(names))


def chunks(text: str, limit: int) -> list[str]:
    return [text[index : index + limit] for index in range(0, len(text), limit)]


class ChatBridge:
    """One platform's bot: a conversation per chat, answering only allowed users."""

    def __init__(
        self,
        platform: str,
        settings: Mapping[str, Any],
        make_session: Callable[[str], ChatSession],
    ) -> None:
        path = f"agent_rt_settings.serve.{platform}"
        token = settings.get("bot_token")
        require_resolved(token)
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"{path}.bot_token is required to serve on {platform}")
        allowed = settings.get("allowed_users") or []
        if not isinstance(allowed, list):
            raise TypeError(f"{path}.allowed_users must be a list of user ids")
        self.platform = platform
        self.token = token.strip()
        self.allowed = {str(user) for user in allowed}
        self.make_session = make_session
        self.sessions: dict[str, ChatSession] = {}
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.tasks: set[asyncio.Task[None]] = set()
        if not self.allowed:
            self.log(f"{path}.allowed_users is empty, so the bot answers nobody yet")

    def log(self, message: str) -> None:
        # httpx errors include request URLs, and Telegram puts the token in the URL.
        print(
            f"agent-rt serve: {self.platform}: {message.replace(self.token, '<bot_token>')}",
            file=sys.stderr,
            flush=True,
        )

    def spawn(self, handler: Any) -> None:
        task = asyncio.create_task(handler)
        self.tasks.add(task)  # keep a reference so the task is not garbage collected
        task.add_done_callback(self.tasks.discard)

    async def reply(self, chat_id: str, user_id: Any, text: str) -> str:
        if str(user_id) not in self.allowed:
            self.log(f"ignored user {user_id}: not in allowed_users")
            return (
                "You are not allowed to use this bot. Ask its owner to add your user id "
                f"{user_id} to agent_rt_settings.serve.{self.platform}.allowed_users."
            )
        if text.strip() == "/new":
            self.sessions.pop(chat_id, None)
            return "Started a new conversation."
        async with self.locks[chat_id]:
            session = self.sessions.get(chat_id)
            if session is None:
                session = self.sessions[chat_id] = self.make_session(self.platform)
            try:
                result = await session.send(text, stream=False)
            except Exception as exc:  # noqa: BLE001 - one failed run must not stop the bot
                self.log(f"run failed: {exc}")
                return "Sorry, that request failed. Please try again."
        response = result.final_response
        reply = _message_text(response.message).strip() if response else ""
        return reply or f"(no reply: {result.termination_reason})"


async def run_telegram(bridge: ChatBridge, client: httpx.AsyncClient) -> None:
    base = f"{TELEGRAM_API}/bot{bridge.token}"
    try:
        me = await client.get(f"{base}/getMe")
    except httpx.HTTPError as exc:
        raise ServeError(f"telegram: {exc}".replace(bridge.token, "<bot_token>")) from None
    if me.status_code in (401, 404):
        raise ServeError("telegram: bot token rejected")
    if me.status_code != 200:
        raise ServeError(f"telegram: getMe returned HTTP {me.status_code}")
    print(f"Serving on Telegram as @{me.json()['result'].get('username')}", flush=True)

    async def handle(chat_id: Any, user_id: Any, text: str) -> None:
        reply = await bridge.reply(str(chat_id), user_id, text)
        for part in chunks(reply, TELEGRAM_LIMIT):
            try:
                sent = await client.post(
                    f"{base}/sendMessage", json={"chat_id": chat_id, "text": part}
                )
                sent.raise_for_status()
            except httpx.HTTPError as exc:
                bridge.log(f"sendMessage failed: {exc}")
                return

    offset = 0
    while True:
        try:
            polled = await client.get(
                f"{base}/getUpdates",
                params={"offset": offset, "timeout": POLL_SECONDS, "allowed_updates": '["message"]'},
                timeout=POLL_SECONDS + 10,
            )
            polled.raise_for_status()
            updates = polled.json()["result"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            bridge.log(f"getUpdates failed: {exc}")
            await asyncio.sleep(RETRY_SECONDS)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or {}
            text = message.get("text")
            user_id = (message.get("from") or {}).get("id")
            chat_id = (message.get("chat") or {}).get("id")
            if text and text.strip() and user_id is not None and chat_id is not None:
                bridge.spawn(handle(chat_id, user_id, text))


async def run_discord(
    bridge: ChatBridge,
    client: httpx.AsyncClient,
    connect: Callable[[str], Any] | None = None,
) -> None:
    from websockets.asyncio.client import connect as websockets_connect
    from websockets.exceptions import ConnectionClosed

    open_gateway = connect or websockets_connect
    while True:
        try:
            async with open_gateway(DISCORD_GATEWAY) as gateway:
                await _discord_session(bridge, client, gateway)
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd else None
            reason = exc.rcvd.reason if exc.rcvd else ""
            if code in DISCORD_FATAL_CLOSE_CODES:
                raise ServeError(f"discord: gateway closed with code {code} ({reason})") from None
            bridge.log(f"gateway closed ({code}), reconnecting")
        except OSError as exc:
            bridge.log(f"gateway connection failed: {exc}")
        await asyncio.sleep(RETRY_SECONDS)


async def _discord_session(bridge: ChatBridge, client: httpx.AsyncClient, gateway: Any) -> None:
    """Identify and handle events until the gateway asks for a reconnect."""
    hello = json.loads(await gateway.recv())
    interval = hello["d"]["heartbeat_interval"] / 1000
    state: dict[str, Any] = {"seq": None, "bot_id": None}
    headers = {"Authorization": f"Bot {bridge.token}"}

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(interval)
            await gateway.send(json.dumps({"op": 1, "d": state["seq"]}))

    async def handle(channel_id: str, user_id: Any, text: str) -> None:
        reply = await bridge.reply(channel_id, user_id, text)
        for part in chunks(reply, DISCORD_LIMIT):
            try:
                sent = await client.post(
                    f"{DISCORD_API}/channels/{channel_id}/messages",
                    headers=headers,
                    # Model text must never ping @everyone, roles, or users.
                    json={"content": part, "allowed_mentions": {"parse": []}},
                )
                sent.raise_for_status()
            except httpx.HTTPError as exc:  # shortcut: no 429 retry, add one if replies get rate limited
                bridge.log(f"sending a message failed: {exc}")
                return

    await gateway.send(
        json.dumps(
            {
                "op": 2,
                "d": {
                    "token": bridge.token,
                    "intents": DISCORD_INTENTS,
                    "properties": {"os": sys.platform, "browser": "agent-rt", "device": "agent-rt"},
                },
            }
        )
    )
    beat = asyncio.create_task(heartbeat())
    try:
        async for raw in gateway:
            event = json.loads(raw)
            state["seq"] = event.get("s") or state["seq"]
            op, kind, data = event.get("op"), event.get("t"), event.get("d") or {}
            if op == 1:
                await gateway.send(json.dumps({"op": 1, "d": state["seq"]}))
            elif op in (7, 9):
                # shortcut: identifies afresh instead of resuming, so messages sent during the gap are missed
                return
            elif kind == "READY":
                state["bot_id"] = data["user"]["id"]
                print(f"Serving on Discord as {data['user'].get('username')}", flush=True)
            elif kind == "MESSAGE_CREATE" and state["bot_id"]:
                author = data.get("author") or {}
                text = data.get("content") or ""
                if author.get("bot"):
                    continue
                if data.get("guild_id"):  # in servers, answer only when mentioned
                    if not any(m.get("id") == state["bot_id"] for m in data.get("mentions") or []):
                        continue
                    text = re.sub(rf"<@!?{state['bot_id']}>", "", text)
                if text.strip():
                    bridge.spawn(handle(data["channel_id"], author.get("id"), text.strip()))
    finally:
        beat.cancel()


RUNNERS = {"telegram": run_telegram, "discord": run_discord}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-rt serve",
        description="Serve Agent RT as a chat bot on Telegram and/or Discord.",
    )
    parser.add_argument(
        "platforms", nargs="*", help="comma-separated platforms: telegram, discord"
    )
    parser.add_argument("-m", "--model", help="provider model name")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to the YAML config file (default: agent-rt.yaml, then ~/.agent-rt/config.yaml)",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        platforms = parse_platforms(args.platforms)
    except ValueError as exc:
        print(f"agent-rt serve: {exc}", file=sys.stderr)
        return 2
    try:
        config, builtin = with_builtin_models(load_config(args.config))
        configure_runtime(config)
        _CLI_SETTINGS.clear()
        _CLI_SETTINGS.update(config.settings.get("cli") or {})
        provider = load_model(config)
        if builtin and not (args.model or "").strip():
            model = (await discover_builtin_models(provider))[0]
        else:
            model = await _startup_model(provider, args.model)
    except Exception as exc:  # noqa: BLE001 - report any config or provider error as one line
        print(f"agent-rt serve: configuration error: {exc}", file=sys.stderr)
        return 2

    def make_session(platform: str) -> ChatSession:
        return ChatSession(
            AgentLoop(provider),
            model=model,
            instructions=SERVE_INSTRUCTIONS.format(platform=platform.title()),
            autosave=False,
        )

    serve = config.settings.get("serve") or {}
    try:
        bridges = [ChatBridge(name, serve.get(name) or {}, make_session) for name in platforms]
    except Exception as exc:  # noqa: BLE001 - missing tokens and bad lists are config errors
        print(f"agent-rt serve: configuration error: {exc}", file=sys.stderr)
        return 2

    print(
        "Note: agent-rt serve is a beta feature; its behaviour and config may change.\n"
        f"Agent RT bot using model {model} (Ctrl+C to stop)",
        flush=True,
    )
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [asyncio.create_task(RUNNERS[b.platform](b, client)) for b in bridges]
        # Runners loop forever, so the first one to finish has failed.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    error = str(next(iter(done)).exception())
    for bridge in bridges:
        error = error.replace(bridge.token, "<bot_token>")
    print(f"agent-rt serve: {error}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130
