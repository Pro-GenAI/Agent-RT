"""Run the Agent RT CLI with another agent CLI's commands, settings, and credentials.

``agent-rt claude -p "hi"``, ``agent-rt codex exec "hi"``, ``agent-rt opencode run "hi"``
and ``agent-rt copilot -p "hi"`` accept that tool's own flags, read the environment
variables and config files that tool documents for models, endpoints, and API keys,
and turn them into Agent RT CLI arguments plus an in-memory config. Like the SDK
migration adapters, these modes deliberately mirror the upstream tool's environment.

Only API-key credentials are reused. Subscription/OAuth logins (Claude.ai, ChatGPT,
GitHub Copilot, opencode OAuth) belong to the official clients and are refused.
Project-level files (``.claude/settings*.json``, ``./opencode.json``) may only choose
the model: endpoints and keys come from user-level config or the environment, so a
cloned repository cannot redirect a real API key to its own server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ext.config import AgentRTConfig

# Claude Code model aliases; ANTHROPIC_DEFAULT_<ALIAS>_MODEL overrides each target.
CLAUDE_MODEL_ALIASES = {
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "haiku": "claude-haiku-5-5",
    "fable": "claude-fable-5-1",
}
DEFAULT_OPENAI_MODEL = "gpt-5.5"
CODEX_LOCAL_PROVIDERS = {"ollama": "http://localhost:11434/v1", "lmstudio": "http://localhost:1234/v1"}
_TRUE = {"1", "true", "yes", "on"}


class ToolCLIError(ValueError):
    """The tool invocation cannot be translated; the message says what to change."""


@dataclass
class ToolInvocation:
    config: AgentRTConfig
    argv: list[str]
    cwd: Path | None = None
    notes: list[str] = field(default_factory=list)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # argparse would print usage and exit
        raise ToolCLIError(f"{self.prog}: {message}")


def _parser(prog: str) -> _Parser:
    return _Parser(prog=prog, add_help=False, allow_abbrev=False)


def _parse(parser: _Parser, argv: Sequence[str]) -> argparse.Namespace:
    args, unknown = parser.parse_known_args(list(argv))
    if unknown:
        raise ToolCLIError(f"{parser.prog}: unsupported option(s) for Agent RT: {' '.join(unknown)}")
    return args


def _config(provider: str, model: str, api_key: str | None, api_base: str | None) -> AgentRTConfig:
    params: dict[str, Any] = {"model": f"{provider}/{model}"}
    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    return AgentRTConfig.from_mapping(
        {
            "model_list": [{"model_name": model, "agent_rt_params": params}],
            "agent_rt_settings": {"default_model": model},
        },
        environ={},
    )


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        merged[key] = _merge(current, value) if isinstance(current, Mapping) and isinstance(value, Mapping) else value
    return merged


def _strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments and trailing commas outside strings (opencode accepts JSONC)."""
    out: list[str] = []
    i, n, in_string = 0, len(text), False
    while i < n:
        char = text[i]
        if in_string:
            if char == "\\":
                out.append(text[i : i + 2])
                i += 2
                continue
            in_string = char != '"'
            out.append(char)
            i += 1
            continue
        if char == '"':
            in_string = True
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        elif char == ",":
            rest = text[i + 1 :].lstrip()
            if rest[:1] in {"}", "]"}:
                i += 1
                continue
        out.append(char)
        i += 1
    return "".join(out)


def _parse_json(text: str, origin: str) -> dict[str, Any]:
    try:
        value = json.loads(_strip_jsonc(text))
    except json.JSONDecodeError as exc:
        raise ToolCLIError(f"unable to parse {origin}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _parse_json(path.read_text(encoding="utf-8-sig"), str(path))
    except OSError as exc:
        raise ToolCLIError(f"unable to read {path}: {exc}") from exc


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if sys.version_info < (3, 11):
        raise ToolCLIError(f"reading {path} requires Python 3.11+ (tomllib)")
    import tomllib

    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ToolCLIError(f"unable to read {path}: {exc}") from exc


def _max_tokens_args(value: str | None, name: str) -> list[str]:
    if not value:
        return []
    if not value.strip().isdigit() or int(value) <= 0:
        raise ToolCLIError(f"{name} must be a positive integer")
    return ["--max-output-tokens", value.strip()]


def _common_args(prompt: str | None, session: str | None = None, resume: bool = False) -> list[str]:
    argv = ["-p", prompt] if prompt is not None else []
    if session:
        argv += ["--session", session] + (["--resume"] if resume else [])
    return argv


def _permission_note(tool: str, flags: Sequence[str]) -> list[str]:
    if not flags:
        return []
    return [f"{tool}: ignoring {', '.join(flags)}; Agent RT keeps its own workspace approval prompts"]


def _ignored_note(tool: str, names: Sequence[str]) -> list[str]:
    return [f"{tool}: {', '.join(names)} not supported by Agent RT; ignored"] if names else []


# ------------------------------------------------------------------ claude


def _claude_model(name: str, env: Mapping[str, str]) -> str:
    alias = re.sub(r"\[.*\]$", "", name.strip())  # e.g. "sonnet[1m]"
    alias = {"default": "sonnet", "opusplan": "opus"}.get(alias, alias)
    if alias in CLAUDE_MODEL_ALIASES:
        return env.get(f"ANTHROPIC_DEFAULT_{alias.upper()}_MODEL") or CLAUDE_MODEL_ALIASES[alias]
    return alias


def _claude(argv: Sequence[str], environ: Mapping[str, str], home: Path, cwd: Path) -> ToolInvocation:
    parser = _parser("claude")
    parser.add_argument("-p", "--print", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--system-prompt")
    parser.add_argument("--append-system-prompt")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("-r", "--resume")
    parser.add_argument("--session-id")
    parser.add_argument("--output-format", choices=["text"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    parser.add_argument("--permission-mode")
    parser.add_argument("prompt", nargs="?")
    args = _parse(parser, argv)
    if args.prompt is not None and not args.print:
        raise ToolCLIError("claude: an initial interactive prompt is not supported; use -p")

    # User settings may set `env` and `model`; project settings may only choose the model.
    config_dir = Path(environ.get("CLAUDE_CONFIG_DIR") or home / ".claude")
    user = _read_json(config_dir / "settings.json")
    project = _merge(_read_json(cwd / ".claude" / "settings.json"), _read_json(cwd / ".claude" / "settings.local.json"))
    env = {**{k: str(v) for k, v in (user.get("env") or {}).items()}, **environ}

    if env.get("CLAUDE_CODE_USE_BEDROCK", "").lower() in _TRUE or env.get("CLAUDE_CODE_USE_VERTEX", "").lower() in _TRUE \
            or env.get("CLAUDE_CODE_USE_FOUNDRY", "").lower() in _TRUE:
        raise ToolCLIError("claude: Bedrock, Vertex, and Foundry are not supported; use ANTHROPIC_API_KEY")
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key:
        if env.get("ANTHROPIC_AUTH_TOKEN"):
            hint = " (ANTHROPIC_AUTH_TOKEN bearer tokens are not supported)"
        elif env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            hint = " (CLAUDE_CODE_OAUTH_TOKEN is a Claude.ai subscription token)"
        else:
            hint = ""
        raise ToolCLIError(
            "claude: set ANTHROPIC_API_KEY (environment or Claude Code settings `env`); "
            f"Claude.ai subscription logins cannot be reused{hint}"
        )
    model = _claude_model(
        args.model
        or env.get("ANTHROPIC_MODEL")
        or project.get("model")
        or user.get("model")
        or env.get("ANTHROPIC_DEFAULT_MODEL")
        or "sonnet",
        env,
    )

    cli = _common_args(
        (args.prompt or "") if args.print else None,
        args.resume or args.session_id,
        resume=bool(args.resume),
    )
    if args.system_prompt or args.append_system_prompt:
        from agent_rt_cli import DEFAULT_INSTRUCTIONS

        base = args.system_prompt or DEFAULT_INSTRUCTIONS
        cli += ["--instructions", "\n\n".join(filter(None, [base, args.append_system_prompt]))]
    if args.max_turns:
        cli += ["--max-turns", str(args.max_turns)]
    cli += _max_tokens_args(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    flags = ["--dangerously-skip-permissions"] * args.dangerously_skip_permissions
    flags += ["--permission-mode"] * bool(args.permission_mode)
    ignored = [name for name in ("ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_BETAS") if env.get(name)]
    return ToolInvocation(
        _config("anthropic", model, api_key, env.get("ANTHROPIC_BASE_URL")),
        cli,
        notes=_permission_note("claude", flags) + _ignored_note("claude", ignored),
    )


# ------------------------------------------------------------------- codex


def _codex_override(config: dict[str, Any], assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep or not key.strip():
        raise ToolCLIError(f"codex: -c expects key=value, got {assignment!r}")
    value: Any = raw
    if sys.version_info >= (3, 11):
        import tomllib

        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            pass  # codex treats unparsable values as literal strings
    target = config
    *parents, leaf = key.strip().split(".")
    for part in parents:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ToolCLIError(f"codex: -c {key} conflicts with a non-table value")
    target[leaf] = value


def _codex(argv: Sequence[str], environ: Mapping[str, str], home: Path, cwd: Path) -> ToolInvocation:
    one_shot = bool(argv) and argv[0] == "exec"
    parser = _parser("codex exec" if one_shot else "codex")
    parser.add_argument("-m", "--model")
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-p", "--profile")
    parser.add_argument("--oss", action="store_true")
    parser.add_argument("--local-provider", choices=sorted(CODEX_LOCAL_PROVIDERS))
    parser.add_argument("-C", "--cd")
    parser.add_argument("-s", "--sandbox")
    parser.add_argument("-a", "--ask-for-approval")
    parser.add_argument("--full-auto", action="store_true")
    parser.add_argument("--dangerously-bypass-approvals-and-sandbox", action="store_true")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--ephemeral", action="store_true")
    parser.add_argument("prompt", nargs="?")
    args = _parse(parser, argv[1:] if one_shot else argv)
    if args.prompt is not None and not one_shot:
        raise ToolCLIError("codex: an initial interactive prompt is not supported; use `codex exec`")

    # Project-local .codex/config.toml is not read: Codex only honors it for trusted projects.
    codex_home = Path(environ.get("CODEX_HOME") or home / ".codex")
    config = _read_toml(codex_home / "config.toml")
    profile_name = args.profile or config.get("profile")
    if profile_name:
        profile_file = codex_home / f"{profile_name}.config.toml"
        profile = _read_toml(profile_file) if profile_file.is_file() else (config.get("profiles") or {}).get(profile_name)
        if not isinstance(profile, dict):
            raise ToolCLIError(f"codex: profile {profile_name!r} not found ({profile_file})")
        config = _merge(config, profile)
    for assignment in args.config:
        _codex_override(config, assignment)

    model = args.model or config.get("model") or DEFAULT_OPENAI_MODEL
    provider_id = (args.local_provider or config.get("oss_provider") or "ollama") if args.oss else (
        config.get("model_provider") or "openai"
    )
    notes: list[str] = []

    def openai_key() -> str:
        auth = _read_json(codex_home / "auth.json")
        key = environ.get("CODEX_API_KEY") or environ.get("OPENAI_API_KEY") or auth.get("OPENAI_API_KEY")
        if not key:
            token = " (CODEX_ACCESS_TOKEN is a ChatGPT token)" if environ.get("CODEX_ACCESS_TOKEN") else ""
            raise ToolCLIError(
                "codex: set CODEX_API_KEY or log in with `codex login --with-api-key`; "
                f"ChatGPT sign-in tokens cannot be reused{token}"
            )
        return key

    if provider_id == "openai":
        api_key: str | None = openai_key()
        api_base = config.get("openai_base_url") or environ.get("OPENAI_BASE_URL")
    elif provider_id in CODEX_LOCAL_PROVIDERS:
        api_key, api_base = None, CODEX_LOCAL_PROVIDERS[provider_id]
    else:
        entry = (config.get("model_providers") or {}).get(provider_id)
        if not isinstance(entry, dict) or not entry.get("base_url"):
            raise ToolCLIError(f"codex: model provider {provider_id!r} needs a base_url in config.toml")
        env_key = entry.get("env_key")
        if env_key:
            api_key = environ.get(env_key)
            if not api_key:
                raise ToolCLIError(f"codex: model provider {provider_id!r} needs {env_key} to be set")
        elif entry.get("requires_openai_auth"):
            api_key = openai_key()
        else:
            api_key = entry.get("experimental_bearer_token")
        api_base = entry["base_url"]
        notes += _ignored_note(
            "codex", [f"model_providers.{provider_id}.{k}" for k in ("http_headers", "env_http_headers", "query_params") if entry.get(k)]
        )

    cli = _common_args((args.prompt or "") if one_shot else None)
    if args.ephemeral:
        cli.append("--no-save")
    flags = [
        name
        for name, on in (
            ("--full-auto", args.full_auto),
            ("--dangerously-bypass-approvals-and-sandbox", args.dangerously_bypass_approvals_and_sandbox),
            ("--sandbox", bool(args.sandbox)),
            ("--ask-for-approval", bool(args.ask_for_approval)),
        )
        if on
    ]
    return ToolInvocation(
        _config("openai", model, api_key, api_base),  # Codex providers are OpenAI-compatible
        cli,
        cwd=(cwd / args.cd) if args.cd else None,
        notes=_permission_note("codex", flags) + notes,
    )


# ---------------------------------------------------------------- opencode


def _substitute(value: Any, environ: Mapping[str, str], base: Path) -> Any:
    """opencode `{env:NAME}` and `{file:path}` substitution."""
    if isinstance(value, Mapping):
        return {k: _substitute(v, environ, base) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, environ, base) for v in value]
    if not isinstance(value, str):
        return value

    def read(match: re.Match[str]) -> str:
        path = Path(match.group(1)).expanduser()
        try:
            return (path if path.is_absolute() else base / path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ToolCLIError(f"opencode: unable to read {{file:{match.group(1)}}}: {exc}") from exc

    value = re.sub(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: environ.get(m.group(1), ""), value)
    return re.sub(r"\{file:([^}]+)\}", read, value)


def _opencode_file(directory: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    for name in ("opencode.json", "opencode.jsonc"):
        if (directory / name).is_file():
            return _substitute(_read_json(directory / name), environ, directory)
    return {}


def _opencode(argv: Sequence[str], environ: Mapping[str, str], home: Path, cwd: Path) -> ToolInvocation:
    one_shot = bool(argv) and argv[0] == "run"
    parser = _parser("opencode run" if one_shot else "opencode")
    parser.add_argument("-m", "--model")
    parser.add_argument("-s", "--session")
    parser.add_argument("--dir")
    parser.add_argument("--format", choices=["default"])
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("message", nargs="*")
    args = _parse(parser, argv[1:] if one_shot else argv)
    if args.message and not one_shot:
        raise ToolCLIError("opencode: a project argument is not supported; use `opencode run`")
    workdir = (cwd / args.dir) if args.dir else cwd

    # Documented order: global < OPENCODE_CONFIG < project < OPENCODE_CONFIG_DIR < OPENCODE_CONFIG_CONTENT.
    config_home = Path(environ.get("XDG_CONFIG_HOME") or home / ".config")
    config = _opencode_file(config_home / "opencode", environ)
    if environ.get("OPENCODE_CONFIG"):
        path = Path(environ["OPENCODE_CONFIG"]).expanduser()
        config = _merge(config, _substitute(_read_json(path), environ, path.parent))
    project = _opencode_file(workdir, environ)
    if project.get("model"):
        config["model"] = project["model"]  # project files may only choose the model
    if environ.get("OPENCODE_CONFIG_DIR"):
        config = _merge(config, _opencode_file(Path(environ["OPENCODE_CONFIG_DIR"]).expanduser(), environ))
    if environ.get("OPENCODE_CONFIG_CONTENT"):
        inline = _parse_json(environ["OPENCODE_CONFIG_CONTENT"], "OPENCODE_CONFIG_CONTENT")
        config = _merge(config, _substitute(inline, environ, workdir))

    spec = args.model or config.get("model")
    if not spec or "/" not in spec:
        raise ToolCLIError("opencode: choose a model as provider/model (--model or `model` in opencode.json)")
    provider_id, _, model = spec.partition("/")
    entry = (config.get("provider") or {}).get(provider_id) or {}
    options = entry.get("options") or {}
    data_home = Path(environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    auth = _read_json(data_home / "opencode" / "auth.json").get(provider_id) or {}
    if auth and auth.get("type") != "api":
        raise ToolCLIError(f"opencode: {provider_id} uses an OAuth login, which cannot be reused; add an API key")
    npm = entry.get("npm", "")
    if provider_id == "anthropic" or npm == "@ai-sdk/anthropic":
        provider, env_name = "anthropic", "ANTHROPIC_API_KEY"
    elif provider_id == "openai" or npm in {"@ai-sdk/openai", "@ai-sdk/openai-compatible"}:
        provider, env_name = "openai", "OPENAI_API_KEY"
    else:
        raise ToolCLIError(f"opencode: provider {provider_id!r} is not OpenAI- or Anthropic-compatible")
    api_key = options.get("apiKey") or auth.get("key") or environ.get(env_name)
    api_base = options.get("baseURL")
    if not api_key and not api_base:
        raise ToolCLIError(f"opencode: no API key for {provider_id}; set {env_name} or run `opencode auth login`")

    cli = _common_args(" ".join(args.message) if one_shot else None, args.session, resume=bool(args.session))
    cli += _max_tokens_args(environ.get("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"), "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX")
    return ToolInvocation(
        _config(provider, model, api_key, api_base),
        cli,
        cwd=workdir if args.dir else None,
        notes=_permission_note("opencode", ["--auto"] * args.auto),
    )


# ----------------------------------------------------------------- copilot


def _copilot(argv: Sequence[str], environ: Mapping[str, str], home: Path, cwd: Path) -> ToolInvocation:
    parser = _parser("copilot")
    parser.add_argument("-p", "--prompt")
    parser.add_argument("--model")
    parser.add_argument("-C", dest="cd")
    parser.add_argument("--resume")
    parser.add_argument("--allow-all-tools", action="store_true")
    parser.add_argument("--allow-all", "--yolo", action="store_true")
    parser.add_argument("--no-auto-update", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = _parse(parser, argv)

    base_url = environ.get("COPILOT_PROVIDER_BASE_URL")
    if not base_url:
        raise ToolCLIError(
            "copilot: set COPILOT_PROVIDER_BASE_URL (BYOK); GitHub Copilot's hosted models "
            "and GitHub sign-in cannot be reused"
        )
    kind = environ.get("COPILOT_PROVIDER_TYPE", "openai").lower()
    if kind not in {"openai", "anthropic"}:
        raise ToolCLIError(f"copilot: COPILOT_PROVIDER_TYPE={kind!r} is not supported (openai or anthropic)")
    bearer = environ.get("COPILOT_PROVIDER_BEARER_TOKEN")
    if bearer and kind == "anthropic":
        raise ToolCLIError("copilot: COPILOT_PROVIDER_BEARER_TOKEN is not supported for anthropic; use COPILOT_PROVIDER_API_KEY")
    # The wire model is what the provider receives; it defaults to COPILOT_MODEL (or --model).
    model = (
        environ.get("COPILOT_PROVIDER_WIRE_MODEL")
        or args.model
        or environ.get("COPILOT_MODEL")
        or environ.get("COPILOT_PROVIDER_MODEL_ID")
    )
    if not model:
        raise ToolCLIError("copilot: set COPILOT_MODEL, COPILOT_PROVIDER_MODEL_ID, or pass --model")

    cli = _common_args(args.prompt, args.resume, resume=bool(args.resume))
    cli += _max_tokens_args(environ.get("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"), "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS")
    flags = [n for n, on in (("--allow-all-tools", args.allow_all_tools), ("--allow-all", args.allow_all)) if on]
    flags += ["COPILOT_ALLOW_ALL"] * (environ.get("COPILOT_ALLOW_ALL", "").lower() in _TRUE)
    ignored = ["COPILOT_PROVIDER_WIRE_API=responses"] * (environ.get("COPILOT_PROVIDER_WIRE_API") == "responses")
    return ToolInvocation(
        # OpenAI-compatible APIs send the API key as a bearer token, so either variable works there.
        _config(kind, model, bearer or environ.get("COPILOT_PROVIDER_API_KEY"), base_url),
        cli,
        cwd=(cwd / args.cd) if args.cd else None,
        notes=_permission_note("copilot", flags) + _ignored_note("copilot", ignored),
    )


TOOLS: dict[str, Callable[[Sequence[str], Mapping[str, str], Path, Path], ToolInvocation]] = {
    "claude": _claude,
    "codex": _codex,
    "opencode": _opencode,
    "copilot": _copilot,
}


def translate_tool_cli(
    tool: str,
    argv: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> ToolInvocation:
    """Translate ``agent-rt <tool> ...`` into Agent RT CLI arguments and config."""
    try:
        handler = TOOLS[tool]
    except KeyError:
        raise ToolCLIError(f"unknown tool {tool!r}; supported: {', '.join(TOOLS)}") from None
    return handler(
        argv,
        os.environ if environ is None else environ,
        Path.home() if home is None else home,
        Path.cwd() if cwd is None else cwd,
    )


__all__ = ("TOOLS", "ToolCLIError", "ToolInvocation", "translate_tool_cli")
