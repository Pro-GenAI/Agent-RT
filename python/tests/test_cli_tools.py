import json
import sys

import pytest

from agent_rt_cli import async_main
from ext.cli_tools import ToolCLIError, translate_tool_cli


def _deployment(invocation):
    (deployment,) = invocation.config.models
    return deployment


def translate(tool, argv, tmp_path, environ=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return translate_tool_cli(
        tool, argv, environ=environ or {}, home=home, cwd=tmp_path
    )


class TestClaude:
    def test_print_prompt_model_alias_and_env(self, tmp_path):
        inv = translate(
            "claude",
            ["-p", "hi", "--model", "opus", "--max-turns", "3"],
            tmp_path,
            {"ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_BASE_URL": "http://gw"},
        )
        d = _deployment(inv)
        assert (d.provider, d.model, d.api_key, d.api_base) == (
            "anthropic",
            "claude-opus-5-5",
            "sk-test",
            "http://gw",
        )
        assert inv.argv == ["-p", "hi", "--max-turns", "3"]
        assert inv.config.default_model == "claude-opus-5-5"

    def test_settings_env_and_model_with_real_env_winning(self, tmp_path):
        (tmp_path / "home" / ".claude").mkdir(parents=True)
        (tmp_path / "home" / ".claude" / "settings.json").write_text(
            json.dumps(
                {"model": "haiku", "env": {"ANTHROPIC_API_KEY": "from-settings"}}
            )
        )
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.local.json").write_text(
            json.dumps({"model": "sonnet"})
        )
        d = _deployment(translate("claude", ["-p", "hi"], tmp_path))
        assert (d.model, d.api_key) == ("claude-sonnet-5-5", "from-settings")
        d = _deployment(
            translate("claude", ["-p", "hi"], tmp_path, {"ANTHROPIC_API_KEY": "real"})
        )
        assert d.api_key == "real"

    def test_system_prompts_resume_and_permission_note(self, tmp_path):
        inv = translate(
            "claude",
            [
                "-p",
                "hi",
                "--system-prompt",
                "S",
                "--append-system-prompt",
                "A",
                "-r",
                "s1",
                "--dangerously-skip-permissions",
            ],
            tmp_path,
            {"ANTHROPIC_API_KEY": "k"},
        )
        assert inv.argv == [
            "-p",
            "hi",
            "--session",
            "s1",
            "--resume",
            "--instructions",
            "S\n\nA",
        ]
        assert "approval" in inv.notes[0]

    def test_alias_overrides_default_model_and_max_tokens(self, tmp_path):
        env = {
            "ANTHROPIC_API_KEY": "k",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "my-sonnet",
            "ANTHROPIC_DEFAULT_MODEL": "sonnet[1m]",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
        }
        inv = translate("claude", ["-p", "hi"], tmp_path, env)
        assert _deployment(inv).model == "my-sonnet"
        assert inv.argv[-2:] == ["--max-output-tokens", "4096"]
        with pytest.raises(ToolCLIError, match="positive integer"):
            translate(
                "claude",
                ["-p", "hi"],
                tmp_path,
                {**env, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "lots"},
            )

    def test_project_settings_cannot_set_credentials_or_endpoint(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text(
            json.dumps(
                {"env": {"ANTHROPIC_BASE_URL": "http://evil", "ANTHROPIC_API_KEY": "x"}}
            )
        )
        d = _deployment(
            translate("claude", ["-p", "hi"], tmp_path, {"ANTHROPIC_API_KEY": "real"})
        )
        assert (d.api_key, d.api_base) == ("real", None)

    @pytest.mark.parametrize(
        ("env", "match"),
        [
            ({"CLAUDE_CODE_OAUTH_TOKEN": "t"}, "CLAUDE_CODE_OAUTH_TOKEN"),
            ({"ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_USE_BEDROCK": "1"}, "Bedrock"),
        ],
    )
    def test_unsupported_auth_is_refused(self, tmp_path, env, match):
        with pytest.raises(ToolCLIError, match=match):
            translate("claude", ["-p", "hi"], tmp_path, env)

    def test_subscription_login_is_refused(self, tmp_path):
        with pytest.raises(ToolCLIError, match="ANTHROPIC_API_KEY"):
            translate("claude", ["-p", "hi"], tmp_path, {"ANTHROPIC_AUTH_TOKEN": "t"})

    @pytest.mark.parametrize(
        "argv", [["--output-format", "json", "-p", "hi"], ["--frobnicate"], ["hi"]]
    )
    def test_unsupported_usage_fails_clearly(self, tmp_path, argv):
        with pytest.raises(ToolCLIError):
            translate("claude", argv, tmp_path, {"ANTHROPIC_API_KEY": "k"})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="config.toml needs tomllib")
class TestCodex:
    def write(self, tmp_path, text):
        codex = tmp_path / "home" / ".codex"
        codex.mkdir(parents=True, exist_ok=True)
        (codex / "config.toml").write_text(text)
        return codex

    def test_custom_provider_profile_and_overrides(self, tmp_path):
        self.write(
            tmp_path,
            'model = "base"\nmodel_provider = "local"\n'
            '[model_providers.local]\nbase_url = "http://127.0.0.1:9/v1"\nenv_key = "LOCAL_KEY"\n'
            '[profiles.fast]\nmodel = "fast-model"\n',
        )
        inv = translate(
            "codex",
            ["exec", "-p", "fast", "--ephemeral", "hi"],
            tmp_path,
            {"LOCAL_KEY": "k"},
        )
        d = _deployment(inv)
        assert (d.provider, d.model, d.api_key, d.api_base) == (
            "openai",
            "fast-model",
            "k",
            "http://127.0.0.1:9/v1",
        )
        assert inv.argv == ["-p", "hi", "--no-save"]
        inv = translate(
            "codex",
            ["exec", "-c", 'model="o"', "-m", "flag", "hi"],
            tmp_path,
            {"LOCAL_KEY": "k"},
        )
        assert _deployment(inv).model == "flag"
        inv = translate(
            "codex", ["exec", "-c", 'model="o"', "hi"], tmp_path, {"LOCAL_KEY": "k"}
        )
        assert _deployment(inv).model == "o"

    def test_openai_key_from_auth_json_and_chatgpt_refused(self, tmp_path):
        codex = self.write(tmp_path, "")
        with pytest.raises(ToolCLIError, match="ChatGPT"):
            translate("codex", ["exec", "hi"], tmp_path)
        (codex / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-auth"}))
        inv = translate("codex", ["exec", "-C", "sub", "hi"], tmp_path)
        assert _deployment(inv).api_key == "sk-auth"
        assert inv.cwd == tmp_path / "sub"

    def test_codex_api_key_profile_file_and_openai_base_url(self, tmp_path):
        codex = self.write(tmp_path, 'openai_base_url = "http://proxy/v1"\n')
        (codex / "work.config.toml").write_text('model = "from-profile-file"\n')
        inv = translate(
            "codex",
            ["exec", "--profile", "work", "hi"],
            tmp_path,
            {"CODEX_API_KEY": "codex-key", "OPENAI_API_KEY": "legacy"},
        )
        d = _deployment(inv)
        assert (d.model, d.api_key, d.api_base) == (
            "from-profile-file",
            "codex-key",
            "http://proxy/v1",
        )

    def test_local_oss_provider_and_ignored_headers(self, tmp_path):
        self.write(
            tmp_path,
            'model_provider = "gw"\n[model_providers.gw]\nbase_url = "http://gw/v1"\n'
            'experimental_bearer_token = "tok"\nhttp_headers = { "X-A" = "1" }\n',
        )
        inv = translate("codex", ["exec", "hi"], tmp_path)
        assert _deployment(inv).api_key == "tok"
        assert "http_headers" in inv.notes[0]
        d = _deployment(
            translate(
                "codex",
                ["exec", "--oss", "--local-provider", "lmstudio", "hi"],
                tmp_path,
            )
        )
        assert (d.api_base, d.api_key) == ("http://localhost:1234/v1", None)

    def test_chatgpt_access_token_refused(self, tmp_path):
        self.write(tmp_path, "")
        with pytest.raises(ToolCLIError, match="CODEX_ACCESS_TOKEN"):
            translate("codex", ["exec", "hi"], tmp_path, {"CODEX_ACCESS_TOKEN": "t"})

    def test_interactive_prompt_needs_exec(self, tmp_path):
        with pytest.raises(ToolCLIError, match="codex exec"):
            translate("codex", ["hi"], tmp_path, {"OPENAI_API_KEY": "k"})


class TestOpencode:
    def global_config(self, tmp_path, text):
        directory = tmp_path / "home" / ".config" / "opencode"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "opencode.jsonc").write_text(text)
        return directory

    def test_jsonc_global_config_env_and_file_substitution(self, tmp_path):
        directory = self.global_config(
            tmp_path,
            """{
          // comments, trailing commas and "//" inside strings are fine
          "model": "local/m1",
          "provider": {"local": {"npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://x/v1?a=//b", "apiKey": "{env:LK}{file:suffix.txt}",},},},
        }""",
        )
        (directory / "suffix.txt").write_text("-tail\n")
        (tmp_path / "opencode.json").write_text(json.dumps({"model": "local/m2"}))
        inv = translate(
            "opencode",
            ["run", "say", "hi"],
            tmp_path,
            {"LK": "secret", "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "900"},
        )
        d = _deployment(inv)
        assert (d.provider, d.model, d.api_key, d.api_base) == (
            "openai",
            "m2",
            "secret-tail",
            "http://x/v1?a=//b",
        )
        assert inv.argv == ["-p", "say hi", "--max-output-tokens", "900"]

    def test_project_config_cannot_define_providers(self, tmp_path):
        (tmp_path / "opencode.json").write_text(
            json.dumps(
                {
                    "model": "evil/m",
                    "provider": {
                        "evil": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {"baseURL": "http://evil/v1"},
                        }
                    },
                }
            )
        )
        with pytest.raises(ToolCLIError, match="not OpenAI- or Anthropic-compatible"):
            translate("opencode", ["run", "hi"], tmp_path, {"OPENAI_API_KEY": "k"})

    def test_config_env_variables_precedence(self, tmp_path):
        self.global_config(tmp_path, json.dumps({"model": "openai/global"}))
        custom = tmp_path / "custom.json"
        custom.write_text(json.dumps({"model": "openai/custom"}))
        env = {"OPENAI_API_KEY": "k", "OPENCODE_CONFIG": str(custom)}
        assert (
            _deployment(translate("opencode", ["run", "hi"], tmp_path, env)).model
            == "custom"
        )
        env["OPENCODE_CONFIG_CONTENT"] = '{"model": "openai/inline"}'
        assert (
            _deployment(translate("opencode", ["run", "hi"], tmp_path, env)).model
            == "inline"
        )

    def test_auth_json_api_key_and_oauth_refused(self, tmp_path):
        auth = tmp_path / "home" / ".local" / "share" / "opencode"
        auth.mkdir(parents=True)
        (auth / "auth.json").write_text(
            json.dumps({"anthropic": {"type": "api", "key": "ak"}})
        )
        d = _deployment(
            translate("opencode", ["run", "-m", "anthropic/claude-x", "hi"], tmp_path)
        )
        assert (d.provider, d.api_key) == ("anthropic", "ak")
        (auth / "auth.json").write_text(json.dumps({"anthropic": {"type": "oauth"}}))
        with pytest.raises(ToolCLIError, match="OAuth"):
            translate("opencode", ["run", "-m", "anthropic/claude-x", "hi"], tmp_path)


class TestCopilot:
    def test_byok_environment(self, tmp_path):
        inv = translate(
            "copilot",
            ["-p", "hi", "--allow-all-tools"],
            tmp_path,
            {
                "COPILOT_PROVIDER_BASE_URL": "http://x/v1",
                "COPILOT_MODEL": "m",
                "COPILOT_PROVIDER_API_KEY": "k",
            },
        )
        d = _deployment(inv)
        assert (d.provider, d.model, d.api_key, d.api_base) == (
            "openai",
            "m",
            "k",
            "http://x/v1",
        )
        assert inv.argv == ["-p", "hi"]
        assert inv.notes

    def test_bearer_wire_model_and_max_tokens(self, tmp_path):
        inv = translate(
            "copilot",
            ["-p", "hi", "--model", "flag"],
            tmp_path,
            {
                "COPILOT_PROVIDER_BASE_URL": "http://x/v1",
                "COPILOT_PROVIDER_WIRE_MODEL": "wire",
                "COPILOT_PROVIDER_API_KEY": "k",
                "COPILOT_PROVIDER_BEARER_TOKEN": "b",
                "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "512",
                "COPILOT_ALLOW_ALL": "true",
            },
        )
        d = _deployment(inv)
        assert (d.model, d.api_key) == ("wire", "b")
        assert inv.argv == ["-p", "hi", "--max-output-tokens", "512"]
        assert "COPILOT_ALLOW_ALL" in inv.notes[0]

    def test_hosted_copilot_refused(self, tmp_path):
        with pytest.raises(ToolCLIError, match="BYOK"):
            translate("copilot", ["-p", "hi"], tmp_path, {"GH_TOKEN": "t"})


async def test_cli_entry_reports_tool_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert await async_main(["claude", "-p", "hi"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
