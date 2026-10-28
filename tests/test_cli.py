import sys

import pytest

from agents import main as cli

ENV_KEYS = ["APIKEY", "API", "CAIRN_API_KEY", "CAIRN_API_BASE", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["cairn", *argv])
    return cli.parse_args()


@pytest.mark.parametrize(
    ("flags", "mode"),
    [((), "default"), (("--yolo",), "bypassPermissions"), (("--plan",), "plan"), (("--accept-edits",), "acceptEdits"), (("--dont-ask",), "dontAsk")],
)
def test_permission_mode_from_flags(monkeypatch, flags, mode):
    assert cli._resolve_permission_mode(parse(monkeypatch, *flags)) == mode


def test_prompt_and_folding_flags(monkeypatch):
    args = parse(monkeypatch, "--no-folding", "--fold-threshold", "0.5", "fix", "the", "bug")
    assert args.prompt == ["fix", "the", "bug"]
    assert args.no_folding is True and args.fold_threshold == 0.5


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.deepseek.com/anthropic", True),
        ("https://proxy.example.com/anthropic/", True),
        ("https://proxy.example.com/anthropic/v1", True),
        ("https://api.openai.com/v1", False),
        (None, False),
    ],
)
def test_anthropic_base_url_detection(url, expected):
    from agents.config import _is_anthropic_compatible_base_url

    assert _is_anthropic_compatible_base_url(url) is expected


def test_generic_config_with_anthropic_url(clean_env):
    clean_env.setenv("APIKEY", "k-generic")
    clean_env.setenv("API", "https://api.deepseek.com/anthropic")
    assert cli._resolve_api_config(None) == ("https://api.deepseek.com/anthropic", "k-generic", False)


def test_cli_api_base_overrides_env_and_selects_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "k-openai")
    clean_env.setenv("API", "https://api.deepseek.com/anthropic")
    assert cli._resolve_api_config("https://my-proxy/v1") == ("https://my-proxy/v1", "k-openai", True)


def test_native_keys_without_base_url(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "k-anthropic")
    assert cli._resolve_api_config(None) == (None, "k-anthropic", False)

    clean_env.delenv("ANTHROPIC_API_KEY")
    clean_env.setenv("OPENAI_API_KEY", "k-openai")
    clean_env.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert cli._resolve_api_config(None) == ("https://api.openai.com/v1", "k-openai", True)


def test_cairn_aliases_and_empty_config(clean_env):
    assert cli._resolve_api_config(None) == (None, None, False)
    clean_env.setenv("CAIRN_API_KEY", "  k-alias  ")
    assert cli._resolve_api_config(None) == (None, "k-alias", False)
