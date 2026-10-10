import pytest

from agents.agent import Agent


@pytest.mark.parametrize(
    ("model", "thinking", "expected"),
    [
        ("claude-sonnet-4-6", True, "adaptive"),
        ("claude-opus-4-6", True, "adaptive"),
        ("claude-sonnet-4-20250514", True, "enabled"),
        ("claude-3-7-sonnet-latest", True, "disabled"),
        ("claude-sonnet-4-6", False, "disabled"),
    ],
)
def test_thinking_mode_resolution(isolated_env, model, thinking, expected):
    agent = Agent(model=model, api_key="test", thinking=thinking)
    assert agent._thinking_mode == expected


def test_thinking_is_disabled_for_non_claude_models(isolated_env):
    agent = Agent(model="gpt-4o", api_base="http://localhost:1/v1", api_key="test", thinking=True)
    assert agent._thinking_mode == "disabled"
