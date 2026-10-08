from pathlib import Path

import pytest

from agents import subagent


@pytest.fixture(autouse=True)
def fresh_agent_cache():
    subagent.reset_agent_cache()
    yield
    subagent.reset_agent_cache()


def tool_names(config):
    return {t["name"] for t in config["tools"]}


def write_agent(base: Path, filename: str, text: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / filename).write_text(text)


def test_explore_and_plan_are_read_only(isolated_env):
    for agent_type in ("explore", "plan"):
        names = tool_names(subagent.get_sub_agent_config(agent_type))
        assert names <= subagent.READ_ONLY_TOOLS
        assert "write_file" not in names and "run_shell" not in names


def test_general_agent_cannot_spawn_agents(isolated_env):
    names = tool_names(subagent.get_sub_agent_config("general"))
    assert "agent" not in names
    assert "write_file" in names


def test_custom_agent_with_tool_allowlist(isolated_env):
    write_agent(
        isolated_env / ".cairn" / "agents",
        "reviewer.md",
        "---\nname: reviewer\ndescription: Reviews diffs\nallowed-tools: read_file, grep_search\n---\nYou review code.",
    )
    config = subagent.get_sub_agent_config("reviewer")
    assert config["system_prompt"] == "You review code."
    assert tool_names(config) == {"read_file", "grep_search"}


def test_project_agent_overrides_user_agent(isolated_env):
    write_agent(Path.home() / ".cairn" / "agents", "helper.md", "---\ndescription: user\n---\nuser prompt")
    write_agent(isolated_env / ".cairn" / "agents", "helper.md", "---\ndescription: project\n---\nproject prompt")
    assert subagent.get_sub_agent_config("helper")["system_prompt"] == "project prompt"


def test_custom_agents_are_listed_in_descriptions(isolated_env):
    assert subagent.build_agent_descriptions() == ""
    subagent.reset_agent_cache()
    write_agent(isolated_env / ".cairn" / "agents", "docs.md", "---\ndescription: Writes docs\n---\nprompt")
    assert "- **docs**: Writes docs" in subagent.build_agent_descriptions()
