import json

import pytest

from agents import tools


@pytest.fixture(autouse=True)
def fresh_permission_rules():
    tools.reset_permission_cache()
    yield
    tools.reset_permission_cache()


@pytest.mark.parametrize("command", ["rm -rf build", "sudo reboot", "git push origin main", "kill 1234"])
def test_dangerous_commands_are_detected(command):
    assert tools.is_dangerous(command)


@pytest.mark.parametrize("command", ["ls -la", "git status", "python -m pytest", "echo removed"])
def test_safe_commands_are_not_flagged(command):
    assert not tools.is_dangerous(command)


def test_read_tools_are_always_allowed(isolated_env):
    assert tools.check_permission("read_file", {"file_path": "x.py"})["action"] == "allow"


def test_dangerous_shell_needs_confirmation(isolated_env):
    result = tools.check_permission("run_shell", {"command": "rm -rf build"})
    assert result == {"action": "confirm", "message": "rm -rf build"}


def test_dont_ask_mode_denies_instead_of_confirming(isolated_env):
    result = tools.check_permission("run_shell", {"command": "rm -rf build"}, mode="dontAsk")
    assert result["action"] == "deny"


def test_plan_mode_blocks_edits_and_shell(isolated_env):
    assert tools.check_permission("edit_file", {"file_path": "a.py"}, mode="plan")["action"] == "deny"
    assert tools.check_permission("run_shell", {"command": "ls"}, mode="plan")["action"] == "deny"


def test_plan_mode_allows_writing_the_plan_file(isolated_env):
    result = tools.check_permission(
        "write_file", {"file_path": "plan.md"}, mode="plan", plan_file_path="plan.md"
    )
    assert result["action"] == "allow"


def test_accept_edits_mode_allows_edits(isolated_env):
    assert tools.check_permission("write_file", {"file_path": "new.py"}, mode="acceptEdits")["action"] == "allow"


def write_settings(directory, permissions):
    (directory / ".cairn").mkdir(parents=True, exist_ok=True)
    (directory / ".cairn" / "settings.json").write_text(json.dumps({"permissions": permissions}))


def test_project_deny_rules_always_apply_but_allow_rules_need_trust(isolated_env):
    from agents.project_trust import trust_project

    write_settings(isolated_env, {"allow": ["run_shell(npm test*)"], "deny": ["write_file(secrets/*)"]})
    assert tools.check_permission("write_file", {"file_path": "secrets/key.txt"})["action"] == "deny"
    assert tools.check_permission("run_shell", {"command": "npm test -- -q"})["action"] == "confirm"

    trust_project()
    tools.reset_permission_cache()
    assert tools.check_permission("run_shell", {"command": "npm test -- -q"})["action"] == "allow"


def test_changing_project_settings_revokes_trust(isolated_env):
    from agents.project_trust import is_project_trusted, trust_project

    write_settings(isolated_env, {"allow": ["run_shell"]})
    trust_project()
    assert is_project_trusted()
    write_settings(isolated_env, {"allow": ["run_shell", "write_file"]})
    assert not is_project_trusted()


def test_every_shell_command_needs_confirmation(isolated_env):
    assert tools.check_permission("run_shell", {"command": "ls"}) == {"action": "confirm", "message": "ls"}
    assert tools.check_permission("run_shell", {"command": "ls"}, mode="acceptEdits")["action"] == "confirm"


def test_edit_file_replaces_unique_match(isolated_env):
    target = isolated_env / "a.py"
    target.write_text("x = 1\ny = 2\n")
    result = tools._edit_file({"file_path": str(target), "old_string": "y = 2", "new_string": "y = 3"})
    assert result.startswith("Successfully edited")
    assert target.read_text() == "x = 1\ny = 3\n"


def test_edit_file_rejects_ambiguous_match(isolated_env):
    target = isolated_env / "a.py"
    target.write_text("x = 1\nx = 1\n")
    result = tools._edit_file({"file_path": str(target), "old_string": "x = 1", "new_string": "x = 2"})
    assert "Must be unique" in result
    assert target.read_text() == "x = 1\nx = 1\n"


def test_edit_file_matches_through_curly_quotes(isolated_env):
    target = isolated_env / "a.py"
    target.write_text("print(“hello”)\n")
    result = tools._edit_file({"file_path": str(target), "old_string": 'print("hello")', "new_string": 'print("hi")'})
    assert "quote normalization" in result
    assert target.read_text() == 'print("hi")\n'


def test_allow_rules_cannot_bypass_plan_mode(isolated_env):
    from pathlib import Path

    write_settings(Path.home(), {"allow": ["run_shell", "write_file"]})

    assert tools.check_permission("run_shell", {"command": "ls"})["action"] == "allow"
    assert tools.check_permission("run_shell", {"command": "ls"}, mode="plan")["action"] == "deny"
    assert tools.check_permission("write_file", {"file_path": "a.py"}, mode="plan")["action"] == "deny"


def test_mcp_tools_need_confirmation_showing_arguments(isolated_env):
    result = tools.check_permission("mcp__browser__navigate", {"url": "https://example.com"})
    assert result["action"] == "confirm"
    assert result["message"] == 'MCP tool call: mcp__browser__navigate\nArguments: {"url": "https://example.com"}'


def test_mcp_tools_are_denied_in_dont_ask_and_plan_modes(isolated_env):
    assert tools.check_permission("mcp__docs__search", {}, mode="dontAsk")["action"] == "deny"
    assert tools.check_permission("mcp__docs__search", {}, mode="plan")["action"] == "deny"


def test_user_allow_rule_can_preapprove_an_mcp_tool(isolated_env):
    from pathlib import Path

    write_settings(Path.home(), {"allow": ["mcp__docs__search"]})
    assert tools.check_permission("mcp__docs__search", {})["action"] == "allow"
    assert tools.check_permission("mcp__docs__search", {}, mode="plan")["action"] == "deny"
