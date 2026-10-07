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


def test_project_settings_rules(isolated_env):
    settings = {"permissions": {"allow": ["run_shell(rm -rf build*)"], "deny": ["write_file(secrets/*)"]}}
    (isolated_env / ".cairn").mkdir()
    (isolated_env / ".cairn" / "settings.json").write_text(json.dumps(settings))

    assert tools.check_permission("run_shell", {"command": "rm -rf build/out"})["action"] == "allow"
    assert tools.check_permission("write_file", {"file_path": "secrets/key.txt"})["action"] == "deny"


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
