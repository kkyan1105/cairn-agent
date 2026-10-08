import json
from pathlib import Path

from agents.mcp_client import McpManager


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_configs_merge_with_later_sources_winning(isolated_env):
    home = Path.home()
    write(home / ".cairn" / "settings.json", {"mcpServers": {
        "shared": {"command": "global-cmd"},
        "global_only": {"command": "g"},
    }})
    write(isolated_env / ".cairn" / "settings.json", {"mcpServers": {"shared": {"command": "project-cmd"}}})
    write(isolated_env / ".mcp.json", {"mcpServers": {"mcp_only": {"command": "m"}}})

    configs = McpManager()._load_configs()

    assert configs["shared"]["command"] == "project-cmd"
    assert set(configs) == {"shared", "global_only", "mcp_only"}


def test_flat_config_format_is_supported(isolated_env):
    write(isolated_env / ".mcp.json", {"flat": {"command": "x", "args": ["--y"]}})
    assert McpManager()._load_configs() == {"flat": {"command": "x", "args": ["--y"]}}


def test_entries_without_command_and_bad_files_are_ignored(isolated_env):
    write(isolated_env / ".cairn" / "settings.json", {"permissions": {"allow": []}, "mcpServers": {"no_cmd": {"args": []}}})
    (isolated_env / ".mcp.json").write_text("{ not valid json")
    assert McpManager()._load_configs() == {}


def test_tool_definitions_fill_in_missing_schema_and_description():
    manager = McpManager()
    manager._tools = [{"name": "ping", "serverName": "srv", "description": "", "inputSchema": None}]
    (definition,) = manager.get_tool_definitions()
    assert definition == {
        "name": "mcp__srv__ping",
        "description": "MCP tool ping from srv",
        "input_schema": {"type": "object", "properties": {}},
    }
