import asyncio
import json
import sys
from pathlib import Path

import pytest

from agents.mcp_client import McpConnection, McpManager

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


def write_mcp_json(project: Path, servers: dict) -> None:
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": servers}))


def test_connection_handshake_list_and_call():
    async def scenario():
        conn = McpConnection("fake", sys.executable, [FAKE_SERVER])
        await conn.connect()
        try:
            await conn.initialize()
            tools = await conn.list_tools()
            assert [t["name"] for t in tools] == ["echo", "fail"]
            assert all(t["serverName"] == "fake" for t in tools)
            assert await conn.call_tool("echo", {"text": "hello"}) == "hello"
            with pytest.raises(RuntimeError, match="tool failed"):
                await conn.call_tool("fail", {})
        finally:
            conn.close()

    asyncio.run(scenario())


def test_manager_exposes_prefixed_tools_and_routes_calls(isolated_env):
    write_mcp_json(isolated_env, {"fake": {"command": sys.executable, "args": [FAKE_SERVER]}})

    async def scenario():
        manager = McpManager()
        await manager.load_and_connect()
        try:
            names = [t["name"] for t in manager.get_tool_definitions()]
            assert names == ["mcp__fake__echo", "mcp__fake__fail"]
            assert manager.is_mcp_tool("mcp__fake__echo")
            assert not manager.is_mcp_tool("read_file")
            assert await manager.call_tool("mcp__fake__echo", {"text": "routed"}) == "routed"
        finally:
            await manager.disconnect_all()

    asyncio.run(scenario())


def test_manager_skips_servers_that_fail_to_start(isolated_env):
    write_mcp_json(isolated_env, {
        "broken": {"command": "definitely-not-a-real-command-xyz"},
        "fake": {"command": sys.executable, "args": [FAKE_SERVER]},
    })

    async def scenario():
        manager = McpManager()
        await manager.load_and_connect()
        try:
            names = [t["name"] for t in manager.get_tool_definitions()]
            assert names == ["mcp__fake__echo", "mcp__fake__fail"]
        finally:
            await manager.disconnect_all()

    asyncio.run(scenario())


def test_call_to_unknown_server_raises(isolated_env):
    async def scenario():
        manager = McpManager()
        with pytest.raises(RuntimeError, match="not connected"):
            await manager.call_tool("mcp__missing__tool", {})
        with pytest.raises(ValueError):
            await manager.call_tool("mcp__bad", {})

    asyncio.run(scenario())
