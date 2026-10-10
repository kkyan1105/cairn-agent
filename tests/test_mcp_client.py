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
            assert [t["name"] for t in tools][:2] == ["echo", "fail"]
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
            assert names[:2] == ["mcp__fake__echo", "mcp__fake__fail"]
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
            assert all(name.startswith("mcp__fake__") for name in names)
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


async def connected(server_name="fake"):
    conn = McpConnection(server_name, sys.executable, [FAKE_SERVER])
    await conn.connect()
    await conn.initialize()
    return conn


def test_heavy_stderr_output_does_not_block_the_server():
    async def scenario():
        conn = await connected()
        try:
            assert await asyncio.wait_for(conn.call_tool("noisy", {}), timeout=10) == "done"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_pending_call_fails_when_server_exits():
    async def scenario():
        conn = await connected()
        try:
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(conn.call_tool("crash", {}), timeout=5)
        finally:
            conn.close()

    asyncio.run(scenario())


def test_timed_out_call_is_cleaned_up_and_connection_stays_usable():
    async def scenario():
        conn = await connected()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(conn.call_tool("hang", {}), timeout=0.5)
            assert conn._pending == {}
            assert await asyncio.wait_for(conn.call_tool("echo", {"text": "still alive"}), timeout=5) == "still alive"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_close_fails_pending_calls_and_is_idempotent():
    async def scenario():
        conn = await connected()
        call = asyncio.create_task(conn.call_tool("hang", {}))
        await asyncio.sleep(0.2)
        conn.close()
        with pytest.raises(RuntimeError, match="closed"):
            await call
        conn.close()

    asyncio.run(scenario())
