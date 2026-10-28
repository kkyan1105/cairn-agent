"""
MCP client module.

This file connects to stdio-based MCP Servers, discovers their tools, and forwards tool calls
issued by the Agent to the corresponding MCP Server.

Implementation:
- No MCP SDK dependency; JSON-RPC messages go directly over stdin/stdout.
- Each MCP Server runs in its own subprocess.
- Each MCP tool is wrapped as `mcp__serverName__toolName` to avoid name clashes with local tools.

Config sources:
- Global config: `~/.cairn/settings.json`
- Project config: `.cairn/settings.json`
- Standard project config: `.mcp.json`

Example config format:
{
    "mcpServers": {
        "name": {
            "command": "...",
            "args": [...],
            "env": {...}
        }
    }
}
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import signal
from pathlib import Path
from typing import Any, Awaitable, Callable

from .ui import print_error, print_info

MCP_TOOL_TIMEOUT_SECONDS = 60.0
MCP_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class McpToolError(RuntimeError):
    """A valid MCP response explicitly reported a failed tool operation."""

    def __init__(self, result: dict):
        self.result = result
        content = result.get("content") or []
        text = "\n".join(str(item.get("text", "")) for item in content
                         if isinstance(item, dict) and item.get("type") == "text")
        super().__init__("MCP tool failed: " + (text or json.dumps(result, ensure_ascii=False)))


# ─── Single MCP connection: one McpConnection per MCP Server subprocess ──────────────────


class BorrowedMcpManager:
    """A child may call its inherited tools, but never owns parent processes."""

    def __init__(self, owner: McpManager, allowed_tools: set[str]):
        self._owner = owner
        self._allowed_tools = frozenset(allowed_tools)

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._allowed_tools

    async def call_tool(self, name: str, args: dict) -> str:
        if name not in self._allowed_tools:
            raise ValueError(f"MCP tool not available in this child: {name}")
        return await self._owner.call_tool(name, args)

    async def disconnect_all(self) -> None:
        # The parent's lifetime encloses all child calls and owns cleanup.
        pass


class McpConnection:
    """Manages a single MCP Server subprocess and the JSON-RPC communication with it."""

    def __init__(self, server_name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        # The MCP Server's name in the config, used later for the tool name prefix and tool call routing.
        self.server_name = server_name
        # Command that starts the MCP Server, e.g. `node`, `python`, or a path to an executable.
        self.command = command
        # Arguments passed to the start command.
        self.args = args or []
        # Extra environment variables, merged with the current process environment on connect.
        self.env = env or {}
        # The MCP Server subprocess. None until connected.
        self._process: asyncio.subprocess.Process | None = None
        # Auto-incrementing JSON-RPC request id counter, used to match requests with responses.
        self._next_id = 1
        # Requests awaiting a response. Key is the JSON-RPC id, value is a Future.
        self._pending: dict[int, asyncio.Future] = {}
        # Background task reading stdout. The MCP Server's responses are read from stdout continuously.
        self._reader_task: asyncio.Task | None = None
        # Background task draining stderr so a chatty server cannot fill the pipe and block.
        self._stderr_task: asyncio.Task | None = None
        self._reader_error: Exception | None = None

    async def connect(self) -> None:
        """Start the MCP Server subprocess and begin reading its stdout in the background."""
        # Subprocess environment = current process environment + extra variables declared in the config.
        merged_env = {**os.environ, **self.env}
        self._reader_error = None
        # Start the MCP Server in stdio mode:
        # - stdin: the client writes JSON-RPC requests to the Server.
        # - stdout: the Server returns JSON-RPC responses to the client.
        # - stderr: keep a separate error pipe so the Server doesn't inherit the current terminal output.
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            start_new_session=os.name != "nt",
            limit=MCP_MAX_MESSAGE_BYTES,
        )
        # Keep reading stdout in the background. This must not block connect(), or initialization couldn't continue.
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _read_loop(self) -> None:
        """Continuously read newline-delimited JSON-RPC responses from the MCP Server's stdout."""
        assert self._process and self._process.stdout
        try:
            while True:
                # MCP stdio usually carries one JSON-RPC message per line.
                line = await self._process.stdout.readline()
                if not line:
                    # Reading nothing means the subprocess's stdout is closed, which usually means the process exited.
                    break
                if len(line) > MCP_MAX_MESSAGE_BYTES:
                    raise ValueError(f"MCP message exceeds {MCP_MAX_MESSAGE_BYTES} bytes")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore invalid JSON so a single garbage line doesn't break the whole connection.
                    continue

                # JSON-RPC responses carry an id; notifications usually don't.
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in msg:
                        # When the Server returns a JSON-RPC error, mark the waiting Future as failed.
                        e = msg["error"]
                        if not fut.done():
                            fut.set_exception(
                                RuntimeError(f"MCP error {e.get('code')}: {e.get('message')}")
                            )
                    elif not fut.done():
                        # For a normal response, only the result part is handed to the caller.
                        fut.set_result(msg.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reader_error = RuntimeError(f"MCP reader failed (message limit {MCP_MAX_MESSAGE_BYTES} bytes): {exc}")
            self._fail_pending(self._reader_error)
        else:
            self._reader_error = RuntimeError(f"MCP server '{self.server_name}' closed")
            self._fail_pending(self._reader_error)

    async def _drain_stderr(self) -> None:
        """Discard the server's stderr. If nobody reads it, the OS pipe buffer fills up and the server blocks on write."""
        assert self._process and self._process.stderr
        try:
            while True:
                chunk = await self._process.stderr.read(65536)
                if not chunk:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _fail_pending(self, error: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and wait for the response with the matching id."""
        if self._reader_error is not None:
            raise RuntimeError(str(self._reader_error))
        assert self._process and self._process.stdin
        # Allocate a unique id for this request. The read loop uses it to find the matching Future.
        req_id = self._next_id
        self._next_id += 1

        # JSON-RPC 2.0 request format:
        # {
        #   "jsonrpc": "2.0",
        #   "id": 1,
        #   "method": "...",
        #   "params": {...}
        # }
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        # The stdio transport uses newlines as message boundaries.
        try:
            self._process.stdin.write((msg + "\n").encode())
            await self._process.stdin.drain()
            return await fut
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise
        except Exception:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification. Notifications have no id and don't wait for a response."""
        if not self._process or not self._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        self._process.stdin.write((msg + "\n").encode())

    async def initialize(self) -> None:
        """Perform the MCP initialization handshake."""
        # initialize is the first step after connecting, used to negotiate the protocol version and client info.
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cairn", "version": "0.1.0"},
        })
        # After a successful initialize request, send the initialized notification per the MCP protocol.
        self._send_notification("notifications/initialized")

    async def list_tools(self) -> list[dict]:
        """Query the list of available tools from this MCP Server."""
        result = await self._send_request("tools/list")
        if not result or not isinstance(result.get("tools"), list):
            return []
        # Keep the original MCP inputSchema and attach serverName so upper layers can prefix tool names and route calls.
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema"),
                "serverName": self.server_name,
            }
            for t in result["tools"]
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        """Call a tool on this MCP Server and return the result as a string."""
        result = await self._send_request("tools/call", {"name": name, "arguments": args})
        if isinstance(result, dict) and result.get("isError") is True:
            raise McpToolError(result)
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            # MCP tool results are usually a list of content blocks. The Agent currently only consumes text blocks.
            text = "\n".join(
                c["text"] for c in result["content"] if c.get("type") == "text"
            )
            if result.get("structuredContent") is not None:
                text += "\n" + json.dumps(result["structuredContent"], ensure_ascii=False)
            return text or json.dumps(result, ensure_ascii=False)
        # If it isn't a standard content list, fall back to a JSON string to avoid losing information.
        return json.dumps(result)

    def close(self) -> None:
        """Shut down the MCP Server subprocess and fail all pending requests."""
        if self._reader_task:
            # Stop the background stdout reader task.
            self._reader_task.cancel()
            self._reader_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self._process:
            try:
                # Kill the subprocess outright so the external MCP Server doesn't linger.
                if getattr(self._process, "stdin", None):
                    self._process.stdin.close()
                if hasattr(self._process, "kill"):
                    if os.name != "nt":
                        os.killpg(self._process.pid, signal.SIGKILL)
                    else:
                        self._process.kill()
            except ProcessLookupError:
                # kill may raise this if the process has already exited; safe to ignore.
                pass
            self._process = None
        # Once the connection is closed, pending requests can never complete, so explicitly fail them.
        self._fail_pending(RuntimeError(f"MCP server '{self.server_name}' closed"))

    async def aclose(self) -> None:
        process = self._process
        tasks = [task for task in (self._reader_task, self._stderr_task) if task]
        self.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        if process:
            await asyncio.wait_for(process.wait(), timeout=5)


# ─── MCP manager: manages all MCP Server connections and tool routing ─────────────────────────────


class McpManager:
    """
    Manages all MCP Server connections.

    Usage:
    1. Call load_and_connect(authorize_start=...) to authorize startup, connect to Servers, and discover tools.
    2. Call get_tool_definitions() to expose MCP tools to the model.
    3. When the model calls a tool named mcp__server__tool, call_tool() routes it to the right Server.
    """

    def __init__(self):
        # Connected MCP Servers. Key is the server name, value is its connection object.
        self._connections: dict[str, McpConnection] = {}
        # Tool definitions discovered from all MCP Servers, kept close to the original MCP format.
        self._tools: list[dict] = []
        # Guards against reconnecting. load_and_connect() should only really run once.
        self._connected = False

    async def load_and_connect(
        self, *, authorize_start: Callable[[str, dict], Awaitable[bool]] | None = None
    ) -> None:
        """Authorize each configured process before connecting and discovering tools."""
        if self._connected:
            return
        self._connected = True

        # Merge global, project, and .mcp.json configs. Configs read later override Servers with the same name.
        configs = self._load_configs()
        if not configs:
            return

        # Wait at most 15 seconds for each Server's initialization and tool discovery so a bad config can't stall startup.
        timeout = 15.0

        for name, cfg in configs.items():
            conn = None
            try:
                # Configuration discovery must never implicitly execute project commands.
                startup = {
                    "server_name": name,
                    "command": shlex.join([cfg["command"], *(cfg.get("args") or [])]),
                    "cwd": str(Path.cwd()),
                }
                if authorize_start is None or not await authorize_start("mcp_start", startup):
                    print_info(f"MCP startup denied: {name}")
                    continue
                conn = McpConnection(name, cfg["command"], cfg.get("args"), cfg.get("env"))
                # Connect subprocess -> MCP initialization handshake -> list tools.
                await conn.connect()
                await asyncio.wait_for(conn.initialize(), timeout=timeout)
                server_tools = await asyncio.wait_for(conn.list_tools(), timeout=timeout)
                # Only register the connection as usable if both initialization and tool discovery succeed.
                self._connections[name] = conn
                self._tools.extend(server_tools)
                print_info(f"MCP connected: {name} ({len(server_tools)} tools)")
            except asyncio.CancelledError:
                if conn is not None:
                    await conn.aclose()
                raise
            except Exception as e:
                # A single Server failing doesn't affect the others. Failed connections are closed to clean up the subprocess.
                print_error(f"MCP failed to connect: {name}: {e}")
                if conn is not None:
                    await conn.aclose()

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions usable directly by the Agent/Anthropic, with MCP tool names prefixed."""
        return [
            {
                # Prefix format: mcp__serverName__toolName.
                # This avoids clashes between MCP tools and built-in tools, and lets call_tool() parse the route back out.
                "name": f"mcp__{t['serverName']}__{t['name']}",
                "description": t.get("description") or f"MCP tool {t['name']} from {t['serverName']}",
                # Anthropic's tool field is input_schema; the original MCP tool field is usually inputSchema.
                "input_schema": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for t in self._tools
        ]

    def is_mcp_tool(self, name: str) -> bool:
        """Check whether a tool name is an MCP tool name."""
        return name.startswith("mcp__")

    async def call_tool(self, prefixed_name: str, args: dict) -> str:
        """Route a prefixed MCP tool call to the correct MCP Server."""
        # Tool name format is mcp__serverName__toolName.
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {prefixed_name}")
        server_name = parts[1]
        # The tool name itself may contain "__", so everything after the third segment is joined back together.
        tool_name = "__".join(parts[2:])  # tool name might contain __
        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(f"MCP server '{server_name}' not connected")
        try:
            return await asyncio.wait_for(
                conn.call_tool(tool_name, args), timeout=MCP_TOOL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP tool '{prefixed_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS:g} seconds"
            ) from exc

    async def disconnect_all(self) -> None:
        """Disconnect all MCP Servers and clear the tool cache."""
        await asyncio.gather(*(conn.aclose() for conn in self._connections.values()))
        self._connections.clear()
        self._tools.clear()
        self._connected = False

    # ─── Config loading ──────────────────────────────────────

    def _load_configs(self) -> dict[str, dict]:
        """Load and merge MCP Server configs in priority order."""
        merged: dict[str, dict] = {}

        # 1. Global config: ~/.cairn/settings.json
        global_path = Path.home() / ".cairn" / "settings.json"
        self._merge_config_file(global_path, merged)

        # 2. Current project config: <cwd>/.cairn/settings.json
        project_path = Path.cwd() / ".cairn" / "settings.json"
        self._merge_config_file(project_path, merged)

        # 3. Standard project config: <cwd>/.mcp.json
        mcp_json_path = Path.cwd() / ".mcp.json"
        self._merge_config_file(mcp_json_path, merged)

        return merged

    def _merge_config_file(self, path: Path, target: dict[str, dict]) -> None:
        """Merge the mcpServers from a single config file into target."""
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            # Two formats are supported:
            # 1. {"mcpServers": {"name": {...}}}
            # 2. {"name": {...}}
            servers = raw.get("mcpServers", raw)
            for name, config in servers.items():
                # Only accept objects that include command; invalid entries are ignored.
                if isinstance(config, dict) and "command" in config:
                    target[name] = config
        except Exception:
            # Skip config files with format errors so one bad config doesn't stop the whole Agent from starting.
            pass
