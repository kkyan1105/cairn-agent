"""Minimal stdio MCP server used by the test suite."""

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
    {"name": "fail", "description": "Always returns a JSON-RPC error"},
]


def respond(msg_id, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue  # notifications need no response
    method = msg["method"]
    if method == "initialize":
        respond(msg["id"], {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake"}})
    elif method == "tools/list":
        sys.stdout.write("not json, should be ignored\n")
        respond(msg["id"], {"tools": TOOLS})
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "echo":
            text = msg["params"]["arguments"].get("text", "")
            respond(msg["id"], {"content": [{"type": "text", "text": text}, {"type": "image", "data": ""}]})
        else:
            respond(msg["id"], error={"code": -32000, "message": "tool failed"})
    else:
        respond(msg["id"], error={"code": -32601, "message": "method not found"})
