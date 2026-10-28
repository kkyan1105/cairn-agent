import asyncio

import pytest


def test_tools_do_not_start_before_the_stream_succeeds(make_agent, monkeypatch):
    """A retried or failed stream must not leave tools half-executed."""
    agent = make_agent(enable_memory_prefetch=False, enable_mcp=False, enable_online_evolution=False)
    executed = []

    async def fake_execute(name, inp):
        executed.append(name)
        return "ok"

    async def failing_stream(on_tool_block_complete=None):
        if on_tool_block_complete:
            on_tool_block_complete({"id": "t1", "name": "read_file", "input": {"file_path": "a.py"}})
        await asyncio.sleep(0)
        raise RuntimeError("stream dropped")

    monkeypatch.setattr(agent, "_execute_tool_call", fake_execute)
    monkeypatch.setattr(agent, "_call_anthropic_stream", failing_stream)

    async def scenario():
        with pytest.raises(RuntimeError, match="stream dropped"):
            await agent._chat_anthropic("read a.py")
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert executed == []

