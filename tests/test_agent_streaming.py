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


def tool_call(call_id, name="read_file", args='{"file_path": "a.py"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def openai_response(tool_calls):
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def tool_messages(agent):
    return {m["tool_call_id"]: m["content"] for m in agent._openai_messages if m.get("role") == "tool"}


def test_every_openai_tool_call_gets_a_result_when_budget_runs_out(make_agent, monkeypatch):
    agent = make_agent("openai", max_turns=0, enable_memory_prefetch=False, enable_mcp=False, enable_online_evolution=False)

    async def stream():
        return openai_response([tool_call("c1"), tool_call("c2")])

    monkeypatch.setattr(agent, "_call_openai_stream", stream)
    asyncio.run(agent._chat_openai("read a.py twice"))

    results = tool_messages(agent)
    assert set(results) == {"c1", "c2"}
    assert all(r.startswith("Tool execution skipped: Turn limit") for r in results.values())


def test_skipped_results_are_not_duplicated():
    from agents.agent import Agent

    agent = Agent.__new__(Agent)
    agent._openai_messages = [{"role": "tool", "tool_call_id": "c1", "content": "real result"}]
    agent._append_openai_skipped_tool_results([tool_call("c1"), tool_call("c2"), {"id": "x", "type": "other"}], "skipped")

    assert tool_messages(agent) == {"c1": "real result", "c2": "skipped"}


def test_tool_exceptions_become_error_results(make_agent, monkeypatch):
    agent = make_agent("openai", enable_memory_prefetch=False, enable_mcp=False, enable_online_evolution=False)
    responses = iter([openai_response([tool_call("c1")]), {"choices": [{"message": {"role": "assistant", "content": "done"}}]}])

    async def stream():
        return next(responses)

    async def broken_tool(name, inp):
        raise OSError("disk unavailable")

    monkeypatch.setattr(agent, "_call_openai_stream", stream)
    monkeypatch.setattr(agent, "_execute_tool_call", broken_tool)
    asyncio.run(agent._chat_openai("read a.py"))

    assert tool_messages(agent)["c1"] == "Error executing tool: disk unavailable"
