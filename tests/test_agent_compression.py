import time

from agents import agent as agent_module
from agents.agent import SNIP_PLACEHOLDER


def anthropic_tool_round(tool_id, name, content, file_path=None):
    tool_input = {"file_path": file_path} if file_path else {}
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]},
    ]


def tool_results(agent):
    return [
        block["content"]
        for msg in agent._anthropic_messages
        if msg["role"] == "user"
        for block in msg["content"]
        if block.get("type") == "tool_result"
    ]


def set_utilization(agent, ratio):
    agent.last_input_token_count = int(agent.effective_window * ratio)


def test_budgeting_truncates_large_results_only_under_pressure(make_agent):
    agent = make_agent()
    agent._anthropic_messages = anthropic_tool_round("t1", "read_file", "x" * 40000)

    set_utilization(agent, 0.3)
    agent._budget_tool_results_anthropic()
    assert len(tool_results(agent)[0]) == 40000

    set_utilization(agent, 0.8)
    agent._budget_tool_results_anthropic()
    (content,) = tool_results(agent)
    assert len(content) < 16000
    assert "chars truncated" in content


def test_snipping_keeps_only_recent_results(make_agent):
    agent = make_agent()
    for i in range(5):
        agent._anthropic_messages += anthropic_tool_round(f"t{i}", "grep_search", f"result {i}")

    set_utilization(agent, 0.65)
    agent._snip_stale_results_anthropic()

    assert tool_results(agent) == [SNIP_PLACEHOLDER, SNIP_PLACEHOLDER, "result 2", "result 3", "result 4"]


def test_snipping_drops_earlier_reads_of_the_same_file(make_agent):
    agent = make_agent()
    agent._anthropic_messages = (
        anthropic_tool_round("a", "read_file", "old a.py", "a.py")
        + anthropic_tool_round("b", "read_file", "b.py", "b.py")
        + anthropic_tool_round("c", "read_file", "c.py", "c.py")
        + anthropic_tool_round("d", "read_file", "new a.py", "a.py")
    )

    set_utilization(agent, 0.65)
    agent._snip_stale_results_anthropic()

    assert tool_results(agent) == [SNIP_PLACEHOLDER, "b.py", "c.py", "new a.py"]


def test_snipping_ignores_tools_that_are_not_snippable(make_agent):
    agent = make_agent()
    for i in range(5):
        agent._anthropic_messages += anthropic_tool_round(f"t{i}", "write_file", f"wrote {i}")

    set_utilization(agent, 0.9)
    agent._snip_stale_results_anthropic()

    assert SNIP_PLACEHOLDER not in tool_results(agent)


def test_microcompaction_clears_old_results_after_idle(make_agent):
    agent = make_agent()
    for i in range(5):
        agent._anthropic_messages += anthropic_tool_round(f"t{i}", "run_shell", f"out {i}")

    agent.last_api_call_time = time.time()
    agent._microcompact_anthropic()
    assert "[Old result cleared]" not in tool_results(agent)

    agent.last_api_call_time = time.time() - agent_module.MICROCOMPACT_IDLE_S - 1
    agent._microcompact_anthropic()
    assert tool_results(agent)[:2] == ["[Old result cleared]"] * 2
    assert tool_results(agent)[2:] == ["out 2", "out 3", "out 4"]


def test_openai_snipping_and_budgeting(make_agent):
    agent = make_agent("openai")
    agent._openai_messages += [{"role": "tool", "tool_call_id": str(i), "content": f"r{i}"} for i in range(4)]
    agent._openai_messages.append({"role": "tool", "tool_call_id": "big", "content": "y" * 40000})

    set_utilization(agent, 0.8)
    agent._budget_tool_results_openai()
    agent._snip_stale_results_openai()

    contents = [m["content"] for m in agent._openai_messages if m["role"] == "tool"]
    assert contents[:2] == [SNIP_PLACEHOLDER] * 2
    assert contents[2:4] == ["r2", "r3"]
    assert "chars truncated" in contents[4]


def test_large_results_are_persisted_with_a_preview(make_agent):
    agent = make_agent()
    small = "fine"
    assert agent._persist_large_result("run_shell", small) == small

    big = "\n".join(f"line {i}" for i in range(10000))
    summary = agent._persist_large_result("run_shell", big)
    assert summary.startswith("[Result too large")
    assert "line 199" in summary and "line 200\n" not in summary

    saved_path = summary.split("Full output saved to ")[1].split(". You can")[0]
    with open(saved_path, encoding="utf-8") as f:
        assert f.read() == big
