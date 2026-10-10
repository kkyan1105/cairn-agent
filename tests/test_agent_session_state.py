def tool_use(tool_id):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "read_file", "input": {}}]}


def tool_result(tool_id):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]}


def test_normalization_keeps_complete_tool_rounds(make_agent):
    agent = make_agent()
    messages = [{"role": "user", "content": "hi"}, tool_use("a"), tool_result("a")]
    assert agent._normalize_anthropic_messages(messages) == messages


def test_normalization_drops_tool_use_without_result(make_agent):
    agent = make_agent()
    messages = [{"role": "user", "content": "hi"}, tool_use("a"), {"role": "user", "content": "next"}]
    assert agent._normalize_anthropic_messages(messages) == [messages[0], messages[2]]


def test_normalization_fixes_tool_use_saved_with_user_role(make_agent):
    agent = make_agent()
    misrouted = dict(tool_use("a"), role="user")
    normalized = agent._normalize_anthropic_messages([misrouted, tool_result("a")])
    assert [m["role"] for m in normalized] == ["assistant", "user"]


def test_restore_session_loads_and_cleans_history(make_agent):
    agent = make_agent()
    agent.restore_session({
        "anthropicMessages": [{"role": "user", "content": "bad \ud800 text"}, tool_use("orphan")],
        "foldedSessionMemories": [{"trigger": "manual"}],
    })
    assert len(agent._anthropic_messages) == 1
    assert "\ud800" not in agent._anthropic_messages[0]["content"]
    assert agent._folded_session_memories == [{"trigger": "manual"}]


def test_cost_budget(make_agent):
    agent = make_agent(max_cost_usd=0.01)
    assert agent._check_budget() == {"exceeded": False}
    agent.total_output_tokens = 1_000
    assert agent._check_budget()["exceeded"] is True


def test_turn_budget(make_agent):
    agent = make_agent(max_turns=2)
    agent.current_turns = 1
    assert agent._check_budget()["exceeded"] is False
    agent.current_turns = 2
    assert "Turn limit" in agent._check_budget()["reason"]
