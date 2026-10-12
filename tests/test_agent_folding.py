import asyncio

import pytest


def run(coro):
    return asyncio.run(coro)


def tool_names(agent):
    return {t["name"] for t in agent.tools}


@pytest.mark.parametrize(
    "kwargs",
    [{"auto_compact_threshold": 0}, {"auto_compact_threshold": 1.2}, {"max_fold_limit": -1}, {"fold_memory_mode": "other"}],
)
def test_invalid_folding_options_are_rejected(make_agent, kwargs):
    with pytest.raises(ValueError):
        make_agent(**kwargs)


def test_disabling_folding_removes_the_compact_tool(make_agent):
    assert "compact_context" in tool_names(make_agent())
    agent = make_agent(enable_folding=False)
    assert "compact_context" not in tool_names(agent)
    assert run(agent._execute_tool_call("compact_context", {})) == "Tool not available in this run: compact_context"
    assert run(agent._compact_conversation()) is False


def test_auto_compaction_respects_threshold_and_switch(make_agent, monkeypatch):
    calls = []

    async def fake_compact(self, *, trigger="manual"):
        calls.append(trigger)
        return True

    from agents.agent import Agent

    monkeypatch.setattr(Agent, "_compact_conversation", fake_compact)

    agent = make_agent(auto_compact_threshold=0.5)
    agent.last_input_token_count = int(agent.effective_window * 0.4)
    run(agent._check_and_compact())
    agent.last_input_token_count = int(agent.effective_window * 0.6)
    run(agent._check_and_compact())
    assert calls == ["auto"]

    disabled = make_agent(enable_folding=False, auto_compact_threshold=0.5)
    disabled.last_input_token_count = disabled.effective_window
    run(disabled._check_and_compact())
    assert calls == ["auto"]


def test_fold_limit_stops_further_compaction(make_agent):
    agent = make_agent(max_fold_limit=1)
    assert agent._fold_limit_reached() is False
    agent._fold_count = 1
    assert agent._fold_limit_reached() is True
    assert "limit reached" in run(agent._execute_compact_context_tool({}))
    assert "folding limit reached" in agent._build_fold_guidance_section()


def test_unknown_tools_are_rejected(make_agent):
    agent = make_agent(custom_tools=[])
    assert run(agent._execute_tool_call("read_file", {"file_path": "x"})) == "Tool not available in this run: read_file"


def test_online_evolution_switch(make_agent, monkeypatch):
    monkeypatch.delenv("CAIRN_AUTO_SKILL_EVOLUTION", raising=False)
    assert make_agent()._online_evolution_enabled() is True
    assert make_agent(enable_online_evolution=False)._online_evolution_enabled() is False
    monkeypatch.setenv("CAIRN_AUTO_SKILL_EVOLUTION", "0")
    assert make_agent()._online_evolution_enabled() is False
