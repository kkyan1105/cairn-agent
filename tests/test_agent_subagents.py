import asyncio

import pytest

from agents.agent import Agent


@pytest.fixture
def captured(monkeypatch):
    created = []

    async def fake_run_once(self, prompt):
        created.append(self)
        return {"text": f"done: {prompt}", "tokens": {"input": 10, "output": 5}}

    monkeypatch.setattr(Agent, "run_once", fake_run_once)
    return created


def run(coro):
    return asyncio.run(coro)


def test_sub_agent_inherits_permissions_credentials_and_remaining_budget(make_agent, captured):
    async def confirm(_message):
        return True

    parent = make_agent("openai", permission_mode="default", confirm_fn=confirm, max_cost_usd=1.0, max_turns=10)
    parent.current_turns = 4

    result = run(parent._execute_agent_tool({"type": "general", "prompt": "list files"}))

    (child,) = captured
    assert result == "done: list files"
    assert child.permission_mode == "default"
    assert child.confirm_fn is confirm
    assert child._api_base == "http://localhost:1/v1" and child._api_key == "test"
    assert child.max_turns == 6
    assert child.max_cost_usd == pytest.approx(1.0)
    assert not {"enter_plan_mode", "exit_plan_mode", "agent"} & {t["name"] for t in child.tools}
    assert parent.total_input_tokens == 10 and parent.total_output_tokens == 5


def test_sub_agent_is_skipped_when_budget_is_exhausted(make_agent, captured):
    parent = make_agent(max_turns=2)
    parent.current_turns = 2
    assert run(parent._execute_agent_tool({"prompt": "anything"})).startswith("Sub-agent skipped")
    assert captured == []


def test_plan_mode_parent_creates_plan_mode_child(make_agent, captured):
    parent = make_agent(permission_mode="plan")
    run(parent._execute_agent_tool({"type": "explore", "prompt": "look around"}))
    assert captured[0].permission_mode == "plan"
