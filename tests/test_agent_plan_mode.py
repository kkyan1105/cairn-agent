import asyncio
from pathlib import Path

import pytest


def run(coro):
    return asyncio.run(coro)


def test_toggle_plan_mode_round_trip(make_agent):
    agent = make_agent(permission_mode="acceptEdits")

    assert agent.toggle_plan_mode() == "plan"
    assert agent._plan_file_path.endswith(f"plan-{agent.session_id}.md")
    assert "Plan Mode Active" in agent._system_prompt

    assert agent.toggle_plan_mode() == "acceptEdits"
    assert agent._plan_file_path is None
    assert "Plan Mode Active" not in agent._system_prompt


def test_plan_mode_tools_update_openai_system_message(make_agent):
    agent = make_agent("openai")
    run(agent._execute_plan_mode_tool("enter_plan_mode"))
    assert "Plan Mode Active" in agent._openai_messages[0]["content"]
    assert run(agent._execute_plan_mode_tool("enter_plan_mode")) == "Already in plan mode."


def test_exit_without_approval_restores_previous_mode(make_agent):
    agent = make_agent()
    run(agent._execute_plan_mode_tool("enter_plan_mode"))
    result = run(agent._execute_plan_mode_tool("exit_plan_mode"))
    assert agent.permission_mode == "default"
    assert "Exited plan mode" in result
    assert run(agent._execute_plan_mode_tool("exit_plan_mode")) == "Not in plan mode."


def enter_plan_with_file(agent, text="1. Do the thing"):
    run(agent._execute_plan_mode_tool("enter_plan_mode"))
    Path(agent._plan_file_path).write_text(text)


@pytest.mark.parametrize(
    ("choice", "expected_mode"),
    [("execute", "acceptEdits"), ("clear-and-execute", "acceptEdits"), ("manual-execute", "default")],
)
def test_approved_plan_switches_mode_and_includes_plan(make_agent, choice, expected_mode):
    agent = make_agent()
    seen = []

    async def approve(plan_content):
        seen.append(plan_content)
        return {"choice": choice}

    agent.set_plan_approval_fn(approve)
    enter_plan_with_file(agent)

    result = run(agent._execute_plan_mode_tool("exit_plan_mode"))

    assert seen == ["1. Do the thing"]
    assert agent.permission_mode == expected_mode
    assert agent._plan_file_path is None
    assert "1. Do the thing" in result


def test_keep_planning_stays_in_plan_mode_with_feedback(make_agent):
    agent = make_agent()

    async def reject(_plan_content):
        return {"choice": "keep-planning", "feedback": "Add tests"}

    agent.set_plan_approval_fn(reject)
    enter_plan_with_file(agent)

    result = run(agent._execute_plan_mode_tool("exit_plan_mode"))

    assert agent.permission_mode == "plan"
    assert "Add tests" in result
