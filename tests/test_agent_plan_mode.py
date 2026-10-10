import asyncio


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
