import json

import pytest

from agents import session, session_memory


def test_save_load_and_latest(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    cwd = str(isolated_env)
    session.save_session("old", {"metadata": {"id": "old", "cwd": cwd, "startTime": "2026-01-01T00:00:00"}})
    session.save_session("new", {"metadata": {"id": "new", "cwd": cwd, "startTime": "2026-02-01T00:00:00"}})
    session.save_session("elsewhere", {"metadata": {"id": "elsewhere", "cwd": str(tmp_path), "startTime": "2026-03-01T00:00:00"}})

    assert session.load_session("new")["metadata"]["id"] == "new"
    assert session.load_session("missing") is None
    # Resume only considers sessions started in the current project.
    assert session.get_latest_session_id() == "new"
    assert session.load_session("../escape") is None


FULL_MEMORY = {
    "episode_memory": {"task_description": "fix bug", "key_events": [], "current_progress": ""},
    "working_memory": {"immediate_goal": "", "current_challenges": "", "next_actions": []},
    "tool_memory": {"tools_used": [], "derived_rules": []},
}


def test_parse_folded_memory_extracts_json_from_noise():
    text = "Here is the memory:\n```json\n" + json.dumps(FULL_MEMORY) + "\n```"
    assert session_memory.parse_folded_memory(text)["episode_memory"]["task_description"] == "fix bug"


@pytest.mark.parametrize(
    "memory",
    [
        {"episode_memory": FULL_MEMORY["episode_memory"]},
        dict(FULL_MEMORY, episode_memory=dict(FULL_MEMORY["episode_memory"], key_events="not a list")),
        dict(FULL_MEMORY, episode_memory=dict(FULL_MEMORY["episode_memory"], task_description="")),
    ],
)
def test_parse_folded_memory_rejects_incomplete_or_empty_state(memory):
    with pytest.raises(ValueError):
        session_memory.parse_folded_memory(json.dumps(memory))


def test_format_produces_reinjectable_text():
    text = session_memory.format_folded_memory(FULL_MEMORY)
    assert text.startswith("<session-folded-memory>")
    assert "fix bug" in text


def test_parse_fold_component_accepts_bare_or_wrapped_objects():
    bare = json.dumps({"immediate_goal": "fix test", "current_challenges": "", "next_actions": []})
    wrapped = json.dumps({"working_memory": json.loads(bare)})
    assert session_memory.parse_fold_component("working_memory", bare)["immediate_goal"] == "fix test"
    assert session_memory.parse_fold_component("working_memory", wrapped)["immediate_goal"] == "fix test"


def test_parse_fold_component_rejects_missing_fields_and_unknown_names():
    with pytest.raises(ValueError, match="missing fields"):
        session_memory.parse_fold_component("tool_memory", json.dumps({"tools_used": []}))
    with pytest.raises(ValueError, match="unknown fold component"):
        session_memory.parse_fold_component("other_memory", "{}")


def test_every_component_has_a_system_prompt():
    assert set(session_memory.FOLD_COMPONENT_SYSTEMS) == {"episode_memory", "working_memory", "tool_memory"}
