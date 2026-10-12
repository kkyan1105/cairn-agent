import json

import pytest

from agents import session, session_memory


def test_save_load_and_latest(isolated_env, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    session.save_session("old", {"metadata": {"id": "old", "startTime": "2026-01-01T00:00:00"}})
    session.save_session("new", {"metadata": {"id": "new", "startTime": "2026-02-01T00:00:00"}})

    assert session.load_session("new")["metadata"]["id"] == "new"
    assert session.load_session("missing") is None
    assert session.get_latest_session_id() == "new"


def test_parse_folded_memory_extracts_json_from_noise():
    text = 'Here is the memory:\n```json\n{"episode_memory": {"task_description": "fix bug"}}\n```'
    memory = session_memory.parse_folded_memory(text)
    assert memory["episode_memory"]["task_description"] == "fix bug"
    assert memory["working_memory"]["next_actions"] == []
    assert memory["tool_memory"] == {"tools_used": [], "derived_rules": []}


def test_parse_folded_memory_normalizes_wrong_types():
    text = json.dumps({"episode_memory": {"key_events": "not a list"}, "tool_memory": []})
    memory = session_memory.parse_folded_memory(text)
    assert memory["episode_memory"]["key_events"] == []
    assert memory["tool_memory"]["tools_used"] == []


def test_fallback_and_format_produce_reinjectable_text():
    memory = session_memory.fallback_folded_memory("user asked to refactor main.py")
    text = session_memory.format_folded_memory(memory)
    assert text.startswith("<session-folded-memory>")
    assert "refactor main.py" in text


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
