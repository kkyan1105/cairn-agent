import json

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
