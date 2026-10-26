import asyncio
import json

import pytest

from agents import online_skill_evolution as evo
from agents import skills


@pytest.fixture(autouse=True)
def fresh_skill_cache():
    skills.reset_skill_cache()
    yield
    skills.reset_skill_cache()


CANDIDATE = {
    "name": "concise_commit_messages",
    "description": "Write short imperative commit messages",
    "when_to_use": "When the user asks for a commit message",
    "instructions": "Use the imperative mood and keep the subject under 60 characters.",
    "evidence": "User: keep commit subjects short and imperative from now on",
    "tags": "git, style",
}


def scripted(extract_reply, manage_reply):
    """Side query that answers the extractor and the skill-set manager differently."""
    calls = []

    async def side_query(system, payload):
        calls.append((system, json.loads(payload)))
        if "Skill Extractor" in system:
            return extract_reply
        return manage_reply

    return side_query, calls


def run(coro):
    return asyncio.run(coro)


def messages():
    return [
        {"role": "user", "content": "Write a commit message"},
        {"role": "assistant", "content": "Added a new feature to the parser module"},
        {"role": "user", "content": "Keep commit subjects short and imperative from now on"},
    ]


def test_extractor_parses_candidate_from_noisy_json():
    side_query, _ = scripted("Here you go: " + json.dumps({"skills": [CANDIDATE]}), "")
    candidate = run(evo.extract_online_skill_candidate(messages=messages(), side_query=side_query))
    assert candidate.name == "concise_commit_messages"
    assert candidate.tags == ["git", "style"]


@pytest.mark.parametrize("reply", ['{"skills": []}', "not json", json.dumps({"skills": [{"name": "x"}]})])
def test_extractor_returns_none_for_weak_or_invalid_output(reply):
    side_query, _ = scripted(reply, "")
    assert run(evo.extract_online_skill_candidate(messages=messages(), side_query=side_query)) is None


def test_ingest_adds_a_new_project_skill(isolated_env):
    side_query, calls = scripted(json.dumps({"skills": [CANDIDATE]}), '{"action": "add", "reason": "new capability"}')

    result = run(evo.online_ingest(messages=messages(), side_query=side_query))

    assert result["action"] == "add" and result["ok"]
    skill_file = isolated_env / ".cairn" / "skills" / "concise_commit_messages" / "SKILL.md"
    assert "imperative mood" in skill_file.read_text()
    assert len(calls) == 2


def test_ingest_merge_writes_merged_body_and_snapshots_the_old_one(isolated_env):
    skill_dir = isolated_env / ".cairn" / "skills" / "commit_style"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: commit_style\ndescription: Commit conventions\n---\n\nReference the issue number.")
    decision = {"action": "merge", "target_skill": "commit_style",
                "merged_instructions": "Reference the issue number. Keep subjects under 60 characters."}
    side_query, _ = scripted(json.dumps({"skills": [CANDIDATE]}), json.dumps(decision))

    result = run(evo.online_ingest(messages=messages(), side_query=side_query))

    assert result["action"] == "merge" and result["ok"]
    body = (skill_dir / "SKILL.md").read_text()
    assert "Reference the issue number." in body and "under 60 characters" in body
    history = "".join(p.read_text() for p in (isolated_env / ".cairn" / "skill-evolution").rglob("*.jsonl"))
    assert '"event": "snapshot"' in history


def test_ingest_respects_denied_write_confirmation(isolated_env):
    side_query, _ = scripted(json.dumps({"skills": [CANDIDATE]}), '{"action": "add"}')

    async def deny(_summary):
        return False

    result = run(evo.online_ingest(messages=messages(), side_query=side_query, confirm_write=deny))

    assert result["action"] == "add_denied"
    assert not (isolated_env / ".cairn" / "skills").exists()


def test_ingest_records_failures_instead_of_raising(isolated_env):
    async def broken(_system, _payload):
        raise RuntimeError("model unavailable")

    result = run(evo.online_ingest(messages=messages(), side_query=broken))
    assert result == {"ok": False, "action": "failed", "error": "model unavailable"}


def test_usage_judgments_with_and_without_a_model():
    hits = [{"name": "commit_style", "score": 0.4, "source": "project"}]

    heuristic = run(evo.judge_retrieved_skill_usage(hits=hits, user_message="q", assistant_text="Following commit_style..."))
    assert heuristic[0]["used"] is True and heuristic[0]["reason"] == "heuristic fallback"

    async def judge(_system, _payload):
        return '{"judgments": [{"name": "commit_style", "relevant": true, "used": false, "reason": "ignored"}]}'

    judged = run(evo.judge_retrieved_skill_usage(hits=hits, user_message="q", assistant_text="a", side_query=judge))
    assert judged[0]["relevant"] is True and judged[0]["used"] is False
