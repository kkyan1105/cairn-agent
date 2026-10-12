from pathlib import Path

import pytest

from agents import skills


@pytest.fixture(autouse=True)
def fresh_skill_cache():
    skills.reset_skill_cache()
    yield
    skills.reset_skill_cache()


def write_skill(base: Path, dirname: str, frontmatter: str, body: str = "Follow the steps.") -> None:
    skill_dir = base / dirname
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}")


def test_discovers_user_and_project_skills(isolated_env):
    write_skill(Path.home() / ".cairn" / "skills", "user_skill", "name: user_skill\ndescription: from user")
    write_skill(isolated_env / ".cairn" / "skills", "project_skill", "name: project_skill\ndescription: from project")

    found = {s.name: s.source for s in skills.discover_skills()}
    assert found == {"user_skill": "user", "project_skill": "project"}


def test_user_skill_wins_over_project_skill_with_same_name(isolated_env):
    write_skill(Path.home() / ".cairn" / "skills", "dup", "name: dup\ndescription: user version")
    write_skill(isolated_env / ".cairn" / "skills", "dup", "name: dup\ndescription: project version")

    (skill,) = skills.discover_skills()
    assert skill.description == "user version"


def test_parses_frontmatter_options(isolated_env):
    write_skill(
        isolated_env / ".cairn" / "skills",
        "opts",
        "description: options\ncontext: fork\nuser-invocable: false\nallowed-tools: read_file, grep_search",
    )
    (skill,) = skills.discover_skills()
    assert skill.name == "opts"  # falls back to the directory name
    assert skill.context == "fork"
    assert skill.user_invocable is False
    assert skill.allowed_tools == ["read_file", "grep_search"]


def test_single_file_skills_are_ignored(isolated_env):
    base = isolated_env / ".cairn" / "skills"
    base.mkdir(parents=True)
    (base / "loose.md").write_text("---\nname: loose\n---\nbody")
    assert skills.discover_skills() == []


def test_prompt_substitutes_arguments(isolated_env):
    write_skill(isolated_env / ".cairn" / "skills", "greet", "name: greet\ndescription: greet", "Say hi to $ARGUMENTS.")
    skill = skills.get_skill_by_name("greet")
    assert skills.resolve_skill_prompt(skill, "Ada") == "Say hi to Ada."


def test_retrieval_ranks_the_matching_skill_first(isolated_env):
    base = isolated_env / ".cairn" / "skills"
    write_skill(base, "code_review", "name: code_review\ndescription: Review code for bugs and security issues")
    write_skill(base, "release_notes", "name: release_notes\ndescription: Write release notes from a changelog")

    hits = skills.retrieve_relevant_skills("please review this diff for bugs")
    assert hits and hits[0]["name"] == "code_review"
    assert all(h["name"] != "release_notes" for h in hits)


def test_retrieval_returns_nothing_for_empty_query(isolated_env):
    write_skill(isolated_env / ".cairn" / "skills", "any", "name: any\ndescription: anything")
    assert skills.retrieve_relevant_skills("   ") == []


def evolve_with_new_instructions(isolated_env, preserve_existing):
    write_skill(isolated_env / ".cairn" / "skills", "style", "name: style\ndescription: style guide", "Use tabs.")
    result = skills.evolve_skill(
        "style",
        lesson="Prefer f-strings",
        instructions="Use f-strings for formatting.",
        preserve_existing=preserve_existing,
    )
    assert result["ok"]
    return skills.get_skill_by_name("style")


def test_evolve_replaces_body_by_default(isolated_env):
    skill = evolve_with_new_instructions(isolated_env, preserve_existing=False)
    assert "Use f-strings" in skill.prompt_template
    assert "Use tabs." not in skill.prompt_template


def test_evolve_merge_keeps_existing_guidance(isolated_env):
    skill = evolve_with_new_instructions(isolated_env, preserve_existing=True)
    body = skill.prompt_template
    assert body.index("Use tabs.") < body.index("## Additional guidance from merge") < body.index("Use f-strings")


def test_evolve_creates_a_version_snapshot(isolated_env):
    skill = evolve_with_new_instructions(isolated_env, preserve_existing=True)
    history = list((isolated_env / ".cairn" / "skill-evolution").rglob("*.jsonl"))
    assert any("snapshot" in path.read_text() for path in history)
    assert skill.name == "style"
