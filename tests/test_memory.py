from agents import memory


def test_memory_dir_is_isolated_per_project(isolated_env, tmp_path, monkeypatch):
    first = memory.get_memory_dir()
    other = tmp_path / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    assert memory.get_memory_dir() != first


def test_save_list_and_index(isolated_env):
    filename = memory.save_memory("Prefers tabs", "Indentation style", "user", "Use tabs.")
    assert filename == "user_prefers_tabs.md"

    entries = memory.list_memories()
    assert [(e.name, e.type, e.content) for e in entries] == [("Prefers tabs", "user", "Use tabs.")]

    index = (memory.get_memory_dir() / "MEMORY.md").read_text()
    assert "[Prefers tabs](user_prefers_tabs.md)" in index


def test_delete_memory_updates_index(isolated_env):
    filename = memory.save_memory("Temp", "to delete", "project", "x")
    assert memory.delete_memory(filename) is True
    assert memory.list_memories() == []
    assert "Temp" not in (memory.get_memory_dir() / "MEMORY.md").read_text()
    assert memory.delete_memory(filename) is False


def test_files_without_required_meta_are_skipped(isolated_env):
    (memory.get_memory_dir() / "broken.md").write_text("---\ndescription: no name\n---\nbody")
    assert memory.list_memories() == []


def test_invalid_type_falls_back_to_project(isolated_env):
    (memory.get_memory_dir() / "odd.md").write_text("---\nname: odd\ntype: unknown\n---\nbody")
    assert memory.list_memories()[0].type == "project"
