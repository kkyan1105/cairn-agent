import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Run a test with a throwaway HOME and working directory."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return project
