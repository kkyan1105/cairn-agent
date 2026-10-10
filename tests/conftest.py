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


@pytest.fixture
def make_agent(isolated_env):
    """Build an Agent without network access, for the Anthropic or OpenAI backend."""
    from agents.agent import Agent

    def factory(backend="anthropic", **kwargs):
        if backend == "openai":
            return Agent(model="gpt-4o", api_base="http://localhost:1/v1", api_key="test", **kwargs)
        return Agent(model="claude-sonnet-4-6", api_key="test", **kwargs)

    return factory
