# Cairn

[![tests](https://github.com/kkyan1105/cairn-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/kkyan1105/cairn-agent/actions/workflows/tests.yml)

Cairn is a self-evolving coding agent harness written in Python.

Like the stone stacks hikers leave to guide those who follow, Cairn turns what it learns from each task into reusable skills for the next one.

The model reasons and proposes tool calls. The harness does everything else: it checks permissions, executes tools, feeds results back to the model, manages context, and turns experience into reusable skills.


## Planned features

- **Agent loop**: model request → tool call → permission check → execution → result feedback.
- **Dual protocol**: works with both OpenAI-compatible and Anthropic-compatible APIs.
- **Built-in tools**: read, write, and edit files, search code, and run shell commands, with permission control.
- **Long-term memory**: per-project memory for preferences, decisions, and context.
- **Skills**: reusable `SKILL.md` playbooks that the agent can retrieve, invoke, and evolve.
- **MCP support**: connect external tools through the Model Context Protocol.
- **Sub-agents**: isolated agents for exploring, planning, and focused tasks.
- **Sessions**: save, resume, and compact conversations.

## Requirements

- Python 3.11+
- An OpenAI-compatible or Anthropic-compatible model endpoint

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
```

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project layout

```text
.cairn/skills/       # Project-level skills (example: code_review)
agents/
├── frontmatter.py   # YAML frontmatter parser for memory and skill files
├── memory.py        # Per-project long-term memory with relevance-based recall
├── prompt.py        # Dynamic system prompt assembly (environment, rules, memory, skills, agents)
├── session.py       # Session save, load, and resume
├── session_memory.py  # Fold long conversations into structured episode, working, and tool memory
├── skill_evolution.py  # Skill persistence, version snapshots, usage stats, and pruning
├── skills.py        # Skill loading, retrieval, and prompt rendering
├── subagent.py      # Built-in and custom sub-agent definitions with scoped tool sets
├── tools.py         # Built-in tools (file I/O, edit, search, shell) and permission checks
└── ui.py            # Terminal UI: colored output, spinner, tool display
tests/               # pytest suite
```
