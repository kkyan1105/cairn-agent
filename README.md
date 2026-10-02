# Bear Agent

Bear Agent is a self-evolving coding agent harness written in Python.

The model reasons and proposes tool calls. The harness does everything else: it checks permissions, executes tools, feeds results back to the model, manages context, and turns experience into reusable skills.

> **Status:** early development. The project is being built module by module, and this README will grow with it.

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

## Project layout

```text
agents/
├── frontmatter.py   # YAML frontmatter parser for memory and skill files
└── ui.py            # Terminal UI: colored output, spinner, tool display
```
