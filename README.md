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

## How the agent loop works

1. Build the system prompt from environment info, git context, project rules, memory, skills, and the available sub-agents.
2. Send the conversation to the model and stream the response.
3. For each tool call, check permissions, then execute it. Read-only tools run concurrently.
4. Feed the tool results back to the model and repeat until it answers without calling a tool.
5. Save the session automatically. Transient API errors are retried with exponential backoff.

### Context management

Long sessions are kept within the model's context window by a multi-level pipeline:

| Level | Trigger | What it does |
|---|---|---|
| Budgeting | Context over 50% full | Truncates oversized tool results, keeping the head and tail |
| Snipping | Context over 60% full | Replaces stale tool results with a placeholder, keeping the 3 most recent and the latest read of each file |
| Microcompaction | 5 minutes idle | Clears old tool results that have already been used |
| Folding | Context over 70% full, or `/compact` | Summarizes history into structured episode, working, and tool memory |

Tool results over 30 KB are saved to `~/.cairn/tool-results/`. Only a preview stays in context.

## Permissions

Every tool call goes through a permission check before it runs. Read-only tools are always allowed. Writing a new file, running a dangerous shell command (such as `rm`, `sudo`, or `git push`), and changing skills all ask for confirmation first.

Permission modes:

| Mode | Behavior |
|---|---|
| `default` | Ask before risky actions |
| `plan` | Read-only. Edits and shell commands are blocked, except writing the plan file |
| `acceptEdits` | File edits are allowed without asking |
| `dontAsk` | Anything that would need confirmation is denied |
| `bypassPermissions` | Everything is allowed |

You can also add allow and deny rules in `~/.cairn/settings.json` or `.cairn/settings.json`. Deny rules are checked first. A trailing `*` matches by prefix.

```json
{
  "permissions": {
    "allow": ["run_shell(npm test*)"],
    "deny": ["write_file(secrets/*)"]
  }
}
```

## MCP servers

Cairn includes its own MCP client, which talks JSON-RPC over stdio. Configure servers in `.mcp.json` at the project root. You can also put them under `mcpServers` in `~/.cairn/settings.json` (global) or `.cairn/settings.json` (project). Later sources override earlier ones when names collide.

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```

Each discovered tool is exposed to the model as `mcp__<server>__<tool>`. A server that fails to start or respond within 15 seconds is skipped, and the other servers keep working.

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project layout

```text
.cairn/skills/       # Project-level skills (example: code_review)
agents/
├── agent.py         # Agent runtime: model calls, tool dispatch, context compression
├── frontmatter.py   # YAML frontmatter parser for memory and skill files
├── mcp_client.py    # Stdio JSON-RPC MCP client and tool routing
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
