# Cairn

[![tests](https://github.com/kkyan1105/cairn-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/kkyan1105/cairn-agent/actions/workflows/tests.yml)

Cairn is a self-evolving coding agent harness written in Python.

Like the stone stacks hikers leave to guide those who follow, Cairn turns what it learns from each task into reusable skills for the next one.

The model reasons and proposes tool calls. The harness does everything else: it checks permissions, executes tools, feeds results back to the model, manages context, and turns experience into reusable skills.


## Features

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
pip install -e .
cp .env.example .env   # then fill in your API key
```

## Usage

```bash
cairn                                   # interactive REPL
cairn "find the bug in utils.py and fix it"   # one-shot task
cairn --plan "how would you refactor the parser?"   # read-only plan, no changes
cairn --accept-edits --max-cost 0.50 "add tests for the CLI"
cairn --resume                          # continue the last session
```

### Docker

The image includes Node.js and Chromium, so the MCP servers in `.mcp.json` work out of the box.

```bash
docker build -t cairn .
docker run -it --rm --env-file .env -v "$PWD":/workspace cairn
```

Run `cairn --help` for all flags and REPL commands such as `/plan`, `/compact`, `/skills`, `/skill-eval`, `/skill-candidates`, `/trust-project`, and `/cleanup-state`. You can also run it without installing it: `python -m agents.main`.

## How the agent loop works

1. Build the system prompt from environment info, git context, project rules, memory, skills, and the available sub-agents.
2. Send the conversation to the model and stream the response.
3. Once the full response has arrived, check permissions for each tool call, then execute it. With OpenAI-compatible backends, consecutive read-only tools run concurrently.
4. Feed the tool results back to the model and repeat until it answers without calling a tool.
5. Save the session automatically. Transient API errors are retried with exponential backoff.

Every tool call is wrapped in a durable checkpoint. If a run is interrupted, a call whose outcome is unknown is reported to the model as "status unknown, inspect before retrying" and is never replayed automatically. Text from a failed or retried stream attempt never reaches the final answer.

### Context management

Long sessions are kept within the model's context window by a multi-level pipeline:

| Level | Trigger | What it does |
|---|---|---|
| Budgeting | Context over 50% full | Truncates oversized tool results, keeping the head and tail |
| Snipping | Context over 60% full | Replaces stale tool results with a placeholder, keeping the 3 most recent and the latest read of each file |
| Microcompaction | 5 minutes idle | Clears old tool results that have already been used |
| Folding | Context over 70% full, or `/compact` | Summarizes history into structured episode, working, and tool memory |

`read_file` returns at most 1000 lines and 24 KiB per call and pages through larger files with `offset` and `limit`. File writes are checked against the version that was read (device, inode, size, and timestamps) and committed atomically, so an external edit in between is never silently overwritten. Tool results over 30 KB are saved to `~/.cairn/tool-results/`. Only a preview stays in context.

Folding is configurable per agent:

| Option | Default | Effect |
|---|---|---|
| `enable_folding` | `True` | Turn folding off entirely, including the `compact_context` tool |
| `auto_compact_threshold` | `0.70` | Fraction of the context window that triggers automatic folding |
| `max_fold_limit` | none | Maximum number of folds per session |
| `fold_memory_mode` | `single_json` | `three_parallel` generates episode, working, and tool memory with three concurrent calls |

### Sub-agents

Sub-agents run with isolated context but inherit the parent's permission mode, confirmation prompt, and remaining cost and turn budget. A sub-agent cannot switch plan mode, and by default it cannot spawn further sub-agents.

## Self-evolving skills

Skills are `SKILL.md` playbooks stored in `.cairn/skills/` (project) or `~/.cairn/skills/` (user). For each request, Cairn ranks the available skills against the request with BM25 over each skill's name, description, and when-to-use. The most relevant skills are offered to the model.

After each reply, an online learning loop runs in the background. It never edits an active skill directly:

1. **Extract.** A side model reads the latest exchange and the user's follow-up feedback, and proposes at most one reusable skill candidate. User turns are the evidence. One-off details, secrets, and assistant-only claims are rejected.
2. **Stage.** A skill-set manager decides whether to **add** a new skill, **merge** into an existing one, or **discard** the candidate. Accepted proposals are staged in `.cairn/skill-evolution/candidates/` with the final body, the write parameters, and a hash of the skill they were based on.
3. **Evaluate.** `/skill-eval` replays reviewed past tasks with the current skill and with candidate variants side by side, under a bounded call budget. A candidate becomes the champion only if it beats the baseline by a margin.
4. **Publish.** `/skill-apply <id>` publishes a candidate whose exact snapshot passed evaluation, or one you approve with `--manual-review`. Publishing is refused if the candidate or the active skill changed in the meantime. `/skill-rollback <skill>` restores the previous version.
5. **Track.** For each retrieved skill, a judge records whether it was relevant and actually used. Skills that keep being retrieved but are never used can be pruned.

Control it with environment variables:

```env
CAIRN_AUTO_SKILL_EVOLUTION=1      # set to 0 to disable
CAIRN_AUTO_SKILL_TARGET=project   # or "user" to share learned skills across projects
```

## Permissions

Every tool call goes through a permission check before it runs. Read-only tools are always allowed. Every shell command, file write, file edit, MCP tool call, and skill change asks for confirmation, each time. Shell commands are never auto-approved by a blacklist, because a command can write files through an interpreter or a pipe.

Permission modes:

| Mode | Behavior |
|---|---|
| `default` | Ask before risky actions |
| `plan` | Read-only. Edits, shell commands, and MCP tools are blocked, except writing the plan file |
| `acceptEdits` | File edits are allowed without asking; shell commands still ask |
| `dontAsk` | Anything that would need confirmation is denied |
| `bypassPermissions` | Everything is allowed |

You can also add allow and deny rules in `~/.cairn/settings.json` or `.cairn/settings.json`. Deny rules are checked first and match every alias of a path (relative, absolute, `..`, or a symlink). Plan mode restrictions apply even when an allow rule matches. A trailing `*` matches by prefix. Allow rules in a project's own settings, and MCP servers it defines, only take effect after you run `/trust-project`. Editing those files revokes the trust.

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

No server process starts without your approval. Each discovered tool is exposed to the model as `mcp__<server>__<tool>`, and each call times out after 60 seconds. A server that fails to start or respond within 15 seconds is skipped, and the other servers keep working. If a server crashes mid-call, the call fails right away instead of hanging.

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project layout

```text
.cairn/skills/            # Project-level skills (example: code_review)
agents/
├── main.py               # CLI entry point and interactive REPL
├── agent.py              # Agent runtime shared by the Anthropic and OpenAI backends
├── config.py             # Startup configuration: keys, endpoints, model, limits, pricing
├── tool_runtime.py       # Tool validation and permission checks for both protocols
├── checkpoints.py        # Durable records around tool calls for safe recovery
├── streaming.py          # Keeps failed stream attempts out of answers
├── context_runtime.py    # Context preflight and bounded memory recall
├── tool_results.py       # Budgeting, snipping, and idle clearing of old tool results
├── session_memory.py     # Folding long conversations into structured memory
├── tools.py              # File, search, and shell tools with permission rules
├── tool_schemas.py       # Tool definitions exposed to the model
├── tool_status.py        # Tool results with an explicit failure flag
├── result_storage.py     # Saves oversized tool output once, keeps previews in context
├── project_trust.py      # Trust records for project-level settings
├── mcp_client.py         # Stdio JSON-RPC MCP client
├── memory.py             # Per-project long-term memory
├── session.py            # Session save, resume, and titles
├── skills.py             # Skill loading, retrieval, and prompt rendering
├── skill_evolution.py    # Skill writes, version snapshots, usage stats, pruning
├── online_skill_evolution.py  # Extracts skill candidates from live feedback
├── skill_candidates.py   # Staged proposals, publishing, and rollback
├── online_skill_eval.py  # Reviewed replay evaluation and champion selection
├── background_jobs.py    # Bounded queue for background skill learning
├── provenance_reader.py  # Incremental index over the evolution log
├── maintenance.py        # Safe cleanup of old local state
├── storage.py            # Atomic writes and cross-process locks
├── validation.py         # Local JSON Schema validation
├── prompt.py             # System prompt assembly
├── subagent.py           # Built-in and custom sub-agents
├── frontmatter.py        # YAML frontmatter parser
└── ui.py                 # Terminal UI
tests/                    # pytest suite
```
