"""
File-based memory system.

Core ideas:
1. Each project has its own memory directory, named by a hash of the current working directory.
2. Each memory is a Markdown file with metadata stored in a YAML frontmatter header.
3. MEMORY.md is an auto-generated index that gives the system prompt a quick view of existing memories.
4. During a conversation, memory file headers are scanned cheaply first, then a side query has the model pick the relevant ones.
5. Recalled memories are injected into the current conversation as <system-reminder> blocks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter, format_frontmatter
from typing import Callable
# A side query is an async function: takes a system prompt and a user prompt, returns the model's text.
# Typed as Any to avoid imposing a complex Awaitable type constraint at runtime.
SideQueryFn = Callable[[str, str], Any]  # actually Awaitable[str]


VALID_TYPES = {"user", "feedback", "project", "reference"}
MAX_INDEX_LINES = 200       # Max lines of MEMORY.md kept before injecting into the system prompt.
MAX_INDEX_BYTES = 25000     # Max bytes of MEMORY.md kept before injecting into the system prompt.


class MemoryEntry:
    """A full memory entry, used for the /memory listing and CRUD operations."""

    __slots__ = ("name", "description", "type", "filename", "content")

    def __init__(self, name: str, description: str, type: str, filename: str, content: str):
        self.name = name
        self.description = description
        self.type = type
        self.filename = filename
        self.content = content




def _project_hash() -> str:
    """Generate a stable hash from the current working directory so memories of different projects stay isolated."""
    return hashlib.sha256(str(Path.cwd()).encode()).hexdigest()[:16]


def get_memory_dir() -> Path:
    """Return the current project's memory directory, creating it if missing."""
    d = Path.home() / ".cairn" / "projects" / _project_hash() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index_path() -> Path:
    """MEMORY.md is the index file for the current project's memory files."""
    return get_memory_dir() / "MEMORY.md"




def _slugify(text: str) -> str:
    """Turn a memory name into a short slug suitable for a filename."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    s = s.strip("_")
    return s[:40]




def list_memories() -> list[MemoryEntry]:
    """Read all memory files of the current project, sorted by modification time, newest first."""
    d = get_memory_dir()
    entries: list[MemoryEntry] = []
    for f in sorted(d.glob("*.md")):
        # MEMORY.md is the index, not an actual memory.
        if f.name == "MEMORY.md":
            continue
        try:
            result = parse_frontmatter(f.read_text())
            meta = result.meta
            # Files without name/type are not valid memories.
            if not meta.get("name") or not meta.get("type"):
                continue
            # Fall back to project for an invalid type so one bad file doesn't break the listing.
            t = meta["type"] if meta["type"] in VALID_TYPES else "project"
            entries.append(MemoryEntry(
                name=meta["name"],
                description=meta.get("description", ""),
                type=t,
                filename=f.name,
                content=result.body,
            ))
        except Exception:
            pass
    # Most recently modified memories first, for /memory display.
    entries.sort(key=lambda e: (d / e.filename).stat().st_mtime, reverse=True)
    return entries


def save_memory(name: str, description: str, type: str, content: str) -> str:
    """Save a memory and refresh the MEMORY.md index."""
    d = get_memory_dir()
    filename = f"{type}_{_slugify(name)}.md"
    text = format_frontmatter({"name": name, "description": description, "type": type}, content)
    (d / filename).write_text(text)
    _update_memory_index()
    return filename


def delete_memory(filename: str) -> bool:
    """Delete a memory by filename and refresh the index on success."""
    filepath = get_memory_dir() / filename
    if not filepath.exists():
        return False
    filepath.unlink()
    _update_memory_index()
    return True




def _update_memory_index() -> None:
    """Regenerate MEMORY.md from the current memory files."""
    memories = list_memories()
    lines = ["# Memory Index", ""]
    for m in memories:
        lines.append(f"- **[{m.name}]({m.filename})** ({m.type}) — {m.description}")
    _get_index_path().write_text("\n".join(lines))


def load_memory_index() -> str:
    """Read MEMORY.md and apply length limits before injecting it into the system prompt."""
    index_path = _get_index_path()
    if not index_path.exists():
        return ""
    content = index_path.read_text()
    lines = content.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... truncated, too many memory entries ...]"
    if len(content.encode()) > MAX_INDEX_BYTES:
        content = content[:MAX_INDEX_BYTES] + "\n\n[... truncated, index too large ...]"
    return content


# ─── Memory Header (lightweight scan) ──────────────────────

class MemoryHeader:
    """Lightweight memory summary containing only the metadata needed for recall filtering."""

    __slots__ = ("filename", "file_path", "mtime_ms", "description", "type")

    def __init__(self, filename: str, file_path: str, mtime_ms: float,
                 description: str | None, type: str | None):
        self.filename = filename
        self.file_path = file_path
        self.mtime_ms = mtime_ms
        self.description = description
        self.type = type


MAX_MEMORY_FILES = 200                    # Max number of memory files considered for recall.
MAX_MEMORY_BYTES_PER_FILE = 4096          # Max bytes of a single memory before injection.
MAX_SESSION_MEMORY_BYTES = 60 * 1024      # Max total memory bytes injected per session.


def scan_memory_headers() -> list[MemoryHeader]:
    """Quickly scan memory file headers without reading full bodies, for low-cost recall filtering."""
    d = get_memory_dir()
    headers: list[MemoryHeader] = []
    for f in d.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            stat = f.stat()
            raw = f.read_text()
            # Only parse the first 30 lines; the frontmatter normally fits at the top of the file.
            first30 = "\n".join(raw.split("\n")[:30])
            result = parse_frontmatter(first30)
            meta = result.meta
            t = meta.get("type")
            headers.append(MemoryHeader(
                filename=f.name,
                file_path=str(f),
                mtime_ms=stat.st_mtime * 1000,
                description=meta.get("description"),
                type=t if t in VALID_TYPES else None,
            ))
        except Exception:
            pass
    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """Format the list of memory summaries into a manifest for the side query to read."""
    lines = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        ts = datetime.fromtimestamp(h.mtime_ms / 1000, tz=timezone.utc).isoformat()
        if h.description:
            lines.append(f"- {tag}{h.filename} ({ts}): {h.description}")
        else:
            lines.append(f"- {tag}{h.filename} ({ts})")
    return "\n".join(lines)


# ─── Memory Age / Freshness ────────────────────────────────

def memory_age(mtime_ms: float) -> str:
    """Convert a modification time into a human-friendly relative time."""
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_warning(mtime_ms: float) -> str:
    """Old memories may be stale; remind the model to check the current code when injecting them."""
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days <= 1:
        return ""
    return (f"This memory is {days} days old. Memories are point-in-time observations, "
            "not live state — claims about code behavior may be outdated. "
            "Verify against current code before asserting as fact.")



SELECT_MEMORIES_PROMPT = """You are selecting memories that will be useful to an AI coding assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" array of filenames for the memories that will clearly be useful (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful, do not include it.
- If no memories would clearly be useful, return an empty array."""


class RelevantMemory:
    """A full memory that was recalled and is ready to be injected into the conversation."""

    __slots__ = ("path", "content", "mtime_ms", "header")

    def __init__(self, path: str, content: str, mtime_ms: float, header: str):
        self.path = path
        self.content = content
        self.mtime_ms = mtime_ms
        self.header = header

    @property
    def size(self) -> int:
        """Bytes used by this memory's content, for the Agent's per-session injection budget."""
        return len(self.content.encode())


async def select_relevant_memories(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
) -> list[RelevantMemory]:
    """
    Select the memories most relevant to the current query from the memory directory.

    Flow:
    1. Scan memory headers to avoid reading every body up front.
    2. Exclude memories already injected in this session to avoid polluting the context with duplicates.
    3. Pass the candidate summaries to a side query and let the model choose up to 5 filenames.
    4. Read the bodies of the selected memories, truncating oversized files.
    5. Wrap them as RelevantMemory for the injection logic downstream.
    """

    # Scan the header info of all memory files.
    headers = scan_memory_headers()

    if not headers:
        return []

    # Exclude memories that have already been surfaced.
    candidates = [h for h in headers if h.file_path not in already_surfaced]
    if not candidates:
        return []

    # The manifest is the list of candidate summaries shown to the side query.
    manifest = format_memory_manifest(candidates)

    # Call side_query so the model picks relevant memories based on filenames and descriptions.
    try:
        text = await side_query(
            SELECT_MEMORIES_PROMPT,
            f"Query: {query}\n\nAvailable memories:\n{manifest}",
        )

        # The side query may return explanatory text; extract only the JSON object.
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []

        # Parse the JSON to get the selected memory filenames.
        parsed = json.loads(match.group(0))
        selected_filenames = set(parsed.get("selected_memories", []))
        # Filter candidate memories by filename, taking at most 5.
        selected = [h for h in candidates if h.filename in selected_filenames][:5]

        result: list[RelevantMemory] = []
        for h in selected:
            # Read the content of each selected memory file.
            content = Path(h.file_path).read_text()
            # Truncate oversized files so a single memory doesn't take up too much context.
            if len(content.encode()) > MAX_MEMORY_BYTES_PER_FILE:
                content = content[:MAX_MEMORY_BYTES_PER_FILE] + "\n\n[... truncated, memory file too large ...]"

            # Build a header from the memory's modification time; old memories get a freshness warning.
            freshness = memory_freshness_warning(h.mtime_ms)
            header_text = (
                f"{freshness}\n\nMemory: {h.file_path}:" if freshness
                else f"Memory (saved {memory_age(h.mtime_ms)}): {h.file_path}:"
            )
            # Return the RelevantMemory list, later formatted as <system-reminder>.
            result.append(RelevantMemory(
                path=h.file_path, content=content,
                mtime_ms=h.mtime_ms, header=header_text,
            ))
        return result
    except Exception as e:
        # A recall failure must not affect the main conversation; cancellation errors are silenced.
        if "cancel" in str(e).lower():
            return []
        print(f"[memory] semantic recall failed: {e}")
        return []


class MemoryPrefetch:
    """Wraps the async memory recall task so the Agent main loop can poll it."""

    def __init__(self, task: asyncio.Task):
        self.task = task
        # consumed marks whether the result has been injected already, so the same task result isn't reused.
        self.consumed = False

    @property
    def settled(self) -> bool:
        """Whether the task has finished."""
        return self.task.done()


def start_memory_prefetch(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
    session_memory_bytes: int,
) -> MemoryPrefetch | None:
    """
    Start memory recall asynchronously, ahead of the main model reply.

    Returns not the memory contents but a MemoryPrefetch handle.
    The Agent main loop later checks whether the task is done and, once it is, injects the memories into the current message.
    """

    # Only multi-word input triggers memory prefetch, so short commands don't each cost a side query.
    if not re.search(r"\s", query.strip()):
        return None

    # This session's memory usage must stay within budget.
    if session_memory_bytes >= MAX_SESSION_MEMORY_BYTES:
        return None

    # The memory directory must actually contain memory files.
    d = get_memory_dir()
    has_memories = any(f.suffix == ".md" and f.name != "MEMORY.md" for f in d.iterdir())
    if not has_memories:
        return None

    # Once the checks pass, create an async task so recall runs in parallel with the main model request.
    task = asyncio.create_task(
        select_relevant_memories(query, side_query, already_surfaced)
    )
    return MemoryPrefetch(task)


def format_memories_for_injection(memories: list[RelevantMemory]) -> str:
    """Wrap recalled memories in a system-reminder for injection into the user message."""
    parts = []
    for m in memories:
        parts.append(f"<system-reminder>\n{m.header}\n\n{m.content}\n</system-reminder>")
    return "\n\n".join(parts)


def build_memory_prompt_section() -> str:
    """
    Generate the Memory System instructions injected into the system prompt.

    These instructions tell the model:
    - where memory files are stored;
    - which memory types exist;
    - how to save a memory via write_file;
    - what should not be saved;
    - which memories are currently in the MEMORY.md index.
    """
    index = load_memory_index()
    memory_dir = str(get_memory_dir())

    return f"""# Memory System

You have a persistent, file-based memory system at `{memory_dir}`.

## Memory Types
- **user**: User's role, preferences, knowledge level
- **feedback**: Corrections and guidance from the user (include Why + How to apply)
- **project**: Ongoing work, goals, deadlines, decisions
- **reference**: Pointers to external resources (URLs, tools, dashboards)

## How to Save Memories
Use the write_file tool to create a memory file with YAML frontmatter:

```markdown
---
name: memory name
description: one-line description
type: user|feedback|project|reference
---
Memory content here.
```

Save to: `{memory_dir}/`
Filename format: `{{type}}_{{slugified_name}}.md`

The MEMORY.md index is auto-updated when you write to the memory directory — do NOT update it manually.

## What NOT to Save
- Code patterns or architecture (read the code instead)
- Git history (use git log)
- Anything already in CLAUDE.md
- Ephemeral task details

## When to Recall
When the user asks you to remember or recall, or when prior context seems relevant.
{chr(10) + "## Current Memory Index" + chr(10) + index if index else chr(10) + "(No memories saved yet.)"}"""
