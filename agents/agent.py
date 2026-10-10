#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Awaitable, Any

import anthropic
import openai

from agents.mcp_client import McpManager
from agents.memory import MemoryPrefetch, start_memory_prefetch, format_memories_for_injection
from agents.prompt import build_system_prompt
from agents.session_memory import (
    FOLD_SESSION_MEMORY_SYSTEM,
    build_anthropic_transcript,
    build_folding_user_prompt,
    build_openai_transcript,
    fallback_folded_memory,
    format_folded_memory,
    parse_folded_memory,
)
from agents.session import save_folded_session_memory, save_session
from agents.subagent import get_sub_agent_config
from agents.tools import ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS, check_permission, \
    get_active_tool_definitions
from agents.ui import print_info, print_divider, print_assistant_text, print_sub_agent_start, print_sub_agent_end, \
    start_spinner, stop_spinner, print_cost, print_tool_call, print_tool_result, print_confirmation, print_retry, \
    print_error


# Exponential backoff retry


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000
}

def _get_context_windows(model:str)->int:
    return MODEL_CONTEXT.get(model, 200000)


# Multi-level compression constants
SNIP_THRESHOLD = 0.60
AUTO_COMPACT_THRESHOLD = 0.70
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes

KEEP_RECENT_RESULTS = 3



def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384

# Convert tools to the OpenAI format
def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class Agent:
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 thinking: bool=False,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 is_sub_agent: bool=False,):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        self._custom_system_prompt = custom_system_prompt
        self.effective_window=_get_context_windows(model) -20000
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0


        self._aborted = False
        # Stores async tasks
        self._current_task:asyncio.Task | None = None
        # Permission allowlist
        self._confirmed_paths: set[str] = set()


        # Variables for Plan Mode state
        self._pre_plan_mode: str | None=None
        self._plan_file_path: str | None=None
        self._plan_approval_fn : Callable[[str], Awaitable[dict]] | None=None
        self._context_cleared : bool=False

        # Thinking mode
        self._thinking_mode = self._resolve_thinking_mode()

        # Output cache for sub-agents
        self._output_buffer: list[str] | None=None
        self._turn_output_buffer: list[str] | None = None

        # Read before edit
        self._read_file_state: dict[str, float] ={}

        # MCP integration
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        # Memory recall
        # Memories the agent has already surfaced
        self._already_surfaced_memories: set[str] = set()
        # Bytes used by the current session
        self._session_memory_bytes = 0

        # Distinguishes historical messages
        self._anthropic_messages: list[str] = []
        self._openai_messages: list[str] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        self._background_skill_tasks: set[asyncio.Task] = set()
        self._folded_session_memories: list[dict[str, Any]] = []
        self._fold_last_time: float = 0.0
        self._fold_count: int = 0
        self._tool_error_streak: int = 0
        self._same_tool_repeat_count: int = 0
        self._last_tool_name: str = ""

        # Build the system prompt
        self._base_system_prompt = custom_system_prompt or build_system_prompt()

        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        # Initialize the LLM client
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

        self._refresh_runtime_system_prompt()

    # Determine the thinking mode for the model
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._model_supports_adaptive_thinking():
            return "adaptive"
        return "enabled"

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False
    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    # Generate the absolute path of the Markdown file used to save the AI's plan.
    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".cairn" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

    # Plan Mode Active

    Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

    ## Plan File: {self._plan_file_path}
    Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

    ## Workflow
    1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
    2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
    3. **Write Plan**: Write a structured plan to the plan file including:
       - **Context**: Why this change is needed
       - **Steps**: Implementation steps with critical file paths
       - **Verification**: How to test the changes
    4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

    IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    # Check whether all current tasks are complete
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    # Factory for LLM calls: builds a sideQuery callable for memory recall, compatible with Anthropic and OpenAI.
    def _build_side_query(self, *, max_tokens: int = 256):
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model
            async def _sq(system:str, user_message:str)->str:

                resp = await client.messages.create(
                    model=model, max_tokens=max(1, int(max_tokens)), system=system,
                messages=[{"role": "user", "content": user_message}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    block_types = [str(getattr(b, "type", "")) for b in getattr(resp, "content", [])]
                    logging.warning(
                        "side_query returned empty Anthropic-compatible response: model=%s stop_reason=%s content_block_types=%s",
                        getattr(resp, "model", model),
                        getattr(resp, "stop_reason", ""),
                        block_types,
                    )
                return text
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model
            async def _sq_openai(system:str, user_message:str)->str:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max(1, int(max_tokens)),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],

                )
                if not resp.choices:
                    logging.warning("side_query returned no OpenAI-compatible choices: model=%s", model)
                    return ""
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    logging.warning(
                        "side_query returned empty OpenAI-compatible response: model=%s finish_reason=%s message=%s",
                        model,
                        getattr(choice, "finish_reason", ""),
                        choice.message,
                    )
                return content
            return _sq_openai
        return None
    # Abort async tasks
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn:Callable[[str], Awaitable[dict]]) -> None:
        self._plan_approval_fn = fn


    # Plan mode toggle ("state switching and context preservation" mechanism)
    def toggle_plan_mode(self) -> str:
        """
               1. Exit plan mode (switch from plan back to the original mode)
               When the current mode is already plan, the if branch runs:
               Restore the previous state: self.permission_mode = self._pre_plan_mode or "default".
                       On entering plan mode, the original mode is saved in _pre_plan_mode. On exit it is assigned back, restoring the state from before the switch.
               Clear plan mode traces: reset _pre_plan_mode and _plan_file_path (the plan file path), and restore _system_prompt to the base _base_system_prompt.
               Sync OpenAI messages: if the OpenAI API is used underneath, also update the first system prompt in the message list so the AI's context switches back too.
               Feedback: print an exit notice and return the restored mode name.

               2. Enter plan mode (switch from another mode into plan)
       When the current mode is not plan, the else branch runs:
       Preserve the current state: self._pre_plan_mode = self.permission_mode. Stash the mode currently in use (e.g. default or auto-accept) so we can return to it later.
       Switch and initialize: set the current mode to "plan", generate a dedicated plan file path, and extend the system prompt. Appending _build_plan_mode_prompt() injects the plan-only instructions: "think, don't act; output a structured plan".
       Sync OpenAI messages: likewise, if OpenAI is used, update the system prompt in the context in real time.
       Feedback and return: print an enter notice (including the plan file path) and return "plan".
        """
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] =self._system_prompt
            print_info(f"Exited plan mode -> {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    # Main entry point

    async def  chat(self, user_message:str)->None:
        # Lazily load MCP servers on the first chat
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        original_user_message = _safe_utf8_text(user_message)
        ready_skill_extraction_window: dict[str, Any] | None = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        if not self.is_sub_agent:
            ready_skill_extraction_window = self._pop_pending_skill_extraction_window(original_user_message)
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )

        self._aborted = False
        self._turn_output_buffer = []
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.create_task(coro)
        try:
            await self._current_task
        except asyncio.CancelledError:
            self._aborted = True

        finally:
            self._current_task = None
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        self._turn_output_buffer = None
        if not self.is_sub_agent and not self._aborted:
            self._schedule_background_skill_task(self._run_skill_usage_tracking(original_user_message, assistant_text))
            if ready_skill_extraction_window:
                self._schedule_background_skill_task(self._run_online_skill_evolution(ready_skill_extraction_window))
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if not self.is_sub_agent:
            print_divider()
            self._auto_save()



   # Run one conversation turn, collect the model's output text for this turn, and return the tokens consumed
    async def run_once(self, prompt:str)->None:
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
        }

    # Output helper: handles model output text in one place. Depending on whether we're in "collect output" mode,
    # decides whether to store the text in a buffer or print it directly to the terminal.
    def _emit_text(self, text:str)->None:
        text = _safe_utf8_text(text)
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _build_fold_guidance_section(self) -> str:
        if self._custom_system_prompt is not None:
            return ""
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0.0
        last_fold = "never" if not self._fold_last_time else f"{int((time.time() - self._fold_last_time) / 60)}m ago"
        return (
            "\n\n# Runtime Fold Guidance\n"
            f"- Current context utilization: {utilization:.0%}\n"
            f"- Recent tool error streak: {self._tool_error_streak}\n"
            f"- Same tool repeat count: {self._same_tool_repeat_count}\n"
            f"- Last fold: {last_fold}\n"
            "- If the context is getting long, the same tool is being retried without progress, or tool failures are accumulating, call `compact_context` before trying more tools.\n"
            "- If you folded very recently and the next step is clear, prefer continuing rather than folding again.\n"
        )

    def _refresh_runtime_system_prompt(self) -> None:
        if self._custom_system_prompt is not None:
            return
        self._base_system_prompt = build_system_prompt()
        if self.permission_mode == "plan":
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt
        self._system_prompt += self._build_fold_guidance_section()
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt

    def _record_tool_outcome(self, tool_name: str, success: bool) -> None:
        if tool_name == self._last_tool_name:
            self._same_tool_repeat_count += 1
        else:
            self._same_tool_repeat_count = 1
        self._last_tool_name = tool_name
        if success:
            self._tool_error_streak = 0
        else:
            self._tool_error_streak += 1

    def _record_fold_event(self) -> None:
        self._fold_last_time = time.time()
        self._fold_count += 1
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    def _looks_like_tool_failure(self, tool_name: str, raw: str, result: str) -> bool:
        text = f"{raw}\n{result}".lower()
        if any(marker in text for marker in ("error", "denied", "timed out", "timeout")):
            return True
        if tool_name == "compact_context" and "no context compaction" in text:
            return True
        return False

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context

            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    def _strip_runtime_injections(self, text: str) -> str:
        return re.sub(r"\n*<retrieved_skills>.*?</retrieved_skills>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def _message_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._message_text(msg)
            if text:
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def _confirm_online_skill_write(self, summary: str) -> bool:
        if self.permission_mode in {"bypassPermissions", "acceptEdits"}:
            return True
        if self.permission_mode in {"plan", "dontAsk"}:
            return False
        if self.confirm_fn is None:
            return False
        print_confirmation(summary)
        try:
            return bool(await self.confirm_fn(summary))
        except Exception:
            return False

    async def _confirm_background_online_skill_write(self, summary: str) -> bool:
        return self.permission_mode in {"bypassPermissions", "acceptEdits"}

    def _online_evolution_enabled(self) -> bool:
        raw = os.environ.get("CAIRN_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _schedule_background_skill_task(self, coro) -> None:
        if self.permission_mode == "plan":
            try:
                coro.close()
            except Exception:
                pass
            return
        task = asyncio.create_task(coro)
        self._background_skill_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._background_skill_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                pass

        task.add_done_callback(_done)

    async def drain_background_skill_tasks(self) -> None:
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            messages.append({"role": "user", "content": feedback})
        pending["messages"] = messages[-10:]
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        if not original_user_message.strip() or not assistant_text.strip():
            return
        self._pending_skill_extraction_window = {
            "messages": self._recent_dialog_messages(max_messages=8),
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
        }

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        messages = list(window.get("messages") or [])
        if not messages:
            return

        side_query = self._build_side_query(max_tokens=2200)
        if side_query is None:
            return

        try:
            from .online_skill_evolution import online_ingest
        except Exception:
            return

        result = await online_ingest(
            messages=messages,
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("CAIRN_AUTO_SKILL_TARGET", "project"),
        )
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        hits = list(self._last_retrieved_skill_hits or [])
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700)
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            result = record_usage_judgments(judgments)
            if result.get("pruned"):
                self._refresh_runtime_system_prompt()
        except Exception:
            return

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        await self._run_online_skill_evolution(window, interactive_confirm=True)
        self._pending_skill_extraction_window = None
        return {"ok": True}


    def clear_history(self)->None:
        self._anthropic_messages = []
        self._openai_messages = []
        self._pending_skill_extraction_window = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content":self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self):
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    # Get the current cost
    def _get_current_cost_usd(self) -> float:
        return (self.total_input_tokens / 1_000_000) * 3 + (self.total_output_tokens / 1_000_000) * 15

    # Check the budget
    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    # Compact the conversation
    async def compact(self)->None:
        compacted = await self._compact_conversation(trigger="manual")
        if not compacted:
            print_info("Nothing to compact yet.")


    # Restore session info
    def restore_session(self, data:dict)->None:
        if data.get("anthropicMessages"):
            self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(data["anthropicMessages"]))
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        if isinstance(data.get("foldedSessionMemories"), list):
            self._folded_session_memories = _sanitize_for_utf8(data["foldedSessionMemories"])
        print_info(f"Session restored ({self._get_message_count()} messages).")



# Clean up Anthropic message history: fix some role errors and drop invalid tool call messages.
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        role_normalized = []
        for msg in messages:
            copied = dict(msg)
            content = copied.get("content")
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized = []
        i = 0
        while i < len(role_normalized):
            msg = role_normalized[i]
            tool_use_ids = self._anthropic_tool_use_ids(msg)
            if not tool_use_ids:
                normalized.append(msg)
                i += 1
                continue

            next_msg = role_normalized[i + 1] if i + 1 < len(role_normalized) else None
            result_ids = self._anthropic_tool_result_ids(next_msg) if next_msg else set()
            if tool_use_ids.issubset(result_ids):
                normalized.append(msg)
                normalized.append(next_msg)
                i += 2
                continue

            i += 1
        return normalized

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "foldedSessionMemories": _sanitize_for_utf8(self._folded_session_memories),
            })
        except Exception:
            pass

    # Auto-compaction
    async def _check_and_compact(self)->None:
        if self.last_input_token_count > self.effective_window * AUTO_COMPACT_THRESHOLD:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(trigger="auto")

    async def _compact_conversation(self, *, trigger: str = "manual")->bool:
        if self.use_openai:
            compacted = await self._compact_openai(trigger=trigger)
        else:
            compacted = await self._compact_anthropic(trigger=trigger)
        if compacted:
            print_info("Conversation compacted.")
        return compacted

    async def _compact_anthropic(self, *, trigger: str)->bool:
        if len (self._anthropic_messages)<4:
            return False

        transcript = build_anthropic_transcript(_sanitize_for_utf8(self._anthropic_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._anthropic_messages = [{"role": "user", "content": format_folded_memory(memory)}]
        self.last_input_token_count = 0
        self._refresh_runtime_system_prompt()
        return True

    async def _compact_openai(self, *, trigger: str)->bool:
        if len (self._openai_messages)<4:
            return False
        system_msg = self._openai_messages[0]
        transcript = build_openai_transcript(_sanitize_for_utf8(self._openai_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._openai_messages=[
            system_msg,
            {"role": "user", "content": format_folded_memory(memory)},
        ]
        self.last_input_token_count=0
        self._refresh_runtime_system_prompt()
        return True

    async def _generate_folded_session_memory(self, transcript: str) -> dict[str, Any]:
        side_query = self._build_side_query(max_tokens=6000)
        if side_query is None:
            return fallback_folded_memory(transcript)
        try:
            raw = await side_query(FOLD_SESSION_MEMORY_SYSTEM, build_folding_user_prompt(transcript))
            return parse_folded_memory(raw)
        except Exception:
            return fallback_folded_memory(transcript)

    def _record_folded_session_memory(self, trigger: str, memory: dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger": trigger,
            "session_id": self.session_id,
            **memory,
        }
        self._folded_session_memories.append(record)
        try:
            save_folded_session_memory(self.session_id, _sanitize_for_utf8(record))
        except Exception:
            pass

    # Multi-level compression pipeline
    def _run_compression_pipeline(self)->None:
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    # Level 1 compression: budget compression
    def _budget_tool_results_anthropic(self)->None:
        # Compute utilization: utilization = tokens used / effective window size.
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        # If utilization is below 50%, there is plenty of room; return without doing anything.
        if utilization < 0.5:
            return
        # Dynamic budget: critical (>70%): if utilization is high, allow 15,000 characters per tool result.
        # Warning (50%-70%): if utilization is moderate, allow only 30000 characters.
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._anthropic_messages:

            # Only handle messages with role "user". In the tool call flow, tool results are usually fed back to the model as the "user".

            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and len(block["content"]) > budget:
                    # Compute the kept length: keep = (budget - 80) // 2, reserving about 80 characters for the notice in the middle and splitting the rest between head and tail.
                    keep = (budget - 80) // 2
                    # Rebuild the content = head + notice + tail
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self)->None:
        # Compute utilization: utilization = tokens used / effective window size.
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        # If utilization is below 50%, there is plenty of room; return without doing anything.
        if utilization < 0.5:
            return
        # Dynamic budget: critical (>70%): if utilization is high, allow 15,000 characters per tool result.
        # Warning (50%-70%): if utilization is moderate, allow only 30000 characters.
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > budget:
                keep = (budget - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]


    # Level 2 strategy: prune stale tool results
    def _snip_stale_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        results = []
        for mindex,  msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue

            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER:
                    tool_use_id = block.get("tool_use_id")
                    # For each tool_result, look up which tool it came from via tool_use_id
                    tool_info = self._find_tool_use_by_id(tool_use_id)
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({"mindex": mindex, "bindex": bindex, "name": tool_info["name"], "file_path": tool_info.get("input", {}).get("file_path")})

        if len(results) <= KEEP_RECENT_RESULTS:
            return

        to_snip =  set()
        seen_files: dict[str, list[int]] = {}

        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)
        # If a file was read multiple times, keep only the last read result and mark all earlier reads as "snipped".
        for indices in seen_files.values():
            if len (indices) >1 :
                for j in indices[:-1]:
                    to_snip.add (j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range (snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self._anthropic_messages[r["mindex"]]["content"][r["bindex"]]["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    # Micro-compaction

    # Time-based context slimming strategy:
    # if there has been no activity for a long time, earlier tool results have already been consumed, so clear them to free up space

    def _microcompact_anthropic(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return

        all_results = []
        for mindex, msg in enumerate(self._anthropic_messages):
            if msg.get("role")!="user" or not isinstance(msg.get("content"), list):
                continue
            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mindex, bindex))

        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"

    def _find_tool_use_by_id(self, tool_use_id: int) -> dict | None:
        for msg in self._anthropic_messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue

            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return {"name": block["name"], "input": block.get("input", {})}

    # Large result persistence
    # If a tool returns a very large result (over 30KB), don't force it into the context; save it to a temp file instead.
    # Then keep only a 'file path' and a 'content preview' in the conversation. If the model needs the full content later, it can call a tool to read the file again

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        THRESHOLD = 30 * 1024  # 30 KB
        # Convert to bytes
        if (len (result.encode())) <= THRESHOLD:
            return result

        d = Path.home() / ".cairn" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        return (
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. "
            f"You can use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    # Tool execution entry point

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        if name == "compact_context":
            return await self._execute_compact_context_tool(inp)
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
            # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        result = await execute_tool(name, inp, self._read_file_state)
        if name in {"skill_create", "skill_evolve"}:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("ok"):
                    self._refresh_runtime_system_prompt()
            except Exception:
                pass
        return result

    async def _execute_compact_context_tool(self, inp: dict) -> str:
        reason = str(inp.get("reason") or "").strip()
        compacted = await self._compact_conversation(trigger="tool")
        if not compacted:
            self._record_tool_outcome("compact_context", False)
            return "No context compaction was performed because there is not enough conversation history yet."
        self._record_tool_outcome("compact_context", True)
        self._context_cleared = True
        suffix = f"\nReason: {reason}" if reason else ""
        return (
            "Context compacted into structured session memory. "
            "Continue from the folded memory now present in the conversation context."
            f"{suffix}"
        )


    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))

        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        # fork means this skill doesn't put its prompt back into the current conversation; it starts a sub-agent to complete the task separately.
        if result["context"] == "fork":
            # result["allowed_tools"] - direct access
            tools = (
                [t for t in self.tools if t["name"] in  result["allowed_tools"] ]
                # result.get("allowed_tools") - safe access
                # key present: returns the corresponding value (may be None, [], ["tool1"], etc.)
                # key missing: returns None (no exception raised)
                if result.get("allowed_tools")
                else  [t for t in self.tools if t["name"] != "agent"]
            )

            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill produced no output)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    async def _execute_plan_mode_tool(self, name):
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path =  self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."
        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text(encoding="utf-8")
            # Interactive approval flow (if there is an approval function)
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice =="keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                # Leave plan mode
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )
                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )
            # Fallback when there is no approval function (e.g. sub-agents)
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt

            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """Clear history but keep the system prompt."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    async def _execute_agent_tool(self, inp:dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")
        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)

        sub_agent = Agent(
            model=self.model,
            api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            is_sub_agent=True,
            permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
        )
        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            print_sub_agent_end(agent_type, description)
            return result["text"] or "(Sub-agent produced no output)"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

#--------------Anthropic backend---------------
    async def  _chat_anthropic(self, user_message: str) -> None:
        self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(self._anthropic_messages))
        user_message = _safe_utf8_text(user_message)
        # First add this turn's user input to the Anthropic message history; every subsequent model call carries this context.
        self._anthropic_messages.append({"role": "user", "content": user_message})

        # Async memory prefetch: only the main agent queries memory; sub-agents don't inject memories.
        # This only starts a background task and doesn't block the current model call.
        memory_prefetch:MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )
        while True:
            # When an external abort is requested, end the whole agent loop.
            if self._aborted:
                break

            # Try to compact the context before each model call so the message history doesn't grow too long.
            self._run_compression_pipeline()

            # If the memory prefetch task is done, append the retrieved memory content to the last user message.
            # consumed ensures the same batch of memories is injected only once.
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._anthropic_messages[-1] if self._anthropic_messages else None
                        if last and last.get("role") == "user":
                            content = last.get("content", "")
                            if isinstance(content, str):
                                # Strings are immutable, so assign back to message.
                                last["content"] = content + "\n\n" + injection_text
                            elif isinstance(content, list):
                                # A list is mutable, so append modifies the list last["content"] points to directly.
                                content.append({"type": "text", "text": injection_text})
                        else:
                            # If the last message isn't a user message, append a separate user message to carry the memories.
                            self._anthropic_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            # Record memories already injected in this session so later retrievals can avoid surfacing them again.
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += m.size
                except:
                    # A failed memory injection must not interrupt the main conversation.
                    pass

            if not self.is_sub_agent:
                start_spinner()


            # Holds tool tasks started early. Key is the tool_use block id returned by Anthropic.
            early_executions: dict[str, asyncio.Task] = {}


            def _on_tool_block(block:dict):
                # During streaming, as soon as a complete tool_use block arrives, if the tool is concurrency-safe and permitted,
                # it can start executing early, cutting the idle time spent waiting for the full model response.
                if block["name"] in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(block["name"], block["input"], self.permission_mode, self._plan_file_path)
                    if perm["action"]=="allow":
                        task =asyncio.create_task(self._execute_tool_call(block["name"], block["input"]))
                        early_executions[block["id"]] = task


            # Call the Anthropic streaming API; _on_tool_block fires when a tool block completes during streaming.
            response = await self._call_anthropic_stream(on_tool_block_complete=_on_tool_block)
            if not self.is_sub_agent:
                stop_spinner()

            # Record the timing and token usage of this model call, for cost display and budget control.
            self.last_api_call_time = time.time()
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.last_input_token_count = response.usage.input_tokens

            # Anthropic's response content may mix text blocks and tool_use blocks; pick out only the tool calls here.
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # Write all content blocks returned by the model to the message history; later tool_results must match these tool_use blocks.
            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            # No tool calls means the model has given its final reply; this turn is over.
            if not tool_uses:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                break

            # With tool calls, move on to the next round of tool execution. Also check the turn/budget limits here.
            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                self._anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool execution skipped: {budget['reason']}",
                        }
                        for tu in tool_uses
                    ],
                })
                break


            # Collect all tool results for this turn, to be sent back to the model as a tool_result message.
            tool_results: list[dict] = []
            context_break = False

            for tu in tool_uses:
                # context_break means a tool cleared the context during execution, so stop processing the remaining tools this turn.
                if context_break or self._aborted:
                    break

                # Convert the tool input to a plain dict for permission checks, printing, and actual execution.
                inp = dict(tu.input) if hasattr(tu, "items") else tu.input
                print_tool_call(tu.name, inp)

                # If this tool already started executing during streaming, just wait for it to finish and collect the result.
                early_task = early_executions.get(tu.id)
                if early_task:
                    try:
                        raw = await early_task
                    except Exception as e:
                        raw = f"Error executing tool: {e}"
                    raw = _safe_utf8_text(raw)
                    res = self._persist_large_result(tu.name, raw)
                    print_tool_result(tu.name, res)
                    self._record_tool_outcome(tu.name, not self._looks_like_tool_failure(tu.name, raw, res))
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                    continue

                # For tools not started early, check permissions right before actually executing.

                perm = check_permission(tu.name, inp, self.permission_mode, self._plan_file_path)
                if perm["action"] == "deny":
                    # When permission is denied, still return a tool_result so the model knows why the tool call failed.
                    print_info(f"Denied: {perm.get('message', '')}")
                    self._record_tool_outcome(tu.name, False)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": f"Action denied: {perm.get('message', '')}"})
                    continue

                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    # High-risk operations need user confirmation; once confirmed for a message it's cached to avoid asking again.
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        self._record_tool_outcome(tu.name, False)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])

                # Once permitted, execute the tool and persist large outputs as a summary or reference that can be sent back.
                try:
                    raw = await self._execute_tool_call(tu.name, inp)
                except Exception as e:
                    raw = f"Error executing tool: {e}"
                raw = _safe_utf8_text(raw)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)
                self._record_tool_outcome(tu.name, not self._looks_like_tool_failure(tu.name, raw, res))

                if self._context_cleared:
                    # If a tool cleared the context during execution, write the result as a new user message
                    # and stop processing the remaining tools this turn, so old and new context don't get mixed.
                    self._context_cleared = False
                    self._anthropic_messages.append({"role": "user", "content": res})
                    context_break = True
                    break

                # Anthropic requires tool_result to reference the earlier tool_use block via tool_use_id.
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                # Anthropic requires assistant/tool_use to be followed immediately by a user/tool_result message,
                # and that message must include results for every tool_use in this turn.
                self._anthropic_messages.append({"role": "user", "content": tool_results})

            self._context_cleared = False

            # Tool results can be long; check whether to compact the context after each round of tool execution.
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):

        async def _do():
            max_output =  _get_max_output_tokens(self.model)

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else 16384,
                "system": _safe_utf8_text(self._system_prompt),
                "tools": _sanitize_for_utf8(get_active_tool_definitions(self.tools)),
                "messages": _sanitize_for_utf8(self._anthropic_messages),
            }
            # If thinking mode is on, add the thinking parameter to the Anthropic request.
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True

            tool_blocks_by_index: dict[int, dict] = {}

            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    # When the event is a tool call start:
                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        # If the block type is tool_use, record this tool call:
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            # Tool argument JSON arrives in streamed chunks, so start with an empty input_json string.
                            tool_blocks_by_index[event.index]= {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }
                    # When the event is a content delta, there are three cases.
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # Case 1, plain text: when the model outputs body text,
                        # call _emit_text(). In normal interactive mode it prints;
                        # in run_once() it writes to _output_buffer.
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        # Case 2, thinking content:
                        # if the model returns thinking content, output it too, prefixed with: [thinking]
                        elif hasattr(delta, 'thinking'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        # Case 3, tool argument JSON fragments: tool call arguments don't arrive all at once
                        # but piece by piece, so keep appending them to input_json.
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += _safe_utf8_text(delta.partial_json)
                    # When a content block ends:
                    # if it's a tool call recorded earlier, parse the assembled JSON:
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            # Then invoke the callback:
                            # its purpose is usually that once a tool call is complete,
                            # the tool can start executing early instead of waiting for the whole assistant message to finish.
                            on_tool_block_complete({
                                "type": "tool_use", "id": _safe_utf8_text(tb["id"]),
                                "name": _safe_utf8_text(tb["name"]), "input": _sanitize_for_utf8(parsed),
                            })
                final_message = await stream.get_final_message()

            # Filter out thinking messages (thinking content generally shouldn't enter the history; it bloats later context and may not match the API message format).
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message
# Call _do(); on retryable errors, _with_retry() handles the retries.
        return await _with_retry(_do)

    # OpenAI backend

    async def _chat_openai(self, user_message:str) -> None:
        user_message = _safe_utf8_text(user_message)
        self._openai_messages.append({"role": "user", "content": user_message})

        # Prefetch handle MemoryPrefetch
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._openai_messages[-1] if self._openai_messages else None

                        if last and last.get("role") == "user":
                            last["content"] = (last.get("content") or "") + "\n\n" + injection_text
                        else:
                            self._openai_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass

            if not self.is_sub_agent:
                start_spinner()

            response = await self._call_openai_stream()

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            if response.get("usage"):
                self.total_input_tokens += response["usage"]["prompt_tokens"]
                self.total_output_tokens += response["usage"]["completion_tokens"]
                self.last_input_token_count = response["usage"]["prompt_tokens"]

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            self._openai_messages.append(message)

            tool_calls = message.get("tool_calls")

            if not tool_calls:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            oai_checked: list[dict] = []
            for tc in tool_calls:
                if self._aborted:
                    break

                if tc.get("type") != "function":
                    continue

                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}

                print_tool_call(fn_name, inp)

                perm = check_permission(fn_name, inp, self.permission_mode, self._plan_file_path)

                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    self._record_tool_outcome(fn_name, False)
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                        "result": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        self._record_tool_outcome(fn_name, False)
                        oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                            "result": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": True})

                oai_batches: list[dict] = []
                for ct in oai_checked:
                    safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                    if safe and oai_batches and oai_batches[-1]["concurrent"]:
                        oai_batches[-1]["items"].append(ct)
                    else:
                        oai_batches.append({"concurrent": safe, "items": [ct]})

                oai_context_break = False
                for batch in oai_batches:
                    if oai_context_break or self._aborted:
                        break

                    if batch["concurrent"]:
                        async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                            raw = await self._execute_tool_call(ct_item["fn"], ct_item["inp"])
                            raw = _safe_utf8_text(raw)
                            res = self._persist_large_result(ct_item["fn"], raw)
                            print_tool_result(ct_item["fn"], res)
                            return ct_item, res

                        results = await asyncio.gather(*[_run_oai_safe(ct) for ct in batch["items"]])
                        for ct_item, res in results:
                            self._record_tool_outcome(
                                ct_item["fn"],
                                not self._looks_like_tool_failure(ct_item["fn"], "", res),
                            )
                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                    else:
                        for ct in batch["items"]:
                            if not ct["allowed"]:
                                self._openai_messages.append(
                                    {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": ct["result"]})
                                continue

                            raw = await self._execute_tool_call(ct["fn"], ct["inp"])
                            raw = _safe_utf8_text(raw)
                            res = self._persist_large_result(ct["fn"], raw)
                            print_tool_result(ct["fn"], res)
                            self._record_tool_outcome(
                                ct["fn"],
                                not self._looks_like_tool_failure(ct["fn"], raw, res),
                            )

                            if self._context_cleared:
                                self._context_cleared = False
                                self._openai_messages.append({"role": "user", "content": res})
                                oai_context_break = True
                                break

                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})

            self._context_cleared = False
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    async def _call_openai_stream(self) -> dict:
        async def _do():
            stream = await self._openai_client.chat.completions.create(
                model=self.model,
                tools=_sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools))),
                messages=_sanitize_for_utf8(self._openai_messages),
                stream=True,
                stream_options={"include_usage": True},
            )

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += _safe_utf8_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                        else:
                            tool_calls[tc.index] = {
                                "id": _safe_utf8_text(tc.id or ""),
                                "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return await _with_retry(_do)

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
