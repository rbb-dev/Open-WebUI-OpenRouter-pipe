"""Tool execution orchestrator for the OpenRouter pipe.

This module handles:
- Tool call execution via queue/worker pipeline
- Direct tool server registry building (Socket.IO bridge)
- Legacy direct execution fallback
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.config import entry_data_url
from ..core.timing_logger import timed, timing_mark
from ..core.url_scheme import split_base64_data_url
from ..core.utils import (
    TOOL_CALL_STATUSES,
    parse_tool_arguments,
    picture_output,
)
from ..core.warn_latch import warn_level

_OWUI_RESULT_WARN_COOLDOWN_S = 300.0
_OWUI_RESULT_WARN_CAP = 256
from ..storage.owui_files import is_linkable_chat
from ..storage.persistence import generate_item_id
from .tool_schema import _advertised_root_params

if TYPE_CHECKING:
    from starlette.requests import Request

    from ..core.circuit_breaker import CircuitBreaker
    from ..pipe import Pipe

from ..streaming.event_emitter import EventEmitter

try:
    from open_webui.utils.middleware import (  # pyright: ignore[reportMissingImports]
        terminal_event_handler as _owui_terminal_event_handler,
    )
except ImportError:
    _owui_terminal_event_handler = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_terminal_event_handler = None  # type: ignore[assignment]

try:
    from open_webui.utils.middleware import (  # pyright: ignore[reportMissingImports]
        store_tool_result_image as _owui_store_tool_result_image,
    )
except ImportError:
    _owui_store_tool_result_image = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_store_tool_result_image = None  # type: ignore[assignment]

try:
    from open_webui.utils.middleware import (  # pyright: ignore[reportMissingImports]
        build_terminal_file_tool_result as _owui_build_terminal_file_tool_result,
    )
except ImportError:
    _owui_build_terminal_file_tool_result = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_build_terminal_file_tool_result = None  # type: ignore[assignment]

try:
    from open_webui.utils.middleware import (
        process_tool_result as _owui_process_tool_result,
    )
except ImportError:
    _owui_process_tool_result = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_process_tool_result = None  # type: ignore[assignment]

try:
    from open_webui.models.users import Users as _Users
except ImportError:
    _Users = None  # type: ignore[assignment,misc]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.users failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _Users = None  # type: ignore[assignment,misc]

try:
    from open_webui.utils.ask_user import (  # pyright: ignore[reportMissingImports]
        normalize_ask_user_request as _owui_normalize_ask_user_request,
    )
except ImportError:
    _owui_normalize_ask_user_request = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.ask_user failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_normalize_ask_user_request = None  # type: ignore[assignment]

try:
    from open_webui.utils.ask_user import (  # pyright: ignore[reportMissingImports]
        get_ask_user_tool_calls as _owui_get_ask_user_tool_calls,
    )
except ImportError:
    _owui_get_ask_user_tool_calls = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.ask_user failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_get_ask_user_tool_calls = None  # type: ignore[assignment]

try:
    from open_webui.utils.tools import (  # pyright: ignore[reportMissingImports]
        get_updated_tool_function as _owui_get_updated_tool_function,
    )
except ImportError:
    _owui_get_updated_tool_function = None  # type: ignore[assignment]
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.tools failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_get_updated_tool_function = None  # type: ignore[assignment]

_ASK_USER_GRACE_SECONDS = 15.0


def resolved_tool_name(call: dict[str, Any]) -> str:
    raw_name = call.get("name")
    return raw_name.strip() if isinstance(raw_name, str) else ""


_BROWSER_TRANSPORT_ERRORS = frozenset({
    "Client session disconnected.",
    "Event call timed out. The browser tab may be inactive or closed.",
    "Tool Server Not Found",
})
_HTTP_ERROR_PREFIX = "HTTP error! Status:"


def _reports_a_browser_transport_failure(error: Any) -> bool:
    if not isinstance(error, str):
        return False
    return error in _BROWSER_TRANSPORT_ERRORS or error.startswith(_HTTP_ERROR_PREFIX)


@dataclass(slots=True)
class _QueuedToolCall:
    """Stores a pending tool call plus execution metadata for worker pools."""
    call: dict[str, Any]
    tool_cfg: dict[str, Any]
    args: dict[str, Any]
    future: asyncio.Future
    allow_batch: bool
    holds_slot: bool = False


@dataclass(slots=True)
class _ToolExecutionContext:
    """Holds shared state for executing tool calls within breaker limits."""
    queue: asyncio.Queue[list[_QueuedToolCall] | None]
    per_request_semaphore: asyncio.Semaphore
    global_semaphore: asyncio.Semaphore | None
    timeout: float
    batch_timeout: float | None
    idle_timeout: float | None
    user_id: str
    event_emitter: EventEmitter | None
    batch_cap: int
    request: Request | None = None
    user: dict[str, Any] | None = None
    resolved_user: Any = None
    resolved_user_done: bool = False
    resolved_user_error: BaseException | None = None
    resolved_user_task: Any = None
    resolved_user_waiters: int = 0
    metadata: dict[str, Any] | None = None
    request_id: str = ""
    fusion_inner: bool = False
    tool_breaker: CircuitBreaker | None = None
    tool_call_budget: int | None = None
    workers: list[asyncio.Task] = field(default_factory=list)
    timeout_error: str | None = None
    on_complete: Callable[[dict, dict], Awaitable[None]] | None = None
    carded_calls: set[str] = field(default_factory=set)
    terminal_files_inline: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)


async def _read_user_row(context: _ToolExecutionContext) -> Any:
    user_obj = context.user
    if isinstance(user_obj, dict) and _Users is not None:
        user_id = user_obj.get("id")
        if user_id:
            user_obj = await _Users.get_user_by_id(user_id)
    return user_obj


async def _resolved_user_obj(context: _ToolExecutionContext) -> Any:
    if context.resolved_user_done:
        if context.resolved_user_error is not None:
            raise context.resolved_user_error
        return context.resolved_user
    if context.resolved_user_task is None:
        context.resolved_user_task = asyncio.ensure_future(_read_user_row(context))
    context.resolved_user_waiters += 1
    try:
        user_obj = await asyncio.shield(context.resolved_user_task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        context.resolved_user_error = exc
        context.resolved_user_done = True
        raise
    finally:
        context.resolved_user_waiters -= 1
        if context.resolved_user_waiters <= 0 and not context.resolved_user_done:
            task = context.resolved_user_task
            context.resolved_user_task = None
            if task is not None and not task.done():
                task.cancel()
    context.resolved_user = user_obj
    context.resolved_user_done = True
    return user_obj


def is_builtin_ask_user(tool_cfg: Any) -> bool:
    return (
        isinstance(tool_cfg, dict)
        and tool_cfg.get("type") == "builtin"
        and tool_cfg.get("tool_id") == "builtin:ask_user"
    )


_DATA_ENTRY_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    "application/zip": ".zip",
    "text/html": ".html",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


def _entry_file_name(mime_type: str) -> str:
    normalised = (mime_type or "").split(";")[0].strip().lower()
    return "tool_result" + _DATA_ENTRY_EXTENSIONS.get(normalised, ".bin")


_DATA_ENTRY_DECODE_QUANTUM = (65536 // 3) * 4


async def _decode_data_entry(payload: str) -> bytes:
    raw = bytearray()
    for offset in range(0, len(payload), _DATA_ENTRY_DECODE_QUANTUM):
        raw += base64.b64decode(payload[offset : offset + _DATA_ENTRY_DECODE_QUANTUM], validate=True)
        await asyncio.sleep(0)
    return bytes(raw)


def _idle_allowance(
    context: _ToolExecutionContext,
    pending: list[tuple[int, dict[str, Any], asyncio.Future, float | None]],
) -> float | None:
    allowance = context.idle_timeout
    if allowance:
        for _index, _call, _future, window in pending:
            if window is not None:
                allowance = max(allowance, window)
    return allowance


def _idle_deadline(
    allowance: float | None,
    started_at: float,
    pre_enqueue_at: float,
    now: float,
) -> float | None:
    if not allowance:
        return None
    return now + (allowance - (pre_enqueue_at - started_at))


class ToolExecutor:
    """Orchestrates tool execution and direct tool server integration."""

    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize tool executor.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging and warnings
        """
        self._pipe = pipe
        self.logger = logger
        self._owui_result_warn_ts: OrderedDict[str, float] = OrderedDict()

    def _owui_warn_level(self, cause: str) -> int:
        latch = self._owui_result_warn_ts
        level = warn_level(latch, cause, cooldown_s=_OWUI_RESULT_WARN_COOLDOWN_S)
        latch.move_to_end(cause)
        while len(latch) > _OWUI_RESULT_WARN_CAP:
            latch.popitem(last=False)
        return level

    # Argument parsing helpers

    def _is_batchable_tool_call(self, args: dict[str, Any]) -> bool:
        """Check if a tool call can be batched or must run sequentially.

        Args:
            args: Parsed tool arguments dictionary

        Returns:
            True if the tool call can be batched, False if it has dependency markers
        """
        blockers = {"depends_on", "_depends_on", "sequential", "no_batch"}
        return not any(key in args for key in blockers)

    @staticmethod
    def _is_builtin_ask_user(tool_cfg: Any) -> bool:
        return is_builtin_ask_user(tool_cfg)

    def _tool_breaker(self, context: _ToolExecutionContext) -> CircuitBreaker | None:
        return context.tool_breaker if context.fusion_inner else self._pipe._circuit_breaker

    @staticmethod
    def _tool_error_text(exc: BaseException) -> str:
        return json.dumps(
            {"error": str(exc) or type(exc).__name__}, indent=2, ensure_ascii=False
        )

    @staticmethod
    def is_process_control(exc: BaseException) -> bool:
        return isinstance(exc, (SystemExit, KeyboardInterrupt, GeneratorExit))

    @staticmethod
    async def _with_current_chat(fn: Any, tool_cfg: dict[str, Any], context: _ToolExecutionContext) -> Any:
        if _owui_get_updated_tool_function is None or tool_cfg.get("direct"):
            return fn
        return await _owui_get_updated_tool_function(
            function=fn,
            extra_params={
                "__messages__": context.messages,
                "__files__": (context.metadata or {}).get("files", []),
            },
        )
    def _ask_user_window(self, tool_cfg: Any, args: dict[str, Any]) -> float | None:
        if _owui_normalize_ask_user_request is None or not self._is_builtin_ask_user(tool_cfg):
            return None
        try:
            timeout_ms = _owui_normalize_ask_user_request(args)["timeout_ms"]
        except ValueError:
            timeout_ms = 120_000
        return timeout_ms / 1000 + _ASK_USER_GRACE_SECONDS

    def _ask_user_refusal(self, calls: list[dict], tools: dict[str, dict[str, Any]]) -> str | None:
        is_ask_user = []
        for call in calls:
            name = call.get("name")
            is_ask_user.append(self._is_builtin_ask_user(tools.get(name.strip() if isinstance(name, str) else "")))
        if not any(is_ask_user) or _owui_get_ask_user_tool_calls is None:
            return None
        _, refusal = _owui_get_ask_user_tool_calls(
            [{"function": {"name": "ask_user" if flag else ""}} for flag in is_ask_user]
        )
        return refusal

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any] | None:
        return parse_tool_arguments(raw_args)

    @timed
    def _terminal_file_result_safe(
        self,
        origin_name: str,
        args: Any,
        raw_result: Any,
        tool_cfg: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> Any:
        if _owui_build_terminal_file_tool_result is None:
            return raw_result
        try:
            reshaped = _owui_build_terminal_file_tool_result(
                origin_name, args if isinstance(args, dict) else {}, raw_result, tool_cfg, metadata
            )
        except Exception:
            self.logger.log(
                self._owui_warn_level(f"terminal-file:{origin_name}"),
                "Open WebUI could not prepare the Open Terminal file result of '%s'; the model receives the result "
                "as the terminal returned it",
                origin_name,
                exc_info=True,
            )
            return raw_result
        return reshaped if reshaped else raw_result

    async def _emit_terminal_events_safe(
        self,
        origin_name: str,
        args: Any,
        result_text: Any,
        event_emitter: EventEmitter | None,
    ) -> None:
        if _owui_terminal_event_handler is None or event_emitter is None:
            return
        try:
            await _owui_terminal_event_handler(
                origin_name, args if isinstance(args, dict) else {}, result_text, event_emitter
            )
        except Exception:
            self.logger.log(
                self._owui_warn_level(f"terminal-events:{origin_name}"),
                "Open WebUI could not tell the chat about the Open Terminal call '%s'; its file browser and preview "
                "are not updated for it",
                origin_name,
                exc_info=True,
            )

    @timed
    async def _process_tool_result_safe(
        self,
        tool_name: str,
        tool_type: str,
        raw_result: Any,
        context: _ToolExecutionContext | None,
        *,
        is_direct_tool: bool = False,
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        """Process tool result to extract text, files, and embeds safely.

        This wraps OpenWebUI's process_tool_result() with proper error handling
        to ensure no tool can crash the pipe. If OpenWebUI's function is unavailable
        or fails, we fall back to simple str() conversion.

        Args:
            tool_name: Name of the tool that produced the result
            tool_type: Type of tool (function, mcp, external, etc.)
            raw_result: Raw result from tool execution (before str conversion)
            context: Tool execution context with request/user/metadata

        Returns:
            Tuple of (output_text, files_list, embeds_list)

        Note:
            Files are typically images/audio from MCP tools or OpenAPI responses.
            Embeds are HTML snippets from HTMLResponse with Content-Disposition: inline.
        """
        timing_mark(f"process_result:{tool_name}")
        files: list[dict[str, Any]] = []
        embeds: list[str] = []

        try:
            if _owui_process_tool_result is not None and context is not None:
                try:
                    user_obj = await _resolved_user_obj(context)
                    processed_result, files, embeds = await _owui_process_tool_result(
                        request=context.request,
                        tool_function_name=tool_name,
                        tool_result=raw_result,
                        tool_type=tool_type,
                        direct_tool=is_direct_tool,
                        metadata=context.metadata,
                        user=user_obj,
                    )
                    output_text = "" if processed_result is None else str(processed_result)
                    timing_mark(f"process_result:{tool_name}:owui_done")
                    return output_text, files, embeds
                except Exception as proc_exc:
                    self.logger.log(
                        self._owui_warn_level(tool_name),
                        "Open WebUI could not process the result of '%s'; the model "
                        "will receive a plain string rendering of the raw payload "
                        "instead: %s",
                        tool_name,
                        proc_exc,
                        exc_info=True,
                    )
                    # Continue to fallback below

            output_text = "" if raw_result is None else str(raw_result)
            timing_mark(f"process_result:{tool_name}:fallback_done")
            return output_text, files, embeds

        except Exception as exc:
            self.logger.warning(
                "Unexpected error processing result for '%s': %s",
                tool_name,
                exc,
                exc_info=True,
            )
            try:
                output_text = "" if raw_result is None else str(raw_result)
            except Exception:
                self.logger.warning(
                    "Tool result for '%s' is not stringifiable; the model will receive a "
                    "placeholder instead of the result",
                    tool_name,
                    exc_info=True,
                )
                output_text = f"[Tool result could not be serialized: {type(raw_result).__name__}]"
            return output_text, [], []

    @timed
    async def _execute_function_calls(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Execute tool calls via the per-request queue/worker pipeline."""

        context = self._pipe._TOOL_CONTEXT.get()
        if context is None:
            raise ValueError(
                "_execute_function_calls called without a tool execution context. "
                "Ensure _TOOL_CONTEXT is set before calling tool execution."
            )

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        pending: list[tuple[int, dict[str, Any], asyncio.Future, float | None]] = []
        batches: list[list[_QueuedToolCall]] = []
        slots: list[dict[str, Any] | None] = [None] * len(calls)
        _on_complete = context.on_complete
        ask_user_refusal = self._ask_user_refusal(calls, tools)
        deferred: list[tuple[int, dict, dict]] = []

        async def _append_and_notify(index: int, call: dict, result: dict) -> None:
            slots[index] = result
            if _on_complete:
                deferred.append((index, call, result))

        async def _flush_deferred(upto: int) -> None:
            handler = _on_complete
            if handler is None:
                return
            while deferred and deferred[0][0] < upto:
                _index, call, result = deferred.pop(0)
                with contextlib.suppress(Exception):
                    await handler(call, result)

        async def _refuse(index: int, call: dict, text: str) -> None:
            await _append_and_notify(index, call, self._build_tool_output(call, text, status="failed"))

        try:
            for index, call in enumerate(calls):
                tool_name = resolved_tool_name(call)
                tool_cfg = tools.get(tool_name)
                if ask_user_refusal and self._is_builtin_ask_user(tool_cfg):
                    await _refuse(index, call, ask_user_refusal)
                    continue
                try:
                    args = parse_tool_arguments(call.get("arguments"))
                except ValueError:
                    await _refuse(
                        index, call, f"Error: Tool call arguments for `{tool_name}` must be a JSON object. Please try again."
                    )
                    continue
                if args is None:
                    await _refuse(
                        index,
                        call,
                        "Error: Tool call arguments could not be parsed. The model generated malformed or "
                        f"incomplete JSON for `{tool_name}`. Please try again.",
                    )
                    continue
                if not tool_cfg:
                    await _refuse(index, call, f'Error: Tool "{tool_name}" not found.')
                    continue
                if _owui_normalize_ask_user_request is not None and self._is_builtin_ask_user(tool_cfg):
                    try:
                        args = _owui_normalize_ask_user_request(args)
                    except ValueError as exc:
                        await _refuse(index, call, f"Invalid arguments: {exc}")
                        continue
                tool_type = (tool_cfg.get("type") or "function").lower()
                breaker = self._tool_breaker(context)
                if breaker is not None and not breaker.tool_allows(
                    context.user_id, tool_type, tool_name
                ):
                    await self._notify_tool_breaker(context, tool_type, call.get("name"))
                    await _append_and_notify(index, call, self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' skipped due to repeated failures.",
                        status="skipped",
                    ))
                    continue
                fn = tool_cfg.get("callable")
                if fn is None:
                    await _append_and_notify(index, call, self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' has no callable configured.",
                        status="failed",
                    ))
                    continue
                if context.tool_call_budget is not None:
                    if context.tool_call_budget <= 0:
                        await _append_and_notify(index, call, self._build_tool_output(
                            call,
                            f"Tool '{call.get('name')}' skipped: fusion tool budget exhausted.",
                            status="skipped",
                        ))
                        continue
                    context.tool_call_budget -= 1

                future: asyncio.Future = loop.create_future()
                allow_batch = self._is_batchable_tool_call(args)
                queued = _QueuedToolCall(
                    call=call,
                    tool_cfg=tool_cfg,
                    args=args,
                    future=future,
                    allow_batch=allow_batch,
                )
                batch = batches[-1] if batches else None
                if (
                    batch is not None
                    and allow_batch
                    and batch[0].allow_batch
                    and len(batch) < context.batch_cap
                    and all(self._can_batch_tool_calls(member, queued) for member in batch)
                ):
                    batch.append(queued)
                else:
                    batches.append([queued])
                origin_source = tool_cfg.get("origin_source")
                origin_name = tool_cfg.get("origin_name")
                if isinstance(origin_source, str) and isinstance(origin_name, str):
                    self.logger.debug(
                        "Enqueued tool %s (origin=%s source=%s batch=%s)",
                        call.get("name"),
                        origin_name,
                        origin_source,
                        allow_batch,
                    )
                else:
                    self.logger.debug("Enqueued tool %s (batch=%s)", call.get("name"), allow_batch)
                pending.append((index, call, future, self._ask_user_window(tool_cfg, args)))

            pre_enqueue_at = loop.time()
            enqueue_allowance = context.batch_timeout
            for _index, _call, _future, window in pending:
                if window is not None:
                    enqueue_allowance = max(enqueue_allowance or 0.0, window)
            enqueue_deadline = pre_enqueue_at + enqueue_allowance if enqueue_allowance else None
            unqueued: list[_QueuedToolCall] = []
            for position, batch in enumerate(batches):
                if enqueue_deadline is None:
                    await context.queue.put(batch)
                    continue
                try:
                    async with asyncio.timeout_at(enqueue_deadline):
                        await context.queue.put(batch)
                except TimeoutError:
                    unqueued.extend(item for later in batches[position:] for item in later)
                    break
            for item in unqueued:
                if not item.future.done():
                    item.future.set_result(self._build_tool_output(
                        item.call,
                        f"Tool '{item.call.get('name')}' was not started: every tool worker was still "
                        f"busy {enqueue_allowance:.0f}s after the round asked for it (queue wait).",
                        status="failed",
                    ))
            if unqueued:
                context.timeout_error = context.timeout_error or (
                    "tool workers were still busy when the round ran out of time to queue its calls"
                )
                self.logger.warning("Tool queue wait: %d call(s) were never started", len(unqueued))

            allowance = _idle_allowance(context, pending)
            collected: dict[int, Any] = {}
            notified: set[int] = set()
            deadline = _idle_deadline(allowance, started_at, pre_enqueue_at, loop.time())
            for pending_index, (index, call, future, _window) in enumerate(pending):
                await _flush_deferred(index)
                try:
                    async with asyncio.timeout_at(deadline) if deadline is not None else contextlib.nullcontext():
                        collected[pending_index] = await future
                except TimeoutError:
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            "Tool '%s' raised while awaiting result (call_id=%s).",
                            call.get("name"),
                            call.get("call_id"),
                            exc_info=True,
                        )
                    collected[pending_index] = self._build_tool_output(
                        call,
                        self._tool_error_text(exc),
                        status="failed",
                    )
                if _on_complete:
                    with contextlib.suppress(Exception):
                        await _on_complete(call, collected[pending_index])
                notified.add(pending_index)

            for pending_index, (index, call, future, _window) in enumerate(pending):
                await _flush_deferred(index)
                result = collected.get(pending_index)
                if result is None and future.done() and not future.cancelled():
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                "Tool '%s' had already failed when the wait for results ended (call_id=%s).",
                                call.get("name"),
                                call.get("call_id"),
                                exc_info=True,
                            )
                        result = self._build_tool_output(call, self._tool_error_text(exc), status="failed")
                if result is None:
                    future.cancel()
                    tool_name = call.get("name")
                    message = (
                        f"Tool '{tool_name}' timed out after {allowance:.0f}s (idle timeout)."
                        if allowance
                        else "Tool idle timeout exceeded."
                    )
                    if context and not context.timeout_error:
                        context.timeout_error = message
                    self.logger.warning("Tool idle timeout: %s", message)
                    result = self._build_tool_output(call, message, status="failed")
                if _on_complete and pending_index not in notified:
                    with contextlib.suppress(Exception):
                        await _on_complete(call, result)
                slots[index] = result

            return [result for result in slots if result is not None]
        finally:
            await asyncio.shield(_flush_deferred(len(calls) + 1))

    @timed
    def _build_direct_tool_server_registry(
        self,
        __metadata__: dict[str, Any],
        *,
        event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        event_emitter: EventEmitter | None,
        strictify: bool = True,
    ) -> dict[str, dict[str, Any]]:
        direct_registry: dict[str, dict[str, Any]] = {}

        try:
            if not isinstance(__metadata__, dict):
                return {}
            resolved = __metadata__.get("tools")
            if not isinstance(resolved, dict) or not resolved:
                return {}
            if event_call is None:
                return {}

            def _browser_call(
                allowed: set[str],
                tool_name: str,
                server_payload: dict[str, Any],
                send: Callable[[dict[str, Any]], Awaitable[Any]],
            ) -> Callable[..., Awaitable[Any]]:
                async def _direct_tool_callable(**kwargs: Any) -> Any:
                    try:
                        filtered = {k: v for k, v in kwargs.items() if k in allowed}
                        session_id = __metadata__.get("session_id")

                        payload = {
                            "type": "execute:tool",
                            "data": {
                                "id": str(uuid.uuid4()),
                                "name": tool_name,
                                "params": filtered,
                                "server": server_payload,
                                "session_id": session_id,
                            },
                        }
                        reply = await send(payload)
                        if isinstance(reply, dict) and _reports_a_browser_transport_failure(reply.get("error")):
                            return [reply, None]
                        return reply
                    except Exception as exc:
                        self.logger.debug("Direct tool '%s' failed: %s", tool_name, exc, exc_info=True)
                        with contextlib.suppress(Exception):
                            await self._pipe._event_emitter_handler._emit_notification(
                                event_emitter,
                                f"Tool '{tool_name}' failed: {exc}",
                                level="warning",
                            )
                        return [{"error": str(exc)}, None]

                return _direct_tool_callable

            for entry in resolved.values():
                try:
                    if not (isinstance(entry, dict) and entry.get("direct") is True):
                        continue
                    spec = entry.get("spec")
                    server = entry.get("server")
                    if not isinstance(spec, dict) or not isinstance(server, dict):
                        continue
                    raw_name = spec.get("name")
                    name = raw_name.strip() if isinstance(raw_name, str) else ""
                    if not name:
                        continue

                    allowed_params: set[str] = set()
                    parameters = spec.get("parameters")
                    if isinstance(parameters, dict):
                        allowed_params = _advertised_root_params(parameters, strictify=strictify)

                    spec_payload = dict(spec)
                    spec_payload["name"] = name
                    server_payload = dict(server)
                    with contextlib.suppress(Exception):
                        server_payload.pop("specs", None)

                    registry_key = f"{name}::direct"
                    direct_registry[registry_key] = {
                        "spec": spec_payload,
                        "direct": True,
                        "server": server_payload,
                        "callable": _browser_call(allowed_params, name, server_payload, event_call),
                        "origin_key": registry_key,
                    }
                except Exception:
                    self.logger.debug("Skipping malformed direct tool spec", exc_info=True)
                    continue

            return direct_registry
        except Exception:
            self.logger.debug("Direct tool server registry build failed", exc_info=True)
            return {}

    async def _notify_tool_breaker(
        self,
        context: _ToolExecutionContext,
        tool_type: str,
        tool_name: str | None,
    ) -> None:
        """Emit notification when tool is skipped due to circuit breaker."""
        if not context.event_emitter:
            return
        try:
            await context.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            f"Skipping {tool_name or tool_type} tools due to repeated failures"
                        ),
                        "done": False,
                    },
                }
            )
        except Exception:
            self.logger.debug("Failed to emit breaker notification", exc_info=True)

    def _build_tool_output(
        self,
        call: dict[str, Any],
        output_text: str,
        *,
        status: str = "completed",
        files: list[dict[str, Any]] | None = None,
        embeds: list[str] | None = None,
        pictures: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build standardized tool output payload.

        Args:
            call: Original tool call dict
            output_text: Tool output or error message
            status: Execution status (completed, failed, skipped, etc.)
            files: Optional list of extracted files (images, audio, etc.)
            embeds: Optional list of HTML embed strings

        Returns:
            Responses API compatible tool output item with optional files/embeds
        """
        call_id = call.get("call_id") or generate_item_id()
        normalized_status = status if status in TOOL_CALL_STATUSES else "incomplete"
        result: dict[str, Any] = {
            "type": "function_call_output",
            "id": generate_item_id(),
            "status": normalized_status,
            "call_id": call_id,
            "output": picture_output(output_text, pictures) if pictures else output_text,
        }
        if files:
            result["files"] = files
        if embeds:
            result["embeds"] = embeds
        return result


    async def _tool_pictures_safe(
        self, files: list[dict[str, Any]], context: _ToolExecutionContext
    ) -> tuple[list[str], list[dict[str, Any]]]:
        pictures: list[str] = []
        shown: list[dict[str, Any]] = []
        for entry in files:
            url = entry.get("url") if isinstance(entry, dict) else None
            if isinstance(entry, dict) and entry.get("type") == "image" and isinstance(url, str) and url.startswith("data:"):
                pictures.append(await self._stored_picture_safe(url, context))
                continue
            if isinstance(entry, dict) and entry_data_url(entry):
                filed = await self._stored_data_entry_safe(entry, context)
                if isinstance(filed, str):
                    pictures.append(filed)
                elif filed is not None:
                    shown.append(filed)
                continue
            shown.append(entry)
            if isinstance(entry, dict) and entry.get("type") == "image" and isinstance(url, str) and url:
                pictures.append(url)
        return pictures, shown

    async def _stored_data_entry_safe(
        self, entry: dict[str, Any], context: _ToolExecutionContext
    ) -> dict[str, Any] | None:
        url = entry_data_url(entry)
        if not url:
            return None
        metadata = context.metadata or {}
        if not is_linkable_chat(metadata.get("chat_id")):
            return None
        parsed = split_base64_data_url(url)
        if parsed is None:
            return None
        header, payload = parsed
        mime_type = header.partition(";")[0].strip().removeprefix("data:").strip()
        if not mime_type:
            mime_type = "application/octet-stream"
        if not self._pipe._file_gateway.validate_base64_size(payload):
            return None
        try:
            raw = await _decode_data_entry(payload)
        except (binascii.Error, ValueError):
            return None
        stored = await self._upload_data_entry_safe(raw, mime_type, context)
        if not isinstance(stored, str) or not stored or stored.startswith("data:"):
            return None
        return {"type": "file", "url": stored, "name": _entry_file_name(mime_type)}

    async def _upload_data_entry_safe(
        self, raw: bytes, mime_type: str, context: _ToolExecutionContext
    ) -> str | None:
        gateway = self._pipe._file_gateway
        try:
            user_obj = await _resolved_user_obj(context)
        except Exception:
            self.logger.debug(
                "Could not resolve a user for a tool's data entry; it is not filed",
                exc_info=True,
            )
            return None
        request, user = await gateway.resolve_storage_context(context.request, user_obj)
        if not request or not user:
            return None
        metadata = context.metadata or {}
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        try:
            return await gateway.upload_to_owui_storage(
                request=request,
                user=user,
                file_data=raw,
                filename=_entry_file_name(mime_type),
                mime_type=mime_type,
                chat_id=chat_id if isinstance(chat_id, str) else None,
                message_id=message_id if isinstance(message_id, str) else None,
                owui_user_id=context.user_id,
            )
        except Exception:
            self.logger.debug(
                "Could not store a tool's data entry; it is dropped rather than inlined",
                exc_info=True,
            )
            return None

    async def _stored_picture_safe(self, url: str, context: _ToolExecutionContext) -> str:
        if _owui_store_tool_result_image is None:
            return url
        try:
            user_obj = await _resolved_user_obj(context)
            stored = await _owui_store_tool_result_image(context.request, url, context.metadata, user_obj)
        except Exception:
            self.logger.debug("Could not store a tool's picture; it stays inline", exc_info=True)
            return url
        return stored if isinstance(stored, str) and stored else url

    def _cancelled_tool_output(self, call: dict[str, Any]) -> dict[str, Any]:
        return self._build_tool_output(call, "Tool execution cancelled", status="cancelled")

    @timed
    async def _tool_worker_loop(self, context: _ToolExecutionContext) -> None:
        """Process queued tool calls with batching/timeouts."""
        while True:
            batch = await context.queue.get()
            try:
                if batch is None:
                    return
                await self._pipe._execute_tool_batch(batch, context)
            finally:
                for leftover in batch or ():
                    if not leftover.future.done():
                        leftover.future.set_result(self._cancelled_tool_output(leftover.call))
                context.queue.task_done()

    def _can_batch_tool_calls(self, first: _QueuedToolCall, candidate: _QueuedToolCall) -> bool:
        """Check if two tool calls can be batched together."""
        if first.call.get("name") != candidate.call.get("name"):
            return False

        dep_keys = {"depends_on", "_depends_on", "sequential", "no_batch"}
        if any(key in first.args or key in candidate.args for key in dep_keys):
            return False

        first_id = first.call.get("call_id")
        candidate_id = candidate.call.get("call_id")
        if first_id and self._args_reference_call(candidate.args, first_id):
            return False
        return not (candidate_id and self._args_reference_call(first.args, candidate_id))

    def _args_reference_call(self, args: Any, call_id: str) -> bool:
        """Check if args contain a reference to another call_id."""
        if isinstance(args, str):
            return call_id in args
        if isinstance(args, dict):
            return any(self._args_reference_call(value, call_id) for value in args.values())
        if isinstance(args, list):
            return any(self._args_reference_call(item, call_id) for item in args)
        return False
