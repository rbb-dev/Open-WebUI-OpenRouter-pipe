"""Tool execution orchestrator for the OpenRouter pipe.

This module handles:
- Tool call execution via queue/worker pipeline
- Direct tool server registry building (Socket.IO bridge)
- Legacy direct execution fallback
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ..storage.persistence import generate_item_id
from ..api.transforms import ResponsesBody
from ..core.timing_logger import timed, timing_mark

if TYPE_CHECKING:
    from ..pipe import Pipe
    from starlette.requests import Request

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

# Import process_tool_result from OpenWebUI (>= 0.7.0)
# Falls back to None if not available - we handle gracefully
try:
    from open_webui.utils.middleware import process_tool_result as _owui_process_tool_result
except ImportError:
    _owui_process_tool_result = None  # type: ignore[assignment]

# Import Users model for dict→UserModel conversion (required by process_tool_result)
try:
    from open_webui.models.users import Users as _Users
except ImportError:
    _Users = None  # type: ignore[assignment,misc]

@dataclass(slots=True)
class _QueuedToolCall:
    """Stores a pending tool call plus execution metadata for worker pools."""
    call: dict[str, Any]
    tool_cfg: dict[str, Any]
    args: dict[str, Any]
    future: asyncio.Future
    allow_batch: bool


@dataclass(slots=True)
class _ToolExecutionContext:
    """Holds shared state for executing tool calls within breaker limits."""
    queue: asyncio.Queue[_QueuedToolCall | None]
    per_request_semaphore: asyncio.Semaphore
    global_semaphore: asyncio.Semaphore | None
    timeout: float
    batch_timeout: float | None
    idle_timeout: float | None
    user_id: str
    event_emitter: "EventEmitter | None"
    batch_cap: int
    # Phase 3: Add request/user/metadata for process_tool_result() integration
    request: "Request | None" = None
    user: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    workers: list[asyncio.Task] = field(default_factory=list)
    timeout_error: Optional[str] = None


class ToolExecutor:
    """Orchestrates tool execution and direct tool server integration."""

    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize tool executor.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging and warnings
        """
        self._pipe = pipe
        self.logger = logger
        self._legacy_tool_warning_emitted = False

    # ------------------------------------------------------------------
    # Argument parsing helpers
    # ------------------------------------------------------------------

    def _is_batchable_tool_call(self, args: dict[str, Any]) -> bool:
        """Check if a tool call can be batched or must run sequentially.

        Args:
            args: Parsed tool arguments dictionary

        Returns:
            True if the tool call can be batched, False if it has dependency markers
        """
        blockers = {"depends_on", "_depends_on", "sequential", "no_batch"}
        return not any(key in args for key in blockers)

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        """Parse raw tool arguments into a dictionary.

        Args:
            raw_args: Raw arguments (dict, JSON string, or other)

        Returns:
            Parsed arguments dictionary

        Raises:
            ValueError: If arguments cannot be parsed
        """
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError("Unable to parse tool arguments") from exc
        raise ValueError(f"Unsupported argument type: {type(raw_args).__name__}")

    @timed
    async def _process_tool_result_safe(
        self,
        tool_name: str,
        tool_type: str,
        raw_result: Any,
        context: "_ToolExecutionContext | None",
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
            # If OpenWebUI's process_tool_result is available and we have context, use it
            if _owui_process_tool_result is not None and context is not None:
                try:
                    # Convert user dict to UserModel if needed (OWUI uses attribute access)
                    user_obj = context.user
                    if isinstance(user_obj, dict) and _Users is not None:
                        user_id = user_obj.get("id")
                        if user_id:
                            # Use thread pool to avoid blocking event loop on sync DB call
                            user_obj = await asyncio.to_thread(_Users.get_user_by_id, user_id)
                    processed_result, files, embeds = _owui_process_tool_result(
                        request=context.request,
                        tool_function_name=tool_name,
                        tool_result=raw_result,
                        tool_type=tool_type,
                        direct_tool=is_direct_tool,
                        metadata=context.metadata,
                        user=user_obj,
                    )
                    # process_tool_result returns stringified result
                    output_text = "" if processed_result is None else str(processed_result)
                    timing_mark(f"process_result:{tool_name}:owui_done")
                    return output_text, files, embeds
                except Exception as proc_exc:
                    # OpenWebUI's function failed - fall back to simple conversion
                    self.logger.debug(
                        "process_tool_result failed for '%s', falling back to str(): %s",
                        tool_name,
                        proc_exc,
                    )
                    # Continue to fallback below

            # Fallback: simple str() conversion (always works, never crashes)
            output_text = "" if raw_result is None else str(raw_result)
            timing_mark(f"process_result:{tool_name}:fallback_done")
            return output_text, files, embeds

        except Exception as exc:
            # Ultimate safety net - should never reach here but guarantees no crash
            self.logger.warning(
                "Unexpected error processing result for '%s': %s",
                tool_name,
                exc,
            )
            try:
                output_text = "" if raw_result is None else str(raw_result)
            except Exception:
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
            self.logger.debug("Using legacy tool execution path")
            # Fallback: legacy direct execution
            return await self._execute_function_calls_legacy(calls, tools)

        loop = asyncio.get_running_loop()
        pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        outputs: list[dict[str, Any]] = []
        enqueued_any = False
        breaker_only_skips = True

        for call in calls:
            raw_name = call.get("name")
            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not tool_name:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        "Tool call missing name",
                        status="failed",
                    )
                )
                continue
            tool_cfg = tools.get(tool_name)
            if not tool_cfg:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        "Tool not found",
                        status="failed",
                    )
                )
                continue
            tool_type = (tool_cfg.get("type") or "function").lower()
            if not self._pipe._circuit_breaker.tool_allows(context.user_id, tool_type):
                await self._notify_tool_breaker(context, tool_type, call.get("name"))
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' skipped due to repeated failures.",
                        status="skipped",
                    )
                )
                continue
            fn = tool_cfg.get("callable")
            if fn is None:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' has no callable configured.",
                        status="failed",
                    )
                )
                continue
            try:
                raw_args_value = call.get("arguments")
                if isinstance(raw_args_value, str) and not raw_args_value.strip():
                    # Avoid silently converting empty-string args to `{}` when the tool declares
                    # required parameters (common OpenRouter `/responses` streaming quirk).
                    required: list[str] = []
                    spec = tool_cfg.get("spec")
                    if isinstance(spec, dict):
                        params = spec.get("parameters")
                        if isinstance(params, dict):
                            req = params.get("required")
                            if isinstance(req, list):
                                required = [r for r in req if isinstance(r, str) and r.strip()]
                    if required:
                        raise ValueError("Missing tool arguments (provider sent empty string)")
                    raw_args_value = "{}"
                if raw_args_value is None:
                    raw_args_value = "{}"
                args = self._parse_tool_arguments(raw_args_value)
            except Exception as exc:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Invalid arguments: {exc}",
                        status="failed",
                    )
                )
                continue

            future: asyncio.Future = loop.create_future()
            allow_batch = self._is_batchable_tool_call(args)
            queued = _QueuedToolCall(
                call=call,
                tool_cfg=tool_cfg,
                args=args,
                future=future,
                allow_batch=allow_batch,
            )
            await context.queue.put(queued)
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
            pending.append((call, future))
            enqueued_any = True
            breaker_only_skips = False

        if not enqueued_any and breaker_only_skips and context.user_id:
            self._pipe._circuit_breaker.record_failure(context.user_id)

        for call, future in pending:
            try:
                if context and context.idle_timeout:
                    result = await asyncio.wait_for(future, timeout=context.idle_timeout)
                else:
                    result = await future
            except asyncio.TimeoutError:
                message = (
                    f"Tool '{call.get('name')}' idle timeout after {context.idle_timeout:.0f}s."
                    if context and context.idle_timeout
                    else "Tool idle timeout exceeded."
                )
                if context:
                    context.timeout_error = context.timeout_error or message
                raise RuntimeError(message)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Tool '%s' raised while awaiting result (call_id=%s).",
                        call.get("name"),
                        call.get("call_id"),
                        exc_info=True,
                    )
                result = self._build_tool_output(
                    call,
                    f"Tool error: {exc}",
                    status="failed",
                )
            outputs.append(result)

        if context and context.timeout_error:
            raise RuntimeError(context.timeout_error)

        return outputs

    @timed
    def _build_direct_tool_server_registry(
        self,
        __metadata__: dict[str, Any],
        *,
        valves: "Pipe.Valves",
        event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        event_emitter: "EventEmitter | None",
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Return OWUI-style "direct tool server" entries (callables + tool specs).

        Open WebUI direct tool servers are executed client-side via Socket.IO.
        The model-visible tool names are plain OpenAPI ``operationId`` values
        (no namespacing). Collisions are preserved here and resolved later by
        the pipe's collision-safe tool registry builder.
        """

        direct_registry: dict[str, dict[str, Any]] = {}
        direct_tool_specs: list[dict[str, Any]] = []

        try:
            if not isinstance(__metadata__, dict):
                return {}, []
            tool_servers = __metadata__.get("tool_servers")
            if not isinstance(tool_servers, list) or not tool_servers:
                return {}, []
            if event_call is None:
                # No Socket.IO bridge means direct tools cannot run; don't advertise them.
                return {}, []

            for server_idx, server in enumerate(tool_servers):
                try:
                    if not isinstance(server, dict):
                        continue
                    specs = server.get("specs")
                    if not isinstance(specs, list) or not specs:
                        # Best-effort fallback: derive specs from raw OpenAPI if present.
                        openapi = server.get("openapi")
                        if isinstance(openapi, dict):
                            try:
                                from open_webui.utils.tools import convert_openapi_to_tool_payload  # type: ignore
                            except Exception:
                                convert_openapi_to_tool_payload = None  # type: ignore[assignment]
                            if callable(convert_openapi_to_tool_payload):
                                try:
                                    specs = convert_openapi_to_tool_payload(openapi)  # type: ignore[misc]
                                except Exception:
                                    specs = []
                    if not isinstance(specs, list) or not specs:
                        continue

                    server_payload = dict(server)
                    with contextlib.suppress(Exception):
                        server_payload.pop("specs", None)

                    for spec_idx, spec in enumerate(specs):
                        try:
                            if not isinstance(spec, dict):
                                continue
                            raw_name = spec.get("name")
                            name = raw_name.strip() if isinstance(raw_name, str) else ""
                            if not name:
                                continue

                            allowed_params: set[str] = set()
                            try:
                                parameters = spec.get("parameters")
                                if isinstance(parameters, dict):
                                    props = parameters.get("properties")
                                    if isinstance(props, dict):
                                        allowed_params = {k for k in props.keys() if isinstance(k, str)}
                            except Exception:
                                allowed_params = set()

                            spec_payload = dict(spec)
                            spec_payload["name"] = name

                            async def _direct_tool_callable(  # noqa: ANN001 - tool kwargs are dynamic
                                _allowed_params: set[str] = allowed_params,
                                _tool_name: str = name,
                                _server_payload: dict[str, Any] = server_payload,
                                _metadata: dict[str, Any] = __metadata__,
                                _event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None = event_call,
                                _event_emitter: "EventEmitter | None" = event_emitter,
                                **kwargs,
                            ) -> Any:
                                try:
                                    filtered: dict[str, Any] = {}
                                    try:
                                        filtered = {k: v for k, v in kwargs.items() if k in _allowed_params}
                                    except Exception:
                                        filtered = {}

                                    session_id = None
                                    try:
                                        session_id = _metadata.get("session_id")
                                    except Exception:
                                        session_id = None

                                    payload = {
                                        "type": "execute:tool",
                                        "data": {
                                            "id": str(uuid.uuid4()),
                                            "name": _tool_name,
                                            "params": filtered,
                                            "server": _server_payload,
                                            "session_id": session_id,
                                        },
                                    }
                                    if _event_call is None:
                                        return [{"error": "Direct tool execution unavailable."}, None]
                                    return await _event_call(payload)  # type: ignore[misc]
                                except Exception as exc:
                                    # Never let tool failures crash the pipe/session.
                                    self.logger.debug("Direct tool '%s' failed: %s", _tool_name, exc, exc_info=True)
                                    with contextlib.suppress(Exception):
                                        await self._pipe._event_emitter_handler._emit_notification(
                                            _event_emitter,
                                            f"Tool '{_tool_name}' failed: {exc}",
                                            level="warning",
                                        )
                                    return [{"error": str(exc)}, None]

                            registry_key = f"{name}::{server_idx}::{spec_idx}"
                            direct_registry[registry_key] = {
                                "spec": spec_payload,
                                "direct": True,
                                "server": server_payload,
                                "callable": _direct_tool_callable,
                                "origin_key": registry_key,
                            }
                        except Exception:
                            # Skip malformed tool specs safely.
                            self.logger.debug("Skipping malformed direct tool spec", exc_info=True)
                            continue
                except Exception:
                    # Skip malformed server entries safely.
                    self.logger.debug("Skipping malformed direct tool server entry", exc_info=True)
                    continue

            if direct_registry:
                try:
                    direct_tool_specs = ResponsesBody.transform_owui_tools(
                        direct_registry,
                        strict=bool(valves.ENABLE_STRICT_TOOL_CALLING)
                        and (getattr(valves, "TOOL_EXECUTION_MODE", "Pipeline") != "Open-WebUI"),
                    )
                except Exception:
                    direct_tool_specs = []
            return direct_registry, direct_tool_specs
        except Exception:
            self.logger.debug("Direct tool server registry build failed", exc_info=True)
            return {}, []

    @timed
    async def _execute_function_calls_legacy(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Legacy direct execution path used when tool context is unavailable.

        Note: In legacy mode, we don't have full context (request/user/metadata)
        so file/embed extraction from process_tool_result() won't work fully.
        This is expected - legacy mode is a fallback.
        """
        if not self._legacy_tool_warning_emitted:
            self._legacy_tool_warning_emitted = True
            self.logger.warning("Tool queue unavailable; falling back to direct execution.")

        # Track tasks with their metadata for result processing
        task_entries: list[tuple[dict, dict[str, Any] | None, Awaitable]] = []

        for call in calls:
            timing_mark(f"legacy_tool_prep:{call.get('name', 'unknown')}")
            try:
                raw_name = call.get("name")
                tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
                if not tool_name:
                    task_entries.append((call, None, asyncio.sleep(0, result=RuntimeError("Tool call missing name"))))
                    continue

                tool_cfg = tools.get(tool_name)
                if not tool_cfg:
                    task_entries.append((call, None, asyncio.sleep(0, result=RuntimeError("Tool not found"))))
                    continue

                fn = tool_cfg.get("callable")
                if fn is None:
                    task_entries.append((call, tool_cfg, asyncio.sleep(0, result=RuntimeError("Tool has no callable configured"))))
                    continue

                raw_args_value = call.get("arguments")
                if isinstance(raw_args_value, str) and not raw_args_value.strip():
                    required: list[str] = []
                    try:
                        spec = tool_cfg.get("spec")
                        if isinstance(spec, dict):
                            params = spec.get("parameters")
                            if isinstance(params, dict):
                                req = params.get("required")
                                if isinstance(req, list):
                                    required = [r for r in req if isinstance(r, str) and r.strip()]
                    except Exception:
                        pass  # Non-critical - continue with empty required list
                    if required:
                        task_entries.append((call, tool_cfg, asyncio.sleep(0, result=RuntimeError("Missing tool arguments (empty string)"))))
                        continue
                    raw_args_value = "{}"

                raw_args = raw_args_value if raw_args_value is not None else "{}"
                try:
                    args = self._parse_tool_arguments(raw_args)
                except Exception as exc:
                    task_entries.append((call, tool_cfg, asyncio.sleep(0, result=RuntimeError(f"Invalid arguments: {exc}"))))
                    continue

                if inspect.iscoroutinefunction(fn):
                    task_entries.append((call, tool_cfg, fn(**args)))
                else:
                    task_entries.append((call, tool_cfg, asyncio.to_thread(fn, **args)))

            except Exception as prep_exc:
                # Safety net for any unexpected error during preparation
                self.logger.debug("Tool preparation failed for '%s': %s", call.get("name"), prep_exc)
                task_entries.append((call, None, asyncio.sleep(0, result=RuntimeError(f"Preparation error: {prep_exc}"))))

        # Execute all tasks, catching exceptions per-task
        tasks = [entry[2] for entry in task_entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs: list[dict] = []
        for (call, tool_cfg, _), result in zip(task_entries, results):
            tool_name = call.get("name", "unknown")
            timing_mark(f"legacy_tool_result:{tool_name}")

            try:
                if isinstance(result, Exception):
                    status = "failed"
                    output_text = f"Tool error: {result}"
                else:
                    status = "completed"
                    # Get tool type for process_tool_result
                    tool_type = "function"
                    if tool_cfg is not None:
                        try:
                            tool_type = (tool_cfg.get("type") or "function").lower()
                        except Exception:
                            pass

                    # Use safe result processing (no context in legacy mode)
                    is_direct = bool(tool_cfg.get("direct")) if tool_cfg else False
                    output_text, files, embeds = await self._process_tool_result_safe(
                        tool_name=tool_name,
                        tool_type=tool_type,
                        raw_result=result,
                        context=None,  # Legacy mode has no context
                        is_direct_tool=is_direct,
                    )

                    # Log if we extracted files/embeds (won't be emitted in legacy mode)
                    if files or embeds:
                        self.logger.debug(
                            "Tool '%s' produced %d files, %d embeds (not emitted in legacy mode)",
                            tool_name, len(files), len(embeds),
                        )

                outputs.append(
                    self._build_tool_output(
                        call,
                        output_text,
                        status=status,
                    )
                )

            except Exception as output_exc:
                # Safety net - ensure we always produce an output
                self.logger.warning("Failed to build output for '%s': %s", tool_name, output_exc)
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Tool output error: {output_exc}",
                        status="failed",
                    )
                )

        return outputs

    async def _notify_tool_breaker(
        self,
        context: "_ToolExecutionContext",
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
            # Event emitter failures (client disconnect, etc.) shouldn't stop pipe
            self.logger.debug("Failed to emit breaker notification", exc_info=True)

    def _build_tool_output(
        self,
        call: dict[str, Any],
        output_text: str,
        *,
        status: str = "completed",
        files: list[dict[str, Any]] | None = None,
        embeds: list[str] | None = None,
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
        # OpenRouter Responses schema does not accept arbitrary status values (e.g. "failed")
        # for tool items in `input`. Encode failures in the output payload and keep status in
        # the accepted enum for compatibility.
        allowed_statuses = {"completed", "incomplete", "in_progress"}
        normalized_status = status if status in allowed_statuses else "completed"
        result: dict[str, Any] = {
            "type": "function_call_output",
            "id": generate_item_id(),
            "status": normalized_status,
            "call_id": call_id,
            "output": output_text,
        }
        # Include files/embeds if provided (for tool card HTML attributes)
        if files:
            result["files"] = files
        if embeds:
            result["embeds"] = embeds
        return result

    # ------------------------------------------------------------------
    # Tool worker loop and batching helpers
    # ------------------------------------------------------------------

    @timed
    async def _tool_worker_loop(self, context: _ToolExecutionContext) -> None:
        """Process queued tool calls with batching/timeouts."""
        pending: list[tuple[_QueuedToolCall | None, bool]] = []
        batch_cap = context.batch_cap
        idle_timeout = context.idle_timeout
        try:
            while True:
                if pending:
                    item, from_queue = pending.pop(0)
                else:
                    try:
                        get_coro = context.queue.get()
                        queued = (
                            await get_coro
                            if idle_timeout is None
                            else await asyncio.wait_for(get_coro, timeout=idle_timeout)
                        )
                    except asyncio.TimeoutError:
                        message = (
                            f"Tool queue idle for {idle_timeout:.0f}s; cancelling pending work."
                            if idle_timeout
                            else "Tool queue idle timeout triggered."
                        )
                        context.timeout_error = context.timeout_error or message
                        self.logger.warning("%s", message)
                        break
                    item, from_queue = queued, True
                if item is None:
                    if from_queue:
                        context.queue.task_done()
                    if pending:
                        continue
                    break

                batch: list[tuple[_QueuedToolCall, bool]] = [(item, from_queue)]
                if item.allow_batch:
                    while len(batch) < batch_cap:
                        try:
                            nxt = context.queue.get_nowait()
                            from_queue_next = True
                        except asyncio.QueueEmpty:
                            break
                        if nxt is None:
                            pending.insert(0, (None, True))
                            break
                        if nxt.allow_batch and self._can_batch_tool_calls(batch[0][0], nxt):
                            batch.append((nxt, from_queue_next))
                        else:
                            pending.insert(0, (nxt, from_queue_next))
                            break

                await self._pipe._execute_tool_batch([itm for itm, _ in batch], context)
                for _, flag in batch:
                    if flag:
                        context.queue.task_done()
        finally:
            # Resolve remaining futures if the worker is stopping unexpectedly
            while pending:
                leftover, from_queue = pending.pop(0)
                if from_queue:
                    context.queue.task_done()
                if leftover is None:
                    continue
                if not leftover.future.done():
                    error_msg = context.timeout_error or "Tool execution cancelled"
                    leftover.future.set_result(
                        self._build_tool_output(
                            leftover.call,
                            error_msg,
                            status="cancelled",
                        )
                    )

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
        if candidate_id and self._args_reference_call(first.args, candidate_id):
            return False
        return True

    def _args_reference_call(self, args: Any, call_id: str) -> bool:
        """Check if args contain a reference to another call_id."""
        if isinstance(args, str):
            return call_id in args
        if isinstance(args, dict):
            return any(self._args_reference_call(value, call_id) for value in args.values())
        if isinstance(args, list):
            return any(self._args_reference_call(item, call_id) for item in args)
        return False
