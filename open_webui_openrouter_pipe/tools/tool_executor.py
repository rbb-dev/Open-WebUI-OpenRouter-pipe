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
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ..storage.persistence import generate_item_id
from ..api.transforms import ResponsesBody
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

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
    workers: list[asyncio.Task] = field(default_factory=list)
    timeout_error: Optional[str] = None


class ToolExecutor:
    """Orchestrates tool execution and direct tool server integration."""

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize tool executor.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging and warnings
        """
        self._pipe = pipe
        self.logger = logger
        self._legacy_tool_warning_emitted = False

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
            if not self._pipe._tool_type_allows(context.user_id, tool_type):
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
                args = self._pipe._parse_tool_arguments(raw_args_value)
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
            allow_batch = self._pipe._is_batchable_tool_call(args)
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
            self._pipe._record_failure(context.user_id)

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

                            @timed
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
                                        await self._pipe._emit_notification(
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
        """Legacy direct execution path used when tool context is unavailable."""

        if not self._legacy_tool_warning_emitted:
            self._legacy_tool_warning_emitted = True
            self.logger.warning("Tool queue unavailable; falling back to direct execution.")

        tasks: list[Awaitable] = []
        for call in calls:
            raw_name = call.get("name")
            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not tool_name:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool call missing name")))
                continue
            tool_cfg = tools.get(tool_name)
            if not tool_cfg:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool not found")))
                continue
            fn = tool_cfg.get("callable")
            if fn is None:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool has no callable configured")))
                continue
            raw_args_value = call.get("arguments")
            if isinstance(raw_args_value, str) and not raw_args_value.strip():
                required: list[str] = []
                spec = tool_cfg.get("spec")
                if isinstance(spec, dict):
                    params = spec.get("parameters")
                    if isinstance(params, dict):
                        req = params.get("required")
                        if isinstance(req, list):
                            required = [r for r in req if isinstance(r, str) and r.strip()]
                if required:
                    tasks.append(asyncio.sleep(0, result=RuntimeError("Missing tool arguments (empty string)")))
                    continue
                raw_args_value = "{}"
            raw_args = raw_args_value if raw_args_value is not None else "{}"
            try:
                args = self._pipe._parse_tool_arguments(raw_args)
            except Exception as exc:
                tasks.append(asyncio.sleep(0, result=RuntimeError(f"Invalid arguments: {exc}")))
                continue
            if inspect.iscoroutinefunction(fn):
                tasks.append(fn(**args))
            else:
                tasks.append(asyncio.to_thread(fn, **args))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[dict] = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                status = "failed"
                output_text = f"Tool error: {result}"
            else:
                status = "completed"
                output_text = "" if result is None else str(result)
            outputs.append(
                self._build_tool_output(
                    call,
                    output_text,
                    status=status,
                )
            )
        return outputs

    @timed
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

    @timed
    def _build_tool_output(
        self,
        call: dict[str, Any],
        output_text: str,
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Build standardized tool output payload.

        Args:
            call: Original tool call dict
            output_text: Tool output or error message
            status: Execution status (completed, failed, skipped, etc.)

        Returns:
            Responses API compatible tool output item
        """
        call_id = call.get("call_id") or generate_item_id()
        # OpenRouter Responses schema does not accept arbitrary status values (e.g. "failed")
        # for tool items in `input`. Encode failures in the output payload and keep status in
        # the accepted enum for compatibility.
        allowed_statuses = {"completed", "incomplete", "in_progress"}
        normalized_status = status if status in allowed_statuses else "completed"
        return {
            "type": "function_call_output",
            "id": generate_item_id(),
            "status": normalized_status,
            "call_id": call_id,
            "output": output_text,
        }
