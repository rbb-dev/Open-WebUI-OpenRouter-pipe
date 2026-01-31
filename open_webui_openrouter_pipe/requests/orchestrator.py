"""Request Processing Orchestrator for OpenRouter.

This module handles the main request processing logic after transformation.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Awaitable, Literal

import aiohttp
from starlette.requests import Request

# Import types needed for request processing
from ..core.config import _ORS_FILTER_FEATURE_FLAG
from ..core.errors import (
    OpenRouterAPIError,
    _is_reasoning_effort_error,
    _parse_supported_effort_values,
    _select_best_effort_fallback,
)
from ..tools.tool_registry import _build_collision_safe_tool_specs_and_registry
from ..models.registry import ModelFamily, OpenRouterModelRegistry
from ..api.transforms import CompletionsBody, ResponsesBody, _chat_tools_to_responses_tools
from ..core.timing_logger import timed, timing_scope

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..streaming.event_emitter import EventEmitter


class RequestOrchestrator:
    """Orchestrates the processing of transformed OpenRouter requests."""

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize RequestOrchestrator.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging
        """
        self._pipe = pipe
        self.logger = logger

    @timed
    async def process_request(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Any,
        __task_body__: Any,
        valves: "Pipe.Valves",
        session: aiohttp.ClientSession,
        openwebui_model_id: str,
        pipe_identifier: str,
        allowlist_norm_ids: set[str],
        enforced_norm_ids: set[str],
        catalog_norm_ids: set[str],
        features: dict[str, Any],
        *,
        user_id: str = "",
    ) -> AsyncGenerator[str, None] | dict[str, Any] | str | None:
        @timed
        def _extract_direct_uploads(metadata: dict[str, Any]) -> dict[str, Any]:
            pipe_meta = metadata.get("openrouter_pipe")
            if not isinstance(pipe_meta, dict):
                return {}
            attachments = pipe_meta.get("direct_uploads")
            if not isinstance(attachments, dict):
                return {}

            extracted: dict[str, Any] = {"files": [], "audio": [], "video": []}
            allowlist_csv = attachments.get("responses_audio_format_allowlist")
            if isinstance(allowlist_csv, str):
                extracted["responses_audio_format_allowlist"] = allowlist_csv
            pdf_parser = attachments.get("pdf_parser")
            if isinstance(pdf_parser, str) and pdf_parser.strip():
                extracted["pdf_parser"] = pdf_parser.strip()
            for key in ("files", "audio", "video"):
                items = attachments.get(key)
                if not isinstance(items, list):
                    continue
                seen: set[str] = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    file_id = item.get("id")
                    if not isinstance(file_id, str) or not file_id.strip():
                        continue
                    file_id = file_id.strip()
                    if file_id in seen:
                        continue
                    seen.add(file_id)
                    cleaned = dict(item)
                    cleaned["id"] = file_id
                    extracted[key].append(cleaned)
            extracted = {k: v for k, v in extracted.items() if v}
            return extracted

        @timed
        async def _inject_direct_uploads_into_messages(
            request_body: dict[str, Any],
            attachments: dict[str, Any],
        ) -> None:
            messages = request_body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("Direct uploads require a chat messages payload.")

            last_user_msg = None
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg
                    break
            if not isinstance(last_user_msg, dict):
                raise ValueError("Direct uploads require at least one user message.")

            content = last_user_msg.get("content")
            content_blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    content_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        content_blocks.append(part)
            elif content is None:
                content_blocks = []
            else:
                raise ValueError("Direct uploads require a supported message content type.")

            max_bytes = int(valves.BASE64_MAX_SIZE_MB) * 1024 * 1024
            chunk_size = int(valves.IMAGE_UPLOAD_CHUNK_BYTES)

            @timed
            def _decode_base64_prefix(data: str, *, byte_count: int = 96) -> bytes:
                """Decode a small prefix of base64 to allow MIME/container sniffing.

                Defensive by design: any anomaly returns empty bytes and forces conservative routing.
                """
                if not data:
                    return b""
                try:
                    wanted = int(byte_count)
                except Exception:
                    wanted = 96
                wanted = max(1, min(wanted, 4096))
                needed = ((wanted + 2) // 3) * 4
                prefix = data[:needed]
                if not prefix:
                    return b""
                base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                for ch in prefix:
                    if ord(ch) > 127 or ch not in base64_chars:
                        return b""
                pad = (-len(prefix)) % 4
                prefix = prefix + ("=" * pad)
                try:
                    decoded = base64.b64decode(prefix, validate=True)
                except Exception:
                    try:
                        decoded = base64.b64decode(prefix, validate=False)
                    except Exception:
                        return b""
                return decoded[:wanted]

            @timed
            def _sniff_audio_format(prefix: bytes) -> str:
                """Best-effort audio container sniff for routing (mp3/wav vs other)."""
                if not prefix:
                    return ""
                if prefix.startswith(b"RIFF") and len(prefix) >= 12 and prefix[8:12] == b"WAVE":
                    return "wav"
                if prefix.startswith(b"ID3"):
                    return "mp3"
                if len(prefix) >= 2 and prefix[0] == 0xFF and (prefix[1] & 0xE0) == 0xE0:
                    return "mp3"
                if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
                    # ISO BMFF container (commonly .m4a for audio uploads).
                    return "m4a"
                if prefix.startswith(b"fLaC"):
                    return "flac"
                if prefix.startswith(b"OggS"):
                    return "ogg"
                if prefix.startswith(b"\x1A\x45\xDF\xA3"):
                    return "webm"
                return ""

            for item in attachments.get("files", []):
                file_id = item.get("id")
                if not isinstance(file_id, str) or not file_id:
                    continue
                name = item.get("name")
                filename = name if isinstance(name, str) else ""
                content_blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "file_id": file_id,
                            "filename": filename,
                        },
                    }
                )

            @timed
            def _csv_set(value: Any) -> set[str]:
                if not isinstance(value, str):
                    return set()
                items = []
                for raw in value.split(","):
                    item = (raw or "").strip().lower()
                    if item:
                        items.append(item)
                return set(items)

            allowlist_key = "responses_audio_format_allowlist"
            allowlist_seen = allowlist_key in attachments
            allowlist_csv = attachments.get(allowlist_key, "") if allowlist_seen else ""
            allowed_for_responses = _csv_set(allowlist_csv) if allowlist_seen else {"mp3", "wav"}

            for item in attachments.get("audio", []):
                file_id = item.get("id")
                if not isinstance(file_id, str) or not file_id:
                    continue
                file_obj = await self._pipe._get_file_by_id(file_id)
                if not file_obj:
                    raise ValueError(f"Native audio attachment '{file_id}' could not be loaded.")
                b64 = await self._pipe._read_file_record_base64(file_obj, chunk_size, max_bytes)
                if not b64:
                    raise ValueError(f"Native audio attachment '{file_id}' could not be encoded.")
                declared = item.get("format")
                declared_format = declared.strip().lower() if isinstance(declared, str) else ""
                sniffed = _sniff_audio_format(_decode_base64_prefix(b64))
                audio_format = (sniffed or declared_format).strip().lower()
                if not audio_format:
                    raise ValueError("Native audio attachment missing required 'format'.")
                # Do not trust upstream metadata; re-sniff and apply the configured /responses eligibility allowlist.
                item["format"] = audio_format
                item["responses_eligible"] = bool(audio_format in allowed_for_responses)
                content_blocks.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": audio_format},
                    }
                )

            for item in attachments.get("video", []):
                file_id = item.get("id")
                if not isinstance(file_id, str) or not file_id:
                    continue
                file_obj = await self._pipe._get_file_by_id(file_id)
                if not file_obj:
                    raise ValueError(f"Native video attachment '{file_id}' could not be loaded.")
                b64 = await self._pipe._read_file_record_base64(file_obj, chunk_size, max_bytes)
                if not b64:
                    raise ValueError(f"Native video attachment '{file_id}' could not be encoded.")
                mime = item.get("content_type")
                if not isinstance(mime, str) or not mime.strip():
                    mime = self._pipe._infer_file_mime_type(file_obj)
                data_url = f"data:{mime.strip()};base64,{b64}"
                content_blocks.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": data_url},
                    }
                )

            last_user_msg["content"] = content_blocks

        user_id = user_id or str(__user__.get("id") or __metadata__.get("user_id") or "")
        chat_id = (__metadata__ or {}).get("chat_id")
        chat_id = chat_id.strip() if isinstance(chat_id, str) else ""
        task_name = self._pipe._task_name(__task__) if __task__ else ""
        # Optional: inject CSS tweak for multi-line statuses when enabled.
        if valves.ENABLE_STATUS_CSS_PATCH:
            if __event_call__:
                payload_user_id = user_id or __metadata__.get("user_id") or __user__.get("id") or "anonymous"
                try:
                    await __event_call__({
                        "type": "execute",
                        "user_id": str(payload_user_id),
                        "data": {
                            "code": """
                            (() => {
                                if (document.getElementById("owui-status-unclamp")) return "ok";
                                const style = document.createElement("style");
                                style.id = "owui-status-unclamp";
                                style.textContent = `
                                    .status-description .line-clamp-1,
                                    .status-description .text-base.line-clamp-1,
                                    .status-description .text-gray-500.text-base.line-clamp-1 {
                                        display: block !important;
                                        overflow: visible !important;
                                        -webkit-line-clamp: unset !important;
                                        -webkit-box-orient: initial !important;
                                        white-space: pre-wrap !important;
                                        word-break: break-word;
                                    }
                                    .status-description .text-base::first-line,
                                    .status-description .text-gray-500.text-base::first-line {
                                        font-weight: 500 !important;
                                    }
                                `;
                                document.head.appendChild(style);
                                return "ok";
                            })();
                            """
                        }
                    })
                except Exception as exc:  # pragma: no cover - UI injection optional
                    self.logger.debug("Status CSS injection failed: %s", exc)
            else:
                    self.logger.debug("Status CSS injection skipped: __event_call__ unavailable.")

        @timed
        def _extract_direct_uploads_warnings(metadata: dict[str, Any]) -> list[str]:
            pipe_meta = metadata.get("openrouter_pipe")
            if not isinstance(pipe_meta, dict):
                return []
            warnings = pipe_meta.get("direct_uploads_warnings")
            if not isinstance(warnings, list):
                return []
            cleaned: list[str] = []
            seen: set[str] = set()
            for warning in warnings:
                if isinstance(warning, str):
                    msg = warning.strip()
                    if msg and msg not in seen:
                        seen.add(msg)
                        cleaned.append(msg)
            return cleaned

        direct_uploads_warnings = _extract_direct_uploads_warnings(__metadata__ or {})
        if direct_uploads_warnings and not __task__:
            summary = direct_uploads_warnings[0]
            extra_count = max(0, len(direct_uploads_warnings) - 1)
            if extra_count:
                summary = f"{summary} (+{extra_count} more)"
            await self._pipe._emit_notification(
                __event_emitter__,
                f"Direct uploads not applied for some attachments: {summary}",
                level="warning",
            )

        direct_uploads = _extract_direct_uploads(__metadata__ or {})
        if __task__ and direct_uploads:
            # IMPORTANT: Open WebUI task requests (title/tags/follow-ups, web_search query generation, etc.)
            # inherit the originating request metadata (`request.state.metadata`) and therefore may carry our
            # `openrouter_pipe.direct_uploads` marker. Do not inject direct uploads into task calls,
            # but also do not mutate shared metadata (it may be reused by the subsequent main chat request).
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Ignoring direct uploads for task request (task=%s chat_id=%s)",
                    task_name or "task",
                    chat_id,
                )
            direct_uploads = {}
        endpoint_override: Literal["responses", "chat_completions"] | None = None
        if direct_uploads:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Injecting direct uploads into chat request (chat_id=%s files=%d audio=%d video=%d)",
                    chat_id,
                    len(direct_uploads.get("files", []) or []),
                    len(direct_uploads.get("audio", []) or []),
                    len(direct_uploads.get("video", []) or []),
                )
            try:
                await _inject_direct_uploads_into_messages(body, direct_uploads)
            except Exception as exc:
                await self._pipe._emit_templated_error(
                    __event_emitter__,
                    template=valves.DIRECT_UPLOAD_FAILURE_TEMPLATE,
                    variables={
                        "requested_model": body.get("model") or "",
                        "reason": str(exc),
                    },
                    log_message="Direct uploads injection failed",
                    log_level=logging.WARNING,
                )
                return ""

            requires_chat = bool(direct_uploads.get("video"))
            if not requires_chat:
                for audio in direct_uploads.get("audio", []):
                    if isinstance(audio, dict) and not bool(audio.get("responses_eligible", False)):
                        requires_chat = True
                        break

            # Note: /chat/completions supports `type:"file"` blocks (data URLs), so direct files can be carried
            # alongside chat-only modalities when we route the request to /chat/completions.

            if requires_chat:
                selected, forced = self._pipe._select_llm_endpoint_with_forced(body.get("model") or "", valves=valves)
                if forced and selected == "responses":
                    await self._pipe._emit_templated_error(
                        __event_emitter__,
                        template=valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
                        variables={
                            "requested_model": body.get("model") or "",
                            "required_endpoint": "chat_completions",
                            "enforced_endpoint": "responses",
                            "reason": "Direct uploads require /chat/completions but the model is forced to /responses.",
                        },
                        log_message="Endpoint override conflict for direct uploads",
                        log_level=logging.WARNING,
                    )
                    return ""
                endpoint_override = "chat_completions"

        # OpenRouter presets (preset field) only work on /chat/completions, not /responses.
        # Force chat completions when a preset parameter is present in the request body.
        # Extract preset from body top-level OR from params.custom_params (Open WebUI may not merge it)
        preset = body.get("preset")
        if not preset:
            # Try extracting from nested params.custom_params location
            params = body.get("params")
            if isinstance(params, dict):
                custom_params = params.get("custom_params")
                if isinstance(custom_params, dict):
                    preset = custom_params.get("preset")
                    if preset:
                        # Promote to top-level for downstream processing
                        body["preset"] = preset

        if preset and endpoint_override is None:
            selected, forced = self._pipe._select_llm_endpoint_with_forced(body.get("model") or "", valves=valves)
            if forced and selected == "responses":
                await self._pipe._emit_templated_error(
                    __event_emitter__,
                    template=valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
                    variables={
                        "requested_model": body.get("model") or "",
                        "required_endpoint": "chat_completions",
                        "enforced_endpoint": "responses",
                        "reason": "Preset parameter requires /chat/completions but the model is forced to /responses.",
                    },
                    log_message="Endpoint override conflict for preset parameter",
                    log_level=logging.WARNING,
                )
                return ""
            endpoint_override = "chat_completions"
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Forcing /chat/completions endpoint for preset parameter (preset=%s)",
                    body.get("preset"),
                )

        completions_body = CompletionsBody.model_validate(body)

        # Resolve full Open WebUI user model for uploads/status events.
        user_model = None
        if user_id:
            try:
                user_model = await self._pipe._get_user_by_id(user_id)
            except Exception:  # pragma: no cover - defensive guard
                user_model = None

        responses_body = await ResponsesBody.from_completions(
            completions_body=completions_body,

            # If chat_id and openwebui_model_id are provided, from_completions() uses them to fetch previously persisted items (function_calls, reasoning, etc.) from DB and reconstruct the input array in the correct order.
            **({"chat_id": __metadata__["chat_id"]} if __metadata__.get("chat_id") else {}),
            **({"openwebui_model_id": openwebui_model_id} if openwebui_model_id else {}),
            artifact_loader=self._pipe._db_fetch,
            pruning_turns=valves.TOOL_OUTPUT_RETENTION_TURNS,
            transformer_context=self._pipe,
            request=__request__,
            user_obj=user_model,
            event_emitter=__event_emitter__,
            transformer_valves=valves,

        )
        self._pipe._sanitize_request_input(responses_body)
        self._pipe._apply_reasoning_preferences(responses_body, valves)
        self._pipe._apply_gemini_thinking_config(responses_body, valves)
        self._pipe._apply_context_transforms(responses_body, valves)

        # Inject provider routing from filter-injected metadata
        # Only for non-task requests - task models (title, tags, follow-ups) use different models
        # that may not have the same providers available
        if __task__ is None and isinstance(__metadata__, dict):
            pipe_meta = __metadata__.get("openrouter_pipe")
            if isinstance(pipe_meta, dict):
                filter_provider = pipe_meta.get("provider")
                if isinstance(filter_provider, dict) and filter_provider:
                    # Merge filter-injected provider settings with any existing provider settings
                    existing_provider = responses_body.provider or {}
                    if isinstance(existing_provider, dict):
                        merged_provider = {**existing_provider, **filter_provider}
                    else:
                        merged_provider = filter_provider
                    responses_body.provider = merged_provider
                    self.logger.debug("Injected provider routing from filter: %s", filter_provider)

        if valves.USE_MODEL_MAX_OUTPUT_TOKENS:
            if responses_body.max_output_tokens is None:
                default_max = ModelFamily.max_completion_tokens(responses_body.model)
                if default_max:
                    responses_body.max_output_tokens = default_max
        else:
            responses_body.max_output_tokens = None

        normalized_model_id = ModelFamily.base_model(responses_body.model)
        task_mode = bool(__task__)
        if task_mode:
            if allowlist_norm_ids and normalized_model_id not in allowlist_norm_ids:
                self.logger.debug(
                    "Bypassing model whitelist for task request (model=%s, task=%s)",
                    normalized_model_id,
                    self._pipe._task_name(__task__) or "task",
                )
        else:
            # Strip variant suffix from model ID for restriction checks
            # Variants like :nitro, :free, :thinking are routing modifiers, not different models
            base_model_for_check = normalized_model_id.rsplit(":", 1)[0] if ":" in normalized_model_id else normalized_model_id

            if catalog_norm_ids and base_model_for_check not in enforced_norm_ids:
                reasons = self._pipe._model_restriction_reasons(
                    base_model_for_check,
                    valves=valves,
                    allowlist_norm_ids=allowlist_norm_ids,
                    catalog_norm_ids=catalog_norm_ids,
                )
                model_id_filter = (valves.MODEL_ID or "").strip()
                free_mode = (getattr(valves, "FREE_MODEL_FILTER", "all") or "all").strip().lower()
                tool_mode = (getattr(valves, "TOOL_CALLING_FILTER", "all") or "all").strip().lower()
                await self._pipe._emit_templated_error(
                    __event_emitter__,
                    template=valves.MODEL_RESTRICTED_TEMPLATE,
                    variables={
                        "requested_model": responses_body.model,
                        "normalized_model_id": normalized_model_id,
                        "restriction_reasons": ", ".join(reasons) if reasons else "restricted",
                        "model_id_filter": model_id_filter if model_id_filter.lower() != "auto" else "",
                        "free_model_filter": free_mode if free_mode != "all" else "",
                        "tool_calling_filter": tool_mode if tool_mode != "all" else "",
                    },
                    log_message=(
                        f"Model restricted (requested={responses_body.model}, normalized={normalized_model_id}, reasons={reasons})"
                    ),
                )
                return ""
        if not features:
            fallback_caps = (
                ModelFamily.capabilities(openwebui_model_id or "")
                or ModelFamily.capabilities(responses_body.model)
            )
            if fallback_caps:
                features = dict(fallback_caps)

        task_effort = None
        if __task__:
            self.logger.debug("Detected task model: %s", __task__)

            requested_model = responses_body.model
            owns_task_model = ModelFamily.base_model(requested_model) in allowlist_norm_ids if allowlist_norm_ids else True
            if owns_task_model:
                task_effort = valves.TASK_MODEL_REASONING_EFFORT
                self._pipe._apply_task_reasoning_preferences(responses_body, task_effort)
                self._pipe._apply_gemini_thinking_config(responses_body, valves)

            result = await self._pipe._run_task_model_request(
                responses_body.model_dump(),
                valves,
                session=session,
                task_context=__task__,
                owui_metadata=__metadata__,
                user_id=user_id or "",
                user_obj=user_model,
                pipe_id=pipe_identifier,
                snapshot_model_id=self._pipe._qualify_model_for_pipe(
                    pipe_identifier,
                    responses_body.model,
                ),
            )
            return result

        tools_registry = __tools__
        if inspect.isawaitable(tools_registry):
            try:
                tools_registry = await tools_registry
            except Exception as exc:
                self.logger.warning("Tool registry unavailable; continuing without tools: %s", exc)
                await self._pipe._emit_notification(
                    __event_emitter__,
                    "Tool registry unavailable; continuing without tools.",
                    level="warning",
                )
                tools_registry = {}
        __tools__ = tools_registry

        # These are executed client-side via Socket.IO (execute:tool) and must not crash the pipe.
        direct_registry: dict[str, dict[str, Any]] = {}
        direct_tool_specs: list[dict[str, Any]] = []
        try:
            direct_registry, direct_tool_specs = self._pipe._build_direct_tool_server_registry(
                __metadata__,
                valves=valves,
                event_call=__event_call__,
                event_emitter=__event_emitter__,
            )
        except Exception:
            direct_registry, direct_tool_specs = {}, []

        merged_extra_tools: list[dict[str, Any]] = []
        try:
            upstream_extra = getattr(completions_body, "extra_tools", None)
            if isinstance(upstream_extra, list):
                merged_extra_tools.extend([t for t in upstream_extra if isinstance(t, dict)])
        except Exception:
            merged_extra_tools = []
        if not merged_extra_tools:
            merged_extra_tools = []

        owui_tool_passthrough = getattr(valves, "TOOL_EXECUTION_MODE", "Pipeline") == "Open-WebUI"
        incoming_tools_raw = body.get("tools")
        incoming_tools = _chat_tools_to_responses_tools(incoming_tools_raw)
        strictify = bool(valves.ENABLE_STRICT_TOOL_CALLING) and (not owui_tool_passthrough)

        # Normalize OWUI tool registry (__tools__) into a dict form when possible.
        owui_registry: dict[str, dict[str, Any]] = {}
        if isinstance(__tools__, dict):
            owui_registry = {k: v for k, v in __tools__.items() if isinstance(v, dict)}
        elif isinstance(__tools__, list):
            for entry in __tools__:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name and isinstance(entry.get("spec"), dict):
                    name = entry["spec"].get("name")
                if isinstance(name, str) and name.strip():
                    owui_registry[name.strip()] = entry

        # OWUI-native tools are already computed by OWUI middleware and attached to metadata["tools"].
        # Prefer using that executor registry instead of re-synthesizing builtins inside the pipe.
        if isinstance(__metadata__, dict):
            metadata_tools_raw = __metadata__.get("tools")
            if isinstance(metadata_tools_raw, dict):
                metadata_tools = {
                    k: v for k, v in metadata_tools_raw.items() if isinstance(k, str) and isinstance(v, dict)
                }
                if metadata_tools:
                    for name, tool_cfg in metadata_tools.items():
                        if name not in owui_registry:
                            owui_registry[name] = tool_cfg

        tools, exec_registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=incoming_tools if incoming_tools else None,
            owui_registry=owui_registry or None,
            direct_registry=direct_registry or None,
            builtin_registry=None,
            extra_tools=merged_extra_tools or None,
            strictify=strictify,
            owui_tool_passthrough=owui_tool_passthrough,
            logger=self.logger,
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            renames = [
                (exposed, origin)
                for exposed, origin in (exposed_to_origin or {}).items()
                if isinstance(exposed, str)
                and isinstance(origin, str)
                and exposed
                and origin
                and exposed != origin
            ]
            self.logger.debug(
                "Tool ingest: request_tools=%d registry_tools=%d direct_tools=%d extra_tools=%d advertised=%d exec_registry=%d renamed=%d",
                len(incoming_tools) if isinstance(incoming_tools, list) else 0,
                len(owui_registry) if isinstance(owui_registry, dict) else 0,
                len(direct_registry) if isinstance(direct_registry, dict) else 0,
                len(merged_extra_tools) if isinstance(merged_extra_tools, list) else 0,
                len(tools) if isinstance(tools, list) else 0,
                len(exec_registry) if isinstance(exec_registry, dict) else 0,
                len(renames),
            )
            if renames:
                self.logger.debug("Tool renames (exposed->origin): %s", json.dumps(renames, ensure_ascii=False))
        __tools__ = exec_registry
        if isinstance(__metadata__, dict) and exposed_to_origin:
            __metadata__["_pipe_exposed_to_origin"] = exposed_to_origin

        if tools:
            if owui_tool_passthrough:
                # Full bypass: do not gate tools on model capability here; forward as OWUI provided.
                responses_body.tools = tools
            elif ModelFamily.supports("function_calling", responses_body.model):
                responses_body.tools = tools

        pdf_parser = direct_uploads.get("pdf_parser") if direct_uploads else None
        if isinstance(pdf_parser, str) and pdf_parser.strip():
            parser_key = pdf_parser.strip()
            engine_map = {
                "Native": "native",
                "PDF Text": "pdf-text",
                "Mistral OCR": "mistral-ocr",
                "native": "native",
                "pdf-text": "pdf-text",
                "mistral-ocr": "mistral-ocr",
            }
            engine = engine_map.get(parser_key)
            if engine:
                plugins = list(responses_body.plugins or [])
                has_file_parser = any(
                    isinstance(entry, dict) and entry.get("id") == "file-parser"
                    for entry in plugins
                )
                if not has_file_parser:
                    plugins.append({"id": "file-parser", "pdf": {"engine": engine}})
                    responses_body.plugins = plugins

        ors_requested = bool(features.get(_ORS_FILTER_FEATURE_FLAG, False))
        if ModelFamily.supports("web_search_tool", responses_body.model) and ors_requested:
            reasoning_cfg = responses_body.reasoning if isinstance(responses_body.reasoning, dict) else {}
            effort = (reasoning_cfg.get("effort") or "").strip().lower()
            if not effort:
                effort = valves.REASONING_EFFORT
            if effort == "minimal":
                self.logger.debug(
                    "Skipping web-search plugin because reasoning.effort is set to 'minimal' (model=%s)",
                    responses_body.model,
                )
            else:
                plugin_payload: dict[str, Any] = {"id": "web"}
                if valves.WEB_SEARCH_MAX_RESULTS is not None:
                    plugin_payload["max_results"] = valves.WEB_SEARCH_MAX_RESULTS
                plugins = list(responses_body.plugins or [])
                plugins.append(plugin_payload)
                responses_body.plugins = plugins

        # Convert the normalized model id back to the original OpenRouter id for the API request.
        setattr(responses_body, "api_model", OpenRouterModelRegistry.api_model_id(normalized_model_id) or normalized_model_id)

        reasoning_retry_attempted = False
        reasoning_effort_retry_attempted = False
        anthropic_prompt_cache_retry_attempted = False
        self.logger.debug(
            "Orchestrator: __request__ type=%s, is_none=%s",
            type(__request__).__name__,
            __request__ is None,
        )
        while True:
            try:
                if responses_body.stream:
                    # Return async generator for partial text
                    return await self._pipe._run_streaming_loop(
                        responses_body,
                        valves,
                        __event_emitter__,
                        __metadata__,
                        __tools__,
                        session=session,
                        user_id=user_id,
                        endpoint_override=endpoint_override,
                        request_context=__request__,
                        user_obj=user_model,
                        pipe_identifier=pipe_identifier,
                    )
                # Return final text (non-streaming)
                return await self._pipe._run_nonstreaming_loop(
                    responses_body,
                    valves,
                    __event_emitter__,
                    __metadata__,
                    __tools__,
                    session=session,
                    user_id=user_id,
                    endpoint_override=endpoint_override,
                    request_context=__request__,
                    user_obj=user_model,
                    pipe_identifier=pipe_identifier,
                )
            except OpenRouterAPIError as exc:
                api_model_label = getattr(responses_body, "api_model", None) or responses_body.model
                if (
                    not anthropic_prompt_cache_retry_attempted
                    and getattr(valves, "ENABLE_ANTHROPIC_PROMPT_CACHING", False)
                    and isinstance(api_model_label, str)
                    and self._pipe._is_anthropic_model_id(api_model_label)
                    and exc.status == 400
                    and Pipe._input_contains_cache_control(getattr(responses_body, "input", None))
                ):
                    self.logger.warning(
                        "Prompt caching payload rejected by provider; retrying without cache_control (model=%s).",
                        api_model_label,
                    )
                    Pipe._strip_cache_control_from_input(responses_body.input)
                    anthropic_prompt_cache_retry_attempted = True
                    continue

                if not reasoning_effort_retry_attempted:
                    error_details = {"provider_raw": exc.provider_raw}
                    if _is_reasoning_effort_error(error_details):
                        original_effort = None
                        if isinstance(responses_body.reasoning, dict):
                            original_effort = responses_body.reasoning.get("effort")

                        error_message = exc.upstream_message or exc.openrouter_message or ""
                        supported_values = _parse_supported_effort_values(error_message)

                        if supported_values:
                            fallback_effort = _select_best_effort_fallback(
                                original_effort or "",
                                supported_values,
                            )
                            if fallback_effort:
                                self.logger.info(
                                    "Reasoning effort '%s' not supported by model %s. Retrying with '%s'. Supported values: %s",
                                    original_effort,
                                    responses_body.model,
                                    fallback_effort,
                                    ", ".join(supported_values),
                                )
                                if __event_emitter__:
                                    try:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": (
                                                        f"Adjusting reasoning effort from '{original_effort}' to "
                                                        f"'{fallback_effort}' (model doesn't support '{original_effort}')"
                                                    ),
                                                    "done": False,
                                                },
                                            }
                                        )
                                    except Exception as emit_error:
                                        self.logger.debug("Failed to emit status update: %s", emit_error)

                                if not isinstance(responses_body.reasoning, dict):
                                    responses_body.reasoning = {}
                                responses_body.reasoning["effort"] = fallback_effort
                                reasoning_effort_retry_attempted = True
                                self._pipe._apply_gemini_thinking_config(responses_body, valves)
                                continue

                if (
                    not reasoning_retry_attempted
                    and self._pipe._should_retry_without_reasoning(exc, responses_body)
                ):
                    reasoning_retry_attempted = True
                    continue

                await self._pipe._report_openrouter_error(
                    exc,
                    event_emitter=__event_emitter__,
                    normalized_model_id=responses_body.model,
                    api_model_id=getattr(responses_body, "api_model", None),
                    template=valves.OPENROUTER_ERROR_TEMPLATE,
                )
                return ""
