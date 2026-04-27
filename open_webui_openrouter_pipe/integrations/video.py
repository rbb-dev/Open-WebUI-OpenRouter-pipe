from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.config import _PIPE_METADATA_KEY, _select_openrouter_http_referer
from ..core.utils import (
    _clean_str,
    _csv_set,
    _find_first_kind_marker_body,
    _iter_kind_marker_spans,
    _serialize_kind_marker,
)
from ..models.registry import OpenRouterModelRegistry
from ..storage.video_persistence import VideoPersistence
from .video_client import OpenRouterVideoClient, extension_for_video_mime
from .video_help import render_video_help
from .video_types import DownloadedVideo, VideoGenerationError, VideoLifecycleResult

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..streaming.event_emitter import EventEmitter


class VideoGenerationAdapter:

    TERMINAL_SUCCESS = {"completed", "succeeded", "success"}
    TERMINAL_FAILURE = {"failed", "cancelled", "canceled", "expired"}
    JOB_MARKER_KIND = "videojob"
    MODEL_MARKER_KIND = "videomodel"

    def __init__(self, *, pipe: "Pipe", logger: logging.Logger) -> None:
        self._pipe = pipe
        self.logger = logger
        self._persistence = VideoPersistence(logger=logger)

    async def generate(
        self,
        *,
        body: dict[str, Any],
        responses_body: Any,
        valves: Any,
        session: Any,
        event_emitter: "EventEmitter | None",
        metadata: dict[str, Any],
        user: dict[str, Any],
        request: Any,
        user_obj: Any,
        normalized_model_id: str,
        api_model_id: str,
    ) -> str:
        prompt = self._extract_prompt(body)
        video_spec = OpenRouterModelRegistry.spec(normalized_model_id)
        video_model = video_spec.get("video_model") if isinstance(video_spec, dict) else {}
        if prompt.strip().lower() == "help":
            content = render_video_help(api_model_id, video_model if isinstance(video_model, dict) else None)
            await self._emit_completion(event_emitter, content)
            return content

        chat_id = _clean_str(metadata.get("chat_id"))
        message_id = _clean_str(metadata.get("message_id"))
        if not chat_id or not message_id:
            content = self._build_failure_content(
                job_id="",
                model_id=api_model_id,
                reason="Video generation requires stable chat_id and message_id metadata.",
            )
            await self._emit_status(event_emitter, "Video generation could not start.", done=True)
            await self._emit_completion(event_emitter, content)
            return content

        key = (chat_id, message_id)
        user_id = _clean_str(user.get("id")) or _clean_str(metadata.get("user_id")) or "anonymous"
        existing = await self._get_active_task(key)
        if existing is not None:
            return await self._await_existing_task(existing, event_emitter)

        message_lock = await self._acquire_message_lock(key)
        global_semaphore: asyncio.Semaphore | None = None
        user_slot_acquired = False
        lifecycle_transferred = False
        job_id = ""

        try:
            existing = await self._get_active_task(key)
            if existing is not None:
                await self._release_message_lock(key, message_lock)
                message_lock = None  # type: ignore[assignment]
                return await self._await_existing_task(existing, event_emitter)

            persisted = await self._persistence.load_message_content(chat_id=chat_id, message_id=message_id)
            if self._looks_like_final_video_content(persisted):
                await self._emit_completion(event_emitter, persisted)
                return persisted
            resume_job_id = self._extract_video_job_marker(persisted)
            if resume_job_id:
                user_slot_acquired = await self._try_acquire_user_slot(user_id, valves)
                if not user_slot_acquired:
                    content = self._build_failure_content(
                        job_id=resume_job_id,
                        model_id=api_model_id,
                        reason=(
                            "Video generation limit reached for this user "
                            f"({valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER} active job(s))."
                        ),
                    )
                    await self._emit_status(event_emitter, "Video generation limit reached.", done=True)
                    await self._emit_completion(event_emitter, content)
                    return content
                global_semaphore = self._ensure_global_semaphore(valves)
                await global_semaphore.acquire()
                job_id = resume_job_id
                await self._add_user_active_job(user_id, job_id)
                await self._emit_status(event_emitter, "Resuming video generation job...", done=False, progress=5)
                bg_task = self._create_lifecycle_task(
                    key=key,
                    job_id=job_id,
                    api_model_id=api_model_id,
                    normalized_model_id=normalized_model_id,
                    valves=valves,
                    event_emitter=event_emitter,
                    user=user,
                    user_obj=user_obj,
                    chat_id=chat_id,
                    message_id=message_id,
                    request=request,
                    user_id=user_id,
                    global_semaphore=global_semaphore,
                    message_lock=message_lock,
                    started_at=time.monotonic(),
                )
                async with self._pipe._video_active_tasks_dict_lock:
                    self._pipe._video_active_tasks[key] = bg_task
                lifecycle_transferred = True
                result = await asyncio.shield(bg_task)
                await self._emit_completion(event_emitter, result.content)
                return result.content

            if not prompt.strip():
                content = self._build_failure_content(
                    job_id="",
                    model_id=api_model_id,
                    reason="Video generation requires a prompt.",
                )
                await self._emit_status(event_emitter, "Video generation could not start.", done=True)
                await self._emit_completion(event_emitter, content)
                return content

            user_slot_acquired = await self._try_acquire_user_slot(user_id, valves)
            if not user_slot_acquired:
                content = self._build_failure_content(
                    job_id="",
                    model_id=api_model_id,
                    reason=(
                        "Video generation limit reached for this user "
                        f"({valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER} active job(s))."
                    ),
                )
                await self._emit_status(event_emitter, "Video generation limit reached.", done=True)
                await self._emit_completion(event_emitter, content)
                return content

            global_semaphore = self._ensure_global_semaphore(valves)
            await global_semaphore.acquire()

            video_meta = self._extract_video_metadata(metadata)
            frame_images = await self._encode_frame_images(video_meta, video_model, valves)
            video_attachment_urls = await self._encode_video_attachments(video_meta, valves)
            audio_attachment_url = await self._encode_audio_attachment(video_meta, valves)
            provider_options = self._extract_provider_options(getattr(responses_body, "provider", None), metadata)
            payload = self._build_payload(
                api_model_id=api_model_id,
                prompt=prompt,
                video_meta=video_meta,
                video_model=video_model,
                frame_images=frame_images,
                provider_options=provider_options,
                video_attachment_urls=video_attachment_urls,
                audio_attachment_url=audio_attachment_url,
            )
            await self._emit_status(event_emitter, "Submitting video generation job...", done=False)

            client = OpenRouterVideoClient(
                session,
                base_url=valves.BASE_URL,
                api_key=self._resolve_api_key(valves),
                logger=self.logger,
                http_referer=_select_openrouter_http_referer(valves),
            )
            accepted = await client.submit(payload)
            job_id = self._extract_job_id(accepted)
            if not job_id:
                raise VideoGenerationError("OpenRouter accepted the request without returning a video job id.")
            await self._add_user_active_job(user_id, job_id)
            if (
                event_emitter is not None
                and isinstance(chat_id, str)
                and not chat_id.startswith("local:")
            ):
                pending_content = self._build_pending_content(
                    job_id=job_id,
                    model_id=api_model_id,
                )
                with contextlib.suppress(Exception):
                    await event_emitter({
                        "type": "message",
                        "data": {"content": pending_content},
                    })
            bg_task = self._create_lifecycle_task(
                key=key,
                job_id=job_id,
                api_model_id=api_model_id,
                normalized_model_id=normalized_model_id,
                valves=valves,
                event_emitter=event_emitter,
                user=user,
                user_obj=user_obj,
                chat_id=chat_id,
                message_id=message_id,
                request=request,
                user_id=user_id,
                global_semaphore=global_semaphore,
                message_lock=message_lock,
                started_at=time.monotonic(),
            )
            async with self._pipe._video_active_tasks_dict_lock:
                self._pipe._video_active_tasks[key] = bg_task
            lifecycle_transferred = True

            try:
                result = await asyncio.shield(bg_task)
            except asyncio.CancelledError:
                raise
            await self._emit_completion(event_emitter, result.content)
            return result.content
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            content = self._build_failure_content(job_id=job_id, model_id=api_model_id, reason=reason)
            await self._emit_status(event_emitter, "Video generation failed.", done=True)
            await self._emit_completion(event_emitter, content)
            return content
        finally:
            if not lifecycle_transferred:
                if global_semaphore is not None:
                    with contextlib.suppress(ValueError):
                        global_semaphore.release()
                if user_slot_acquired:
                    await self._release_user_slot(user_id, job_id)
                if message_lock is not None:
                    await self._release_message_lock(key, message_lock)

    def _create_lifecycle_task(
        self,
        *,
        key: tuple[str, str],
        job_id: str,
        api_model_id: str,
        normalized_model_id: str,
        valves: Any,
        event_emitter: "EventEmitter | None",
        user: dict[str, Any],
        user_obj: Any,
        chat_id: str,
        message_id: str,
        request: Any,
        user_id: str,
        global_semaphore: asyncio.Semaphore,
        message_lock: asyncio.Lock,
        started_at: float,
    ) -> "asyncio.Task[VideoLifecycleResult]":
        task: asyncio.Task[VideoLifecycleResult] = asyncio.create_task(
            self._run_lifecycle_after_submit(
                key=key,
                job_id=job_id,
                api_model_id=api_model_id,
                normalized_model_id=normalized_model_id,
                valves=valves,
                event_emitter=event_emitter,
                user=user,
                user_obj=user_obj,
                chat_id=chat_id,
                message_id=message_id,
                request=request,
                user_id=user_id,
                global_semaphore=global_semaphore,
                message_lock=message_lock,
                started_at=started_at,
            ),
            name=f"openrouter-video-{job_id}",
        )
        task.add_done_callback(self._consume_background_exception)
        return task

    async def _run_lifecycle_after_submit(
        self,
        *,
        key: tuple[str, str],
        job_id: str,
        api_model_id: str,
        normalized_model_id: str,
        valves: Any,
        event_emitter: "EventEmitter | None",
        user: dict[str, Any],
        user_obj: Any,
        chat_id: str,
        message_id: str,
        request: Any,
        user_id: str,
        global_semaphore: asyncio.Semaphore,
        message_lock: asyncio.Lock,
        started_at: float,
    ) -> VideoLifecycleResult:
        content = ""
        failed = False
        usage: dict[str, Any] = {}
        file_id: str | None = None
        output_mime = ""
        description = ""
        downloaded = None
        try:
            await self._emit_status(event_emitter, "Video generation job accepted.", done=False, progress=5)
            session = self._pipe._create_http_session(valves)
            try:
                client = OpenRouterVideoClient(
                    session,
                    base_url=valves.BASE_URL,
                    api_key=self._resolve_api_key(valves),
                    logger=self.logger,
                    http_referer=_select_openrouter_http_referer(valves),
                )
                status_payload = await self._poll_until_terminal(client, job_id, valves, event_emitter)
                usage = self._coerce_video_usage(status_payload.get("usage"))
                status = _clean_str(status_payload.get("status")).lower()
                if status not in self.TERMINAL_SUCCESS:
                    raise VideoGenerationError(self._status_failure_reason(status_payload, status))

                max_bytes = int(valves.REMOTE_VIDEO_MAX_SIZE_MB) * 1024 * 1024
                allowed_mimes = _csv_set(valves.VIDEO_OUTPUT_MIME_ALLOWLIST)
                await self._emit_status(event_emitter, "Downloading generated video...", done=False, progress=80)
                content_url = client.content_url(job_id)
                bearer = client.bearer_header()
                tmp_dir = Path(tempfile.mkdtemp(prefix="openrouter-video-"))
                tmp_path = tmp_dir / f"job-{job_id}.bin"
                download_result = await self._pipe._multimodal_handler._download_remote_url_streaming(
                    content_url,
                    tmp_path,
                    chunk_size=int(valves.VIDEO_DOWNLOAD_CHUNK_SIZE),
                    max_size_bytes=max_bytes,
                    mime_allowlist=allowed_mimes,
                    extra_headers=bearer,
                )
                if not download_result:
                    with contextlib.suppress(Exception):
                        if tmp_path.exists():
                            tmp_path.unlink()
                    raise VideoGenerationError(
                        "Generated video could not be downloaded from OpenRouter."
                    )
                downloaded = DownloadedVideo(
                    path=download_result["path"],
                    mime_type=download_result["mime_type"] or "",
                    size_bytes=int(download_result["size_bytes"] or 0),
                )
                output_mime = downloaded.mime_type
            finally:
                with contextlib.suppress(Exception):
                    await session.close()

            elapsed = max(0.0, time.monotonic() - started_at)
            extension = extension_for_video_mime(output_mime)
            if downloaded is None:
                raise VideoGenerationError("Generated video download did not complete.")
            file_id = await self._pipe._multimodal_handler._upload_to_owui_storage_from_path(
                request=request,
                user=user_obj or user,
                source_path=downloaded.path,
                filename=f"openrouter-video-{job_id}{extension}",
                mime_type=output_mime,
                chat_id=chat_id,
                message_id=message_id,
                owui_user_id=user_id,
            )
            if not file_id:
                raise VideoGenerationError(
                    "Generated video could not be stored in Open WebUI; the upload failed."
                )
            with contextlib.suppress(Exception):
                downloaded.path.unlink(missing_ok=True)
            content = self._build_success_content(
                job_id=job_id,
                model_id=api_model_id,
                file_id=file_id,
                elapsed=elapsed,
                usage=usage,
            )
            description = self._format_final_status(elapsed=elapsed, usage=usage, valves=valves)
            await self._emit_status(event_emitter, description, done=True, progress=100)
            return VideoLifecycleResult(
                content=content,
                status_description=description,
                usage=usage,
                job_id=job_id,
                file_id=file_id,
                failed=False,
                elapsed=elapsed,
                model_id=api_model_id,
                output_mime=output_mime,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed = True
            elapsed = max(0.0, time.monotonic() - started_at)
            reason = str(exc) or exc.__class__.__name__
            content = self._build_failure_content(job_id=job_id, model_id=api_model_id, reason=reason)
            description = f"Video generation failed: {reason}"
            await self._emit_status(event_emitter, description, done=True)
            return VideoLifecycleResult(
                content=content,
                status_description=description,
                usage=usage,
                job_id=job_id,
                file_id=file_id,
                failed=failed,
                elapsed=elapsed,
                model_id=api_model_id,
                output_mime=output_mime,
            )
        finally:
            if downloaded is not None:
                with contextlib.suppress(Exception):
                    downloaded.path.unlink(missing_ok=True)
            async with self._pipe._video_active_tasks_dict_lock:
                current = self._pipe._video_active_tasks.get(key)
                if current is asyncio.current_task():
                    self._pipe._video_active_tasks.pop(key, None)
            await self._release_user_slot(user_id, job_id)
            with contextlib.suppress(ValueError):
                global_semaphore.release()
            await self._release_message_lock(key, message_lock)

    async def _poll_until_terminal(
        self,
        client: OpenRouterVideoClient,
        job_id: str,
        valves: Any,
        event_emitter: "EventEmitter | None",
    ) -> dict[str, Any]:
        initial_delay = float(valves.VIDEO_INITIAL_POLL_DELAY_SECONDS)
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        interval = float(valves.VIDEO_POLL_INTERVAL_SECONDS)
        max_interval = float(valves.VIDEO_POLL_INTERVAL_MAX_SECONDS)
        backoff = float(valves.VIDEO_POLL_BACKOFF_FACTOR)
        deadline = time.monotonic() + int(valves.VIDEO_MAX_POLL_TIME_SECONDS)
        consecutive_errors = 0
        last_emit_status = ""
        last_emit_progress = -1
        last_emit_at = 0.0
        min_emit_interval = max(10.0, min(30.0, max_interval))

        async def _maybe_emit(status_value: str, progress: int) -> None:
            nonlocal last_emit_status, last_emit_progress, last_emit_at
            now = time.monotonic()
            changed = status_value != last_emit_status or progress != last_emit_progress
            if not changed and now - last_emit_at < min_emit_interval:
                return
            last_emit_status = status_value
            last_emit_progress = progress
            last_emit_at = now
            label = status_value.replace("_", " ") if status_value else "in progress"
            await self._emit_status(event_emitter, f"Video generation {label}...", done=False, progress=progress)

        while True:
            if time.monotonic() > deadline:
                raise VideoGenerationError("Video generation timed out before OpenRouter reported completion.")
            try:
                payload = await client.status(job_id)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= int(valves.VIDEO_STATUS_POLL_MAX_ERRORS):
                    raise
                await asyncio.sleep(min(interval, max_interval))
                interval = min(max_interval, interval * backoff)
                continue

            status = _clean_str(payload.get("status")).lower()
            if status in self.TERMINAL_SUCCESS | self.TERMINAL_FAILURE:
                if status in self.TERMINAL_SUCCESS:
                    await self._emit_status(event_emitter, "Video generation completed.", done=False, progress=100)
                return payload
            progress = 5 if status == "pending" else 50
            await _maybe_emit(status, progress)
            await asyncio.sleep(min(interval, max_interval))
            interval = min(max_interval, interval * backoff)

    async def _await_existing_task(
        self,
        task: "asyncio.Task[VideoLifecycleResult]",
        event_emitter: "EventEmitter | None",
    ) -> str:
        await self._emit_status(event_emitter, "Waiting for active video generation job...", done=False)
        result = await asyncio.shield(task)
        await self._emit_status(event_emitter, result.status_description, done=True)
        await self._emit_completion(event_emitter, result.content)
        return result.content

    async def _get_active_task(self, key: tuple[str, str]) -> "asyncio.Task[VideoLifecycleResult] | None":
        async with self._pipe._video_active_tasks_dict_lock:
            task = self._pipe._video_active_tasks.get(key)
            if task is not None and task.done():
                self._pipe._video_active_tasks.pop(key, None)
                return None
            return task

    async def _acquire_message_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        async with self._pipe._video_message_locks_dict_lock:
            lock = self._pipe._video_message_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._pipe._video_message_locks[key] = lock
            self._pipe._video_message_lock_refs[key] = self._pipe._video_message_lock_refs.get(key, 0) + 1
        try:
            await lock.acquire()
        except BaseException:
            async with self._pipe._video_message_locks_dict_lock:
                new_refs = self._pipe._video_message_lock_refs.get(key, 0) - 1
                if new_refs <= 0:
                    self._pipe._video_message_lock_refs.pop(key, None)
                    if (
                        self._pipe._video_message_locks.get(key) is lock
                        and not lock.locked()
                    ):
                        self._pipe._video_message_locks.pop(key, None)
                else:
                    self._pipe._video_message_lock_refs[key] = new_refs
            raise
        return lock

    async def _release_message_lock(self, key: tuple[str, str], lock: asyncio.Lock) -> None:
        if lock.locked():
            lock.release()
        async with self._pipe._video_message_locks_dict_lock:
            refs = self._pipe._video_message_lock_refs.get(key, 0) - 1
            if refs <= 0:
                self._pipe._video_message_lock_refs.pop(key, None)
                if self._pipe._video_message_locks.get(key) is lock:
                    self._pipe._video_message_locks.pop(key, None)
            else:
                self._pipe._video_message_lock_refs[key] = refs

    def _ensure_global_semaphore(self, valves: Any) -> asyncio.Semaphore:
        limit = int(valves.MAX_CONCURRENT_VIDEO_GENS)
        cls = type(self._pipe)
        if cls._video_global_semaphore is None or cls._video_global_limit != limit:
            cls._video_global_semaphore = asyncio.Semaphore(limit)
            cls._video_global_limit = limit
        return cls._video_global_semaphore

    async def _try_acquire_user_slot(self, user_id: str, valves: Any) -> bool:
        limit = int(valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER)
        async with self._pipe._video_user_locks_dict_lock:
            lock = self._pipe._video_user_locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._pipe._video_user_locks[user_id] = lock
        async with lock:
            current = int(self._pipe._video_user_active_counts.get(user_id, 0))
            if current >= limit:
                return False
            self._pipe._video_user_active_counts[user_id] = current + 1
            return True

    async def _add_user_active_job(self, user_id: str, job_id: str) -> None:
        async with self._pipe._video_user_locks_dict_lock:
            lock = self._pipe._video_user_locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._pipe._video_user_locks[user_id] = lock
        async with lock:
            if job_id:
                self._pipe._video_user_active_jobs.setdefault(user_id, set()).add(job_id)

    async def _release_user_slot(self, user_id: str, job_id: str = "") -> None:
        async with self._pipe._video_user_locks_dict_lock:
            lock = self._pipe._video_user_locks.get(user_id)
        if lock is None:
            return
        async with lock:
            if job_id:
                jobs = self._pipe._video_user_active_jobs.get(user_id)
                if jobs is not None:
                    jobs.discard(job_id)
                    if not jobs:
                        self._pipe._video_user_active_jobs.pop(user_id, None)
            current = int(self._pipe._video_user_active_counts.get(user_id, 0))
            if current <= 1:
                self._pipe._video_user_active_counts.pop(user_id, None)
            else:
                self._pipe._video_user_active_counts[user_id] = current - 1

    def _build_payload(
        self,
        *,
        api_model_id: str,
        prompt: str,
        video_meta: dict[str, Any],
        video_model: Any,
        frame_images: list[dict[str, Any]],
        provider_options: dict[str, Any],
        video_attachment_urls: list[str] | None = None,
        audio_attachment_url: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": api_model_id,
            "prompt": prompt,
        }
        params = video_meta.get("params")
        allowed = self._allowed_passthrough(video_model)
        if isinstance(params, dict):
            for key, value in params.items():
                if value is None or value == "":
                    continue
                target = self._select_passthrough_key(key, allowed)
                if target:
                    payload[target] = value
        if video_attachment_urls:
            if "videos" in allowed and len(video_attachment_urls) > 1:
                payload["videos"] = [{"url": u} for u in video_attachment_urls]
            elif "video" in allowed:
                payload["video"] = video_attachment_urls[0]
            elif "videos" in allowed:
                payload["videos"] = [{"url": video_attachment_urls[0]}]
        if audio_attachment_url and "audio" in allowed:
            payload["audio"] = audio_attachment_url
        self._validate_passthrough_urls(payload)
        if frame_images:
            payload["frame_images"] = frame_images
        if provider_options:
            payload["provider"] = {"options": self._normalise_provider_options(provider_options)}
        return payload

    def _validate_passthrough_urls(self, payload: dict[str, Any]) -> None:
        url_fields = ("audio", "last_image", "video")
        array_fields = ("videos", "images")
        handler = self._pipe._multimodal_handler

        def _check(url: str, field_name: str) -> None:
            if url.startswith("data:"):
                return
            if not handler._is_safe_url_blocking(url):
                raise VideoGenerationError(
                    f"Refusing to forward unsafe URL in '{field_name}'. Use https:// or "
                    f"an allowlisted http:// destination."
                )

        for field_name in url_fields:
            raw = payload.get(field_name)
            if not isinstance(raw, str):
                continue
            cleaned = raw.strip()
            if not cleaned:
                payload.pop(field_name, None)
                continue
            _check(cleaned, field_name)
            payload[field_name] = cleaned

        for field_name in array_fields:
            items = payload.get(field_name)
            if not isinstance(items, list):
                continue
            validated: list[Any] = []
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    url = item.get("url")
                elif isinstance(item, str):
                    url = item
                else:
                    continue
                if not isinstance(url, str) or not url.strip():
                    continue
                cleaned = url.strip()
                _check(cleaned, f"{field_name}[{idx}]")
                if isinstance(item, dict):
                    new_item = dict(item)
                    new_item["url"] = cleaned
                    validated.append(new_item)
                else:
                    validated.append(cleaned)
            payload[field_name] = validated

    @staticmethod
    def _normalise_provider_options(provider_options: dict[str, Any]) -> dict[str, Any]:
        normalised: dict[str, Any] = {}
        for slug, payload in provider_options.items():
            if not isinstance(slug, str) or not slug.strip() or not isinstance(payload, dict):
                continue
            if isinstance(payload.get("parameters"), dict):
                normalised[slug.strip()] = payload
            else:
                normalised[slug.strip()] = {"parameters": dict(payload)}
        return normalised

    async def _encode_frame_images(
        self,
        video_meta: dict[str, Any],
        video_model: Any,
        valves: Any,
    ) -> list[dict[str, Any]]:
        raw_frames = video_meta.get("frame_images")
        if not isinstance(raw_frames, list) or not raw_frames:
            return []
        supported = self._supported_frame_types(video_model)
        if not supported:
            raise VideoGenerationError("The selected video model does not accept frame images.")
        max_bytes = int(valves.VIDEO_FRAME_IMAGE_MAX_BYTES)
        total_max = int(valves.VIDEO_FRAME_TOTAL_MAX_BYTES)
        chunk_size = int(getattr(valves, "IMAGE_UPLOAD_CHUNK_BYTES", 1024 * 1024))
        allowed_mimes = _csv_set(valves.VIDEO_FRAME_IMAGE_MIME_ALLOWLIST)
        encoded: list[dict[str, Any]] = []
        total_bytes = 0
        seen_frame_types: set[str] = set()

        for item in raw_frames:
            if not isinstance(item, dict):
                continue
            file_id = _clean_str(item.get("id"))
            frame_type = _clean_str(item.get("frame_type")) or "first_frame"
            if not file_id:
                continue
            if frame_type not in supported:
                raise VideoGenerationError(
                    f"Frame type '{frame_type}' is not supported by this model. Supported: {', '.join(sorted(supported))}."
                )
            if frame_type in seen_frame_types:
                continue
            file_obj = await self._pipe._multimodal_handler._get_file_by_id(file_id)
            if not file_obj:
                raise VideoGenerationError(f"Frame image '{file_id}' could not be loaded from Open WebUI storage.")
            mime = self._pipe._multimodal_handler._infer_file_mime_type(file_obj)
            mime = _clean_str(mime).split(";", 1)[0].lower()
            if mime not in allowed_mimes:
                raise VideoGenerationError(f"Frame image MIME '{mime or 'unknown'}' is not allowed.")
            b64 = await self._pipe._multimodal_handler._read_file_record_base64(file_obj, chunk_size, max_bytes)
            if not b64:
                raise VideoGenerationError(f"Frame image '{file_id}' could not be encoded.")
            try:
                decoded_len = len(base64.b64decode(b64, validate=False))
            except Exception as exc:
                raise VideoGenerationError(f"Frame image '{file_id}' contains invalid base64 data.") from exc
            total_bytes += decoded_len
            if total_bytes > total_max:
                raise VideoGenerationError(
                    f"Frame images exceed the total request limit ({total_bytes} bytes; max {total_max} bytes)."
                )
            encoded.append(
                {
                    "type": "image_url",
                    "frame_type": frame_type,
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            seen_frame_types.add(frame_type)
        return encoded

    async def _encode_attachment_data_url(
        self,
        item: dict[str, Any],
        valves: Any,
        kind: str,
    ) -> str:
        file_id = _clean_str(item.get("id"))
        if not file_id:
            raise VideoGenerationError(f"{kind} attachment is missing a file id.")
        chunk_size = int(getattr(valves, "IMAGE_UPLOAD_CHUNK_BYTES", 1024 * 1024))
        max_bytes = int(getattr(valves, "REMOTE_VIDEO_MAX_SIZE_MB", 500)) * 1024 * 1024
        file_obj = await self._pipe._multimodal_handler._get_file_by_id(file_id)
        if not file_obj:
            raise VideoGenerationError(f"{kind} attachment '{file_id}' could not be loaded from Open WebUI storage.")
        mime = self._pipe._multimodal_handler._infer_file_mime_type(file_obj)
        mime = _clean_str(mime).split(";", 1)[0].lower() or _clean_str(item.get("content_type")).split(";", 1)[0].lower()
        if not mime:
            raise VideoGenerationError(f"{kind} attachment '{file_id}' has no detectable MIME type.")
        b64 = await self._pipe._multimodal_handler._read_file_record_base64(file_obj, chunk_size, max_bytes)
        if not b64:
            raise VideoGenerationError(f"{kind} attachment '{file_id}' could not be encoded.")
        return f"data:{mime};base64,{b64}"

    async def _encode_video_attachments(
        self,
        video_meta: dict[str, Any],
        valves: Any,
    ) -> list[str]:
        raw = video_meta.get("video_attachments")
        if not isinstance(raw, list) or not raw:
            return []
        urls: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                urls.append(await self._encode_attachment_data_url(item, valves, "Video"))
        return urls

    async def _encode_audio_attachment(
        self,
        video_meta: dict[str, Any],
        valves: Any,
    ) -> str:
        raw = video_meta.get("audio_attachments")
        if not isinstance(raw, list) or not raw:
            return ""
        for item in raw:
            if isinstance(item, dict):
                return await self._encode_attachment_data_url(item, valves, "Audio")
        return ""

    def _extract_video_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        pipe_meta = metadata.get(_PIPE_METADATA_KEY) if isinstance(metadata, dict) else None
        if not isinstance(pipe_meta, dict):
            return {}
        video_meta = pipe_meta.get("video_generation")
        return dict(video_meta) if isinstance(video_meta, dict) else {}

    def _extract_provider_options(self, response_provider: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        provider = response_provider if isinstance(response_provider, dict) else {}
        options = provider.get("options") if isinstance(provider, dict) else None
        if not isinstance(options, dict):
            pipe_meta = metadata.get(_PIPE_METADATA_KEY) if isinstance(metadata, dict) else None
            provider_meta = pipe_meta.get("provider") if isinstance(pipe_meta, dict) else None
            options = provider_meta.get("options") if isinstance(provider_meta, dict) else None
        return dict(options) if isinstance(options, dict) else {}

    def _allowed_passthrough(self, video_model: Any) -> set[str]:
        if not isinstance(video_model, dict):
            return set()
        raw = video_model.get("allowed_passthrough_parameters")
        if not isinstance(raw, list):
            allowed: set[str] = set()
        else:
            allowed = {item for item in raw if isinstance(item, str) and item}
        if (
            isinstance(video_model.get("supported_aspect_ratios"), list)
            and not allowed.intersection({"aspect_ratio", "aspectRatio", "ratio"})
        ):
            allowed.add("aspect_ratio")
        if (
            isinstance(video_model.get("supported_durations"), list)
            and not allowed.intersection({"duration", "duration_seconds"})
        ):
            allowed.add("duration")
        if isinstance(video_model.get("supported_resolutions"), list):
            allowed.add("resolution")
        if isinstance(video_model.get("supported_sizes"), list) or isinstance(video_model.get("supported_size_options"), list):
            allowed.add("size")
        if video_model.get("seed") is True:
            allowed.add("seed")
        if video_model.get("generate_audio") is True:
            allowed.add("generate_audio")
        return allowed

    def _supported_frame_types(self, video_model: Any) -> set[str]:
        if not isinstance(video_model, dict):
            return set()
        raw = video_model.get("supported_frame_images")
        if not isinstance(raw, list):
            return set()
        return {item for item in raw if isinstance(item, str) and item}

    def _select_passthrough_key(self, key: str, allowed: set[str]) -> str:
        raw = _clean_str(key)
        if not raw:
            return ""
        aliases = {
            "aspect_ratio": ["aspect_ratio", "aspectRatio", "ratio"],
            "duration": ["duration", "duration_seconds"],
            "generate_audio": ["generate_audio"],
            "negative_prompt": ["negative_prompt", "negativePrompt"],
            "resolution": ["resolution"],
            "seed": ["seed"],
            "size": ["size"],
        }
        candidates = aliases.get(raw, [raw])
        if not allowed:
            return ""
        for candidate in candidates:
            if candidate in allowed:
                return candidate
        return ""

    def _extract_prompt(self, body: dict[str, Any]) -> str:
        messages = body.get("messages") if isinstance(body, dict) else None
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text:
                                parts.append(text)
                    return "\n".join(parts)
        prompt = body.get("prompt") if isinstance(body, dict) else ""
        return prompt if isinstance(prompt, str) else ""

    def _extract_job_id(self, payload: dict[str, Any]) -> str:
        for key in ("id", "job_id", "jobId"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = payload.get("data")
        if isinstance(data, dict):
            return self._extract_job_id(data)
        return ""

    @staticmethod
    def _status_failure_reason(payload: dict[str, Any], status: str) -> str:
        for key in ("error", "message", "detail", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                message = value.get("message") or value.get("detail")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        return f"OpenRouter reported video job status '{status or 'failed'}'."

    def _build_success_content(
        self,
        *,
        job_id: str,
        model_id: str,
        file_id: str,
        elapsed: float,
        usage: dict[str, Any],
    ) -> str:
        footer = f"*Generated in {elapsed:.1f}s*"
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            footer = f"*Generated in {elapsed:.1f}s · ${cost:.4f}*"
        return (
            f"{_serialize_kind_marker(self.JOB_MARKER_KIND, job_id)}\n"
            f"{_serialize_kind_marker(self.MODEL_MARKER_KIND, model_id)}\n\n"
            f"<video>\n"
            f"/api/v1/files/{file_id}/content\n"
            f"</video>\n\n"
            f"{footer}\n"
        )

    def _build_failure_content(self, *, job_id: str, model_id: str, reason: str) -> str:
        markers = ""
        if job_id:
            markers += f"{_serialize_kind_marker(self.JOB_MARKER_KIND, job_id)}\n"
        if model_id:
            markers += f"{_serialize_kind_marker(self.MODEL_MARKER_KIND, model_id)}\n"
        if markers:
            markers += "\n"
        return f"{markers}### Video generation failed\n\n{reason}"

    def _build_pending_content(self, *, job_id: str, model_id: str) -> str:
        return (
            f"{_serialize_kind_marker(self.JOB_MARKER_KIND, job_id)}\n"
            f"{_serialize_kind_marker(self.MODEL_MARKER_KIND, model_id)}\n\n"
            "Video generation is running..."
        )

    def _extract_video_job_marker(self, content: str) -> str:
        if not isinstance(content, str) or not content:
            return ""
        return _find_first_kind_marker_body(content, kind=self.JOB_MARKER_KIND).strip()

    def _looks_like_final_video_content(self, content: str) -> bool:
        if not isinstance(content, str):
            return False
        if not _iter_kind_marker_spans(content, kind=self.JOB_MARKER_KIND):
            return False
        return "<video>" in content or "### Video generation failed" in content

    def _format_final_status(self, *, elapsed: float, usage: dict[str, Any], valves: Any) -> str:
        try:
            return self._pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=elapsed,
                total_usage=usage,
                valves=valves,
                stream_duration=None,
            )
        except Exception:
            return f"Video generated in {elapsed:.1f}s"

    @staticmethod
    def _coerce_video_usage(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out = dict(raw)
        cost = out.get("cost")
        if isinstance(cost, str):
            try:
                out["cost"] = float(cost)
            except (TypeError, ValueError):
                out.pop("cost", None)
        elif not isinstance(cost, (int, float)):
            out.pop("cost", None)
        return out

    async def _emit_status(
        self,
        emitter: "EventEmitter | None",
        description: str,
        *,
        done: bool,
        progress: int | None = None,
    ) -> None:
        data: dict[str, Any] = {"description": description, "done": done}
        if progress is not None:
            data["progress"] = progress
        await self._safe_emit(
            emitter,
            {
                "type": "status",
                "data": data,
            },
        )

    async def _emit_completion(self, emitter: "EventEmitter | None", content: str) -> None:
        await self._safe_emit(
            emitter,
            {
                "type": "chat:message:delta",
                "data": {"content": content},
            },
        )
        await self._safe_emit(
            emitter,
            {
                "type": "chat:completion",
                "data": {"content": content, "done": True},
            },
        )

    async def _safe_emit(self, emitter: "EventEmitter | None", event: dict[str, Any]) -> None:
        if emitter is None:
            return
        try:
            await emitter(event)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception as exc:
            self.logger.debug("Video generation event emit failed: %s", exc)

    def _resolve_api_key(self, valves: Any) -> str:
        api_key, api_key_error = self._pipe._resolve_openrouter_api_key(valves)
        if api_key_error or not api_key:
            raise VideoGenerationError(api_key_error or "OpenRouter API key is not configured.")
        return api_key

    @staticmethod
    def _consume_background_exception(task: asyncio.Future[Any]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()
