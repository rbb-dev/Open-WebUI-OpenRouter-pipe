from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import functools
import logging
import re
import secrets
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlsplit

from ..api.gateway.responses_adapter import _record_failed_call
from ..core.config import _PIPE_METADATA_KEY, _select_openrouter_http_referer
from ..core.costs import maybe_dump_costs_snapshot
from ..core.errors import OpenRouterAPIError, RequiredInternalFileError
from ..core.utils import (
    _clean_str,
    _csv_set,
    _find_first_kind_marker_body,
    _iter_kind_marker_spans,
    _serialize_kind_marker,
    summarise_names,
)
from ..core.warn_latch import warn_level
from ..media import (
    FrameExtractionError,
    extract_frame,
    make_thumbnail,
    probe_video,
)
from ..models.registry import OpenRouterModelRegistry
from ..requests.fusion_engine import asks_for_help, latest_user_text
from ..storage.multimodal import (
    ADDRESS_CHECK_BUDGET_SECONDS,
    ADDRESS_CHECK_SECONDS,
)
from ..storage.owui_files import (
    PUBLISHING_NEEDS_OWNERSHIP,
    InlineFileTooLargeError,
    authorize_file_publication,
    declared_file_size,
    get_file_by_id,
    infer_file_mime_type,
    is_linkable_chat,
    is_temporary_chat,
    materialize_owui_file_to_temp,
)
from ..storage.video_persistence import VideoPersistence
from .image_types import capability_declared_off, prompt_with_system
from .media_relay import (
    MAX_RELAY_SECONDS_PER_REQUEST,
    RELAY_HOSTS,
    MediaRelayError,
    host_keeps_forever,
    megabytes,
    relay_to_public_url,
    usable_media_type,
)
from .provider_options import (
    VIDEO_PROVIDER_KEYS,
    UnvettableRequest,
    carrier_slug,
    merge_provider_options,
    options_key,
    payload_addresses,
    refuse_past_the_scan_depth,
    requested_provider_block,
    requested_provider_options,
    restrict_provider_block,
)
from .request_fields import VIDEO_REQUEST_FIELDS
from .video_client import OpenRouterVideoClient, extension_for_video_mime
from .video_help import render_video_help
from .video_intent import (
    FramePlanEntry,
    VideoIntentResult,
    _admin_intent_floor,
    _hash_chat_id,
    collect_attachments_from_video_meta,
    emit_telemetry_log,
    render_clarification_message,
    render_intent_disclosure_block,
    resolve_intent,
    resolve_intent_user_setting,
    resolve_reference_items,
    should_emit_confirmation_footer,
)
from .video_types import (
    DownloadedVideo,
    VideoGenerationError,
    VideoGenerationStalled,
    VideoLifecycleResult,
)

_warned_provider_slug_guess: set[str] = set()

_warned_video_provider_keys: set[str] = set()

_warned_reference_scope: set[str] = set()

_warned_reference_scope_entry: set[str] = set()

_MAX_PASSTHROUGH_URLS = 16

_OPTIONS_HOP_DEPTH = 3

_MAX_VIDEO_OUTPUTS = 16

_REFERENCE_KINDS_NEEDING_A_LINK = frozenset({"audio_url", "video_url"})

_INTENT_NOTIFIED_WINDOW = 300

_INTENT_CHAT_COUNT_WINDOW = 300

_INTENT_BREAKER_SWEEP_INTERVAL_SECONDS = 60.0

_INTENT_NO_CHAT_ID_KEY = "__no_chat_id__"


def _reference_write_key(video_meta: dict[str, Any]) -> str:
    for key in ("input_references", "video_attachments"):
        items = video_meta.get(key)
        if isinstance(items, list) and items:
            return key
    return "input_references"


def _declared_input_kinds(video_model: Any) -> Callable[[str], bool]:
    source = video_model if isinstance(video_model, dict) else {}
    declared = source.get("input_modalities")
    if not isinstance(declared, list):
        declared = ((source.get("architecture") or {}) if isinstance(source.get("architecture"), dict) else {}).get(
            "input_modalities"
        )
    if not isinstance(declared, list) or not declared:
        return lambda _family: True
    kinds = {item for item in declared if isinstance(item, str)}
    return lambda family: family in kinds


def _reference_kind_refused(family: str) -> str:
    spoken = {"video": "a clip", "audio": "a sound file", "image": "a picture"}.get(family, family)
    return f"this model does not take {spoken} as a reference"


def _attachment_position(entry: dict[str, Any]) -> int:
    declared = entry.get("attachment_index")
    if isinstance(declared, int) and not isinstance(declared, bool):
        return declared
    return 0


def _shortfall_note(missing: int, attempted: int, fetched: int) -> str:
    if attempted - fetched > 0:
        cause = "could not be fetched"
    elif fetched - attempted + missing > 0:
        cause = "could not be saved to storage"
    else:
        cause = "could not be shown"
    return (
        f"\n{missing} of the {attempted} clips this job delivered {cause} "
        "and are not shown above.\n"
    )


def _promotable_as_frame(entry: dict[str, Any], valves: Any = None) -> bool:
    raw = getattr(valves, "VIDEO_FRAME_IMAGE_MIME_ALLOWLIST", None) if valves is not None else None
    allowed = _csv_set("image/jpeg,image/png,image/webp") if raw is None else _csv_set(raw)
    return _clean_str(entry.get("content_type")).split(";", 1)[0].lower() in allowed


def _host_entries(raw: Any) -> tuple[frozenset[str], frozenset[str]]:
    entries: set[str] = set()
    unusable: set[str] = set()
    for entry in _csv_set(raw):
        head, sep, port = entry.rpartition(":")
        if sep and head and port.isdigit():
            head = head.rstrip(".")
        else:
            head = entry.rstrip(".")
        if not head:
            continue
        if any(ch in head for ch in "!/") or "://" in head:
            unusable.add(head)
        entries.add(head)
    return frozenset(entries), frozenset(unusable)


def _host_in_scope(url: str, entries: frozenset[str]) -> bool:
    if not entries:
        return True
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.lower().rstrip(".")
    return any(host == entry or host.endswith("." + entry) for entry in entries)


def _is_pipe_published(field_name: str, url: str, relayed: frozenset[str]) -> bool:
    return field_name.startswith("input_references[") and url in relayed


_REFERENCE_FAMILY_FIELDS = frozenset(
    {"audio", "last_image", "video", "videos", "images", "input_references"}
)


def _reference_family(label: str) -> str:
    head = label.split(".")[-1].split("[")[0].strip()
    return head if head in _REFERENCE_FAMILY_FIELDS else ""


async def _decoded_length(b64: str) -> int:
    return len(await asyncio.to_thread(base64.b64decode, b64, validate=False))


async def _decoded_payload(b64: str) -> bytes:
    return await asyncio.to_thread(base64.b64decode, b64, validate=False)


@dataclass(frozen=True, slots=True)
class _AcceptedReference:
    file_id: str
    kind: str
    family: str
    mime: str
    b64: str
    via_file_host: bool
    filename: str
    blob: bytes


_FLOOR_UNPUBLISHED = (
    "unpublished: no OpenRouter page, model listing or recorded reply states this figure"
)

_FLOOR_OBSERVED_PREFIX = "observed: "


@dataclass(frozen=True, slots=True)
class _InputPixelFloor:
    pixels: int
    source: str

    @property
    def whose_rule(self) -> str:
        if self.source == _FLOOR_UNPUBLISHED:
            return (
                "a floor this pipe applies itself, because nobody publishes one for this "
                "model; a larger clip goes through"
            )
        if self.source.startswith(_FLOOR_OBSERVED_PREFIX):
            return (
                "a floor read off a refusal this model sent back: "
                f"{self.source[len(_FLOOR_OBSERVED_PREFIX):]}"
            )
        return f"a floor this model's own documentation states, at {self.source}"


_INPUT_PIXEL_FLOORS: dict[str, _InputPixelFloor] = {
    "bytedance/seedance-2.5": _InputPixelFloor(407696, _FLOOR_UNPUBLISHED),
    "bytedance/seedance-2.0": _InputPixelFloor(407696, _FLOOR_UNPUBLISHED),
    "bytedance/seedance-2.0-fast": _InputPixelFloor(407696, _FLOOR_UNPUBLISHED),
    "bytedance/seedance-2.0-mini": _InputPixelFloor(407696, _FLOOR_UNPUBLISHED),
    "bytedance/seedance-1-5-pro": _InputPixelFloor(407696, _FLOOR_UNPUBLISHED),
}


def _validate_input_pixel_floors(floors: dict[str, _InputPixelFloor]) -> None:
    for model_id, floor in floors.items():
        if floor.pixels <= 0:
            raise ValueError(
                f"{model_id!r} declares a pixel floor of {floor.pixels!r}, which would "
                "drop nothing or everything"
            )
        if not (
            floor.source == _FLOOR_UNPUBLISHED
            or floor.source.startswith(_FLOOR_OBSERVED_PREFIX)
            or floor.source.startswith("https://")
        ):
            raise ValueError(
                f"{model_id!r} leaves a user's clip out of a request they are paying for "
                "on the strength of a pixel count, so the table must record where that "
                "count came from - the page that publishes it, the refusal it was read "
                f"off, or the explicit note that nothing publishes it; got {floor.source!r}"
            )


_validate_input_pixel_floors(_INPUT_PIXEL_FLOORS)


_AUDIO_NEEDS_A_COMPANION = (
    "OpenRouter only takes a sound reference alongside a picture or a clip, so it was left out"
)

_REFERENCE_NEEDS_A_LINK = (
    "OpenRouter takes sound and video references as https links, not as uploaded files, "
    "so a file attached here cannot be sent. Paste a public https link to it instead"
)

_NO_SLUG = (
    "this model's catalog entry does not name the company to send it to"
)

_NOT_IN_SCHEMA = (
    "OpenRouter's video API does not define this key; set it on a chat model instead"
)

_OVER_URL_BUDGET = (
    f"only the first {_MAX_PASSTHROUGH_URLS} links in a request are forwarded"
)

_OVER_REFERENCE_BUDGET = (
    "the request's combined reference budget was already spent"
)

_A_COPY_MAY_ALREADY_BE_THERE = (
    "{host} may already hold a copy of it{second}"
)

_COULD_NOT_SAY_IT_FIRST = (
    "Your attachment has to be uploaded to a public file host before a video model can "
    "read it, and this chat could not be told that before it happened, because {cause}. "
    "Nothing was uploaded. Reload the chat and send it again."
)

_BLANK_NOTICE_CAUSE = (
    "the FILE_HOST_NOTICE setting has no wording to send"
)

_NO_EMITTER_CAUSE = (
    "there is no channel to this chat to send it on"
)

_NOTICE_REFUSED_CAUSE = (
    "this chat would not accept the notice"
)

_VIDEO_IS_STILL_RUNNING = (
    "OpenRouter is still working on this video. Nothing came back from the job for "
    "{waited}, so this message stopped waiting for it. The job was not cancelled and is "
    "still billed."
)

_VIDEO_CAN_BE_PICKED_BACK_UP = (
    " Press Continue Response on this message to pick this same job back up; Regenerate "
    "starts a fresh job under a new message instead."
)

_VIDEO_IS_STILL_RUNNING_STATUS = "Video generation is still running at OpenRouter."

_VIDEO_GENERATION_FAILED_STATUS = "No video was produced."


def _spoken_duration(seconds: float) -> str:
    if seconds < 120:
        return f"{round(seconds)} seconds"
    return f"{round(seconds / 60)} minutes"


def _intent_counters_enabled(valves: Any) -> tuple[int, int]:
    return (
        int(getattr(valves, "VIDEO_INTENT_MAX_CALLS_PER_CHAT", 0) or 0),
        int(getattr(valves, "VIDEO_INTENT_MAX_CALLS_PER_USER_DAY", 0) or 0),
    )


def _video_stall_window(valves: Any) -> float:
    total = getattr(valves, "HTTP_TOTAL_TIMEOUT_SECONDS", None)
    one_request = float(total) if total else float(getattr(valves, "HTTP_SOCK_READ_SECONDS", 0) or 0)
    return max(
        float(valves.VIDEO_MAX_POLL_TIME_SECONDS),
        float(valves.VIDEO_POLL_INTERVAL_MAX_SECONDS) + one_request,
    )


def _video_still_running_note(window: float) -> str:
    return _VIDEO_IS_STILL_RUNNING.format(waited=_spoken_duration(window))


_FILE_HOST_RECORD = (
    "> **Your {kind} {was} uploaded to {host}.** Anyone holding the link can open "
    "{it}, {retention}.\n"
)

RELAY_BLOCK_START = "relay_block_start"

RELAY_BLOCK_END = "relay_block_end"

WITHHELD_BLOCK_START = "withheld_block_start"

WITHHELD_BLOCK_END = "withheld_block_end"

_RELAY_BLOCK_REGION_RE = re.compile(
    r"\[openrouter:v1:" + re.escape(RELAY_BLOCK_START) + r":[^\]]+\]: #"
    r".*?"
    r"\[openrouter:v1:" + re.escape(RELAY_BLOCK_END) + r":[^\]]+\]: #\s*\n?",
    re.DOTALL,
)

_WITHHELD_BLOCK_REGION_RE = re.compile(
    r"\[openrouter:v1:" + re.escape(WITHHELD_BLOCK_START) + r":[^\]]+\]: #"
    r".*?"
    r"\[openrouter:v1:" + re.escape(WITHHELD_BLOCK_END) + r":[^\]]+\]: #\s*\n?",
    re.DOTALL,
)

_WITHHELD_RECORD = "> **Not sent with this video:** {items}\n"

_MAX_INPUT_REFERENCES = 16

_OVER_REFERENCE_COUNT = (
    f"only the first {_MAX_INPUT_REFERENCES} attachments in one request are sent as "
    "references"
)

_UNTYPED_REFERENCE = (
    "the file carries no media type the video API has a reference kind for"
)

_PUBLISHING_NEEDS_A_STORED_TYPE = (
    "Open WebUI has no media type recorded for this file, and the type your browser "
    "declared for it is not enough to put a file on a public host under"
)

_REFERENCE_KINDS: dict[str, str] = {
    "image": "image_url",
    "audio": "audio_url",
    "video": "video_url",
}

_SIZE_FIXES_THE_PIXELS = (
    "the exact size already fixes the pixels, and this pipe does not also send a "
    "resolution tier that disagrees with them"
)

_SIZE_IS_A_TIER = (
    "the size chosen is itself a resolution tier, so this pipe sends only one of the two"
)

_SIZE_CONTRADICTS_THE_RATIO = (
    "it is not the shape of the exact size chosen, and this pipe sends the size on its "
    "own rather than a request that contradicts itself. OpenRouter's video API does not "
    "say which of the two it would have honoured"
)

_ASPECT_RATIO_TOLERANCE = 0.025
"""How far a chosen ratio may sit from the pixels before this pipe stops sending it.

This pipe's number, not OpenRouter's: their video schema says only that ``size`` is
interchangeable with ``resolution`` + ``aspect_ratio``, and publishes no rule for the
two disagreeing. The rejection language belongs to the image API.
"""

_warned_dropped_video_param: set[str] = set()

_warned_pinned_attachment: set[str] = set()

_DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS: frozenset[str] = VIDEO_REQUEST_FIELDS

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..streaming.event_emitter import EventEmitter


def _write_and_close(handle: int, blob: bytes) -> None:
    with open(handle, "wb") as sink:
        sink.write(blob)


class VideoResizableSemaphore(asyncio.Semaphore):
    _debt: int

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self._limit = value
        self._debt = 0

    def release(self) -> None:
        if self._debt > 0:
            self._debt -= 1
            return
        super().release()

    def resize(self, new_limit: int) -> None:
        new_limit = int(new_limit)
        if new_limit > self._limit:
            for _ in range(new_limit - self._limit):
                if self._debt > 0:
                    self._debt -= 1
                else:
                    super().release()
        else:
            in_flight = self._debt + self._limit - self._value
            if in_flight > new_limit:
                self._debt = in_flight - new_limit
                self._value = 0
            else:
                self._debt = 0
                self._value = new_limit - in_flight
        self._limit = new_limit


class VideoGenerationAdapter:

    TERMINAL_SUCCESS: ClassVar[set[str]] = {"completed", "succeeded", "success"}
    TERMINAL_FAILURE: ClassVar[set[str]] = {"failed", "cancelled", "canceled", "expired"}
    JOB_MARKER_KIND = "videojob"
    MODEL_MARKER_KIND = "videomodel"

    def __init__(self, *, pipe: Pipe, logger: logging.Logger) -> None:
        self._pipe = pipe
        self.logger = logger
        self._persistence = VideoPersistence(logger=logger)
        self._intent_call_counts_per_chat: OrderedDict[str, int] = OrderedDict()
        self._intent_call_counts_per_user_day: dict[tuple[str, str], int] = {}
        self._intent_pruned_day: str = ""
        self._intent_breaker_until_ts: dict[str, float] = {}
        self._intent_breaker_swept_at: float = 0.0
        self._intent_failure_notified_chats: OrderedDict[str, None] = OrderedDict()

    async def generate(
        self,
        *,
        body: dict[str, Any],
        responses_body: Any,
        valves: Any,
        session: Any,
        event_emitter: EventEmitter | None,
        metadata: dict[str, Any],
        user: dict[str, Any],
        request: Any,
        user_obj: Any,
        normalized_model_id: str,
        api_model_id: str,
        outcome_sink: dict[str, Any] | None = None,
        breaker_key: str | None = None,
    ) -> str:
        prompt = self._extract_prompt(body)
        video_spec = OpenRouterModelRegistry.spec(normalized_model_id)
        video_model = video_spec.get("video_model") if isinstance(video_spec, dict) else {}
        if asks_for_help(self._extract_user_prompt(body)):
            content = render_video_help(
                api_model_id,
                video_model if isinstance(video_model, dict) else None,
                admin_valves=valves,
            )
            await self._emit_completion(event_emitter, content)
            return content

        chat_id = _clean_str(metadata.get("chat_id"))
        message_id = _clean_str(metadata.get("message_id"))
        if not (chat_id and message_id):
            from ..core.logging_system import SessionLogger

            request_id = SessionLogger.request_id.get() or secrets.token_hex(8)
            key = (f"api:{request_id}", "")
            self.logger.warning(
                "Video generation keyed to a request-scoped id: no chat_id/message_id metadata (request_id=%s).",
                request_id,
            )
        else:
            key = (chat_id, message_id)
        user_id = _clean_str(user.get("id")) or _clean_str(metadata.get("user_id")) or "anonymous"
        existing = await self._get_active_task(key)
        if existing is not None:
            return await self._await_existing_task(existing, event_emitter, outcome_sink)

        message_lock = await self._acquire_message_lock(key)
        global_semaphore: asyncio.Semaphore | None = None
        global_slot_acquired = False
        user_slot_acquired = False
        user_slot_lock: asyncio.Lock | None = None
        lifecycle_transferred = False
        submitted = False
        job_id = ""
        disclosure_block = ""
        withheld_record_written = False
        withheld: list[tuple[str, str]] = []
        relayed_families: set[tuple[str, str]] = set()

        try:
            existing = await self._get_active_task(key)
            if existing is not None:
                await self._release_message_lock(key, message_lock)
                message_lock = None  # type: ignore[assignment]
                return await self._await_existing_task(existing, event_emitter, outcome_sink)

            persisted = await self._persistence.load_message_content(chat_id=chat_id, message_id=message_id)
            if self._looks_like_final_video_content(persisted):
                await self._emit_completion(event_emitter, persisted)
                return persisted
            resume_job_id = self._extract_video_job_marker(persisted)
            if resume_job_id:
                user_slot_acquired, user_slot_lock = await self._try_acquire_user_slot(user_id, valves)
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
                global_slot_acquired = True
                job_id = resume_job_id
                await self._add_user_active_job(user_id, job_id, user_slot_lock)
                await self._emit_status(event_emitter, "Resuming video generation job...", done=False, progress=5)
                resumed_disclosure = self._recover_the_file_host_record(persisted)
                resumed_disclosure += self._recover_the_withheld_record(persisted)
                if persisted:
                    from .video_intent import _INTENT_BLOCK_REGION_RE
                    m = _INTENT_BLOCK_REGION_RE.search(persisted)
                    if m:
                        resumed_disclosure += m.group(0)
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
                    disclosure_block=resumed_disclosure,
                    breaker_key=breaker_key,
                )
                lifecycle_transferred = True
                async with self._pipe._video_active_tasks_dict_lock:
                    self._pipe._video_active_tasks[key] = bg_task
                result = await asyncio.shield(bg_task)
                self._settle_request(result, outcome_sink)
                await self._emit_completion(event_emitter, result.content, usage=result.usage)
                return result.content

            video_meta_pre = self._extract_video_metadata(metadata)
            intent_result: VideoIntentResult | None = None

            if self._intent_classifier_should_run(
                valves=valves,
                persisted_content=persisted,
                prompt=prompt,
                body=body,
                video_meta=video_meta_pre,
                metadata=metadata if isinstance(metadata, dict) else None,
                chat_id=chat_id if isinstance(chat_id, str) else "",
                user_id=user_id if isinstance(user_id, str) else "",
            ):
                try:
                    await self._emit_status(
                        event_emitter, "Analyzing request...", done=False,
                    )
                    intent_result = await resolve_intent(
                        body=body,
                        video_meta=video_meta_pre,
                        video_model=video_model or {},
                        valves=valves,
                        request=request,
                        user_obj=user_obj or user,
                        chat_id=chat_id if isinstance(chat_id, str) else "",
                        logger=self.logger,
                        fallback_prompt_text=prompt,
                        metadata=metadata if isinstance(metadata, dict) else None,
                    )
                    self._intent_record_call(
                        chat_id if isinstance(chat_id, str) else "",
                        user_id if isinstance(user_id, str) else "",
                        valves=valves,
                    )
                    if intent_result.classifier_failed:
                        self._intent_record_failure(user_id if isinstance(user_id, str) else "")
                        self.logger.warning(
                            "video_intent classifier_failed=True; reason=%s; "
                            "breaker tripped",
                            intent_result.failure_reason or "<unknown>",
                        )
                        latch_key = self._intent_latch_key(chat_id)
                        log_key = self._intent_notice_key(chat_id) or "<not retained>"
                        if self._intent_was_failure_notified(latch_key):
                            self.logger.debug(
                                "first-failure toast suppressed (chat already "
                                "notified): chat_key=%s", log_key,
                            )
                        elif event_emitter is None:
                            self.logger.debug(
                                "first-failure toast suppressed: event_emitter "
                                "is None (chat_key=%s)", log_key,
                            )
                        else:
                            self._intent_note_failure_notified(latch_key)
                            try:
                                await event_emitter({
                                    "type": "notification",
                                    "data": {
                                        "type": "warning",
                                        "content": (
                                            "Intent inference unavailable; "
                                            "using simple text-to-video."
                                        ),
                                    },
                                })
                                self.logger.info(
                                    "first-failure toast emitted (chat_key=%s)",
                                    log_key,
                                )
                            except Exception as exc:
                                self.logger.warning(
                                    "first-failure toast emission raised "
                                    "(suppressed): %s", exc, exc_info=True,
                                )
                    if (
                        intent_result.clarification is not None
                        and intent_result.clarification.needs
                    ):
                        clar_content = render_clarification_message(intent_result)
                        await self._emit_status(
                            event_emitter, "Need a quick clarification", done=True,
                        )
                        await self._emit_completion(event_emitter, clar_content)
                        self._emit_intent_telemetry(intent_result, valves=valves, chat_id=chat_id)
                        return clar_content
                    reused_frame_pref_raw = resolve_intent_user_setting(
                        metadata, "frame_extraction_index",
                        valves, "VIDEO_INTENT_FRAME_EXTRACTION_INDEX", "last",
                    )
                    reused_frame_pref: Literal["first", "last"] = (
                        "first" if reused_frame_pref_raw == "first" else "last"
                    )
                    self._apply_uploaded_attachment_retargeting(
                        intent_result, video_meta_pre, valves,
                    )
                    thumbs = await self._materialise_frame_plan(
                        intent=intent_result,
                        video_meta=video_meta_pre,
                        request=request,
                        user_obj=user_obj or user,
                        chat_id=chat_id if isinstance(chat_id, str) else "",
                        message_id=message_id if isinstance(message_id, str) else "",
                        reused_frame_index=reused_frame_pref,
                    )
                    if isinstance(metadata, dict):
                        pipe_meta = metadata.setdefault(_PIPE_METADATA_KEY, {})
                        if isinstance(pipe_meta, dict):
                            pipe_meta["video_generation"] = video_meta_pre
                    confirm_mode = str(
                        resolve_intent_user_setting(
                            metadata, "confirm_mode",
                            valves, "VIDEO_INTENT_CONFIRM_MODE", "on_reference",
                        )
                        or "on_reference"
                    )
                    if should_emit_confirmation_footer(
                        intent_result, confirm_mode=confirm_mode,
                    ):
                        disclosure_block = render_intent_disclosure_block(
                            intent=intent_result,
                            thumb_urls=thumbs,
                        )
                    if intent_result.prompt:
                        prompt = intent_result.prompt
                    self._emit_intent_telemetry(intent_result, valves=valves, chat_id=chat_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.logger.warning(
                        "video_intent classifier failed (degrade-open): %s", exc, exc_info=True
                    )
                    self._intent_record_failure(user_id if isinstance(user_id, str) else "")
                    latch_key = self._intent_latch_key(chat_id)
                    log_key = self._intent_notice_key(chat_id) or "<not retained>"
                    already = self._intent_was_failure_notified(latch_key)
                    if not already:
                        if event_emitter is None:
                            self.logger.debug(
                                "first-failure toast suppressed: event_emitter "
                                "is None (chat_key=%s)", log_key,
                            )
                        else:
                            self._intent_note_failure_notified(latch_key)
                            with contextlib.suppress(Exception):
                                await event_emitter({
                                    "type": "notification",
                                    "data": {
                                        "type": "warning",
                                        "content": (
                                            "Intent inference unavailable; "
                                            "using simple text-to-video."
                                        ),
                                    },
                                })
                    intent_result = None
                    disclosure_block = ""

            user_slot_acquired, user_slot_lock = await self._try_acquire_user_slot(user_id, valves)
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

            video_meta = self._extract_video_metadata(metadata)
            frame_images = await self._encode_frame_images(
                video_meta, video_model, valves, user_obj=user_obj or user,
            )
            vetted_addresses: dict[str, bool] = {}
            try:
                input_references = await self._encode_input_references(
                    video_meta, valves, withheld=withheld, user_obj=user_obj or user,
                    video_model=video_model, relayed=relayed_families,
                    companions=bool(frame_images), event_emitter=event_emitter,
                    vetted=vetted_addresses,
                )
            finally:
                disclosure_block = self._with_the_file_host_record(
                    disclosure_block, valves, relayed_families
                )
            if not prompt.strip():
                content = self._build_failure_content(
                    job_id="",
                    model_id=api_model_id,
                    reason=(
                        "Video generation needs a prompt in your message. Add words describing "
                        "the video you want \u2014 an attachment alone is not enough."
                    ),
                )
                if disclosure_block:
                    content = disclosure_block + "\n" + content
                await self._emit_status(event_emitter, "Video generation could not start.", done=True)
                await self._emit_completion(event_emitter, content)
                return content
            provider_block = requested_provider_block(
                SimpleNamespace(provider=getattr(responses_body, "provider", None)), metadata
            )
            provider_options = self._extract_provider_options(getattr(responses_body, "provider", None), metadata)
            payload = await self._build_payload(
                api_model_id=api_model_id,
                prompt=prompt,
                video_meta=video_meta,
                video_model=video_model,
                provider_block=provider_block,
                frame_images=frame_images,
                input_references=input_references,
                provider_options=provider_options,
                withheld=withheld,
                vetted=vetted_addresses,
                video_reference_allowed_domains=getattr(
                    valves, "VIDEO_REFERENCE_ALLOWED_DOMAINS", ""
                ),
            )
            global_semaphore = self._ensure_global_semaphore(valves)
            await global_semaphore.acquire()
            global_slot_acquired = True

            disclosure_block = self._with_the_withheld_record(disclosure_block, withheld)
            withheld_record_written = True
            await self._emit_status(event_emitter, "Submitting video generation job...", done=False)

            client = OpenRouterVideoClient(
                session,
                base_url=valves.BASE_URL,
                api_key=self._resolve_api_key(valves),
                logger=self.logger,
                http_referer=_select_openrouter_http_referer(valves),
                valves=valves,
                user=user_obj,
                owui_chat_id=chat_id,
            )
            submitted = True
            accepted = await client.submit(payload)
            job_id = self._extract_job_id(accepted)
            if not job_id:
                raise VideoGenerationError("OpenRouter accepted the request without returning a video job id.")
            await self._add_user_active_job(user_id, job_id, user_slot_lock)
            if (
                event_emitter is not None
                and isinstance(chat_id, str)
                and is_linkable_chat(chat_id)
            ):
                pending_content = self._build_pending_content(
                    job_id=job_id,
                    model_id=api_model_id,
                )
                if disclosure_block:
                    pending_content = disclosure_block + "\n" + pending_content
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
                disclosure_block=disclosure_block,
                breaker_key=breaker_key,
            )
            lifecycle_transferred = True
            async with self._pipe._video_active_tasks_dict_lock:
                self._pipe._video_active_tasks[key] = bg_task

            result = await asyncio.shield(bg_task)
            self._settle_request(result, outcome_sink)
            await self._emit_completion(event_emitter, result.content, usage=result.usage)
            return result.content
        except asyncio.CancelledError:
            raise
        except OpenRouterAPIError as exc:
            self._count_a_failed_start(submitted and not lifecycle_transferred, outcome_sink, breaker_key)
            self.logger.warning("Video generation rejected (job_id=%s): %s", job_id, exc)
            return await self._pipe._ensure_error_formatter()._report_openrouter_error(
                exc,
                event_emitter=event_emitter,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
                partial_answer=disclosure_block,
            )
        except Exception as exc:
            self._count_a_failed_start(submitted and not lifecycle_transferred, outcome_sink, breaker_key)
            self.logger.exception("Video generation request failed (job_id=%s)", job_id)
            if not withheld_record_written:
                disclosure_block = self._with_the_withheld_record(disclosure_block, withheld)
            reason = str(exc) or exc.__class__.__name__
            content = self._build_failure_content(job_id=job_id, model_id=api_model_id, reason=reason)
            if disclosure_block:
                content = disclosure_block + "\n" + content
            await self._emit_status(event_emitter, _VIDEO_GENERATION_FAILED_STATUS, done=True)
            await self._emit_completion(event_emitter, content)
            return content
        finally:
            if not lifecycle_transferred:
                if global_slot_acquired and global_semaphore is not None:
                    global_semaphore.release()
                await asyncio.shield(
                    self._release_presubmit_slots(
                        key,
                        user_id,
                        job_id,
                        message_lock,
                        release_user_slot=user_slot_acquired,
                    )
                )

    @staticmethod
    def _settle_request(result: VideoLifecycleResult, outcome_sink: dict[str, Any] | None) -> None:
        if outcome_sink is not None:
            outcome_sink["error_occurred"] = result.failed or not result.file_id

    def _count_a_failed_start(
        self, sent_to_openrouter: bool, outcome_sink: dict[str, Any] | None, breaker_key: str | None
    ) -> None:
        if outcome_sink is not None:
            outcome_sink["error_occurred"] = True
        if sent_to_openrouter:
            _record_failed_call(self._pipe, breaker_key)

    async def _cleanup_step(
        self, phase: str, key: tuple[str, str], what: str, coro: Awaitable[None]
    ) -> None:
        """Run one release and swallow its failure so the next release still runs.

        Cancellation propagates; anything else is logged. Two byte-near-identical
        copies of this lived 250 lines apart, differing only in the word "pre-submit",
        so changing which exceptions propagate in one left the other on the old
        contract -- and the symptom is a stranded semaphore that surfaces hours later
        as a generation that never starts.
        """
        try:
            await coro
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception:
            self.logger.warning(
                "Video %s cleanup step '%s' failed for %s; its resource is stranded",
                phase,
                what,
                key,
                exc_info=True,
            )

    async def _release_presubmit_slots(
        self,
        key: tuple[str, str],
        user_id: str,
        job_id: str,
        message_lock: asyncio.Lock | None,
        *,
        release_user_slot: bool,
    ) -> None:
        """Release what generate() acquired before handing off to the lifecycle task.

        Each step is independent: a failure in one must not skip the others. Without
        the isolation the first failure propagates out of the shielded coroutine and
        every release below it is skipped, stranding those resources for the lifetime
        of the process.
        """

        _step = functools.partial(self._cleanup_step, "pre-submit", key)

        if release_user_slot:
            await _step("user slot", self._release_user_slot(user_id, job_id))
        if message_lock is not None:
            await _step("message lock", self._release_message_lock(key, message_lock))

    def _create_lifecycle_task(
        self,
        *,
        key: tuple[str, str],
        job_id: str,
        api_model_id: str,
        normalized_model_id: str,
        valves: Any,
        event_emitter: EventEmitter | None,
        user: dict[str, Any],
        user_obj: Any,
        chat_id: str,
        message_id: str,
        request: Any,
        user_id: str,
        global_semaphore: asyncio.Semaphore,
        message_lock: asyncio.Lock,
        started_at: float,
        disclosure_block: str = "",
        breaker_key: str | None = None,
    ) -> asyncio.Task[VideoLifecycleResult]:
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
                disclosure_block=disclosure_block,
                breaker_key=breaker_key,
            ),
            name=f"openrouter-video-{job_id}",
        )
        task.add_done_callback(self._consume_background_exception)
        return task

    async def _record_what_it_cost(
        self,
        *,
        usage: dict[str, Any],
        billing: dict[str, bool],
        valves: Any,
        user_id: str,
        api_model_id: str,
        user_obj: Any,
    ) -> None:
        """Record what a terminal poll says the job cost, once, whatever happens next.

        The latch is raised BEFORE the write, not after: a cancel delivered inside the
        write leaves the job recorded and unwinds through the caller's cancel handler,
        which would otherwise see an unraised latch and record it a second time.
        """
        if not usage or billing["costed"]:
            return
        billing["costed"] = True
        with contextlib.suppress(Exception):
            await maybe_dump_costs_snapshot(
                self._pipe,
                valves,
                user_id=user_id,
                model_id=api_model_id,
                usage=usage,
                user_obj=user_obj,
                pipe_id=self._pipe.id,
            )

    async def _run_lifecycle_after_submit(
        self,
        *,
        key: tuple[str, str],
        job_id: str,
        api_model_id: str,
        normalized_model_id: str,
        valves: Any,
        event_emitter: EventEmitter | None,
        user: dict[str, Any],
        user_obj: Any,
        chat_id: str,
        message_id: str,
        request: Any,
        user_id: str,
        global_semaphore: asyncio.Semaphore,
        message_lock: asyncio.Lock,
        started_at: float,
        disclosure_block: str = "",
        breaker_key: str | None = None,
    ) -> VideoLifecycleResult:
        content = ""
        failed = False
        usage: dict[str, Any] = {}
        billing: dict[str, bool] = {"costed": False}
        file_id: str | None = None
        output_mime = ""
        description = ""
        downloads: list[DownloadedVideo] = []
        tmp_dir: Path | None = None
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
                    valves=valves,
                    user=user_obj,
                    owui_chat_id=chat_id,
                )
                status_payload = await self._poll_until_terminal(client, job_id, valves, event_emitter)
                usage = self._coerce_video_usage(status_payload.get("usage"))
                status = _clean_str(status_payload.get("status")).lower()
                if status not in self.TERMINAL_SUCCESS:
                    raise VideoGenerationError(self._status_failure_reason(status_payload, status))

                generation_id = _clean_str(status_payload.get("generation_id"))
                if generation_id:
                    self.logger.info(
                        "Video job %s produced generation %s", job_id, generation_id
                    )
                counter: Any = getattr(client, "output_count", None)
                reported = 1 if counter is None else int(counter(status_payload))
                outputs = min(reported, _MAX_VIDEO_OUTPUTS)
                if reported > outputs:
                    self.logger.warning(
                        "Video job %s reported %d outputs; downloading the first %d",
                        job_id, reported, outputs,
                    )
                max_bytes = int(valves.REMOTE_VIDEO_MAX_SIZE_MB) * 1024 * 1024
                allowed_mimes = _csv_set(valves.VIDEO_OUTPUT_MIME_ALLOWLIST)
                await self._emit_status(event_emitter, "Downloading generated video...", done=False, progress=80)
                bearer = client.bearer_header()
                tmp_dir = Path(tempfile.mkdtemp(prefix="openrouter-video-"))
                for index in range(outputs):
                    tmp_path = tmp_dir / f"job-{job_id}-{index}.bin"
                    download_result = await self._pipe._multimodal_handler._download_remote_url_streaming(
                        client.content_url(job_id, index=index),
                        tmp_path,
                        chunk_size=int(valves.VIDEO_DOWNLOAD_CHUNK_SIZE),
                        max_size_bytes=max_bytes,
                        mime_allowlist=allowed_mimes,
                        extra_headers=bearer,
                    )
                    if not download_result:
                        with contextlib.suppress(Exception):
                            tmp_path.unlink(missing_ok=True)
                        if downloads:
                            self.logger.warning(
                                "Video job %s reported %d outputs but clip %d could not be "
                                "downloaded; the clips already fetched are kept",
                                job_id, reported, index,
                            )
                        else:
                            self.logger.warning(
                                "Video job %s reported %d outputs but clip %d could not be "
                                "downloaded",
                                job_id, reported, index,
                            )
                        continue
                    downloads.append(
                        DownloadedVideo(
                            path=download_result["path"],
                            mime_type=download_result["mime_type"] or "",
                            size_bytes=int(download_result["size_bytes"] or 0),
                        )
                    )
                output_mime = downloads[0].mime_type if downloads else ""
            finally:
                # Shielded: `suppress(Exception)` does not catch CancelledError, so a
                # cancel delivered here abandons the close and strands the connector.
                with contextlib.suppress(Exception):
                    await asyncio.shield(session.close())

            elapsed = max(0.0, time.monotonic() - started_at)
            if not downloads:
                raise VideoGenerationError(
                    f"None of the {outputs} clips this job delivered could not be fetched from "
                    "OpenRouter."
                    if reported > 1 else
                    "Generated video could not be downloaded from OpenRouter."
                )
            file_ids: list[str] = []
            for index, clip in enumerate(downloads):
                suffix = "" if len(downloads) == 1 else f"-{index}"
                stored = await self._pipe._file_gateway.upload_to_owui_storage_from_path(
                    request=request,
                    user=user_obj or user,
                    source_path=clip.path,
                    filename=(
                        f"openrouter-video-{job_id}{suffix}"
                        f"{extension_for_video_mime(clip.mime_type)}"
                    ),
                    mime_type=clip.mime_type,
                    chat_id=chat_id,
                    message_id=message_id,
                    owui_user_id=user_id,
                )
                if stored:
                    file_ids.append(stored)
                    with contextlib.suppress(Exception):
                        clip.path.unlink(missing_ok=True)
                else:
                    self.logger.warning(
                        "Video job %s downloaded %d clips but clip %d could not be stored "
                        "in Open WebUI; the clips already stored are kept",
                        job_id,
                        len(downloads),
                        index,
                    )
            if not file_ids:
                raise VideoGenerationError(
                    "Generated video could not be stored in Open WebUI; the upload failed."
                )
            file_id = file_ids[0]
            content = self._build_success_content(
                job_id=job_id,
                model_id=api_model_id,
                file_ids=file_ids,
                elapsed=elapsed,
                usage=usage,
                attempted=outputs,
                fetched=len(downloads),
            )
            if disclosure_block:
                content = disclosure_block + "\n" + content
            description = self._format_final_status(elapsed=elapsed, usage=usage, valves=valves)
            await self._emit_status(event_emitter, description, done=True, progress=100)
            await self._record_what_it_cost(
                usage=usage,
                billing=billing,
                valves=valves,
                user_id=user_id,
                api_model_id=api_model_id,
                user_obj=user_obj,
            )
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
            await asyncio.shield(
                self._record_what_it_cost(
                    usage=usage,
                    billing=billing,
                    valves=valves,
                    user_id=user_id,
                    api_model_id=api_model_id,
                    user_obj=user_obj,
                )
            )
            raise
        except VideoGenerationStalled as exc:
            self.logger.warning("Video job %s outlasted its status window: %s", job_id, exc)
            elapsed = max(0.0, time.monotonic() - started_at)
            note = str(exc)
            if is_linkable_chat(chat_id):
                note += _VIDEO_CAN_BE_PICKED_BACK_UP
            content = self._build_pending_content(
                job_id=job_id, model_id=api_model_id, note=note
            )
            if disclosure_block:
                content = disclosure_block + "\n" + content
            description = _VIDEO_IS_STILL_RUNNING_STATUS
            await self._emit_status(event_emitter, description, done=True)
            await self._record_what_it_cost(
                usage=usage,
                billing=billing,
                valves=valves,
                user_id=user_id,
                api_model_id=api_model_id,
                user_obj=user_obj,
            )
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
        except Exception as exc:
            self.logger.exception("Video lifecycle failed (job_id=%s)", job_id)
            failed = True
            _record_failed_call(self._pipe, breaker_key)
            elapsed = max(0.0, time.monotonic() - started_at)
            reason = str(exc) or exc.__class__.__name__
            content = self._build_failure_content(job_id=job_id, model_id=api_model_id, reason=reason)
            if disclosure_block:
                content = disclosure_block + "\n" + content
            description = _VIDEO_GENERATION_FAILED_STATUS
            await self._emit_status(event_emitter, description, done=True)
            await self._record_what_it_cost(
                usage=usage,
                billing=billing,
                valves=valves,
                user_id=user_id,
                api_model_id=api_model_id,
                user_obj=user_obj,
            )
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
            for clip in downloads:
                try:
                    clip.path.unlink(missing_ok=True)
                except Exception:
                    self.logger.warning(
                        "Could not remove the temp video file %s; it will accumulate",
                        clip.path,
                        exc_info=True,
                    )
            if tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir)
                except FileNotFoundError:
                    pass
                except OSError:
                    self.logger.warning(
                        "Could not remove the per-job temp directory %s; these leak one "
                        "empty directory per generation",
                        tmp_dir,
                        exc_info=True,
                    )
            global_semaphore.release()
            await asyncio.shield(
                self._finalize_generation(
                    key, user_id, job_id, message_lock, asyncio.current_task()
                )
            )

    async def _finalize_generation(
        self,
        key: tuple[str, str],
        user_id: str,
        job_id: str,
        message_lock: asyncio.Lock,
        owner: asyncio.Task | None = None,
    ) -> None:
        _step = functools.partial(self._cleanup_step, "lifecycle", key)

        async def _drop_active_task() -> None:
            async with self._pipe._video_active_tasks_dict_lock:
                current = self._pipe._video_active_tasks.get(key)
                if current is (owner or asyncio.current_task()):
                    self._pipe._video_active_tasks.pop(key, None)

        await _step("active-task entry", _drop_active_task())
        await _step("user slot", self._release_user_slot(user_id, job_id))
        await _step("message lock", self._release_message_lock(key, message_lock))

    async def _poll_until_terminal(
        self,
        client: OpenRouterVideoClient,
        job_id: str,
        valves: Any,
        event_emitter: EventEmitter | None,
    ) -> dict[str, Any]:
        initial_delay = float(valves.VIDEO_INITIAL_POLL_DELAY_SECONDS)
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        interval = float(valves.VIDEO_POLL_INTERVAL_SECONDS)
        max_interval = float(valves.VIDEO_POLL_INTERVAL_MAX_SECONDS)
        backoff = float(valves.VIDEO_POLL_BACKOFF_FACTOR)
        stall_window = _video_stall_window(valves)
        deadline = time.monotonic() + stall_window
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

        polling_url = ""
        while True:
            if time.monotonic() > deadline:
                raise VideoGenerationStalled(_video_still_running_note(stall_window))
            try:
                payload = await client.status(job_id, polling_url=polling_url)
                consecutive_errors = 0
                polling_url = _clean_str(payload.get("polling_url"))
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
            deadline = time.monotonic() + stall_window
            progress = 5 if status == "pending" else 50
            await _maybe_emit(status, progress)
            await asyncio.sleep(min(interval, max_interval))
            interval = min(max_interval, interval * backoff)

    async def _await_existing_task(
        self,
        task: asyncio.Task[VideoLifecycleResult],
        event_emitter: EventEmitter | None,
        outcome_sink: dict[str, Any] | None = None,
    ) -> str:
        await self._emit_status(event_emitter, "Waiting for active video generation job...", done=False)
        result = await asyncio.shield(task)
        self._settle_request(result, outcome_sink)
        await self._emit_status(event_emitter, result.status_description, done=True)
        await self._emit_completion(event_emitter, result.content, usage=result.usage)
        return result.content

    async def _get_active_task(self, key: tuple[str, str]) -> asyncio.Task[VideoLifecycleResult] | None:
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

    async def _release_user_lock(self, user_id: str, lock: asyncio.Lock) -> None:
        async with self._pipe._video_user_locks_dict_lock:
            refs = self._pipe._video_user_lock_refs.get(user_id, 0) - 1
            if refs <= 0:
                self._pipe._video_user_lock_refs.pop(user_id, None)
                if (
                    self._pipe._video_user_locks.get(user_id) is lock
                    and not lock.locked()
                ):
                    self._pipe._video_user_locks.pop(user_id, None)
            else:
                self._pipe._video_user_lock_refs[user_id] = refs

    def _ensure_global_semaphore(self, valves: Any) -> asyncio.Semaphore:
        limit = int(valves.MAX_CONCURRENT_VIDEO_GENS)
        cls = type(self._pipe)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for attr in ("_video_global_semaphore",):
            sem = getattr(cls, attr, None)
            if sem is None:
                continue
            try:
                sem_loop = getattr(sem, "_get_loop", lambda: None)()
            except RuntimeError:
                sem_loop = None
            if current_loop is not None and sem_loop is not current_loop:
                setattr(cls, attr, None)
        if cls._video_global_semaphore is None:
            cls._video_global_semaphore = VideoResizableSemaphore(limit)
        else:
            if int(getattr(cls._video_global_semaphore, "_limit", limit)) != limit:
                cast(VideoResizableSemaphore, cls._video_global_semaphore).resize(limit)
        cls._video_global_limit = limit
        return cls._video_global_semaphore

    async def _try_acquire_user_slot(self, user_id: str, valves: Any) -> tuple[bool, asyncio.Lock]:
        limit = int(valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER)
        async with self._pipe._video_user_locks_dict_lock:
            lock = self._pipe._video_user_locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._pipe._video_user_locks[user_id] = lock
            self._pipe._video_user_lock_refs[user_id] = self._pipe._video_user_lock_refs.get(user_id, 0) + 1
        refused = False
        async with lock:
            current = int(self._pipe._video_user_active_counts.get(user_id, 0))
            if current >= limit:
                refused = True
            else:
                self._pipe._video_user_active_counts[user_id] = current + 1
        if refused:
            await asyncio.shield(self._release_user_lock(user_id, lock))
            return False, lock
        return True, lock

    async def _add_user_active_job(
        self, user_id: str, job_id: str, lock: asyncio.Lock | None = None
    ) -> None:
        if lock is None:
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
                lock = asyncio.Lock()
                self._pipe._video_user_locks[user_id] = lock
        try:
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
        finally:
            await asyncio.shield(self._release_user_lock(user_id, lock))

    async def _build_payload(
        self,
        *,
        api_model_id: str,
        prompt: str,
        video_meta: dict[str, Any],
        video_model: Any,
        frame_images: list[dict[str, Any]],
        provider_options: dict[str, Any],
        provider_block: dict[str, Any] | None = None,
        input_references: list[dict[str, Any]] | None = None,
        withheld: list[tuple[str, str]] | None = None,
        vetted: dict[str, bool] | None = None,
        video_reference_allowed_domains: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": api_model_id,
            "prompt": prompt,
        }
        params = video_meta.get("params")
        top_level, passthrough = self._split_allowed_parameters(video_model)
        provider_params: dict[str, Any] = {}
        if isinstance(params, dict):
            for key, value in params.items():
                if value is None or value == "":
                    continue
                if key in top_level:
                    payload[key] = value
                    continue
                target = self._select_passthrough_key(key, passthrough)
                if target:
                    provider_params[target] = value
                    continue
                documented = _clean_str(key) in _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS
                self.logger.log(
                    warn_level(_warned_dropped_video_param, f"{api_model_id}:{key}"),
                    "Dropping video parameter %r for %r: %s",
                    key,
                    api_model_id,
                    "it is a documented top-level field, but this model's catalog entry publishes "
                    "no matching supported_* list, so the pipe will neither validate it nor ship "
                    "it inside provider.options"
                    if documented
                    else "the catalog entry does not list it as an allowed passthrough parameter",
                )
        self._apply_size_consistency(payload, video_model, withheld)
        if frame_images:
            payload["frame_images"] = frame_images
        if input_references:
            payload["input_references"] = input_references
        merged_options = dict(provider_options) if isinstance(provider_options, dict) else {}
        if provider_params:
            bulky = {
                key: provider_params.pop(key)
                for key in ("video", "videos", "audio", "images", "last_image")
                if key in provider_params
            }
            candidates = self._provider_slug_candidates(video_model, api_model_id)
            if not candidates and withheld is not None:
                withheld.extend(
                    (name, _NO_SLUG) for name in sorted(provider_params) + sorted(bulky)
                )
            carrier = carrier_slug(provider_block or {}, candidates, pin_routes=False)
            if bulky and len(candidates) > 1:
                self.logger.log(
                    warn_level(_warned_pinned_attachment, api_model_id),
                    "%r lists %d providers; the attached %s is sent under %r only. The video "
                    "API accepts no provider routing controls, so if OpenRouter routes "
                    "elsewhere the attachment is ignored.",
                    api_model_id,
                    len(candidates),
                    ", ".join(sorted(bulky)),
                    carrier,
                )
            targets = candidates
            for slug in targets:
                existing = merged_options.get(slug)
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(provider_params)
                if merged:
                    merged_options[slug] = merged
            if bulky and carrier:
                existing = merged_options.get(carrier)
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(bulky)
                merged_options[carrier] = merged
        normalised = self._normalise_provider_options(merged_options)
        routing = {
            key: value
            for key, value in (provider_block or {}).items()
            if key != "options"
        }
        block, unsupported = restrict_provider_block(
            merge_provider_options({**routing, "options": normalised}, None, {}),
            VIDEO_PROVIDER_KEYS,
        )
        if unsupported:
            if withheld is not None:
                withheld.extend((name, _NOT_IN_SCHEMA) for name in unsupported)
            self.logger.log(
                warn_level(_warned_video_provider_keys, str(payload.get("model", ""))),
                "Provider preferences not accepted by the video API were not sent for %r: %s. "
                "Sending them would read as a control in force while nothing enforces it.",
                payload.get("model"),
                ", ".join(unsupported),
            )
        if block:
            payload["provider"] = block
        own = payload.get("input_references")
        relayed = frozenset(
            entry[kind]["url"]
            for entry in (own if isinstance(own, list) else ())
            if isinstance(entry, dict)
            for kind in entry
            if kind != "type"
            and isinstance(entry[kind], dict)
            and isinstance(entry[kind].get("url"), str)
            and not entry[kind]["url"].startswith("data:")
        ) & frozenset(link for link, ok in (vetted or {}).items() if ok and link)
        await self._validate_passthrough_urls(
            payload, withheld, vetted=vetted,
            relayed=relayed,
            video_reference_allowed_domains=video_reference_allowed_domains,
        )
        return payload

    @staticmethod
    def _withheld_notice(withheld: list[tuple[str, str]]) -> str:
        """Group withheld items by cause, so one sentence never speaks for three of them.

        A key the schema does not define, a parameter with no slug to key it under, and a
        URL list past the request ceiling are three different problems with three different
        remedies; rendering them under one clause makes two of the three untrue.
        """
        grouped: dict[str, list[str]] = {}
        for item, reason in withheld:
            grouped.setdefault(reason, []).append(item)
        return "; ".join(
            f"{summarise_names(items, 16)} was not sent ({reason})"
            for reason, items in grouped.items()
        )

    async def _validate_passthrough_urls(
        self,
        payload: dict[str, Any],
        withheld: list[tuple[str, str]] | None = None,
        seen: dict[str, bool] | None = None,
        budget: list[int] | None = None,
        vetted: dict[str, bool] | None = None,
        depth: int = 0,
        deadline: float | None = None,
        video_reference_allowed_domains: str | None = None,
        relayed: frozenset[str] | None = None,
    ) -> None:
        root = seen is None
        if seen is None:
            seen = dict(vetted) if isinstance(vetted, dict) else {}
        entries, unusable = _host_entries(video_reference_allowed_domains)
        for entry in sorted(unusable):
            self.logger.log(
                warn_level(_warned_reference_scope_entry, f"video_reference_scope_entry:{entry}"),
                "VIDEO_REFERENCE_ALLOWED_DOMAINS entry %r cannot match any host, so every "
                "reference URL is refused. Write a bare host such as "
                "media.example.com: a ! block entry, a whole URL and a CIDR range are "
                "not accepted here.",
                entry,
            )
        relayed = frozenset(relayed or ())
        if deadline is None:
            deadline = time.monotonic() + ADDRESS_CHECK_BUDGET_SECONDS
        budget = [_MAX_PASSTHROUGH_URLS] if budget is None else budget
        provider = payload.get("provider")
        options = provider.get("options") if isinstance(provider, dict) else None
        if isinstance(options, dict):
            try:
                refuse_past_the_scan_depth(depth + _OPTIONS_HOP_DEPTH)
            except UnvettableRequest as exc:
                raise VideoGenerationError(str(exc)) from exc
            for nested in options.values():
                if isinstance(nested, dict):
                    await self._validate_passthrough_urls(
                        nested, withheld, seen, budget,
                        depth=depth + _OPTIONS_HOP_DEPTH,
                        deadline=deadline,
                        video_reference_allowed_domains=video_reference_allowed_domains,
                        vetted=vetted,
                        relayed=relayed,
                    )
        url_fields = ("audio", "last_image", "video")
        array_fields = ("videos", "images")
        handler = self._pipe._multimodal_handler

        def _spend(url: str) -> bool:
            """One pool for the whole request: breadth, depth and array length draw on it."""
            if url in seen:
                return True
            if budget[0] <= 0:
                return False
            budget[0] -= 1
            return True

        async def _check(url: str, field_name: str) -> None:
            if url.startswith("data:"):
                return
            safe = seen.get(url)
            if safe is None:
                safe = bool(await handler._is_safe_url(
                    url, seconds=min(ADDRESS_CHECK_SECONDS, deadline - time.monotonic()),
                ))
                seen[url] = safe
            if not safe:
                raise VideoGenerationError(
                    f"Refusing to forward unsafe URL in '{field_name}'. Use https:// or "
                    f"an allowlisted http:// destination."
                )
            if entries and not _is_pipe_published(field_name, url, relayed) and not _host_in_scope(url, entries):
                self.logger.log(
                    warn_level(
                        _warned_reference_scope,
                        f"video_reference_scope:{_reference_family(field_name) or 'provider_options'}",
                    ),
                    "Refusing to forward '%s': its host is outside the hosts this "
                    "deployment allows. Add it to VIDEO_REFERENCE_ALLOWED_DOMAINS, or "
                    "clear that valve.",
                    field_name,
                )
                raise VideoGenerationError(
                    f"Refusing to forward '{field_name}': its host is outside the hosts "
                    f"this deployment allows. Add it to VIDEO_REFERENCE_ALLOWED_DOMAINS, "
                    f"or clear that valve."
                )

        for field_name in url_fields:
            raw = payload.get(field_name)
            if not isinstance(raw, str):
                continue
            cleaned = raw.strip()
            if not cleaned:
                payload.pop(field_name, None)
                continue
            if not _spend(cleaned):
                if withheld is not None:
                    withheld.append((field_name, _OVER_URL_BUDGET))
                payload.pop(field_name, None)
                continue
            await _check(cleaned, field_name)
            payload[field_name] = cleaned

        for field_name in array_fields:
            items = payload.get(field_name)
            if not isinstance(items, list):
                continue
            dropped = 0
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
                if not _spend(cleaned):
                    dropped += 1
                    continue
                await _check(cleaned, f"{field_name}[{idx}]")
                if isinstance(item, dict):
                    new_item = dict(item)
                    new_item["url"] = cleaned
                    validated.append(new_item)
                else:
                    validated.append(cleaned)
            if dropped and withheld is not None:
                withheld.append(
                    (f"{dropped} of {len(items)} {field_name} entries", _OVER_URL_BUDGET)
                )
            payload[field_name] = validated

        if not root:
            return
        try:
            addresses = list(payload_addresses(payload))
        except UnvettableRequest as exc:
            raise VideoGenerationError(str(exc)) from exc
        for url, where in addresses:
            if not _spend(url):
                raise VideoGenerationError(
                    f"Refusing to send '{where}' unchecked: a request is checked against "
                    f"at most {_MAX_PASSTHROUGH_URLS} addresses and this one is past that."
                )
            await _check(url, where)

    @staticmethod
    def _parse_pixel_size(value: Any) -> tuple[int, int] | None:
        parts = _clean_str(value).lower().replace("\u00d7", "x").split("x")
        if len(parts) != 2:
            return None
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        return (width, height) if width > 0 and height > 0 else None

    @staticmethod
    def _aspect_ratio_value(value: Any) -> float | None:
        parts = _clean_str(value).split(":")
        if len(parts) != 2:
            return None
        try:
            width, height = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        return width / height if width > 0 and height > 0 else None

    @staticmethod
    def _tier_is_the_only_one_published(video_model: Any, resolution: str) -> bool:
        published = video_model.get("supported_resolutions") if isinstance(video_model, dict) else None
        if not isinstance(published, list) or len(published) != 1:
            return False
        return _clean_str(published[0]) == resolution

    def _apply_size_consistency(
        self,
        payload: dict[str, Any],
        video_model: Any,
        withheld: list[tuple[str, str]] | None,
    ) -> None:
        size = payload.get("size")
        if not size:
            return
        pixels = self._parse_pixel_size(size)
        resolution = _clean_str(payload.get("resolution"))
        if pixels is None:
            if resolution and resolution != _clean_str(size):
                payload.pop("resolution", None)
                if withheld is not None:
                    withheld.append(("resolution", _SIZE_IS_A_TIER))
            return
        if resolution and not self._tier_is_the_only_one_published(video_model, resolution):
            payload.pop("resolution", None)
            if withheld is not None:
                withheld.append(("resolution", _SIZE_FIXES_THE_PIXELS))
        declared = self._aspect_ratio_value(payload.get("aspect_ratio"))
        if declared is None:
            return
        width, height = pixels
        if abs(width / height - declared) / declared > _ASPECT_RATIO_TOLERANCE:
            payload.pop("aspect_ratio", None)
            if withheld is not None:
                withheld.append(("aspect_ratio", _SIZE_CONTRADICTS_THE_RATIO))

    @staticmethod
    def _normalise_provider_options(provider_options: dict[str, Any]) -> dict[str, Any]:
        normalised: dict[str, Any] = {}
        for slug, payload in provider_options.items():
            if not isinstance(slug, str) or not slug.strip() or not isinstance(payload, dict):
                continue
            normalised[slug.strip()] = dict(payload)
        return normalised

    async def _encode_frame_images(
        self,
        video_meta: dict[str, Any],
        video_model: Any,
        valves: Any,
        *,
        user_obj: Any = None,
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
        encoded: list[dict[str, Any]] = []
        total_bytes = 0
        seen_frame_types: set[str] = set()
        seen_file_ids: set[str] = set()

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
            if file_id in seen_file_ids:
                continue
            file_obj = await get_file_by_id(file_id, self._pipe.logger)
            if not file_obj:
                raise VideoGenerationError(f"Frame image '{file_id}' could not be loaded from Open WebUI storage.")
            mime = infer_file_mime_type(file_obj)
            mime = _clean_str(mime).split(";", 1)[0].lower()
            if not _promotable_as_frame({"content_type": mime}, valves):
                raise VideoGenerationError(f"Frame image MIME '{mime or 'unknown'}' is not allowed.")
            try:
                b64 = await self._pipe._file_gateway.read_file_record_base64(
                    file_obj, chunk_size, max_bytes, user=user_obj,
                )
            except RequiredInternalFileError as exc:
                if exc.kind == "size":
                    raise VideoGenerationError(
                        f"Frame image '{file_id}' is larger than the "
                        f"{megabytes(max_bytes)} this deployment sends per frame."
                    ) from exc
                raise VideoGenerationError(exc.user_message) from exc
            except ValueError as exc:
                if "File exceeds BASE64_MAX_SIZE_MB limit" not in str(exc):
                    raise
                raise VideoGenerationError(
                    f"Frame image '{file_id}' is larger than the "
                    f"{megabytes(max_bytes)} this deployment sends per frame."
                ) from exc
            if not b64:
                raise VideoGenerationError(f"Frame image '{file_id}' could not be encoded.")
            try:
                decoded_len = await _decoded_length(b64)
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
            seen_file_ids.add(file_id)
        return encoded

    async def _encode_input_references(
        self,
        video_meta: dict[str, Any],
        valves: Any,
        *,
        withheld: list[tuple[str, str]] | None = None,
        user_obj: Any = None,
        video_model: Any = None,
        relayed: set[tuple[str, str]] | None = None,
        companions: bool = False,
        event_emitter: Any = None,
        vetted: dict[str, bool] | None = None,
    ) -> list[dict[str, Any]]:
        raw = resolve_reference_items(video_meta)
        if not raw:
            return []
        image_max = int(valves.VIDEO_FRAME_IMAGE_MAX_BYTES)
        asset_max = int(getattr(valves, "REMOTE_VIDEO_MAX_SIZE_MB", 500)) * 1024 * 1024
        total_max = int(valves.VIDEO_FRAME_TOTAL_MAX_BYTES)
        relay_max = int(getattr(valves, "MEDIA_FILE_HOST_MAX_SIZE_MB", 200)) * 1024 * 1024
        chunk_size = int(getattr(valves, "IMAGE_UPLOAD_CHUNK_BYTES", 1024 * 1024))
        allowed_images = _csv_set(valves.VIDEO_FRAME_IMAGE_MIME_ALLOWLIST)
        accepted: list[_AcceptedReference] = []
        model_takes = _declared_input_kinds(video_model)
        blob_floor = _INPUT_PIXEL_FLOORS.get(_clean_str(video_meta.get("model_id")))
        encoded: list[dict[str, Any]] = []
        total_bytes = 0
        relay_bytes = 0
        past_the_count = 0

        _over_reference_single = (
            f"it is larger than the {megabytes(image_max)} one reference may be"
        )

        def _skip(name: str, cause: str, text: str) -> None:
            self.logger.log(
                warn_level(_warned_dropped_video_param, f"input_reference:{cause}"),
                "Reference file %r was not sent with the video request: %s", name, text,
            )
            if withheld is not None:
                withheld.append((name, text))

        for item in raw:
            if not isinstance(item, dict):
                continue
            file_id = _clean_str(item.get("id"))
            if not file_id:
                continue
            if len(accepted) >= _MAX_INPUT_REFERENCES:
                past_the_count += 1
                continue
            file_obj = await get_file_by_id(file_id, self._pipe.logger)
            if not file_obj:
                _skip(file_id, "unreadable", "it could not be loaded from Open WebUI storage")
                continue
            stored_mime = usable_media_type(infer_file_mime_type(file_obj))
            mime = stored_mime or usable_media_type(item.get("content_type"))
            family = mime.split("/", 1)[0]
            kind = _REFERENCE_KINDS.get(family)
            if not kind:
                _skip(file_id, "untyped", _UNTYPED_REFERENCE)
                continue
            if family == "image" and mime not in allowed_images:
                _skip(
                    file_id,
                    "allowlist",
                    f"the type {mime!r} is not on the reference image allowlist",
                )
                continue
            if not model_takes(family):
                _skip(file_id, "refused-kind", _reference_kind_refused(family))
                continue
            via_file_host = self._file_host_wanted(valves, family)
            if kind in _REFERENCE_KINDS_NEEDING_A_LINK and not via_file_host:
                _skip(file_id, "needs-a-link", _REFERENCE_NEEDS_A_LINK)
                continue
            if via_file_host and not stored_mime:
                _skip(file_id, "unstored-type", _PUBLISHING_NEEDS_A_STORED_TYPE)
                continue
            if via_file_host and not authorize_file_publication(file_obj, user_obj):
                _skip(file_id, "not-owner", PUBLISHING_NEEDS_OWNERSHIP)
                continue
            if family == "image" and via_file_host:
                declared = declared_file_size(file_obj)
                if declared and declared > image_max:
                    _skip(file_id, "over-single", _over_reference_single)
                    continue
            if via_file_host:
                self._refuse_over_the_relay_cap(
                    declared_file_size(file_obj), relay_bytes, relay_max
                )
            try:
                b64 = await self._pipe._file_gateway.read_file_record_base64(
                    file_obj,
                    chunk_size,
                    relay_max if via_file_host else (image_max if family == "image" else asset_max),
                    user=user_obj,
                )
            except RequiredInternalFileError as exc:
                if family == "image" and getattr(exc, "kind", None) == "size":
                    _skip(file_id, "over-single", _over_reference_single)
                    continue
                _skip(file_id, "not-allowed", exc.user_message)
                continue
            except InlineFileTooLargeError:
                if family == "image":
                    _skip(file_id, "over-single", _over_reference_single)
                    continue
                raise
            except ValueError as exc:
                if not via_file_host:
                    raise
                raise VideoGenerationError(
                    f"The attached {family} is larger than the "
                    f"{megabytes(relay_max)} this deployment sends to a file host."
                ) from exc
            if not b64:
                _skip(file_id, "unencodable", "it could not be encoded")
                continue
            try:
                blob = await _decoded_payload(b64)
            except Exception as exc:
                self.logger.debug(
                    "input_references asset %s is not valid base64: %s", file_id, exc,
                    exc_info=True,
                )
                _skip(file_id, "not-base64", "it contains invalid base64 data")
                continue
            decoded_len = len(blob)
            if family == "image":
                if decoded_len > image_max:
                    _skip(file_id, "over-single", _over_reference_single)
                    continue
                if total_bytes + decoded_len > total_max:
                    _skip(file_id, "over-budget", _OVER_REFERENCE_BUDGET)
                    continue
                total_bytes += decoded_len
            if via_file_host:
                self._refuse_over_the_relay_cap(decoded_len, relay_bytes, relay_max)
                if family == "video":
                    note = await self._clip_too_small_note(blob_floor, blob, mime)
                    if note:
                        _skip(file_id, "clip-size", note)
                        continue
                relay_bytes += decoded_len
            accepted.append(
                _AcceptedReference(
                    file_id=file_id,
                    kind=kind,
                    family=family,
                    mime=mime,
                    b64=b64,
                    via_file_host=via_file_host,
                    filename=_clean_str(getattr(file_obj, "filename", "")),
                    blob=blob,
                )
            )
        if past_the_count:
            _skip(
                f"{past_the_count} further attachment(s)", "over-count", _OVER_REFERENCE_COUNT
            )
        if accepted and not companions and all(
            entry.kind == "audio_url" for entry in accepted
        ):
            for entry in accepted:
                _skip(entry.file_id, "audio-alone", _AUDIO_NEEDS_A_COMPANION)
            return []
        disclosed = await self._disclose_the_file_host(
            valves,
            {entry.family for entry in accepted if entry.via_file_host},
            event_emitter,
        )
        used: set[tuple[str, str]] = set()
        relay_deadline = time.monotonic() + MAX_RELAY_SECONDS_PER_REQUEST
        for entry in accepted:
            if entry.via_file_host:
                link, host = await self._relay_reference(
                    valves, entry.blob, filename=entry.filename,
                    mime=entry.mime, family=entry.family,
                    deadline=relay_deadline,
                )
                encoded.append({"type": entry.kind, entry.kind: {"url": link}})
                if vetted is not None:
                    vetted[link] = True
                used.add((entry.family, host))
                if relayed is not None:
                    relayed.add((entry.family, host))
                continue
            encoded.append(
                {
                    "type": entry.kind,
                    entry.kind: {"url": f"data:{entry.mime};base64,{entry.b64}"},
                }
            )
        if used and used != disclosed:
            await self._emit_file_host_notice(valves, used, event_emitter)
        return encoded

    @staticmethod
    def _refuse_over_the_relay_cap(size: int | None, already: int, cap: int) -> None:
        if cap <= 0 or size is None or size <= 0:
            return
        if size > cap:
            raise VideoGenerationError(
                f"The attachment is {megabytes(size)} and the limit for sending media "
                f"to a file host is {megabytes(cap)}."
            )
        if already + size > cap:
            raise VideoGenerationError(
                f"The attachments come to {megabytes(already + size)} and one request "
                f"sends at most {megabytes(cap)} to a file host."
            )

    @staticmethod
    def _relay_hosts(valves: Any) -> list[str]:
        chosen = str(getattr(valves, "MEDIA_FILE_HOST", "litterbox"))
        if not bool(getattr(valves, "USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN", False)):
            return [chosen]
        return [chosen] + [name for name in RELAY_HOSTS if name != chosen]

    async def _disclose_the_file_host(
        self, valves: Any, families: set[str], event_emitter: Any
    ) -> set[tuple[str, str]]:
        if not families:
            return set()
        planned = {
            (family, host)
            for family in families
            for host in self._relay_hosts(valves)
        }
        said, cause = await self._emit_file_host_notice(valves, planned, event_emitter)
        if not said:
            self.logger.warning(
                "Could not tell this chat the upload is about to happen: %s. Set "
                "TELL_USERS_ABOUT_THE_FILE_HOST off if you warn your users another way, "
                "or give FILE_HOST_NOTICE a sentence to send.",
                cause,
            )
            raise VideoGenerationError(_COULD_NOT_SAY_IT_FIRST.format(cause=cause))
        return planned

    async def _emit_file_host_notice(
        self, valves: Any, pairs: set[tuple[str, str]], event_emitter: Any
    ) -> tuple[bool, str]:
        if not bool(getattr(valves, "TELL_USERS_ABOUT_THE_FILE_HOST", True)):
            return True, ""
        if event_emitter is None:
            return False, _NO_EMITTER_CAUSE
        notice = self._file_host_notice(valves, pairs)
        if not notice.strip():
            return False, _BLANK_NOTICE_CAUSE
        delivered = bool(
            await self._pipe._event_emitter_handler._emit_notification(
                event_emitter, notice, level="info"
            )
        )
        return delivered, "" if delivered else _NOTICE_REFUSED_CAUSE

    @staticmethod
    def _relay_hosts_named(relayed: set[tuple[str, str]]) -> list[str]:
        return sorted({used for _family, used in relayed})

    @staticmethod
    def _relay_kinds_named(relayed: set[tuple[str, str]]) -> list[str]:
        kinds = {"video": "clip", "audio": "sound file", "image": "picture"}
        return sorted({kinds.get(family, family) for family, _used in relayed})

    @classmethod
    def _relay_kinds_spoken(cls, relayed: set[tuple[str, str]]) -> str:
        spoken = cls._relay_kinds_named(relayed)
        if not spoken:
            return ""
        return spoken[0] if len(spoken) == 1 else " and ".join(
            [", ".join(spoken[:-1]), spoken[-1]]
        )

    @staticmethod
    def _relay_retention_words(
        valves: Any, hosts: list[str], *, plural: bool = False
    ) -> str:
        host = hosts[0] if hosts else str(getattr(valves, "MEDIA_FILE_HOST", "litterbox"))
        if any(host_keeps_forever(used) for used in hosts) or (
            not hosts and host_keeps_forever(host)
        ):
            if plural:
                return (
                    "and stay there for good, because the uploads carry no account and "
                    "nothing here can take them down again"
                )
            return (
                "and stays there for good, because the upload carries no account and "
                "nothing here can take it down again"
            )
        spans = {"1h": "an hour", "12h": "12 hours", "24h": "a day", "72h": "three days"}
        span = str(getattr(valves, "MEDIA_FILE_HOST_RETENTION", "1h"))
        deleted = "are deleted" if plural else "is deleted"
        return f"and {deleted} again {spans.get(span, span)} later"

    @classmethod
    def _file_host_notice(cls, valves: Any, relayed: set[tuple[str, str]]) -> str:
        template = str(getattr(valves, "FILE_HOST_NOTICE", "") or "")
        hosts = cls._relay_hosts_named(relayed)
        host = hosts[0] if hosts else str(getattr(valves, "MEDIA_FILE_HOST", "litterbox"))
        try:
            return template.format(
                kind=cls._relay_kinds_spoken(relayed),
                host=" and ".join(hosts) or host,
                retention=cls._relay_retention_words(valves, hosts),
            )
        except (KeyError, IndexError, ValueError):
            return template

    @classmethod
    def _with_the_file_host_record(
        cls, block: str, valves: Any, relayed: set[tuple[str, str]]
    ) -> str:
        record = cls._file_host_record(valves, relayed)
        if not record:
            return block
        return f"{block}{record}" if block else record

    @classmethod
    def _file_host_record(cls, valves: Any, relayed: set[tuple[str, str]]) -> str:
        if not relayed:
            return ""
        hosts = cls._relay_hosts_named(relayed)
        many = len(cls._relay_kinds_named(relayed)) > 1
        said = _FILE_HOST_RECORD.format(
            kind=cls._relay_kinds_spoken(relayed),
            was=("were" if many else "was"),
            it=("them" if many else "it"),
            host=" and ".join(hosts),
            retention=cls._relay_retention_words(valves, hosts, plural=many),
        )
        return (
            f"{_serialize_kind_marker(RELAY_BLOCK_START, '1')}\n"
            f"\n{said}\n"
            f"{_serialize_kind_marker(RELAY_BLOCK_END, '1')}\n"
        )

    @staticmethod
    def _recover_the_file_host_record(persisted: str) -> str:
        if not isinstance(persisted, str) or not persisted:
            return ""
        found = _RELAY_BLOCK_REGION_RE.search(persisted)
        return found.group(0) if found else ""

    @classmethod
    def _with_the_withheld_record(
        cls, block: str, withheld: list[tuple[str, str]]
    ) -> str:
        record = cls._withheld_record(withheld)
        if not record:
            return block
        return f"{block}{record}" if block else record

    @classmethod
    def _withheld_record(cls, withheld: list[tuple[str, str]]) -> str:
        if not withheld:
            return ""
        return (
            f"{_serialize_kind_marker(WITHHELD_BLOCK_START, '1')}\n"
            f"\n{_WITHHELD_RECORD.format(items=cls._withheld_notice(withheld))}\n"
            f"{_serialize_kind_marker(WITHHELD_BLOCK_END, '1')}\n"
        )

    @staticmethod
    def _recover_the_withheld_record(persisted: str) -> str:
        if not isinstance(persisted, str) or not persisted:
            return ""
        found = _WITHHELD_BLOCK_REGION_RE.search(persisted)
        return found.group(0) if found else ""

    @staticmethod
    def _file_host_wanted(valves: Any, family: str) -> bool:
        if not bool(getattr(valves, "SEND_MEDIA_VIA_FILE_HOST", False)):
            return False
        per_kind = {
            "video": "SEND_VIDEO_VIA_FILE_HOST",
            "audio": "SEND_AUDIO_VIA_FILE_HOST",
            "image": "SEND_IMAGES_VIA_FILE_HOST",
        }.get(family)
        return bool(per_kind and getattr(valves, per_kind, False))

    async def _clip_too_small_note(
        self, floor: _InputPixelFloor | None, payload: str | bytes, mime: str
    ) -> str:
        if floor is None:
            return ""
        if isinstance(payload, (bytes, bytearray)):
            blob = bytes(payload)
        else:
            try:
                blob = await asyncio.to_thread(base64.b64decode, payload, validate=False)
            except (binascii.Error, ValueError):
                return ""
        suffix = extension_for_video_mime(mime) or ".mp4"
        handle, path = tempfile.mkstemp(suffix=suffix, prefix="openrouter-clip-")
        temp = Path(path)
        try:
            await asyncio.to_thread(_write_and_close, handle, blob)
            meta = await probe_video(temp)
        except FrameExtractionError:
            return ""
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, RuntimeError):
            return ""
        finally:
            with contextlib.suppress(OSError):
                temp.unlink()
        pixels = int(meta.width) * int(meta.height)
        if pixels <= 0 or pixels >= floor.pixels:
            return ""
        return (
            f"it is {meta.width} by {meta.height}, which is {pixels:,} pixels a frame, and "
            f"anything under {floor.pixels:,} is left out here: {floor.whose_rule}"
        )

    async def _relay_reference(
        self, valves: Any, payload: str | bytes, *, filename: str, mime: str,
        family: str, deadline: float,
    ) -> tuple[str, str]:
        if isinstance(payload, (bytes, bytearray)):
            blob = bytes(payload)
        else:
            try:
                blob = await asyncio.to_thread(base64.b64decode, payload, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise VideoGenerationError(
                    f"The attached {family} could not be read, so it was not sent."
                ) from exc
        hosts = self._relay_hosts(valves)
        failures: list[str] = []
        async with self._pipe._create_http_session(valves) as http:
            for host in hosts:
                try:
                    link = await relay_to_public_url(
                        http,
                        blob,
                        filename=filename or f"reference.{mime.split('/', 1)[-1]}",
                        mime=mime,
                        host=host,
                        retention=str(getattr(valves, "MEDIA_FILE_HOST_RETENTION", "1h")),
                        max_bytes=int(getattr(valves, "MEDIA_FILE_HOST_MAX_SIZE_MB", 200))
                        * 1024
                        * 1024,
                        seconds_left=deadline - time.monotonic(),
                    )
                    return link, host
                except MediaRelayError as exc:
                    failures.append(str(exc))
                    self.logger.warning(
                        "Could not put the attached %s behind a link via %s: %s",
                        family, host, exc,
                    )
                    if getattr(exc, "may_have_stored_it", True):
                        failures.append(
                            _A_COPY_MAY_ALREADY_BE_THERE.format(
                                host=host,
                                second=", so no second host was tried"
                                if host != hosts[-1] else "",
                            )
                        )
                        break
        raise VideoGenerationError(
            f"The attached {family} could not be sent: {'; '.join(failures)}."
        )

    def _extract_video_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        pipe_meta = metadata.get(_PIPE_METADATA_KEY) if isinstance(metadata, dict) else None
        if not isinstance(pipe_meta, dict):
            return {}
        video_meta = pipe_meta.get("video_generation")
        return dict(video_meta) if isinstance(video_meta, dict) else {}

    def _intent_classifier_should_run(
        self,
        *,
        valves: Any,
        persisted_content: str,
        prompt: str,
        body: dict[str, Any],
        video_meta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        chat_id: str = "",
        user_id: str = "",
    ) -> bool:
        """Apply short-circuit conditions for the intent classifier.

        """
        if not bool(_admin_intent_floor(valves, "VIDEO_INTENT_ENABLED", True)):
            return False
        if not bool(resolve_intent_user_setting(
            metadata, "enabled", valves, "VIDEO_INTENT_ENABLED", True,
        )):
            return False
        if not (prompt or "").strip():
            return False
        if asks_for_help(self._extract_user_prompt(body)):
            return False
        if persisted_content and self._extract_video_job_marker(persisted_content):
            return False
        if getattr(valves, "VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT", True):
            messages = body.get("messages") if isinstance(body, dict) else None
            if isinstance(messages, list) and len([
                m for m in messages
                if isinstance(m, dict) and (m.get("role") or "").lower() not in ("system", "developer")
            ]) <= 1:
                has_attachments = bool(collect_attachments_from_video_meta(video_meta))
                if not has_attachments:
                    return False
        cap_chat, cap_day = _intent_counters_enabled(valves)
        if self._intent_breaker_open(user_id):
            return False
        charge_chat = cap_chat > 0 and bool(chat_id)
        tally_key = _hash_chat_id(chat_id) if charge_chat else ""
        if charge_chat and self._intent_call_counts_per_chat.get(tally_key, 0) >= cap_chat:
            return False
        charge_day = cap_day > 0 and bool(user_id)
        day = ""
        if charge_day:
            from datetime import datetime
            day = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            if self._intent_call_counts_per_user_day.get((user_id, day), 0) >= cap_day:
                return False
        if charge_chat:
            per_chat = self._intent_call_counts_per_chat
            per_chat[tally_key] = per_chat.get(tally_key, 0) + 1
            per_chat.move_to_end(tally_key)
            while len(per_chat) > _INTENT_CHAT_COUNT_WINDOW:
                per_chat.popitem(last=False)
        if charge_day:
            if self._intent_pruned_day != day:
                self._intent_call_counts_per_user_day = {
                    key: count for key, count in self._intent_call_counts_per_user_day.items()
                    if key[1] == day
                }
                self._intent_pruned_day = day
            self._intent_call_counts_per_user_day[(user_id, day)] = (
                self._intent_call_counts_per_user_day.get((user_id, day), 0) + 1
            )
        return True

    def _intent_record_call(self, chat_id: str, user_id: str, *, valves: Any) -> None:
        return

    def _intent_breaker_open(self, user_id: str) -> bool:
        now = time.time()
        if now - self._intent_breaker_swept_at >= _INTENT_BREAKER_SWEEP_INTERVAL_SECONDS:
            self._intent_breaker_swept_at = now
            for expired in [k for k, until in self._intent_breaker_until_ts.items() if until <= now]:
                self._intent_breaker_until_ts.pop(expired, None)
        key = user_id or ""
        until = self._intent_breaker_until_ts.get(key, 0.0)
        if now < until:
            return True
        self._intent_breaker_until_ts.pop(key, None)
        return False

    def _intent_record_failure(self, user_id: str = "") -> None:
        """Open the in-process circuit breaker for 60 seconds after auth/quota errors."""
        now = time.time()
        key = user_id or ""
        self._intent_breaker_until_ts[key] = max(
            self._intent_breaker_until_ts.get(key, 0.0),
            now + _INTENT_BREAKER_SWEEP_INTERVAL_SECONDS,
        )

    def _intent_latch_key(self, chat_id: Any) -> str:
        if not isinstance(chat_id, str) or not chat_id:
            return _INTENT_NO_CHAT_ID_KEY
        if is_temporary_chat(chat_id):
            return ""
        return chat_id

    def _intent_notice_key(self, chat_id: Any) -> str:
        key = self._intent_latch_key(chat_id)
        return "" if key == _INTENT_NO_CHAT_ID_KEY else key

    def _intent_was_failure_notified(self, chat_key: str) -> bool:
        if not chat_key or is_temporary_chat(chat_key):
            return False
        return chat_key in self._intent_failure_notified_chats

    def _intent_note_failure_notified(self, chat_key: str) -> None:
        if not chat_key or is_temporary_chat(chat_key):
            return
        notified = self._intent_failure_notified_chats
        notified[chat_key] = None
        notified.move_to_end(chat_key)
        while len(notified) > _INTENT_NOTIFIED_WINDOW:
            notified.popitem(last=False)

    def _emit_intent_telemetry(
        self,
        intent: VideoIntentResult,
        *,
        valves: Any,
        chat_id: Any,
    ) -> None:
        """Emit the structured video_intent telemetry line. Called once per
        terminal path (clarify / modify-fallback / classifier_failed-or-success)
        AFTER materialise has run, so `frames_extracted` reads accurate.

        Wrapped in suppress so a telemetry hiccup never breaks the user's video
        request.
        """
        try:
            emit_telemetry_log(
                intent,
                logger=self.logger,
                chat_id=chat_id if isinstance(chat_id, str) else "",
                log_decisions_enabled=bool(
                    getattr(valves, "VIDEO_INTENT_LOG_DECISIONS", False)
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.logger.debug("emit_telemetry_log raised (suppressed): %s", exc, exc_info=True)

    def _apply_uploaded_attachment_retargeting(
        self,
        intent: VideoIntentResult,
        video_meta: dict[str, Any],
        valves: Any = None,
    ) -> None:
        if not intent.frame_plan:
            return
        frame_images = video_meta.get("frame_images")
        if not isinstance(frame_images, list):
            return
        reference_list = resolve_reference_items(video_meta)
        ordered_attachments: list[dict[str, Any]] = [
            item for item in (
                [f for f in frame_images if isinstance(f, dict)]
                + [r for r in reference_list if isinstance(r, dict)]
            )
        ]
        ordered_attachments.sort(key=lambda entry: _attachment_position(entry))
        retargeted = 0
        moved: list[dict[str, Any]] = []
        demoted: list[dict[str, Any]] = []
        for entry in intent.frame_plan:
            if entry.source != "uploaded_attachment":
                continue
            idx = entry.source_index
            if not isinstance(idx, int) or idx < 0 or idx >= len(ordered_attachments):
                intent.downgrades.append(
                    f"retarget_skipped_invalid_index_{idx}"
                )
                continue
            target = ordered_attachments[idx]
            if not isinstance(target, dict):
                continue
            if entry.target == "input_reference":
                if target in frame_images:
                    moved.append(target)
                continue
            if target not in frame_images:
                if not _promotable_as_frame(target, valves):
                    intent.downgrades.append(f"retarget_skipped_non_image_frame_{idx}")
                    continue
                displaced = next(
                    (
                        other
                        for other in frame_images
                        if isinstance(other, dict)
                        and other.get("frame_type") == entry.target
                        and other is not target
                    ),
                    None,
                )
                if displaced is not None:
                    frame_images[frame_images.index(displaced)] = target
                    demoted.append(displaced)
                else:
                    frame_images.append(target)
                claimed = video_meta.get("input_references")
                if isinstance(claimed, list) and target in claimed:
                    claimed.remove(target)
            existing_frame_type = target.get("frame_type")
            if existing_frame_type != entry.target:
                target["frame_type"] = entry.target
                retargeted += 1
        if moved or demoted:
            references = video_meta.setdefault(_reference_write_key(video_meta), [])
            if not isinstance(references, list):
                references = []
                video_meta["input_references"] = references
            ordered: list[dict[str, Any]] = []
            for item in list(demoted) + list(moved):
                if not any(item is seen for seen in ordered):
                    ordered.append(item)
            for item in ordered:
                while item in frame_images:
                    frame_images.remove(item)
            picked: list[dict[str, Any]] = list(ordered)
            for item in picked:
                if not isinstance(item, dict):
                    continue
                references.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "content_type": item.get("content_type"),
                    "size": item.get("size"),
                    "attachment_index": item.get("attachment_index"),
                })
                retargeted += 1
        if retargeted:
            intent.frames_retargeted += retargeted
            self.logger.debug(
                "_apply_uploaded_attachment_retargeting: applied %d classifier "
                "frame instruction(s), %d moved to input_references",
                retargeted, len({id(x) for x in moved}),
            )

    async def _materialise_frame_plan(
        self,
        *,
        intent: VideoIntentResult,
        video_meta: dict[str, Any],
        request: Any,
        user_obj: Any,
        chat_id: str,
        message_id: str,
        reused_frame_index: Literal["first", "last"] = "last",
    ) -> list[str]:
        """For each prior_video_* entry in frame_plan, extract the frame from
        the prior video file, upload it as a new OWUI image, and inject into
        video_meta["frame_images"] so the existing _encode_frame_images path
        wires it into the /videos call.

        Any per-entry failure is logged and the entry is dropped (downgrade
        noted in intent.downgrades). Returns a list of thumbnail URLs (one
        per resolved frame) for the disclosure block — empty strings for
        entries that failed.
        """
        thumb_urls: list[str] = []
        if not intent.frame_plan:
            return thumb_urls

        if is_temporary_chat(chat_id):
            intent.downgrades.append("frame_plan_dropped_temporary_chat")
            return ["" for _entry in intent.frame_plan]

        materialised: dict[str, Path] = {}
        try:
            for position, entry in enumerate(intent.frame_plan):
                if entry.source == "uploaded_attachment":
                    thumb_urls.append("")
                    continue
                if not entry.source.startswith("prior_video_"):
                    thumb_urls.append("")
                    continue

                try:
                    file_id = await self._resolve_prior_video_file_id(
                        entry, intent=intent, user_obj=user_obj,
                    )
                    if not file_id:
                        intent.downgrades.append(
                            f"prior_video_index_{entry.source_index}_unresolvable_at_{position}"
                        )
                        thumb_urls.append("")
                        continue

                    tmp_path = materialised.get(file_id)
                    if tmp_path is None:
                        tmp_path = await self._resolve_owui_file_path(
                            file_id=file_id, request=request, user_obj=user_obj,
                        )
                        if tmp_path is not None:
                            materialised[file_id] = tmp_path
                    if tmp_path is None:
                        intent.downgrades.append(
                            f"prior_video_download_failed_idx_{entry.source_index}_at_{position}"
                        )
                        thumb_urls.append("")
                        continue

                    try:
                        if entry.source == "prior_video_first_frame":
                            target = "first_frame"
                            ts = None
                        elif entry.source == "prior_video_last_frame":
                            target = "last_frame"
                            ts = None
                        else:
                            target = "at_timestamp"
                            ts = entry.timestamp_seconds

                        frame = await extract_frame(
                            tmp_path, target=target, timestamp_seconds=ts,
                            fallback_to_last_on_overshoot=True,
                            reused_frame_index=reused_frame_index,
                            logger=self.logger,
                        )
                        if frame.downgrade_note:
                            intent.downgrades.append(frame.downgrade_note)
                    except FrameExtractionError as exc:
                        if (
                            entry.source == "prior_video_first_frame"
                            and getattr(exc, "pixel_cap", False)
                        ):
                            try:
                                frame = await extract_frame(
                                    tmp_path, target="at_timestamp", timestamp_seconds=0.0,
                                    fallback_to_last_on_overshoot=True,
                                    reused_frame_index="last",
                                    logger=self.logger,
                                )
                            except FrameExtractionError:
                                raise exc
                            if frame.downgrade_note:
                                intent.downgrades.append(frame.downgrade_note)
                        else:
                            self.logger.warning(
                                "frame extraction failed for entry %s: %s",
                                entry.source_index, exc,
                            )
                            intent.downgrades.append(
                                f"frame_extract_failed_idx_{entry.source_index}_at_{position}"
                            )
                            thumb_urls.append("")
                            continue

                    frame_file_id = await self._pipe._file_gateway.upload_to_owui_storage(
                        request=request,
                        user=user_obj,
                        file_data=frame.image_bytes,
                        filename=f"intent-frame-{entry.source}-{entry.source_index}.png",
                        mime_type="image/png",
                        chat_id=chat_id or None,
                        message_id=message_id or None,
                    )
                    if not frame_file_id:
                        intent.downgrades.append(
                            f"frame_upload_failed_idx_{entry.source_index}_at_{position}"
                        )
                        thumb_urls.append("")
                        continue

                    intent.frames_extracted += 1

                    if entry.target in ("first_frame", "last_frame"):
                        extracted = getattr(frame, "resolved_target", entry.target)
                        if extracted not in ("first_frame", "last_frame"):
                            extracted = entry.target
                        fi_list = video_meta.setdefault("frame_images", [])
                        if isinstance(fi_list, list):
                            fi_list.append({
                                "id": frame_file_id,
                                "frame_type": extracted,
                                "name": f"intent-frame-{extracted}.png",
                                "content_type": "image/png",
                            })
                    elif entry.target == "input_reference":
                        ir_list = video_meta.setdefault(_reference_write_key(video_meta), [])
                        if isinstance(ir_list, list):
                            ir_list.append({
                                "id": frame_file_id,
                                "name": "intent-frame-input_reference.png",
                                "content_type": "image/png",
                            })

                    try:
                        thumb = await asyncio.to_thread(make_thumbnail, frame.image_bytes)
                        thumb_file_id = await self._pipe._file_gateway.upload_to_owui_storage(
                            request=request,
                            user=user_obj,
                            file_data=thumb.image_bytes,
                            filename=f"intent-thumb-{entry.source}-{entry.source_index}.jpg",
                            mime_type="image/jpeg",
                            chat_id=chat_id or None,
                            message_id=message_id or None,
                        )
                        if thumb_file_id:
                            thumb_urls.append(f"/api/v1/files/{thumb_file_id}/content")
                        else:
                            intent.downgrades.append(
                                f"thumbnail_upload_failed_idx_{entry.source_index}_at_{position}"
                            )
                            thumb_urls.append("")
                    except Exception as exc:
                        self.logger.warning(
                            "thumbnail generation failed for entry %s: %s",
                            entry.source_index, exc, exc_info=True,
                        )
                        intent.downgrades.append(
                            f"thumbnail_generation_failed_idx_{entry.source_index}_at_{position}"
                        )
                        thumb_urls.append("")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.logger.warning(
                        "_materialise_frame_plan entry failed (degrade-open): %s", exc, exc_info=True
                    )
                    intent.downgrades.append("materialise_failed")
                    thumb_urls.append("")
        finally:
            for path in materialised.values():
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    self.logger.warning(
                        "prior-video temp unlink failed for %s: %s", path, exc,
                    )

        return thumb_urls

    async def _resolve_prior_video_file_id(
        self,
        entry: FramePlanEntry,
        *,
        intent: VideoIntentResult,
        user_obj: Any,
    ) -> str:
        del user_obj
        idx = entry.source_index
        if not isinstance(idx, int) or not intent.prior_videos:
            return ""
        if idx < 0 or idx >= len(intent.prior_videos):
            return ""
        prior = intent.prior_videos[idx]
        url = str(prior.get("file_url") or "")
        clean = url.split("?", 1)[0].split("#", 1)[0]
        marker = "/api/v1/files/"
        start = clean.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        tail = clean[start:]
        if not tail or tail.startswith("/"):
            return ""
        candidate = tail.split("/", 1)[0]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", candidate):
            return ""
        return candidate

    async def _resolve_owui_file_path(
        self,
        *,
        file_id: str,
        request: Any,
        user_obj: Any,
    ) -> Path | None:
        """Materialise a prior-video file to a private temp for best-effort frame extraction.

        Authorises and routes the read through the OWUI Storage gateway (any
        backend). Returns a caller-owned temp Path on success; returns None on
        auth denial, size, containment, or any other failure so the caller can
        degrade with a downgrade note. The caller must unlink the returned temp.
        """
        del request
        try:
            file_obj = await get_file_by_id(file_id, self._pipe.logger)
            if file_obj is None:
                return None
            max_bytes = int(self._pipe.valves.VIDEO_MAX_SIZE_MB) * 1024 * 1024
            try:
                temp = await materialize_owui_file_to_temp(
                    file_obj,
                    user=user_obj,
                    logger=self._pipe.logger,
                    max_bytes=max_bytes,
                    allow_unknown_size=bool(
                        getattr(self._pipe.valves, "ALLOW_UNKNOWN_SIZE_CLOUD_READS", False)
                    ),
                    allowed_suffixes={".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"},
                )
            except RequiredInternalFileError as exc:
                self.logger.warning(
                    "video_intent: prior-video file unavailable (file_id=%s): %s",
                    file_id, exc.user_message,
                )
                return None
            return temp
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "Could not read the referenced video file %s; the generation will "
                "proceed without it: %s",
                file_id,
                exc,
                exc_info=True,
            )
            return None

    def _extract_provider_options(self, response_provider: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        return requested_provider_options(
            SimpleNamespace(provider=response_provider), metadata
        )

    def _split_allowed_parameters(self, video_model: Any) -> tuple[set[str], set[str]]:
        passthrough = set()
        if isinstance(video_model, dict):
            raw = video_model.get("allowed_passthrough_parameters")
            if isinstance(raw, list):
                passthrough = {item for item in raw if isinstance(item, str) and item}
        top_level: set[str] = set()
        if isinstance(video_model, dict):
            if isinstance(video_model.get("supported_aspect_ratios"), list):
                top_level.add("aspect_ratio")
            if isinstance(video_model.get("supported_durations"), list):
                top_level.add("duration")
            if isinstance(video_model.get("supported_resolutions"), list):
                top_level.add("resolution")
            if isinstance(video_model.get("supported_sizes"), list) or isinstance(
                video_model.get("supported_size_options"), list
            ):
                top_level.add("size")
            if "seed" in video_model and not capability_declared_off(video_model.get("seed")):
                top_level.add("seed")
            if "generate_audio" in video_model and not capability_declared_off(
                video_model.get("generate_audio")
            ):
                top_level.add("generate_audio")
            if isinstance(video_model.get("upscale_factor"), dict):
                top_level.add("upscale_factor")
            if isinstance(video_model.get("creativity"), list):
                top_level.add("creativity")
        passthrough -= _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS
        return top_level, passthrough

    def _provider_slug_candidates(self, video_model: Any, api_model_id: str) -> list[str]:
        model_id = api_model_id
        if isinstance(video_model, dict):
            candidate = video_model.get("id")
            if isinstance(candidate, str) and candidate:
                model_id = candidate
        model_id = model_id.lstrip("~")

        candidates: list[str] = []
        catalog_manager = getattr(self._pipe, "_catalog_manager", None)
        getter = getattr(catalog_manager, "get_cached_provider_map", None)
        if callable(getter):
            cached = getter() or {}
            entry = cached.get(model_id) if isinstance(cached, dict) else None
            providers = entry.get("providers") if isinstance(entry, dict) else None
            if isinstance(providers, list):
                candidates.extend(
                    dict.fromkeys(key for p in providers if (key := options_key(p)))
                )

        if not candidates:
            self.logger.log(
                warn_level(_warned_provider_slug_guess, model_id),
                "No catalog provider slug for %r. The vendor prefix of a model id is not a "
                "provider slug -- google/veo is served by google-vertex, kwaivgi/kling by "
                "atlas-cloud -- so guessing one would key every provider parameter to a "
                "provider OpenRouter will never match, and it would drop them silently.",
                model_id,
            )
        return candidates

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
            "negative_prompt": ["negative_prompt", "negativePrompt"],
        }
        candidates = aliases.get(raw, [raw])
        if not allowed:
            return ""
        for candidate in candidates:
            if candidate in allowed:
                return candidate
        return ""

    @staticmethod
    def _read_prompt(body: dict[str, Any], read: Callable[[Any], str]) -> str:
        text = read(body.get("messages") if isinstance(body, dict) else None)
        if text:
            return text
        prompt = body.get("prompt") if isinstance(body, dict) else ""
        return prompt if isinstance(prompt, str) else ""

    def _extract_prompt(self, body: dict[str, Any]) -> str:
        return self._read_prompt(body, prompt_with_system)

    def _extract_user_prompt(self, body: dict[str, Any]) -> str:
        return self._read_prompt(body, latest_user_text)

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
        file_ids: list[str],
        elapsed: float,
        usage: dict[str, Any],
        attempted: int = 0,
        fetched: int = 0,
    ) -> str:
        clips = "".join(
            f"<video>\n/api/v1/files/{file_id}/content\n</video>\n\n" for file_id in file_ids
        )
        shortfall = ""
        missing = max(attempted - len(file_ids), 0)
        if missing > 0:
            shortfall = _shortfall_note(missing, attempted, fetched)
        return (
            f"{_serialize_kind_marker(self.JOB_MARKER_KIND, job_id)}\n"
            f"{_serialize_kind_marker(self.MODEL_MARKER_KIND, model_id)}\n\n"
            f"{clips}{shortfall}"
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

    def _build_pending_content(
        self, *, job_id: str, model_id: str, note: str = "Video generation is running..."
    ) -> str:
        return (
            f"{_serialize_kind_marker(self.JOB_MARKER_KIND, job_id)}\n"
            f"{_serialize_kind_marker(self.MODEL_MARKER_KIND, model_id)}\n\n"
            f"{note}"
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
                stream_duration=elapsed,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            self.logger.warning(
                "Could not render the final video usage status; falling back to a "
                "plain duration line",
                exc_info=True,
            )
            return f"Video generated in {elapsed:.1f}s"

    @staticmethod
    def _coerce_video_usage(raw: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            if key == "cost":
                if isinstance(value, str):
                    try:
                        out["cost"] = float(value)
                    except (TypeError, ValueError):
                        continue
                elif isinstance(value, (int, float)):
                    out["cost"] = value
            elif key in {"total_tokens", "input_tokens", "output_tokens"}:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[key] = int(value)
            else:
                out[key] = value
        return out

    async def _emit_status(
        self,
        emitter: EventEmitter | None,
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

    async def _emit_completion(
        self,
        emitter: EventEmitter | None,
        content: str,
        *,
        usage: dict[str, Any] | None = None,
    ) -> None:
        await self._pipe._event_emitter_handler._emit_unstreamed_answer(
            emitter,
            content=content,
            usage=usage if isinstance(usage, dict) else None,
        )

    async def _safe_emit(self, emitter: EventEmitter | None, event: dict[str, Any]) -> None:
        if emitter is None:
            return
        try:
            await emitter(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.debug("Video generation event emit failed: %s", exc, exc_info=True)

    def _resolve_api_key(self, valves: Any) -> str:
        api_key, api_key_error = self._pipe._resolve_openrouter_api_key(valves)
        if api_key_error or not api_key:
            raise VideoGenerationError(api_key_error or "OpenRouter API key is not configured.")
        return api_key

    @staticmethod
    def _consume_background_exception(task: asyncio.Future[Any]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()
