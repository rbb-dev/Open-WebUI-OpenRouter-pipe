"""Video intent classifier — orchestration, validation, history hygiene.

Wraps the structured_task module to invoke a task model with JSON-schema
strict response_format, then validates the parsed output against schema
invariants.

Degrade-open universally: any failure returns a fallback `VideoIntentResult`
(intent=text_to_video, frame_plan=[], prompt extracted from latest user
message). NEVER raises to the caller.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from ..core.config import _PIPE_METADATA_KEY
from ..core.utils import _iter_kind_marker_spans, _safe_marker_body, _serialize_kind_marker
from ..structured_task import (
    build_response_format,
    call_with_candidates,
    resolve_task_model_candidates,
)
from .video_intent_prompts import (
    INTENT_JSON_SCHEMA,
    INTENT_SCHEMA_NAME,
    INTENT_SYSTEM_PROMPT,
)

# -----------------------------------------------------------------------------
# Marker tags for the hidden intent-disclosure block
# -----------------------------------------------------------------------------

INTENT_BLOCK_START = "intent_block_start"
INTENT_BLOCK_END = "intent_block_end"
INTENT_MODE = "intent_mode"
INTENT_FRAME = "intent_frame"
INTENT_PROMPT = "intent_prompt"
INTENT_LANG = "intent_lang"
INTENT_CONFIDENCE = "intent_confidence"
INTENT_THUMBNAIL = "intent_thumbnail"
INTENT_CLARIFICATION = "intent_clarification"

VIDEO_JOB_MARKER = "videojob"
VIDEO_MODEL_MARKER = "videomodel"


def resolve_intent_user_setting(
    metadata: Any,
    field: str,
    valves: Any,
    admin_field: str,
    default: Any,
    *,
    coerce: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Prefer the per-request user value pushed by the per-model video filter
    inlet, fall back to the admin valve.

    The video filter (when admin VIDEO_INTENT_ENABLED=True) writes user-set
    intent values into ``metadata[_PIPE_METADATA_KEY]["video_intent"]`` keyed
    by the short field name (``enabled``, ``max_clarifications``,
    ``frame_extraction_index``, ``confirm_mode``). When admin disables the
    feature, the filter never writes that key, so callers fall straight
    through to the admin valve — which itself is False, short-circuiting
    the classifier.

    Args:
        metadata: the request metadata dict (may be None/non-dict).
        field: short key under ``video_intent`` (e.g. "enabled").
        valves: the merged Valves instance from the pipe call.
        admin_field: full admin valve attribute name (e.g.
            "VIDEO_INTENT_ENABLED").
        default: returned when both the metadata key and admin valve are
            missing.
        coerce: optional callable applied to the user value before return —
            useful for normalising str → enum-like values that came from
            JSON serialisation.
    """
    if isinstance(metadata, dict):
        pipe_meta = metadata.get(_PIPE_METADATA_KEY)
        if isinstance(pipe_meta, dict):
            intent_meta = pipe_meta.get("video_intent")
            if isinstance(intent_meta, dict) and field in intent_meta:
                value = intent_meta[field]
                if value is not None:
                    return coerce(value) if coerce is not None else value
    return getattr(valves, admin_field, default)

_INTENT_BLOCK_REGION_RE = re.compile(
    r"\[openrouter:v1:" + re.escape(INTENT_BLOCK_START) + r":[^\]]+\]: #"
    r".*?"
    r"\[openrouter:v1:" + re.escape(INTENT_BLOCK_END) + r":[^\]]+\]: #\s*\n?",
    re.DOTALL,
)


_PLACEHOLDER_RE = re.compile(r"\[(?:video|image):\d+\]")

_CONTROL_TOKEN_RE = re.compile(
    r"<\|[^|]*\|>"
    r"|\[/?(?:SYSTEM|INST|s|/s|/INST)\]"
    r"|```"
    r"|~~~",
    re.IGNORECASE,
)

_ZERO_WIDTH_CHARS = ("​", "‌", "‍", "﻿", "⁠")


def neutralise_control_tokens(text: str) -> str:
    """Replace prompt-injection control tokens with spaces.

    Defense-in-depth: applies NFKC normalisation to fold fullwidth lookalikes
    (`＜｜...｜＞`) into ASCII, strips zero-width characters that could split
    tokens, then runs the regex.
    """
    if not text:
        return ""
    normalised = unicodedata.normalize("NFKC", text)
    for zw in _ZERO_WIDTH_CHARS:
        normalised = normalised.replace(zw, "")
    return _CONTROL_TOKEN_RE.sub(lambda m: " " * len(m.group(0)), normalised)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

IntentLiteral = Literal[
    "text_to_video",
    "image_to_video",
    "modify_prior_video",
    "continue_prior_video",
    "ambiguous",
]

FrameSourceLiteral = Literal[
    "uploaded_attachment",
    "prior_video_first_frame",
    "prior_video_last_frame",
    "prior_video_at_timestamp",
]

FrameTargetLiteral = Literal["first_frame", "last_frame", "input_reference"]

ConfidenceLiteral = Literal["high", "medium", "low"]


@dataclass
class FramePlanEntry:
    source: FrameSourceLiteral
    source_index: Optional[int]
    timestamp_seconds: Optional[float]
    target: FrameTargetLiteral


@dataclass
class ClarificationPayload:
    needs: bool
    question: str
    options: Optional[list[str]]
    reason: str


@dataclass
class VideoIntentResult:
    intent: IntentLiteral
    frame_plan: list[FramePlanEntry]
    prompt: str
    use_user_prompt: bool
    language: str
    confidence: ConfidenceLiteral
    clarification: Optional[ClarificationPayload]
    reason: str
    downgrades: list[str] = field(default_factory=list)
    discarded_plan: bool = False
    prior_videos: list[dict[str, Any]] = field(default_factory=list)
    classifier_failed: bool = False
    failure_reason: str = ""
    frames_extracted: int = 0
    frames_retargeted: int = 0
    task_model_latency_ms: int = 0
    task_model_fallback_triggered: bool = False


# -----------------------------------------------------------------------------
# History hygiene
# -----------------------------------------------------------------------------

def strip_intent_blocks(content: str) -> str:
    """Remove `[openrouter:v1:intent_block_start]: #` ... `[openrouter:v1:intent_block_end]: #`
    regions from a chat-content string. Preserves videojob/videomodel/<video>
    markers (those sit OUTSIDE the intent block).
    """
    if not content:
        return ""
    return _INTENT_BLOCK_REGION_RE.sub("", content)


def count_prior_clarifications(messages: list[dict[str, Any]]) -> int:
    """Count consecutive most-recent assistant turns containing the
    `intent_clarification` marker. Used for the MAX_CLARIFICATIONS soft cap.

    Looks for the dedicated `intent_clarification` marker emitted only by
    `render_clarification_message` — independent of the `intent_mode` enum
    so future mode renames cannot silently break the loop guard.

    Multi-modal (list) assistant content is converted to a flat text string
    before scanning so it doesn't break the streak.

    The trailing message in a chat-completion request is always the current
    user turn; skip it before walking backwards so the streak counter sees
    the immediately preceding assistant turn(s). Counting stops at the next
    user message above the run, which is the user's original ambiguous turn.
    """
    if not isinstance(messages, list):
        return 0
    end = len(messages)
    if end > 0 and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        end -= 1
    count = 0
    for idx in range(end - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            break
        role = msg.get("role")
        if role == "user":
            break
        if role != "assistant":
            continue
        content = msg.get("content") or ""
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text") or ""))
            content = "\n".join(text_parts)
        if not isinstance(content, str):
            continue
        is_clarify = any(
            True for _ in _iter_kind_marker_spans(content, kind=INTENT_CLARIFICATION)
        )
        if is_clarify:
            count += 1
        else:
            break
    return count



# -----------------------------------------------------------------------------
# Conversation + attachments collection
# -----------------------------------------------------------------------------

_VIDEO_TAG_RE = re.compile(r"<video[^>]*>([\s\S]*?)</video>", re.IGNORECASE)


_FILE_URL_SHAPE_RE = re.compile(r"^/api/v1/files/[A-Za-z0-9_-]{1,128}(/content)?$")


def collect_prior_videos_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract prior assistant videos from chat history.

    Returns chronological list of {index, message_index, file_url, job_id,
    model_id_if_known}. Strips intent-disclosure blocks before parsing so
    thumbnails are NOT mistaken for prior videos. Pairs each <video> tag
    with the nearest preceding videojob/videomodel markers via span offsets,
    so multi-video assistant turns are correctly enumerated (F9).
    Validates each file_url against the OWUI shape regex (S11).
    """
    if not isinstance(messages, list):
        return []
    results: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str) or not content:
            continue
        cleaned = strip_intent_blocks(content)
        videojob_spans = _iter_kind_marker_spans(cleaned, kind=VIDEO_JOB_MARKER)
        videomodel_spans = _iter_kind_marker_spans(cleaned, kind=VIDEO_MODEL_MARKER)
        for video_match in _VIDEO_TAG_RE.finditer(cleaned):
            file_url = video_match.group(1).strip()
            if not file_url or not _FILE_URL_SHAPE_RE.match(file_url):
                continue
            video_start = video_match.start()
            job_id = ""
            for span in videojob_spans:
                if span["start"] < video_start:
                    job_id = str(span.get("body") or "")
            model_id = ""
            for span in videomodel_spans:
                if span["start"] < video_start:
                    model_id = str(span.get("body") or "")
            results.append({
                "index": len(results),
                "message_index": message_index,
                "file_url": file_url,
                "job_id": job_id,
                "model_id_if_known": model_id,
            })
    return results


def collect_attachments_from_video_meta(
    video_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect attachments from filter-injected video_meta as a flat list.

    Only image/video attachments are reported (other kinds excluded). Order is
    preserved per source list. Index is 0-based across the flat output.
    """
    if not isinstance(video_meta, dict):
        return []
    flat: list[dict[str, Any]] = []
    for kind_key, kind in (
        ("frame_images", "image"),
        ("video_attachments", "video"),
    ):
        items = video_meta.get(kind_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mime_type = str(item.get("content_type") or item.get("mime_type") or "").lower()
            flat.append({
                "index": len(flat),
                "kind": kind,
                "mime_type": mime_type,
                "id": item.get("id"),
                "name": item.get("name"),
                "size": item.get("size"),
            })
    return flat


# -----------------------------------------------------------------------------
# Task payload builder
# -----------------------------------------------------------------------------

def build_task_payload(
    *,
    latest_user_text: str,
    conversation: list[dict[str, Any]],
    prior_videos: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
    selected_model: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON payload sent as 'user' role content to the task model.

    Conversation is ALREADY pre-processed (intent blocks stripped) by the caller.
    Image bytes are NEVER included — markers and metadata only.
    """
    sm = {
        "id": selected_model.get("id") if isinstance(selected_model, dict) else "",
        "supported_frame_images": (
            selected_model.get("supported_frame_images") or []
            if isinstance(selected_model, dict)
            else []
        ),
    }
    from .video_intent_prompts import SCHEMA_VERSION
    return {
        "_version": SCHEMA_VERSION,
        "latest_user_text": latest_user_text,
        "conversation": conversation,
        "prior_videos": prior_videos,
        "attachments": attachments,
        "selected_model": sm,
    }


# -----------------------------------------------------------------------------
# Validator (schema invariants)
# -----------------------------------------------------------------------------

_VALID_INTENTS = {
    "text_to_video", "image_to_video", "modify_prior_video",
    "continue_prior_video", "ambiguous",
}
_VALID_SOURCES = {
    "uploaded_attachment", "prior_video_first_frame",
    "prior_video_last_frame", "prior_video_at_timestamp",
}
_VALID_TARGETS = {"first_frame", "last_frame", "input_reference"}
_VALID_CONFIDENCE = {"high", "medium", "low"}
_PROMPT_MAX_LEN = 2000
_FRAME_PLAN_MAX = 4


def _strip_placeholders(text: str) -> str:
    if not text:
        return ""
    return _PLACEHOLDER_RE.sub("", text).strip()


def _normalize_prior_video_index(
    raw: Any, prior_videos: list[dict[str, Any]]
) -> Optional[int]:
    """Resolve -1 to last index; bounds-check; return None if invalid."""
    if not isinstance(raw, int) or not prior_videos:
        return None
    if raw == -1:
        return len(prior_videos) - 1
    if 0 <= raw < len(prior_videos):
        return raw
    return None


def validate_intent_params(
    raw: dict[str, Any],
    *,
    attachments_count: int,
    prior_videos: list[dict[str, Any]],
    video_model: dict[str, Any],
    explicit_frame_images_present: bool,
    prior_clarifications_in_session: int,
    max_clarifications: int,
    fallback_prompt: str,
) -> VideoIntentResult:
    """Validate raw classifier output against schema invariants.

    Drop-not-raise pattern for per-entry failures. Top-level missing-required
    fields fall back to safe defaults.
    """
    downgrades: list[str] = []

    intent_raw = raw.get("intent")
    intent: IntentLiteral = (
        intent_raw if isinstance(intent_raw, str) and intent_raw in _VALID_INTENTS
        else "text_to_video"  # type: ignore[assignment]
    )

    # Confidence
    confidence_raw = raw.get("confidence")
    confidence: ConfidenceLiteral = (
        confidence_raw if isinstance(confidence_raw, str) and confidence_raw in _VALID_CONFIDENCE
        else "low"  # type: ignore[assignment]
    )

    language_raw = str(raw.get("language") or "en").strip()
    if not re.fullmatch(r"[a-z]{2,8}", language_raw.lower()):
        language = "en"
    else:
        language = language_raw.lower()

    use_user_prompt = bool(raw.get("use_user_prompt"))

    prompt_raw = raw.get("prompt")
    prompt = _strip_placeholders(prompt_raw if isinstance(prompt_raw, str) else "")
    if len(prompt) > _PROMPT_MAX_LEN:
        prompt = prompt[:_PROMPT_MAX_LEN].rstrip()

    reason = str(raw.get("reason") or "").strip()[:500]

    # Clarification
    clarification: Optional[ClarificationPayload] = None
    clar_raw = raw.get("clarification")
    if isinstance(clar_raw, dict):
        needs = bool(clar_raw.get("needs"))
        question = str(clar_raw.get("question") or "").strip()
        options_raw = clar_raw.get("options")
        options = (
            [str(opt)[:200] for opt in options_raw[:5] if isinstance(opt, str)]
            if isinstance(options_raw, list) else None
        )
        clar_reason = str(clar_raw.get("reason") or "").strip()
        if needs and (not question or len(question) > 280):
            needs = False
        clarification = ClarificationPayload(
            needs=needs, question=question, options=options, reason=clar_reason,
        )

    frame_plan: list[FramePlanEntry] = []
    raw_plan = raw.get("frame_plan")
    supported_frame_types: set[str] = {
        str(t) for t in (
            video_model.get("supported_frame_images")
            if isinstance(video_model, dict) else []
        ) or []
    }
    seen_targets: set[str] = set()  
    if isinstance(raw_plan, list):
        for entry_raw in raw_plan[:_FRAME_PLAN_MAX]:
            if not isinstance(entry_raw, dict):
                continue
            source = entry_raw.get("source")
            target = entry_raw.get("target")
            if source not in _VALID_SOURCES or target not in _VALID_TARGETS:
                continue

            source_index_raw = entry_raw.get("source_index")
            timestamp_seconds_raw = entry_raw.get("timestamp_seconds")
            source_index: Optional[int]
            timestamp_seconds: Optional[float]

            if source == "uploaded_attachment":
                if not isinstance(source_index_raw, int):
                    continue
                if not (0 <= source_index_raw < attachments_count):
                    downgrades.append(f"dropped_uploaded_attachment_index_{source_index_raw}")
                    continue
                source_index = source_index_raw
                timestamp_seconds = None
            elif source in ("prior_video_first_frame", "prior_video_last_frame",
                            "prior_video_at_timestamp"):
                resolved = _normalize_prior_video_index(source_index_raw, prior_videos)
                if resolved is None:
                    downgrades.append(f"dropped_invalid_prior_video_index_{source_index_raw}")
                    continue
                source_index = resolved
                if source == "prior_video_at_timestamp":
                    if not isinstance(timestamp_seconds_raw, (int, float)) or timestamp_seconds_raw < 0:
                        downgrades.append("dropped_invalid_timestamp")
                        continue
                    timestamp_seconds = float(timestamp_seconds_raw)
                else:
                    timestamp_seconds = None
            else:
                continue

            if target in ("first_frame", "last_frame"):
                if not supported_frame_types:
                    downgrades.append(
                        f"dropped_{target}_no_frame_support_for_model"
                    )
                    continue
                if target not in supported_frame_types:
                    downgrades.append(
                        f"downgraded_unsupported_target_{target}_to_input_reference"
                    )
                    target = "input_reference"

            if target == "input_reference" and not supported_frame_types:
                downgrades.append("dropped_input_reference_no_frame_support_for_model")
                continue

            if target in ("first_frame", "last_frame"):
                if target in seen_targets:
                    downgrades.append(f"deduped_duplicate_{target}_target")
                    continue
                seen_targets.add(target)

            frame_plan.append(FramePlanEntry(
                source=source,  # type: ignore[arg-type]
                source_index=source_index,
                timestamp_seconds=timestamp_seconds,
                target=target,  # type: ignore[arg-type]
            ))
            if len(frame_plan) >= _FRAME_PLAN_MAX:
                break

    # Cross-field invariants
    if intent == "text_to_video" and frame_plan:
        downgrades.append("forced_empty_frame_plan_for_text_to_video")
        frame_plan = []

    discarded_plan = False
    if explicit_frame_images_present and frame_plan:
        prior_entries = [e for e in frame_plan if e.source != "uploaded_attachment"]
        retarget_entries = [e for e in frame_plan if e.source == "uploaded_attachment"]
        if prior_entries:
            downgrades.append("dropped_prior_video_entries_explicit_attachments_present")
        frame_plan = retarget_entries
        if not retarget_entries:
            discarded_plan = True
        if intent in ("modify_prior_video", "continue_prior_video"):
            downgrades.append("intent_downgraded_due_to_explicit_attachments")
            intent = "image_to_video"

    if intent == "image_to_video" and not any(
        e.source == "uploaded_attachment" for e in frame_plan
    ):
        downgrades.append("image_to_video_without_attachment_downgraded_to_text")
        intent = "text_to_video"
        frame_plan = []
    if intent in ("modify_prior_video", "continue_prior_video") and not frame_plan:
        downgrades.append(f"{intent}_dropped_to_text_no_frame_support")
        intent = "text_to_video"

    if (
        clarification is not None
        and clarification.needs
        and prior_clarifications_in_session >= max_clarifications
    ):
        downgrades.append("clarification_capped_max_reached")
        clarification = ClarificationPayload(needs=False, question="", options=None, reason="")
        if intent == "ambiguous":
            intent = "text_to_video"
            frame_plan = []
            if not prompt:
                prompt = fallback_prompt

    if intent == "ambiguous" and not (clarification and clarification.needs):
        downgrades.append("ambiguous_without_clarification_downgraded_to_text")
        intent = "text_to_video"
        frame_plan = []
        if not prompt:
            prompt = fallback_prompt
    elif intent != "ambiguous" and not prompt:
        prompt = fallback_prompt

    return VideoIntentResult(
        intent=intent,
        frame_plan=frame_plan,
        prompt=prompt,
        use_user_prompt=use_user_prompt,
        language=language,
        confidence=confidence,
        clarification=clarification,
        reason=reason,
        downgrades=downgrades,
        discarded_plan=discarded_plan,
    )


def _intent_mode_for_telemetry(result: VideoIntentResult) -> str:
    """Compress intent + frame_plan into a single telemetry-friendly mode label."""
    if result.clarification and result.clarification.needs:
        return "clarify"
    if not result.frame_plan:
        return "text2video"
    sources = {e.source for e in result.frame_plan}
    if any(s.startswith("prior_video_") for s in sources):
        return "image2video_priorframe"
    if "uploaded_attachment" in sources:
        return "image2video_attached"
    return result.intent


def emit_telemetry_log(
    result: VideoIntentResult,
    *,
    logger: logging.Logger,
    chat_id: str,
    log_decisions_enabled: bool,
    task_model_latency_ms: Optional[int] = None,
    task_model_fallback_triggered: Optional[bool] = None,
) -> None:
    """Emit one structured INFO log line per video intent turn.

    `task_model_latency_ms` and `task_model_fallback_triggered` are optional
    overrides; when omitted, values are read from the result (which the
    classifier orchestrator stamps at decision time). The caller (video.py)
    emits this AFTER the terminal path runs (materialise / block / clarify)
    so `result.frames_extracted` reflects what actually happened — not what
    the classifier requested.
    """
    requested_prior = sum(
        1 for e in result.frame_plan if e.source.startswith("prior_video_")
    )
    extracted = int(result.frames_extracted)
    latency_ms = (
        task_model_latency_ms
        if task_model_latency_ms is not None
        else result.task_model_latency_ms
    )
    fallback_triggered = (
        task_model_fallback_triggered
        if task_model_fallback_triggered is not None
        else result.task_model_fallback_triggered
    )
    payload = {
        "event": "video_intent",
        "chat_id_hash": _hash_chat_id(chat_id),
        "intent_mode": _intent_mode_for_telemetry(result),
        "intent": result.intent,
        "confidence": result.confidence,
        "language": result.language,
        "frame_plan_size": len(result.frame_plan),
        "clarification_emitted": bool(
            result.clarification and result.clarification.needs
        ),
        "task_model_latency_ms": int(latency_ms),
        "task_model_fallback_triggered": bool(fallback_triggered),
        "classifier_failed": bool(result.classifier_failed),
        "failure_reason": result.failure_reason or "",
        "prior_video_frame_extracted": extracted > 0,
        "prior_video_frames_extracted_count": extracted,
        "prior_video_frames_requested_count": requested_prior,
        "frames_retargeted_count": int(result.frames_retargeted),
        "downgrades_count": len(result.downgrades),
        "discarded_plan": result.discarded_plan,
    }
    if log_decisions_enabled:
        logger.info("video_intent telemetry: %s", json.dumps(payload, default=str))
    else:
        logger.debug("video_intent telemetry: %s", json.dumps(payload, default=str))


def _hash_chat_id(chat_id: str) -> str:
    """Hash chat_id for log-safe identification (8-char prefix)."""
    if not chat_id:
        return ""
    import hashlib
    return hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:8]


def fallback_intent_result(prompt: str) -> VideoIntentResult:
    """Produce the degrade-open VideoIntentResult: text_to_video with the
    user's raw prompt and an empty frame plan. Used on every failure path."""
    return VideoIntentResult(
        intent="text_to_video",
        frame_plan=[],
        prompt=prompt or "",
        use_user_prompt=False,
        language="en",
        confidence="low",
        clarification=None,
        reason="degrade_open_fallback",
        downgrades=[],
        discarded_plan=False,
    )


# -----------------------------------------------------------------------------
# resolve_intent — main orchestration (degrade-open)
# -----------------------------------------------------------------------------

async def resolve_intent(
    *,
    body: dict[str, Any],
    video_meta: dict[str, Any],
    video_model: dict[str, Any],
    valves: Any,
    request: Any,
    user_obj: Any,
    chat_id: str,
    logger: logging.Logger,
    invoke_chat_completion: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None,
    fallback_prompt_text: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> VideoIntentResult:
    """Run the classifier; ALWAYS returns a VideoIntentResult — never raises.

    Args:
        body: OWUI request body (must contain 'messages').
        video_meta: Filter-injected metadata (`frame_images`, `params`, etc.).
        video_model: Model spec snapshot (with `supported_frame_images`).
        valves: Pipe Valves object.
        request: FastAPI Request.
        user_obj: Resolved UserModel (or compatible). Required by OWUI's
            generate_chat_completion.
        chat_id: Chat session id (for telemetry).
        logger: Pipe logger.
        invoke_chat_completion: Async callable that takes form_data and returns
            the chat-completion response. Defaults to OWUI's generate_chat_completion
            via lazy import; can be overridden for tests.
        fallback_prompt_text: Pre-computed fallback prompt (typically the latest
            user message text). Used by the degrade-open path.
    """
    fallback = fallback_intent_result(fallback_prompt_text)

    try:
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list):
            return fallback
        conversation = _build_conversation(messages)
        prior_videos = collect_prior_videos_from_messages(messages)
        attachments = collect_attachments_from_video_meta(video_meta)
        explicit_frame_images_present = bool(
            isinstance(video_meta, dict) and video_meta.get("frame_images")
        )
        prior_clar = count_prior_clarifications(messages)
        max_clar_raw = resolve_intent_user_setting(
            metadata, "max_clarifications",
            valves, "VIDEO_INTENT_MAX_CLARIFICATIONS", 1,
        )
        try:
            max_clar = max(0, min(3, int(max_clar_raw)))
        except (TypeError, ValueError):
            max_clar = 1

        task_payload = build_task_payload(
            latest_user_text=neutralise_control_tokens(fallback_prompt_text),
            conversation=conversation,
            prior_videos=prior_videos,
            attachments=attachments,
            selected_model=video_model or {},
        )

        candidates = resolve_task_model_candidates(
            request=request,
            mode=getattr(valves, "VIDEO_INTENT_TASK_MODEL_MODE", "external"),
            fallback=getattr(valves, "VIDEO_INTENT_TASK_MODEL_FALLBACK", "other_task_model"),
        )
        if not candidates:
            logger.debug("video_intent: no task-model candidates configured; degrade-open")
            return fallback

        response_format = build_response_format(
            name=INTENT_SCHEMA_NAME, schema=INTENT_JSON_SCHEMA, strict=True,
        )
        supported_params = video_model.get("supported_parameters") if isinstance(video_model, dict) else None
        del supported_params  # unused for now — task-model spec lookup is future work

        def _build_form_data(model_id: str) -> dict[str, Any]:
            return {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": json.dumps(task_payload, ensure_ascii=True)},
                ],
                "temperature": 0,
                "stream": False,
                "response_format": response_format,
                "metadata": {"task": INTENT_SCHEMA_NAME, "chat_id": chat_id},
            }

        if invoke_chat_completion is None:
            invoke_chat_completion = _make_default_invoke(request, user_obj)

        timeout_s = float(getattr(valves, "VIDEO_INTENT_TIMEOUT_S", 8))
        if timeout_s <= 0:
            timeout_s = 8.0
        _t0 = time.monotonic()
        params = await call_with_candidates(
            candidates=candidates,
            build_form_data=_build_form_data,
            invoke=invoke_chat_completion,
            timeout_s=timeout_s,
            logger=logger,
        )
        _latency_ms = int((time.monotonic() - _t0) * 1000)

        result = validate_intent_params(
            params,
            attachments_count=len(attachments),
            prior_videos=prior_videos,
            video_model=video_model or {},
            explicit_frame_images_present=explicit_frame_images_present,
            prior_clarifications_in_session=prior_clar,
            max_clarifications=max_clar,
            fallback_prompt=fallback_prompt_text,
        )
        result.prior_videos = prior_videos
        result.task_model_latency_ms = _latency_ms
        result.task_model_fallback_triggered = False
        return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("video_intent classifier failed (degrade-open): %s", exc)
        fallback.classifier_failed = True
        fallback.failure_reason = f"{type(exc).__name__}: {exc}"
        return fallback


def _build_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact conversation snapshot for the task-model payload.

    Strips intent disclosure blocks. Reports per-turn:
    {message_index, role, text, has_video_marker, attached_image_count}.
    Does NOT include image data — markers/counts only.
    """
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")
        text = ""
        attached = 0
        if isinstance(content, str):
            text = strip_intent_blocks(content)
        elif isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text") or ""))
                elif part.get("type") in ("image_url", "input_image"):
                    attached += 1
            text = strip_intent_blocks("\n".join(text_parts))
        text = neutralise_control_tokens(text)
        has_video = bool(_VIDEO_TAG_RE.search(text)) if role == "assistant" else False
        out.append({
            "message_index": i,
            "role": role,
            "text": text[:2000],  
            "has_video_marker": has_video,
            "attached_image_count": attached,
        })
    return out


def _make_default_invoke(
    request: Any, user_obj: Any
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """Build the default invoke closure that calls OWUI's generate_chat_completion."""
    async def _invoke(form_data: dict[str, Any]) -> Any:
        # Lazy import to avoid pulling OWUI at module load
        from open_webui.utils.chat import generate_chat_completion  # type: ignore[import-not-found]
        return await generate_chat_completion(
            request=request, form_data=form_data, user=user_obj,
        )
    return _invoke


# -----------------------------------------------------------------------------
# Intent Disclosure Block rendering
# -----------------------------------------------------------------------------

_DOWNGRADE_USER_MESSAGES: dict[str, str] = {
    "frame_extract_failed": "Could not extract frame from previous video.",
    "prior_video_download_failed": "Previous video could not be loaded.",
    "prior_video_index_unresolvable": "Referenced previous video not found.",
    "prior_video_unauthorized": "Cannot access referenced video (different user).",
    "frame_upload_failed": "Frame could not be uploaded.",
    "input_reference_target_skipped": "Style reference skipped (not supported by model yet).",
    "materialise_failed": "A non-critical step was skipped.",
    "discarded_classifier_plan_explicit_attachments_present": "Used your uploaded frame instead of an inferred one.",
    "intent_downgraded_due_to_explicit_attachments": "Used your uploaded frame instead of an inferred one.",
    "clarification_capped_max_reached": "Proceeding with best-effort interpretation.",
}


def _user_facing_downgrade_message(code: str) -> str:
    """Map an internal downgrade code to a user-friendly message.

    Strips the optional `_idx_N` / `_*` suffix and looks up the base code.
    Falls back to a generic message — never echoes raw exception text.
    """
    if not isinstance(code, str) or not code:
        return "A non-critical step was skipped."
    for prefix in _DOWNGRADE_USER_MESSAGES:
        if code.startswith(prefix):
            return _DOWNGRADE_USER_MESSAGES[prefix]
    return "A non-critical step was skipped."


def should_emit_confirmation_footer(
    intent: VideoIntentResult,
    *,
    confirm_mode: str,
) -> bool:
    """Decide whether to render the Intent Disclosure block based on the
    `VIDEO_INTENT_CONFIRM_MODE` valve.

    - `always`: render whenever the classifier produced a frame_plan.
    - `on_reference` (default): render when the plan reuses prior video
      content OR there are multiple frames (i.e. anything beyond a simple
      single-attachment image-to-video).
    - `low_confidence`: render only when the classifier flagged its decision
      as low-confidence — the user gets to verify the guess.
    - `never`: never render.

    Empty frame_plans never render regardless of mode (nothing to disclose).
    Unknown mode strings fall through to `on_reference` semantics.
    """
    if not intent.frame_plan:
        return False
    if confirm_mode == "never":
        return False
    if confirm_mode == "always":
        return True
    if confirm_mode == "low_confidence":
        return intent.confidence == "low"
    has_prior = any(e.source != "uploaded_attachment" for e in intent.frame_plan)
    multi_frame = len(intent.frame_plan) > 1
    return has_prior or multi_frame


def render_intent_disclosure_block(
    intent: VideoIntentResult,
    *,
    thumb_urls: list[str],
) -> str:
    """Render the markdown intent-disclosure block.

    Hidden markers wrap a visible blockquote with thumbnails + cleaned prompt.
    Only renders when `intent.frame_plan` is non-empty. Returns empty string on
    any rendering failure (defense-in-depth — never crashes the pipe).
    """
    if not intent.frame_plan:
        return ""

    try:
        lines: list[str] = []
        lines.append(_serialize_kind_marker(INTENT_BLOCK_START, "1"))
        lines.append(_serialize_kind_marker(INTENT_MODE, _safe_marker_body(intent.intent)))
        lines.append(_serialize_kind_marker(INTENT_CONFIDENCE, _safe_marker_body(intent.confidence)))
        lines.append(_serialize_kind_marker(INTENT_LANG, _safe_marker_body(intent.language or "en")))
        for idx, entry in enumerate(intent.frame_plan):
            thumb = thumb_urls[idx] if idx < len(thumb_urls) else ""
            if not thumb:
                continue
            body = (
                f"{idx}|source={entry.source}|src_idx={entry.source_index}"
                f"|ts={entry.timestamp_seconds}|target={entry.target}|thumb={thumb}"
            )
            lines.append(_serialize_kind_marker(INTENT_FRAME, _safe_marker_body(body)))
        if intent.prompt:
            prompt_marker_body = (intent.prompt[:200] + "…") if len(intent.prompt) > 200 else intent.prompt
            lines.append(_serialize_kind_marker(INTENT_PROMPT, _safe_marker_body(prompt_marker_body)))

        lines.append("")
        mode_label = {
            "modify_prior_video": "🎬 **Modifying previous video**",
            "continue_prior_video": "🎬 **Continuing previous video**",
            "image_to_video": "🎬 **Generating from attached image**",
            "text_to_video": "🎬 **Generating new video**",
            "ambiguous": "🤔 **Clarifying intent**",
        }.get(intent.intent, "🎬 **Generating video**")
        lines.append(f"> {mode_label}")
        if thumb_urls:
            for url in thumb_urls:
                if url:
                    lines.append("> ")
                    lines.append(f"> ![ref]({url})")
        if intent.prompt:
            lines.append("> ")
            lines.append(f"> Prompt: *\"{intent.prompt}\"*")
        if intent.downgrades:
            for note in intent.downgrades:
                lines.append("> ")
                user_msg = _user_facing_downgrade_message(note)
                lines.append(f"> ⚠️ {user_msg}")
        lines.append("")
        lines.append(_serialize_kind_marker(INTENT_BLOCK_END, "1"))
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def render_clarification_message(intent: VideoIntentResult) -> str:
    """Render the clarification chat message."""
    if not intent.clarification or not intent.clarification.needs:
        return ""
    lines: list[str] = []
    lines.append(_serialize_kind_marker(INTENT_BLOCK_START, "1"))
    lines.append(_serialize_kind_marker(INTENT_MODE, "ambiguous"))
    lines.append(_serialize_kind_marker(INTENT_CLARIFICATION, "1"))
    lines.append(_serialize_kind_marker(INTENT_LANG, intent.language or "en"))
    lines.append("")
    lines.append("🤔 **Quick question**")
    lines.append("")
    lines.append(intent.clarification.question)
    if intent.clarification.options:
        lines.append("")
        for i, opt in enumerate(intent.clarification.options, start=1):
            lines.append(f"{i}. **{opt}**")
        lines.append("")
        lines.append("_Reply with `1`, `2`, or rephrase._")
    lines.append("")
    lines.append(_serialize_kind_marker(INTENT_BLOCK_END, "1"))
    return "\n".join(lines) + "\n"
