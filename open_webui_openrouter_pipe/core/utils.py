"""Shared utility functions for the OpenRouter pipe.

This module contains reusable helper functions used across the codebase:
- Template rendering (_render_error_template, etc.)
- JSON helpers (_safe_json_loads, _pretty_json)
- Type coercion (_coerce_positive_int, _coerce_bool)
- String normalization
- ULID generation and marker system
- Path and identifier sanitization

These utilities have minimal dependencies and can be used by any module.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any, Iterator, Literal, Optional, cast

# Import constants needed for template rendering and ULID generation
from .config import (
    DEFAULT_OPENROUTER_ERROR_TEMPLATE,
    ULID_LENGTH,
    CROCKFORD_ALPHABET,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_TEMPLATE_IF_TOKEN_RE = re.compile(r"\{\{\s*(#if\s+(\w+)|/if)\s*\}\}")
_MARKER_SUFFIX = "]: #"  # Suffix for artifact markers in text
_CROCKFORD_SET = frozenset(CROCKFORD_ALPHABET)

# -----------------------------------------------------------------------------
# Template Rendering
# -----------------------------------------------------------------------------

def _render_error_template(template: str, values: dict[str, Any]) -> str:
    """Render a user-supplied template, honoring {{#if}} conditionals."""
    if not template:
        template = DEFAULT_OPENROUTER_ERROR_TEMPLATE

    rendered_lines: list[str] = []
    condition_stack: list[bool] = []

    def _conditions_active() -> bool:
        """Return True when the current {{#if}} stack has no falsy guards."""
        return all(condition_stack) if condition_stack else True

    for raw_line in template.splitlines():
        last_index = 0
        line_parts: list[str] = []

        for match in _TEMPLATE_IF_TOKEN_RE.finditer(raw_line):
            segment = raw_line[last_index:match.start()]
            if segment and _conditions_active():
                line_parts.append(segment)

            token = match.group(1) or ""
            var_name = match.group(2)
            if token.startswith("#if"):
                condition_stack.append(bool(values.get(var_name or "")))
            else:
                if condition_stack:
                    condition_stack.pop()

            last_index = match.end()

        tail_segment = raw_line[last_index:]
        if tail_segment and _conditions_active():
            line_parts.append(tail_segment)

        if not line_parts:
            if raw_line.strip():
                continue
            if not _conditions_active():
                continue
            rendered_lines.append("")
            continue

        line = "".join(line_parts)

        drop_line = False
        for name, value in values.items():
            placeholder = f"{{{name}}}"
            if placeholder in line:
                if not _template_value_present(value):
                    drop_line = True
                line = line.replace(placeholder, "" if value is None else str(value))
        if drop_line:
            continue
        rendered_lines.append(line)
    return "\n".join(rendered_lines).strip()


def _pretty_json(value: Any) -> str:
    """Return a human-readable JSON string or an empty string when not applicable."""
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return text.strip()
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


# -----------------------------------------------------------------------------
# JSON Helpers
# -----------------------------------------------------------------------------

def _safe_json_loads(payload: Optional[str]) -> Any:
    """Return parsed JSON or None without raising."""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Type Coercion and String Normalization
# -----------------------------------------------------------------------------

def _coerce_positive_int(value: Any) -> Optional[int]:
    """Convert strings/bools into positive integers (MB)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _coerce_bool(value: Any) -> Optional[bool]:
    """Best-effort coercion of truthy string/int flags into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _normalize_string_list(value: Any) -> list[str]:
    """Return a list of trimmed strings."""
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _normalize_optional_str(entry)
        if text:
            items.append(text)
    return items



# -----------------------------------------------------------------------------
# ULID Generation and Marker System
# -----------------------------------------------------------------------------


def _extract_marker_ulid(line: str) -> str | None:
    """Return the ULID embedded in a hidden marker line, if present."""
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("[") or not stripped.endswith(_MARKER_SUFFIX):
        return None
    body = stripped[1 : -len(_MARKER_SUFFIX)]
    if len(body) != ULID_LENGTH:
        return None
    for char in body:
        if char not in _CROCKFORD_SET:
            return None
    return body



def contains_marker(text: str) -> bool:
    """Fast check: does the text contain any embedded ULID markers?

    Args:
        text: Text to scan.

    Returns:
        bool: True if the sentinel substring is present; otherwise False.
    """
    return bool(_iter_marker_spans(text))


def split_text_by_markers(text: str) -> list[dict]:
    """Split text into a sequence of literal segments and marker segments.

    Args:
        text: Source text possibly containing embedded markers.

    Returns:
        list[dict]: A list like:
            [
              {"type": "text",   "text": "..."},
              {"type": "marker", "marker": "01H...Q4"},
              ...
            ]
    """
    segments: list[dict[str, Any]] = []
    last = 0
    for span in _iter_marker_spans(text):
        if span["start"] > last:
            segments.append({"type": "text", "text": text[last:span["start"]]})
        segments.append(
            {
                "type": "marker",
                "marker": span["marker"],
            }
        )
        last = span["end"]
    if last < len(text):
        segments.append({"type": "text", "text": text[last:]})
    return segments


# -----------------------------------------------------------------------------
# Configuration and Environment Helpers
# -----------------------------------------------------------------------------

# Global cache for Open WebUI config module
_OPEN_WEBUI_CONFIG_MODULE: Any | None = None


def _get_open_webui_config_module() -> Any | None:
    """Return the cached open_webui.config module if available."""
    global _OPEN_WEBUI_CONFIG_MODULE
    if _OPEN_WEBUI_CONFIG_MODULE is not None:
        return _OPEN_WEBUI_CONFIG_MODULE
    try:
        import open_webui.config as ow_config  # type: ignore
    except Exception:
        return None
    _OPEN_WEBUI_CONFIG_MODULE = ow_config
    return ow_config


def _unwrap_config_value(value: Any) -> Any:
    """Return the raw value from a PersistentConfig-like object."""
    if value is None:
        return None
    return getattr(value, "value", value)


# -----------------------------------------------------------------------------
# Payload and Content Utilities
# -----------------------------------------------------------------------------

# Marker for redacted data URLs
_REDACTED_DATA_URL_MARKER = "[REDACTED]"


def _redact_payload_blobs(value: Any, *, max_chars: int = 256) -> Any:
    """Return a copy of ``value`` with large data:...;base64,... blobs truncated.

    This keeps DEBUG logs readable and prevents multi-megabyte payloads from being emitted
    when images/files are inlined for providers.
    """

    def _redact_data_url(text: str) -> str:
        candidate = text.strip()
        if not candidate.startswith("data:") or ";base64," not in candidate:
            return text
        header, b64 = candidate.split(",", 1)
        if len(candidate) <= max_chars:
            return candidate
        keep = max(8, min(64, max_chars // 4))
        return f"{header},{b64[:keep]}…{_REDACTED_DATA_URL_MARKER}({len(b64)} chars)…"

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        if isinstance(obj, str):
            return _redact_data_url(obj)
        return obj

    safe_max = int(max_chars) if isinstance(max_chars, int) else 256
    safe_max = max(64, min(safe_max, 8192))
    max_chars = safe_max
    return _walk(value)


def _extract_plain_text_content(content: Any) -> str:
    """Collapse Open WebUI-style content blocks into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text_val = block.get("text") or block.get("content")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    if isinstance(content, dict):
        text_val = content.get("text") or content.get("content")
        if isinstance(text_val, str):
            return text_val
    return str(content or "")


# -----------------------------------------------------------------------------
# Feature Flags and Metadata
# -----------------------------------------------------------------------------

def _extract_feature_flags(__metadata__: dict[str, Any]) -> dict[str, Any]:
    """Return flat feature flags from Open WebUI metadata.

    Open WebUI sends feature flags as a flat dict under ``metadata["features"]``
    (e.g. ``{"web_search": True, "code_interpreter": False}``).
    """
    raw_features = __metadata__.get("features") if isinstance(__metadata__, dict) else None
    return dict(raw_features) if isinstance(raw_features, dict) else {}


def merge_usage_stats(total, new):
    """Recursively merge nested usage statistics.

    For numeric values, sums are accumulated; for dicts, the function recurses;
    other values overwrite the prior value when non-None.

    Args:
        total: Accumulator dictionary to update.
        new:   Newly reported usage block to merge into `total`.

    Returns:
        dict: The updated accumulator dictionary (`total`).
    """
    for k, v in new.items():
        if isinstance(v, dict):
            total[k] = merge_usage_stats(total.get(k, {}), v)
        elif isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = v if v is not None else total.get(k, 0)
    return total


# -----------------------------------------------------------------------------
# Formatting Utilities
# -----------------------------------------------------------------------------

def wrap_code_block(text: str, language: str = "python") -> str:
    """Wrap text in a fenced Markdown code block.

    The fence length adapts to the longest backtick run within the text to avoid
    prematurely closing the block.

    Args:
        text:     The code or content to wrap.
        language: Markdown fence language tag.

    Returns:
        str: Markdown code block.
    """
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


# -----------------------------------------------------------------------------
# Template and String Utilities
# -----------------------------------------------------------------------------

def _template_value_present(value: Any) -> bool:
    """Return True when a placeholder value should be rendered."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, (int, float)):
        return True
    return bool(value)


def _normalize_optional_str(value: Any) -> Optional[str]:
    """Convert arbitrary input into a trimmed string or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _sanitize_path_component(value: str, *, fallback: str = "unknown", max_length: int = 128) -> str:
    """Return a filesystem-safe path component to prevent traversal/odd characters."""
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    cleaned = cleaned.strip("._-") or fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._-") or fallback
    return cleaned


# -----------------------------------------------------------------------------
# HTTP and Timing Utilities
# -----------------------------------------------------------------------------

def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Convert Retry-After header value into seconds."""
    import datetime
    import email.utils

    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        seconds = float(trimmed)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(trimmed)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        seconds = (dt - now).total_seconds()
        return max(0.0, seconds)
    except (TypeError, ValueError, OverflowError):
        return None


# -----------------------------------------------------------------------------
# ULID Marker System
# -----------------------------------------------------------------------------

def _serialize_marker(ulid: str) -> str:
    """Return the hidden marker representation for ``ulid``."""
    return f"[{ulid}{_MARKER_SUFFIX}"


def _iter_marker_spans(text: str) -> list[dict[str, Any]]:
    """Return ordered ULID marker spans."""
    if not text:
        return []

    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in text.splitlines(True):
        stripped = segment.strip()
        marker_ulid = _extract_marker_ulid(stripped)
        if marker_ulid:
            offset = segment.find(stripped)
            start = cursor + (offset if offset >= 0 else 0)
            spans.append(
                {
                    "start": start,
                    "end": start + len(stripped),
                    "marker": marker_ulid,
                }
            )
        cursor += len(segment)

    spans.sort(key=lambda span: span["start"])
    return spans
