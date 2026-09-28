"""Introspect valves and overlay enrichment (title/group/help) for the Config tab."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import typing
from typing import Any

import annotated_types as at
from cryptography.fernet import Fernet, InvalidToken
from pydantic import ValidationError

from ...core.config import EncryptedStr, _is_template_valve, _valve_schema
from .config_meta import CONFIG_META

logger = logging.getLogger(__name__)

_UNCATEGORIZED_TOP = "Uncategorized"
_ACRONYMS = frozenset(
    {"ZDR", "SSRF", "TTL", "MIME", "URL", "URLS", "ID", "IDS", "DB", "HTTP", "LZ4",
     "API", "SSE", "JSON", "LLM", "RAG", "HMAC", "IP", "CSV", "MB", "KB", "GB", "MS"}
)
_BRAND = {"OPENROUTER": "OpenRouter", "WEBUI": "WebUI", "OWUI": "Open WebUI"}


def is_secret(annotation: Any) -> bool:
    """True iff the field is an ``EncryptedStr`` (directly or under Optional)."""
    if annotation is EncryptedStr:
        return True
    return any(arg is EncryptedStr for arg in typing.get_args(annotation))


def _literal_options(annotation: Any) -> list[Any] | None:
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))
    for arg in typing.get_args(annotation):
        found = _literal_options(arg)
        if found:
            return found
    return None


def _base_type(annotation: Any) -> tuple[Any, bool]:
    """Collapse ``Optional[X]`` / ``X | None`` to ``(X, nullable)``."""
    args = typing.get_args(annotation)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        return (non_none[0] if non_none else annotation), True
    return annotation, False


def _bounds(field: Any) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for meta in field.metadata:
        if isinstance(meta, at.Ge):
            out["ge"] = meta.ge
        elif isinstance(meta, at.Le):
            out["le"] = meta.le
        elif isinstance(meta, at.Gt):
            out["gt"] = meta.gt
        elif isinstance(meta, at.Lt):
            out["lt"] = meta.lt
    return out or None


def _humanize(name: str) -> str:
    parts = name.split("_")
    words: list[str] = []
    for i, part in enumerate(parts):
        if part in _BRAND:
            words.append(_BRAND[part])
        elif part in _ACRONYMS:
            words.append(part)
        else:
            words.append(part.capitalize() if i == 0 else part.lower())
    return " ".join(words)


def _widget(base: Any, enum: list[Any] | None, secret: bool) -> str:
    if secret:
        return "masked secret"
    if enum:
        return "dropdown (enum)"
    if base is bool:
        return "toggle (bool)"
    if base is int:
        return "number (int)"
    if base is float:
        return "number (float)"
    return "text"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def describe_valves(valves_cls: type) -> list[dict[str, Any]]:
    """Return the static per-field spec (structure + enrichment overlay).

    No live values — the caller merges the current saved values in separately.
    """
    specs: list[dict[str, Any]] = []
    for name, fld in valves_cls.model_fields.items():
        annotation = fld.annotation
        secret = is_secret(annotation)
        base, nullable = _base_type(annotation)
        enum = _literal_options(annotation)

        meta = CONFIG_META.get(name)
        enriched = meta is not None
        title = (meta or {}).get("title") or _humanize(name)
        group = (meta or {}).get("group") or f"{_UNCATEGORIZED_TOP}/General"
        detail = (meta or {}).get("detail") or (fld.description or "")
        top, _, sub = group.partition("/")

        specs.append(
            {
                "name": name,
                "title": title,
                "top": top,
                "sub": sub or "General",
                "detail": detail,
                "enriched": enriched,
                "widget": _widget(base, enum, secret),
                "enum": [str(opt) for opt in enum] if enum else None,
                "bounds": _bounds(fld),
                "nullable": nullable,
                "secret": secret,
                "is_template": _is_template_valve(name),
                "default": None if secret else json_safe(fld.get_default(call_default_factory=True)),
            }
        )
    return specs


def drift(valves_cls: type) -> dict[str, list[str]]:
    """Enrichment drift: valves with no entry, and entries with no valve."""
    live = set(valves_cls.model_fields)
    mapped = set(CONFIG_META)
    return {"unenriched": sorted(live - mapped), "orphaned": sorted(mapped - live)}


def readable_stored(valves_cls: type, stored: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    kept = {k: v for k, v in stored.items() if v is not None and k in valves_cls.model_fields}
    try:
        built = valves_cls(**kept)
    except ValidationError as exc:
        bad = {str(err["loc"][0]) for err in exc.errors() if err.get("loc")}
        return {k: v for k, v in kept.items() if k not in bad}, sorted(bad)
    bad = {k for k in kept if k not in built.model_fields_set}
    bad = {k for k in bad if not _is_blanked_nullable(valves_cls, k, kept.get(k))}
    return {k: v for k, v in kept.items() if k not in bad}, sorted(bad)


def _is_blanked_nullable(valves_cls: type, name: str, value: Any) -> bool:
    from ...core.valve_salvage import _admits_none

    return _admits_none(valves_cls, name) and isinstance(value, str) and not value.strip()


def _is_clear_edit(fld: Any, value: Any, current: dict[str, Any]) -> bool:
    return bool(is_secret(fld.annotation)) and value is None


class _ClientMessage(RuntimeError):
    pass


async def _raw_valve_column(pipe_id: str) -> Any:
    try:
        from open_webui.internal.db import get_async_db_context
        from open_webui.models.functions import Function
        from sqlalchemy import select

        async with get_async_db_context() as db:
            result = await db.execute(select(Function.valves).filter_by(id=pipe_id))
            return result.scalar_one_or_none()
    except Exception:
        logger.debug(
            "pipe_dashboard: raw valve column unavailable; cannot tell an unset row "
            "from one this server cannot decrypt",
            exc_info=True,
        )
        return None


def _raw_column_decodes(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return True
    secret = os.getenv("WEBUI_SECRET_KEY")
    if not secret:
        return True
    key = secret.encode()
    if len(secret) != 44:
        key = base64.urlsafe_b64encode(hashlib.sha256(key).digest())
    try:
        Fernet(key).decrypt(raw.encode())
    except (InvalidToken, ValueError, TypeError):
        logger.warning(
            "pipe_dashboard: the stored configuration did not decode under the current "
            "WEBUI_SECRET_KEY (a rotated key does this); the Config tab will refuse to "
            "show or write it rather than serve factory defaults over it"
        )
        return False
    return True


async def stored_row_readable(pipe_id: str, stored: Any) -> tuple[bool, str]:
    if stored is None:
        return False, "the stored configuration could not be read from the database"
    if stored == {} and not _raw_column_decodes(await _raw_valve_column(pipe_id)):
        return False, (
            "the stored configuration could not be read from the database: it is "
            "encrypted with a different WEBUI_SECRET_KEY"
        )
    if not isinstance(stored, dict):
        return False, "the stored configuration could not be read from the database"
    return True, ""


def merge_for_save_with_drops(
    valves_cls: type, current: dict[str, Any], edits: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    stored, dropped = readable_stored(valves_cls, current)
    if dropped:
        logger.warning(
            "pipe_dashboard: dropped stored valves the current schema rejects: %s",
            ", ".join(dropped),
        )
    merged = dict(stored)
    for key, value in edits.items():
        fld = valves_cls.model_fields.get(key)
        if fld is None:
            continue
        if is_secret(fld.annotation):
            if _is_clear_edit(fld, value, merged):
                merged.pop(key, None)
                continue
            if value == "":
                continue
        _, nullable = _base_type(fld.annotation)
        if nullable and isinstance(value, str) and not value.strip():
            merged[key] = None
            continue
        _valve_schema(valves_cls)(**{key: value})
        merged[key] = value
    full = valves_cls(**merged).model_dump()
    defaults = valves_cls().model_dump()
    out: dict[str, Any] = {}
    for name, fld in valves_cls.model_fields.items():
        if name not in full:
            continue
        if is_secret(fld.annotation):
            stored_value = str(full.get(name) or "")
            plain = EncryptedStr.decrypt(stored_value)
            default_plain = EncryptedStr.decrypt(str(defaults.get(name) or ""))
            unreadable_encrypted = EncryptedStr.is_unreadable(stored_value)
            if unreadable_encrypted or (plain and plain != default_plain):
                out[name] = full[name]
        elif full.get(name) != defaults.get(name):
            out[name] = full[name]
    return out, dropped


def merge_for_save(valves_cls: type, current: dict[str, Any], edits: dict[str, Any]) -> dict[str, Any]:
    return merge_for_save_with_drops(valves_cls, current, edits)[0]
