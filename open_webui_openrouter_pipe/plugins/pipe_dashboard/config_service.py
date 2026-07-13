"""Introspect valves and overlay enrichment (title/group/help) for the Config tab."""

from __future__ import annotations

import typing
from typing import Any

import annotated_types as at

from ...core.config import EncryptedStr
from .config_meta import CONFIG_META

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
                "is_template": name.endswith("_TEMPLATE"),
                "default": None if secret else json_safe(fld.get_default(call_default_factory=True)),
            }
        )
    return specs


def drift(valves_cls: type) -> dict[str, list[str]]:
    """Enrichment drift: valves with no entry, and entries with no valve."""
    live = set(valves_cls.model_fields)
    mapped = set(CONFIG_META)
    return {"unenriched": sorted(live - mapped), "orphaned": sorted(mapped - live)}


def merge_for_save(valves_cls: type, current: dict[str, Any], edits: dict[str, Any]) -> dict[str, Any]:
    """Return only the custom subset (fields differing from default) to persist, validating edits over current."""
    merged = dict(current)
    for key, value in edits.items():
        fld = valves_cls.model_fields.get(key)
        if fld is None:
            continue
        if is_secret(fld.annotation) and (value is None or value == ""):
            continue
        merged[key] = value
    full = valves_cls(**merged).model_dump()
    defaults = valves_cls().model_dump()
    out: dict[str, Any] = {}
    for name, fld in valves_cls.model_fields.items():
        if name not in full:
            continue
        if is_secret(fld.annotation):
            plain = EncryptedStr.decrypt(str(full.get(name) or ""))
            default_plain = EncryptedStr.decrypt(str(defaults.get(name) or ""))
            if plain and plain != default_plain:
                out[name] = full[name]
        elif full.get(name) != defaults.get(name):
            out[name] = full[name]
    return out
