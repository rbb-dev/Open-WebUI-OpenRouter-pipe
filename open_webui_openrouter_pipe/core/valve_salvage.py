from __future__ import annotations

import logging
import typing
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from .warn_latch import warn_level

logger = logging.getLogger(__name__)

_VALVE_SCHEMA_CACHE: dict[type, type] = {}
_warned_stale_valves: dict[str, float] = {}
_STALE_VALVES_WARN_EVERY_S = 300.0
_RENAMED_VALVES = {
    "VIDEO_INTENT_MAX_CALLS_PER_CHAT": "VIDEO_INTENT_MAX_TURNS_PER_CHAT",
    "VIDEO_INTENT_MAX_CALLS_PER_USER_DAY": "VIDEO_INTENT_MAX_TURNS_PER_USER_DAY",
}


def carry_renamed_valves(cls: type, values: Any) -> Any:
    if not isinstance(values, Mapping):
        return values
    for old, new in _RENAMED_VALVES.items():
        if new not in cls.model_fields or old not in values:
            continue
        value = values[old]
        carried = {k: v for k, v in values.items() if k != old}
        if value is not None and new not in values:
            carried[new] = value
        logger.log(
            warn_level(_warned_stale_valves, old, cooldown_s=_STALE_VALVES_WARN_EVERY_S),
            "pipe: stored setting %s has been renamed to %s; its value was carried over.",
            old, new,
        )
        values = carried
    return values


def _encrypted_type() -> Any:
    from .config import EncryptedStr

    return EncryptedStr


def _field_annotation(cls: type, name: str) -> Any:
    field = cls.model_fields.get(name)
    return None if field is None else field.annotation


def is_secret_annotation(annotation: Any) -> bool:
    secret = _encrypted_type()
    if annotation is secret:
        return True
    return any(arg is secret for arg in typing.get_args(annotation))


def is_secret_field(cls: type, name: str) -> bool:
    return is_secret_annotation(_field_annotation(cls, name))


def _admits_none(cls: type, name: str) -> bool:
    return type(None) in typing.get_args(_field_annotation(cls, name))


def _secret_as_str(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return "" if value is None else str(value)


def _valve_schema(cls: type) -> type:
    shell = _VALVE_SCHEMA_CACHE.get(cls)
    if shell is None:
        fields: dict[str, Any] = {
            name: (field.annotation, field) for name, field in cls.model_fields.items()
        }
        shell = create_model(
            f"_StoredValveSchema{len(_VALVE_SCHEMA_CACHE)}",
            __base__=BaseModel,
            __config__=typing.cast("typing.Any", dict(cls.model_config)),
            **fields,
        )
        _VALVE_SCHEMA_CACHE[cls] = shell
    return shell


def drop_unvalidatable(cls: type, values: Any) -> Any:
    values = carry_renamed_valves(cls, values)
    if not isinstance(values, Mapping):
        return values
    kept = dict(values)
    unread: list[tuple[str, str]] = []
    blanked: set[str] = set()
    for _ in range(len(kept) + 2):
        try:
            _valve_schema(cls)(**kept)
            break
        except ValidationError as exc:
            names = {str(err["loc"][0]) for err in exc.errors() if err.get("loc")}
            for name in names:
                if not is_secret_field(cls, name) or isinstance(kept.get(name), str):
                    continue
                value = kept.get(name)
                if value is None:
                    kept.pop(name, None)
                    note = "was not set and has been cleared"
                elif (
                    isinstance(value, (list, tuple))
                    and len(value) == 1
                    and isinstance(value[0], str)
                ):
                    kept[name] = _secret_as_str(value)
                    note = "was not text and has been read as the text it holds"
                else:
                    kept[name] = _secret_as_str(value)
                    note = (
                        "was kept and is not a key this release can use; re-enter it "
                        "where you configure the pipe"
                    )
                if all(existing != name for existing, _ in unread):
                    unread.append((name, note))
            bad = {name for name in names if not is_secret_field(cls, name)}
            if not bad:
                break
            for name in bad:
                blank = _admits_none(cls, name) and isinstance(kept.get(name), str) and not kept[name].strip()
                if blank:
                    blanked.add(name)
                kept.pop(name, None)
    for name, note in unread:
        logger.log(
            warn_level(_warned_stale_valves, name, cooldown_s=_STALE_VALVES_WARN_EVERY_S),
            "pipe: stored setting %s %s.",
            name,
            note,
        )
    if len(kept) != len(values):
        stored = values
        for name in sorted(set(values) - set(kept) - {n for n, _ in unread} - blanked):
            default = cls.model_fields[name].get_default(call_default_factory=True)
            shown = "<redacted>" if is_secret_field(cls, name) else stored[name]
            logger.log(
                warn_level(_warned_stale_valves, name, cooldown_s=_STALE_VALVES_WARN_EVERY_S),
                "pipe: stored setting %s (was %r) is not accepted by this release and "
                "was left at its default %r. The pipe keeps running; set it again where "
                "you configure the pipe.",
                name, shown, default,
            )
    for name in sorted(set(values) - set(cls.model_fields)):
        logger.log(
            warn_level(_warned_stale_valves, name, cooldown_s=_STALE_VALVES_WARN_EVERY_S),
            "pipe: stored setting %s is not a setting this release has, and is not used. "
            "The pipe keeps running; nothing is read from it.",
            name,
        )
    return kept
