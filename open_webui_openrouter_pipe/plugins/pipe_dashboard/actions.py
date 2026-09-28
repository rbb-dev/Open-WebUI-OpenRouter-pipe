"""Action registry + dispatcher for the pipe_dashboard HTTP action route.

Transport-agnostic: no FastAPI/OWUI imports. Authorization is delegated to
authz.can_view/can_act (read vs write per action). Every terminal outcome is
audited; write outcomes include args + client_ip.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from .authz import can_act, can_view
from .config_service import (
    _ClientMessage,
    describe_valves,
    drift,
    is_secret,
    json_safe,
    merge_for_save_with_drops,
    readable_stored,
    stored_row_readable,
)
from .dashboard_socket import emit_config_changed
from .update_service import UpdateError

logger = logging.getLogger(__name__)

_PD_ACTION_MIN_INTERVAL = 1.0
_rate_state: dict[tuple[str, str], float] = {}
_config_write_locks: dict[tuple[str, int], asyncio.Lock] = {}


class _OptionalKey(NamedTuple):
    types: type | tuple[type, ...]


def optional(types: type | tuple[type, ...]) -> _OptionalKey:
    return _OptionalKey(types)


SchemaValue = type | tuple[type, ...] | _OptionalKey


@dataclass
class ActionEntry:
    name: str
    permission: str
    schema: Mapping[str, SchemaValue] | None
    handler: Callable[..., Awaitable[dict[str, Any]]]
    needs_request: bool = False
    admin_only: bool = False


ACTIONS: dict[str, ActionEntry] = {}


def register_action(
    name: str,
    *,
    permission: str = "write",
    schema: Mapping[str, SchemaValue] | None = None,
    needs_request: bool = False,
    admin_only: bool = False,
):
    def deco(fn):
        ACTIONS[name] = ActionEntry(
            name=name, permission=permission, schema=schema, handler=fn,
            needs_request=needs_request, admin_only=admin_only,
        )
        return fn

    return deco


def _validate(args: Any, schema: Mapping[str, SchemaValue] | None) -> tuple[bool, str]:
    if schema is None:
        return True, ""
    if not isinstance(args, dict):
        return False, "args must be an object"
    for key, typ in schema.items():
        if isinstance(typ, _OptionalKey):
            if key in args and not isinstance(args[key], typ.types):
                return False, f"missing or invalid: {key}"
            continue
        if key not in args or not isinstance(args[key], typ):
            return False, f"missing or invalid: {key}"
    return True, ""


def _rate_limited(user_id: str, name: str) -> bool:
    now = time.monotonic()
    key = (user_id, name)
    last = _rate_state.get(key, 0.0)
    if now - last < _PD_ACTION_MIN_INTERVAL:
        return True
    _rate_state[key] = now
    return False


def _scrub(value: Any, limit: int = 200) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")[:limit]


def _secret_fields(pipe: Any) -> Mapping[str, Any]:
    fields = getattr(type(getattr(pipe, "valves", None)), "model_fields", None)
    if not fields:
        try:
            from ...pipe import Pipe

            fields = Pipe.Valves.model_fields
        except Exception:  # noqa: BLE001
            return {}
    return fields


def _is_secret_key(key: Any, fields: Mapping[str, Any]) -> bool:
    field = fields.get(key) if isinstance(key, str) else None
    return field is not None and is_secret(field.annotation)


def _is_valve_key(key: Any, fields: Mapping[str, Any]) -> bool:
    return isinstance(key, str) and key in fields


def _protocol_keys(entry: ActionEntry | None) -> frozenset[str]:
    if entry is not None and entry.schema:
        return frozenset(entry.schema)
    keys: set[str] = set()
    for candidate in ACTIONS.values():
        if candidate.schema:
            keys |= set(candidate.schema)
    return frozenset(keys)


def _marker(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "<redacted list>"
    if isinstance(value, Mapping):
        return "<redacted dict>"
    return f"<redacted {type(value).__name__}>"


def _mask(
    key: Any,
    value: Any,
    fields: Mapping[str, Any],
    protocol: frozenset[str],
    depth: int = 0,
) -> Any:
    if _is_secret_key(key, fields):
        return "<redacted>"
    vouched = _is_valve_key(key, fields) or (isinstance(key, str) and key in protocol)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if depth >= 32:
            return value
        if not vouched:
            return _marker(value)
        if isinstance(value, Mapping):
            return {k: _mask(k, v, fields, protocol, depth + 1) for k, v in value.items()}
        return [_marker(v) for v in value]
    if vouched:
        return value
    return _marker(value)


def _redacted_args(pipe: Any, args: Any, entry: ActionEntry | None = None) -> Any:
    if not isinstance(args, dict):
        return args
    fields = _secret_fields(pipe)
    if not fields:
        return args
    protocol = _protocol_keys(entry)
    return {k: _mask(k, v, fields, protocol) for k, v in args.items()}


def _audit(user: Any, name: str, outcome: str, client_ip: Any, args: Any = None) -> None:
    uid = getattr(user, "id", None)
    level = logger.debug if outcome in ("ok", "disabled") else logger.warning
    level(
        "pipe_dashboard action user=%s action=%s outcome=%s ip=%s args=%s",
        _scrub(uid), _scrub(name), outcome, _scrub(client_ip),
        _scrub(args) if args is not None else "-",
    )


def _dashboard_enabled(pipe: Any) -> bool:
    valves = getattr(pipe, "valves", None)
    if valves is None or not hasattr(valves, "PIPE_DASHBOARD_ENABLE"):
        return True
    return bool(valves.PIPE_DASHBOARD_ENABLE)


async def dispatch_action(
    pipe: Any, user: Any, name: str, args: Any, *, client_ip: Any = None, request: Any = None
) -> tuple[int, dict[str, Any]]:
    entry = ACTIONS.get(name)
    required = entry.permission if entry else "read"
    allowed = await (can_act if required == "write" else can_view)(user, pipe)
    if not allowed:
        _audit(user, name, "forbidden", client_ip)
        return 403, {"error": "forbidden"}
    if entry is not None and entry.admin_only and getattr(user, "role", None) != "admin":
        _audit(user, name, "forbidden", client_ip)
        return 403, {"error": "forbidden"}
    if entry is None:
        _audit(user, name, "unknown", client_ip)
        return 404, {"error": "unknown action"}
    ok, err = _validate(args, entry.schema)
    if not ok:
        _audit(user, name, "bad_args", client_ip)
        return 400, {"error": err}
    if _rate_limited(getattr(user, "id", ""), name):
        _audit(user, name, "rate_limited", client_ip)
        return 429, {"error": "rate limited"}
    if entry.needs_request and request is None:
        _audit(user, name, "bad_args", client_ip)
        return 400, {"error": "request unavailable"}
    if not _dashboard_enabled(pipe):
        _audit(user, name, "disabled", client_ip)
        return 404, {"error": "unknown action"}
    write = entry.permission == "write"
    try:
        if entry.needs_request:
            result = await entry.handler(pipe, user, args, request=request)
        else:
            result = await entry.handler(pipe, user, args)
    except Exception as exc:  # noqa: BLE001
        cause = str(exc).strip() if isinstance(exc, _ClientMessage) else type(exc).__name__
        logger.warning("pipe_dashboard action %s failed: %s", name, cause)
        _audit(user, name, "error", client_ip, args=_redacted_args(pipe, args, entry) if write else None)
        return 500, {"error": "action failed", "detail": cause}
    _audit(user, name, "ok", client_ip, args=_redacted_args(pipe, args, entry) if write else None)
    return 200, {"ok": True, "result": result}


@register_action("whoami", permission="read", schema=None)
async def _whoami(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    return {
        "user_id": getattr(user, "id", None),
        "role": getattr(user, "role", None),
        "can_view": await can_view(user, pipe),
        "can_act": await can_act(user, pipe),
    }


@register_action("echo", permission="write", schema={"message": str})
async def _echo(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    return {"message": args["message"]}


def _pipe_dashboard_plugin(pipe: Any) -> Any:
    registry = getattr(pipe, "_plugin_registry", None)
    for plugin in getattr(registry, "_plugins", []) or []:
        if getattr(plugin, "plugin_id", "") == "pipe-dashboard":
            return plugin
    return None


@register_action(
    "usage_stats",
    permission="read",
    schema={"range": str, "tz_offset_min": int, "include_tasks": bool},
)
async def _usage_stats(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    plugin = _pipe_dashboard_plugin(pipe)
    if plugin is None:
        return {"available": False, "reason": "plugin unavailable"}
    from .usage_queries import run_usage_query

    return await run_usage_query(plugin, pipe, args)


async def _current_config_rev(pipe: Any) -> Any:
    """Return the function's stored ``updated_at``, or None."""
    try:
        from open_webui.models.functions import Functions

        function = await Functions.get_function_by_id(getattr(pipe, "id", ""))
        if function is None:
            logger.warning(
                "pipe_dashboard: the stored function row could not be read, so the "
                "config revision is unknown; concurrent-edit protection is unavailable"
            )
            return None
        return getattr(function, "updated_at", None)
    except Exception:
        logger.warning(
            "pipe_dashboard: could not read the stored config revision; "
            "concurrent-edit protection is unavailable for this call",
            exc_info=True,
        )
        return None


async def _read_stored_valves(pipe_id: str) -> tuple[dict[str, Any] | None, bool]:
    from ...core.utils import _await_if_needed

    stored: Any = None
    try:
        from open_webui.models.functions import Functions

        stored = await _await_if_needed(Functions.get_function_valves_by_id(pipe_id))
    except Exception:
        logger.warning(
            "pipe_dashboard: stored valve read failed; the config view is showing "
            "in-memory values instead of the persisted ones",
            exc_info=True,
        )
        return None, False
    return stored, True


async def _effective_valves_and_state(
    pipe: Any,
) -> tuple[Any, list[str], dict[str, Any] | None, bool]:
    stored, read_ok = await _read_stored_valves(getattr(pipe, "id", ""))
    valves_cls = type(pipe.valves)
    if not read_ok:
        return pipe.valves, [], None, False
    if stored is None:
        logger.warning(
            "pipe_dashboard: the stored valve set could not be read, so the config view "
            "is showing in-memory values rather than the persisted ones, and the Config "
            "tab will not save over them"
        )
        return pipe.valves, [], None, False
    readable, reason = await stored_row_readable(getattr(pipe, "id", ""), stored)
    if not readable:
        logger.warning(
            "pipe_dashboard: the stored valve set could not be read, so the config view "
            "refuses to show a current state it cannot confirm and the Config tab will "
            "not save over it (%s)",
            reason,
        )
        return valves_cls(), [], stored, False
    if not stored:
        return valves_cls(), [], {}, True
    kept, dropped = readable_stored(valves_cls, stored)
    if dropped:
        logger.warning(
            "pipe_dashboard: the persisted valve set does not validate against the "
            "current schema; %s cannot be shown or saved and reads as its default",
            ", ".join(dropped),
        )
    return valves_cls(**kept), dropped, stored, True


async def _effective_valves_and_drops(pipe: Any) -> tuple[Any, list[str]]:
    valves, dropped, _stored, _read_ok = await _effective_valves_and_state(pipe)
    return valves, dropped


async def _effective_valves(pipe: Any) -> Any:
    return (await _effective_valves_and_drops(pipe))[0]


def _config_snapshot(
    valves: Any, stored: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Valve specs with current values; secret values masked to None."""
    valves_cls = type(valves)
    specs = describe_valves(valves_cls)
    for spec in specs:
        name = spec["name"]
        if spec["secret"]:
            spec["value"] = None
            spec["secret_set"] = bool(str(getattr(valves, name, "") or ""))
            spec["secret_stored"] = bool(stored is not None and name in stored)
        else:
            spec["value"] = json_safe(getattr(valves, name, None))
    return {"valves": specs, "drift": drift(valves_cls)}


async def _saved_values(pipe: Any, names: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    wanted = set(names)
    try:
        effective, reset, stored, _read_ok = await _effective_valves_and_state(pipe)
    except _ClientMessage:
        logger.warning(
            "pipe_dashboard: the store became unreadable while echoing a completed save; "
            "the write is committed, so the echo is dropped rather than reported as a failure"
        )
        return {}, []
    snapshot = _config_snapshot(effective, stored)
    return {
        spec["name"]: spec["value"]
        for spec in snapshot["valves"]
        if spec["name"] in wanted and not spec["secret"]
    }, list(reset)


@register_action("config_get", permission="read", schema=None, admin_only=True)
async def _config_get(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    effective, dropped, stored, read_ok = await _effective_valves_and_state(pipe)
    snapshot = _config_snapshot(effective, stored)
    snapshot["reset"] = dropped
    snapshot["rev"] = await _current_config_rev(pipe)
    snapshot["config_unreadable"] = not read_ok
    return snapshot


@register_action(
    "config_set",
    permission="write",
    schema={"edits": dict, "rev": optional((int, str, type(None)))},
    admin_only=True,
)
async def _config_set(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    """Merge edits into the stored custom subset (not the live model) and persist; rev-guarded."""
    async with _config_write_lock(getattr(pipe, "id", "")):
        result, committed = await _persist_config_edit(pipe, user, args)
    if committed:
        result["values"], result["post_reset"] = await _saved_values(pipe, args["edits"])
    return result


def _config_write_lock(pipe_id: str) -> asyncio.Lock:
    key = (pipe_id, id(asyncio.get_running_loop()))
    lock = _config_write_locks.get(key)
    if lock is None:
        lock = _config_write_locks[key] = asyncio.Lock()
    return lock


async def _persist_config_edit(pipe: Any, user: Any, args: Any) -> tuple[dict[str, Any], bool]:
    current_rev = await _current_config_rev(pipe)
    client_rev = args.get("rev")
    # An unreadable revision is a conflict on its own, independent of what the caller
    # sent. Open WebUI's get_function_by_id catches its own DB errors and returns None,
    # so the except arm below can never fire for a real fault -- and config_get then
    # hands the client rev: null, which it echoes back, making `client_rev is not None`
    # False and letting the write through with no concurrency check at all.
    if current_rev is None or (client_rev is not None and client_rev != current_rev):
        effective, _dropped, stored, conflict_read_ok = await _effective_valves_and_state(pipe)
        if not conflict_read_ok and stored is None:
            return {
                "unreadable": "the stored configuration could not be read from the database",
                "rev": current_rev,
            }, False
        stale = _config_snapshot(effective, stored)
        stale["conflict"] = True
        stale["rev"] = current_rev
        stale["config_unreadable"] = not conflict_read_ok
        return stale, False
    edits = args["edits"]
    if not edits:
        return {"saved": 0, "rev": current_rev}, False
    from open_webui.models.functions import Functions

    current = await Functions.get_function_valves_by_id(getattr(pipe, "id", ""))
    readable, reason = await stored_row_readable(getattr(pipe, "id", ""), current)
    if not readable or current is None:
        if current is None:
            return {
                "unreadable": reason
                or "the stored configuration could not be read from the database",
                "rev": current_rev,
            }, False
        refused = _config_snapshot(pipe.valves)
        refused["conflict"] = True
        refused["rev"] = current_rev
        refused["config_unreadable"] = True
        return refused, False
    _stored, stored_read_ok = await _read_stored_valves(getattr(pipe, "id", ""))
    if not stored_read_ok:
        return {
            "unreadable": "the stored configuration could not be read from the database",
            "rev": current_rev,
        }, False
    to_save, dropped, not_saved, cleared = merge_for_save_with_drops(
        type(pipe.valves), current, edits
    )
    result = await Functions.update_function_valves_by_id(getattr(pipe, "id", ""), to_save)
    if result is None:
        raise RuntimeError("valve update rejected by store")
    rev = getattr(result, "updated_at", None)
    await emit_config_changed(rev)
    return {
        "saved": len(edits) - len(not_saved - cleared),
        "not_saved": sorted(not_saved - cleared),
        "rev": rev,
        "reset": dropped,
        "post_reset": [],
        "values": {},
    }, True


def _update_service_of(pipe: Any) -> Any:
    plugin = _pipe_dashboard_plugin(pipe)
    return getattr(plugin, "update_service", None) if plugin is not None else None


async def _update_enabled(pipe: Any) -> tuple[bool, str]:
    """Gate on the PERSISTED valve, not the in-memory copy (which lags on idle workers)."""
    svc = _update_service_of(pipe)
    if svc is None:
        return bool(
            getattr(getattr(pipe, "valves", None), "PIPE_DASHBOARD_UPDATE_ENABLE", True)
        ), "disabled"
    try:
        valves, stored_read_ok = await svc._row_valves_checked()
        if not stored_read_ok:
            logger.warning(
                "pipe_dashboard: the persisted PIPE_DASHBOARD_UPDATE_ENABLE valve is "
                "unreadable; refusing update actions rather than falling back to the "
                "in-memory copy, which would let a failed read override an operator's "
                "disable"
            )
            return False, "valve_unreadable"
        return bool(valves.get("PIPE_DASHBOARD_UPDATE_ENABLE", True)), "disabled"
    except Exception:
        logger.warning(
            "pipe_dashboard: cannot read the persisted PIPE_DASHBOARD_UPDATE_ENABLE valve; "
            "refusing update actions until it can be confirmed",
            exc_info=True,
        )
        return False, "valve_unreadable"


async def _run_update_call(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await coro
    except UpdateError as exc:
        result: dict[str, Any] = {"error": exc.code, "message": exc.message}
        reset = str(getattr(exc, "reset", "") or "")
        if reset:
            result["reset"] = reset
        return result


@register_action("update_check", permission="read", schema={"force": optional(bool)})
async def _update_check(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    enabled, reason = await _update_enabled(pipe)
    if not enabled:
        return {"enabled": False, "reason": reason}
    svc = _update_service_of(pipe)
    if svc is None:
        return {"error": "unavailable", "message": "update service not initialized"}
    force = bool(args.get("force", False)) and getattr(user, "role", None) == "admin"
    return await _run_update_call(svc.check(force=force))


@register_action(
    "update_apply",
    permission="write",
    schema={"rev": (int, str), "compressed": optional(bool)},
    needs_request=True,
    admin_only=True,
)
async def _update_apply(pipe: Any, user: Any, args: Any, request: Any = None) -> dict[str, Any]:
    enabled, reason = await _update_enabled(pipe)
    if not enabled:
        return {"error": reason}
    svc = _update_service_of(pipe)
    if svc is None:
        return {"error": "unavailable", "message": "update service not initialized"}
    actor = str(getattr(user, "id", "") or "admin")
    return await _run_update_call(
        svc.apply(dict(args), actor=actor, actor_id=actor, request=request)
    )


@register_action(
    "update_restore",
    permission="write",
    schema={"file_id": str, "rev": (int, str)},
    needs_request=True,
    admin_only=True,
)
async def _update_restore(pipe: Any, user: Any, args: Any, request: Any = None) -> dict[str, Any]:
    enabled, reason = await _update_enabled(pipe)
    if not enabled:
        return {"error": reason}
    svc = _update_service_of(pipe)
    if svc is None:
        return {"error": "unavailable", "message": "update service not initialized"}
    actor = str(getattr(user, "id", "") or "admin")
    return await _run_update_call(
        svc.restore(dict(args), actor=actor, actor_id=actor, request=request)
    )


@register_action(
    "update_snapshot_delete",
    permission="write",
    schema={"file_id": str, "sha256": str},
    admin_only=True,
)
async def _update_snapshot_delete(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    enabled, reason = await _update_enabled(pipe)
    if not enabled:
        return {"error": reason}
    svc = _update_service_of(pipe)
    if svc is None:
        return {"error": "unavailable", "message": "update service not initialized"}
    return await _run_update_call(svc.snapshot_delete(dict(args)))
