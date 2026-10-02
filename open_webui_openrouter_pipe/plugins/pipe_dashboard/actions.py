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
from typing import Any, NamedTuple, cast

from pydantic import ValidationError

from .authz import can_act, can_view
from .config_service import (
    _ClientInput,
    _ClientMessage,
    describe_valves,
    drift,
    is_secret,
    json_safe,
    merge_for_save_with_drops,
    persisted_dashboard_enabled,
    readable_stored,
    stored_row_readable,
)
from .dashboard_socket import (
    emit_config_changed,
    publish_valves_changed,
    read_config_rev,
)
from .update_service import UpdateError, UpdateService, _distributed_lock

logger = logging.getLogger(__name__)

_PD_ACTION_MIN_INTERVAL = 1.0
_VALUE_ERROR_PREFIX = "Value error, "
_rate_state: dict[tuple[str, str], float] = {}
_config_write_locks: dict[tuple[str, int], asyncio.Lock] = {}
_PD_CONFIG_LOCK_SUFFIX = "config_lock"
_PD_CONFIG_LOCK_TIMEOUT_S = 30


class _ConfigLease:
    __slots__ = ("lock",)

    def __init__(self, lock: Any) -> None:
        self.lock = lock


async def _acquire_config_lease() -> _ConfigLease | None:
    lock = _distributed_lock(_PD_CONFIG_LOCK_SUFFIX, _PD_CONFIG_LOCK_TIMEOUT_S)
    if lock is None:
        logger.warning(
            "pipe_dashboard: the cross-worker configuration lease is unavailable, so a "
            "save on this worker is guarded by the in-process lock and the revision alone"
        )
        return None
    acquire = getattr(lock, "acquire_lock", None) or getattr(lock, "aquire_lock", None)
    if acquire is None:
        logger.warning(
            "pipe_dashboard: the configuration lease exposes no way to take it, so a save "
            "on this worker is guarded by the in-process lock and the revision alone"
        )
        return None
    try:
        acquired = await asyncio.to_thread(acquire)
    except Exception:
        logger.warning(
            "pipe_dashboard: the configuration lease could not be taken, so a save on this "
            "worker is guarded by the in-process lock and the revision alone",
            exc_info=True,
        )
        return None
    if not acquired:
        UpdateService._dispose_lock(lock)
        raise _ClientMessage(
            "another administrator is saving the configuration on another worker, so nothing "
            "was saved; try again in a moment"
        )
    return _ConfigLease(lock)


async def _release_config_lease(lease: _ConfigLease | None) -> None:
    lock = getattr(lease, "lock", None)
    if lease is None or lock is None:
        return
    lease.lock = None
    try:
        await asyncio.to_thread(lock.release_lock)
    except Exception:
        logger.warning(
            "pipe_dashboard: the cross-worker configuration lease could not be released",
            exc_info=True,
        )
    UpdateService._dispose_lock(lock)


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
    last = _rate_state.get(key)
    if last is not None and 0.0 <= now - last < _PD_ACTION_MIN_INTERVAL:
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


async def _dashboard_enabled(pipe: Any) -> bool:
    return (await persisted_dashboard_enabled(pipe))[0]


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
    if not await _dashboard_enabled(pipe):
        _audit(user, name, "disabled", client_ip)
        return 404, {"error": "dashboard_off"}
    write = entry.permission == "write"
    try:
        if entry.needs_request:
            result = await entry.handler(pipe, user, args, request=request)
        else:
            result = await entry.handler(pipe, user, args)
    except _ClientMessage as exc:
        cause = str(exc).strip()
        status = int(getattr(exc, "status", 500))
        logger.warning("pipe_dashboard action %s failed: %s", name, cause)
        _audit(user, name, "bad_args" if status == 400 else "error", client_ip,
               args=_redacted_args(pipe, args, entry) if write else None)
        return status, {"error": "action failed", "detail": cause}
    except Exception as exc:  # noqa: BLE001
        cause = type(exc).__name__
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
    schema={
        "range": optional(str),
        "tz_offset_min": optional(int),
        "include_tasks": optional(bool),
    },
)
async def _usage_stats(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    plugin = _pipe_dashboard_plugin(pipe)
    if plugin is None:
        return {"available": False, "reason": "plugin unavailable"}
    from .usage_queries import run_usage_query

    return await run_usage_query(plugin, pipe, args)


async def _current_config_rev(pipe: Any) -> Any:
    """Return the function's stored ``updated_at``, or None."""
    rev = await read_config_rev(getattr(pipe, "id", ""))
    if rev is None:
        logger.warning(
            "pipe_dashboard: the stored function row could not be read, so the "
            "config revision is unknown; concurrent-edit protection is unavailable"
        )
    return rev


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
        return pipe.valves, [], stored, False
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


async def _saved_values(
    pipe: Any, names: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted = set(names)
    effective, _reset, stored, read_ok = await _effective_valves_and_state(pipe)
    if not read_ok:
        logger.warning(
            "pipe_dashboard: the store became unreadable while echoing a completed save; "
            "the write is committed, so the echo is dropped rather than reported as a failure"
        )
        return {}, {}
    snapshot = _config_snapshot(effective, stored)
    return (
        {
            spec["name"]: spec["value"]
            for spec in snapshot["valves"]
            if spec["name"] in wanted and not spec["secret"]
        },
        {
            spec["name"]: {
                "set": bool(spec["secret_set"]),
                "stored": bool(spec["secret_stored"]),
            }
            for spec in snapshot["valves"]
            if spec["name"] in wanted and spec["secret"]
        },
    )


@register_action("config_get", permission="read", schema=None, admin_only=True)
async def _config_get(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    effective, dropped, stored, read_ok = await _effective_valves_and_state(pipe)
    snapshot = _config_snapshot(effective, stored)
    snapshot["reset"] = dropped
    snapshot["rev"] = await _current_config_rev(pipe)
    snapshot["config_unreadable"] = not read_ok
    return snapshot


def _save_refusal_message(exc: ValidationError) -> str:
    errors = [err for err in exc.errors() if err.get("loc")]
    names = sorted({str(err["loc"][0]) for err in errors})
    reasons: list[str] = []
    for err in errors:
        if err.get("type") != "value_error":
            continue
        text = str(err.get("msg") or "").removeprefix(_VALUE_ERROR_PREFIX)
        if text and text not in reasons:
            reasons.append(text)
    message = f"the stored settings would not accept: {', '.join(names)}"
    if reasons:
        message = f"{message} - {'; '.join(reasons)}"
    return message


@register_action(
    "config_set",
    permission="write",
    schema={
        "edits": dict,
        "rev": optional((int, str, type(None))),
        "base": optional(dict),
    },
    needs_request=True,
    admin_only=True,
)
async def _config_set(
    pipe: Any, user: Any, args: Any, request: Any = None
) -> dict[str, Any]:
    """Merge edits into the stored custom subset (not the live model) and persist; rev-guarded."""
    lease = await _acquire_config_lease()
    try:
        async with _config_write_lock(getattr(pipe, "id", "")):
            result, committed = await _persist_config_edit(pipe, user, args, request, lease)
            if committed:
                result["values"], result["secrets"] = await _saved_values(
                    pipe, args["edits"]
                )
    finally:
        await _release_config_lease(lease)
    return result


def _config_write_lock(pipe_id: str) -> asyncio.Lock:
    running = asyncio.get_running_loop()
    for key in [
        k
        for k, lock in _config_write_locks.items()
        if k[0] == pipe_id
        and (getattr(lock, "_loop", None) or running).is_closed()
    ]:
        del _config_write_locks[key]
    key = (pipe_id, id(running))
    lock = _config_write_locks.get(key)
    if lock is not None:
        try:
            lock_loop = getattr(cast(Any, lock), "_get_loop", lambda: None)()
            if lock_loop is not running:
                lock = None
        except RuntimeError:
            lock = None
    if lock is None:
        lock = _config_write_locks[key] = asyncio.Lock()
    return lock


def _base_still_holds(valves_cls: type, effective: Any, name: str, base: Any) -> bool:
    if not isinstance(base, dict) or name not in base:
        return False
    fld = valves_cls.model_fields.get(name)
    if fld is not None and is_secret(fld.annotation):
        return bool(base[name]) == bool(str(getattr(effective, name, "") or ""))
    return json_safe(getattr(effective, name, None)) == json_safe(base[name])


async def _write_config_edits(
    pipe: Any,
    user: Any,
    current: dict[str, Any],
    current_rev: Any,
    request: Any,
    edits: dict[str, Any],
    conflicts: list[str],
    refused: dict[str, Any] | None,
    lease: _ConfigLease | None = None,
) -> tuple[dict[str, Any], bool]:
    from open_webui.models.functions import Functions

    readable, _reason = await stored_row_readable(getattr(pipe, "id", ""), current)
    if not readable:
        answer = _config_snapshot(pipe.valves) if refused is None else refused
        answer["conflict"] = True
        answer["rev"] = current_rev
        answer["config_unreadable"] = True
        if conflicts:
            answer["conflicts"] = conflicts
        return answer, False
    try:
        to_save, dropped, not_saved, cleared = merge_for_save_with_drops(
            type(pipe.valves), current, edits
        )
    except ValidationError as exc:
        raise _ClientInput(_save_refusal_message(exc)) from exc
    result = await Functions.update_function_valves_by_id(getattr(pipe, "id", ""), to_save)
    if result is None:
        raise _ClientMessage("the database refused the write, so nothing was saved")
    rev = getattr(result, "updated_at", None)
    await _release_config_lease(lease)
    await emit_config_changed(rev)
    await publish_valves_changed(getattr(pipe, "id", ""), user, request)
    payload: dict[str, Any] = {
        "saved": len(edits) - len(not_saved - cleared),
        "not_saved": sorted(not_saved - cleared),
        "rev": rev,
        "reset": dropped,
        "post_reset": [],
        "values": {},
    }
    if conflicts:
        payload["conflict"] = True
        payload["conflicts"] = conflicts
    return payload, True


async def _persist_base_checked_edit(
    pipe: Any,
    user: Any,
    args: Any,
    request: Any,
    current_rev: Any,
    base: dict[str, Any],
    lease: _ConfigLease | None = None,
) -> tuple[dict[str, Any], bool]:
    effective, _dropped, stored, read_ok = await _effective_valves_and_state(pipe)
    if not read_ok and stored is None:
        return {
            "unreadable": "the stored configuration could not be read from the database",
            "rev": current_rev,
        }, False
    valves_cls = type(pipe.valves)
    conflicts = sorted(
        name for name in args["edits"] if not _base_still_holds(valves_cls, effective, name, base)
    )
    stale = _config_snapshot(effective, stored)
    stale["rev"] = current_rev
    stale["config_unreadable"] = not read_ok
    if len(conflicts) == len(args["edits"]):
        stale["conflict"] = True
        stale["conflicts"] = conflicts
        return stale, False
    allowed = {name: value for name, value in args["edits"].items() if name not in conflicts}
    return await _write_config_edits(
        pipe, user, stored or {}, current_rev, request, allowed, conflicts, stale, lease
    )


async def _persist_config_edit(
    pipe: Any, user: Any, args: Any, request: Any = None, lease: _ConfigLease | None = None
) -> tuple[dict[str, Any], bool]:
    current_rev = await _current_config_rev(pipe)
    client_rev = args.get("rev")
    base = args.get("base")
    stale = current_rev is None or client_rev is None or client_rev != current_rev
    if stale and not isinstance(base, dict):
        effective, _dropped, stored, conflict_read_ok = await _effective_valves_and_state(pipe)
        if not conflict_read_ok and stored is None:
            return {
                "unreadable": "the stored configuration could not be read from the database",
                "rev": current_rev,
            }, False
        stale_payload = _config_snapshot(effective, stored)
        stale_payload["conflict"] = True
        stale_payload["rev"] = current_rev
        stale_payload["config_unreadable"] = not conflict_read_ok
        return stale_payload, False
    if isinstance(base, dict):
        return await _persist_base_checked_edit(
            pipe, user, args, request, current_rev, base, lease
        )
    edits = args["edits"]
    if not edits:
        return {"saved": 0, "rev": current_rev}, False
    current, stored_read_ok = await _read_stored_valves(getattr(pipe, "id", ""))
    if not stored_read_ok or current is None:
        return {
            "unreadable": "the stored configuration could not be read from the database",
            "rev": current_rev,
        }, False
    return await _write_config_edits(
        pipe, user, current, current_rev, request, edits, [], None, lease
    )


def _update_service_of(pipe: Any) -> Any:
    plugin = _pipe_dashboard_plugin(pipe)
    return getattr(plugin, "update_service", None) if plugin is not None else None


async def _update_enabled(pipe: Any) -> tuple[bool, str]:
    """Gate on the PERSISTED valve, not the in-memory copy (which lags on idle workers)."""
    svc = _update_service_of(pipe)
    if svc is None:
        return False, "service_unavailable"
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


_UPDATE_UNAVAILABLE: dict[str, Any] = {
    "error": "unavailable",
    "message": "update service not initialized",
}


def _update_check_refusal(reason: str) -> dict[str, Any]:
    return {"enabled": False, "reason": reason}


def _update_write_refusal(reason: str) -> dict[str, Any]:
    return {"error": reason}


async def _update_gate(
    pipe: Any, refusal: Callable[[str], dict[str, Any]]
) -> tuple[Any, dict[str, Any] | None]:
    svc = _update_service_of(pipe)
    if svc is None:
        return None, dict(_UPDATE_UNAVAILABLE)
    enabled, reason = await _update_enabled(pipe)
    if not enabled:
        return None, refusal(reason)
    return svc, None


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
    svc, refused = await _update_gate(pipe, _update_check_refusal)
    if refused is not None:
        return refused
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
    svc, refused = await _update_gate(pipe, _update_write_refusal)
    if refused is not None:
        return refused
    actor = str(getattr(user, "id", "") or "admin")
    return await _run_update_call(
        svc.apply(dict(args), actor=actor, actor_id=actor, request=request, actor_user=user)
    )


@register_action(
    "update_restore",
    permission="write",
    schema={"file_id": str, "rev": (int, str)},
    needs_request=True,
    admin_only=True,
)
async def _update_restore(pipe: Any, user: Any, args: Any, request: Any = None) -> dict[str, Any]:
    svc, refused = await _update_gate(pipe, _update_write_refusal)
    if refused is not None:
        return refused
    actor = str(getattr(user, "id", "") or "admin")
    return await _run_update_call(
        svc.restore(dict(args), actor=actor, actor_id=actor, request=request, actor_user=user)
    )


@register_action(
    "update_snapshot_delete",
    permission="write",
    schema={"file_id": str, "sha256": str},
    admin_only=True,
)
async def _update_snapshot_delete(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    svc, refused = await _update_gate(pipe, _update_write_refusal)
    if refused is not None:
        return refused
    return await _run_update_call(svc.snapshot_delete(dict(args)))
