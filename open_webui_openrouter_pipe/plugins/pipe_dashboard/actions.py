"""Action registry + dispatcher for the pipe_dashboard HTTP action route.

Transport-agnostic: no FastAPI/OWUI imports. Authorization is delegated to
authz.can_view/can_act (read vs write per action). Every terminal outcome is
audited; write outcomes include args + client_ip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .authz import can_act, can_view
from .config_service import describe_valves, drift, json_safe, merge_for_save
from .dashboard_socket import emit_config_changed

_pd_actions_log = logging.getLogger(__name__)

_PD_ACTION_MIN_INTERVAL = 1.0
_rate_state: dict[tuple[str, str], float] = {}


@dataclass
class ActionEntry:
    name: str
    permission: str
    schema: dict[str, type] | None
    handler: Callable[..., Awaitable[dict[str, Any]]]


ACTIONS: dict[str, ActionEntry] = {}


def register_action(name: str, *, permission: str = "write", schema: dict[str, type] | None = None):
    def deco(fn):
        ACTIONS[name] = ActionEntry(name=name, permission=permission, schema=schema, handler=fn)
        return fn

    return deco


def _validate(args: Any, schema: dict[str, type] | None) -> tuple[bool, str]:
    if schema is None:
        return True, ""
    if not isinstance(args, dict):
        return False, "args must be an object"
    for key, typ in schema.items():
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


def _audit(user: Any, name: str, outcome: str, client_ip: Any, args: Any = None) -> None:
    uid = getattr(user, "id", None)
    level = _pd_actions_log.debug if outcome == "ok" else _pd_actions_log.warning
    level(
        "pipe_dashboard action user=%s action=%s outcome=%s ip=%s args=%s",
        _scrub(uid), _scrub(name), outcome, _scrub(client_ip),
        _scrub(args) if args is not None else "-",
    )


async def dispatch_action(pipe: Any, user: Any, name: str, args: Any, *, client_ip: Any = None) -> tuple[int, dict[str, Any]]:
    entry = ACTIONS.get(name)
    required = entry.permission if entry else "read"
    allowed = await (can_act if required == "write" else can_view)(user, pipe)
    if not allowed:
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
    write = entry.permission == "write"
    try:
        result = await entry.handler(pipe, user, args)
    except Exception:
        _audit(user, name, "error", client_ip, args=args if write else None)
        return 500, {"error": "action failed"}
    _audit(user, name, "ok", client_ip, args=args if write else None)
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
        return getattr(function, "updated_at", None)
    except Exception:
        return None


async def _effective_valves(pipe: Any) -> Any:
    """Reconstruct valves from the stored custom subset so reads reflect persisted state."""
    try:
        from open_webui.models.functions import Functions

        stored = await Functions.get_function_valves_by_id(getattr(pipe, "id", ""))
    except Exception:
        return pipe.valves
    valves_cls = type(pipe.valves)
    if not stored:
        return valves_cls()
    try:
        return valves_cls(**{k: v for k, v in stored.items() if v is not None})
    except Exception:
        return pipe.valves


def _config_snapshot(valves: Any) -> dict[str, Any]:
    """Valve specs with current values; secret values masked to None."""
    valves_cls = type(valves)
    specs = describe_valves(valves_cls)
    for spec in specs:
        name = spec["name"]
        if spec["secret"]:
            spec["value"] = None
            spec["secret_set"] = bool(str(getattr(valves, name, "") or ""))
        else:
            spec["value"] = json_safe(getattr(valves, name, None))
    return {"valves": specs, "drift": drift(valves_cls)}


@register_action("config_get", permission="read", schema=None)
async def _config_get(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    effective = await _effective_valves(pipe)
    snapshot = _config_snapshot(effective)
    snapshot["rev"] = await _current_config_rev(pipe)
    return snapshot


@register_action("config_set", permission="write", schema={"edits": dict})
async def _config_set(pipe: Any, user: Any, args: Any) -> dict[str, Any]:
    """Merge edits into the stored custom subset (not the live model) and persist; rev-guarded."""
    current_rev = await _current_config_rev(pipe)
    client_rev = args.get("rev")
    if client_rev is not None and current_rev is not None and client_rev != current_rev:
        effective = await _effective_valves(pipe)
        stale = _config_snapshot(effective)
        stale["conflict"] = True
        stale["rev"] = current_rev
        return stale
    edits = args["edits"]
    if not edits:
        return {"saved": 0, "rev": current_rev}
    from open_webui.models.functions import Functions

    current = await Functions.get_function_valves_by_id(getattr(pipe, "id", ""))
    if current is None:
        raise RuntimeError("could not read current config")
    to_save = merge_for_save(type(pipe.valves), current, edits)
    result = await Functions.update_function_valves_by_id(getattr(pipe, "id", ""), to_save)
    if result is None:
        raise RuntimeError("valve update rejected by store")
    rev = getattr(result, "updated_at", None)
    await emit_config_changed(rev)
    return {"saved": len(edits), "rev": rev}
