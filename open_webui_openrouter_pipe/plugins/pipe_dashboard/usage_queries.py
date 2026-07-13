"""Range analytics over the usage table for the Usage tab."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable

from ...storage.persistence import _db_session

USAGE_RANGES: dict[str, tuple[int, int]] = {
    "1h": (3600, 60),
    "6h": (21600, 120),
    "24h": (86400, 300),
    "7d": (604800, 3600),
    "30d": (2592000, 14400),
}

_UQ_MEMO: dict[tuple[str, bool, int], tuple[float, dict[str, Any]]] = {}
_UQ_MEMO_TTL = 30.0
_UQ_MEMO_MAX = 256


def _new_acc() -> dict[str, float]:
    return {
        "sessions": 0, "failed": 0, "cancelled": 0, "retried": 0,
        "tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0, "tokens_cached": 0,
        "cost": 0.0, "task_cost": 0.0, "tools": 0, "tools_failed": 0, "savings": 0.0,
    }


def _add(acc: dict[str, float], r: Any, is_task: bool) -> None:
    if not is_task:
        acc["sessions"] += 1
        if r.status == "failed":
            acc["failed"] += 1
        elif r.status == "cancelled":
            acc["cancelled"] += 1
        if (r.retries or 0) > 0:
            acc["retried"] += 1
    acc["tokens_in"] += r.tokens_in or 0
    acc["tokens_out"] += r.tokens_out or 0
    acc["tokens_reasoning"] += r.tokens_reasoning or 0
    acc["tokens_cached"] += r.tokens_cached or 0
    acc["cost"] += r.cost or 0.0
    if is_task:
        acc["task_cost"] += r.cost or 0.0
    acc["tools"] += (r.tools_ok or 0) + (r.tools_failed or 0)
    acc["tools_failed"] += r.tools_failed or 0
    acc["savings"] += r.cache_savings or 0.0


def _cards(acc: dict[str, float]) -> dict[str, Any]:
    sessions = int(acc["sessions"])
    tin = int(acc["tokens_in"])
    return {
        "sessions": {
            "count": sessions,
            "failed": int(acc["failed"]),
            "cancelled": int(acc["cancelled"]),
            "retried": int(acc["retried"]),
        },
        "tokens": {
            "total": tin + int(acc["tokens_out"]),
            "input": tin,
            "cached": int(acc["tokens_cached"]),
            "output": int(acc["tokens_out"]),
            "reasoning": int(acc["tokens_reasoning"]),
        },
        "cost": {
            "total": round(acc["cost"], 6),
            "avg_per_session": round(acc["cost"] / sessions, 6) if sessions else 0.0,
            "task_portion": round(acc["task_cost"], 6),
        },
        "tools": {"count": int(acc["tools"]), "failed": int(acc["tools_failed"])},
        "errors": {"rate": round(acc["failed"] / sessions, 4) if sessions else 0.0},
        "cached": {
            "pct": round(acc["tokens_cached"] / tin, 4) if tin else 0.0,
            "savings": round(acc["savings"], 6),
        },
    }


def query_usage_stats(
    model: Any,
    session_factory: Any,
    *,
    now: float,
    range_key: str,
    tz_offset_min: int,
    include_tasks: bool,
    name_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Synchronous aggregation — call inside the store's DB executor."""
    import datetime

    span, bucket_s = USAGE_RANGES[range_key]
    start = now - span
    prev_start = start - span
    off = int(tz_offset_min) * 60

    with _db_session(session_factory) as session:
        from sqlalchemy import case, func

        totals_q = session.query(
            func.sum(case((model.kind == "task", 0), else_=1)), func.min(model.ts),
            func.sum(model.tokens_in), func.sum(model.tokens_cached), func.sum(model.tokens_out),
            func.sum(model.tools_ok + model.tools_failed), func.sum(model.cost),
        )
        if not include_tasks:
            totals_q = totals_q.filter(model.kind != "task")
        total_count, min_ts, tot_tin, tot_tcached, tot_tout, tot_tools, tot_cost = totals_q.one()

        rows = (
            session.query(model)
            .filter(model.ts >= datetime.datetime.fromtimestamp(prev_start))
            .all()
        )

    cur = _new_acc()
    prev = _new_acc()
    buckets: dict[int, dict[str, float]] = {}
    by_model: dict[tuple[str, bool], dict[str, Any]] = {}
    by_user: dict[str, dict[str, Any]] = {}

    for r in rows:
        try:
            ts = r.ts.timestamp()
        except Exception:
            continue
        is_task = (r.kind or "chat") == "task"
        if is_task and not include_tasks:
            continue
        acc = cur if ts >= start else prev
        _add(acc, r, is_task)
        if acc is not cur:
            continue

        b = int((ts + off) // bucket_s * bucket_s - off)
        bucket = buckets.setdefault(b, {"tokens": 0, "cost": 0.0, "sessions": 0, "tools": 0})
        bucket["tokens"] += (r.tokens_in or 0) + (r.tokens_out or 0)
        bucket["cost"] += r.cost or 0.0
        bucket["sessions"] += 0 if is_task else 1
        bucket["tools"] += (r.tools_ok or 0) + (r.tools_failed or 0)

        uid = r.user_id or "?"
        user = by_user.setdefault(uid, {
            "user_id": uid, "user_name": r.user_name or "?", "sessions": 0,
            "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0, "tools": 0, "tools_failed": 0,
            "cost": 0.0, "last_active": 0.0,
        })
        user["sessions"] += 0 if is_task else 1
        user["tokens_in"] += r.tokens_in or 0
        user["tokens_cached"] += r.tokens_cached or 0
        user["tokens_out"] += r.tokens_out or 0
        user["tools"] += (r.tools_ok or 0) + (r.tools_failed or 0)
        user["tools_failed"] += r.tools_failed or 0
        user["cost"] += r.cost or 0.0
        user["last_active"] = max(user["last_active"], ts)
        if r.user_name:
            user["user_name"] = r.user_name

        _model_add(by_model, (r.model_id or "?", is_task), r, count_session=True)

    total_cost_window = cur["cost"] or 0.0
    model_rows = []
    for (mid, synthetic), agg in by_model.items():
        name = name_fn(mid) if name_fn else mid
        if synthetic:
            name = f"{name} (tasks)"
        sessions = agg["sessions"]
        model_rows.append({
            "model_id": mid,
            "model_name": name,
            "sessions": sessions,
            "tokens_in": agg["tokens_in"],
            "tokens_cached": agg["tokens_cached"],
            "tokens_out": agg["tokens_out"],
            "tools": agg["tools"],
            "tools_failed": agg["tools_failed"],
            "cost": round(agg["cost"], 6),
            "avg_cost": round(agg["cost"] / sessions, 6) if sessions else 0.0,
            "share_pct": round(agg["cost"] / total_cost_window * 100, 1) if total_cost_window else 0.0,
        })
    model_rows.sort(key=lambda row: -row["cost"])

    user_out = [{
        "user_id": u["user_id"],
        "user_name": u["user_name"],
        "sessions": u["sessions"],
        "tokens_in": u["tokens_in"],
        "tokens_cached": u["tokens_cached"],
        "tokens_out": u["tokens_out"],
        "tools": u["tools"],
        "tools_failed": u["tools_failed"],
        "cost": round(u["cost"], 6),
        "last_active": int(u["last_active"]) or None,
    } for u in sorted(by_user.values(), key=lambda u: -u["cost"])]

    bucket_rows = [
        {"t": t, "tokens": int(v["tokens"]), "cost": round(v["cost"], 6),
         "sessions": int(v["sessions"]), "tools": int(v["tools"])}
        for t, v in sorted(buckets.items())
    ]

    since = None
    try:
        since = int(min_ts.timestamp()) if min_ts is not None else None
    except Exception:
        since = None
    have_prev = since is not None and since <= int(prev_start)

    return {
        "available": True,
        "cards": _cards(cur),
        "prev": _cards(prev) if have_prev else None,
        "buckets": bucket_rows,
        "by_model": model_rows,
        "by_user": user_out,
        "totals": {
            "sessions": int(total_count or 0),
            "tools": int(tot_tools or 0),
            "tokens_in": int(tot_tin or 0),
            "tokens_cached": int(tot_tcached or 0),
            "tokens_out": int(tot_tout or 0),
            "cost": round(float(tot_cost or 0.0), 6),
        },
        "meta": {
            "range": range_key,
            "bucket_s": bucket_s,
            "start": int(start),
            "now": int(now),
            "since": since,
            "include_tasks": include_tasks,
        },
    }


def _model_add(by_model: dict, key: tuple[str, bool], r: Any, *, count_session: bool) -> None:
    agg = by_model.setdefault(key, {
        "sessions": 0, "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0,
        "tools": 0, "tools_failed": 0, "cost": 0.0,
    })
    if count_session:
        agg["sessions"] += 1
    agg["tokens_in"] += r.tokens_in or 0
    agg["tokens_cached"] += r.tokens_cached or 0
    agg["tokens_out"] += r.tokens_out or 0
    agg["tools"] += (r.tools_ok or 0) + (r.tools_failed or 0)
    agg["tools_failed"] += r.tools_failed or 0
    agg["cost"] += r.cost or 0.0


def _warm_usage_store(store: Any, usage_store: Any, valves: Any, pipe_id: str) -> str | None:
    """Initialize a cold artifact store + usage model off-loop; return the failure type name or None."""
    try:
        if getattr(store, "_session_factory", None) is None:
            store._ensure_artifact_store(valves, pipe_id)
        usage_store.ensure(store)
        return None
    except Exception as exc:
        return type(exc).__name__


async def run_usage_query(plugin: Any, pipe: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Async front: validate, memoize (30s), ensure the table, run in executor."""
    range_key = str(args.get("range") or "24h")
    include_tasks = bool(args.get("include_tasks", True))
    tz_offset_min = int(args.get("tz_offset_min") or 0)
    tz_offset_min = max(-900, min(900, round(tz_offset_min / 15) * 15))

    valves = plugin.ctx.valves
    collect_on = bool(getattr(valves, "PIPE_DASHBOARD_USAGE_COLLECT", False))
    retention_days = int(getattr(valves, "PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30) or 30)
    base_meta = {"collect_on": collect_on, "retention_days": retention_days, "range": range_key}

    if range_key not in USAGE_RANGES:
        return {"available": False, "reason": "unknown range", "meta": base_meta}
    if USAGE_RANGES[range_key][0] > retention_days * 86400:
        return {"available": False, "reason": "range exceeds retention", "meta": base_meta}

    store = getattr(pipe, "_artifact_store", None)
    usage_store = plugin._usage_store
    if store is None:
        return {"available": False, "reason": "storage unavailable", "meta": base_meta}

    warm_error: str | None = None
    if not usage_store.enabled or getattr(store, "_session_factory", None) is None:
        loop = asyncio.get_running_loop()
        warm_error = await loop.run_in_executor(
            None, _warm_usage_store, store, usage_store,
            getattr(pipe, "valves", None), getattr(pipe, "id", "") or "",
        )
    if not usage_store.enabled:
        reason = f"storage unavailable ({warm_error})" if warm_error else "storage unavailable"
        return {"available": False, "reason": reason, "meta": base_meta}

    memo_key = (range_key, include_tasks, tz_offset_min)
    now = time.time()
    hit = _UQ_MEMO.get(memo_key)
    if hit is not None and now - hit[0] < _UQ_MEMO_TTL:
        return copy.deepcopy(hit[1])

    executor = getattr(store, "_db_executor", None)
    session_factory = getattr(store, "_session_factory", None)
    if executor is None or session_factory is None:
        reason = f"storage unavailable ({warm_error})" if warm_error else "storage unavailable"
        return {"available": False, "reason": reason, "meta": base_meta}

    from .plugin import _registry_model_name

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: query_usage_stats(
            usage_store._model,
            session_factory,
            now=now,
            range_key=range_key,
            tz_offset_min=tz_offset_min,
            include_tasks=include_tasks,
            name_fn=_registry_model_name,
        ),
    )
    info = await usage_store.table_info()
    result["meta"].update(base_meta)
    result["meta"]["records"] = info.get("records")
    result["meta"]["approx_bytes"] = info.get("approx_bytes")
    if len(_UQ_MEMO) >= _UQ_MEMO_MAX:
        for stale_key in [k for k, (ts, _) in _UQ_MEMO.items() if now - ts >= _UQ_MEMO_TTL]:
            _UQ_MEMO.pop(stale_key, None)
        if len(_UQ_MEMO) >= _UQ_MEMO_MAX:
            _UQ_MEMO.clear()
    _UQ_MEMO[memo_key] = (now, result)
    return copy.deepcopy(result)
