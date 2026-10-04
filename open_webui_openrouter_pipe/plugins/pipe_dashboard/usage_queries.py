"""Range analytics over the usage table for the Usage tab."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from typing import Any

from ...core.warn_latch import warn_level
from ...storage.persistence import _db_session
from .usage_store import epoch_from_usage_ts, usage_ts_from_epoch

USAGE_RANGES: dict[str, tuple[int, int]] = {
    "1h": (3600, 60),
    "6h": (21600, 120),
    "24h": (86400, 300),
    "7d": (604800, 3600),
    "30d": (2592000, 14400),
}

_UQ_MEMO: dict[
    tuple[str | None, str, bool, int, bool, bool, int], tuple[float, dict[str, Any]]
] = {}
_UQ_MEMO_TTL = 30.0
_UQ_MEMO_MAX = 256

logger = logging.getLogger(__name__)

_warned_row_timestamps: set[str] = set()


def _new_acc() -> dict[str, float]:
    return {
        "sessions": 0, "failed": 0, "cancelled": 0, "retried": 0,
        "tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0, "tokens_cached": 0,
        "cost": 0.0, "task_cost": 0.0, "tools": 0, "tools_failed": 0,
        "tools_skipped": 0, "savings": 0.0,
    }


def _tools_sum(model: Any) -> Any:
    from sqlalchemy import func

    return func.coalesce(model.tools_ok, 0) + func.coalesce(model.tools_failed, 0) + func.coalesce(model.tools_skipped, 0)


def _int(value: Any) -> int:
    return 0 if value is None else int(value)


def _float(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _is_chat_row(model: Any) -> Any:
    from sqlalchemy import or_

    return or_(model.kind != "task", model.kind.is_(None))


def _counted_rows(model: Any, include_tasks: bool) -> Any:
    from sqlalchemy import true

    return true() if include_tasks else _is_chat_row(model)


def _grouped_key(column: Any) -> Any:
    from sqlalchemy import case, or_

    return case((or_(column.is_(None), column == ""), "?"), else_=column)


def _epoch_or_zero(value: Any) -> float:
    try:
        return epoch_from_usage_ts(value)
    except (AttributeError, OSError, OverflowError, ValueError):
        _level = warn_level(_warned_row_timestamps, "unusable_row_timestamp")
        logger.log(
            _level,
            "usage query: a stored row has an unusable timestamp; the time it reports "
            "is being left out of the answer",
            exc_info=True,
        )
        return 0.0


def _window_cards(model: Any, session: Any, lo: Any, hi: Any, counted: Any) -> dict[str, float]:
    from sqlalchemy import and_, case, func, select

    chat = _is_chat_row(model)
    stmt = select(
        func.sum(case((chat, 1), else_=0)),
        func.sum(case((and_(chat, model.status == "failed"), 1), else_=0)),
        func.sum(case((and_(chat, model.status == "cancelled"), 1), else_=0)),
        func.sum(case((and_(chat, model.retries > 0), 1), else_=0)),
        func.sum(func.coalesce(model.tokens_in, 0)),
        func.sum(func.coalesce(model.tokens_out, 0)),
        func.sum(func.coalesce(model.tokens_reasoning, 0)),
        func.sum(func.coalesce(model.tokens_cached, 0)),
        func.sum(func.coalesce(model.cost, 0.0)),
        func.sum(case((model.kind == "task", func.coalesce(model.cost, 0.0)), else_=0.0)),
        func.sum(_tools_sum(model)),
        func.sum(func.coalesce(model.tools_failed, 0)),
        func.sum(func.coalesce(model.tools_skipped, 0)),
        func.sum(func.coalesce(model.cache_savings, 0.0)),
    ).where(model.ts >= lo, counted)
    if hi is not None:
        stmt = stmt.where(model.ts < hi)
    row = session.execute(stmt).one()
    acc = _new_acc()
    acc["sessions"] = _int(row[0])
    acc["failed"] = _int(row[1])
    acc["cancelled"] = _int(row[2])
    acc["retried"] = _int(row[3])
    acc["tokens_in"] = _int(row[4])
    acc["tokens_out"] = _int(row[5])
    acc["tokens_reasoning"] = _int(row[6])
    acc["tokens_cached"] = _int(row[7])
    acc["cost"] = _float(row[8])
    acc["task_cost"] = _float(row[9])
    acc["tools"] = _int(row[10])
    acc["tools_failed"] = _int(row[11])
    acc["tools_skipped"] = _int(row[12])
    acc["savings"] = _float(row[13])
    return acc


def _bucket_edges(start: float, now: float, bucket_s: int, off: int) -> list[int]:
    first = int((start + off) // bucket_s * bucket_s - off)
    last = int((now + off) // bucket_s * bucket_s - off)
    return list(range(first, last + bucket_s, bucket_s))


def _window_buckets(
    model: Any, session: Any, lo: Any, edges: list[int], bucket_s: int, counted: Any
) -> dict[int, dict[str, float]]:
    from sqlalchemy import and_, case, func, select

    if not edges:
        return {}
    index = case(
        *[
            (
                and_(
                    model.ts >= usage_ts_from_epoch(b),
                    model.ts < usage_ts_from_epoch(b + bucket_s),
                ),
                position,
            )
            for position, b in enumerate(edges)
        ],
        else_=None,
    )
    stmt = (
        select(
            index.label("bucket"),
            func.sum(func.coalesce(model.tokens_in, 0) + func.coalesce(model.tokens_out, 0)),
            func.sum(func.coalesce(model.cost, 0.0)),
            func.sum(case((_is_chat_row(model), 1), else_=0)),
            func.sum(case((and_(_is_chat_row(model), model.status == "failed"), 1), else_=0)),
            func.sum(_tools_sum(model)),
            func.sum(func.coalesce(model.tokens_in, 0)),
            func.sum(func.coalesce(model.tokens_cached, 0)),
        )
        .where(model.ts >= lo, counted)
        .group_by(index)
    )
    out: dict[int, dict[str, float]] = {}
    for position, tokens, cost, sessions, failed, tools, tin, tcached in session.execute(stmt):
        if position is None:
            continue
        out[edges[int(position)]] = {
            "tokens": _int(tokens),
            "cost": _float(cost),
            "sessions": _int(sessions),
            "failed": _int(failed),
            "tools": _int(tools),
            "tokens_in": _int(tin),
            "tokens_cached": _int(tcached),
        }
    return out


def _window_models(model: Any, session: Any, lo: Any, counted: Any) -> list[dict[str, Any]]:
    from sqlalchemy import func, select

    kind = func.coalesce(model.kind, "chat")
    mid = _grouped_key(model.model_id)
    stmt = (
        select(
            mid.label("model_id"),
            kind.label("kind"),
            func.count(),
            func.sum(func.coalesce(model.tokens_in, 0)),
            func.sum(func.coalesce(model.tokens_cached, 0)),
            func.sum(func.coalesce(model.tokens_out, 0)),
            func.sum(_tools_sum(model)),
            func.sum(func.coalesce(model.tools_failed, 0)),
            func.sum(func.coalesce(model.tools_skipped, 0)),
            func.sum(func.coalesce(model.cost, 0.0)),
        )
        .where(model.ts >= lo, counted)
        .group_by(mid, kind)
    )
    return [
        {
            "model_id": row[0],
            "is_task": row[1] == "task",
            "sessions": _int(row[2]),
            "tokens_in": _int(row[3]),
            "tokens_cached": _int(row[4]),
            "tokens_out": _int(row[5]),
            "tools": _int(row[6]),
            "tools_failed": _int(row[7]),
            "tools_skipped": _int(row[8]),
            "cost": _float(row[9]),
        }
        for row in session.execute(stmt)
    ]


def _newest_user_names(model: Any, session: Any, lo: Any, counted: Any) -> dict[str, str]:
    from sqlalchemy import func, select

    uid = _grouped_key(model.user_id)
    ranked = (
        select(
            uid.label("user_id"),
            model.user_name.label("user_name"),
            func.row_number().over(partition_by=uid, order_by=model.ts.desc()).label("rank"),
        )
        .where(
            model.ts >= lo,
            counted,
            model.user_name.isnot(None),
            model.user_name != "",
        )
        .subquery()
    )
    stmt = select(ranked.c.user_id, ranked.c.user_name).where(ranked.c.rank == 1)
    return {user_id: name for user_id, name in session.execute(stmt)}


_BY_USER_CAP = 10


def _window_user_sums(model: Any, session: Any, lo: Any, counted: Any) -> dict[str, float]:
    from sqlalchemy import case, func, select

    stmt = select(
        func.sum(case((_is_chat_row(model), 1), else_=0)),
        func.sum(func.coalesce(model.tokens_in, 0)),
        func.sum(func.coalesce(model.tokens_cached, 0)),
        func.sum(func.coalesce(model.tokens_out, 0)),
        func.sum(_tools_sum(model)),
        func.sum(func.coalesce(model.tools_failed, 0)),
        func.sum(func.coalesce(model.tools_skipped, 0)),
        func.sum(func.coalesce(model.cost, 0.0)),
        func.sum(case((model.kind == "task", func.coalesce(model.cost, 0.0)), else_=0.0)),
    ).where(model.ts >= lo, counted)
    row = session.execute(stmt).one()
    return {
        "sessions": _int(row[0]),
        "tokens_in": _int(row[1]),
        "tokens_cached": _int(row[2]),
        "tokens_out": _int(row[3]),
        "tools": _int(row[4]),
        "tools_failed": _int(row[5]),
        "tools_skipped": _int(row[6]),
        "cost": _float(row[7]),
        "task_cost": _float(row[8]),
    }


def _window_users(
    model: Any, session: Any, lo: Any, counted: Any
) -> tuple[list[dict[str, Any]], int]:
    from sqlalchemy import case, func, select

    uid = _grouped_key(model.user_id)
    cost_sum = func.sum(func.coalesce(model.cost, 0.0))
    stmt = (
        select(
            uid.label("user_id"),
            func.sum(case((_is_chat_row(model), 1), else_=0)),
            func.sum(func.coalesce(model.tokens_in, 0)),
            func.sum(func.coalesce(model.tokens_cached, 0)),
            func.sum(func.coalesce(model.tokens_out, 0)),
            func.sum(_tools_sum(model)),
            func.sum(func.coalesce(model.tools_failed, 0)),
            func.sum(func.coalesce(model.tools_skipped, 0)),
            cost_sum,
            func.sum(case((model.kind == "task", func.coalesce(model.cost, 0.0)), else_=0.0)),
            func.max(model.ts),
            func.count().over().label("user_total"),
        )
        .where(model.ts >= lo, counted)
        .group_by(uid)
        .order_by(cost_sum.desc(), uid)
        .limit(_BY_USER_CAP)
    )
    names = _newest_user_names(model, session, lo, counted)
    out: list[dict[str, Any]] = []
    user_total = 0
    for row in session.execute(stmt):
        if not out:
            user_total = _int(row[11])
        out.append(
            {
                "user_id": row[0],
                "user_name": names.get(row[0]) or "?",
                "sessions": _int(row[1]),
                "tokens_in": _int(row[2]),
                "tokens_cached": _int(row[3]),
                "tokens_out": _int(row[4]),
                "tools": _int(row[5]),
                "tools_failed": _int(row[6]),
                "tools_skipped": _int(row[7]),
                "cost": _float(row[8]),
                "task_cost": _float(row[9]),
                "last_active": _epoch_or_zero(row[10]),
            }
        )
    return out, user_total


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
        "tools": {
            "count": int(acc["tools"]),
            "failed": int(acc["tools_failed"]),
            "skipped": int(acc["tools_skipped"]),
        },
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
    span, bucket_s = USAGE_RANGES[range_key]
    start = now - span
    prev_start = start - span
    off = int(tz_offset_min) * 60

    with _db_session(session_factory) as session:
        from sqlalchemy import case, func, or_

        totals_q = session.query(
            func.sum(case((model.kind == "task", 0), else_=1)), func.min(model.ts),
            func.sum(model.tokens_in), func.sum(model.tokens_cached), func.sum(model.tokens_out),
            func.sum(_tools_sum(model)),
            func.sum(model.cost),
        )
        if not include_tasks:
            totals_q = totals_q.filter(or_(model.kind != "task", model.kind.is_(None)))
        total_count, min_ts, tot_tin, tot_tcached, tot_tout, tot_tools, tot_cost = totals_q.one()

        counted = _counted_rows(model, include_tasks)
        lo = usage_ts_from_epoch(start)
        cur = _window_cards(model, session, lo, None, counted)
        prev = _window_cards(model, session, usage_ts_from_epoch(prev_start), lo, counted)
        buckets = _window_buckets(
            model, session, lo, _bucket_edges(start, now, bucket_s, off), bucket_s, counted
        )
        grouped_models = _window_models(model, session, lo, counted)
        grouped_users, user_total = _window_users(model, session, lo, counted)
        user_sums = _window_user_sums(model, session, lo, counted)

    total_cost_window = cur["cost"] or 0.0
    model_rows = []
    for agg in grouped_models:
        name = name_fn(agg["model_id"]) if name_fn else agg["model_id"]
        if agg["is_task"]:
            name = f"{name} (tasks)"
        sessions = agg["sessions"]
        model_rows.append({
            "model_id": agg["model_id"],
            "model_name": name,
            "sessions": sessions,
            "tokens_in": agg["tokens_in"],
            "tokens_cached": agg["tokens_cached"],
            "tokens_out": agg["tokens_out"],
            "tools": agg["tools"],
            "tools_failed": agg["tools_failed"],
            "tools_skipped": agg["tools_skipped"],
            "cost": round(agg["cost"], 6),
            "avg_cost": round(agg["cost"] / sessions, 6) if sessions else 0.0,
            "share_pct": round(agg["cost"] / total_cost_window * 100, 1) if total_cost_window else 0.0,
        })
    model_rows.sort(key=lambda row: -row["cost"])

    user_out = [{
        "user_name": u["user_name"],
        "sessions": u["sessions"],
        "tokens_in": u["tokens_in"],
        "tokens_cached": u["tokens_cached"],
        "tokens_out": u["tokens_out"],
        "tools": u["tools"],
        "tools_failed": u["tools_failed"],
        "tools_skipped": u["tools_skipped"],
        "cost": round(u["cost"], 6),
        "task_cost": round(u["task_cost"], 6),
        "last_active": int(u["last_active"]) or None,
    } for u in grouped_users]

    bucket_rows = [
        {"t": t, "tokens": int(v["tokens"]), "cost": round(v["cost"], 6),
         "sessions": int(v["sessions"]), "tools": int(v["tools"]),
         "err_rate": round(v["failed"] / v["sessions"], 4) if v["sessions"] else 0.0,
         "cached_pct": round(v["tokens_cached"] / v["tokens_in"], 4) if v["tokens_in"] else 0.0}
        for t, v in sorted(buckets.items())
    ]

    since = None
    try:
        since = int(epoch_from_usage_ts(min_ts)) if min_ts is not None else None
    except (AttributeError, OSError, OverflowError, ValueError):
        logger.warning(
            "usage query: could not derive the earliest retained timestamp; "
            "previous-period comparison is unavailable",
            exc_info=True,
        )
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
            "user_count": int(user_total),
            "by_user_sums": user_sums,
            "start": int(start),
            "now": int(now),
            "since": since,
            "include_tasks": include_tasks,
        },
    }


def _warm_usage_store(store: Any, usage_store: Any, valves: Any, pipe_id: str) -> str | None:
    """Initialize a cold artifact store + usage model off-loop; return the failure type name or None."""
    try:
        if getattr(store, "_session_factory", None) is None:
            store._ensure_artifact_store(valves, pipe_id)
        usage_store.ensure(store)
        return None
    except Exception as exc:
        logger.warning("usage store is unavailable", exc_info=True)
        return type(exc).__name__


async def run_usage_query(plugin: Any, pipe: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Async front: validate, memoize (30s), ensure the table, run in executor."""
    range_key = str(args.get("range") or "24h")
    include_tasks = bool(args.get("include_tasks", True))
    tz_offset_min = int(args.get("tz_offset_min") or 0)
    tz_offset_min = max(-900, min(900, round(tz_offset_min / 15) * 15))

    from .actions import _update_service_of

    svc = _update_service_of(pipe)
    stored_read_ok = False
    row: dict[str, Any] = {}
    if svc is None:
        valves = plugin.ctx.valves
        collect_on = bool(getattr(valves, "PIPE_DASHBOARD_USAGE_COLLECT", False))
        retention_days = int(getattr(valves, "PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30) or 30)
        logger.warning(
            "pipe_dashboard: no update service is registered, so the persisted usage "
            "valves were never consulted; the Usage tab reports the in-memory copy and "
            "says the read did not happen rather than claiming one that did not"
        )
    else:
        try:
            row, stored_read_ok = await svc._row_valves_checked()
            if not stored_read_ok:
                logger.warning(
                    "pipe_dashboard: the persisted usage valves are unreadable; the Usage tab "
                    "is reporting their defaults rather than the in-memory copy, which would "
                    "let a failed read override an operator's disable"
                )
        except Exception:
            logger.warning(
                "pipe_dashboard: cannot read the persisted usage valves; the Usage tab "
                "is reporting their defaults rather than the in-memory copy, which would "
                "let a failed read override an operator's disable",
                exc_info=True,
            )
            row, stored_read_ok = {}, False
        collect_on = bool(row.get("PIPE_DASHBOARD_USAGE_COLLECT", False)) if stored_read_ok else False
        retention_days = int(row.get("PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30) or 30) if stored_read_ok else 30
    base_meta = {
        "collect_on": collect_on,
        "valves_read_ok": bool(stored_read_ok),
        "retention_days": retention_days,
        "range": range_key,
    }

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

    memo_key = (usage_store._table_name, range_key, include_tasks, tz_offset_min,
                bool(stored_read_ok), collect_on, retention_days)
    now = time.time()
    hit = _UQ_MEMO.get(memo_key)
    if hit is not None and now - hit[0] < _UQ_MEMO_TTL:
        return await asyncio.get_running_loop().run_in_executor(None, copy.deepcopy, hit[1])

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
    result["meta"]["persist_failed"] = info.get("persist_failed")
    if len(_UQ_MEMO) >= _UQ_MEMO_MAX:
        for stale_key in [k for k, (ts, _) in _UQ_MEMO.items() if now - ts >= _UQ_MEMO_TTL]:
            _UQ_MEMO.pop(stale_key, None)
        if len(_UQ_MEMO) >= _UQ_MEMO_MAX:
            _UQ_MEMO.clear()
    _UQ_MEMO[memo_key] = (now, result)
    return await asyncio.get_running_loop().run_in_executor(None, copy.deepcopy, result)
