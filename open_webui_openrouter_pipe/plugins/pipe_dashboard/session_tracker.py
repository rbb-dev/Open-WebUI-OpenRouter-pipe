"""In-memory live-session tracking for the pipe_dashboard plugin.

Sessions are keyed by the pipe's per-request id and observed entirely via
plugin hooks: created at ``on_request``, updated by the wrapped emitter's
``chat:completion`` usage snapshots and the ``on_tool_result`` /
``on_request_retry`` events, and finalized exactly once by
``on_generation_complete``. Completed entries stay in a bounded recent ring
for the dashboard's retention window; task entries fold their cost into the
parent chat session (matched by chat id). DB persistence is delegated to a
finalize callback so this module stays storage-free.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

_st_log = logging.getLogger(__name__)

_ST_ACTIVE_CAP = 30
_ST_RECENT_CAP = 300
_ST_RECENT_MAX_AGE_S = 10800.0
_ST_ABANDON_S = 7200.0

_ST_STATUS_MAP = {"ok": "completed", "failed": "failed", "cancelled": "cancelled"}


def _usage_numbers(usage: Any) -> dict[str, float]:
    numbers = {"tin": 0, "tout": 0, "treason": 0, "tcached": 0, "cost": 0.0, "discount": 0.0}
    if not isinstance(usage, dict):
        return numbers
    try:
        itd = usage.get("input_tokens_details") or {}
        otd = usage.get("output_tokens_details") or {}
        numbers["tin"] = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        numbers["tout"] = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        numbers["treason"] = int((otd or {}).get("reasoning_tokens") or 0)
        numbers["tcached"] = int((itd or {}).get("cached_tokens") or 0)
        numbers["cost"] = float(usage.get("cost") or 0.0)
        numbers["discount"] = float(usage.get("cache_discount") or 0.0)
    except Exception:
        _st_log.debug("usage parse failed", exc_info=True)
    return numbers


class SessionTracker:
    """Thread-safe registry of active + recently-completed sessions."""

    def __init__(
        self,
        pricing_fn: Callable[[str], dict[str, Any] | None] | None = None,
        name_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._recent: list[dict[str, Any]] = []
        self._pricing_fn = pricing_fn
        self._name_fn = name_fn
        self._pid = os.getpid()
        self.on_finalize: Callable[[dict[str, Any]], None] | None = None

    def start(
        self,
        request_id: str,
        *,
        body: dict[str, Any] | None = None,
        user: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        task: Any = None,
    ) -> None:
        if not request_id:
            return
        body = body if isinstance(body, dict) else {}
        user = user if isinstance(user, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        task_name = self._task_name(task)
        model_id = str(body.get("model") or "")
        entry = {
            "request_id": request_id,
            "kind": "task" if task_name else "chat",
            "task_name": task_name or None,
            "user_id": str(user.get("id") or metadata.get("user_id") or ""),
            "user_name": str(user.get("name") or user.get("email") or "?"),
            "chat_id": str(metadata.get("chat_id") or ""),
            "session_id": str(metadata.get("session_id") or ""),
            "model_id": model_id,
            "model_name": self._resolve_name(model_id),
            "status": "queued",
            "started": time.time(),
            "done": None,
            "tin": 0,
            "tout": 0,
            "treason": 0,
            "tcached": 0,
            "cost": 0.0,
            "discount": 0.0,
            "task_cost": 0.0,
            "tools_ok": 0,
            "tools_failed": 0,
            "tools_skipped": 0,
            "retries": 0,
            "current_tool": None,
        }
        with self._lock:
            self._active[request_id] = entry

    @staticmethod
    def _task_name(task: Any) -> str:
        if isinstance(task, str):
            return task.strip()
        if isinstance(task, dict):
            name = task.get("type") or task.get("task") or task.get("name")
            return name.strip() if isinstance(name, str) else ""
        return ""

    def _resolve_name(self, model_id: str) -> str:
        fn = self._name_fn
        if fn is None or not model_id:
            return model_id
        try:
            return str(fn(model_id) or model_id)
        except Exception:
            return model_id

    def mark_streaming(self, request_id: str) -> None:
        with self._lock:
            entry = self._active.get(request_id)
            if entry is not None and entry["status"] == "queued":
                entry["status"] = "streaming"

    def tool_started(self, request_id: str, tool_name: str) -> None:
        with self._lock:
            entry = self._active.get(request_id)
            if entry is not None:
                entry["current_tool"] = str(tool_name or "?")
                entry["status"] = "tool"

    def tool_result(self, request_id: str, status: str) -> None:
        with self._lock:
            entry = self._active.get(request_id)
            if entry is None:
                return
            if status == "failed":
                entry["tools_failed"] += 1
            elif status == "skipped":
                entry["tools_skipped"] += 1
            else:
                entry["tools_ok"] += 1
            entry["current_tool"] = None
            if entry["status"] == "tool":
                entry["status"] = "streaming"

    def retry(self, request_id: str) -> None:
        with self._lock:
            entry = self._active.get(request_id)
            if entry is not None:
                entry["retries"] += 1

    def update_usage(self, request_id: str, usage: Any) -> None:
        numbers = _usage_numbers(usage)
        with self._lock:
            entry = self._active.get(request_id)
            if entry is None:
                return
            for key in ("tin", "tout", "treason", "tcached", "cost", "discount"):
                if numbers[key]:
                    entry[key] = numbers[key]
            if entry["status"] == "queued":
                entry["status"] = "streaming"

    def finalize(self, request_id: str, usage: Any, status_word: str) -> None:
        numbers = _usage_numbers(usage)
        row: dict[str, Any] | None = None
        with self._lock:
            entry = self._active.pop(request_id, None)
            if entry is None:
                return
            for key in ("tin", "tout", "treason", "tcached", "cost", "discount"):
                if numbers[key]:
                    entry[key] = numbers[key]
            entry["status"] = _ST_STATUS_MAP.get(status_word, "failed")
            entry["done"] = time.time()
            entry["current_tool"] = None
            entry["savings"] = self._cache_savings(entry)
            self._fold_task_into_parent(entry)
            self._recent.append(entry)
            self._trim_recent_locked()
            row = dict(entry)
        callback = self.on_finalize
        if callback is not None and row is not None:
            try:
                callback(row)
            except Exception:
                _st_log.debug("finalize callback failed", exc_info=True)

    def _cache_savings(self, entry: dict[str, Any]) -> float:
        discount = float(entry.get("discount") or 0.0)
        if discount:
            return abs(discount)
        cached = int(entry.get("tcached") or 0)
        if not cached or self._pricing_fn is None:
            return 0.0
        try:
            pricing = self._pricing_fn(str(entry.get("model_id") or "")) or {}
            prompt_rate = float(pricing.get("prompt") or 0.0)
            cache_rate = float(pricing.get("input_cache_read") or 0.0)
            if prompt_rate > cache_rate > 0:
                return cached * (prompt_rate - cache_rate)
        except Exception:
            _st_log.debug("cache savings compute failed", exc_info=True)
        return 0.0

    def _fold_task_into_parent(self, entry: dict[str, Any]) -> None:
        if entry.get("kind") != "task":
            return
        chat_id = entry.get("chat_id")
        cost = float(entry.get("cost") or 0.0)
        if not chat_id or not cost:
            return
        candidates = [
            item
            for item in list(self._active.values()) + self._recent
            if item.get("kind") == "chat" and item.get("chat_id") == chat_id
        ]
        if not candidates:
            return
        parent = max(candidates, key=lambda item: item.get("started") or 0.0)
        parent["task_cost"] = float(parent.get("task_cost") or 0.0) + cost

    def _trim_recent_locked(self) -> None:
        cutoff = time.time() - _ST_RECENT_MAX_AGE_S
        self._recent = [item for item in self._recent if (item.get("done") or 0.0) >= cutoff]
        if len(self._recent) > _ST_RECENT_CAP:
            self._recent = self._recent[-_ST_RECENT_CAP:]

    def sweep(self) -> None:
        cutoff = time.time() - _ST_ABANDON_S
        with self._lock:
            stale = [rid for rid, item in self._active.items() if (item.get("started") or 0.0) < cutoff]
        for rid in stale:
            self.finalize(rid, None, "failed")

    def live_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._live_sessions_locked(time.time())

    def task_costs_by_chat(self) -> dict[str, float]:
        with self._lock:
            return self._task_costs_locked()

    def live_snapshot(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        with self._lock:
            return self._live_sessions_locked(time.time()), self._task_costs_locked()

    def _live_sessions_locked(self, now: float) -> list[dict[str, Any]]:
        actives = sorted(self._active.values(), key=lambda item: item.get("started") or 0.0, reverse=True)[:_ST_ACTIVE_CAP]
        self._trim_recent_locked()
        recents = list(reversed(self._recent))[:_ST_RECENT_CAP]
        rows = []
        for item in actives + recents:
            if item.get("kind") == "task":
                continue
            status = item.get("status") or "queued"
            if status == "tool":
                status = f"tool:{item.get('current_tool') or '?'}"
            done = item.get("done")
            elapsed = (done or now) - (item.get("started") or now)
            rows.append({
                "user": item.get("user_name") or "?",
                "model_id": item.get("model_id") or "",
                "model_name": item.get("model_name") or item.get("model_id") or "",
                "kind": item.get("kind") or "chat",
                "status": status,
                "started": round(item.get("started") or 0.0, 1),
                "done": round(done, 1) if done else None,
                "elapsed_s": round(max(0.0, elapsed), 1),
                "tokens_in": int(item.get("tin") or 0),
                "tokens_cached": int(item.get("tcached") or 0),
                "tokens_out": int(item.get("tout") or 0),
                "tools_ok": int(item.get("tools_ok") or 0),
                "tools_failed": int(item.get("tools_failed") or 0),
                "cost": round(float(item.get("cost") or 0.0) + float(item.get("task_cost") or 0.0), 6),
                "task_cost": round(float(item.get("task_cost") or 0.0), 6),
                "worker_pid": self._pid,
                "chat_id": item.get("chat_id") or "",
            })
        return rows

    def _task_costs_locked(self) -> dict[str, float]:
        chat_ids = {
            e.get("chat_id")
            for e in list(self._active.values()) + self._recent
            if e.get("kind") == "chat" and e.get("chat_id")
        }
        out: dict[str, float] = {}
        for e in self._recent:
            if e.get("kind") != "task":
                continue
            cid = e.get("chat_id")
            cost = float(e.get("cost") or 0.0)
            if not cid or not cost or cid in chat_ids:
                continue
            out[cid] = out.get(cid, 0.0) + cost
        return out

    def db_row(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Map a finalized entry to the UsageStore row schema."""
        import datetime

        done = float(entry.get("done") or time.time())
        started = float(entry.get("started") or done)
        return {
            "ts": datetime.datetime.fromtimestamp(done),
            "started_at": datetime.datetime.fromtimestamp(started),
            "kind": entry.get("kind") or "chat",
            "user_id": entry.get("user_id") or "",
            "user_name": entry.get("user_name") or "",
            "chat_id": entry.get("chat_id") or "",
            "session_id": entry.get("session_id") or "",
            "model_id": entry.get("model_id") or "",
            "task_name": entry.get("task_name"),
            "status": {"completed": "ok"}.get(str(entry.get("status")), str(entry.get("status") or "ok")),
            "duration_ms": int(max(0.0, done - started) * 1000),
            "tokens_in": int(entry.get("tin") or 0),
            "tokens_out": int(entry.get("tout") or 0),
            "tokens_reasoning": int(entry.get("treason") or 0),
            "tokens_cached": int(entry.get("tcached") or 0),
            "tools_ok": int(entry.get("tools_ok") or 0),
            "tools_failed": int(entry.get("tools_failed") or 0),
            "retries": int(entry.get("retries") or 0),
            "cost": float(entry.get("cost") or 0.0),
            "cache_savings": float(entry.get("savings") or 0.0),
            "worker_pid": self._pid,
        }
