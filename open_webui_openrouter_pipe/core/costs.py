"""Cost tracking and usage snapshot utilities.

This module provides functions for dumping usage/cost snapshots to Redis
for billing and analytics purposes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional, TYPE_CHECKING

from .timing_logger import timed
from .utils import _await_if_needed
from ..storage.persistence import _sanitize_table_fragment

if TYPE_CHECKING:
    from ..pipe import Pipe


@timed
async def maybe_dump_costs_snapshot(
    pipe: "Pipe",
    valves: "Pipe.Valves",
    *,
    user_id: str,
    model_id: Optional[str],
    usage: dict[str, Any] | None,
    user_obj: Optional[Any] = None,
    pipe_id: Optional[str] = None,
) -> None:
    """Push usage snapshots to Redis when enabled, namespaced per pipe.

    Args:
        pipe: The Pipe instance (for accessing redis_client and logger)
        valves: Valve configuration (supports per-request overrides)
        user_id: User GUID
        model_id: Model identifier
        usage: Usage dictionary with token counts etc.
        user_obj: Optional user object with email/name fields
        pipe_id: Optional explicit pipe ID (defaults to pipe.id)
    """
    if not valves.COSTS_REDIS_DUMP:
        return
    if not (pipe._redis_enabled and pipe._redis_client):
        return
    if not user_id:
        return

    def _user_field(obj: Any, field: str) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            value = obj.get(field)
        else:
            value = getattr(obj, field, None)
        return str(value) if value is not None else None

    email = _user_field(user_obj, "email")
    name = _user_field(user_obj, "name")
    snapshot_usage = usage if isinstance(usage, dict) else {}
    model_value = (model_id or "").strip() if isinstance(model_id, str) else (model_id or "")

    missing_fields: list[str] = []
    if not user_id:
        missing_fields.append("guid")
    if not email:
        missing_fields.append("email")
    if not name:
        missing_fields.append("name")
    if not model_value:
        missing_fields.append("model")
    if not snapshot_usage:
        missing_fields.append("usage")
    if missing_fields:
        pipe.logger.debug(
            "Skipping cost snapshot due to missing fields: %s",
            ", ".join(sorted(missing_fields)),
        )
        return

    ttl = valves.COSTS_REDIS_TTL_SECONDS
    ts = int(time.time())
    raw_pipe_id = pipe_id or getattr(pipe, "id", None)
    if not raw_pipe_id:
        pipe.logger.debug("Skipping cost snapshot due to missing pipe identifier.")
        return
    pipe_namespace = _sanitize_table_fragment(raw_pipe_id)
    key = f"costs:{pipe_namespace}:{user_id}:{uuid.uuid4()}:{ts}"
    payload = {
        "guid": user_id,
        "email": str(email),
        "name": str(name),
        "model": model_value,
        "usage": snapshot_usage,
        "ts": ts,
    }
    try:
        await _await_if_needed(
            pipe._redis_client.set(key, json.dumps(payload, default=str), ex=ttl)
        )
    except Exception as exc:  # pragma: no cover - Redis failures logged, not fatal
        pipe.logger.debug("Cost snapshot write failed for user=%s: %s", user_id, exc)
