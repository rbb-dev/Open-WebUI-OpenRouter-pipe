"""Cost tracking and usage snapshot utilities.

This module provides functions for dumping usage/cost snapshots to Redis
for billing and analytics purposes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from ..storage.persistence import _sanitize_table_fragment
from .timing_logger import timed
from .utils import _await_if_needed

if TYPE_CHECKING:
    from ..pipe import Pipe


@timed
async def maybe_dump_costs_snapshot(
    pipe: Pipe,
    valves: Pipe.Valves,
    *,
    user_id: str,
    model_id: str | None,
    usage: dict[str, Any] | None,
    user_obj: Any | None = None,
    pipe_id: str | None = None,
    chat_id: str | None = None,
    message_id: str | None = None,
    kind: str | None = None,
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

    def _user_field(obj: Any, field: str) -> str | None:
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
    if chat_id:
        payload["chat_id"] = str(chat_id)
    if message_id:
        payload["message_id"] = str(message_id)
    if kind:
        payload["kind"] = str(kind)
    try:
        await _await_if_needed(
            pipe._redis_client.set(key, json.dumps(payload, default=str), ex=ttl)
        )
    except Exception as exc:  # pragma: no cover - Redis failures logged, not fatal
        pipe.logger.debug(
            "Cost snapshot write failed for user=%s: %s", user_id, exc, exc_info=True
        )


def chat_usage_to_responses_usage(raw_usage: Any) -> dict[str, Any]:
    """Normalise Chat Completions usage counters into the Responses-style keys this pipe reads."""
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, Any] = {}

    prompt_tokens = raw_usage.get("prompt_tokens")
    completion_tokens = raw_usage.get("completion_tokens")
    total_tokens = raw_usage.get("total_tokens")
    if prompt_tokens is not None:
        usage["input_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["output_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens

    for key in ("cost", "cache_discount", "cache_discount_pct"):
        if key in raw_usage:
            usage[key] = raw_usage[key]

    cost_details = raw_usage.get("cost_details")
    if isinstance(cost_details, dict) and cost_details:
        usage["cost_details"] = dict(cost_details)

    prompt_details = raw_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and prompt_details:
        usage["input_tokens_details"] = dict(prompt_details)

    completion_details = raw_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict) and completion_details:
        usage["output_tokens_details"] = dict(completion_details)

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in raw_usage and key not in usage:
            usage[key] = raw_usage[key]

    return usage
