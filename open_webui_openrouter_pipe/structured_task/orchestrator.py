"""Task-model selection: resolves candidate model IDs from OWUI config.

Ports the pattern from `.external/seedream.py:670-693`.

The `chat_model` fallback was removed in v2.6.x: media-generation flows route
through video/image models that cannot process structured-output classification
prompts, so falling back to the user's selected chat model produced
classifier-failure-then-degrade-open instead of useful results. The two
remaining strategies (`none`, `other_task_model`) cover the realistic
deployment shapes.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

logger = logging.getLogger(__name__)

TaskModelMode = Literal["internal", "external"]
TaskModelFallback = Literal["none", "other_task_model"]

_OWUI_TASK_MODEL_KEYS = ("task.model.default", "task.model.external")
_OWUI_TASK_MODEL_PARAMS_KEY = "task.model.params"

_OWUI_REQUEST_SCOPED_PARAM_KEYS = frozenset(
    {
        "stream_response",
        "stream_delta_chunk_size",
        "function_calling",
        "reasoning_tags",
        "compact_token_threshold",
        "system",
        "note_id",
        "tool_approval_mode",
    }
)


async def read_task_model_config() -> tuple[dict[str, str] | None, dict[str, Any]]:
    try:
        from open_webui.models.config import Config as _OwuiConfig
    except Exception:
        logger.debug(
            "structured_task: Open WebUI's config table is not importable",
            exc_info=True,
        )
        return None, {}

    try:
        rows = await _OwuiConfig.get_many(*_OWUI_TASK_MODEL_KEYS, _OWUI_TASK_MODEL_PARAMS_KEY)
    except Exception:
        logger.debug(
            "structured_task: Open WebUI's Task Model settings could not be read",
            exc_info=True,
        )
        return None, {}

    raw = rows.get(_OWUI_TASK_MODEL_PARAMS_KEY) if isinstance(rows, dict) else None
    params = (
        {key: value for key, value in raw.items() if value is not None and value != ""}
        if isinstance(raw, dict)
        else {}
    )
    ids = {
        key: str(rows[key]).strip()
        for key in _OWUI_TASK_MODEL_KEYS
        if rows.get(key)
    }
    return ids, params


async def read_task_model_params() -> dict[str, Any]:
    return (await read_task_model_config())[1]


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(value, Mapping):
            base = dict(current) if isinstance(current, Mapping) else {}
            target[key] = _deep_update(base, value)
        else:
            target[key] = value
    return target


def merge_task_model_params(payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(params, dict) or not params:
        return payload

    mergeable = {
        key: value
        for key, value in params.items()
        if key not in payload and key not in _OWUI_REQUEST_SCOPED_PARAM_KEYS
    }
    custom_params = mergeable.pop("custom_params", None)
    if isinstance(custom_params, dict) and custom_params:
        decoded: dict[str, Any] = {}
        for key, value in custom_params.items():
            if isinstance(value, str):
                try:
                    decoded[key] = json.loads(value)
                except (TypeError, ValueError):
                    decoded[key] = value
            else:
                decoded[key] = value
        effective = _deep_update(dict(mergeable), decoded)
    else:
        effective = mergeable

    for key, value in effective.items():
        if value is not None and key not in payload and key not in _OWUI_REQUEST_SCOPED_PARAM_KEYS:
            payload[key] = value
    return payload


def select_task_model_candidates(
    rows: dict[str, str] | None,
    *,
    request: Any,
    mode: TaskModelMode,
    fallback: TaskModelFallback,
) -> list[str]:
    """Return ordered list of task-model IDs per OWUI config.

    Empty strings are filtered. Duplicates are deduped while preserving order.

    Args:
        mode: "internal" picks TASK_MODEL primary, "external" picks TASK_MODEL_EXTERNAL.
        fallback: "none" returns only primary; "other_task_model" appends the
            other OWUI task model.
    """
    if rows is None:
        config = getattr(getattr(request, "app", None), "state", None)
        config = getattr(config, "config", None) if config is not None else None
        internal = (getattr(config, "TASK_MODEL", "") or "").strip() if config else ""
        external = (getattr(config, "TASK_MODEL_EXTERNAL", "") or "").strip() if config else ""
    else:
        internal = rows.get("task.model.default", "")
        external = rows.get("task.model.external", "")

    primary = internal if mode == "internal" else external
    other = external if mode == "internal" else internal

    candidates: list[str] = []
    if primary:
        candidates.append(primary)

    if fallback == "other_task_model" and other and other not in candidates:
        candidates.append(other)

    return candidates


async def resolve_task_model_candidates(
    *,
    request: Any,
    mode: TaskModelMode,
    fallback: TaskModelFallback,
) -> list[str]:
    rows, _params = await read_task_model_config()
    return select_task_model_candidates(rows, request=request, mode=mode, fallback=fallback)
