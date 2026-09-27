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

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

TaskModelMode = Literal["internal", "external"]
TaskModelFallback = Literal["none", "other_task_model"]

_OWUI_TASK_MODEL_KEYS = ("task.model.default", "task.model.external")


async def _owui_task_model_ids() -> dict[str, str] | None:
    try:
        from open_webui.models.config import Config as _OwuiConfig
    except Exception:
        logger.debug(
            "structured_task: Open WebUI's config table is not importable",
            exc_info=True,
        )
        return None

    try:
        rows = await _OwuiConfig.get_many(*_OWUI_TASK_MODEL_KEYS)
    except Exception:
        logger.debug(
            "structured_task: Open WebUI's Task Model settings could not be read",
            exc_info=True,
        )
        return None

    return {
        key: str(rows[key]).strip()
        for key in _OWUI_TASK_MODEL_KEYS
        if rows.get(key)
    }


async def resolve_task_model_candidates(
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
    rows = await _owui_task_model_ids()
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
