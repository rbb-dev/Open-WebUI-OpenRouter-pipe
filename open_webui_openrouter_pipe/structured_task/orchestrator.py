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

from typing import Any, Literal

TaskModelMode = Literal["internal", "external"]
TaskModelFallback = Literal["none", "other_task_model"]


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
    internal = external = ""
    try:
        from open_webui.models.config import Config

        values = await Config.get_many("task.model.default", "task.model.external")
        internal = (values.get("task.model.default") or "").strip()
        external = (values.get("task.model.external") or "").strip()
    except Exception:  # noqa: BLE001
        config = getattr(getattr(request, "app", None), "state", None)
        config = getattr(config, "config", None) if config is not None else None
        internal = (getattr(config, "TASK_MODEL", "") or "").strip() if config else ""
        external = (getattr(config, "TASK_MODEL_EXTERNAL", "") or "").strip() if config else ""

    primary = internal if mode == "internal" else external
    other = external if mode == "internal" else internal

    candidates: list[str] = []
    if primary:
        candidates.append(primary)

    if fallback == "other_task_model" and other and other not in candidates:
        candidates.append(other)

    return candidates
