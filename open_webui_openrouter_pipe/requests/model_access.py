from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from ..core.fusion_defaults import (
    DEFAULT_FUSION_PRESET,
    FUSION_PRESET_PANELS,
    FusionPanelTooLargeError,
    find_fusion_entry,
    resolve_fusion_run,
)
from ..models.registry import ModelFamily, OpenRouterModelRegistry

logger = logging.getLogger(__name__)

FUSION_SERVER_TOOL_TYPE = "openrouter:fusion"
FUSION_ROLES = ("panel", "judge")
MODEL_NAMING_TOOL_ROLES = {
    "openrouter:advisor": "advisor",
    "openrouter:subagent": "subagent",
}
_NAME_SHOWN_LIMIT = 200


def _open_webui_model_access() -> Any:
    from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL
    from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL
    from open_webui.models.models import Models
    from open_webui.utils.models import check_model_access

    return SimpleNamespace(
        Models=Models,
        check_model_access=check_model_access,
        bypass_admin_access=BYPASS_ADMIN_ACCESS_CONTROL,
        bypass_model_access=BYPASS_MODEL_ACCESS_CONTROL,
    )


def models_the_request_would_use(
    *,
    plugins: Any,
    tools: Any,
    member_tools: Any = None,
    saved_plugins: list[Any],
    saved_tools: list[Any],
    internal: bool,
    fusion_on: bool,
    pipe_id: str,
) -> dict[str, tuple[str, list[str]]]:
    wanted: dict[str, tuple[str, list[str]]] = {}

    def add(model: Any, role: str) -> None:
        shown = model.strip() if isinstance(model, str) else ""
        if not shown:
            return
        _, roles = wanted.setdefault(ModelFamily.base_model(shown, pipe_id), (shown, []))
        if role not in roles:
            roles.append(role)

    def add_fusion_run(entry: dict[str, Any] | None, *, hosted: bool) -> None:
        try:
            plan = resolve_fusion_run(entry)
        except FusionPanelTooLargeError:
            return
        for model in plan.panel_models:
            add(model, "panel")
        if hosted and plan.panel_from_preset:
            for model in FUSION_PRESET_PANELS[DEFAULT_FUSION_PRESET]:
                add(model, "panel")
        if not hosted or not plan.judge_from_preset:
            add(plan.judge_model, "judge")

    def add_fusion_tool(tool: dict[str, Any]) -> None:
        parameters = tool.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        add_fusion_run(
            {
                "id": "fusion",
                "analysis_models": parameters.get("analysis_models"),
                "model": parameters.get("model"),
            },
            hosted=True,
        )

    def add_model_naming_tools(candidates: Any) -> None:
        if not isinstance(candidates, list):
            return
        for tool in candidates:
            if not isinstance(tool, dict):
                continue
            kind = tool.get("type")
            role = MODEL_NAMING_TOOL_ROLES.get(kind) if isinstance(kind, str) else None
            parameters = tool.get("parameters")
            if role is not None and isinstance(parameters, dict):
                add(parameters.get("model"), role)

    if fusion_on:
        if internal:
            add_fusion_run(find_fusion_entry(plugins), hosted=False)
        else:
            entry = find_fusion_entry(plugins)
            if entry is None:
                entry = find_fusion_entry(saved_plugins)
            if entry is not None and entry.get("enabled") is not False:
                add_fusion_run(entry, hosted=True)
    if internal:
        add_model_naming_tools(member_tools)
        return wanted

    sent_tools = [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []
    for tool in sent_tools:
        if tool.get("type") == FUSION_SERVER_TOOL_TYPE and fusion_on:
            add_fusion_tool(tool)
    add_model_naming_tools(sent_tools)
    if fusion_on and not any(tool.get("type") == FUSION_SERVER_TOOL_TYPE for tool in sent_tools):
        for tool in saved_tools:
            if isinstance(tool, dict) and tool.get("type") == FUSION_SERVER_TOOL_TYPE:
                add_fusion_tool(tool)
    return wanted


def user_for_access_check(user_model: Any, *, user_id: str, role: Any) -> Any:
    return user_model if user_model is not None else SimpleNamespace(id=user_id, role=role)


def _published_row_ids(pipe: Any, pipe_id: str) -> dict[str, str]:
    rows = pipe._select_model_rows(OpenRouterModelRegistry.list_models())
    return {
        ModelFamily.base_model(row["id"], pipe_id): row["id"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


async def models_the_user_may_not_use(
    pipe: Any,
    *,
    pipe_id: str,
    user: Any,
    wanted: dict[str, tuple[str, list[str]]],
    restricted_by_pipe: Callable[[str], bool],
    catalogue_known: bool,
) -> list[tuple[str, list[str]]]:
    rows = _published_row_ids(pipe, pipe_id)
    open_webui = _open_webui_model_access()
    bypass = bool(
        open_webui.bypass_model_access
        or (getattr(user, "role", None) == "admin" and open_webui.bypass_admin_access)
    )
    unavailable: list[tuple[str, list[str]]] = []
    for norm, (shown, roles) in wanted.items():
        if restricted_by_pipe(norm):
            continue
        row_id = rows.get(norm)
        if row_id is None:
            if catalogue_known:
                unavailable.append((shown, roles))
            continue
        open_webui_id = f"{pipe_id}.{row_id}"
        row = await open_webui.Models.get_model_by_id(open_webui_id)
        if row is not None and getattr(row, "is_active", True) is False:
            unavailable.append((shown, roles))
            continue
        if bypass:
            continue
        try:
            await open_webui.check_model_access(user, {"id": open_webui_id}, model_info=row)
        except Exception:
            logger.debug(
                "Open WebUI refused %s to user %s",
                open_webui_id,
                getattr(user, "id", None),
                exc_info=True,
            )
            unavailable.append((shown, roles))
    return unavailable


def names_a_fusion_model(unavailable: list[tuple[str, list[str]]]) -> bool:
    return any(role in FUSION_ROLES for _shown, roles in unavailable for role in roles)


def describe_unavailable_models(
    unavailable: list[tuple[str, list[str]]], *, judge_writes_the_answer: bool
) -> str:
    judge_label = "judge and final answer" if judge_writes_the_answer else "judge"
    parts = []
    for role, label in (
        ("panel", "panel"),
        ("judge", judge_label),
        ("advisor", "advisor"),
        ("subagent", "subagent"),
    ):
        names = [
            shown if len(shown) <= _NAME_SHOWN_LIMIT else shown[:_NAME_SHOWN_LIMIT] + "…"
            for shown, roles in unavailable
            if role in roles
        ]
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts)


def refusal_sentence(summary: str, *, fusion: bool) -> str:
    if fusion:
        return (
            "Fusion was not run: models it would call are not available to this user in "
            f"Open WebUI ({summary})"
        )
    return (
        "The request was not sent: models its advisor or subagent tool names are not "
        f"available to this user in Open WebUI ({summary})"
    )
