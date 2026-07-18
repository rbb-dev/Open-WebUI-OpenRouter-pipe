"""Renderer for the standalone OWUI "OpenRouter Fusion" filter.

This module emits the *source code* of a self-contained Open WebUI Filter
function (installed separately into OWUI's Functions table). The rendered filter
configures OpenRouter's Fusion multi-model judge panel by injecting a
``{"id": "fusion", ...}`` entry into ``body["plugins"]``, and optionally forces
Fusion to run via ``tool_choice="required"``.

Design:
- Gated to the ``openrouter/fusion`` model unless ``ALLOW_ON_NON_FUSION_MODELS``.
- Configuration (preset / panel / judge / max_tool_calls) maps to the Fusion
  plugin object; sentinels (empty string / 0) mean "use Fusion's default".
- ``FUSION_FORCE_TOOL_CALL`` (default off) sets ``tool_choice="required"`` on the
  request — exactly as OpenRouter documents for forcing Fusion. On the fusion
  models this is redundant: the pipe forces ``required`` on every chat request
  itself, so the switch only matters on non-fusion models an admin attached the
  filter to. If the caller adds other tools the model may pick one of those
  instead (documented OpenRouter behaviour, see docs/openrouter_fusion.md).
"""
from __future__ import annotations

import re

# Canonical id of the dedicated Fusion model and the OWUI function id.
FUSION_MODEL_SLUG = "openrouter/fusion"
FUSION_FILTER_FUNCTION_ID = "openrouter_fusion"
FUSION_FILTER_DISPLAY_NAME = "OpenRouter Fusion"

# The fusion model appears in several id forms across the pipe:
#   raw slug            "openrouter/fusion"                              (catalog original_id, /v1/models id)
#   pipe-prefixed slug  "openrouter.openrouter/fusion"                   (prefix + raw slug)
#   sanitized id        "openrouter.fusion"                              (sanitize_model_id rewrites '/'->'.')
#   OWUI full model id  "open_webui_openrouter_pipe.openrouter.fusion"  (runtime body["model"])
# The first two keep the vendor/model slash; the last two are all-dots. Match both
# shapes — the slash form via the anchored pattern, the dotted form via a suffix match.
_FUSION_MODEL_PATTERN = re.compile(r"^openrouter/fusion(?:-flash)?$")
_FUSION_DOTTED_PATTERN = re.compile(r"(?:^|\.)openrouter\.fusion(?:-flash)?$")


def canonical_model_slug(raw: str) -> str:
    """Strip the OWUI pipe prefix (``openrouter.openrouter/fusion`` -> ``openrouter/fusion``)."""
    if not isinstance(raw, str) or "/" not in raw:
        return raw if isinstance(raw, str) else ""
    head, slash, tail = raw.partition("/")
    return head.rsplit(".", 1)[-1] + slash + tail


def is_fusion_model(model_id: str) -> bool:
    """True if ``model_id`` is the dedicated openrouter/fusion model, in any id form the
    pipe produces: raw slug, pipe-prefixed slug, sanitized dot-form, or full OWUI id.

    The single source of truth for "which model gets the Fusion filter auto-wired",
    reused by FilterManager/catalog attach decisions and the rendered filter inlet.
    """
    if not isinstance(model_id, str) or not model_id:
        return False
    if _FUSION_MODEL_PATTERN.match(canonical_model_slug(model_id)):
        return True
    return bool(_FUSION_DOTTED_PATTERN.search(model_id))


# The rendered filter source. ``__MARKER__`` / ``__FILTER_ID__`` are substituted
# by render_openrouter_fusion_filter_source(). The body is a plain string (not an
# f-string) so dict/brace literals need no escaping.
_FUSION_FILTER_TEMPLATE = '''"""
title: OpenRouter Fusion
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: __FILTER_ID__
description: Configure OpenRouter Fusion (multi-model judge panel) for the OpenRouter pipe.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover - OWUI not importable in some contexts
    SRC_LOG_LEVELS = {}

OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"
# Match the fusion model in slash forms ("openrouter/fusion", "<prefix>.openrouter/fusion")
# and all-dots forms ("openrouter.fusion", "<funcid>.openrouter.fusion"). The pipe
# sanitizes '/'->'.' in OWUI model ids, so the runtime body["model"] is all-dots.
_FUSION_MODEL_PATTERN = re.compile(r"^openrouter/fusion(?:-flash)?$")
_FUSION_DOTTED_PATTERN = re.compile(r"(?:^|\\.)openrouter\\.fusion(?:-flash)?$")


class FusionConfigError(Exception):
    """Raised at inlet when Fusion configuration is invalid (e.g. panel > 8 models)."""


def _canonical_model_slug(raw: str) -> str:
    if not isinstance(raw, str) or "/" not in raw:
        return raw if isinstance(raw, str) else ""
    head, slash, tail = raw.partition("/")
    return head.rsplit(".", 1)[-1] + slash + tail


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        ALLOW_ON_NON_FUSION_MODELS: bool = Field(
            default=False,
            title="Allow on non-Fusion models",
            description=(
                "Off (default): this filter only acts on the openrouter/fusion model and "
                "no-ops if attached to anything else. On: it adds the Fusion panel to any "
                "model it is attached to."
            ),
        )

    class UserValves(BaseModel):
        FUSION_PRESET: Literal["", "general-high", "general-budget", "general-fast"] = Field(
            default="",
            title="Preset",
            description=(
                "Curated Fusion panel + judge bundle. 'general-high' = the strongest "
                "frontier trio with a frontier judge; 'general-budget' = a fast low-cost "
                "trio with the same frontier judge; 'general-fast' = the same low-cost "
                "trio with a quicker judge (lowest latency). Empty = general-high. "
                "Explicit panel/judge below override a preset."
            ),
        )
        FUSION_ANALYSIS_MODELS: str = Field(
            default="",
            title="Panel models (comma-separated)",
            description=(
                "1-8 model IDs that answer in parallel, comma-separated, e.g. "
                "'anthropic/claude-opus-latest, openai/gpt-latest'. Each model adds cost. "
                "Empty = use the preset / Fusion default panel."
            ),
        )
        FUSION_JUDGE_MODEL: str = Field(
            default="",
            title="Judge model",
            description=(
                "Model that reviews the panel's answers and writes the final analysis. "
                "Empty = Fusion default."
            ),
        )
        FUSION_MAX_TOOL_CALLS: int = Field(
            default=0,
            ge=0,
            le=16,
            title="Max tool calls per model",
            description=(
                "Tool budget for each panel and judge model (1-16; 0 = default 8). On the "
                "OpenRouter engine this caps web search/fetch steps. On the pipe's "
                "internal engine it is enforced as a hard per-model cap on individual "
                "tool invocations (knowledge bases, tool servers, web tools alike); "
                "excess calls are skipped, and it also bounds the model's tool rounds."
            ),
        )
        FUSION_FORCE_TOOL_CALL: bool = Field(
            default=False,
            title="Always run Fusion",
            description=(
                "On the dedicated fusion models this switch has no effect: deliberation is "
                "guaranteed on both engines (the OpenRouter engine via tool_choice="
                "'required'; the internal engine always deliberates). It matters only "
                "when an admin has attached this filter to a "
                "non-fusion model (ALLOW_ON_NON_FUSION_MODELS). There — Off (default): the "
                "model decides whether the prompt needs the multi-model panel (cheaper; some "
                "replies answer directly). On: force the panel to run every message. Only "
                "reliable when Fusion is the only tool in the request; other tool "
                "integrations let the model satisfy the forcing with a different tool."
            ),
        )

    def __init__(self) -> None:
        self.toggle = True
        self.valves = self.Valves()
        self.log = logging.getLogger("openrouter.fusion.filter")
        try:
            self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        except Exception:
            pass

    def _read_user_valves(self, __user__: Any) -> "Filter.UserValves":
        if isinstance(__user__, dict):
            raw = __user__.get("valves")
            if isinstance(raw, self.UserValves):
                return raw
            if isinstance(raw, BaseModel):
                try:
                    return self.UserValves(**raw.model_dump())
                except Exception:
                    return self.UserValves()
            if isinstance(raw, dict):
                try:
                    return self.UserValves(**raw)
                except Exception:
                    return self.UserValves()
        return self.UserValves()

    def inlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __user__: dict | None = None,
    ) -> dict:
        try:
            if not isinstance(body, dict):
                return body
            model_id = body.get("model") or ""
            is_fusion = isinstance(model_id, str) and bool(
                _FUSION_MODEL_PATTERN.match(_canonical_model_slug(model_id))
                or _FUSION_DOTTED_PATTERN.search(model_id)
            )
            if not is_fusion and not self.valves.ALLOW_ON_NON_FUSION_MODELS:
                return body

            uv = self._read_user_valves(__user__)

            plugins = body.get("plugins")
            plugins = list(plugins) if isinstance(plugins, list) else []
            idx = next(
                (i for i, p in enumerate(plugins) if isinstance(p, dict) and p.get("id") == "fusion"),
                None,
            )
            cfg = dict(plugins[idx]) if idx is not None else {"id": "fusion"}

            preset = (uv.FUSION_PRESET or "").strip()
            if preset:
                cfg["preset"] = preset
            panel = [m.strip() for m in (uv.FUSION_ANALYSIS_MODELS or "").split(",") if m.strip()]
            if panel:
                if len(panel) > 8:
                    raise FusionConfigError(
                        "Fusion panel accepts 1-8 models, got {}.".format(len(panel))
                    )
                cfg["analysis_models"] = panel
            judge = (uv.FUSION_JUDGE_MODEL or "").strip()
            if judge:
                cfg["model"] = judge
            if isinstance(uv.FUSION_MAX_TOOL_CALLS, int) and uv.FUSION_MAX_TOOL_CALLS > 0:
                cfg["max_tool_calls"] = uv.FUSION_MAX_TOOL_CALLS

            has_overrides = bool(set(cfg) - {"id"})
            if is_fusion and idx is None and not has_overrides:
                pass
            elif idx is not None:
                plugins[idx] = cfg  # replace in place (preserve position)
                body["plugins"] = plugins
            else:
                plugins.append(cfg)
                body["plugins"] = plugins

            if uv.FUSION_FORCE_TOOL_CALL:
                # Force Fusion by requiring a tool call, exactly as OpenRouter
                # documents. On the openrouter/fusion alias Fusion is the only
                # injected tool, so "required" forces it; if the caller also adds
                # other tools the model may pick one of those (OpenRouter's
                # documented behaviour). Skip when the caller already chose a tool,
                # a legacy function_call is present, or a Fusion plugin is disabled.
                tc_unset = ("tool_choice" not in body) or (body.get("tool_choice") is None)
                any_fusion_disabled = any(
                    isinstance(p, dict) and p.get("id") == "fusion" and p.get("enabled") is False
                    for p in plugins
                )
                if tc_unset and "function_call" not in body and not any_fusion_disabled:
                    body["tool_choice"] = "required"
            return body
        except FusionConfigError:
            raise
        except Exception:
            self.log.exception("OpenRouter Fusion filter inlet failed; passing body through unchanged")
            return body
'''


def render_openrouter_fusion_filter_source(*, marker: str) -> str:
    """Return the canonical OWUI filter source for the OpenRouter Fusion filter."""
    return (
        _FUSION_FILTER_TEMPLATE
        .replace("__MARKER__", marker)
        .replace("__FILTER_ID__", FUSION_FILTER_FUNCTION_ID)
    )
