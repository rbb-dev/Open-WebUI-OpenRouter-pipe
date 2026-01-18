"""Reasoning model configuration and retry logic.

This module handles:
- Automatic reasoning trace enablement for supported models
- Task-specific reasoning effort overrides
- Gemini thinking_config translation
- Reasoning-related error detection and retry logic
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
import logging
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..api.transforms import ResponsesBody

from .registry import ModelFamily
from ..core.errors import OpenRouterAPIError
# Lazy import to avoid circular dependency.

LOGGER = logging.getLogger(__name__)


class ReasoningConfigManager:
    """Manages reasoning model configuration and retry logic.

    This class encapsulates all reasoning-related configuration logic:
    - Applies reasoning preferences based on valve settings
    - Handles task-specific reasoning effort overrides
    - Translates reasoning config to Gemini thinking_config
    - Detects reasoning errors and determines if retry is appropriate
    """

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize the ReasoningConfigManager.

        Args:
            pipe: Reference to parent Pipe instance for accessing valves
            logger: Logger instance for diagnostic output
        """
        self._pipe = pipe
        self.logger = logger
        self.valves = pipe.valves

    # ----------------------------------------------------------------------
    # _apply_reasoning_preferences (35 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_reasoning_preferences(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Automatically request reasoning traces when supported and enabled."""
        if not valves.ENABLE_REASONING:
            return

        supported = ModelFamily.supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        summary_mode = valves.REASONING_SUMMARY_MODE
        requested_summary: Optional[str] = None
        if summary_mode != "disabled":
            requested_summary = summary_mode

        target_effort = valves.REASONING_EFFORT

        if supports_reasoning:
            cfg: dict[str, Any] = {}
            if isinstance(responses_body.reasoning, dict):
                cfg = dict(responses_body.reasoning)
            if target_effort and "effort" not in cfg:
                cfg["effort"] = target_effort
            if requested_summary and "summary" not in cfg:
                cfg["summary"] = requested_summary
            cfg.setdefault("enabled", True)
            responses_body.reasoning = cfg or None
            if getattr(responses_body, "include_reasoning", None) is not None:
                setattr(responses_body, "include_reasoning", None)
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", ""}
            setattr(responses_body, "include_reasoning", desired)
        else:
            responses_body.reasoning = None
            setattr(responses_body, "include_reasoning", False)


    # ----------------------------------------------------------------------
    # _apply_task_reasoning_preferences (29 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_task_reasoning_preferences(self, responses_body: ResponsesBody, effort: str) -> None:
        """Override reasoning effort for task models."""
        if not effort:
            return
        supported = ModelFamily.supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        target_effort = effort.strip().lower()

        if supports_reasoning:
            cfg = (
                responses_body.reasoning
                if isinstance(responses_body.reasoning, dict)
                else {}
            )
            cfg = dict(cfg) if cfg else {}
            cfg["effort"] = target_effort
            cfg.setdefault("enabled", True)
            responses_body.reasoning = cfg
            if getattr(responses_body, "include_reasoning", None) is not None:
                setattr(responses_body, "include_reasoning", None)
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", "minimal"}
            setattr(responses_body, "include_reasoning", desired)
        else:
            responses_body.reasoning = None
            setattr(responses_body, "include_reasoning", False)


    # ----------------------------------------------------------------------
    # _apply_gemini_thinking_config (47 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_gemini_thinking_config(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Translate reasoning preferences into Vertex thinking_config for Gemini models."""
        # Lazy import to avoid circular dependency
        from .registry import (
            _classify_gemini_thinking_family,
            _map_effort_to_gemini_level,
            _map_effort_to_gemini_budget,
        )

        normalized_model = ModelFamily.base_model(responses_body.model)
        family = _classify_gemini_thinking_family(normalized_model)
        if not family:
            responses_body.thinking_config = None
            return

        reasoning_cfg = (
            responses_body.reasoning if isinstance(responses_body.reasoning, dict) else {}
        )
        include_flag = getattr(responses_body, "include_reasoning", None)
        effort_hint = (reasoning_cfg.get("effort") or "").strip().lower()
        if not effort_hint:
            effort_hint = valves.REASONING_EFFORT

        enabled = reasoning_cfg.get("enabled", True)
        exclude = reasoning_cfg.get("exclude", False)
        if effort_hint == "none":
            enabled = False

        reasoning_requested = bool(include_flag) or (reasoning_cfg and enabled and not exclude)
        if not reasoning_requested:
            responses_body.thinking_config = None
            setattr(responses_body, "include_reasoning", False)
            return

        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if family == "gemini-3":
            level = _map_effort_to_gemini_level(effort_hint, valves.GEMINI_THINKING_LEVEL)
            if level is None:
                responses_body.thinking_config = None
                setattr(responses_body, "include_reasoning", False)
                return
            thinking_config["thinking_level"] = level
        elif family == "gemini-2.5":
            budget = _map_effort_to_gemini_budget(effort_hint, valves.GEMINI_THINKING_BUDGET)
            if budget is None:
                responses_body.thinking_config = None
                setattr(responses_body, "include_reasoning", False)
                return
            thinking_config["thinking_budget"] = budget

        responses_body.thinking_config = thinking_config
        responses_body.reasoning = None
        setattr(responses_body, "include_reasoning", None)

    # ----------------------------------------------------------------------
    # _should_retry_without_reasoning (38 lines)
    # ----------------------------------------------------------------------

    @timed
    def _should_retry_without_reasoning(
        self,
        error: OpenRouterAPIError,
        responses_body: ResponsesBody,
    ) -> bool:
        """Return True when we can retry the request after disabling reasoning."""

        include_flag = getattr(responses_body, "include_reasoning", None)
        has_reasoning_dict = bool(getattr(responses_body, "reasoning", None))
        has_thinking_config = bool(getattr(responses_body, "thinking_config", None))
        if not any((include_flag, has_reasoning_dict, has_thinking_config)):
            return False

        trigger_phrases = (
            "thinking_config.include_thoughts is only enabled when thinking is enabled",
            "include_thoughts is only enabled when thinking is enabled",
        )
        message_candidates = [
            error.upstream_message,
            error.openrouter_message,
            str(error),
        ]

        for message in message_candidates:
            if not isinstance(message, str):
                continue
            lowered = message.lower()
            if any(trigger in lowered for trigger in trigger_phrases):
                setattr(responses_body, "include_reasoning", False)
                responses_body.reasoning = None
                responses_body.thinking_config = None
                self.logger.info(
                    "Retrying without reasoning for model '%s' after provider rejected include_reasoning without thinking.",
                    responses_body.model,
                )
                return True

        return False
