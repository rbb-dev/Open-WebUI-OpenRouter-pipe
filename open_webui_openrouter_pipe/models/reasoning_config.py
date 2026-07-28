"""Reasoning model configuration and retry logic.

This module handles:
- Automatic reasoning trace enablement for supported models
- Task-specific reasoning effort overrides
- Gemini thinking_config translation
- Reasoning-related error detection and retry logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..api.transforms import ResponsesBody
    from ..pipe import Pipe

from ..core.errors import OpenRouterAPIError
from ..integrations.anthropic import _is_anthropic_model_id
from .registry import ModelFamily

# Valve effort value that triggers verbosity: "max" for Claude models.
_XHIGH_EFFORT = "xhigh"
_MAX_VERBOSITY = "max"


class ReasoningConfigManager:
    """Manages reasoning model configuration and retry logic.

    This class encapsulates all reasoning-related configuration logic:
    - Applies reasoning preferences based on valve settings
    - Handles task-specific reasoning effort overrides
    - Translates reasoning config to Gemini thinking_config
    - Detects reasoning errors and determines if retry is appropriate
    """

    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize the ReasoningConfigManager.

        Args:
            pipe: Reference to parent Pipe instance for accessing valves
            logger: Logger instance for diagnostic output
        """
        self._pipe = pipe
        self.logger = logger
        self.valves = pipe.valves

    def _apply_reasoning_preferences(self, responses_body: ResponsesBody, valves: Pipe.Valves) -> None:
        """Automatically request reasoning traces when supported and enabled."""
        if not valves.ENABLE_REASONING:
            return

        supported = ModelFamily.supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        summary_mode = valves.REASONING_SUMMARY_MODE
        requested_summary: str | None = None
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
                responses_body.include_reasoning = None
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", ""}
            responses_body.include_reasoning = desired
        else:
            responses_body.reasoning = None
            responses_body.include_reasoning = False


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
                responses_body.include_reasoning = None
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", "minimal"}
            responses_body.include_reasoning = desired
        else:
            responses_body.reasoning = None
            responses_body.include_reasoning = False


    def _apply_gemini_thinking_config(self, responses_body: ResponsesBody, valves: Pipe.Valves) -> None:
        """Translate reasoning preferences into Vertex thinking_config for Gemini models."""
        # Lazy import to avoid circular dependency
        from .registry import (
            _classify_gemini_thinking_family,
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
            responses_body.include_reasoning = False
            return

        thinking_config: dict[str, Any] = {"include_thoughts": True}
        budget = _map_effort_to_gemini_budget(effort_hint, valves.GEMINI_THINKING_BUDGET)
        # A budget of 0 (GEMINI_THINKING_BUDGET=0, the documented "disable
        # thinking" value) means thinking is off, same as None — valid budgets
        # are always >= 1. Emitting include_thoughts=True with thinking_budget=0
        # is contradictory and the provider rejects it, so disable here.
        if not budget:
            responses_body.thinking_config = None
            responses_body.include_reasoning = False
            return
        thinking_config["thinking_budget"] = budget

        responses_body.thinking_config = thinking_config
        responses_body.reasoning = None
        responses_body.include_reasoning = None

    def _apply_anthropic_verbosity(self, responses_body: ResponsesBody, valves: Pipe.Valves) -> None:
        """Map xhigh effort to verbosity: "max" for Claude Opus/Sonnet models.

        OpenRouter's ``verbosity`` parameter maps to Anthropic's
        ``output_config.effort``.  The ``"max"`` level is only supported by
        Claude 4.6 Opus/Sonnet and later, but older Claude models gracefully
        fall back to ``"high"`` on the OpenRouter side, so the broad pattern
        match (``anthropic.claude-opus-*`` / ``anthropic.claude-sonnet-*``) is
        safe.

        This is intentionally a no-op when the user has already set
        ``verbosity`` explicitly (e.g. via a custom parameter), to avoid
        overriding their choice.
        """
        from .registry import _is_claude_reasoning_model

        # Don't override if the user already set verbosity explicitly.
        if responses_body.verbosity is not None:
            return

        normalized = ModelFamily.base_model(responses_body.model)
        if not _is_claude_reasoning_model(normalized):
            return

        # Determine effective effort: request-level reasoning.effort takes
        # priority, then fall back to the valve default.
        effort = ""
        if isinstance(responses_body.reasoning, dict):
            effort = (responses_body.reasoning.get("effort") or "").strip().lower()
        if not effort:
            effort = (valves.REASONING_EFFORT or "").strip().lower()

        if effort == _XHIGH_EFFORT:
            responses_body.verbosity = _MAX_VERBOSITY

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
                responses_body.include_reasoning = False
                responses_body.reasoning = None
                responses_body.thinking_config = None
                self.logger.info(
                    "Retrying without reasoning for model '%s' after provider rejected include_reasoning without thinking.",
                    responses_body.model,
                )
                return True

        return False

    def _should_retry_dropping_signed_reasoning(
        self,
        error: OpenRouterAPIError,
        responses_body: ResponsesBody,
    ) -> bool:
        """Strip replayed thinking blocks and retry when an Anthropic provider rejects a stale thinking-block signature on a 400."""
        if getattr(error, "status", None) != 400:
            return False
        target_model = getattr(responses_body, "api_model", None)
        if not (isinstance(target_model, str) and target_model.strip()):
            target_model = str(getattr(responses_body, "model", "") or "")
        if not _is_anthropic_model_id(target_model):
            return False
        message_candidates = [
            error.upstream_message,
            error.openrouter_message,
            str(error),
        ]
        is_signature_error = False
        for message in message_candidates:
            if not isinstance(message, str):
                continue
            lowered = message.lower()
            if ("signature" in lowered and "thinking" in lowered) or (
                "thinking block" in lowered and "cannot be modified" in lowered
            ):
                is_signature_error = True
                break
        if not is_signature_error:
            return False
        if not self._strip_replayed_reasoning(responses_body):
            return False
        self.logger.info(
            "Retrying without replayed thinking blocks for model '%s' after a thinking-block signature was rejected.",
            responses_body.model,
        )
        return True

    @staticmethod
    def _strip_replayed_reasoning(responses_body: ResponsesBody) -> bool:
        """Remove replayed reasoning items and message-level reasoning_details from the request input; return True if anything changed."""
        input_items = getattr(responses_body, "input", None)
        if not isinstance(input_items, list):
            return False
        changed = False
        cleaned = []
        for item in input_items:
            if isinstance(item, dict):
                if item.get("type") == "reasoning":
                    changed = True
                    continue
                if "reasoning_details" in item:
                    item = {k: v for k, v in item.items() if k != "reasoning_details"}
                    changed = True
            cleaned.append(item)
        if changed:
            responses_body.input = cleaned
        return changed
