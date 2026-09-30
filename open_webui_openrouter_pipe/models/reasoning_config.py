"""Reasoning model configuration and retry logic.

This module handles:
- Automatic reasoning trace enablement for supported models
- Task-specific reasoning effort overrides
- Reasoning-related error detection and retry logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..api.transforms import ResponsesBody
    from ..pipe import Pipe

from ..core.errors import OpenRouterAPIError
from ..core.utils import _select_best_effort_fallback
from ..integrations.anthropic import _is_anthropic_model_id
from .registry import ModelFamily

# Valve effort value that triggers verbosity: "max" for Claude models.
_XHIGH_EFFORT = "xhigh"
_MAX_VERBOSITY = "max"
_EFFORT_REASONING_OFF = frozenset({"none", ""})
_NO_EFFORT = "none"
_ANSWER_RESERVE_TOKENS = 64


def _normalised_effort(cfg: dict[str, Any]) -> str:
    return str(cfg.get("effort") or "").strip().lower()


class ReasoningConfigManager:
    """Manages reasoning model configuration and retry logic.

    This class encapsulates all reasoning-related configuration logic:
    - Applies reasoning preferences based on valve settings
    - Handles task-specific reasoning effort overrides
    - Detects reasoning errors and determines if retry is appropriate
    """

    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize the ReasoningConfigManager.

        Args:
            logger: Logger instance for diagnostic output
        """
        self._pipe = pipe
        self.logger = logger

    @staticmethod
    def _set_include_reasoning(responses_body: ResponsesBody, value: bool | None) -> None:
        if value is not None and "include_reasoning" not in ModelFamily.catalog_supported_parameters(responses_body.model):
            value = None
        responses_body.include_reasoning = value

    @classmethod
    def _model_requires_reasoning(cls, model_id: str) -> bool:
        return ModelFamily.reasoning_contract(model_id).get("mandatory") is True

    @staticmethod
    def _request_asks_for_no_reasoning(cfg: dict[str, Any]) -> bool:
        if cfg.get("enabled") is False or cfg.get("exclude") is True:
            return True
        effort = cfg.get("effort")
        return isinstance(effort, str) and _normalised_effort(cfg) == _NO_EFFORT

    @classmethod
    def _reserve_the_answer_room(
        cls, responses_body: ResponsesBody, cfg: dict[str, Any]
    ) -> None:
        cap = responses_body.max_output_tokens
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            return
        if cls._request_asks_for_no_reasoning(cfg):
            return
        budget = cfg.get("max_tokens")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            return
        fitted = min(budget, cap - _ANSWER_RESERVE_TOKENS)
        if fitted < 1:
            cfg.pop("max_tokens", None)
            return
        cfg["max_tokens"] = fitted

    @classmethod
    def _refuse_off_on_mandatory_model(
        cls,
        model_id: str,
        cfg: dict[str, Any],
        *,
        off_from_settings: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        if not cls._request_asks_for_no_reasoning(cfg):
            return cfg, False
        if not cls._model_requires_reasoning(model_id):
            return cfg, False
        consumed = (
            ("enabled", "exclude", "max_tokens") if off_from_settings else ("enabled", "exclude")
        )
        repaired = {k: v for k, v in cfg.items() if k not in consumed}
        repaired["enabled"] = True
        lowest = _select_best_effort_fallback(
            _NO_EFFORT,
            [e for e in ModelFamily.reasoning_contract(model_id).get("supported_efforts") or [] if e != _NO_EFFORT],
        )
        if lowest:
            repaired["effort"] = lowest
        elif _normalised_effort(repaired) in _EFFORT_REASONING_OFF:
            repaired.pop("effort", None)
        return repaired, True

    def _apply_reasoning_preferences(self, responses_body: ResponsesBody, valves: Pipe.Valves) -> str | None:
        supported = ModelFamily.catalog_supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        summary_mode = valves.REASONING_SUMMARY_MODE
        requested_summary: str | None = None
        if summary_mode != "disabled":
            requested_summary = summary_mode

        target_effort = valves.REASONING_EFFORT
        refused: bool = False

        if supports_reasoning:
            cfg: dict[str, Any] = {}
            if isinstance(responses_body.reasoning, dict):
                cfg = dict(responses_body.reasoning)
            if target_effort in _EFFORT_REASONING_OFF or (target_effort and "effort" not in cfg):
                cfg["effort"] = target_effort
            if summary_mode == "disabled":
                cfg.pop("summary", None)
            elif requested_summary and "summary" not in cfg:
                cfg["summary"] = requested_summary
            cfg.setdefault("enabled", True)
            cfg, refused = self._refuse_off_on_mandatory_model(
                responses_body.model, cfg, off_from_settings=target_effort == _NO_EFFORT
            )
            self._reserve_the_answer_room(responses_body, cfg)
            responses_body.reasoning = cfg or None
            self._set_include_reasoning(responses_body, None)
        elif supports_legacy_only:
            carried = responses_body.reasoning if isinstance(responses_body.reasoning, dict) else {}
            responses_body.reasoning = None
            off = target_effort in _EFFORT_REASONING_OFF or self._request_asks_for_no_reasoning(carried)
            self._set_include_reasoning(responses_body, not off)

        return responses_body.model if refused else None

    def _apply_task_reasoning_preferences(self, responses_body: ResponsesBody, effort: str) -> str | None:
        """Override reasoning effort for task models."""
        if not effort:
            return None
        supported = ModelFamily.catalog_supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        target_effort = effort.strip().lower()
        refused: bool = False

        if supports_reasoning:
            cfg = (
                responses_body.reasoning
                if isinstance(responses_body.reasoning, dict)
                else {}
            )
            cfg = dict(cfg) if cfg else {}
            cfg["effort"] = target_effort
            cfg.setdefault("enabled", True)
            cfg, refused = self._refuse_off_on_mandatory_model(
                responses_body.model, cfg, off_from_settings=target_effort == _NO_EFFORT
            )
            self._reserve_the_answer_room(responses_body, cfg)
            responses_body.reasoning = cfg
            self._set_include_reasoning(responses_body, None)
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in _EFFORT_REASONING_OFF
            self._set_include_reasoning(responses_body, desired)

        return responses_body.model if refused else None

    def _apply_gemini_thinking_config(
        self,
        responses_body: ResponsesBody,
        valves: Pipe.Valves,
        *,
        honour_existing_budget: bool = True,
    ) -> str | None:
        # Lazy import to avoid circular dependency
        from .registry import (
            _classify_gemini_thinking_family,
            _map_effort_to_gemini_budget,
        )

        responses_body.thinking_config = None
        if not _classify_gemini_thinking_family(ModelFamily.base_model(responses_body.model)):
            return None
        if "reasoning" not in ModelFamily.catalog_supported_parameters(responses_body.model):
            return None
        cfg = dict(responses_body.reasoning) if isinstance(responses_body.reasoning, dict) else {}
        requested = bool(responses_body.include_reasoning) or bool(cfg and cfg.get("enabled", True) and not cfg.get("exclude", False))
        if not requested:
            self._set_include_reasoning(responses_body, None)
            return None

        if valves.GEMINI_THINKING_BUDGET == 0:
            mandatory = self._model_requires_reasoning(responses_body.model)
            off = {**cfg, "effort": _NO_EFFORT} if mandatory else {"effort": _NO_EFFORT}
            off, refused = self._refuse_off_on_mandatory_model(
                responses_body.model, off, off_from_settings=False
            )
            responses_body.reasoning = off
            self._set_include_reasoning(responses_body, None)
            return responses_body.model if refused else None

        requested_budget = cfg.get("max_tokens")
        brought = (
            honour_existing_budget
            and isinstance(requested_budget, int)
            and not isinstance(requested_budget, bool)
            and requested_budget >= 1
            and cfg.get("enabled") is not False
            and cfg.get("exclude") is not True
            and _normalised_effort(cfg) != _NO_EFFORT
        )
        if brought and isinstance(requested_budget, int):
            budget = int(requested_budget)
            cfg.pop("effort", None)
        else:
            effort = _normalised_effort(cfg) or valves.REASONING_EFFORT
            budget = _map_effort_to_gemini_budget(effort, valves.GEMINI_THINKING_BUDGET)
            if not budget:
                return None
            cfg.pop("effort", None)
            cfg.setdefault("enabled", True)
        cap = responses_body.max_output_tokens
        if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1 and budget:
            budget = min(budget, cap - _ANSWER_RESERVE_TOKENS)
            if budget < 1:
                responses_body.reasoning = None
                self._set_include_reasoning(responses_body, None)
                return None
        cfg["max_tokens"] = budget
        responses_body.reasoning = cfg
        self._set_include_reasoning(responses_body, None)
        return None

    def _fit_effort_none_to_model(self, responses_body: ResponsesBody, *, settings_applied: bool) -> None:
        cfg = responses_body.reasoning
        if not isinstance(cfg, dict) or _normalised_effort(cfg) != "none":
            return
        row = ModelFamily.reasoning_contract(responses_body.model)
        if row.get("mandatory") is True:
            fitted = {key: value for key, value in cfg.items() if key != "effort"}
            lowest = _select_best_effort_fallback("none", [e for e in row.get("supported_efforts") or [] if e != "none"])
            if lowest:
                fitted["effort"] = lowest
            responses_body.reasoning = fitted or None
        elif settings_applied and "reasoning" in ModelFamily.catalog_supported_parameters(responses_body.model):
            responses_body.reasoning = {"effort": "none"}
            self._set_include_reasoning(responses_body, None)

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
        if not ModelFamily.supports_verbosity(normalized):
            return

        # Determine effective effort: request-level reasoning.effort takes
        # priority, then fall back to the valve default.
        effort = ""
        if isinstance(responses_body.reasoning, dict):
            effort = _normalised_effort(responses_body.reasoning)
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

        if self._model_requires_reasoning(responses_body.model):
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
                self._set_include_reasoning(responses_body, False)
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
                "thinking" in lowered and "cannot be modified" in lowered
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
