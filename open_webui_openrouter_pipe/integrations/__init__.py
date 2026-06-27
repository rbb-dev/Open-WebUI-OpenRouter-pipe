"""Vendor-specific integrations module.

This module contains vendor-specific integration logic for various LLM providers
when routing through OpenRouter. Each vendor's specific features and quirks are
isolated in dedicated modules.

Modules:
    anthropic: Anthropic-specific features like prompt caching
"""

from .anthropic import (
    _is_anthropic_model_id,
    _maybe_apply_anthropic_prompt_caching,
    _maybe_apply_responses_toplevel_cache_control,
)

__all__ = [
    "_maybe_apply_anthropic_prompt_caching",
    "_maybe_apply_responses_toplevel_cache_control",
    "_is_anthropic_model_id",
]
