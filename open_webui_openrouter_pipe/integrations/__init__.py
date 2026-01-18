"""Vendor-specific integrations module.

This module contains vendor-specific integration logic for various LLM providers
when routing through OpenRouter. Each vendor's specific features and quirks are
isolated in dedicated modules.

Modules:
    anthropic: Anthropic-specific features like prompt caching
"""

from .anthropic import _maybe_apply_anthropic_prompt_caching

__all__ = [
    "_maybe_apply_anthropic_prompt_caching",
]
