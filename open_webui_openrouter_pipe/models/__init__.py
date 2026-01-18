"""Model management subsystem.

This module handles model catalog synchronization, reasoning configuration,
and model registry management for the OpenRouter pipe.

Submodules:
    catalog_manager: Model metadata synchronization from OpenRouter to OWUI
    reasoning_config: Reasoning model configuration and retry logic
    registry: Model capability registry and family classification
"""

from .catalog_manager import ModelCatalogManager
from .reasoning_config import ReasoningConfigManager

__all__ = [
    "ModelCatalogManager",
    "ReasoningConfigManager",
]
