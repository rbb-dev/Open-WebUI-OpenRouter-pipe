"""Plugin system for the OpenRouter pipe.

In package mode, this module auto-imports all plugin submodules so their
``@PluginRegistry.register`` decorators fire. In bundled mode, the bundler
already inlines all code, so decorators fire automatically.
"""

from __future__ import annotations

import logging

from .base import PluginBase, PluginContext
from .registry import PluginRegistry

__all__ = ["PluginBase", "PluginContext", "PluginRegistry"]

_plugins_init_logger = logging.getLogger(__name__)

# ── Auto-import plugin submodules in package mode ──
# Excluded modules are infrastructure, not plugins.
import importlib  # noqa: E402
import pkgutil  # noqa: E402

_PLUGINS_EXCLUDED = {"base", "registry", "_utils"}
try:
    for _finder, _name, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        _short = _name.rsplit(".", 1)[-1]
        if _short not in _PLUGINS_EXCLUDED:
            try:
                importlib.import_module(_name)
            except Exception:
                _plugins_init_logger.debug("Failed to import plugin module %s", _name, exc_info=True)
except Exception:
    pass  # Bundled mode: pkgutil may fail, but decorator code already executed

# Clean up module namespace
del importlib, pkgutil, _PLUGINS_EXCLUDED

# Explicit imports — redundant in package mode (already loaded by pkgutil above)
# but essential in compressed-bundle mode where __path__ is empty.
from . import pipe_stats as _pipe_stats  # noqa: E402, F401
from . import think_streaming as _think_streaming  # noqa: E402, F401
