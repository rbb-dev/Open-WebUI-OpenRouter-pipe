"""Plugin system for the OpenRouter pipe.

Plugins self-register via the ``@PluginRegistry.register`` decorator when their
module executes. This package discovers and imports plugin sub-packages so those
decorators fire — naming no specific plugin, in every packaging mode:

- Package mode: ``pkgutil.iter_modules`` walks the real ``plugins/`` directory.
- Flat bundle: every module body is inlined and executed at import, so plugins
  register on their own; discovery here is a harmless no-op.
- Compressed bundle: modules are lazy-loaded by the bundle's import hook, so
  ``pkgutil`` finds nothing (``__path__`` is empty). We instead ask the bundle's
  finder for its module manifest and import the plugin packages it lists.

Infrastructure modules (``base``, ``registry``, ``_utils``) are never plugins.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

from .base import PluginBase, PluginContext
from .registry import PluginRegistry

__all__ = ["PluginBase", "PluginContext", "PluginRegistry"]

_plugins_init_logger = logging.getLogger(__name__)

# Infrastructure — the framework itself, never a plugin.
_PLUGINS_EXCLUDED = frozenset({"base", "registry", "_utils"})


def _import_plugin_module(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception:
        _plugins_init_logger.debug("Failed to import plugin module %s", name, exc_info=True)


def _discover_plugins() -> None:
    """Import every plugin sub-package so its ``@register`` decorator fires.

    Names no specific plugin: discovery is purely by enumeration of what is
    present, from the filesystem (package mode) or the bundle manifest (bundle).
    """
    prefix = __name__ + "."
    seen: set[str] = set()

    # 1) Package mode — walk the real filesystem package.
    try:
        for mod in pkgutil.iter_modules(__path__, prefix):
            short = mod.name[len(prefix):]
            if short not in _PLUGINS_EXCLUDED:
                _import_plugin_module(mod.name)
                seen.add(mod.name)
    except Exception:
        # Bundled mode: __path__ may be empty; the manifest branch handles it.
        _plugins_init_logger.debug("pkgutil plugin discovery unavailable", exc_info=True)

    # 2) Bundle mode — ask any bundle finder for its module manifest and import
    #    the top-level plugin packages it lists. Fully generic; no plugin names.
    for finder in list(sys.meta_path):
        lister = getattr(finder, "bundled_module_names", None)
        if not callable(lister):
            continue
        try:
            names = lister()
        except Exception:
            continue
        if not isinstance(names, (list, tuple, set, frozenset)):
            continue
        for full in names:
            if not full.startswith(prefix):
                continue
            short = full[len(prefix):]
            # Only top-level plugin packages (e.g. "<pkg>.plugins.<name>").
            if "." in short or short in _PLUGINS_EXCLUDED:
                continue
            if full not in seen:
                _import_plugin_module(full)
                seen.add(full)


_discover_plugins()
