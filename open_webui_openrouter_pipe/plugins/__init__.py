from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

from .base import PluginBase, PluginContext
from .registry import PluginRegistry

__all__ = ["PluginBase", "PluginContext", "PluginRegistry"]

logger = logging.getLogger(__name__)

_PLUGINS_EXCLUDED = frozenset({"base", "registry", "_utils"})


def _import_plugin_module(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception:
        logger.debug("Failed to import plugin module %s", name, exc_info=True)


def _discover_plugins() -> None:
    prefix = __name__ + "."
    seen: set[str] = set()

    try:
        for mod in pkgutil.iter_modules(__path__, prefix):
            short = mod.name[len(prefix):]
            if short not in _PLUGINS_EXCLUDED:
                _import_plugin_module(mod.name)
                seen.add(mod.name)
    except Exception:
        logger.debug("pkgutil plugin discovery unavailable", exc_info=True)

    for finder in list(sys.meta_path):
        lister = getattr(finder, "bundled_module_names", None)
        if not callable(lister):
            continue
        try:
            names = lister()
        except Exception:
            logger.warning(
                "plugin discovery: bundle finder %r could not list its modules; any "
                "plugins it carries will not load",
                type(finder).__name__,
                exc_info=True,
            )
            continue
        if not isinstance(names, (list, tuple, set, frozenset)):
            continue
        for full in names:
            if not full.startswith(prefix):
                continue
            short = full[len(prefix):]
            if "." in short or short in _PLUGINS_EXCLUDED:
                continue
            if full not in seen:
                _import_plugin_module(full)
                seen.add(full)


_discover_plugins()
