from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

try:
    for _finder, _name, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        try:
            importlib.import_module(_name)
        except Exception:
            logger.debug("Failed to import command module %s", _name, exc_info=True)
except (AttributeError, NameError, TypeError):
    logger.debug(
        "command auto-discovery unavailable; relying on the explicit imports below",
        exc_info=True,
    )

del importlib, pkgutil

from . import dashboard_cmd as _dashboard_cmd  # noqa: F401
from . import help_cmd as _help_cmd  # noqa: F401
