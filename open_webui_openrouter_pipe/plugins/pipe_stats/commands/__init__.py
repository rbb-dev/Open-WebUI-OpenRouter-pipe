"""Import all command modules to trigger @register_command decorators.

In package mode, pkgutil discovers modules automatically.  In bundled mode
(especially compressed bundles where ``__path__`` is empty), pkgutil finds
nothing — the explicit imports below guarantee commands are always registered.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

_ps_cmd_logger = logging.getLogger(__name__)

try:
    for _finder, _name, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        try:
            importlib.import_module(_name)
        except Exception:
            _ps_cmd_logger.debug("Failed to import command module %s", _name, exc_info=True)
except Exception:
    pass  # Bundled mode

del importlib, pkgutil

# Explicit imports — redundant in package mode (already loaded by pkgutil above)
# but essential in compressed-bundle mode where pkgutil returns nothing.
from . import config_cmd as _config_cmd  # noqa: E402, F401
from . import health_cmd as _health_cmd  # noqa: E402, F401
from . import help_cmd as _help_cmd  # noqa: E402, F401
from . import stats_cmd as _stats_cmd  # noqa: E402, F401
