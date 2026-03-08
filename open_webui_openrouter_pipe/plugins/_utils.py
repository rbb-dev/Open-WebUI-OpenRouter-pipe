"""Shared utilities for plugin subsystem."""

from __future__ import annotations

import logging
from typing import Any

_plugin_utils_log = logging.getLogger(__name__)


def get_owui_app() -> Any | None:
    """Import and return the OWUI FastAPI app, or ``None`` if unavailable."""
    try:
        from open_webui.main import app
        return app
    except ImportError:
        _plugin_utils_log.debug("OWUI app not available")
        return None


def ensure_route_before_spa(app: Any) -> None:
    """Push the SPA catch-all mount to the end of ``app.routes``.

    Starlette evaluates routes in list order.  OWUI's SPAStaticFiles mount
    at ``/`` catches everything, so dynamically-registered API routes must
    appear *before* it.  This helper moves the SPA mount to the end,
    allowing all prior routes to match first.  Idempotent.
    """
    for i, route in enumerate(app.routes):
        if getattr(route, "name", "") == "spa-static-files":
            app.routes.append(app.routes.pop(i))
            break
