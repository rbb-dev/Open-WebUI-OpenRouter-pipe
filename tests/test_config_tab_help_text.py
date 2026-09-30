"""Config-tab valve editing: every valve survives edit -> save -> round-trip, and a save stores only the custom subset."""

from __future__ import annotations

import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_webui_openrouter_pipe.core.config import EncryptedStr, Valves
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import config_service as cs
from typing import Any, cast


from tests._config_tab_shared import _FIXTURES
_HELP_KEY = "sk-help"


def _record_refresh_failure(cache_seconds: int, api_key: str = _HELP_KEY) -> float:
    """One failing fetch through the real recorder, for one named credential.

    The recorder hands back the window that failure computed, which is the same value
    `ensure_loaded` gives the settle, so a caller that wants the wait reads it here
    rather than off the shared freshness clock.
    """
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    return OpenRouterModelRegistry._record_refresh_failure(
        RuntimeError("boom"), cache_seconds, api_key
    )


def _record_refresh_success(cache_seconds: int, api_key: str = _HELP_KEY) -> None:
    """One successful fetch through the real recorder, under the same convention."""
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    OpenRouterModelRegistry._record_refresh_success(cache_seconds, api_key)
