"""Config-tab valve editing: every valve survives edit -> save -> round-trip, and a save stores only the custom subset."""

from __future__ import annotations

import inspect
import re
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_webui_openrouter_pipe.core.config import EncryptedStr, Valves
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import config_service as cs
from typing import Any, cast


from tests._config_tab_shared import _FIXTURES
