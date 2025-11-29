"""Pytest bootstrap helpers to keep local runs deterministic.

This plugin is loaded via ``pytest.ini`` using ``-p`` so that it executes
before pytest auto-loads entry point plugins.  We use the opportunity to
set environment variables that otherwise must be exported manually:

* Force ``tempfile`` to use ``/tmp`` instead of ``/mnt/c`` on WSL/Windows,
  avoiding ``FileNotFoundError`` when pytest's FD capture truncates files.
* Disable automatic loading of globally-installed pytest plugins.  The host
  Python has dozens of unrelated plugins that slow down or hang collection.
"""

from __future__ import annotations

import os
import tempfile
from typing import Final

_POSIX_TMPDIR: Final[str] = "/tmp"


def _force_posix_tmpdir() -> None:
    """Ensure ``tempfile`` operates on the Linux side of WSL."""

    if os.name != "posix" or not os.path.isdir(_POSIX_TMPDIR):
        return

    current = (
        os.environ.get("TMPDIR")
        or os.environ.get("TMP")
        or os.environ.get("TEMP")
    )
    if current and current.startswith(_POSIX_TMPDIR):
        return

    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = _POSIX_TMPDIR
    tempfile.tempdir = _POSIX_TMPDIR


def _disable_global_pytest_plugins() -> None:
    """Prevent dozens of unrelated system plugins from being auto-loaded."""

    if not os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


_force_posix_tmpdir()
_disable_global_pytest_plugins()
