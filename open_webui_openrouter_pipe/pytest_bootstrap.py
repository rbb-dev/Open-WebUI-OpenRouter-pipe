from __future__ import annotations

import os
import tempfile
from typing import Final

_POSIX_TMPDIR: Final[str] = "/tmp"


def _force_posix_tmpdir() -> None:
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
    if not os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


_force_posix_tmpdir()
_disable_global_pytest_plugins()
