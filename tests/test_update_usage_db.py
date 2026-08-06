"""The cost-ingest drain must survive a bad LOG_LEVEL rather than die at startup.

`scripts/update_usage_db.py` is the operator-facing script that drains OpenRouter cost
snapshots into Postgres. It runs unattended, so the failure that matters is the one
that happens before it does any work: `getattr(logging, name, ERROR)` returns ANY
attribute of the logging module, so `LOG_LEVEL=BASIC_FORMAT` yields a format string
and `basicConfig` raises `ValueError: Unknown level`. A one-character typo in an
environment variable stopped the drain instead of logging at ERROR.

The same defect was fixed across the package by `core.logging_system.resolve_level`.
This script cannot use it: it deliberately imports nothing from the pipe package so it
stays runnable on its own, so it carries the equivalent inline and this pins it.

Loaded by path with the two drivers stubbed. `psycopg` is not a test dependency, and
importing the module is the only way to reach the real function -- reimplementing it
here would test the copy rather than the script.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_usage_db.py"


@pytest.fixture(scope="module")
def drain():
    stubs = {}
    for name in ("psycopg", "redis"):
        if name not in sys.modules:
            stubs[name] = types.ModuleType(name)
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        name = "_update_usage_db_under_test"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves sys.modules[cls.__module__] while building the class
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(name, None)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.parametrize(
    "value",
    [
        "BASIC_FORMAT",
        "NOTSET",
        "nonsense",
        "",
        "   ",
        "root",
        "raiseExceptions",
        "10; DROP TABLE",
    ],
)
def test_an_unusable_log_level_falls_back_instead_of_killing_the_drain(drain, value, monkeypatch):
    """Asserts the level the script actually installed, not that a call did not raise.

    BASIC_FORMAT is the live one: it is a real uppercase attribute of the logging
    module holding a format string, so the old `getattr` returned it and basicConfig
    rejected it. NOTSET resolves to 0, which is "inherit" rather than a threshold, and
    would have turned the drain's own logger wide open.
    """
    monkeypatch.setenv("LOG_LEVEL", value)
    drain.configure_logging()
    assert drain.LOGGER.level == logging.ERROR, (
        f"LOG_LEVEL={value!r} left the drain at level {drain.LOGGER.level}; it must fall "
        "back to ERROR. A level that is not a real threshold either crashes basicConfig "
        "or silently changes how much this unattended job logs."
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),
        (" warning ", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_a_real_level_is_honoured(drain, value, expected, monkeypatch):
    """Anti-vacuity: a fallback that swallowed everything would pass the test above."""
    monkeypatch.setenv("LOG_LEVEL", value)
    drain.configure_logging()
    assert drain.LOGGER.level == expected


def test_the_debug_flag_wins_over_the_environment(drain, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    drain.configure_logging(debug=True)
    assert drain.LOGGER.level == logging.DEBUG
