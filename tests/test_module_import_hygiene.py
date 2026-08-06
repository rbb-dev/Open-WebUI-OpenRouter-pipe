"""Every module must import standalone, with none of the package pre-loaded.

The rest of the suite imports ``Pipe`` before anything else (conftest does it at
collection time), which warms ``sys.modules`` with the whole dependency graph in one
fixed order. Under that warm cache a circular import is invisible: whichever module
loses the race finds its partner already fully initialised. Open WebUI, the bundler's
topological sort, and ``python -c "import x"`` all enter at arbitrary modules, so a
cycle that the suite cannot see still breaks the product.

The sweep runs in a subprocess. Clearing the package out of ``sys.modules`` and
re-executing every module body has side effects that do not live in ``sys.modules`` --
logger handler lists, registry singletons, module-level caches -- and restoring the
mapping does not undo them. A separate interpreter cannot contaminate this one.
"""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import open_webui_openrouter_pipe as pkg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFIX = pkg.__name__

IS_BUNDLED = bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH"))

if not IS_BUNDLED and not hasattr(pkg, "__path__"):
    raise RuntimeError(
        f"{PREFIX} is not a package and OWUI_PIPE_BUNDLE_PATH is unset. The skip below "
        "must never engage in package mode -- a silently-disabled guard is how the "
        "cycle regression it exists for shipped green in the first place."
    )

bundled = pytest.mark.skipif(
    IS_BUNDLED,
    reason=(
        "a bundle aliases every submodule name to its own module object, so clearing "
        "them and re-importing resolves to the on-disk package instead -- the sweep "
        "would test source while claiming to test the bundle"
    ),
)

_SWEEP = """
import importlib, json, pkgutil, sys
sys.path.insert(0, {tests!r})
import owui_stubs  # the stubs only: conftest also imports Pipe, which this sweep deletes
import open_webui_openrouter_pipe as pkg

PREFIX = pkg.__name__
targets = {targets!r} or sorted(m.name for m in pkgutil.walk_packages(pkg.__path__, PREFIX + "."))
failures = []
imported = []
for name in targets:
    for key in [k for k in sys.modules if k == PREFIX or k.startswith(PREFIX + ".")]:
        del sys.modules[key]
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures.append(f"{{name}}: {{type(exc).__name__}}: {{exc}}")
    else:
        imported.append(name)
print("@@RESULT@@" + json.dumps({{"imported": imported, "failures": failures}}))
"""


def _sweep_chunk(targets: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _SWEEP.format(tests=str(PROJECT_ROOT / "tests"), targets=targets)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        timeout=300,
    )
    marker = "@@RESULT@@"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
    assert line, (
        "sweep subprocess produced no result.\n"
        f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    import json

    return json.loads(line[len(marker) :])


_CYCLE_PARTICIPANTS = [
    "open_webui_openrouter_pipe.api.transforms",
    "open_webui_openrouter_pipe.requests.transformer",
    "open_webui_openrouter_pipe.requests.fusion_engine",
    "open_webui_openrouter_pipe.requests.nonstreaming_adapter",
]


def _discover() -> list[str]:
    """Every module under the package, plus the named cycle participants.

    A curated subset was tried and was wrong: the package root is a PEP 562 lazy
    loader, so importing it executes no submodule at all, and Open WebUI's real entry
    -- ``from open_webui_openrouter_pipe import Pipe`` -- makes ``.pipe`` the first
    module to run. Deciding by hand which modules "can" be entered first left 30 of
    107 module bodies unexecuted, ``.pipe`` among them, while the test kept its name
    and stayed green. The full sweep costs 1.6s more than the subset, measured, so
    there is nothing to buy by choosing.
    """
    return sorted(
        {*(m.name for m in pkgutil.walk_packages(pkg.__path__, PREFIX + ".")), *_CYCLE_PARTICIPANTS}
    )


_SWEEP_CACHE: dict[str, dict] = {}


def _sweep() -> dict:
    """Import every module first, from a cleared package, spread over several interpreters.

    Each chunk still clears the package out of ``sys.modules`` before every single import,
    so every module is the first package module its interpreter loads -- the property is
    per-module, not per-process, and splitting the list cannot weaken it. Serially the
    sweep costs ~20s of a ~50s suite for work that is entirely independent per module.
    """
    if "all" not in _SWEEP_CACHE:
        names = _discover()
        workers = min(4, max(1, os.cpu_count() or 2))
        chunks = [c for c in (names[i::workers] for i in range(workers)) if c]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(_sweep_chunk, chunks))
        # `imported` is recorded INSIDE the subprocess loop, so it is what the work
        # produced. Reporting the target list back would make the assertion below
        # `_discover() == _discover()`, which is true however few modules ran.
        _SWEEP_CACHE["all"] = {
            "imported": sorted(i for part in parts for i in part["imported"]),
            "failures": sorted(f for part in parts for f in part["failures"]),
        }
    return _SWEEP_CACHE["all"]


@bundled
def test_every_module_imports_as_the_entry_point():
    result = _sweep()
    expected = set(_discover())
    # `imported` is appended only after the import RETURNS, and a module that raised is
    # named in `failures`. Their union is therefore what actually executed or tried to:
    # appending before the call would let a `continue` report a full list having run
    # nothing.
    reached = set(result["imported"]) | {f.split(":")[0] for f in result["failures"]}
    assert reached == expected, (
        "the modules a subprocess actually imported do not match what walk_packages "
        "finds. A chunk that goes missing leaves module bodies unexecuted while this "
        "test stays green.\n"
        f"missing: {sorted(expected - reached)}\n"
        f"unexpected: {sorted(reached - expected)}"
    )
    assert PREFIX + ".pipe" in reached, (
        "open_webui_openrouter_pipe.pipe is what Open WebUI's `from ... import Pipe` "
        "executes first; a sweep that omits it cannot see the regression it exists for"
    )
    assert not result["failures"], "modules that cannot be imported first:\n  " + "\n  ".join(
        result["failures"]
    )


@bundled
def test_known_cycle_participants_import_first():
    """Named guards for the modules a real cycle regression went through.

    The sweep imports these under the same rules as the other 100-odd; these exist so a
    reintroduced cycle names the module instead of hiding in a list of 106. Membership is
    asserted too, so renaming one of them retires its guard loudly rather than leaving a
    filter that matches nothing and passes.
    """
    modules = [
        "open_webui_openrouter_pipe.api.transforms",
        "open_webui_openrouter_pipe.requests.transformer",
        "open_webui_openrouter_pipe.requests.fusion_engine",
        "open_webui_openrouter_pipe.requests.nonstreaming_adapter",
    ]
    result = _sweep()
    missing = [name for name in modules if name not in result["imported"]]
    assert not missing, f"named cycle participants are no longer swept: {missing}"
    named = [f for f in result["failures"] if f.split(":")[0] in set(modules)]
    assert not named, "\n  ".join(named)
