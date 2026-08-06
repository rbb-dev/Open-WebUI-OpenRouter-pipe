"""One subprocess per artifact: load it the way Open WebUI does, answer every question.

Three tests each spawned their own interpreter and compiled the same 2.5 MB artifact --
two of them the very same file -- for ~12s of a ~90s suite. The load is the expensive
part and it is identical for all of them, so it happens once per artifact and the
answers are cached.

Isolation is unchanged: the load still happens in a separate interpreter, because
executing a bundle installs package aliases and rewrites ``sys.modules`` for whatever
process does it, and popping the probe's own module name afterwards undoes none of it.

The loader is Open WebUI's own: a bare ``types.ModuleType`` named ``function_<id>``,
then ``exec(compile(source))`` into its ``__dict__``. That is what ``plugin.py`` does,
and it differs from ``spec_from_file_location`` in ways the artifact can observe -- the
module gets no ``__file__`` or ``__spec__``, and the pipe reads its own id off the
``function_`` prefix in ``__name__``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

_SCRIPT = '''
import importlib.util, json, sys, types

sys.path.insert(0, __TESTS__)
import owui_stubs  # noqa: F401 - the stand-ins, without conftest's package import

import logging

out = {"ok": False, "error": None, "schema": {}, "proxy": {}}
try:
    path = __ARTIFACT__
    source = open(path, "r", encoding="utf-8").read()
    module = types.ModuleType("function_owui_faithful_smoke")
    sys.modules["function_owui_faithful_smoke"] = module
    exec(compile(source, path, "exec"), module.__dict__)

    pipe_cls = getattr(module, "Pipe", None)
    if pipe_cls is None:
        raise AssertionError("Pipe class missing after exec")
    valves = getattr(pipe_cls, "Valves", None)
    user_valves = getattr(pipe_cls, "UserValves", None)
    if valves is None:
        raise AssertionError("Pipe.Valves missing")
    if user_valves is None:
        raise AssertionError("Pipe.UserValves missing")
    out["schema"] = {
        "valves": len(valves.model_json_schema().get("properties") or {}),
        "user_valves": len(user_valves.model_json_schema().get("properties") or {}),
    }

    proxy = module.__dict__.get("logging")
    attrs = sorted(getattr(module, "_SUBMODULE_ATTRS", None) or [])
    # A stdlib logging submodule is an attribute of the package only once something has
    # imported it. Leaving that to chance made the compared set a function of whatever
    # the probe process happened to load: with the real Open WebUI it was
    # ['config', 'debug'], with the stand-ins ['debug'], and the module-valued half of
    # the shadowing rule went uncompared without any assertion noticing.
    for name in attrs:
        if importlib.util.find_spec(f"logging.{name}") is not None:
            importlib.import_module(f"logging.{name}")
    collisions, wrong = [], []
    for name in attrs:
        expected = getattr(logging, name, None)
        if expected is None:
            continue
        if name in getattr(proxy, "_children", frozenset()):
            continue
        collisions.append(name)
        if getattr(proxy, name, None) is not expected:
            wrong.append(name)
    out["proxy"] = {
        "has_proxy": proxy is not None,
        "is_stdlib": proxy is logging,
        "collisions": collisions,
        "wrong": wrong,
    }
    out["ok"] = True
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("PROBE_JSON:" + json.dumps(out))
'''

_CACHE: dict[str, dict] = {}


def probe(artifact: Path) -> dict:
    """Load ``artifact`` in a fresh interpreter and return every observation about it.

    Cached per artifact path, so the second caller pays nothing. Never raises for a
    failed load: the caller gets ``ok=False`` and the reason, because a probe that
    aborts silently is indistinguishable from one whose checks all passed.
    """
    key = str(Path(artifact).resolve())
    if key in _CACHE:
        return _CACHE[key]

    script = _SCRIPT.replace("__TESTS__", repr(str(TESTS_DIR))).replace(
        "__ARTIFACT__", repr(key)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("PROBE_JSON:")), None
    )
    if line is None:
        _CACHE[key] = {
            "ok": False,
            "error": (
                f"probe produced no result (rc={result.returncode}); "
                f"stdout={result.stdout[-400:]!r} stderr={result.stderr[-400:]!r}"
            ),
            "schema": {},
            "proxy": {},
        }
    else:
        _CACHE[key] = json.loads(line[len("PROBE_JSON:") :])
    return _CACHE[key]
