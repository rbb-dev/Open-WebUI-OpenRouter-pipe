"""One read-and-parse of each source tree, shared by the tests that scan it.

Keyed by root, so a caller scanning `scripts/` or `filters/` shares the cache with any
other caller scanning the same root and pays nothing for the roots it does not touch.

CONTRACT: the returned trees are SHARED. Walk them, read them, never mutate them --
`ast.fix_missing_locations`, attribute assignment on a node, or anything from
`ast.NodeTransformer` would be seen by every later test in the session. Callers that
need to transform must `ast.parse` their own copy; the source text is right there.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.metadata
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def parsed_sources(root: str) -> tuple[tuple[Path, str, ast.Module], ...]:
    """(path, source, tree) for every `.py` under *root*, sorted, parsed once.

    *root* is repo-relative (`"open_webui_openrouter_pipe"`, `"scripts"`, `"filters"`).
    A file that will not parse is omitted rather than raising, so one syntactically
    broken scratch file cannot take down every census at import time; callers that care
    about coverage assert a floor on what they found.
    """
    out: list[tuple[Path, str, ast.Module]] = []
    for path in sorted((REPO_ROOT / root).rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            out.append((path, source, ast.parse(source, filename=str(path))))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
    return tuple(out)


@lru_cache(maxsize=None)
def open_webui_package_dir() -> Path:
    try:
        located = Path(str(importlib.metadata.distribution("open-webui").locate_file("open_webui")))
    except importlib.metadata.PackageNotFoundError:
        located = None
    if located is not None and (located / "__init__.py").is_file():
        return located
    spec = importlib.machinery.PathFinder.find_spec("open_webui")
    for folder in (spec.submodule_search_locations or []) if spec is not None else []:
        if (Path(folder) / "__init__.py").is_file():
            return Path(folder)
    first = (
        "no distribution named open-webui is installed"
        if located is None
        else f"it names {located}, which holds no __init__.py"
    )
    pytest.fail(
        "Open WebUI's source files were not found, and this test reads them from disk without importing Open WebUI.\n"
        "Looked in two places:\n"
        f"  1. the open-webui distribution's metadata (importlib.metadata): {first}\n"
        "  2. a top-level open_webui package on sys.path, found without importing it (importlib.machinery.PathFinder): "
        "none\n"
        "Install open-webui into the interpreter that runs the tests, or put the folder that contains open_webui "
        "on PYTHONPATH.",
        pytrace=False,
    )
