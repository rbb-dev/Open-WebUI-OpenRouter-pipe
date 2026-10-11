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
from functools import lru_cache
from pathlib import Path

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
