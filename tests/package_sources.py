"""One read-and-parse of each source tree, shared by the tests that scan it.

Walking `open_webui_openrouter_pipe/` with `rglob("*.py")` and parsing all 108 files
costs about 0.38 s (0.18 s reading, 0.20 s parsing) and produced byte-identical results
in every scanner that did it. Six of those walks now share this cache; thirteen
`rglob` walks remain in `tests/`, deliberately:

- `test_owui_seam` records the files that FAIL to parse, which this cache omits.
- `test_silent_suppression_inventory` and `test_swallowed_failure_diagnostics` are
  observed by `test_every_source_census_walks_the_same_roots`, which monkeypatches
  `Path.rglob` to record what each census reads; a cached walk records nothing on the
  second call and would disarm that guard.
- `test_bundle_line_numbering`, `test_bundle_string_fidelity` and
  `test_bundle_name_collisions` compare the package tree against a built artifact and
  must see EVERY file, including one that would not parse -- which this cache omits.
- `test_warn_latch_isolation` and `test_async_test_markers` walk with a regex over raw
  text and never parse; they could adopt this and have simply not been converted.

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
