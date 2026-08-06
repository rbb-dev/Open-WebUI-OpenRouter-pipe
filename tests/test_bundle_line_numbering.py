"""The bundler's idea of "a line" must match the one ``ast`` and ``tokenize`` use.

``str.splitlines`` breaks on ``\\v \\f \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029`` as well
as ``\\n``; ``ast`` and ``tokenize`` line numbers count only ``\\n``. The bundler deletes
import statements by AST ``lineno`` and collapses blank runs by token position, both by
indexing into a list of lines -- so where the two disagree it edits the wrong line, and
only in the shipped artifact, which no test suite executes.

This is not hypothetical for this package: ``core/utils.py`` lists U+2028 and U+2029 as
literals among the separators it forbids in a rendered body. The second test below
fails if that stops being true, because at that point the first test is passing
vacuously and something has to say so.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLER = PROJECT_ROOT / "scripts" / "bundle_v2.py"
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"


def _bundler() -> Any:
    spec = importlib.util.spec_from_file_location("_bundle_v2_line_numbering", BUNDLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _package_sources() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
    ]


def test_the_bundlers_split_matches_python_line_numbering_for_every_module():
    """Agreement with ``readlines()``, which is how CPython itself numbers lines.

    Checked against the real package rather than a synthetic string, because the
    bundler runs against the real package and a synthetic case only proves the helper
    handles the input someone thought to write.
    """
    split = _bundler().split_physical_lines
    sources = _package_sources()
    assert sources, "no package sources found; this test is inspecting nothing"

    disagreements = [
        str(path.relative_to(PROJECT_ROOT))
        for path, source in sources
        if split(source) != io.StringIO(source).readlines()
    ]
    assert not disagreements, (
        "the bundler splits these files into different lines than Python does, so an "
        "AST lineno or a token position indexes the wrong one:\n  "
        + "\n  ".join(disagreements)
    )


def test_the_package_still_contains_the_separators_that_make_this_matter():
    """Proves the check above is load-bearing rather than trivially satisfied.

    ``splitlines`` and ``readlines`` agree on any file free of the exotic separators,
    so on such a tree the test above passes no matter how the bundler splits. It has
    to be shown that at least one real module distinguishes them.
    """
    affected = {
        str(path.relative_to(PROJECT_ROOT)): (
            len(source.splitlines(keepends=True)),
            len(io.StringIO(source).readlines()),
        )
        for path, source in _package_sources()
        if source.splitlines(keepends=True) != io.StringIO(source).readlines()
    }
    assert affected, (
        "no package module contains a non-newline line separator any more, so "
        "test_the_bundlers_split_matches_python_line_numbering_for_every_module can no "
        "longer fail and both tests should be reconsidered together"
    )


def test_an_ast_lineno_addresses_the_statement_it_names():
    """The property the bundler actually depends on, end to end.

    ``process_module_body`` deletes import statements by slicing ``source_lines`` with
    the ``lineno`` the AST reports. Any drift between the two silently deletes a
    neighbouring statement instead.
    """
    split = _bundler().split_physical_lines
    source = (
        "import os\n"
        'SEPARATORS = "  "\n'
        "import sys\n"
        "VALUE = 1\n"
    )
    lines = split(source)
    imports = [n for n in ast.parse(source).body if isinstance(n, ast.Import)]
    assert len(imports) == 2

    addressed = [lines[node.lineno - 1] for node in imports]
    assert addressed == ["import os\n", "import sys\n"], (
        f"an AST lineno addressed {addressed!r}; the bundler would delete those lines "
        "believing them to be the imports"
    )


def test_every_import_comment_the_bundler_drops_is_accounted_for():
    """A suppression lost on the way into the header must fail, not scroll past.

    The shared header is rendered from a deduplicated token set, so a ``# noqa`` or
    ``# type: ignore`` on a top-level external import cannot be carried across. The
    bundler prints what it dropped, but a print in a CI log fails nothing -- so a
    suppression that works in package mode disappears from the bundle and resurfaces
    later as a bundle-only lint or type error with no obvious cause.

    Pinning the exact set turns that into a test failure at the moment it happens. If
    this goes red, either carry the suppression another way or add it here knowing the
    bundle will not have it.
    """
    bundler = _bundler()
    modules = bundler.discover_modules(PACKAGE_DIR)
    ordered = [modules[name] for name in sorted(modules)]
    for module in ordered:
        bundler.analyze_module(module, modules)

    dropped = {
        line.strip()
        for module in ordered
        for line in getattr(module, "dropped_import_comments", [])
    }
    # Empty on purpose. A trailing comment that instructs a tool is now carried onto
    # the hoisted import, because a suppression that holds for the package and
    # vanishes from the artifact produces a bundle-only failure with no obvious
    # cause. Anything landing here is prose, which the bundle cannot carry.
    expected: set[str] = set()
    assert dropped == expected, (
        "the set of import comments the bundle cannot carry has changed.\n"
        f"  no longer dropped: {sorted(expected - dropped)}\n"
        f"  newly dropped:     {sorted(dropped - expected)}\n"
        "A newly dropped suppression is silently absent from every shipped artifact."
    )
