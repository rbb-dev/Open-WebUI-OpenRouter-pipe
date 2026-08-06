"""Every collected async test must carry an asyncio marker.

Without one pytest does not await the coroutine -- it errors with "async def
functions are not natively supported", and if that error is ever suppressed the test
silently never runs. This is easy to cause by accident: inserting a new test directly
above an existing one takes over the decorator that belonged to it, leaving the
original bare. That happened here, and the whole file still reported green apart from
one confusing error.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _class_has_asyncio_pytestmark(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
        ):
            value = stmt.value
            marks = value.elts if isinstance(value, ast.Tuple | ast.List) else [value]
            if any(_is_asyncio_marker(m) for m in marks):
                return True
    return False


def _is_asyncio_marker(node) -> bool:
    """A pytest.mark.asyncio marker, not merely the word "asyncio" somewhere.

    A substring test is satisfied by `@mock.patch("asyncio.sleep")` or by a skipif
    whose reason mentions asyncio -- neither of which makes pytest await the
    coroutine, which is the entire property being asserted.
    """
    import ast

    target = node.func if isinstance(node, ast.Call) else node
    return ast.unparse(target).split("(")[0].strip().endswith("mark.asyncio")


def test_every_collected_async_test_has_an_asyncio_marker():
    offenders: list[str] = []
    checked = 0

    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        module_marked = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
            and any(
                _is_asyncio_marker(m)
                for m in (n.value.elts if isinstance(n.value, ast.Tuple | ast.List) else [n.value])
            )
            for n in tree.body
        )

        candidates: list[tuple[ast.AsyncFunctionDef, bool]] = [
            (n, module_marked) for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
        ]
        def _collect_class(node: ast.ClassDef, inherited: bool) -> None:
            inherited = inherited or _class_has_asyncio_pytestmark(node)
            for member in node.body:
                if isinstance(member, ast.AsyncFunctionDef):
                    candidates.append((member, inherited))
                elif isinstance(member, ast.ClassDef):
                    _collect_class(member, inherited)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                _collect_class(node, module_marked)

        for fn, inherited in candidates:
            if not fn.name.startswith("test_"):
                continue
            checked += 1
            if inherited:
                continue
            if not any(_is_asyncio_marker(d) for d in fn.decorator_list):
                offenders.append(f"{path.name}:{fn.lineno} {fn.name}")

    assert checked >= 2000, (
        f"only {checked} async tests found, expected ~2114; the scan is stale. A floor "
        "of 50 against an actual of 2114 absorbed dropping the entire class-method "
        "branch (1061 tests) without going red."
    )
    assert not offenders, (
        "these async tests carry no asyncio marker, so pytest will not await them:\n  "
        + "\n  ".join(offenders)
    )
