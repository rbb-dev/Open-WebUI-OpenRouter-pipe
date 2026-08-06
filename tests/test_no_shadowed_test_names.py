"""Every test written in this suite must actually be collected.

`tests/test_pipe.py` defined `class TestToolTypeBreaker` twice. The module body runs top
to bottom, so the second binding replaced the first before pytest collected anything, and
the two tests in the first class did not exist at collection time. Both were mutated to
raise unconditionally and the whole suite stayed green at 5885.

pytest emits no warning for this: by collection time the shadowed class is gone, so there
is nothing left to warn about. Nothing short of reading the source can see it.
"""

from __future__ import annotations

import ast
import pathlib


def _module_level_bindings(tree: ast.Module) -> list[tuple[str, int]]:
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            out.append((node.name, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            out.append((node.name, node.lineno))
    return out


def _method_bindings(cls: ast.ClassDef) -> list[tuple[str, int]]:
    return [
        (n.name, n.lineno)
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]


def test_no_test_name_is_bound_twice_in_one_module():
    """A rebound name silently deletes the earlier definition and everything in it."""
    offenders = []
    for path in sorted(pathlib.Path(__file__).resolve().parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        seen: dict[str, int] = {}
        for name, lineno in _module_level_bindings(tree):
            if name in seen:
                offenders.append(f"{path.name}: {name} bound at lines {seen[name]} and {lineno}")
            seen[name] = lineno

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            seen_methods: dict[str, int] = {}
            for name, lineno in _method_bindings(node):
                if name in seen_methods:
                    offenders.append(
                        f"{path.name}: {node.name}.{name} bound at lines "
                        f"{seen_methods[name]} and {lineno}"
                    )
                seen_methods[name] = lineno

    assert not offenders, (
        "these names are bound twice, so the first definition and every test inside it "
        "is discarded before collection and can never fail:\n  " + "\n  ".join(offenders)
    )
