"""One reader for `warn_level(<latch>, <cause>)` cause keys, shared by every census.

Three censuses had grown three copies of this. Two required the cause to be an
`ast.Constant` and silently dropped everything else -- and the house idiom is an
f-string, `f"<site>:{type(exc).__name__}"`, which pipe.py uses at ten of its eleven
sites. A shape they could not match was ABSENT from the discovered set rather than
reported, so a new seam spelled the house way never appeared in a driver table and its
latch could be inverted with the whole suite green. Two of them also required the LATCH
to be a bare Name, and four call sites in the package already hold theirs on an object.

The failure direction is inverted here: a shape this cannot resolve is RETURNED in
`unresolvable` for the caller to assert empty. An unknown spelling is red, not silence.
"""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

UNRESOLVABLE_MESSAGE = (
    "these warn_level causes could not be reduced to a key, so this census cannot see "
    "the site at all and its driver table cannot report the omission. Give the cause a "
    'literal head (f"<site>:{...}") or teach latch_causes the new shape:\n  '
)


def _bound_name(node: ast.expr) -> str | None:
    """The trailing identifier of `x`, `a.x` or `self.x`; None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _cause_key(node: ast.expr) -> str | None:
    """The stable PREFIX of a cause key, or None when the shape is not static.

    A string constant is its own key. An f-string keys on the literal text before its
    first placeholder with a trailing ':' removed, so `f"web_tools:{type(exc).__name__}"`
    keys on `web_tools` -- which is what a driver table can name and what the callers
    already group by when they read the latch back.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        if (
            isinstance(head, ast.Constant)
            and isinstance(head.value, str)
            and head.value.rstrip(":")
        ):
            return head.value.rstrip(":")
    return None


def causes_in_source(source: str, latch: str) -> tuple[set[str], list[str]]:
    """(cause keys, unresolvable descriptions) for one latch in one source text."""
    tree = ast.parse(source)
    causes: set[str] = set()
    unresolvable: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        if _bound_name(node.func) != "warn_level":
            continue
        if _bound_name(node.args[0]) != latch:
            continue
        key = _cause_key(node.args[1])
        if key is None:
            unresolvable.append(f"line {node.lineno}: {ast.unparse(node.args[1])}")
        else:
            causes.add(key)
    return causes, unresolvable


def warn_level_causes(module: ModuleType, latch: str) -> tuple[set[str], list[str]]:
    """Every cause key on *latch* in *module*, plus the shapes that did not resolve."""
    return causes_in_source(inspect.getsource(module), latch)
