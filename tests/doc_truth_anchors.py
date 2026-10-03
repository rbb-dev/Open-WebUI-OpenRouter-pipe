"""One anchor idiom for the doc-truth tests, so a claim is found the same way everywhere.

Every test in this group asks the same question of a shipped text: does the sentence
an operator reads there say what the code does? They differ only in which file and
which boundary they point at. Repeating the "split on the anchor, assert the anchor
exists exactly once, return the span" idiom in each file is how one of them ends up
silently passing on an empty span -- `str.split` on a missing anchor still returns a
list, just the wrong one -- so the idiom lives here once and is imported by each.

The boundary is named, never a line number. A doc-truth test that pins line 80 fails
the next time a paragraph above it grows a line, and the fix is then to bump the
number, which trains the reader to ignore red without reading it. A heading or a
sentence is a thing the document either still has or genuinely no longer has, and
either answer is worth failing on.

The same holds for the help texts a valve ships. Those are Python strings, and a regex
over the source reads the *literal*, not the value: `"..." + _CONST` -- an ordinary
refactor once one sentence is shared between two valves -- reads short, so the guard
quietly checks less than it claims. The dashboard reader fails in the other direction,
running on into the next entry, where a claim can be satisfied by a neighbour's wording.
So the two readers below parse the module and fold the string expression, and what comes
back is the words the Config tab renders. They read the files, never import them, so
every caller still collects under the release gate's no-plugins artifact.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_SOURCE = _ROOT / "open_webui_openrouter_pipe" / "core" / "config.py"
_CONFIG_META_SOURCE = (
    _ROOT / "open_webui_openrouter_pipe" / "plugins" / "pipe_dashboard" / "config_meta.py"
)


def doc(name: str) -> str:
    """A docs/ file's text, or an AssertionError naming the file that went missing."""
    path = _ROOT / "docs" / name
    assert path.is_file(), f"{name} is gone from docs/; the test pointing at it is stale"
    return path.read_text(encoding="utf-8")


def section(text: str, start_heading: str, end_heading: str, *, what: str) -> str:
    """The span from `start_heading`'s own line up to `end_heading`'s own line.

    Both boundaries are full ATX headings and must occur exactly once, so a document
    that duplicates a section cannot make the extraction quietly stop at the first
    copy and hand the assertions a span that no longer says what it used to. A
    subheading (`###`) cannot close a `###` section, so the two are normally written
    as siblings and the level is not part of the contract.
    """
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if line == start_heading]
    assert len(starts) == 1, (
        f"{what}: the heading {start_heading!r} occurs {len(starts)} times as a whole "
        f"line, so the section's start is not the single place this test thinks it is. "
        f"Either the section was duplicated, or it was renamed and the test is stale."
    )
    tail = lines[starts[0] + 1 :]
    ends = [n for n, line in enumerate(tail) if line == end_heading]
    assert len(ends) == 1, (
        f"{what}: the section opened by {start_heading!r} has {len(ends)} whole lines "
        f"equal to {end_heading!r} after it, so its end is not where this test thinks it "
        f"is. The section was probably duplicated, or it was renamed and the test is stale."
    )
    return "\n".join(lines[starts[0] : starts[0] + 1 + ends[0]])


def between(text: str, anchor: str, *, what: str) -> str:
    """The anchor plus the paragraph it opens, up to the next blank line.

    Doc-truth assertions are about a claim, and a claim is written as a paragraph
    starting with a sentence this test can name. Reading to the next blank line keeps
    the assertion on the claim without pinning how many sentences the paragraph has.
    """
    assert anchor in text, (
        f"{what}: the claim {anchor!r} is not in the document any more, so this test can "
        f"no longer tell which text it was reading"
    )
    head = text.index(anchor)
    return text[head : head + len(text[head:].split("\n\n", 1)[0])].strip()


def _fold(node: ast.expr, constants: dict[str, str]) -> str | None:
    """The string a node evaluates to, or None when it is not a constant expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left, constants), _fold(node.right, constants)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


@cache
def _parsed(path: Path) -> tuple[ast.Module, dict[str, str]]:
    """A module's tree, and its own top-level string constants, folded in source order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        folded = _fold(value, constants)
        if folded is not None:
            constants[target.id] = folded
    return tree, constants


def valve_field_description(name: str) -> str:
    """The `description=` a Config-tab valve carries, as the string the tab shows.

    Located by the `Valves` annotation and read off the `Field(...)` call, so the
    answer does not move when the arguments are reordered, the value is re-wrapped
    across source lines, or a sentence is hoisted into a module constant.
    """
    path = _CONFIG_SOURCE
    assert path.is_file(), f"core/config.py is gone; the test reading {name} is stale"
    tree, constants = _parsed(path)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        if node.target.id != name or not isinstance(node.value, ast.Call):
            continue
        if getattr(node.value.func, "id", "") != "Field":
            continue
        for keyword in node.value.keywords:
            if keyword.arg != "description":
                continue
            folded = _fold(keyword.value, constants)
            assert folded is not None, (
                f"{name}'s Field description is not a constant string expression, so this "
                "reader cannot say what the Config tab shows for it"
            )
            return folded
    raise AssertionError(f"{name} has no Field(...) description in core/config.py")


def config_meta_detail(name: str) -> str:
    """The `detail=` a dashboard `CONFIG_META` row carries, as the string the row shows.

    The key is matched as a dict key rather than as source text, so the row's own
    layout is irrelevant and a capture can never run on into the entry after it.
    """
    path = _CONFIG_META_SOURCE
    assert path.is_file(), f"config_meta.py is gone; the test reading {name} is stale"
    tree, constants = _parsed(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == name):
                continue
            if not isinstance(value, ast.Dict):
                continue
            for inner_key, inner in zip(value.keys, value.values):
                if not (isinstance(inner_key, ast.Constant) and inner_key.value == "detail"):
                    continue
                folded = _fold(inner, constants)
                assert folded is not None, (
                    f"{name}'s CONFIG_META detail is not a constant string expression, so "
                    "this reader cannot say what the dashboard row shows for it"
                )
                return folded
    raise AssertionError(f"{name} has no CONFIG_META detail in config_meta.py")
