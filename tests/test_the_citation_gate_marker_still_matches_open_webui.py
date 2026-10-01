"""The citation gate's marker still is Open WebUI's own marker.

`streaming_core.py` routes a tool result to Open WebUI's citation extractor only when the registry
entry behind the exposed name carries both halves of the marker Open WebUI's `get_builtin_tools`
stamps: a `type` of `builtin` and a `tool_id` under the `builtin:` prefix. Both halves are read off a
private field of Open WebUI, so an upstream rename silently rewrites the gate: every citation stops
being a citation and the suite stays green, because the fixtures in `tests/test_streaming_handler.py`
stamped the same literal the gate reads.

This node reads both sides and compares them, which is what makes them unable to drift. Upstream is
read from the installed distribution's source by AST, so the check never imports Open WebUI and
survives a bundled or zipped install. The pipe is read as source, scanning the whole file rather than
the gate helper, so a legal refactor that inlines or renames the gate keeps the node green.

Two of the pipe's readers of the marker's `type` half live outside `streaming_core.py`: the shared
predicate the registry claims a builtin through, and the ask-user check it is composed beside. The
`type`-only reader below finds them by scanning whole modules, and the cross-file clause is what stops
a refactor from leaving one site on a literal of its own that the others no longer agree with.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPE_SOURCE = _REPO_ROOT / "open_webui_openrouter_pipe" / "streaming" / "streaming_core.py"
_UPSTREAM_RELATIVE = "open_webui/utils/tools.py"

# The `type` literal the gate compares by equality. The window is bounded so a pair cannot be
# stitched together out of two compares far apart in the file, and closed at the semicolon and at any
# new `and` so the `startswith` paired with it is the marker half of the same boolean and not a
# comparison the same expression happens to sit near.
_PIPE_GATE_MARKER = re.compile(
    r'\.get\(\s*["\']type["\']\s*\)\s*==\s*["\'](?P<type>[^"\']+)["\']'
    r"(?P<between>(?:(?![;\n]).){0,200}?)"
    r'\.startswith\(\s*["\'](?P<prefix>[^"\']+)["\']\s*\)',
    re.DOTALL,
)

def _pipe_gate_marker() -> tuple[str, str]:
    """The two literals the pipe's citation gate compares, as `(type, tool_id prefix)`.

    Scans the whole module, so inlining the gate into its caller, renaming the helper, or reading the
    marker as `str(cfg.get("tool_id", ""))` all leave the pair findable. Rewriting either literal does
    not: a gate that compares a renamed marker has no pair to find.
    """
    source = _PIPE_SOURCE.read_text(encoding="utf-8")
    pairs = {
        (match.group("type"), match.group("prefix"))
        for match in _PIPE_GATE_MARKER.finditer(source)
    }
    if not pairs:
        raise AssertionError(
            f"no `type` comparison followed by a `startswith` on the marker expression was found "
            f"anywhere in {_PIPE_SOURCE}, so the citation gate no longer reads a type literal and a "
            f"tool_id prefix together and cannot be pinned against Open WebUI's {_UPSTREAM_RELATIVE}"
        )
    if len(pairs) > 1:
        raise AssertionError(
            f"{_PIPE_SOURCE.name} compares more than one candidate marker, so which is the citation "
            f"gate's is ambiguous: {sorted(pairs)}"
        )
    return pairs.pop()


def _string_of(expression: ast.expr) -> str:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    return ast.unparse(expression)


def _upstream_tool_id_stamps(source: str | None = None) -> list[tuple[str, str]]:
    """Every `(type, tool_id)` Open WebUI's own `utils/tools.py` stamps, as source expressions.

    That module builds several shapes carrying both keys; only the builtin one is this marker's
    producer, and the caller tells them apart on the type half. `source` overrides the installed
    file's text, which is how the rename table drives this reader over a renamed upstream.
    """
    if source is None:
        path = Path(str(importlib.metadata.distribution("open-webui").locate_file(_UPSTREAM_RELATIVE)))
        source = path.read_text(encoding="utf-8")
    stamps: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries: dict[str, ast.expr] = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "tool_id" in entries and "type" in entries:
            stamps.append(
                (_string_of(entries["type"]), ast.unparse(entries["tool_id"]))
            )
    return stamps


def _tool_id_head(tool_id_expression: ast.expr) -> str | None:
    """The literal head of a stamped tool_id: the `Constant` parts up to the first interpolation.

    `f'builtin:{func.__name__}'` has the head `'builtin:'`. Dumping the whole expression instead
    yields `"builtin:{func.__name__}"`, which no prefix test can compare against; and reading it
    as a substring is worse, because `builtin:` occurs inside `native_builtin:`, `toolbuiltin:`
    and `xbuiltin:` -- three renames the gate rejects and a substring check accepts.
    """
    if not isinstance(tool_id_expression, ast.JoinedStr):
        return None
    parts: list[str] = []
    for part in tool_id_expression.values:
        if isinstance(part, ast.FormattedValue):
            break
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        else:
            return None
    return "".join(parts)


def _stamped_heads_of(source: str) -> dict[str, list[str]]:
    """The tool_id literal heads upstream stamps, grouped by the `type` they carry."""
    heads: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        entries: dict[str, ast.expr] = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "tool_id" not in entries or "type" not in entries:
            continue
        head = _tool_id_head(entries["tool_id"])
        if head is not None:
            heads.setdefault(_string_of(entries["type"]), []).append(head)
    return heads


def _gate_prefix_still_stamped(source: str, pipe_type: str, pipe_prefix: str) -> bool:
    """Whether some dict stamped with `pipe_type` writes a tool_id the gate's prefix opens.

    A prefix test on the literal head, not a substring test on the dumped expression: the gate calls
    `startswith`, so only a head that starts with the prefix opens it, and `builtin:` occurs inside
    `native_builtin:`, `toolbuiltin:` and `xbuiltin:` -- three renames that leave every citation
    silently uncited while a substring test calls the marker stamped.
    """
    return any(
        head.startswith(pipe_prefix) for head in _stamped_heads_of(source).get(pipe_type, [])
    )


def _pipe_builtin_marker() -> tuple[str, str]:
    """The pipe's citation marker, asserted to be one Open WebUI still stamps.

    Split out so `tests/test_streaming_handler.py` can build its fixtures' marker from the same read
    the pin checks, so the fixtures cannot be built from a literal the gate no longer reads.
    """
    pipe_type, pipe_prefix = _pipe_gate_marker()
    path = Path(str(importlib.metadata.distribution("open-webui").locate_file(_UPSTREAM_RELATIVE)))
    source = path.read_text(encoding="utf-8")
    stamps = _upstream_tool_id_stamps(source)
    assert stamps, (
        f"no dict in Open WebUI's {_UPSTREAM_RELATIVE} carries both a `tool_id` and a `type`, so the "
        f"marker the pipe's gate reads has no producer upstream"
    )
    of_the_pipes_type = [
        tool_id_expression
        for upstream_type, tool_id_expression in stamps
        if upstream_type == pipe_type
    ]
    heads = _stamped_heads_of(source).get(pipe_type, [])
    assert _gate_prefix_still_stamped(source, pipe_type, pipe_prefix), (
        f"the citation gate in {_PIPE_SOURCE.name} reads a `type` of {pipe_type!r} and a tool_id "
        f"starting with {pipe_prefix!r}, but Open WebUI's {_UPSTREAM_RELATIVE} stamps no such "
        f"tool_id: what it writes with a type of {pipe_type!r} is {of_the_pipes_type or 'nothing'}, "
        f"whose literal head(s) are {heads or 'none'}. An upstream rename leaves the gate rejecting "
        f"every builtin, so every citation is silently dropped while the suite stays green; the "
        f"pipe's literal and Open WebUI's must change together."
    )
    return pipe_type, pipe_prefix


def _owui_builtin_tool_id_prefix() -> str:
    """The tool_id prefix a citation fixture stamps, read from the pipe's gate and checked against
    Open WebUI. `tests/test_streaming_handler.py` builds every builtin registry fixture from this, so
    a fixture and the gate cannot disagree about what a builtin looks like.

    Called per use rather than bound to a module constant: the check is an assertion, and evaluating
    it at import would turn a drift into a collection error that takes the whole file down with it
    instead of failing the one node that names it.
    """
    return _pipe_builtin_marker()[1]
