#!/usr/bin/env python3
"""Report prose added to production modules since a baseline commit.

A grep for added ``#`` lines sees one of three ways to add prose, so moving a comment
into an adjacent string literal removes nothing and still satisfies it. This walks each
changed module with ``tokenize`` for ``#`` tokens and with ``ast`` for
``Expr(Constant(str))`` statements, intersects both with the lines the diff actually
added, and reports the three categories separately.

The diff runs from the baseline commit to the WORKING TREE. ``base..HEAD`` cannot see a
staged or unstaged edit, which is the state this gate runs in.

A re-indent is not prose, so it is not reported: a four-space re-indent of a whole region
rewrites every line's text, and without an exemption the most conservative change this
codebase can make is scored as thirty lines of new prose. Exemption is decided by aligning
the added lines against the removed ones and keeping only a pair that differs by leading
whitespace and nothing else.

Directive comments survive, sourced from the bundler's own rule rather than a second
copy of it: the bundler is what decides which comments are load-bearing.

Usage:
    python scripts/check_added_prose.py 045e46e
    python scripts/check_added_prose.py 045e46e --allow docstring
    python scripts/check_added_prose.py 045e46e --path open_webui_openrouter_pipe --json
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import re
import subprocess
import sys
import tokenize
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLER = PROJECT_ROOT / "scripts" / "bundle_v2.py"

CATEGORIES = ("comment", "bare-string", "docstring")

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

_REINDENT_MIN_BLOCK = 2   # a re-indent rewrites a contiguous REGION, so its lines arrive as a
                          # multi-line aligned block; a lone match is a move, or a coincidence


def directive_comment_re(bundler: Path | None = None) -> re.Pattern[str]:
    """The bundler's own directive rule, loaded rather than restated."""
    target = bundler or BUNDLER
    spec = importlib.util.spec_from_file_location("_bundle_v2_for_prose_check", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the bundler at {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    pattern = getattr(module, "_DIRECTIVE_COMMENT_RE", None)
    if not isinstance(pattern, re.Pattern):
        raise TypeError(
            "scripts/bundle_v2.py no longer exposes _DIRECTIVE_COMMENT_RE as a compiled "
            "pattern, so this check would report every noqa and type: ignore as new prose"
        )
    return pattern


def _untracked_files(tree_root: Path, path_spec: str) -> list[str]:
    """Paths git is not tracking yet, which no diff against a commit can show.

    ``git diff <commit>`` compares the commit to the working tree for files git already
    knows about, so a brand-new module that has not been staged is absent from it
    entirely -- and a changeset that introduces one would score zero prose for it.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", path_spec],
        cwd=tree_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in listing.splitlines() if line.strip()]


def _diff_positions(
    base: str, path_spec: str, *, root: Path | None = None
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Added line numbers per file, and the subset that only re-indents a removed line.

    A re-indent rewrites every line's text, so ``git diff`` reports the whole region as
    added. Without the second half, the most conservative change this codebase can
    make -- one that moves code and says nothing -- is scored as thirty lines of new
    prose.

    The alignment is over the whole file, not per hunk. A four-space re-indent leaves
    only blank lines and a few coincidences byte-identical, so ``git diff`` shatters
    one contiguous region into dozens of hunks whose added and removed sides no longer
    line up; pairing within a hunk then misses most of the region. Across the file the
    two sides are the same lines in the same order, which is what the matcher needs.
    """
    tree_root = root or PROJECT_ROOT
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", base, "--", path_spec],
        cwd=tree_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    per_file: dict[str, set[int]] = {}
    added_by_file: dict[str, list[tuple[int, str]]] = {}
    removed_by_file: dict[str, list[str]] = {}
    current: str | None = None
    cursor = 0

    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target == "/dev/null":
                current = None
            else:
                current = target.removeprefix("b/")
                per_file.setdefault(current, set())
                added_by_file.setdefault(current, [])
                removed_by_file.setdefault(current, [])
            continue
        if raw.startswith("@@"):
            match = _HUNK_RE.match(raw)
            if match is not None:
                cursor = int(match.group(1))
            continue
        if current is None:
            continue
        if raw.startswith("+"):
            per_file[current].add(cursor)
            added_by_file[current].append((cursor, raw[1:]))
            cursor += 1
        elif raw.startswith("-"):
            removed_by_file[current].append(raw[1:])

    reindented: dict[str, set[int]] = {}
    for rel, added_lines_here in added_by_file.items():
        removed_here = removed_by_file[rel]
        if not added_lines_here or not removed_here:
            continue
        _exempt_reindents(rel, added_lines_here, removed_here, reindented)

    for rel in _untracked_files(tree_root, path_spec):
        target = tree_root / rel
        if not target.is_file():
            continue
        body = target.read_text(encoding="utf-8", errors="replace")
        per_file.setdefault(rel, set()).update(range(1, len(body.splitlines()) + 1))
    return per_file, reindented


def _exempt_reindents(
    rel: str,
    added_lines_here: list[tuple[int, str]],
    removed_here: list[str],
    reindented: dict[str, set[int]],
) -> None:
    """Exempt added lines that only re-indent a line already there.

    A line is exempt when it pairs, in order, with a removed line carrying the same
    stripped text and differs from it by leading whitespace and nothing else.

    Order alone is not enough and text alone is not enough. A line that MOVED within a
    hunk keeps its exact text, so text matching would exempt prose that changed
    position to explain something else. Two clauses separate the two. A line that
    carries text is exempt only when it is by definition NOT byte-identical to the
    removed line it pairs with, so a moved line -- byte-identical to the one it
    replaced -- is never exempt; and the pair must arrive as part of a multi-line
    block, because a re-indent rewrites a contiguous region while a move is a lone
    line. The first clause alone is not sufficient: a line that moves into a deeper
    scope both changes position and re-indents, so it is no longer byte-identical and
    it is that second clause, the block, that keeps it counted.

    A blank line is exempt whenever it pairs with a blank one, with or without the
    indentation change, because a blank line is not prose and there is nothing in it to
    have moved. This is not a hole: a blank line is only ever counted because it falls
    inside some other node's line range, and that node's text lines are matched on the
    strict rule above and still reported. Exempting the interior blank of a docstring
    that the gate has already flagged on its first line changes nothing about the
    verdict. The blank line inside a multi-line docstring is the case that needs it: a
    re-indent shifts such a line right, but a formatter that left it empty means it
    arrives byte-identical.
    """
    matcher = SequenceMatcher(
        a=[text.strip() for _, text in added_lines_here],
        b=[text.strip() for text in removed_here],
        autojunk=False,
    )
    for a_start, b_start, size in matcher.get_matching_blocks():
        if size < _REINDENT_MIN_BLOCK:
            continue
        for offset in range(size):
            line_no, text = added_lines_here[a_start + offset]
            paired = removed_here[b_start + offset]
            if not text.strip():
                if not paired.strip():
                    reindented.setdefault(rel, set()).add(line_no)
            elif text != text.lstrip() and paired != text:
                reindented.setdefault(rel, set()).add(line_no)


def added_lines(base: str, path_spec: str, *, root: Path | None = None) -> dict[str, set[int]]:
    """Line numbers added in the working tree, per repo-relative path."""
    return _diff_positions(base, path_spec, root=root)[0]


def _docstring_statements(tree: ast.AST) -> set[int]:
    """Every string-expression statement in a real docstring position, by node id."""
    heads: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            heads.add(id(body[0]))
    return heads


def prose_lines(source: str, directive: re.Pattern[str]) -> dict[str, set[int]]:
    """Lines carrying prose in *source*, split by category."""
    found: dict[str, set[int]] = {name: set() for name in CATEGORIES}

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not directive.search(token.string):
            found["comment"].add(token.start[0])

    tree = ast.parse(source)
    heads = _docstring_statements(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        category = "docstring" if id(node) in heads else "bare-string"
        end = node.end_lineno if node.end_lineno is not None else node.lineno
        found[category].update(range(node.lineno, end + 1))
    return found


def sweep(
    base: str,
    path_spec: str,
    *,
    root: Path | None = None,
    bundler: Path | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Added prose per changed module, per category."""
    tree_root = root or PROJECT_ROOT
    directive = directive_comment_re(bundler)
    per_file, reindented = _diff_positions(base, path_spec, root=tree_root)
    report: dict[str, dict[str, list[int]]] = {}
    for rel, added in sorted(per_file.items()):
        if not rel.endswith(".py"):
            continue
        target = tree_root / rel
        if not target.is_file():
            continue
        counted = added - reindented.get(rel, set())
        hits = {
            name: sorted(lines & counted)
            for name, lines in prose_lines(
                target.read_text(encoding="utf-8"), directive
            ).items()
        }
        if any(hits.values()):
            report[rel] = hits
    return report


def render(report: dict[str, dict[str, list[int]]]) -> str:
    """The human-readable report, one line per file and category."""
    out: list[str] = []
    for rel, hits in report.items():
        for name in CATEGORIES:
            if hits[name]:
                out.append(f"{rel}: {len(hits[name])} {name} line(s) added: {hits[name]}")
    totals = {name: sum(len(hits[name]) for hits in report.values()) for name in CATEGORIES}
    out.append(
        "totals: "
        + ", ".join(f"{name}={totals[name]}" for name in CATEGORIES)
        + f" across {len(report)} module(s)"
    )
    return "\n".join(out)


def failing_categories(
    report: dict[str, dict[str, list[int]]], allowed: set[str]
) -> set[str]:
    """Categories with findings that the caller did not allow."""
    return {
        name
        for name in CATEGORIES
        if name not in allowed and any(hits[name] for hits in report.values())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report prose added since a baseline commit.")
    parser.add_argument("base", help="baseline commit; the diff runs to the working tree")
    parser.add_argument(
        "--path", default="open_webui_openrouter_pipe", help="pathspec to limit the diff to"
    )
    parser.add_argument(
        "--allow",
        default="",
        help=f"comma-separated categories excluded from the exit code: {', '.join(CATEGORIES)}",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    allowed = {piece.strip() for piece in args.allow.split(",") if piece.strip()}
    unknown = sorted(allowed - set(CATEGORIES))
    if unknown:
        parser.error(f"unknown category in --allow: {unknown}; known: {list(CATEGORIES)}")

    report = sweep(args.base, args.path)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 1 if failing_categories(report, allowed) else 0


if __name__ == "__main__":
    sys.exit(main())
