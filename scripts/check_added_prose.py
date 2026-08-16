#!/usr/bin/env python3
"""Report prose added to production modules since a baseline commit.

A grep for added ``#`` lines sees one of three ways to add prose, so moving a comment
into an adjacent string literal removes nothing and still satisfies it. This walks each
changed module with ``tokenize`` for ``#`` tokens and with ``ast`` for
``Expr(Constant(str))`` statements, intersects both with the lines the diff actually
added, and reports the three categories separately.

The diff runs from the baseline commit to the WORKING TREE. ``base..HEAD`` cannot see a
staged or unstaged edit, which is the state this gate runs in.

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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLER = PROJECT_ROOT / "scripts" / "bundle_v2.py"

CATEGORIES = ("comment", "bare-string", "docstring")

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


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


def added_lines(base: str, path_spec: str, *, root: Path | None = None) -> dict[str, set[int]]:
    """Line numbers added in the working tree, per repo-relative path."""
    tree_root = root or PROJECT_ROOT
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", base, "--", path_spec],
        cwd=tree_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    per_file: dict[str, set[int]] = {}
    current: set[int] | None = None
    cursor = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            current = (
                None
                if target == "/dev/null"
                else per_file.setdefault(target.removeprefix("b/"), set())
            )
            continue
        if raw.startswith("@@"):
            match = _HUNK_RE.match(raw)
            if match is not None:
                cursor = int(match.group(1))
            continue
        if current is not None and raw.startswith("+"):
            current.add(cursor)
            cursor += 1
    for rel in _untracked_files(tree_root, path_spec):
        target = tree_root / rel
        if not target.is_file():
            continue
        body = target.read_text(encoding="utf-8", errors="replace")
        per_file.setdefault(rel, set()).update(range(1, len(body.splitlines()) + 1))
    return per_file


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
    report: dict[str, dict[str, list[int]]] = {}
    for rel, added in sorted(added_lines(base, path_spec, root=tree_root).items()):
        if not rel.endswith(".py"):
            continue
        target = tree_root / rel
        if not target.is_file():
            continue
        hits = {
            name: sorted(lines & added)
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
