#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import re
import sys
import tokenize
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import NoReturn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ("comment", "bare-string", "docstring")

KEPT_DIRECTIVES = {
    "ruff": r"noqa(?::[ \t]*[A-Z]+[0-9]+(?:[ \t,]+[A-Z]+[0-9]+)*)?",
    "pyright and mypy": r"type:[ \t]*ignore(?:\[[A-Za-z0-9_, \t-]*\])?",
    "pyright": (
        r"pyright:[ \t]*(?:ignore(?:\[[A-Za-z0-9_, \t]*\])?|basic|standard|strict"
        r"|report[A-Za-z]+[ \t]*=[ \t]*[A-Za-z0-9_]+)"
    ),
    "coverage.py": r"pragma:[ \t]*no[ \t]+(?:cover|branch)",
    "formatters": r"fmt:[ \t]*(?:off|on|skip)",
}
KEPT_LEADING_COMMENTS = {
    "shebang": (1, r"#!"),
    "encoding cookie": (2, r"#.*coding[:=][ \t]*[-\w.]+"),
}
KEPT_MODULE_DOCSTRINGS = {
    "STUB_FILE": "the key: value lines Open WebUI and the bundler parse, and the docstring delimiters",
}
KEPT_FRONTMATTER_LINE = r"^\s*([a-z_]+):\s*(.*)\s*$"
BUNDLER_INPUTS = ("PACKAGE_DIR", "STUB_FILE", "ANYIO_WORKAROUND_FILE")

FRONTMATTER_LINE = re.compile(KEPT_FRONTMATTER_LINE, re.IGNORECASE)
DIRECTIVE_FORMS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in KEPT_DIRECTIVES.values())
LEADING_FORMS = tuple((row, re.compile(pattern)) for row, pattern in KEPT_LEADING_COMMENTS.values())
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
NON_CODE_TOKENS = (
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
)


class ProseError(Exception):
    pass


@dataclass(frozen=True)
class ScopedFile:
    path: Path
    rel: str
    keeps_docstring: bool


@dataclass(frozen=True)
class Rules:
    keeps_docstring: bool
    first_party: frozenset[str]


@dataclass(frozen=True)
class Follower:
    row: int
    kind: str
    module_level: bool


@dataclass
class Expectation:
    placeholders: set[int] = field(default_factory=set)
    rewrites: dict[int, str] = field(default_factory=dict)


@dataclass
class Plan:
    delete: set[int] = field(default_factory=set)
    replace: dict[int, str] = field(default_factory=dict)
    found: dict[str, set[int]] = field(
        default_factory=lambda: {name: set() for name in CATEGORIES}
    )
    header_rows: set[int] = field(default_factory=set)
    followers: dict[int, Follower] = field(default_factory=dict)
    unremovable: list[str] = field(default_factory=list)

    @property
    def dirty(self) -> bool:
        return any(self.found.values())


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_bundler(root: Path) -> ModuleType:
    target = root / "scripts" / "bundle_v2.py"
    spec = importlib.util.spec_from_file_location("_bundle_v2_for_strip_prose", target)
    if spec is None or spec.loader is None:
        fail(f"cannot load the bundler at {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def scope_files(root: Path) -> list[ScopedFile]:
    bundler = load_bundler(root)
    missing = [name for name in BUNDLER_INPUTS if not hasattr(bundler, name)]
    if missing:
        fail(
            f"scripts/bundle_v2.py no longer defines {', '.join(missing)}, so the files "
            "it copies into a bundle cannot be listed"
        )
    package = Path(bundler.PACKAGE_DIR)
    manifest = Path(bundler.STUB_FILE)
    workaround = Path(bundler.ANYIO_WORKAROUND_FILE)
    if not package.is_dir() or not manifest.is_file():
        fail(f"the bundler's package {package} or manifest {manifest} is missing")
    paths = {path for path in package.rglob("*.py") if "__pycache__" not in path.parts}
    paths.add(manifest)
    if workaround.is_file():
        paths.add(workaround)
    keeps = {Path(getattr(bundler, name)).resolve() for name in KEPT_MODULE_DOCSTRINGS}
    return [
        ScopedFile(path, path.relative_to(root).as_posix(), path.resolve() in keeps)
        for path in sorted(paths)
    ]


def skip_separators(text: str, position: int) -> int:
    return len(text) - len(text[position:].lstrip(" \t#"))


def directive_chain(text: str, start: int) -> tuple[list[str], int]:
    forms: list[str] = []
    end = start
    position = skip_separators(text, start)
    while True:
        hit = next(
            (match for form in DIRECTIVE_FORMS if (match := form.match(text, position))),
            None,
        )
        if hit is None:
            return forms, end
        forms.append(hit.group(0))
        end = hit.end()
        position = skip_separators(text, end)


def normalized_comment(text: str) -> str | None:
    forms, end = directive_chain(text, 1)
    if forms:
        return text if not text[end:].strip() else text[:end]
    for index in range(1, len(text)):
        if text[index] == "#":
            forms, _end = directive_chain(text, index + 1)
            if forms:
                return "# " + "  # ".join(forms)
    return None


def is_kept_leading_comment(row: int, text: str) -> bool:
    return any(row <= last_row and form.match(text) for last_row, form in LEADING_FORMS)


def is_string_statement(node: object) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def is_ellipsis_statement(node: object) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    )


def statement_lists(tree: ast.AST) -> list[tuple[ast.AST, str, list[ast.stmt]]]:
    found: list[tuple[ast.AST, str, list[ast.stmt]]] = []
    for node in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            statements = getattr(node, name, None)
            if isinstance(statements, list) and statements and isinstance(statements[0], ast.stmt):
                found.append((node, name, statements))
    return found


def first_row(statement: ast.stmt) -> int:
    decorators = getattr(statement, "decorator_list", [])
    return min([statement.lineno, *(decorator.lineno for decorator in decorators)])


def import_section(node: ast.stmt, first_party: frozenset[str]) -> str:
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return "local"
        module = node.module or ""
    elif isinstance(node, ast.Import):
        module = node.names[0].name
    else:
        return "none"
    top = module.split(".")[0]
    if top == "__future__":
        return "future"
    if top in sys.stdlib_module_names:
        return "stdlib"
    return "first-party" if top in first_party else "third-party"


def follower_kind(current: ast.stmt, following: ast.stmt, first_party: frozenset[str]) -> str:
    if isinstance(following, (ast.Import, ast.ImportFrom)):
        same = import_section(current, first_party) == import_section(following, first_party)
        return "same-section-import" if same else "new-section-import"
    if isinstance(following, (*FUNCTIONS, ast.ClassDef)):
        return "definition"
    return "other"


def header_end_rows(tokens: list[tokenize.TokenInfo]) -> set[int]:
    rows: set[int] = set()
    last: tokenize.TokenInfo | None = None
    for token in tokens:
        if token.type in NON_CODE_TOKENS:
            continue
        if token.type == tokenize.NEWLINE:
            if last is not None and last.type == tokenize.OP and last.string == ":":
                rows.add(token.start[0])
            last = None
            continue
        last = token
    return rows


def placeholder_for(owner: ast.AST, parents: dict[int, list[ast.stmt]]) -> str:
    if isinstance(owner, (*FUNCTIONS, ast.ClassDef)):
        for sibling in parents.get(id(owner), []):
            if (
                sibling is not owner
                and isinstance(sibling, FUNCTIONS)
                and len(sibling.body) == 1
                and is_ellipsis_statement(sibling.body[0])
            ):
                return "..."
    return "pass"


def shares_its_lines(lines: list[str], statement: ast.stmt) -> bool:
    last_row = statement.end_lineno or statement.lineno
    first = lines[statement.lineno - 1].encode("utf-8")
    last = lines[last_row - 1].encode("utf-8")
    before = first[: statement.col_offset].decode("utf-8")
    after = last[statement.end_col_offset or 0 :].decode("utf-8")
    return bool(before.strip() or after.strip())


def reduce_frontmatter(value: str) -> str:
    pieces = value.split("\n")
    if len(pieces) < 3:
        return value
    keys = [piece for piece in pieces[1:-1] if FRONTMATTER_LINE.match(piece)]
    return "\n".join([pieces[0], *keys, pieces[-1]])


def trim_frontmatter(lines: list[str], statement: ast.stmt, plan: Plan) -> None:
    first = statement.lineno
    last = statement.end_lineno or first
    prose = [
        row for row in range(first + 1, last) if not FRONTMATTER_LINE.match(lines[row - 1])
    ]
    if not prose:
        return
    if lines[first - 1].strip() != '"""' or lines[last - 1].strip() != '"""':
        plan.unremovable.append(
            f"line {first}: the frontmatter docstring does not open and close on its own lines"
        )
    plan.found["docstring"].update(prose)
    plan.delete.update(prose)


def physical_lines(text: str) -> tuple[list[str], bool]:
    if not text:
        return [], False
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
        return parts, True
    return parts, False


def plan_source(text: str, rules: Rules) -> Plan:
    lines, _ = physical_lines(text)
    tree = ast.parse(text)
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    plan = Plan(header_rows=header_end_rows(tokens))
    lists = statement_lists(tree)
    parents = {id(item): statements for _owner, _name, statements in lists for item in statements}

    for owner, _name, statements in lists:
        for current, following in pairwise(statements):
            if isinstance(current, (ast.Import, ast.ImportFrom)):
                plan.followers[current.end_lineno or current.lineno] = Follower(
                    first_row(following),
                    follower_kind(current, following, rules.first_party),
                    isinstance(owner, ast.Module),
                )

    for owner, name, statements in lists:
        slot = name == "body" and isinstance(owner, DOCSTRING_OWNERS)
        doomed: list[tuple[ast.stmt, str]] = []
        for index, statement in enumerate(statements):
            if not is_string_statement(statement):
                continue
            is_docstring = slot and index == 0
            if is_docstring and isinstance(owner, ast.Module) and rules.keeps_docstring:
                trim_frontmatter(lines, statement, plan)
                continue
            doomed.append((statement, "docstring" if is_docstring else "bare-string"))
        for statement, category in doomed:
            if shares_its_lines(lines, statement):
                plan.unremovable.append(
                    f"line {statement.lineno}: a string statement shares its line with other code"
                )
            rows = range(statement.lineno, (statement.end_lineno or statement.lineno) + 1)
            plan.found[category].update(rows)
            plan.delete.update(rows)
        if doomed and len(doomed) == len(statements) and not isinstance(owner, ast.Module):
            first = doomed[0][0]
            raw = lines[first.lineno - 1]
            indent = raw[: len(raw) - len(raw.lstrip())]
            carriage = "\r" if lines[(first.end_lineno or first.lineno) - 1].endswith("\r") else ""
            plan.delete.discard(first.lineno)
            plan.replace[first.lineno] = indent + placeholder_for(owner, parents) + carriage

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, col = token.start
        original = token.string
        line = lines[row - 1]
        whole_line = not line[:col].strip()
        if whole_line and is_kept_leading_comment(row, original):
            continue
        revised = normalized_comment(original)
        if revised == original:
            continue
        if row in plan.delete:
            if revised is not None:
                plan.unremovable.append(f"line {row}: a directive comment sits on a removed line")
            continue
        plan.found["comment"].add(row)
        rest = line[col + len(original) :]
        if revised is None:
            if whole_line:
                plan.delete.add(row)
            else:
                plan.replace[row] = line[:col].rstrip() + rest
        else:
            plan.replace[row] = line[:col] + revised + rest
    return plan


def blank_rows_to_keep(
    run: range, plan: Plan, total: int
) -> tuple[list[int], int]:
    above, below = run.start - 1, run.stop
    if above == 0 or below > total or above in plan.header_rows:
        return [], 0
    largest = largest_blank_group(run, plan.delete)
    follower = plan.followers.get(above)
    if follower is None or follower.row != below:
        return largest, 0
    if follower.kind == "same-section-import":
        return [], 0
    if follower.kind == "new-section-import":
        return largest[:1], 0
    if follower.kind == "definition" and follower.module_level:
        return largest, max(0, 2 - len(largest))
    return largest, 0


def settle_blank_runs(lines: list[str], plan: Plan) -> tuple[set[int], dict[int, int]]:
    total = len(lines)
    removed = set(plan.delete)
    inserted: dict[int, int] = {}

    def blank(row: int) -> bool:
        return row not in plan.delete and row not in plan.replace and not lines[row - 1].strip()

    row = 1
    while row <= total:
        if row not in plan.delete and not blank(row):
            row += 1
            continue
        end = row
        while end <= total and (end in plan.delete or blank(end)):
            end += 1
        run = range(row, end)
        if any(r in plan.delete for r in run):
            keep, extra = blank_rows_to_keep(run, plan, total)
            removed.update(r for r in run if r not in plan.delete and r not in keep)
            if extra:
                inserted[row - 1] = extra
        row = end
    return removed, inserted


def largest_blank_group(run: range, deleted: set[int]) -> list[int]:
    groups: list[list[int]] = []
    previous_blank = False
    for r in run:
        if r in deleted:
            previous_blank = False
            continue
        if previous_blank:
            groups[-1].append(r)
        else:
            groups.append([r])
        previous_blank = True
    return max(groups, key=len, default=[])


def render(text: str, plan: Plan) -> str:
    lines, final_newline = physical_lines(text)
    removed, inserted = settle_blank_runs(lines, plan)
    kept: list[str] = []
    for row in range(1, len(lines) + 1):
        if row not in removed:
            kept.append(plan.replace.get(row, lines[row - 1]))
        if row in inserted:
            carriage = "\r" if lines[row - 1].endswith("\r") else ""
            kept.extend([carriage] * inserted[row])
    if not kept:
        return ""
    ends_with_newline = final_newline or len(lines) in removed
    return "\n".join(kept) + ("\n" if ends_with_newline else "")


def prune_strings(tree: ast.AST, keeps_docstring: bool) -> Expectation:
    expectation = Expectation()
    for owner, name, statements in statement_lists(tree):
        slot = name == "body" and isinstance(owner, DOCSTRING_OWNERS)
        keeps_head = slot and isinstance(owner, ast.Module) and keeps_docstring
        head = statements[0]
        if keeps_head and is_string_statement(head) and isinstance(head, ast.Expr):
            constant = head.value
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                expectation.rewrites[id(constant)] = reduce_frontmatter(constant.value)
        survivors = [
            statement
            for index, statement in enumerate(statements)
            if not is_string_statement(statement) or (keeps_head and index == 0)
        ]
        if len(survivors) == len(statements):
            continue
        if not survivors and not isinstance(owner, ast.Module):
            marker = ast.Pass()
            expectation.placeholders.add(id(marker))
            survivors = [marker]
        statements[:] = survivors
    return expectation


def same_tree(old: object, new: object, expectation: Expectation) -> bool:
    if id(old) in expectation.placeholders:
        return isinstance(new, ast.Pass) or is_ellipsis_statement(new)
    if id(old) in expectation.rewrites:
        return isinstance(new, ast.Constant) and new.value == expectation.rewrites[id(old)]
    if isinstance(old, ast.AST):
        return type(old) is type(new) and all(
            same_tree(getattr(old, name, None), getattr(new, name, None), expectation)
            for name in old._fields
        )
    if isinstance(old, list):
        return (
            isinstance(new, list)
            and len(old) == len(new)
            and all(same_tree(a, b, expectation) for a, b in zip(old, new, strict=True))
        )
    return type(old) is type(new) and old == new


def verify(old_text: str, new_text: str, rules: Rules) -> None:
    expected = ast.parse(old_text)
    expectation = prune_strings(expected, rules.keeps_docstring)
    if not same_tree(expected, ast.parse(new_text), expectation):
        raise ProseError("the result differs from the source by more than prose")
    if plan_source(new_text, rules).dirty:
        raise ProseError("a second pass would change the result")


def render_report(findings: dict[str, dict[str, list[int]]], verb: str) -> str:
    out = []
    for rel, hits in findings.items():
        for name in CATEGORIES:
            if hits[name]:
                out.append(f"{rel}: {len(hits[name])} {name} line(s) {verb}: {hits[name]}")
    totals = {name: sum(len(hits[name]) for hits in findings.values()) for name in CATEGORIES}
    out.append(
        "totals: "
        + ", ".join(f"{name}={totals[name]}" for name in CATEGORIES)
        + f" across {len(findings)} module(s)"
    )
    return "\n".join(out)


def first_party_names(root: Path) -> frozenset[str]:
    names = {path.name for path in root.iterdir() if path.is_dir()}
    names.update(path.stem for path in root.glob("*.py"))
    return frozenset(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove prose from the files the bundler ships, or fail when any is present."
    )
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 on any prose")
    parser.add_argument("--scope", action="store_true", help="list the files in scope and exit")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="tree to work on")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    scoped = scope_files(root)
    if args.scope:
        for item in scoped:
            print(item.rel + ("  (frontmatter docstring reduced to its key lines)" if item.keeps_docstring else ""))
        return 0

    first_party = first_party_names(root)
    findings: dict[str, dict[str, list[int]]] = {}
    rewrites: list[tuple[ScopedFile, str]] = []
    for item in scoped:
        text = item.path.read_bytes().decode("utf-8")
        rules = Rules(item.keeps_docstring, first_party)
        new_text = text
        try:
            plan = plan_source(text, rules)
            if plan.dirty and not args.check:
                if plan.unremovable:
                    raise ProseError(plan.unremovable[0])
                new_text = render(text, plan)
                verify(text, new_text, rules)
        except (ProseError, SyntaxError, tokenize.TokenError, RecursionError) as exc:
            fail(f"{item.rel}: {exc}")
        if plan.dirty:
            findings[item.rel] = {name: sorted(plan.found[name]) for name in CATEGORIES}
            rewrites.append((item, new_text))

    print(render_report(findings, "found" if args.check else "removed"))
    if args.check:
        return 1 if findings else 0
    for item, new_text in rewrites:
        item.path.write_bytes(new_text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
