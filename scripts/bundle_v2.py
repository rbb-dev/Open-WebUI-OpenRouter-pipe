#!/usr/bin/env python3
"""
Bundler v2 — True monolith bundler.

Two output formats:
- Raw (default): a single flat .py file with all code inlined as real Python.
  No string-embedded modules, no import hooks, no exec().
- Compressed (--compress): zlib+base64 string-blob format with a sys.meta_path
  import hook.  Smaller file size at the cost of exec()-based module loading.

Usage:
    python scripts/bundle_v2.py
    python scripts/bundle_v2.py --compress
    python scripts/bundle_v2.py --output PATH
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import re
import sys
import textwrap
import tokenize
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# Constants

PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"
STUB_FILE = PROJECT_ROOT / "open_webui_openrouter_pipe.py"
PACKAGE_NAME = "open_webui_openrouter_pipe"

DEFAULT_OUTPUT_RAW = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled.py"
DEFAULT_OUTPUT_COMPRESSED = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"
DEFAULT_OUTPUT_RAW_NO_PLUGINS = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled_no_plugins.py"
DEFAULT_OUTPUT_COMPRESSED_NO_PLUGINS = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled_compressed_no_plugins.py"
ANYIO_WORKAROUND_FILE = PROJECT_ROOT / "scripts" / "anyio_1111_workaround.py"


ANYIO_WORKAROUND_MARKER = "_apply_anyio_1111_workaround"

NAME_COLLISIONS: list[tuple[str, str, str]] = []


def split_physical_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, keeping the terminator.

    ``str.splitlines`` also breaks on ``\\v \\f \\x1c \\x1d \\x1e \\x85 \\u2028
    \\u2029``, and ``ast``/``tokenize`` line numbers count none of them. The package
    contains U+2028 and U+2029 as literals -- ``core/utils.py`` lists them in the
    separators it forbids in a rendered body -- so that one line reads as three, and
    every line below it in the same file is displaced. Indexing a token position or an
    AST ``lineno`` into ``splitlines`` output then addresses the wrong line, which in
    this program means deleting or rewriting the wrong one, in the shipped artifact
    only.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _load_anyio_workaround_block() -> str:
    if not ANYIO_WORKAROUND_FILE.exists():
        print(
            "bundling without the anyio #1111 workaround (source not present)",
            file=sys.stderr,
        )
        return ""
    return ANYIO_WORKAROUND_FILE.read_text(encoding="utf-8").rstrip()

STDLIB_NAMES: set[str] = set(getattr(sys, "stdlib_module_names", set()))

SKIP_FILES = {"pytest_bootstrap.py"}


# Data structures

@dataclass
class LineRange:
    """Inclusive line range (1-indexed, matching ast node lineno)."""
    start: int
    end: int


@dataclass
class AliasedImport:
    """An internal import with an alias: `from ..X import Y as Z`."""
    original_name: str
    alias: str
    lineno: int
    end_lineno: int


@dataclass
class TryExceptImport:
    """A try/except ImportError block that imports an optional dependency."""
    source_lines: list[str]
    imported_names: set[str]
    line_range: LineRange


@dataclass
class TypeCheckingBlock:
    """An `if TYPE_CHECKING:` block."""
    has_internal: bool
    has_external: bool
    external_import_lines: list[str]
    else_lines: list[str]
    has_else: bool
    line_range: LineRange


@dataclass
class ModuleInfo:
    dotted_name: str
    file_path: Path
    raw_source: str
    source_lines: list[str]
    tree: ast.Module
    # Dependency tracking
    internal_deps: set[str] = field(default_factory=set)
    delete_ranges: list[LineRange] = field(default_factory=list)
    alias_mappings: dict[str, str] = field(default_factory=dict)
    external_import_lines: list[str] = field(default_factory=list)
    dropped_import_comments: list[str] = field(default_factory=list)
    # Optional try/except blocks
    try_except_blocks: list[TryExceptImport] = field(default_factory=list)
    # TYPE_CHECKING blocks
    type_checking_blocks: list[TypeCheckingBlock] = field(default_factory=list)
    is_init: bool = False
    # Top-level names defined
    top_level_names: set[str] = field(default_factory=set)


def discover_modules(package_dir: Path, *, exclude_plugins: bool = False) -> dict[str, ModuleInfo]:
    """Walk the package directory and create ModuleInfo for each .py file.

    When ``exclude_plugins`` is set, files inside a plugin sub-package
    (``plugins/<name>/...``) are skipped; the framework (``plugins/*.py``) stays.
    """
    modules: dict[str, ModuleInfo] = {}

    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        if py_file.name in SKIP_FILES:
            continue

        relative = py_file.relative_to(package_dir)
        parts = list(relative.parts)

        if exclude_plugins and len(parts) >= 3 and parts[0] == "plugins":
            continue

        is_init = parts[-1] == "__init__.py"
        if is_init:
            parts = parts[:-1]
            dotted = PACKAGE_NAME if not parts else f"{PACKAGE_NAME}.{'.'.join(parts)}"
        else:
            parts[-1] = parts[-1][:-3]
            dotted = f"{PACKAGE_NAME}.{'.'.join(parts)}"

        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        modules[dotted] = ModuleInfo(
            dotted_name=dotted,
            file_path=py_file,
            raw_source=source,
            source_lines=split_physical_lines(source),
            tree=tree,
            is_init=is_init,
        )

    return modules


def _is_stdlib(module_name: str) -> bool:
    top = module_name.split(".")[0]
    return top in STDLIB_NAMES


def _is_internal(module_name: str) -> bool:
    return module_name.startswith(PACKAGE_NAME)


def _resolve_relative_import(node: ast.ImportFrom, module_dotted: str) -> str | None:
    """Resolve a relative import to an absolute dotted name."""
    if node.level == 0:
        return node.module
    parts = module_dotted.split(".")
    if node.level > len(parts):
        return None
    base_parts = parts[: -node.level] if node.level <= len(parts) else []
    if node.module:
        return ".".join(base_parts) + "." + node.module
    return ".".join(base_parts)


def _find_try_except_import_blocks(tree: ast.Module, source_lines: list[str]) -> list[TryExceptImport]:
    """Find try/except ImportError blocks at the MODULE TOP LEVEL only.

    Only top-level optional-dependency guards are hoisted.  Nested try/except
    blocks (inside functions) are left in place — they are runtime logic, not
    optional-import guards.
    """
    blocks: list[TryExceptImport] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = False
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id in ("ImportError", "ModuleNotFoundError"):
                catches_import_error = True
            elif isinstance(handler.type, ast.Tuple):
                for elt in handler.type.elts:
                    if isinstance(elt, ast.Name) and elt.id in ("ImportError", "ModuleNotFoundError"):
                        catches_import_error = True

        if not catches_import_error:
            continue

        has_import = False
        imported_names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                has_import = True
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        imported_names.add(alias.asname or alias.name.split(".")[-1])
                elif isinstance(child, ast.ImportFrom):
                    for alias in child.names:
                        imported_names.add(alias.asname or alias.name)

        for handler in node.handlers:
            for child in ast.walk(handler):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            imported_names.add(target.id)

        if not has_import:
            continue

        start = node.lineno
        end = node.end_lineno or node.lineno
        lines = source_lines[start - 1: end]

        blocks.append(TryExceptImport(
            source_lines=lines,
            imported_names=imported_names,
            line_range=LineRange(start, end),
        ))

    return blocks


def _find_type_checking_blocks(
    tree: ast.Module,
    source_lines: list[str],
    module_dotted: str,
    dropped_comments: list[str] | None = None,
) -> list[TypeCheckingBlock]:
    """Find `if TYPE_CHECKING:` blocks at the module top level."""
    blocks: list[TypeCheckingBlock] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        is_tc = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        ) or (
            isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.attr == "TYPE_CHECKING"
        )
        if not is_tc:
            continue

        has_internal = False
        has_external = False
        external_lines: list[str] = []

        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.ImportFrom):
                    resolved = _resolve_relative_import(child, module_dotted)
                    if (resolved and _is_internal(resolved)) or child.level > 0:
                        has_internal = True
                    else:
                        has_external = True
                        _n = ", ".join(
                            a.name + (f" as {a.asname}" if a.asname else "")
                            for a in child.names
                        )
                        _stmt = f"from {'.' * child.level}{child.module or ''} import {_n}"
                        _c = _trailing_comment(
                            source_lines, child.lineno, child.end_lineno or child.lineno
                        )
                        if _c and _is_directive_comment(_c):
                            _stmt = f"{_stmt}{_c}"
                        elif _c and dropped_comments is not None:
                            dropped_comments.append(f"{_stmt}{_c}")
                        external_lines.append(_stmt)
                elif isinstance(child, ast.Import):
                    has_external = True
                    _stmt = "import " + ", ".join(
                        a.name + (f" as {a.asname}" if a.asname else "")
                        for a in child.names
                    )
                    _c = _trailing_comment(
                        source_lines, child.lineno, child.end_lineno or child.lineno
                    )
                    if _c and _is_directive_comment(_c):
                        _stmt = f"{_stmt}{_c}"
                    elif _c and dropped_comments is not None:
                        dropped_comments.append(f"{_stmt}{_c}")
                    external_lines.append(_stmt)
            elif isinstance(child, ast.Pass):
                pass
            else:
                has_internal = True

        has_else = bool(node.orelse)
        else_lines: list[str] = []
        if has_else:
            else_body_start = node.orelse[0].lineno
            else_body_end = node.orelse[-1].end_lineno or node.orelse[-1].lineno
            if_body_end = node.body[-1].end_lineno or node.body[-1].lineno
            else_kw_line = else_body_start - 1
            for scan in range(else_body_start - 2, if_body_end - 1, -1):
                if source_lines[scan].strip().startswith("else"):
                    else_kw_line = scan + 1
                    break
            else_lines = source_lines[else_kw_line - 1: else_body_end]

        start = node.lineno
        end = node.end_lineno or node.lineno

        blocks.append(TypeCheckingBlock(
            has_internal=has_internal,
            has_external=has_external,
            external_import_lines=external_lines,
            else_lines=else_lines,
            has_else=has_else,
            line_range=LineRange(start, end),
        ))

    return blocks


def analyze_module(mod: ModuleInfo, all_modules: dict[str, ModuleInfo]) -> None:
    """Analyze a module's imports and classify them."""
    tree = mod.tree
    lines = mod.source_lines

    mod.try_except_blocks = _find_try_except_import_blocks(tree, lines)
    te_ranges = {(b.line_range.start, b.line_range.end) for b in mod.try_except_blocks}

    # Find TYPE_CHECKING blocks
    mod.type_checking_blocks = _find_type_checking_blocks(
        tree, lines, mod.dotted_name, mod.dropped_import_comments
    )
    tc_ranges = {(b.line_range.start, b.line_range.end) for b in mod.type_checking_blocks}

    def _in_special_block(lineno: int) -> bool:
        for s, e in te_ranges | tc_ranges:
            if s <= lineno <= e:
                return True
        return False

    top_level_nodes: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        top_level_nodes.add(id(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _in_special_block(node.lineno):
                continue

            is_top_level = id(node) in top_level_nodes

            if node.module == "__future__":
                mod.delete_ranges.append(LineRange(node.lineno, node.end_lineno or node.lineno))
                continue

            resolved = _resolve_relative_import(node, mod.dotted_name)
            if node.level > 0 or (resolved and _is_internal(resolved)):
                if resolved and is_top_level:
                    dep = resolved
                    while dep and dep not in all_modules and "." in dep:
                        dep = dep.rsplit(".", 1)[0]
                    if dep and dep in all_modules:
                        mod.internal_deps.add(dep)

                for alias in node.names:
                    if alias.asname and alias.asname != alias.name:
                        mod.alias_mappings[alias.asname] = alias.name

                # Mark for deletion
                mod.delete_ranges.append(LineRange(node.lineno, node.end_lineno or node.lineno))
                continue

            if is_top_level:
                if resolved and resolved.startswith("open_webui."):
                    pass
                else:
                    start = node.lineno
                    end = node.end_lineno or node.lineno
                    _names = ", ".join(
                        a.name + (f" as {a.asname}" if a.asname else "")
                        for a in node.names
                    )
                    stmt = f"from {'.' * node.level}{node.module or ''} import {_names}"
                    comment = _trailing_comment(lines, start, end)
                    if comment and _is_directive_comment(comment):
                        stmt = f"{stmt}{comment}"
                    elif comment:
                        mod.dropped_import_comments.append(f"{stmt}{comment}")
                    mod.external_import_lines.append(stmt)
                    mod.delete_ranges.append(LineRange(start, end))

        elif isinstance(node, ast.Import):
            if _in_special_block(node.lineno):
                continue
            is_top_level = id(node) in top_level_nodes
            if is_top_level:
                start = node.lineno
                end = node.end_lineno or node.lineno
                stmt = "import " + ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "")
                    for a in node.names
                )
                comment = _trailing_comment(lines, start, end)
                if comment and _is_directive_comment(comment):
                    stmt = f"{stmt}{comment}"
                elif comment:
                    mod.dropped_import_comments.append(f"{stmt}{comment}")
                mod.external_import_lines.append(stmt)
                mod.delete_ranges.append(LineRange(start, end))

    # Mark module docstring for deletion
    if (tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)):
        doc_node = tree.body[0]
        mod.delete_ranges.append(LineRange(doc_node.lineno, doc_node.end_lineno or doc_node.lineno))

    # Mark __all__ for deletion
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    mod.delete_ranges.append(LineRange(node.lineno, node.end_lineno or node.lineno))

    # Collect top-level names
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            mod.top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mod.top_level_names.add(target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            mod.top_level_names.add(node.target.id)
        elif isinstance(node, ast.Try):
            mod.top_level_names |= _try_block_bindings(node)


def topological_sort(modules: dict[str, ModuleInfo]) -> list[ModuleInfo]:
    """Kahn's algorithm — returns modules in dependency order (leaves first)."""
    # Build adjacency and in-degree
    in_degree: dict[str, int] = {name: 0 for name in modules}
    dependents: dict[str, list[str]] = defaultdict(list)

    for name, mod in modules.items():
        for dep in mod.internal_deps:
            if dep in modules and dep != name:
                dependents[dep].append(name)
                in_degree[name] += 1

    queue: deque[str] = deque()
    for name, deg in in_degree.items():
        if deg == 0:
            queue.append(name)

    ordered: list[str] = []
    while queue:
        current = sorted(queue)
        queue.clear()
        for name in current:
            ordered.append(name)
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    if len(ordered) != len(modules):
        missing = set(modules) - set(ordered)
        print(f"WARNING: Circular dependency detected involving: {missing}", file=sys.stderr)
        # Add remaining modules anyway
        ordered.extend(sorted(missing))

    return [modules[name] for name in ordered]


def _replace_name_token(source: str, old_name: str, new_name: str) -> str:
    """Replace occurrences of *old_name* with *new_name*, touching only NAME tokens.

    Strings, comments, and docstrings are never modified.  The original
    formatting (whitespace, newlines) is preserved byte-for-byte except at
    the replacement sites.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return re.sub(rf"\b{re.escape(old_name)}\b", new_name, source)

    positions: list[tuple[int, int]] = []
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == old_name:
            positions.append(tok.start)

    if not positions:
        return source

    lines = split_physical_lines(source)
    for row, col in reversed(positions):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + new_name + line[col + len(old_name):]

    return "".join(lines)


def process_module_body(mod: ModuleInfo) -> str:
    """Strip imports and return the processed module body."""
    lines = mod.source_lines[:]
    n = len(lines)

    delete_lines: set[int] = set()
    for r in mod.delete_ranges:
        for ln in range(r.start, r.end + 1):
            delete_lines.add(ln)

    for te in mod.try_except_blocks:
        for ln in range(te.line_range.start, te.line_range.end + 1):
            delete_lines.add(ln)

    # Also delete TYPE_CHECKING blocks
    for tc in mod.type_checking_blocks:
        for ln in range(tc.line_range.start, tc.line_range.end + 1):
            delete_lines.add(ln)

    result_lines: list[str] = []
    for i in range(n):
        lineno = i + 1
        if lineno not in delete_lines:
            result_lines.append(lines[i])

    text = "".join(result_lines)
    for alias_name, original_name in mod.alias_mappings.items():
        text = _replace_name_token(text, alias_name, original_name)

    src_lines = split_physical_lines(text)
    in_string = _lines_inside_multiline_strings(text)
    keep = [True] * len(src_lines)
    for i, line in enumerate(src_lines):
        stripped = line.strip()
        if not stripped or not stripped.startswith("#"):
            continue
        if (i + 1) in in_string:
            continue
        j = i + 1
        while j < len(src_lines) and not src_lines[j].strip():
            j += 1
        if j >= len(src_lines) or (
            src_lines[j].strip().startswith("#") and j > i + 1
        ):
            keep[i] = False
    text = "".join(line for line, k in zip(src_lines, keep) if k)

    text = _collapse_blank_runs_outside_strings(text)

    text = text.strip("\n")
    return text


def _lines_inside_multiline_strings(text: str) -> set[int]:
    """1-indexed line numbers spanned by a multi-line string literal."""
    import tokenize
    from io import StringIO

    spans: set[int] = set()
    fstring_kinds = {
        getattr(tokenize, name)
        for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }
    try:
        for tok in tokenize.generate_tokens(StringIO(text).readline):
            if tok.type in {tokenize.STRING, *fstring_kinds} and tok.end[0] > tok.start[0]:
                spans.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return spans


def _collapse_blank_runs_outside_strings(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2, but never inside a string literal.

    A plain ``re.sub`` over the source rewrote the *contents* of embedded templates:
    filter_manager renders the Open WebUI filters it installs from triple-quoted
    source held in this package, and a run of blank lines inside one of those strings
    was silently shortened. The flat bundle then installed a filter whose source
    differed from the one the package installs -- invisible unless something compares
    them byte for byte, and unbounded in principle, since any embedded template may
    depend on its own whitespace.
    """
    in_string = _lines_inside_multiline_strings(text)

    lines = split_physical_lines(text)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip():
            out.append(lines[i])
            i += 1
            continue
        run_start = i
        while i < len(lines) and not lines[i].strip():
            i += 1
        run = lines[run_start:i]
        # 1-indexed line numbers for the run
        touches_string = any(n in in_string for n in range(run_start + 1, i + 1))
        out.extend(run if touches_string or len(run) <= 2 else run[:2])
    return "".join(out)


def _is_directive_comment(comment: str) -> bool:
    """A trailing comment that instructs a tool rather than describing the code.

    These must survive hoisting: the bundle is linted and type-checked in CI, so a
    suppression that holds for the package and vanishes from the artifact produces a
    bundle-only failure with no obvious cause.
    """
    return bool(_DIRECTIVE_COMMENT_RE.search(comment))


_DIRECTIVE_COMMENT_RE = re.compile(
    r"#\s*(noqa|type:\s*ignore|pragma|pyright:|mypy:|pylint:|ruff:|flake8:)",
    re.IGNORECASE,
)


def _trailing_comment(lines: list[str], start: int, end: int) -> str:
    """Return the trailing comments on a hoisted import, or an empty string.

    Scans every physical line of the statement, not just the last: a parenthesised
    import ends on ``)``, so reading only that line misses a suppression written on
    the opening line or on any member line -- exactly the multi-line case the
    dropped-comment warning exists to surface.
    """
    found: list[str] = []
    for raw in lines[start - 1 : end]:
        text = raw.rstrip("\n")
        idx = text.find("#")
        if idx == -1:
            continue
        if text.count('"', 0, idx) % 2 or text.count("'", 0, idx) % 2:
            continue
        comment = text[idx:].strip()
        if comment not in found:
            found.append(comment)
    return "  " + " ".join(found) if found else ""


def collect_and_dedup_external_imports(ordered_modules: list[ModuleInfo]) -> tuple[list[str], list[str], set[str]]:
    """Collect all external imports, merge names per module, split into stdlib/third-party.

    For ``from X import a, b`` style imports, names are merged per module so that
    multiple modules importing different names from the same package produce a
    single combined import line.  Bare ``import X`` lines are deduplicated by
    their full module path.

    Returns (stdlib_lines, third_party_lines, all_imported_names) where the
    third element is the set of all local names brought into scope by the
    rendered imports (used to filter redundant TYPE_CHECKING imports).
    """
    from_imports: dict[str, set[str]] = defaultdict(set)
    import_directives: dict[str, str] = {}
    bare_directives: dict[str, str] = {}
    bare_imports: set[str] = set()

    for mod in ordered_modules:
        for raw_line in mod.external_import_lines:
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", line)
            if m:
                module_path = m.group(1)
                names_part = m.group(2).strip()
                # Split the directive off BEFORE the comma split. Left inline, it
                # travels with whichever name it followed; the names are then re-sorted,
                # and if that name no longer sorts last the comment swallows every name
                # after it -- the header binds fewer names than the module bodies lost,
                # and the build fails naming an unrelated module.
                comment = ""
                hash_at = names_part.find("#")
                if hash_at != -1:
                    comment = names_part[hash_at:].strip()
                    names_part = names_part[:hash_at].rstrip().rstrip(",").rstrip()
                names_part = names_part.strip("()")
                if comment:
                    existing = import_directives.get(module_path, "")
                    if comment not in existing:
                        import_directives[module_path] = (existing + " " + comment).strip()
                for token in names_part.split(","):
                    token = token.strip()
                    if token:
                        from_imports[module_path].add(token)
                continue
            m2 = re.match(r"^import\s+(.+)$", line)
            if m2:
                for token in m2.group(1).split(","):
                    token = token.strip()
                    if not token:
                        continue
                    # Split the directive off BEFORE storing. The token is the key the
                    # local name is derived from, and a trailing `# type: ignore` rode
                    # into it -- so `bare_local` read `iio  # type: ignore[...]` and
                    # could never match a colliding `from` import. The header still
                    # emitted both lines and the collision report stayed silent.
                    head, sep, comment = token.partition("#")
                    token = head.strip()
                    if not token:
                        continue
                    if sep and _is_directive_comment(sep + comment):
                        bare_directives[token] = (sep + comment).strip()
                    bare_imports.add(token)
                continue

    def _top_module(module_path: str) -> str:
        return module_path.split(".")[0]

    def _is_stdlib(module_path: str) -> bool:
        return _top_module(module_path) in STDLIB_NAMES

    def _sort_key_for_name(name: str) -> str:
        """Sort imported names: plain names first, then aliases, case-insensitive."""
        return name.lower()

    stdlib_lines: list[str] = []
    third_party_lines: list[str] = []
    seen_local_names: dict[str, str] = {}
    NAME_COLLISIONS.clear()
    seen_local_sections: dict[str, bool] = {}
    for module_path in sorted(from_imports):
        unique_names: list[str] = []
        for token in sorted(from_imports[module_path], key=_sort_key_for_name):
            local = token.split(" as ")[-1].strip()
            origin = f"from {module_path} import {token.strip()}"
            if local not in seen_local_names:
                seen_local_names[local] = origin
                seen_local_sections[local] = _is_stdlib(module_path)
                unique_names.append(token)
            else:
                NAME_COLLISIONS.append((local, seen_local_names[local], origin))
        if not unique_names:
            continue
        rendered = f"from {module_path} import {', '.join(unique_names)}"
        directive = import_directives.get(module_path, "")
        if directive:
            rendered = f"{rendered}  {directive}"
        if _is_stdlib(module_path):
            stdlib_lines.append(rendered)
        else:
            third_party_lines.append(rendered)

    for token in sorted(bare_imports):
        directive = bare_directives.get(token, "")
        rendered = f"import {token}" + (f"  {directive}" if directive else "")
        if " as " in token:
            bare_local = token.split(" as ")[-1].strip()
        else:
            bare_local = token.split(".")[0].strip()
        origin = f"import {token.strip()}"
        bare_is_stdlib = token.split(".")[0].split()[0] in STDLIB_NAMES
        if bare_local in seen_local_names:
            prior_is_stdlib = seen_local_sections.get(bare_local, bare_is_stdlib)
            if bare_is_stdlib != prior_is_stdlib:
                bare_wins = not bare_is_stdlib
            else:
                bare_wins = True
            if bare_wins:
                NAME_COLLISIONS.append((bare_local, origin, seen_local_names[bare_local]))
            else:
                NAME_COLLISIONS.append((bare_local, seen_local_names[bare_local], origin))
        else:
            seen_local_names[bare_local] = origin
            seen_local_sections[bare_local] = bare_is_stdlib
        top = token.split(".")[0].split()[0]
        if top in STDLIB_NAMES:
            stdlib_lines.append(rendered)
        else:
            third_party_lines.append(rendered)

    stdlib_lines.sort()
    third_party_lines.sort()

    all_imported_names: set[str] = set()
    for names_set in from_imports.values():
        for token in names_set:
            parts = token.split(" as ")
            all_imported_names.add(parts[-1].strip())
    for token in bare_imports:
        parts = token.split(" as ")
        all_imported_names.add(parts[-1].strip())

    return stdlib_lines, third_party_lines, all_imported_names


def collect_and_dedup_try_except(ordered_modules: list[ModuleInfo]) -> list[str]:
    """Collect try/except ImportError blocks, deduplicated by imported names."""
    seen_names: set[str] = set()
    result_lines: list[str] = []

    for mod in ordered_modules:
        for te in mod.try_except_blocks:
            # Check if any of the names are new
            new_names = te.imported_names - seen_names
            if not new_names:
                continue
            seen_names |= te.imported_names

            is_internal = False
            for line in te.source_lines:
                stripped = line.strip()
                if stripped.startswith("from .") or (
                    stripped.startswith("from ") and PACKAGE_NAME in stripped.split("import")[0]
                ):
                    is_internal = True
                    break

            if is_internal:
                continue

            # Emit the block
            block_text = "".join(te.source_lines)
            # Dedent to top level if needed
            block_text = textwrap.dedent(block_text)
            result_lines.append(block_text.rstrip("\n"))
            result_lines.append("")

    return result_lines


def collect_external_type_checking(
    ordered_modules: list[ModuleInfo],
    runtime_imported_names: set[str],
) -> tuple[list[str], list[str]]:
    """Collect external TYPE_CHECKING imports and else-branch lines.

    *runtime_imported_names* contains names already imported in the regular
    (non-TYPE_CHECKING) import section.  TYPE_CHECKING imports whose names are
    all covered by runtime imports are skipped — they'd be redundant.
    """
    tc_import_lines: list[str] = []
    else_lines: list[str] = []
    seen: set[str] = set()

    for mod in ordered_modules:
        for tc in mod.type_checking_blocks:
            if not tc.has_external:
                continue
            for line in tc.external_import_lines:
                normalized = line.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)

                m = re.match(r"^from\s+\S+\s+import\s+(.+)$", normalized)
                if m:
                    names_part = m.group(1).strip().strip("()")
                    local_names = set()
                    for token in names_part.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        parts = token.split(" as ")
                        local_names.add(parts[-1].strip())
                    if local_names and local_names <= runtime_imported_names:
                        continue

                tc_import_lines.append(f"    {normalized}")

            if tc.has_else and tc.else_lines:
                for line in tc.else_lines:
                    stripped = line.rstrip("\n")
                    if stripped.strip() and stripped.strip() not in seen:
                        else_lines.append(stripped)

    return tc_import_lines, else_lines


def _read_stub_version(stub_path: Path) -> str:
    content = stub_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\n]+)", content, re.MULTILINE)
    return match.group(1).strip() if match else "0.0.0"


def _render_header(*, version: str, compressed: bool) -> str:
    description_suffix = " (minified)" if compressed else ""
    return f'''"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: OpenRouter Responses API integration for Open WebUI (flat monolith{description_suffix})
required_open_webui_version: 0.9.1
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, imageio, imageio-ffmpeg, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow, yarl
license: MIT
"""'''


def _render_package_alias_shim(all_submodules: list[str]) -> str:
    """Generate the sys.modules shim so `from open_webui_openrouter_pipe.X import Y` works."""
    sub_list = ", ".join(repr(s) for s in sorted(all_submodules))

    component_names: set[str] = set()
    for sub in all_submodules:
        for part in sub.split("."):
            component_names.add(part)
    attr_set = ", ".join(repr(n) for n in sorted(component_names))

    intermediate_names: set[str] = set()
    for sub in all_submodules:
        parts = sub.split(".")
        for i in range(len(parts) - 1):
            intermediate_names.add(parts[i])
    intermediate_set = ", ".join(repr(n) for n in sorted(intermediate_names))

    # Children of each intermediate package, so a proxy answers from what the package
    # ACTUALLY contains rather than from the package-wide name set. `_SUBMODULE_ATTRS`
    # is every path component anywhere in the tree, so it holds "config" (from
    # core/config.py) and a `logging` proxy consulting it hands back the monolith for
    # `logging.config`, shadowing the stdlib module.
    children: dict[str, set[str]] = {}
    for sub in all_submodules:
        parts = sub.split(".")
        for i in range(len(parts) - 1):
            children.setdefault(parts[i], set()).add(parts[i + 1])
    children_map = ", ".join(
        f"{n!r}: frozenset({{{', '.join(repr(c) for c in sorted(cs))}}})"
        for n, cs in sorted(children.items())
    )

    return f'''
# =============================================================================
# PACKAGE ALIAS SHIM
# =============================================================================
# Makes `from open_webui_openrouter_pipe import X` and
# `from open_webui_openrouter_pipe.core.config import Y` work when running
# as a flat monolith file (not a real package).
# Also supports `import open_webui_openrouter_pipe.pipe as pipe_module`
# and `from open_webui_openrouter_pipe.core import config as cfg` via
# module-level __getattr__ (PEP 562).
#
# When a subpackage name shadows a global import (e.g., our "logging"
# subpackage vs stdlib "logging"), a lightweight proxy module is created
# that delegates attribute access to both the flat module (for submodule
# names) and the shadowed module (for its original API).

_SUBMODULE_ATTRS: frozenset[str] = frozenset({{{attr_set}}})
_INTERMEDIATE_PACKAGES: frozenset[str] = frozenset({{{intermediate_set}}})
_PACKAGE_CHILDREN: dict = {{{children_map}}}

def __getattr__(name: str):
    """Allow attribute access for submodule names (PEP 562).

    When Python executes ``import X.Y as alias``, the bytecode does
    ``getattr(sys.modules["X"], "Y")``.  Since every submodule maps to
    this same flat module, we return ourselves for any known submodule
    component name.
    """
    if name in _SUBMODULE_ATTRS:
        return sys.modules[__name__]
    raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")


class _PackageProxy(types.ModuleType):
    """Proxy module for subpackage names that shadow global imports.

    Delegates attribute access to the flat monolith module for submodule
    names, and to the shadowed module (e.g., stdlib ``logging``) for
    everything else.  This allows both ``logging.getLogger(...)`` and
    ``import open_webui_openrouter_pipe.logging.session_log_manager``
    to work correctly.
    """
    def __init__(self, fullname: str, flat_mod: types.ModuleType, shadowed: types.ModuleType, children=frozenset()):
        super().__init__(fullname)
        self._children = children
        self.__path__ = []
        self.__package__ = fullname
        self.__file__ = "<bundled-proxy>"
        self._flat = flat_mod
        self._shadowed = shadowed

    def __getattr__(self, name: str):
        # Only this package's OWN children resolve to the monolith. Consulting the
        # package-wide component set instead shadowed every stdlib attribute that
        # happened to share a name with any module anywhere in the tree.
        if name in self._children:
            return self._flat
        return getattr(self._shadowed, name)


def _install_package_alias() -> None:
    _this = sys.modules[__name__]
    _pkg = "{PACKAGE_NAME}"
    sys.modules[_pkg] = _this

    # Register submodule entries in sys.modules
    for _sub in [{sub_list}]:
        _full = f"{{_pkg}}.{{_sub}}"
        sys.modules[_full] = _this

    # For intermediate package names that shadow existing globals (e.g., our
    # "logging" subpackage vs stdlib "logging"), create proxy modules so that
    # both the original API and submodule imports work.
    for _name in _INTERMEDIATE_PACKAGES:
        _existing = _this.__dict__.get(_name)
        if _existing is not None and isinstance(_existing, types.ModuleType) and _existing is not _this:
            _proxy = _PackageProxy(
                f"{{_pkg}}.{{_name}}", _this, _existing, _PACKAGE_CHILDREN.get(_name, frozenset())
            )
            sys.modules[f"{{_pkg}}.{{_name}}"] = _proxy
            setattr(_this, _name, _proxy)

_install_package_alias()
del _install_package_alias
'''


def _render_entry_point() -> str:
    return '''
# =============================================================================
# ENTRY POINT
# =============================================================================
# Import and export the Pipe class for Open WebUI

_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX):] if __name__.startswith(_MODULE_PREFIX) else Pipe.id

class _BundledPipe(Pipe):
    id = _runtime_id

Pipe = _BundledPipe  # type: ignore[misc]

__all__ = ["Pipe"]
'''


def assemble(
    *,
    version: str,
    compressed: bool,
    stdlib_imports: list[str],
    third_party_imports: list[str],
    optional_imports: list[str],
    tc_import_lines: list[str],
    tc_else_lines: list[str],
    module_blocks: list[tuple[str, str]],
    submodule_list: list[str],
) -> str:
    """Assemble the final output."""
    parts: list[str] = []

    # 1. Header
    parts.append(_render_header(version=version, compressed=compressed))
    parts.append("")

    parts.append("from __future__ import annotations")
    parts.append("")
    parts.append(f'__version__ = "{version}"')
    parts.append("")

    workaround = _load_anyio_workaround_block()
    if workaround:
        parts.append(workaround)
        parts.append("")

    # 3. Stdlib imports
    if stdlib_imports:
        parts.append("# =============================================================================")
        parts.append("# STDLIB IMPORTS")
        parts.append("# =============================================================================")
        parts.append("")
        for _i, line in enumerate(stdlib_imports):
            parts.append(line + ("  # noqa: I001" if _i == 0 else ""))
        parts.append("")

    # 4. Third-party imports
    if third_party_imports:
        parts.append("# =============================================================================")
        parts.append("# THIRD-PARTY IMPORTS")
        parts.append("# =============================================================================")
        parts.append("")
        parts.extend(third_party_imports)
        parts.append("")

    # 5. Optional dependencies
    if optional_imports:
        parts.append("# =============================================================================")
        parts.append("# OPTIONAL DEPENDENCIES")
        parts.append("# =============================================================================")
        parts.append("")
        parts.extend(optional_imports)
        parts.append("")

    if tc_import_lines:
        parts.append("# =============================================================================")
        parts.append("# TYPE_CHECKING (external types only)")
        parts.append("# =============================================================================")
        parts.append("")
        parts.append("if TYPE_CHECKING:")
        parts.extend(tc_import_lines)
        if tc_else_lines:
            parts.extend(tc_else_lines)
        parts.append("")

    # 7. Module bodies
    parts.append("# =============================================================================")
    parts.append("# MODULE BODIES")
    parts.append("# =============================================================================")
    parts.append("")

    for short_name, body in module_blocks:
        parts.append(f"# -- {short_name} " + "-" * max(1, 77 - len(short_name) - 4))
        parts.append("")
        parts.append(body)
        parts.append("")
        parts.append("")

    # 9. Package alias shim
    parts.append(_render_package_alias_shim(submodule_list))

    # 10. Entry point
    parts.append(_render_entry_point())

    return "\n".join(parts) + "\n"


def _collect_all_modules(package_dir: Path, *, exclude_plugins: bool = False) -> dict[str, str]:
    """Collect all Python modules under *package_dir* as ``{dotted_name: source}``.

    Includes ``__init__.py`` files. When ``exclude_plugins`` is set, files inside
    a plugin sub-package (``plugins/<name>/...``) are skipped; the framework
    (``plugins/*.py``) stays — so the bundle manifest simply lists no plugins.

    Sources are embedded verbatim and executed only when first imported, so each is
    parsed here. Without that, a module with a syntax error produces a bundle whose
    own ``ast.parse`` succeeds, which is the only gate release publishing applies.
    """
    modules: dict[str, str] = {}
    package_name = package_dir.name

    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        relative = py_file.relative_to(package_dir)
        parts = list(relative.parts)

        if exclude_plugins and len(parts) >= 3 and parts[0] == "plugins":
            continue

        if py_file.name in SKIP_FILES:
            continue

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            module_path = package_name if not parts else f"{package_name}.{'.'.join(parts)}"
        else:
            parts[-1] = parts[-1][:-3]
            module_path = f"{package_name}.{'.'.join(parts)}"

        source = py_file.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            raise SyntaxError(
                f"{py_file} does not parse, so the compressed bundle would embed a "
                f"module that fails only when first imported: {exc}"
            ) from exc
        modules[module_path] = source

    return modules


def _compress_source_zlib_base64(source: str) -> str:
    raw = source.encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.b64encode(comp).decode("ascii")


def _b64_chunks_expr(b64_text: str, *, chunk_size: int = 120) -> str:
    """Return a Python expression for a long base64 string using adjacent string literals."""
    if len(b64_text) <= chunk_size:
        return repr(b64_text)

    chunks = [b64_text[i : i + chunk_size] for i in range(0, len(b64_text), chunk_size)]
    lines = ["("]
    for chunk in chunks:
        lines.append(f"    {chunk!r}")
    lines.append(")")
    return "\n".join(lines)


def _render_header_compressed(*, version: str) -> str:
    return f'''"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: OpenRouter Responses API integration for Open WebUI (bundled and compressed monolith)
required_open_webui_version: 0.9.1
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, imageio, imageio-ffmpeg, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow, yarl
license: MIT
"""'''


def _generate_compressed_runtime() -> str:
    lines: list[str] = []
    lines.append("# =============================================================================")
    lines.append("# BUNDLED IMPORT HOOK")
    lines.append("# =============================================================================")
    lines.append("# - Loads open_webui_openrouter_pipe.* from embedded sources")
    lines.append("# - Populates linecache so inspect.getsource() works")
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import base64")
    lines.append("import io")
    lines.append("import linecache")
    lines.append("import sys")
    lines.append("import zlib")
    lines.append("from importlib.abc import Loader, MetaPathFinder")
    lines.append("from importlib.machinery import ModuleSpec")
    lines.append("")
    lines.append("_BUNDLED_SOURCES_Z: dict[str, str] = {}")
    lines.append("_BUNDLED_SOURCES: dict[str, str] = {}  # decompressed cache")
    lines.append("")
    lines.append("def _bundled_source(fullname: str) -> str:")
    lines.append("    cached = _BUNDLED_SOURCES.get(fullname)")
    lines.append("    if cached is not None:")
    lines.append("        return cached")
    lines.append("    payload = _BUNDLED_SOURCES_Z.get(fullname)")
    lines.append('    if payload is None:')
    lines.append('        return ""')
    lines.append("    raw = zlib.decompress(base64.b64decode(payload))")
    lines.append('    text = raw.decode(\"utf-8\")')
    lines.append("    _BUNDLED_SOURCES[fullname] = text")
    lines.append("    return text")
    lines.append("")
    lines.append("def _bundled_has_module(fullname: str) -> bool:")
    lines.append("    return fullname in _BUNDLED_SOURCES_Z")
    lines.append("")
    lines.append("def _bundled_is_package(fullname: str) -> bool:")
    lines.append('    prefix = fullname + "."')
    lines.append("    return any(name.startswith(prefix) for name in _BUNDLED_SOURCES_Z)")
    lines.append("")
    lines.append("class _BundledModuleFinder(MetaPathFinder):")
    lines.append("    def find_spec(self, fullname, path, target=None):")
    lines.append("        if not _bundled_has_module(fullname):")
    lines.append("            return None")
    lines.append("        return ModuleSpec(")
    lines.append("            fullname,")
    lines.append("            _BundledModuleLoader(fullname),")
    lines.append("            is_package=_bundled_is_package(fullname),")
    lines.append("        )")
    lines.append("")
    lines.append("    def bundled_module_names(self):")
    lines.append("        # Generic manifest accessor: lets the plugin system discover")
    lines.append("        # plugin packages in a compressed bundle with no hard-coded names.")
    lines.append("        return list(_BUNDLED_SOURCES_Z)")
    lines.append("")
    lines.append("class _BundledModuleLoader(Loader):")
    lines.append("    def __init__(self, fullname: str):")
    lines.append("        self.fullname = fullname")
    lines.append("")
    lines.append("    def create_module(self, spec):")
    lines.append("        return None  # default module creation")
    lines.append("")
    lines.append("    def exec_module(self, module):")
    lines.append("        if _bundled_is_package(self.fullname):")
    lines.append("            module.__path__ = []")
    lines.append("            module.__package__ = self.fullname")
    lines.append("        else:")
    lines.append('            module.__package__ = self.fullname.rpartition(\".\")[0] or self.fullname')
    lines.append("")
    lines.append('        module.__file__ = f\"<bundled:{self.fullname}>\"')
    lines.append("")
    lines.append("        source = _bundled_source(self.fullname)")
    lines.append("        if not source.strip():")
    lines.append("            return")
    lines.append("")
    lines.append("        # Make inspect.getsource() work for bundled modules")
    lines.append("        linecache.cache[module.__file__] = (")
    lines.append("            len(source),")
    lines.append("            None,")
    lines.append("            io.StringIO(source).readlines(),")
    lines.append("            module.__file__,")
    lines.append("        )")
    lines.append("")
    lines.append('        code = compile(source, module.__file__, \"exec\")')
    lines.append(
        "        exec(code, module.__dict__)  # noqa: S102 - executing the embedded "
        "modules is the whole mechanism of the compressed bundle"
    )
    lines.append("")
    lines.append("def _install_bundled_finder() -> None:")
    lines.append("    sys.meta_path[:] = [")
    lines.append("        _f for _f in sys.meta_path")
    lines.append("        if type(_f).__name__ != '_BundledModuleFinder'")
    lines.append("    ]")
    lines.append("    for _key in list(sys.modules):")
    lines.append("        if _key == 'open_webui_openrouter_pipe' or _key.startswith('open_webui_openrouter_pipe.'):")
    lines.append("            del sys.modules[_key]")
    lines.append("    sys.meta_path.insert(0, _BundledModuleFinder())")
    lines.append("")
    lines.append("_install_bundled_finder()")
    lines.append("")
    return "\n".join(lines)


def _generate_compressed_entry_point() -> str:
    return """\
# =============================================================================
# ENTRY POINT
# =============================================================================
# Import and export the Pipe class for Open WebUI

from open_webui_openrouter_pipe import Pipe as BasePipe

_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX):] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id

class Pipe(BasePipe):
    id = _runtime_id

__all__ = ["Pipe"]
"""


def _bundle_compressed(*, output_path: Path, version: str, no_plugins: bool = False) -> None:
    """Produce a compressed bundle using zlib+base64 string blobs with an import hook."""
    modules = _collect_all_modules(PACKAGE_DIR, exclude_plugins=no_plugins)

    parts: list[str] = []
    parts.append(_render_header_compressed(version=version))
    parts.append("")
    parts.append(_generate_compressed_runtime())

    workaround = _load_anyio_workaround_block()
    if workaround:
        parts.append(workaround)
        parts.append("")

    parts.append("# =============================================================================")
    parts.append("# BUNDLED MODULE SOURCES")
    parts.append("# =============================================================================")
    parts.append(f"# Total modules: {len(modules)}")
    parts.append("")

    for module_name in sorted(modules):
        b64 = _compress_source_zlib_base64(modules[module_name])
        expr = _b64_chunks_expr(b64)
        parts.append(f"# --- {module_name} ---")
        parts.append(f"_BUNDLED_SOURCES_Z[{module_name!r}] = {expr}")
        parts.append("")

    parts.append(_generate_compressed_entry_point())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path} (compressed, {size_kb:.1f} KB)")


def _try_block_bindings(node: ast.Try) -> set[str]:
    """Names a module-level try/except binds when the module body runs.

    The collection above walks `ast.iter_child_nodes`, so a name bound inside a
    module-level `try` is a grandchild and was never seen -- which meant the collision
    gate could not fire for any optional-import guard. That hole was invisible while
    the guards were spelled `except ImportError`, because those blocks were hoisted
    into the bundle header and deduplicated on the way; respelling the handlers as
    `except Exception` left them inline, still binding names, still uncollected.
    """
    bound: set[str] = set()

    def _walk(stmts: list[ast.stmt]) -> None:
        # NOT ast.walk: that descends into nested defs, so a fallback helper written
        # inside an `except` contributed its LOCALS as module-level names -- and since
        # the collision report became a raise, a local shared by two modules hard-fails
        # the build with "Rename one of them" pointing at nothing.
        for sub in stmts:
            if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                bound.add(sub.name)
                continue
            if isinstance(sub, ast.Assign):
                bound.update(t.id for t in sub.targets if isinstance(t, ast.Name))
            elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                bound.add(sub.target.id)
            elif isinstance(sub, ast.Import | ast.ImportFrom):
                bound.update((a.asname or a.name).split(".")[0] for a in sub.names)
            elif isinstance(sub, ast.If | ast.While | ast.For | ast.AsyncFor):
                _walk(sub.body)
                _walk(sub.orelse)
            elif isinstance(sub, ast.With | ast.AsyncWith):
                _walk(sub.body)
            elif isinstance(sub, ast.Try):
                _walk(sub.body)
                for handler in sub.handlers:
                    _walk(handler.body)
                _walk(sub.orelse)
                _walk(sub.finalbody)

    _walk(node.body)
    for handler in node.handlers:
        _walk(handler.body)
    _walk(node.orelse)
    _walk(node.finalbody)
    return bound


def _binding_signatures(mod: ModuleInfo, name: str) -> set[tuple] | None:
    """What *name* is bound TO in *mod*, canonically -- or None if not classifiable.

    Compared instead of source text because the guards differ only in type-checker
    suppression comments, which cannot change what object the name receives. Two
    modules that both run `from open_webui.models.files import Files` with a `None`
    fallback bind the same object in the collapsed namespace no matter how either is
    annotated; two that import the same NAME from different modules do not, and that
    difference survives here.

    Returning None means "no opinion" -- the caller falls back to the stricter textual
    rule rather than guessing.
    """
    # Abstain if anything OUTSIDE a top-level `try` also binds the name: this scan sees
    # only Try nodes, so a later `name = ...` at module scope is invisible to it and two
    # modules with matching guards were declared equivalent while the flat bundle bound
    # the rebound value for both. None means "no opinion"; the caller falls back to the
    # stricter textual rule.
    for child in ast.iter_child_nodes(mod.tree):
        if isinstance(child, ast.Try):
            continue
        if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            rebinds = child.name == name
        elif isinstance(child, ast.Assign):
            rebinds = any(isinstance(t, ast.Name) and t.id == name for t in child.targets)
        elif isinstance(child, ast.AnnAssign):
            rebinds = isinstance(child.target, ast.Name) and child.target.id == name
        elif isinstance(child, ast.Import | ast.ImportFrom):
            rebinds = any((a.asname or a.name).split(".")[0] == name for a in child.names)
        else:
            rebinds = False
        if rebinds:
            return None

    signatures: set[tuple] = set()
    for parent in ast.iter_child_nodes(mod.tree):
        if not isinstance(parent, ast.Try):
            continue
        for sub in ast.walk(parent):
            if isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    if (alias.asname or alias.name) == name:
                        signatures.add(("from", sub.module, sub.level, alias.name))
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    if (alias.asname or alias.name).split(".")[0] == name:
                        signatures.add(("import", alias.name, alias.asname))
            elif isinstance(sub, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in sub.targets
            ):
                if not isinstance(sub.value, ast.Constant):
                    return None
                signatures.add(("const", repr(sub.value.value)))
            elif (
                isinstance(sub, ast.AnnAssign)
                and isinstance(sub.target, ast.Name)
                and sub.target.id == name
            ):
                return None
    return signatures or None


def _bindings_are_equivalent(
    name: str,
    origins: list[str],
    mods_by_short: dict[str, ModuleInfo],
) -> bool:
    """True when every module binds *name* to demonstrably the same thing."""
    seen: list[set[tuple]] = []
    for origin in origins:
        signatures = _binding_signatures(mods_by_short[origin], name)
        if signatures is None:
            return False
        seen.append(signatures)
    return all(s == seen[0] for s in seen)


def _is_idempotent_definition(node: ast.AST) -> bool:
    """Assignments safe to re-execute in one namespace.

    Only a `logging.getLogger(...)` call qualifies: it returns the same object for the
    same name, so re-executing it in one namespace is a no-op.

    A bare `None` sentinel deliberately does NOT qualify. It looks idempotent, but it
    is the canonical shape for a module global that is later reassigned via `global`
    -- precisely the case where the flat bundle collapses two modules into one slot
    and the last writer silently wins for both. AnnAssign is covered by the same rule:
    it is not an ast.Assign, so it falls out below.
    """
    if not isinstance(node, ast.Assign):
        return False
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "getLogger"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "logging"
    )


def _definitions_textually_identical(
    name: str,
    origins: list[str],
    mods_by_short: dict[str, ModuleInfo],
) -> bool:
    """True when every top-level definition of *name* across *origins* is idempotent and textually identical."""
    texts: set[str] = set()
    for origin in origins:
        mod = mods_by_short[origin]
        # An origin this loop cannot classify -- a module-level `try` binding the name,
        # which _try_block_bindings does register -- must make the answer False, not sit
        # the vote out. Otherwise the surviving origins agree with themselves and a real
        # collision is reported benign.
        seen_here = False
        for node in ast.iter_child_nodes(mod.tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != name:
                    continue
            elif isinstance(node, ast.Assign):
                if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    continue
            elif isinstance(node, ast.AnnAssign):
                if not (isinstance(node.target, ast.Name) and node.target.id == name):
                    continue
            else:
                continue
            if not _is_idempotent_definition(node):
                return False
            segment = ast.get_source_segment(mod.raw_source, node)
            if not segment:
                return False
            seen_here = True
            texts.add(segment)
        if not seen_here:
            return False
    if not texts:
        return False
    return len(texts) == 1



def _report_name_collisions() -> None:
    """Surface every external name the header could bind only once.

    Benign when both modules re-export the same object (``fastapi.Request`` *is*
    ``starlette.requests.Request``); silently wrong otherwise, and silently wrong is
    the failure mode with no NameError to catch it. Proving identity needs the
    dependency installed, which a repo build has and the gh-pages in-browser builder
    does not, so the proof lives in test_bundle_name_collisions.py and this only
    reports.
    """
    if not NAME_COLLISIONS:
        return
    print(
        f"Deduplicated {len(NAME_COLLISIONS)} external name(s) in the shared header:",
        file=sys.stderr,
    )
    for name, kept, dropped in NAME_COLLISIONS:
        print(f"  {name}: kept from {kept}, dropped from {dropped}", file=sys.stderr)


def _report_dropped_import_comments(ordered_modules: list[ModuleInfo]) -> None:
    """Warn when a hoisted import's trailing comment will not reach the bundle.

    Tool directives (``# noqa``, ``# type: ignore``, ...) ride along with the hoisted
    statement -- see ``_is_directive_comment``. What reaches this report is descriptive
    prose, which a deduplicated header cannot carry.  That is fine as long
    as it is visible: otherwise a suppression that works in package mode silently
    disappears and resurfaces as a bundle-only CI failure with no obvious cause.
    """
    dropped = [(m.dotted_name, line) for m in ordered_modules for line in m.dropped_import_comments]
    if dropped:
        print(
            f"Dropped {len(dropped)} trailing comment(s) on hoisted imports "
            "(the bundle header cannot carry them):",
            file=sys.stderr,
        )
        for name, line in dropped:
            print(f"  {name}: {line}", file=sys.stderr)


def _assert_hoisted_imports_rebound(
    ordered_modules: list[ModuleInfo], rendered_lines: list[str]
) -> None:
    """Fail the build if a hoisted import was deleted but never re-emitted.

    The bundler removes top-level external imports from each module body and collects
    them into a shared header.  If a statement is lost in between, the bundle still
    parses and still imports -- it only raises NameError when the affected code path
    first runs.  The comparison is against the names actually bound by the RENDERED
    header, so it covers capture, parsing, and the renderer's duplicate-name dedup.
    """
    bound_by_header: set[str] = set()
    for line in rendered_lines:
        try:
            hdr = ast.parse(line.split("  #")[0]).body[0]
        except SyntaxError:
            continue
        if isinstance(hdr, ast.ImportFrom):
            bound_by_header.update(a.asname or a.name for a in hdr.names)
        elif isinstance(hdr, ast.Import):
            bound_by_header.update(a.asname or a.name.split(".")[0] for a in hdr.names)

    missing: dict[str, str] = {}
    for mod in ordered_modules:
        for stmt in mod.external_import_lines:
            try:
                node = ast.parse(stmt).body[0]
            except SyntaxError:
                raise RuntimeError(f"bundler: unparsable hoisted import {stmt!r}") from None
            if isinstance(node, ast.ImportFrom):
                bound = [a.asname or a.name for a in node.names]
            elif isinstance(node, ast.Import):
                bound = [a.asname or a.name.split(".")[0] for a in node.names]
            else:
                continue
            for name in bound:
                if name not in bound_by_header:
                    missing.setdefault(name, f"{mod.dotted_name}: {stmt}")
    if missing:
        detail = "\n".join(f"  {n}  <- {src}" for n, src in sorted(missing.items()))
        raise RuntimeError(
            "bundler: these imports were removed from module bodies but never re-emitted "
            f"in the header:\n{detail}"
        )


def validate_output(source: str, output_path: Path) -> bool:
    """Validate the generated bundle."""
    errors: list[str] = []

    # 1. Valid Python syntax
    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

    # 2. No leftover internal imports
    for i, line in enumerate(split_physical_lines(source), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check for relative imports
        if re.match(r"^\s*from\s+\.\S*\s+import", stripped):
            errors.append(f"Line {i}: leftover relative import: {stripped}")
        if (re.match(rf"^\s*from\s+{PACKAGE_NAME}\.\S+\s+import", stripped)
            and "_sys.modules" not in line
            and "sys.modules" not in line
            and not stripped.startswith(("#", '"', "'"))):
            errors.append(f"Line {i}: leftover absolute internal import: {stripped}")

    # 3. Pipe class exists
    if "class Pipe" not in source:
        errors.append("No 'class Pipe' found in output")

    # 4. __all__ exists
    if "__all__" not in source:
        errors.append("No '__all__' found in output")

    if errors:
        print(f"VALIDATION FAILED for {output_path}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    print(f"Validation passed: {output_path}")
    return True


# Main bundler

def bundle(*, output_path: Path, compressed: bool, no_plugins: bool = False) -> None:
    version = _read_stub_version(STUB_FILE)

    if compressed:
        _bundle_compressed(output_path=output_path, version=version, no_plugins=no_plugins)
        return


    all_modules = discover_modules(PACKAGE_DIR, exclude_plugins=no_plugins)
    print(f"Discovered {len(all_modules)} modules")

    for mod in all_modules.values():
        analyze_module(mod, all_modules)

    content_modules = {
        name: mod for name, mod in all_modules.items()
        if not mod.is_init
    }
    print(f"Content modules (non-__init__): {len(content_modules)}")

    ordered = topological_sort(content_modules)
    print(f"Topological order: {[m.dotted_name.removeprefix(PACKAGE_NAME + '.') for m in ordered]}")

    name_origins: dict[str, list[str]] = defaultdict(list)
    mods_by_short: dict[str, ModuleInfo] = {}
    for mod in ordered:
        short = mod.dotted_name.removeprefix(PACKAGE_NAME + ".")
        mods_by_short[short] = mod
        for name in mod.top_level_names:
            name_origins[name].append(short)
    collisions = {n: origins for n, origins in name_origins.items() if len(origins) > 1}
    benign = {
        name: origins
        for name, origins in collisions.items()
        if _definitions_textually_identical(name, origins, mods_by_short)
        or _bindings_are_equivalent(name, origins, mods_by_short)
    }
    for name in benign:
        collisions.pop(name)
    if benign:
        summary = ", ".join(f"{name} x{len(benign[name])}" for name in sorted(benign))
        print(f"Identical redefinitions (benign): {summary}")
    if collisions:
        detail = "\n".join(
            f"  {name}: defined in {', '.join(origins)}"
            for name, origins in sorted(collisions.items())
        )
        raise RuntimeError(
            f"{len(collisions)} top-level name(s) are defined by more than one module and "
            f"are not textually identical:\n{detail}\n"
            "The flat bundle collapses every module into one namespace, so the last "
            "definition wins and every call site that meant the other one silently "
            "changes behaviour -- with no NameError to catch it, and only in the shipped "
            "artifact, never in the package the tests import. Rename one of them."
        )

    stdlib_imports, third_party_imports, runtime_names = collect_and_dedup_external_imports(ordered)

    _assert_hoisted_imports_rebound(ordered, stdlib_imports + third_party_imports)
    _report_dropped_import_comments(ordered)
    _report_name_collisions()
    optional_imports = collect_and_dedup_try_except(ordered)
    tc_import_lines, tc_else_lines = collect_external_type_checking(ordered, runtime_names)

    if not any("import types" in line for line in stdlib_imports):
        stdlib_imports.append("import types")
        stdlib_imports.sort()

    module_blocks: list[tuple[str, str]] = []
    for mod in ordered:
        body = process_module_body(mod)
        short = mod.dotted_name.removeprefix(PACKAGE_NAME + ".")
        if body.strip():
            module_blocks.append((short, body))

    # Compute submodule list for the shim
    submodule_list: list[str] = []
    for name in all_modules:
        if name == PACKAGE_NAME:
            continue
        sub = name.removeprefix(PACKAGE_NAME + ".")
        submodule_list.append(sub)
        parts = sub.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in submodule_list:
                submodule_list.append(parent)
    submodule_list = sorted(set(submodule_list))

    output = assemble(
        version=version,
        compressed=False,
        stdlib_imports=stdlib_imports,
        third_party_imports=third_party_imports,
        optional_imports=optional_imports,
        tc_import_lines=tc_import_lines,
        tc_else_lines=tc_else_lines,
        module_blocks=module_blocks,
        submodule_list=submodule_list,
    )

    ok = validate_output(output, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

    if not ok:
        print("ERROR: Validation failed — output written for inspection but exiting with error", file=sys.stderr)
        sys.exit(1)

    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path} (readable, {size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle v2: create a true flat monolith from open_webui_openrouter_pipe"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use zlib+base64 compressed string blobs with import hook (smaller file)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (defaults depend on --compress / --no-plugins)",
    )
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="Exclude plugin sub-packages (plugins/<name>/...); bundle the framework only",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        if args.no_plugins:
            output = DEFAULT_OUTPUT_COMPRESSED_NO_PLUGINS if args.compress else DEFAULT_OUTPUT_RAW_NO_PLUGINS
        else:
            output = DEFAULT_OUTPUT_COMPRESSED if args.compress else DEFAULT_OUTPUT_RAW

    bundle(output_path=output, compressed=bool(args.compress), no_plugins=bool(args.no_plugins))


if __name__ == "__main__":
    main()
