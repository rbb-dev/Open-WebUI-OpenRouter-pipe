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
import re
import sys
import textwrap
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"
STUB_FILE = PROJECT_ROOT / "open_webui_openrouter_pipe.py"
PACKAGE_NAME = "open_webui_openrouter_pipe"

DEFAULT_OUTPUT_RAW = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled.py"
DEFAULT_OUTPUT_COMPRESSED = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"

# Modules known to be stdlib (Python 3.11+).  We use sys.stdlib_module_names
# at runtime, but keep a fallback for safety.
STDLIB_NAMES: set[str] = set(getattr(sys, "stdlib_module_names", set()))

SKIP_FILES = {"pytest_bootstrap.py"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
    source_lines: list[str]  # verbatim lines to preserve
    imported_names: set[str]  # names introduced (for dedup)
    line_range: LineRange


@dataclass
class TypeCheckingBlock:
    """An `if TYPE_CHECKING:` block."""
    has_internal: bool  # references internal modules?
    has_external: bool  # references external modules?
    external_import_lines: list[str]  # external import lines to hoist
    else_lines: list[str]  # else-branch lines (if any)
    has_else: bool
    line_range: LineRange  # covers the entire if/else


@dataclass
class ModuleInfo:
    dotted_name: str
    file_path: Path
    raw_source: str
    source_lines: list[str]
    tree: ast.Module
    # Dependency tracking
    internal_deps: set[str] = field(default_factory=set)
    # Lines to delete (1-indexed)
    delete_ranges: list[LineRange] = field(default_factory=list)
    # Alias mappings: alias_name → original_name (for text replacement, no assignments emitted)
    alias_mappings: dict[str, str] = field(default_factory=dict)
    # External imports (stdlib + third-party)
    external_import_lines: list[str] = field(default_factory=list)
    # Optional try/except blocks
    try_except_blocks: list[TryExceptImport] = field(default_factory=list)
    # TYPE_CHECKING blocks
    type_checking_blocks: list[TypeCheckingBlock] = field(default_factory=list)
    # Is this an __init__.py?
    is_init: bool = False
    # Top-level names defined
    top_level_names: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Step 1: Discovery
# ---------------------------------------------------------------------------

def discover_modules(package_dir: Path) -> dict[str, ModuleInfo]:
    """Walk the package directory and create ModuleInfo for each .py file."""
    modules: dict[str, ModuleInfo] = {}

    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        if py_file.name in SKIP_FILES:
            continue

        relative = py_file.relative_to(package_dir)
        parts = list(relative.parts)

        is_init = parts[-1] == "__init__.py"
        if is_init:
            parts = parts[:-1]
            dotted = PACKAGE_NAME if not parts else f"{PACKAGE_NAME}.{'.'.join(parts)}"
        else:
            parts[-1] = parts[-1][:-3]  # strip .py
            dotted = f"{PACKAGE_NAME}.{'.'.join(parts)}"

        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        modules[dotted] = ModuleInfo(
            dotted_name=dotted,
            file_path=py_file,
            raw_source=source,
            source_lines=source.splitlines(keepends=True),
            tree=tree,
            is_init=is_init,
        )

    return modules


# ---------------------------------------------------------------------------
# Step 2: Import classification & analysis
# ---------------------------------------------------------------------------

def _is_stdlib(module_name: str) -> bool:
    top = module_name.split(".")[0]
    return top in STDLIB_NAMES


def _is_internal(module_name: str) -> bool:
    return module_name.startswith(PACKAGE_NAME)


def _resolve_relative_import(node: ast.ImportFrom, module_dotted: str) -> str | None:
    """Resolve a relative import to an absolute dotted name."""
    if node.level == 0:
        return node.module
    # Compute the base package from the importing module's dotted name
    parts = module_dotted.split(".")
    # Go up `level` packages
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
        # Check if any handler catches ImportError or ModuleNotFoundError.
        # We intentionally exclude bare `except:` and `except Exception:` —
        # those are not optional-import guards.
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

        # Check if the try body contains an import
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

        # Also capture names assigned in except handler (e.g., `lz4frame = None`)
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
) -> list[TypeCheckingBlock]:
    """Find `if TYPE_CHECKING:` blocks at the module top level."""
    blocks: list[TypeCheckingBlock] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        # Check if test is `TYPE_CHECKING`
        is_tc = False
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            is_tc = True
        elif isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name):
            if node.test.attr == "TYPE_CHECKING":
                is_tc = True
        if not is_tc:
            continue

        has_internal = False
        has_external = False
        external_lines: list[str] = []

        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.ImportFrom):
                    resolved = _resolve_relative_import(child, module_dotted)
                    if resolved and _is_internal(resolved):
                        has_internal = True
                    elif child.level > 0:
                        has_internal = True  # relative import = internal
                    else:
                        has_external = True
                        # Capture the raw source lines for this import
                        istart = child.lineno
                        iend = child.end_lineno or child.lineno
                        external_lines.extend(source_lines[istart - 1: iend])
                elif isinstance(child, ast.Import):
                    has_external = True
                    istart = child.lineno
                    iend = child.end_lineno or child.lineno
                    external_lines.extend(source_lines[istart - 1: iend])
            elif isinstance(child, ast.Pass):
                pass  # skip `pass` statements
            else:
                # Non-import code in TYPE_CHECKING — treat as internal
                has_internal = True

        has_else = bool(node.orelse)
        else_lines: list[str] = []
        if has_else:
            else_start = node.orelse[0].lineno
            else_end = node.orelse[-1].end_lineno or node.orelse[-1].lineno
            # Include the `else:` line itself (one line before the else body)
            else_lines = source_lines[else_start - 2: else_end]

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

    # Find try/except ImportError blocks FIRST (so we can exclude their ranges)
    mod.try_except_blocks = _find_try_except_import_blocks(tree, lines)
    te_ranges = {(b.line_range.start, b.line_range.end) for b in mod.try_except_blocks}

    # Find TYPE_CHECKING blocks
    mod.type_checking_blocks = _find_type_checking_blocks(tree, lines, mod.dotted_name)
    tc_ranges = {(b.line_range.start, b.line_range.end) for b in mod.type_checking_blocks}

    def _in_special_block(lineno: int) -> bool:
        for s, e in te_ranges | tc_ranges:
            if s <= lineno <= e:
                return True
        return False

    # Determine which nodes are top-level (direct children of the module)
    top_level_nodes: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        top_level_nodes.add(id(node))

    # Walk the ENTIRE AST to find imports at any depth
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _in_special_block(node.lineno):
                continue

            is_top_level = id(node) in top_level_nodes

            # from __future__ import annotations
            if node.module == "__future__":
                mod.delete_ranges.append(LineRange(node.lineno, node.end_lineno or node.lineno))
                continue

            # Relative imports (level > 0) or absolute internal imports
            resolved = _resolve_relative_import(node, mod.dotted_name)
            if node.level > 0 or (resolved and _is_internal(resolved)):
                # Track dependency (only from top-level imports for ordering)
                if resolved and is_top_level:
                    dep = resolved
                    while dep and dep not in all_modules and "." in dep:
                        dep = dep.rsplit(".", 1)[0]
                    if dep and dep in all_modules:
                        mod.internal_deps.add(dep)

                # Record aliases for text replacement in the module body.
                # Instead of emitting `alias = original` assignments (which
                # can overwrite same-named globals from other modules), we
                # replace references to the alias with the original name.
                for alias in node.names:
                    if alias.asname and alias.asname != alias.name:
                        mod.alias_mappings[alias.asname] = alias.name

                # Mark for deletion
                mod.delete_ranges.append(LineRange(node.lineno, node.end_lineno or node.lineno))
                continue

            # External import — only hoist if top-level
            if is_top_level:
                if resolved and resolved.startswith("open_webui."):
                    # Open WebUI imports outside try/except — keep in place
                    pass
                else:
                    start = node.lineno
                    end = node.end_lineno or node.lineno
                    # Join multi-line imports into one statement string
                    stmt = "".join(lines[start - 1: end]).strip()
                    mod.external_import_lines.append(stmt)
                    mod.delete_ranges.append(LineRange(start, end))

        elif isinstance(node, ast.Import):
            if _in_special_block(node.lineno):
                continue
            is_top_level = id(node) in top_level_nodes
            if is_top_level:
                start = node.lineno
                end = node.end_lineno or node.lineno
                stmt = "".join(lines[start - 1: end]).strip()
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
        if isinstance(node, ast.ClassDef):
            mod.top_level_names.add(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            mod.top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mod.top_level_names.add(target.id)


# ---------------------------------------------------------------------------
# Step 3: Topological sort
# ---------------------------------------------------------------------------

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
        # Sort for determinism within same level
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
        for name in sorted(missing):
            ordered.append(name)

    return [modules[name] for name in ordered]


# ---------------------------------------------------------------------------
# Step 4: Process module body
# ---------------------------------------------------------------------------

def process_module_body(mod: ModuleInfo) -> str:
    """Strip imports and return the processed module body."""
    lines = mod.source_lines[:]  # copy
    n = len(lines)

    # Build a set of line numbers to delete
    delete_lines: set[int] = set()
    for r in mod.delete_ranges:
        for ln in range(r.start, r.end + 1):
            delete_lines.add(ln)

    # Also delete try/except blocks that will be hoisted
    for te in mod.try_except_blocks:
        for ln in range(te.line_range.start, te.line_range.end + 1):
            delete_lines.add(ln)

    # Also delete TYPE_CHECKING blocks
    for tc in mod.type_checking_blocks:
        for ln in range(tc.line_range.start, tc.line_range.end + 1):
            delete_lines.add(ln)

    result_lines: list[str] = []
    for i in range(n):
        lineno = i + 1  # 1-indexed
        if lineno not in delete_lines:
            result_lines.append(lines[i])

    # Apply alias replacements (alias → original_name) at word boundaries.
    # This replaces references to aliased imports with the original name,
    # producing natural code without alias assignment lines.
    text = "".join(result_lines)
    for alias_name, original_name in mod.alias_mappings.items():
        text = re.sub(rf"\b{re.escape(alias_name)}\b", original_name, text)

    # Clean up orphaned comment lines that remain after import removal.
    # Strategy: mark a standalone comment line as orphaned if the next
    # non-blank line is ALSO a standalone comment (not a code line).
    # Multi-line comment blocks that precede code are preserved.
    src_lines = text.splitlines(keepends=True)
    keep = [True] * len(src_lines)
    for i, line in enumerate(src_lines):
        stripped = line.strip()
        if not stripped or not stripped.startswith("#"):
            continue
        # Standalone comment line — look ahead past blanks
        j = i + 1
        while j < len(src_lines) and not src_lines[j].strip():
            j += 1
        if j >= len(src_lines):
            keep[i] = False  # comment at EOF with nothing after → orphaned
        elif src_lines[j].strip().startswith("#"):
            # Next non-blank is also a comment — but only mark as orphaned
            # if there's a blank line between them (import section pattern)
            if j > i + 1:
                keep[i] = False
    text = "".join(line for line, k in zip(src_lines, keep) if k)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    text = text.strip("\n")
    return text


# ---------------------------------------------------------------------------
# Step 5: External import hoisting and deduplication
# ---------------------------------------------------------------------------

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
    # {module_path: {name_or_alias, ...}}  for "from X import a, b as c" style
    from_imports: dict[str, set[str]] = defaultdict(set)
    # set of full "import X" / "import X as Y" tokens
    bare_imports: set[str] = set()

    for mod in ordered_modules:
        for raw_line in mod.external_import_lines:
            line = raw_line.strip()
            if not line:
                continue
            # Parse "from X import a, b as c"
            m = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", line)
            if m:
                module_path = m.group(1)
                names_part = m.group(2).strip()
                # Handle parenthesised form (shouldn't occur, but be safe)
                names_part = names_part.strip("()")
                for token in names_part.split(","):
                    token = token.strip()
                    if token:
                        from_imports[module_path].add(token)
                continue
            # Parse "import X, Y as Z"
            m2 = re.match(r"^import\s+(.+)$", line)
            if m2:
                for token in m2.group(1).split(","):
                    token = token.strip()
                    if token:
                        bare_imports.add(token)
                continue

    def _top_module(module_path: str) -> str:
        return module_path.split(".")[0]

    def _is_stdlib(module_path: str) -> bool:
        return _top_module(module_path) in STDLIB_NAMES

    def _sort_key_for_name(name: str) -> str:
        """Sort imported names: plain names first, then aliases, case-insensitive."""
        return name.lower()

    # Render "from X import ..." lines (merged), deduplicating local names
    # across modules (e.g., `Request` from both fastapi and starlette).
    stdlib_lines: list[str] = []
    third_party_lines: list[str] = []
    seen_local_names: set[str] = set()
    for module_path in sorted(from_imports):
        unique_names: list[str] = []
        for token in sorted(from_imports[module_path], key=_sort_key_for_name):
            # "X as Y" → local name is Y; "X" → local name is X
            local = token.split(" as ")[-1].strip()
            if local not in seen_local_names:
                seen_local_names.add(local)
                unique_names.append(token)
        if not unique_names:
            continue
        rendered = f"from {module_path} import {', '.join(unique_names)}"
        if _is_stdlib(module_path):
            stdlib_lines.append(rendered)
        else:
            third_party_lines.append(rendered)

    # Render bare "import X" lines
    for token in sorted(bare_imports):
        rendered = f"import {token}"
        top = token.split(".")[0].split()[0]  # handle "X as Y"
        if top in STDLIB_NAMES:
            stdlib_lines.append(rendered)
        else:
            third_party_lines.append(rendered)

    stdlib_lines.sort()
    third_party_lines.sort()

    # Build the set of all local names introduced by these imports.
    all_imported_names: set[str] = set()
    for names_set in from_imports.values():
        for token in names_set:
            # "X as Y" → local name is Y; "X" → local name is X
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
                continue  # All names already defined
            seen_names |= te.imported_names

            # Check if any import inside is internal (from .X or from open_webui_openrouter_pipe.X)
            is_internal = False
            for line in te.source_lines:
                stripped = line.strip()
                if stripped.startswith("from .") or (
                    stripped.startswith("from ") and PACKAGE_NAME in stripped.split("import")[0]
                ):
                    is_internal = True
                    break

            if is_internal:
                continue  # Skip internal try/except blocks

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

                # Parse the import to extract local names and check redundancy.
                m = re.match(r"^from\s+\S+\s+import\s+(.+)$", normalized)
                if m:
                    names_part = m.group(1).strip().strip("()")
                    local_names = set()
                    for token in names_part.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        # "X as Y" → local name is Y; "X" → local name is X
                        parts = token.split(" as ")
                        local_names.add(parts[-1].strip())
                    if local_names and local_names <= runtime_imported_names:
                        continue  # all names already available at runtime

                tc_import_lines.append(f"    {normalized}")

            if tc.has_else and tc.else_lines:
                for line in tc.else_lines:
                    stripped = line.rstrip("\n")
                    if stripped.strip() and stripped.strip() not in seen:
                        else_lines.append(stripped)

    return tc_import_lines, else_lines


# ---------------------------------------------------------------------------
# Step 6: Output assembly
# ---------------------------------------------------------------------------

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
required_open_webui_version: 0.8.0
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow
license: MIT
"""'''


def _render_package_alias_shim(all_submodules: list[str]) -> str:
    """Generate the sys.modules shim so `from open_webui_openrouter_pipe.X import Y` works."""
    # Build the list of all sub-module dotted paths
    sub_list = ", ".join(repr(s) for s in sorted(all_submodules))

    # Collect all unique path components so `import X.Y as alias` works.
    # Python's IMPORT_FROM opcode does getattr(parent, "child") — the child
    # must be an attribute on the parent module.  Since all submodules point
    # to the same flat module we use a module-level __getattr__ (PEP 562) to
    # return the module itself for any submodule component name.
    component_names: set[str] = set()
    for sub in all_submodules:
        for part in sub.split("."):
            component_names.add(part)
    attr_set = ", ".join(repr(n) for n in sorted(component_names))

    # Find intermediate package names (not leaf modules) for proxy generation.
    # These are components that have children: "core" in "core.config".
    intermediate_names: set[str] = set()
    for sub in all_submodules:
        parts = sub.split(".")
        for i in range(len(parts) - 1):
            intermediate_names.add(parts[i])
    # Also add top-level names that represent packages (have dot-separated children)
    intermediate_set = ", ".join(repr(n) for n in sorted(intermediate_names))

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
    def __init__(self, fullname: str, flat_mod: types.ModuleType, shadowed: types.ModuleType):
        super().__init__(fullname)
        self.__path__ = []
        self.__package__ = fullname
        self.__file__ = "<bundled-proxy>"
        self._flat = flat_mod
        self._shadowed = shadowed

    def __getattr__(self, name: str):
        if name in _SUBMODULE_ATTRS:
            return self._flat
        return getattr(self._shadowed, name)


def _install_package_alias() -> None:
    _this = sys.modules[__name__]
    _pkg = "{PACKAGE_NAME}"
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _this

    # Register submodule entries in sys.modules
    for _sub in [{sub_list}]:
        _full = f"{{_pkg}}.{{_sub}}"
        if _full not in sys.modules:
            sys.modules[_full] = _this

    # For intermediate package names that shadow existing globals (e.g., our
    # "logging" subpackage vs stdlib "logging"), create proxy modules so that
    # both the original API and submodule imports work.
    for _name in _INTERMEDIATE_PACKAGES:
        _existing = _this.__dict__.get(_name)
        if _existing is not None and isinstance(_existing, types.ModuleType) and _existing is not _this:
            _proxy = _PackageProxy(f"{{_pkg}}.{{_name}}", _this, _existing)
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
    module_blocks: list[tuple[str, str]],  # (short_name, processed_body)
    submodule_list: list[str],
) -> str:
    """Assemble the final output."""
    parts: list[str] = []

    # 1. Header
    parts.append(_render_header(version=version, compressed=compressed))
    parts.append("")

    # 2. Future annotations + version
    parts.append("from __future__ import annotations")
    parts.append("")
    parts.append(f'__version__ = "{version}"')
    parts.append("")

    # 3. Stdlib imports
    if stdlib_imports:
        parts.append("# =============================================================================")
        parts.append("# STDLIB IMPORTS")
        parts.append("# =============================================================================")
        parts.append("")
        for line in stdlib_imports:
            parts.append(line)
        parts.append("")

    # 4. Third-party imports
    if third_party_imports:
        parts.append("# =============================================================================")
        parts.append("# THIRD-PARTY IMPORTS")
        parts.append("# =============================================================================")
        parts.append("")
        for line in third_party_imports:
            parts.append(line)
        parts.append("")

    # 5. Optional dependencies
    if optional_imports:
        parts.append("# =============================================================================")
        parts.append("# OPTIONAL DEPENDENCIES")
        parts.append("# =============================================================================")
        parts.append("")
        for line in optional_imports:
            parts.append(line)
        parts.append("")

    # 6. TYPE_CHECKING (external only)
    if tc_import_lines:
        parts.append("# =============================================================================")
        parts.append("# TYPE_CHECKING (external types only)")
        parts.append("# =============================================================================")
        parts.append("")
        parts.append("if TYPE_CHECKING:")
        for line in tc_import_lines:
            parts.append(line)
        if tc_else_lines:
            for line in tc_else_lines:
                parts.append(line)
        parts.append("")

    # 7. Module bodies
    parts.append("# =============================================================================")
    parts.append("# MODULE BODIES")
    parts.append("# =============================================================================")
    parts.append("")

    for short_name, body in module_blocks:
        parts.append(f"# ── {short_name} " + "─" * max(1, 77 - len(short_name) - 4))
        parts.append("")
        parts.append(body)
        parts.append("")
        parts.append("")

    # 9. Package alias shim
    parts.append(_render_package_alias_shim(submodule_list))

    # 10. Entry point
    parts.append(_render_entry_point())

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Step 7: Compressed bundle (zlib+base64 string blobs)
# ---------------------------------------------------------------------------
# When --compress is used, each module's source is zlib-compressed and
# base64-encoded into a dict.  A lightweight sys.meta_path import hook
# decompresses and executes modules on demand at import time.


def _collect_all_modules(package_dir: Path) -> dict[str, str]:
    """Collect all Python modules under *package_dir* as ``{dotted_name: source}``.

    Includes ``__init__.py`` files and does not skip any files — every module
    is needed for the import-hook approach.
    """
    modules: dict[str, str] = {}
    package_name = package_dir.name

    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        relative = py_file.relative_to(package_dir)
        parts = list(relative.parts)

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            module_path = package_name if not parts else f"{package_name}.{'.'.join(parts)}"
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py
            module_path = f"{package_name}.{'.'.join(parts)}"

        modules[module_path] = py_file.read_text(encoding="utf-8")

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
        lines.append(f"    {repr(chunk)}")
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
required_open_webui_version: 0.8.0
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow
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
    lines.append("import linecache")
    lines.append("import sys")
    lines.append("from importlib.abc import Loader, MetaPathFinder")
    lines.append("from importlib.machinery import ModuleSpec")
    lines.append("import base64")
    lines.append("import zlib")
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
    lines.append("            source.splitlines(True),")
    lines.append("            module.__file__,")
    lines.append("        )")
    lines.append("")
    lines.append('        code = compile(source, module.__file__, \"exec\")')
    lines.append("        exec(code, module.__dict__)")
    lines.append("")
    lines.append("def _install_bundled_finder() -> None:")
    lines.append("    for finder in sys.meta_path:")
    lines.append("        if isinstance(finder, _BundledModuleFinder):")
    lines.append("            return")
    lines.append("    sys.meta_path.insert(0, _BundledModuleFinder())")
    lines.append("")
    lines.append("_install_bundled_finder()")
    lines.append("")
    return "\n".join(lines)


def _generate_compressed_entry_point() -> str:
    return "\n".join(
        [
            "# =============================================================================",
            "# ENTRY POINT",
            "# =============================================================================",
            "# Import and export the Pipe class for Open WebUI",
            "",
            "from open_webui_openrouter_pipe import Pipe as BasePipe",
            "",
            '_MODULE_PREFIX = "function_"',
            "_runtime_id = __name__[len(_MODULE_PREFIX):] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id",
            "",
            "class Pipe(BasePipe):",
            "    id = _runtime_id",
            "",
            '__all__ = ["Pipe"]',
            "",
        ]
    )


def _bundle_compressed(*, output_path: Path, version: str) -> None:
    """Produce a compressed bundle using zlib+base64 string blobs with an import hook."""
    modules = _collect_all_modules(PACKAGE_DIR)

    parts: list[str] = []
    parts.append(_render_header_compressed(version=version))
    parts.append("")
    parts.append(_generate_compressed_runtime())

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


# ---------------------------------------------------------------------------
# Step 8: Validation
# ---------------------------------------------------------------------------

def validate_output(source: str, output_path: Path) -> bool:
    """Validate the generated bundle."""
    errors: list[str] = []

    # 1. Valid Python syntax
    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

    # 2. No leftover internal imports
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check for relative imports
        if re.match(r"^\s*from\s+\.\S*\s+import", stripped):
            errors.append(f"Line {i}: leftover relative import: {stripped}")
        # Check for absolute internal imports (but not in the shim or string literals)
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


# ---------------------------------------------------------------------------
# Main bundler
# ---------------------------------------------------------------------------

def bundle(*, output_path: Path, compressed: bool) -> None:
    version = _read_stub_version(STUB_FILE)

    # Compressed mode: completely separate code path (string blobs + import hook)
    if compressed:
        _bundle_compressed(output_path=output_path, version=version)
        return

    # --- Flat monolith (default) ---

    # Step 1: Discover
    all_modules = discover_modules(PACKAGE_DIR)
    print(f"Discovered {len(all_modules)} modules")

    # Step 2: Analyze
    for mod in all_modules.values():
        analyze_module(mod, all_modules)

    # Filter out __init__.py files — they are pure re-exports
    content_modules = {
        name: mod for name, mod in all_modules.items()
        if not mod.is_init
    }
    print(f"Content modules (non-__init__): {len(content_modules)}")

    # Step 3: Topological sort
    ordered = topological_sort(content_modules)
    print(f"Topological order: {[m.dotted_name.removeprefix(PACKAGE_NAME + '.') for m in ordered]}")

    # Name collision detection: warn about top-level names defined in multiple modules
    name_origins: dict[str, list[str]] = defaultdict(list)
    for mod in ordered:
        for name in mod.top_level_names:
            name_origins[name].append(mod.dotted_name.removeprefix(PACKAGE_NAME + "."))
    collisions = {n: origins for n, origins in name_origins.items() if len(origins) > 1}
    if collisions:
        print(f"WARNING: {len(collisions)} name collision(s) detected:", file=sys.stderr)
        for name, origins in sorted(collisions.items()):
            print(f"  {name}: defined in {', '.join(origins)}", file=sys.stderr)

    # Step 5: Collect external imports
    stdlib_imports, third_party_imports, runtime_names = collect_and_dedup_external_imports(ordered)
    optional_imports = collect_and_dedup_try_except(ordered)
    tc_import_lines, tc_else_lines = collect_external_type_checking(ordered, runtime_names)

    # Ensure `types` is available (used by the package alias shim)
    if not any("import types" in line for line in stdlib_imports):
        stdlib_imports.append("import types")
        stdlib_imports.sort()

    # Step 4: Process module bodies
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
        # Also add intermediate packages (e.g., "core" for "core.config")
        parts = sub.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in submodule_list:
                submodule_list.append(parent)
    submodule_list = sorted(set(submodule_list))

    # Step 6: Assemble
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

    # Step 8: Validate
    ok = validate_output(output, output_path)
    if not ok:
        print("WARNING: Validation failed but writing output anyway", file=sys.stderr)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

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
        help="Output file path (defaults depend on --compress)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_COMPRESSED if args.compress else DEFAULT_OUTPUT_RAW

    bundle(output_path=output, compressed=bool(args.compress))


if __name__ == "__main__":
    main()
