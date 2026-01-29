#!/usr/bin/env python3
"""
Bundle the modular open_webui_openrouter_pipe package into a single monolithic .py file.

This bundler uses a "single-file package" strategy:
- Embed each package module as source text
- Install a sys.meta_path finder/loader to serve imports from the embedded sources

Two output formats are supported:
- Raw (default): embedded sources are readable/editable Python strings
- Compressed (--compress): embedded sources are zlib+base64 blobs inflated at import time

Usage:
    python scripts/bundle.py
    python scripts/bundle.py --compress
    python scripts/bundle.py --output PATH

Output (defaults):
    open_webui_openrouter_pipe_bundled.py
    open_webui_openrouter_pipe_bundled_compressed.py   (with --compress)
"""

from __future__ import annotations

import argparse
import base64
import re
import zlib
from pathlib import Path


# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"
STUB_FILE = PROJECT_ROOT / "open_webui_openrouter_pipe.py"

DEFAULT_OUTPUT_RAW = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled.py"
DEFAULT_OUTPUT_COMPRESSED = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"


def _read_stub_version(stub_path: Path) -> str:
    content = stub_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\n]+)", content, re.MULTILINE)
    return match.group(1).strip() if match else "0.0.0"


def _render_header(*, version: str, compressed: bool) -> str:
    description_prefix = " and compressed " if compressed else " "
    return f'''"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: OpenRouter Responses API integration for Open WebUI (bundled{description_prefix}monolith)
required_open_webui_version: 0.7.0
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow
license: MIT
"""'''


def collect_modules(package_dir: Path) -> dict[str, str]:
    """Collect all Python modules under `package_dir` as {module_path: source}."""
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


def _literal_lines_for_text(text: str) -> list[str]:
    """Render `text` as one or more adjacent Python string literals.

    Intended to be placed inside parentheses, one literal per line, so Python
    concatenates them at compile time.
    """
    if not text:
        return []
    if "'''" in text:
        raise ValueError("text contains triple-single-quote delimiter")

    # Raw strings cannot end with an odd number of backslashes. Split one off.
    if text.endswith("\\"):
        trailing = len(text) - len(text.rstrip("\\"))
        if trailing % 2 == 1:
            if len(text) == 1:
                return [repr("\\")]
            return [f"r'''{text[:-1]}'''", repr("\\")]

    return [f"r'''{text}'''"]


def _source_expr_raw(source: str) -> str:
    """Return a Python expression which evaluates to `source` (readable raw-string form)."""
    parts = source.split("'''")
    expr_lines: list[str] = ["("]
    for idx, part in enumerate(parts):
        for lit in _literal_lines_for_text(part):
            expr_lines.append(f"    {lit}")
        if idx < len(parts) - 1:
            expr_lines.append('    "\'\'\'"')
    expr_lines.append(")")
    return "\n".join(expr_lines)


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


def _compress_source_zlib_base64(source: str) -> str:
    raw = source.encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.b64encode(comp).decode("ascii")


def _generate_runtime(*, compressed: bool) -> str:
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
    if compressed:
        lines.append("import base64")
        lines.append("import zlib")
    lines.append("")

    if compressed:
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
    else:
        lines.append("_BUNDLED_SOURCES: dict[str, str] = {}")
        lines.append("")
        lines.append("def _bundled_source(fullname: str) -> str:")
        lines.append('    return _BUNDLED_SOURCES.get(fullname, "")')
        lines.append("")
        lines.append("def _bundled_has_module(fullname: str) -> bool:")
        lines.append("    return fullname in _BUNDLED_SOURCES")

    lines.append("")
    lines.append("def _bundled_is_package(fullname: str) -> bool:")
    lines.append('    prefix = fullname + "."')
    if compressed:
        lines.append("    return any(name.startswith(prefix) for name in _BUNDLED_SOURCES_Z)")
    else:
        lines.append("    return any(name.startswith(prefix) for name in _BUNDLED_SOURCES)")
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


def _generate_entry_point() -> str:
    return "\n".join(
        [
            "# =============================================================================",
            "# ENTRY POINT",
            "# =============================================================================",
            "# Import and export the Pipe class for Open WebUI",
            "",
            "from open_webui_openrouter_pipe import Pipe as BasePipe",
            "",
            '_MODULE_PREFIX = \"function_\"',
            "_runtime_id = __name__[len(_MODULE_PREFIX):] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id",
            "",
            "class Pipe(BasePipe):",
            "    id = _runtime_id",
            "",
            '__all__ = [\"Pipe\"]',
            "",
        ]
    )


def bundle(*, output_path: Path, compressed: bool) -> None:
    version = _read_stub_version(STUB_FILE)
    modules = collect_modules(PACKAGE_DIR)

    parts: list[str] = []
    parts.append(_render_header(version=version, compressed=compressed))
    parts.append("")
    parts.append(_generate_runtime(compressed=compressed))

    parts.append("# =============================================================================")
    parts.append("# BUNDLED MODULE SOURCES")
    parts.append("# =============================================================================")
    parts.append(f"# Total modules: {len(modules)}")
    parts.append("")

    if compressed:
        for module_name in sorted(modules):
            b64 = _compress_source_zlib_base64(modules[module_name])
            expr = _b64_chunks_expr(b64)
            parts.append(f"# --- {module_name} ---")
            parts.append(f"_BUNDLED_SOURCES_Z[{module_name!r}] = {expr}")
            parts.append("")
    else:
        for module_name in sorted(modules):
            expr = _source_expr_raw(modules[module_name])
            parts.append(f"# --- {module_name} ---")
            parts.append(f"_BUNDLED_SOURCES[{module_name!r}] = {expr}")
            parts.append("")

    parts.append(_generate_entry_point())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    mode = "compressed" if compressed else "raw"
    print(f"Wrote {output_path} ({mode}, {size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle open_webui_openrouter_pipe into a single monolithic .py file"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Store module sources as zlib+base64 blobs and inflate at import time",
    )
    parser.add_argument(
        "--output",
        "-o",
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
