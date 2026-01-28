#!/usr/bin/env python3
"""
Bundle the modular open_webui_openrouter_pipe package into a single monolithic .py file.

This script creates a standalone version of the pipe that can be pasted directly
into Open WebUI without requiring pip installation from GitHub.

Usage:
    python scripts/bundle.py [--output PATH]

Output:
    dist/open_webui_openrouter_pipe_bundled.py (default)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"
STUB_FILE = PROJECT_ROOT / "open_webui_openrouter_pipe.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "open_webui_openrouter_pipe_bundled.py"


def extract_stub_header(stub_path: Path) -> str:
    """Extract the docstring header from the stub file and transform for bundled version."""
    content = stub_path.read_text(encoding="utf-8")

    # Find the docstring (first triple-quoted block)
    match = re.match(r'^("""[\s\S]*?""")', content)
    if match:
        header = match.group(1)
    else:
        header = None

    # Build a proper bundled header
    # Extract version from stub if present (match "version:" at start of line, not "required_open_webui_version:")
    version_match = re.search(r'^version:\s*([^\n]+)', content, re.MULTILINE)
    version = version_match.group(1).strip() if version_match else "2.0.3"

    return f'''"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: OpenRouter Responses API integration for Open WebUI (bundled monolith - no pip install required)
required_open_webui_version: 0.7.0
version: {version}
requirements: aiohttp, cryptography, fastapi, httpx, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper, cairosvg, Pillow
license: MIT
"""'''


def collect_modules(package_dir: Path) -> Dict[str, str]:
    """
    Walk the package directory and collect all module sources.

    Returns a dict mapping module paths (e.g., "open_webui_openrouter_pipe.core.config")
    to their source code.
    """
    modules: Dict[str, str] = {}
    package_name = package_dir.name

    for py_file in sorted(package_dir.rglob("*.py")):
        # Skip __pycache__ and test files
        if "__pycache__" in str(py_file):
            continue

        # Calculate module path
        relative = py_file.relative_to(package_dir)
        parts = list(relative.parts)

        # Handle __init__.py -> package name
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            if not parts:
                module_path = package_name
            else:
                module_path = f"{package_name}.{'.'.join(parts)}"
        else:
            # Regular module
            parts[-1] = parts[-1][:-3]  # Remove .py
            module_path = f"{package_name}.{'.'.join(parts)}"

        # Read source
        source = py_file.read_text(encoding="utf-8")
        modules[module_path] = source

    return modules


def escape_source(source: str) -> str:
    """Escape source code for embedding in a triple-quoted string."""
    # We'll use a raw string approach with base64 for problematic content
    # But first, try simple escaping

    # Replace backslash-quote sequences that could break triple quotes
    # and escape any triple quotes in the source
    escaped = source.replace("\\", "\\\\")
    escaped = escaped.replace('"""', '\\"\\"\\"')

    return escaped


def generate_import_hook() -> str:
    """Generate the import hook code that loads bundled modules."""
    return '''
# =============================================================================
# BUNDLED IMPORT HOOK
# =============================================================================
# This hook intercepts imports for the bundled package and loads module sources
# from the _BUNDLED_MODULES dictionary instead of the filesystem.

import sys
import types
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

class _BundledModuleFinder(MetaPathFinder):
    """Meta path finder that locates bundled modules."""

    def find_spec(self, fullname, path, target=None):
        if fullname in _BUNDLED_MODULES:
            return ModuleSpec(fullname, _BundledModuleLoader(fullname), is_package=self._is_package(fullname))
        return None

    def _is_package(self, fullname):
        """Check if the module is a package (has submodules)."""
        prefix = fullname + "."
        return any(k.startswith(prefix) for k in _BUNDLED_MODULES)


class _BundledModuleLoader(Loader):
    """Loader that executes bundled module source code."""

    def __init__(self, fullname):
        self.fullname = fullname

    def create_module(self, spec):
        # Use default module creation
        return None

    def exec_module(self, module):
        # Set up package attributes
        if any(k.startswith(self.fullname + ".") for k in _BUNDLED_MODULES):
            module.__path__ = []
            module.__package__ = self.fullname
        else:
            module.__package__ = self.fullname.rpartition(".")[0] or self.fullname

        # Set __file__ to a synthetic path for debugging
        module.__file__ = f"<bundled:{self.fullname}>"

        # Execute the module source
        source = _BUNDLED_MODULES.get(self.fullname, "")
        if source.strip():
            code = compile(source, module.__file__, "exec")
            exec(code, module.__dict__)


# Install the import hook BEFORE any imports from the bundled package
sys.meta_path.insert(0, _BundledModuleFinder())

'''


def generate_modules_dict(modules: Dict[str, str]) -> str:
    """Generate the _BUNDLED_MODULES dictionary definition."""
    lines = ["# ============================================================================="]
    lines.append("# BUNDLED MODULE SOURCES")
    lines.append("# =============================================================================")
    lines.append(f"# Total modules: {len(modules)}")
    lines.append("")
    lines.append("_BUNDLED_MODULES = {")

    # Sort modules to ensure consistent output
    for module_path in sorted(modules.keys()):
        source = modules[module_path]
        escaped = escape_source(source)
        lines.append(f'    "{module_path}": """')
        lines.append(escaped)
        lines.append('""",')
        lines.append("")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_entry_point() -> str:
    """Generate the entry point that exports the Pipe class."""
    return '''
# =============================================================================
# ENTRY POINT
# =============================================================================
# Import and export the Pipe class for Open WebUI

from open_webui_openrouter_pipe import Pipe

# Handle Open WebUI's module naming convention
_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX):] if __name__.startswith(_MODULE_PREFIX) else Pipe.id

# Create a wrapper class with the runtime ID
class Pipe(Pipe):
    id = _runtime_id

__all__ = ["Pipe"]
'''


def bundle(output_path: Path) -> None:
    """Bundle the package into a single file."""
    print(f"Bundling {PACKAGE_DIR.name}...")

    # Collect all module sources
    modules = collect_modules(PACKAGE_DIR)
    print(f"  Found {len(modules)} modules")

    # Calculate total lines
    total_lines = sum(source.count("\n") + 1 for source in modules.values())
    print(f"  Total source lines: {total_lines:,}")

    # Build the bundled file
    parts = []

    # 1. Header (generated for bundled version)
    header = extract_stub_header(STUB_FILE)
    parts.append(header)
    parts.append("")

    # 2. Import hook
    parts.append(generate_import_hook())

    # 3. Modules dictionary
    parts.append(generate_modules_dict(modules))

    # 4. Entry point
    parts.append(generate_entry_point())

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(parts)
    output_path.write_text(content, encoding="utf-8")

    # Report size
    size_kb = len(content.encode("utf-8")) / 1024
    print(f"  Output: {output_path}")
    print(f"  Size: {size_kb:.1f} KB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Bundle open_webui_openrouter_pipe into a single monolithic .py file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    bundle(args.output)


if __name__ == "__main__":
    main()
