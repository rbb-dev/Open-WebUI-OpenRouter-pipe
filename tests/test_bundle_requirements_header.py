"""Regression test: every runtime dependency declared in pyproject.toml must
also appear in the bundle's `requirements:` header line.

OWUI deployments that load a bundle file install dependencies by parsing the
header line — pyproject.toml is never read on the target machine. So if a
new top-level import is added in the source but the bundle header isn't
updated, the bundle ships broken (e.g. `ModuleNotFoundError: No module named
'imageio'` at request time).

This test catches that drift before it ships. It is run against:
  - The two `_render_header*` functions in `scripts/bundle_v2.py` directly,
    so the test catches the gap even if the bundle hasn't been regenerated.
  - The two on-disk bundle files (when present), so the test also catches
    the gap if a stale bundle is still around.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
READABLE_BUNDLE = ROOT / "open_webui_openrouter_pipe_bundled.py"
COMPRESSED_BUNDLE = ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"
BUNDLE_SCRIPT = ROOT / "scripts" / "bundle_v2.py"

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml  # type: ignore[import-not-found]


def _normalise_pkg(name: str) -> str:
    """PEP 503 normalisation: collapse runs of `[-_.]` to `-`, lowercase.

    `pydantic-core`, `pydantic_core`, and `Pydantic.Core` all refer to the
    same package on PyPI, and pip treats them as equivalent for install.
    """
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _read_pyproject_runtime_deps() -> set[str]:
    data = _toml.loads(PYPROJECT.read_text(encoding="utf-8"))
    raw_deps = data["project"]["dependencies"]
    return {_normalise_pkg(re.split(r"[<>=!~;\[]", d, 1)[0]) for d in raw_deps}


def _parse_requirements_line(text: str) -> set[str]:
    """Pull the comma-separated package names out of an OWUI bundle header
    `requirements: a, b, c` line."""
    match = re.search(r"^requirements:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise AssertionError("No `requirements:` line found")
    return {_normalise_pkg(p) for p in match.group(1).split(",") if p.strip()}


def _read_render_header_requirements() -> tuple[set[str], set[str]]:
    """Extract the two hardcoded `requirements:` lines from bundle_v2.py
    without executing the script. They live inside f-strings rendered by
    `_render_header` and `_render_header_compressed`."""
    text = BUNDLE_SCRIPT.read_text(encoding="utf-8")
    lines = [
        line for line in text.splitlines()
        if line.startswith("requirements:")
    ]
    if len(lines) != 2:
        raise AssertionError(
            f"Expected exactly 2 `requirements:` lines in bundle_v2.py "
            f"(one per header renderer), found {len(lines)}: {lines}"
        )
    return tuple(_parse_requirements_line(line) for line in lines)  # type: ignore[return-value]


@pytest.fixture(scope="module")
def runtime_deps() -> set[str]:
    return _read_pyproject_runtime_deps()


def test_pyproject_has_runtime_deps(runtime_deps):
    # Sanity check: pyproject.toml is parseable and has a non-empty deps list.
    # If this fails, the rest of the tests can't be trusted.
    assert runtime_deps, "pyproject.toml [project].dependencies is empty"


def test_render_header_requirements_match_pyproject(runtime_deps):
    """Both `_render_header*` functions must declare every runtime dep."""
    readable_reqs, compressed_reqs = _read_render_header_requirements()
    missing_readable = runtime_deps - readable_reqs
    missing_compressed = runtime_deps - compressed_reqs
    assert not missing_readable, (
        f"`_render_header` (readable bundle) missing pyproject deps: "
        f"{sorted(missing_readable)}. Update `scripts/bundle_v2.py:794`."
    )
    assert not missing_compressed, (
        f"`_render_header_compressed` (compressed bundle) missing pyproject "
        f"deps: {sorted(missing_compressed)}. Update `scripts/bundle_v2.py:1080`."
    )


def test_readable_bundle_header_matches_pyproject(runtime_deps):
    if not READABLE_BUNDLE.exists():
        pytest.skip("readable bundle not built")
    bundle_reqs = _parse_requirements_line(READABLE_BUNDLE.read_text(encoding="utf-8"))
    missing = runtime_deps - bundle_reqs
    assert not missing, (
        f"Readable bundle header missing pyproject deps: {sorted(missing)}. "
        f"Regenerate with `python scripts/bundle_v2.py`."
    )


def test_compressed_bundle_header_matches_pyproject(runtime_deps):
    if not COMPRESSED_BUNDLE.exists():
        pytest.skip("compressed bundle not built")
    bundle_reqs = _parse_requirements_line(COMPRESSED_BUNDLE.read_text(encoding="utf-8"))
    missing = runtime_deps - bundle_reqs
    assert not missing, (
        f"Compressed bundle header missing pyproject deps: {sorted(missing)}. "
        f"Regenerate with `python scripts/bundle_v2.py --compress`."
    )


def test_bundles_are_in_sync_with_renderer():
    """Catches stale-on-disk bundles: header line must match bundle_v2.py output."""
    readable_render, compressed_render = _read_render_header_requirements()
    if READABLE_BUNDLE.exists():
        readable_disk = _parse_requirements_line(
            READABLE_BUNDLE.read_text(encoding="utf-8")
        )
        assert readable_disk == readable_render, (
            f"Readable bundle on disk has different requirements than the "
            f"renderer would produce. Regenerate the bundle.\n"
            f"  on-disk:   {sorted(readable_disk)}\n"
            f"  renderer:  {sorted(readable_render)}"
        )
    if COMPRESSED_BUNDLE.exists():
        compressed_disk = _parse_requirements_line(
            COMPRESSED_BUNDLE.read_text(encoding="utf-8")
        )
        assert compressed_disk == compressed_render, (
            f"Compressed bundle on disk has different requirements than the "
            f"renderer would produce. Regenerate the bundle.\n"
            f"  on-disk:   {sorted(compressed_disk)}\n"
            f"  renderer:  {sorted(compressed_render)}"
        )


def test_imageio_and_ffmpeg_present_in_bundles():
    """Explicit check for the two libraries the video intent classifier
    depends on. If a refactor drops them from the import path in source but
    they remain in pyproject, the more general checks above won't fire — if
    someone drops them from BOTH, this test forces a decision instead of
    silent breakage."""
    if not BUNDLE_SCRIPT.exists():
        pytest.skip("bundle script not present")
    readable_render, compressed_render = _read_render_header_requirements()
    for name in ("imageio", "imageio-ffmpeg"):
        normalised = _normalise_pkg(name)
        assert normalised in readable_render, (
            f"`{name}` missing from `_render_header` readable-bundle "
            f"requirements line. The video intent classifier needs it for "
            f"frame extraction."
        )
        assert normalised in compressed_render, (
            f"`{name}` missing from `_render_header_compressed` "
            f"compressed-bundle requirements line."
        )
