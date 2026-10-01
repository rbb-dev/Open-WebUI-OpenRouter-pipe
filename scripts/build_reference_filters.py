#!/usr/bin/env python3
"""Regenerate the reference copies under filters/ from the pipe's own renderers.

    python scripts/build_reference_filters.py
    python scripts/build_reference_filters.py --filters-dir somewhere/else

The files under `filters/` are REFERENCE COPIES: reviewed in the repository and
installed by hand, beside the copies FilterManager renders and the pipe installs into
Open WebUI. Each one is the output of a renderer, so a renderer edit has to reach
both copies; this script is how it reaches the one in this repository. The parity
test in tests/test_web_tools_filter.py is the other half: it fails CI when the two
diverge, this is what a maintainer runs when they are meant to agree.

Write only when the bytes differ, and print what changed. A step that rewrites
unchanged files makes `git status` noise out of the one signal it should carry.

Only the renderers with a twin are listed. FilterManager has five; `video` and
`image` have no reference copy under `filters/`, and generating one here would
publish a file nothing asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILTERS_DIR = PROJECT_ROOT / "filters"

# Run from anywhere without PYTHONPATH set: the renderer lives in the package, and a
# regeneration step that only works when the caller remembered the environment is a
# step the next maintainer does not run. Everything else in `scripts/` reaches the
# package the same way -- gate.sh exports PYTHONPATH=. -- but this one is documented
# in filters/README.md as a bare command, so it has to stand on its own.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (filename under filters/, FilterManager renderer name, its kwargs).
# One definition: tests/test_reference_filters_are_regenerable.py reads THIS table
# rather than restating it, and tests/test_web_tools_filter.py renders through it.
REFERENCE_FILTERS: tuple[tuple[str, str, dict[str, object]], ...] = (
    (
        "openrouter_web_tools.py",
        "render_openrouter_web_tools_filter_source",
        {"enable_web_search": True, "enable_web_fetch": True, "enable_datetime": True},
    ),
    (
        "openrouter_image_gen.py",
        "render_openrouter_image_gen_filter_source",
        {"dedicated_image_api": True},
    ),
    (
        "openrouter_direct_uploads_toggle.py",
        "render_direct_uploads_filter_source",
        {},
    ),
)


def render_all() -> dict[str, str]:
    """Every reference filter's current content, keyed by filename."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    return {
        filename: getattr(FilterManager, renderer)(**kwargs)
        for filename, renderer, kwargs in REFERENCE_FILTERS
    }


def regenerate(filters_dir: Path) -> list[str]:
    """Write every reference filter whose bytes differ, and report what changed.

    Silent when nothing changed. A step that narrates a no-op every run is a step
    whose output nobody reads, and a rewrite-without-comparing guard would leave the
    narration as the only evidence it did anything at all.
    """
    rendered = render_all()
    changed = []
    for filename, content in rendered.items():
        path = filters_dir / filename
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            continue
        path.write_text(content, encoding="utf-8")
        changed.append(filename)
        print(f"wrote {path}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filters-dir",
        type=Path,
        default=DEFAULT_FILTERS_DIR,
        help="directory holding the reference copies (default: %(default)s)",
    )
    args = parser.parse_args()
    args.filters_dir.mkdir(parents=True, exist_ok=True)
    regenerate(args.filters_dir)


if __name__ == "__main__":
    main()