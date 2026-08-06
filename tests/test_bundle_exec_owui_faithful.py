"""Exec both bundles via OWUI's loader pattern in a subprocess and check schemas."""
from __future__ import annotations

from pathlib import Path

import pytest
from bundle_probe import probe

ROOT = Path(__file__).parent.parent

BUNDLES = [
    ("readable", ROOT / "open_webui_openrouter_pipe_bundled.py"),
    ("compressed", ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"),
]


# Whichever consumer of bundle_probe runs first pays the load; the rest read the
# cache. A budget on only one of them makes the suite's verdict depend on
# collection order, and this file sorts first.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("name,bundle_path", BUNDLES, ids=[b[0] for b in BUNDLES])
def test_bundle_loads_via_owui_faithful_loader(name: str, bundle_path: Path) -> None:
    if not bundle_path.exists():
        pytest.skip(f"{name} bundle not built (run scripts/bundle_v2.py first)")
    data = probe(bundle_path)
    assert data["ok"], f"{name}: bundle exec failed -- {data['error']}"
    assert data["schema"]["valves"] > 0, (
        f"{name}: Pipe.Valves schema has no properties, so the artifact exposes no "
        "configuration to Open WebUI"
    )
    assert data["schema"]["user_valves"] > 0, (
        f"{name}: Pipe.UserValves schema has no properties, so per-user settings are "
        "absent from the artifact"
    )
