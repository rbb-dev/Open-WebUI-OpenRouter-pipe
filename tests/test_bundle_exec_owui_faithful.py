"""Exec both bundles via OWUI's loader pattern in a subprocess and check schemas."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from bundle_probe import probe

from open_webui_openrouter_pipe.models.registry import ModelFamily

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


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name,bundle_path", BUNDLES, ids=[b[0] for b in BUNDLES])
def test_bundle_id_survives_a_per_load_module_name(name: str, bundle_path: Path) -> None:
    """Open WebUI 0.12 execs the artifact as ``function_<id>_<32hex>`` and drops it from
    ``sys.modules`` on return; the id the bundle publishes must still be the admin's."""
    if not bundle_path.exists():
        pytest.skip(f"{name} bundle not built (run scripts/bundle_v2.py first)")
    data = probe(
        bundle_path,
        module_name=f"function_open_webui_openrouter_pipe_{uuid4().hex}",
    )
    assert data["ok"], f"{name}: bundle exec failed -- {data['error']}"
    assert data["pipe_id"] == "open_webui_openrouter_pipe", data["pipe_id"]
    assert (
        ModelFamily._normalize_catalog_id(
            "open_webui_openrouter_pipe.openrouter/gpt-x", pipe_id=data["pipe_id"]
        )
        == "openrouter.gpt-x"
    )


@pytest.mark.timeout(600)
def test_bundle_submodule_alias_survives_a_dropped_module() -> None:
    """0.12's loader pops the module from ``sys.modules`` on return; the PEP 562 fallback
    the readable flat bundle installs for submodule names must not consult the registry
    then. The compressed artifact carries no such shim -- its import hook answers from
    sys.modules directly -- so only the readable one is probed."""
    bundle_path = ROOT / "open_webui_openrouter_pipe_bundled.py"
    if not bundle_path.exists():
        pytest.skip("readable bundle not built (run scripts/bundle_v2.py first)")
    data = probe(
        bundle_path,
        module_name=f"function_open_webui_openrouter_pipe_{uuid4().hex}",
        pop_module=True,
    )
    assert data["ok"], f"bundle exec failed -- {data['error']}"
    assert data["alias_probe_attr"], "no unbound submodule name to probe the alias with"
    assert data["alias_resolves"] is True, data["alias_resolves"]
