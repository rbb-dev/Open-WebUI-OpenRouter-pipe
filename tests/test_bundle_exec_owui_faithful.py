"""Exec both bundles via OWUI's loader pattern in a subprocess and check schemas."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

BUNDLES = [
    ("readable", ROOT / "open_webui_openrouter_pipe_bundled.py"),
    ("compressed", ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"),
]

_PROBE = """
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'tests'))
import conftest  # noqa: F401

import types
bundle_path = sys.argv[1]
src = open(bundle_path, 'r', encoding='utf-8').read()
mod_name = 'function_owui_faithful_smoke'
module = types.ModuleType(mod_name)
sys.modules[mod_name] = module
exec(compile(src, bundle_path, 'exec'), module.__dict__)
assert hasattr(module, 'Pipe'), 'Pipe class missing after exec'
Pipe = module.Pipe
assert hasattr(Pipe, 'Valves'), 'Pipe.Valves missing'
assert hasattr(Pipe, 'UserValves'), 'Pipe.UserValves missing'
v = Pipe.Valves.model_json_schema()
assert v.get('properties'), 'Pipe.Valves schema has no properties'
u = Pipe.UserValves.model_json_schema()
assert u.get('properties'), 'Pipe.UserValves schema has no properties'
print(f'OK: valves={len(v[\"properties\"])} user_valves={len(u[\"properties\"])}')
"""


@pytest.mark.parametrize("name,bundle_path", BUNDLES, ids=[b[0] for b in BUNDLES])
def test_bundle_loads_via_owui_faithful_loader(name: str, bundle_path: Path) -> None:
    if not bundle_path.exists():
        pytest.skip(f"{name} bundle not built (run scripts/bundle_v2.py first)")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(bundle_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"{name}: bundle exec failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.strip().startswith("OK:"), (
        f"{name}: probe did not report OK; stdout: {result.stdout!r}"
    )
