"""A repo-built bundle must carry the anyio #1111 workaround.

The workaround patches a cancellation spin that pegs a core at 100% when a request is
aborted, so a bundle that quietly ships without it degrades under exactly the traffic
that already hurts. The bundler cannot enforce this by refusing to build: the gh-pages
in-browser builder deliberately makes the workaround an opt-in checkbox and fetches its
source only when ticked, so raising on absence breaks the public builder's default path.

The guarantee belongs here instead, where the source is always present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLER = PROJECT_ROOT / "scripts" / "bundle_v2.py"

ARTIFACTS = [
    "open_webui_openrouter_pipe_bundled.py",
    "open_webui_openrouter_pipe_bundled_compressed.py",
    "open_webui_openrouter_pipe_bundled_no_plugins.py",
    "open_webui_openrouter_pipe_bundled_compressed_no_plugins.py",
]


def _bundler() -> Any:
    spec = importlib.util.spec_from_file_location("_bundle_v2_under_test", BUNDLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_workaround_source_is_present_in_the_repo():
    bundler = _bundler()
    assert bundler.ANYIO_WORKAROUND_FILE.exists(), (
        f"{bundler.ANYIO_WORKAROUND_FILE} is missing. Every bundle built from this "
        "checkout would omit the cancellation-spin workaround."
    )
    assert bundler.ANYIO_WORKAROUND_MARKER in bundler._load_anyio_workaround_block()


def test_absent_source_degrades_instead_of_failing_the_build(tmp_path, capsys):
    """The gh-pages builder's default path must still produce a bundle."""
    bundler = _bundler()
    bundler.ANYIO_WORKAROUND_FILE = tmp_path / "not-fetched.py"

    block = bundler._load_anyio_workaround_block()

    assert block == ""
    assert "without the anyio #1111 workaround" in capsys.readouterr().err


@pytest.mark.parametrize("compressed", [False, True])
def test_a_freshly_built_bundle_carries_the_workaround(tmp_path, compressed):
    """Builds from source, so it holds even in a checkout with no artifacts yet."""
    bundler = _bundler()
    output = tmp_path / "built.py"

    bundler.bundle(output_path=output, compressed=compressed)

    text = output.read_text(encoding="utf-8", errors="replace")
    marker = bundler.ANYIO_WORKAROUND_MARKER
    shape = "compressed" if compressed else "flat"

    assert f"def {marker}(" in text, f"{shape} bundle does not define {marker}()"
    call_lines = [
        ln for ln in text.splitlines() if ln.strip() == f"{marker}()"
    ]
    assert any(not ln.startswith(" ") for ln in call_lines), (
        f"{shape} bundle defines {marker}() but never calls it unconditionally at "
        "module level, so the cancellation-spin workaround is never applied"
    )


def test_every_built_artifact_carries_the_workaround():
    """Covers the artifacts CI actually publishes, whichever of them exist.

    Artifacts are gitignored, so a fresh checkout has none and this checks nothing --
    which is why the guarantee lives in the build-from-source test above. This one
    catches a stale or hand-edited artifact sitting in the tree, and says how many it
    actually inspected so a silent zero is visible rather than passing as a green tick.
    """
    marker = _bundler().ANYIO_WORKAROUND_MARKER
    checked, missing = [], []
    for name in ARTIFACTS:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        checked.append(name)
        text = path.read_text(encoding="utf-8", errors="replace")
        if f"def {marker}(" not in text or not any(
            ln.strip() == f"{marker}()" and not ln.startswith((" ", "\t", "#"))
            for ln in text.splitlines()
        ):
            missing.append(name)
    assert not missing, f"built artifacts missing {marker}(): {missing}"
    if not checked:
        pytest.skip(
            "no bundles built; the guarantee is covered by the build-from-source test"
        )


def test_the_version_gate_applies_the_workaround_below_the_fixed_version():
    """Presence in the bundle is not applicability.

    Inverting the version comparison makes the workaround never apply while every
    presence check still passes -- the block ships, and the cancellation spin it
    exists to prevent comes straight back.
    """
    import ast

    source = (PROJECT_ROOT / "scripts" / "anyio_1111_workaround.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    fixed_in = next(
        (
            n.value.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_FIXED_IN" for t in n.targets)
            and isinstance(n.value, ast.Constant)
        ),
        None,
    )
    assert fixed_in, "_FIXED_IN is gone; this guard is stale"

    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_before_fix"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {}
    exec(compile(module, "<probe>", "exec"), ns)
    before_fix = ns["_before_fix"]

    assert before_fix("4.12.1", fixed_in), (
        f"anyio 4.12.1 is below {fixed_in} but the gate says the workaround is "
        "unnecessary -- it would never apply"
    )
    assert not before_fix(fixed_in, fixed_in), "the fixed version must not be patched"
    assert not before_fix("5.0.0", fixed_in), "a later version must not be patched"


_GATE_PROBE = r'''
import importlib.metadata, importlib.util, sys
importlib.metadata.version = lambda name: FAKE_VERSION if name == "anyio" else "0"
spec = importlib.util.spec_from_file_location("_anyio_probe", WORKAROUND)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from anyio._backends._asyncio import CancelScope
patched = getattr(CancelScope, "_deliver_cancellation", None)
print("APPLIED" if getattr(patched, "_anyio_1111_workaround_applied", False) else "SKIPPED")
'''


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("4.12.1", "APPLIED"),
        ("4.14.1", "APPLIED"),
        ("4.14.2", "SKIPPED"),
        ("4.20.0", "SKIPPED"),
        ("5.0.0", "SKIPPED"),
    ],
)
def test_the_gate_decides_whether_cancelscope_is_actually_patched(installed, expected):
    """Run the real gate and look at CancelScope, not at the comparator.

    The sibling test execs `_before_fix` standalone, so inverting its *call site*
    (`if not _before_fix(...)` -> `if _before_fix(...)`) leaves it green while the
    workaround applies only to anyio versions that already contain the fix -- exactly
    the outcome the block exists to prevent.

    Out of process because the function returns early whenever pytest is imported.
    Importing the module *is* the exercise: it self-invokes and then deletes the
    symbol, which is exactly what the bundle head does.
    """
    import subprocess

    workaround = PROJECT_ROOT / "scripts" / "anyio_1111_workaround.py"
    probe = (
        f"FAKE_VERSION = {installed!r}\nWORKAROUND = {str(workaround)!r}\n" + _GATE_PROBE
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().endswith(expected), (
        f"anyio {installed} should be {expected}, got {result.stdout.strip()!r}. "
        "The version gate is not deciding what actually happens to CancelScope."
    )
