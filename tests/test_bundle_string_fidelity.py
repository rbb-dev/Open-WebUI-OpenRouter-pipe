"""A bundled string constant must be byte-identical to the package's.

The bundler rewrites source text: it deletes import statements, prunes orphaned
comments and collapses blank runs. Every one of those operates on lines, and a line
inside a triple-quoted literal looks exactly like a line of code to anything that does
not tokenise. When one of them reaches into a string, the artifact still parses, still
imports, still passes every test that runs against the package -- and ships different
data.

It happened. The orphan-comment pruner deleted ``## Step 3. Classify intent (rules in
priority order; earlier wins)`` out of INTENT_SYSTEM_PROMPT, so the flat bundle -- the
default download -- ran the video-intent classifier on a prompt that says "run these
steps in order" and then skips from 2 to 4. The package and both compressed bundles
were correct, so the same model behaved differently depending on which artifact the
operator installed, and nothing in 5600 tests could see it: tests run against the
package or against a bundle, never comparing the two.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "open_webui_openrouter_pipe"
BUNDLES = (
    "open_webui_openrouter_pipe_bundled.py",
    "open_webui_openrouter_pipe_bundled_no_plugins.py",
    "open_webui_openrouter_pipe_bundled_compressed.py",
    "open_webui_openrouter_pipe_bundled_compressed_no_plugins.py",
)
MIN_LENGTH = 16


def _assign_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


def _artifact_source(path: Path) -> tuple[str, bool]:
    """The artifact's Python, plus whether it carries its modules separately.

    A compressed bundle stores each module as `zlib(base64(...))` in
    `_BUNDLED_SOURCES_Z`, so parsing the file alone finds none of the package's string
    constants and the comparison silently covers nothing.

    The flag decides whether the package's `__init__.py` files are compared: an
    artifact holding each module's own source still has them intact, so their contents
    must match too. A flat bundle synthesises its own head instead and has none of it.
    """
    import base64
    import zlib

    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    parts = [text]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and getattr(target.value, "id", None) == "_BUNDLED_SOURCES_Z"
        ):
            continue
        payload = node.value
        if isinstance(payload, ast.Constant) and isinstance(payload.value, str):
            blob = payload.value
        elif isinstance(payload, ast.JoinedStr):
            continue
        else:
            try:
                blob = ast.literal_eval(payload)
            except Exception:
                continue
        if not isinstance(blob, str):
            continue
        try:
            parts.append(zlib.decompress(base64.b64decode(blob)).decode("utf-8"))
        except Exception:
            continue
    return "\n".join(parts), len(parts) > 1


@lru_cache(maxsize=None)
def _assigned_strings(source: str) -> tuple[tuple[str, str], ...]:
    """(name, value) for every string reachable from an assignment.

    Reachable, not "the value IS a string": the previous version matched only
    `NAME = "..."`, which excluded every f-string. Python models `f"a{x}b"` as a
    JoinedStr whose literal segments are Constants underneath it, so a template built
    with one interpolation had none of its text compared. Walking the value also
    reaches strings inside dict, list and tuple literals. Measured against this
    package, that takes the compared set from 121 constants to 3024, and the multi-line
    ones -- the only shape the bundler's line-based rewrites can damage -- from 37 to
    280.

    Returned as pairs, not a dict: several assignments in one module share a name
    (`msg`, `reason`), and keying by name keeps only the last -- which silently drops
    most values from the comparison on both sides.
    """
    out: list[tuple[str, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        targets = _assign_targets(node)
        value = getattr(node, "value", None)
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if value is None or not names:
            continue
        for sub in ast.walk(value):
            if not isinstance(sub, ast.Constant) or not isinstance(sub.value, str):
                continue
            if len(sub.value) < MIN_LENGTH:
                continue
            for name in names:
                out.append((name, sub.value))
    return tuple(out)


def _package_strings(*, package_machinery: bool = True) -> dict[str, str]:
    """Every compared constant, keyed uniquely.

    `package_machinery=False` drops the `__init__.py` files. Those exist to assemble a
    package -- `__all__`, the lazy-import table, the submodule inventory, the
    distribution name handed to importlib.metadata -- and a flat bundle is not a
    package: `scripts/bundle_v2.py` special-cases every `__init__.py`, synthesises its
    own `__version__` and `__all__` for the artifact head, and carries no lazy-import
    table at all. Their values are absent there by construction, not corrupted.

    `test_no_package_data_hides_behind_the_init_exemption` keeps that exemption from
    growing into a hiding place.
    """
    found: dict[str, str] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.name == "__init__.py" and not package_machinery:
            continue
        for name, text in _assigned_strings(path.read_text(encoding="utf-8")):
            found[f"{path.relative_to(PACKAGE_DIR)}::{name}::{len(found)}"] = text
    return found


@pytest.mark.parametrize("artifact", BUNDLES)
def test_no_bundled_string_constant_differs_from_the_package(artifact):
    """Compared by VALUE, so a renamed or relocated constant still matches."""
    path = PROJECT_ROOT / artifact
    if not path.exists():
        pytest.fail(
            f"{artifact} is not built, so the only comparison between the package and "
            "a shipped artifact ran against nothing. Build the bundles "
            "(python scripts/bundle_v2.py [--compress] [--no-plugins]) before running "
            "this file; skipping made the guarantee conditional on build order."
        )

    source, carries_module_sources = _artifact_source(path)
    pkg = _package_strings(package_machinery=carries_module_sources)
    assert pkg, "no module-level string constants found; this test is comparing nothing"

    # The --no-plugins artifacts omit plugins/ by design, so their constants are
    # legitimately absent rather than corrupted.
    if "no_plugins" in artifact:
        pkg = {k: v for k, v in pkg.items() if not k.startswith("plugins/")}
        assert pkg, "excluding plugins/ left nothing to compare"

    bundled = {v for _, v in _assigned_strings(source)}
    missing = sorted(k for k, v in pkg.items() if v not in bundled)

    assert not missing, (
        f"{artifact} carries a different value for these string constants than the "
        "package does. The bundler rewrote the contents of a literal -- the artifact "
        "parses and imports, so only a comparison like this one can see it:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="reads the source tree to build the expectation; the artifact is compared to it, not the other way round",
)
def test_the_package_still_has_strings_long_enough_to_compare():
    """Anti-vacuity: an empty expectation would make the check above pass on anything."""
    pkg = _package_strings()
    assert len(pkg) >= 800, (
        f"only {len(pkg)} strings of >= {MIN_LENGTH} chars found; the comparison above "
        "is close to asserting nothing. The floor is set above the 121 that the "
        "value-is-a-Constant scan reached, so reverting to it fails here rather than "
        "quietly shrinking the compared set back to a tenth of the package."
    )
    for required in ("INTENT_SYSTEM_PROMPT", "SOCKETIO_UMD_SHA384", "_OPENROUTER_FRONTEND_MODELS_URL"):
        assert any(required in k for k in pkg), (
            f"{required} dropped out of the compared set. The first is the constant the "
            "bundler actually corrupted; the other two are a Subresource Integrity hash "
            "and an API URL, where a single wrong character is worst and both were "
            "excluded while the length floor was 200."
        )
    assert any("INTENT_SYSTEM_PROMPT" in k for k in pkg), (
        "INTENT_SYSTEM_PROMPT is no longer among the compared constants -- it is the "
        "one the bundler actually corrupted, so losing it from this scan removes the "
        "regression test for the bug this file was written for"
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="reads the source tree to build the expectation; the artifact is compared to it, not the other way round",
)
def test_the_compared_set_reaches_inside_f_strings():
    """The scan must reach text an f-string interpolates around.

    A template assembled with one `{value}` is a JoinedStr, and matching only
    `NAME = "..."` skipped all of them -- which is where the filter renderers live, the
    ones whose constants ARE the source of the Open WebUI filters this project ships.
    Counting long constants alone would not notice their return: a thousand ordinary
    literals satisfy any total. This asserts on the f-string mass specifically.
    """
    interpolated = [
        (path, segment)
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and any(isinstance(t, ast.Name) for t in _assign_targets(node))
        and node.value is not None
        for joined in ast.walk(node.value)
        if isinstance(joined, ast.JoinedStr)
        for segment in joined.values
        if isinstance(segment, ast.Constant)
        and isinstance(segment.value, str)
        and len(segment.value) >= MIN_LENGTH
    ]
    assert len(interpolated) >= 80, (
        f"only {len(interpolated)} assigned f-string segments of >= {MIN_LENGTH} chars "
        "exist in the package; if the templates genuinely moved, retune this floor "
        "deliberately rather than letting the guard pass on an empty scan"
    )

    compared = set(_package_strings().values())
    absent = [
        f"{path.relative_to(PACKAGE_DIR)}: {str(segment.value)[:60]!r}"
        for path, segment in interpolated
        if segment.value not in compared
    ]
    assert not absent, (
        f"{len(absent)} f-string segments are not in the compared set, so the bundler "
        "could rewrite them unnoticed:\n  " + "\n  ".join(sorted(absent)[:20])
    )

    multiline = sum(1 for v in compared if "\n" in v)
    assert multiline >= 200, (
        f"only {multiline} multi-line constants are compared. That is the shape the "
        "bundler's line-based rewrites can actually damage, and the scan this file "
        "replaced reached 37 of them; a number back down near that means the walk "
        "stopped descending into values."
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="reads the source tree to build the expectation; the artifact is compared to it, not the other way round",
)
def test_no_package_data_hides_behind_the_init_exemption():
    """The `__init__.py` exemption must stay a description of wiring, not a hiding place.

    Those files are exempt from the flat-bundle comparison because a flat bundle is not
    a package and has none of their contents. The exemption is keyed on the filename,
    so anything moved into an `__init__.py` inherits it silently -- a prompt template
    parked there would stop being compared with nothing to say so.

    What makes the exemption safe is that these files hold only assembly declarations,
    and every value in them is a short symbol name. So this bounds them: nothing
    multi-line, nothing long. Multi-line is the shape the bundler's line-based rewrites
    can damage in the first place, which is the whole subject of this file.
    """
    exempt: list[tuple[str, str, str]] = []
    for path in sorted(PACKAGE_DIR.rglob("__init__.py")):
        for name, text in _assigned_strings(path.read_text(encoding="utf-8")):
            exempt.append((str(path.relative_to(PACKAGE_DIR)), name, text))

    assert exempt, (
        "no constants were found in any __init__.py. Either the package stopped "
        "assembling itself there, or the scan stopped seeing them -- and in the second "
        "case this guard is watching nothing."
    )

    oversized = sorted(
        f"{where}::{name} ({len(text)} chars)"
        for where, name, text in exempt
        if "\n" in text or len(text) > 120
    )
    assert not oversized, (
        "these __init__.py constants are too big to be package wiring, so real data is "
        "sitting in a file the flat-bundle comparison skips and would ship unchecked. "
        "Move them into a module:\n  " + "\n  ".join(oversized)
    )

    everything = _package_strings()
    assert len(exempt) < len(everything) // 4, (
        f"{len(exempt)} of {len(everything)} compared constants live in an __init__.py. "
        "The exemption is meant to cover package wiring, not a quarter of the package."
    )
