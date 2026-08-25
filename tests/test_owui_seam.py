"""Static drift guard for the Open WebUI import seam.

Every symbol this pipe imports from ``open_webui.*`` must still exist in the
INSTALLED Open WebUI. This test derives that checklist by AST-scanning our own
source (so it is self-maintaining: a new ``from open_webui... import X`` is
covered automatically) and resolves each symbol by parsing Open WebUI's source
files off disk.

It NEVER imports or executes Open WebUI, which makes it:
  * fast (~0.3s) and offline — no torch/transformers/retrieval import;
  * stub-proof — the rest of the suite replaces ``open_webui`` with fakes in
    ``sys.modules`` (conftest), but this reads the real package's source files,
    located via installed distribution metadata, not ``sys.modules``.

A failure means Open WebUI renamed/moved/removed a symbol we depend on — caught
at development time, before it ships and silently breaks in production.
"""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

_OUR_PKG = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"


def _is_owui(module: str) -> bool:
    """True for real Open WebUI modules, excluding our own package."""
    return module == "open_webui" or module.startswith("open_webui.")


def _owui_base() -> Path | None:
    """Directory of the installed ``open_webui`` package (via metadata, no import)."""
    try:
        dist = importlib.metadata.distribution("open-webui")
        init = Path(str(dist.locate_file("open_webui/__init__.py")))
    except Exception:
        return None
    return init.parent if init.is_file() else None


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    exc = handler.type
    if exc is None:
        return True  # bare except
    candidates = exc.elts if isinstance(exc, ast.Tuple) else [exc]
    return any(
        isinstance(c, ast.Name) and c.id in ("ImportError", "ModuleNotFoundError", "Exception")
        for c in candidates
    )


_UNPARSED: list[str] = []


def _our_owui_imports() -> list[tuple[str, str | None, bool]]:
    """Every (module, symbol, is_optional) imported from open_webui across our tree.

    ``symbol`` is None for a bare ``import open_webui.x``. ``is_optional`` is True
    when the import sits inside a ``try/except ImportError`` (graceful degradation).
    """
    found: set[tuple[str, str | None, bool]] = set()

    def walk(nodes: list[ast.stmt], optional: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module and _is_owui(node.module):
                for alias in node.names:
                    if alias.name != "*":
                        found.add((node.module, alias.name, optional))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_owui(alias.name):
                        found.add((alias.name, None, optional))
            # Recurse into compound statements, tracking try/except-ImportError scope.
            if isinstance(node, ast.Try):
                inner = optional or any(_catches_import_error(h) for h in node.handlers)
                walk(node.body, inner)
                for h in node.handlers:
                    walk(h.body, optional)
                walk(node.orelse, optional)
                walk(node.finalbody, optional)
            elif isinstance(node, ast.If):
                walk(node.body, optional)
                walk(node.orelse, optional)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                walk(node.body, optional)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                walk(node.body, optional)
                walk(node.orelse, optional)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(node.body, optional)

    for py_file in _OUR_PKG.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            _UNPARSED.append(py_file.name)
            continue
        walk(tree.body, False)

    return sorted(found, key=lambda t: (t[0], t[1] or ""))


def _module_file(base: Path, module: str) -> Path | None:
    rel = "" if module == "open_webui" else module[len("open_webui.") :]
    parts = rel.split(".") if rel else []
    candidate_py = base.joinpath(*parts).with_suffix(".py") if parts else None
    candidate_init = base.joinpath(*parts, "__init__.py")
    if candidate_py is not None and candidate_py.is_file():
        return candidate_py
    if candidate_init.is_file():
        return candidate_init
    return None


def _toplevel_names(path: Path) -> set[str]:
    """Names bound at import time in ``path`` (defs/classes/assignments/imports),
    recursing through control flow but NOT into def/class bodies."""
    names: set[str] = set()

    def targets(node: ast.expr) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [name for elt in node.elts for name in targets(elt)]
        return []

    def walk(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    names.update(targets(tgt))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.If):
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for h in node.handlers:
                    walk(h.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                walk(node.body)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                walk(node.body)
                walk(node.orelse)

    try:
        walk(ast.parse(path.read_text(encoding="utf-8")).body)
    except SyntaxError:
        return names
    return names


def _resolves(base: Path, module: str, symbol: str | None) -> bool:
    if symbol is None:  # bare `import open_webui.x` — the module itself must exist
        rel = "" if module == "open_webui" else module[len("open_webui.") :]
        return _module_file(base, module) is not None or base.joinpath(*rel.split(".")).is_dir()
    src = _module_file(base, module)
    if src is not None and symbol in _toplevel_names(src):
        return True
    if _module_file(base, f"{module}.{symbol}") is not None:
        return True  # symbol is a submodule
    rel = "" if module == "open_webui" else module[len("open_webui.") :]
    parts = rel.split(".") if rel else []
    return base.joinpath(*parts, symbol).is_dir()  # namespace-package submodule


_BASE = _owui_base()
_IMPORTS = _our_owui_imports()

_NEWER_THAN_OUR_FLOOR: dict[tuple[str, str], str] = {
    ("open_webui.utils.chat_id", "NON_SAVED_CHAT_ID_PREFIXES"): "0.11.0",
}


@pytest.mark.skipif(_BASE is None, reason="open_webui is not installed")
@pytest.mark.parametrize(
    "module,symbol,is_optional",
    _IMPORTS,
    ids=[f"{m}.{s}" if s else f"{m}(module)" for m, s, _ in _IMPORTS],
)
def test_owui_import_seam_resolves(module: str, symbol: str | None, is_optional: bool) -> None:
    """`is_optional` decides what a non-resolving symbol means, rather than only how it reads.

    It was computed and then used for nothing but the message, so every import was held
    to the REQUIRED rule. That forbids the one pattern a guarded import exists for:
    preferring an API from a NEWER Open WebUI than the manifest floor, and falling back
    on the versions that predate it. Blocking that pushes the code back onto a hand-copy
    of upstream's value, which is what drifts.

    Absence still has to be declared. An unresolved optional import that nobody listed
    is indistinguishable from one Open WebUI deleted underneath us, which is exactly the
    drift this file exists to catch.
    """
    assert _BASE is not None
    target = module if symbol is None else f"{module}.{symbol}"
    if _resolves(_BASE, module, symbol):
        return
    assert is_optional, (
        f"OWUI seam drift: `{target}` (REQUIRED import) no longer resolves in the "
        f"installed open_webui at {_BASE}. Open WebUI renamed/moved/removed it — "
        f"update our import before this ships."
    )
    assert symbol is not None and (module, symbol) in _NEWER_THAN_OUR_FLOOR, (
        f"OWUI seam drift: `{target}` is imported under a guard and does not resolve in "
        f"the installed open_webui at {_BASE}, so its fallback is what runs — "
        "permanently, and silently. If Open WebUI removed it, drop our import. If it "
        "predates our floor, add it to _NEWER_THAN_OUR_FLOOR with the version that "
        "introduces it."
    )


def _version_parts(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _installed_owui_version() -> str | None:
    try:
        return importlib.metadata.version("open-webui")
    except Exception:
        return None


_MANIFEST = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe.py"


def _manifest_floor() -> str | None:
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("required_open_webui_version:"):
            return line.split(":", 1)[1].strip()
    return None


def test_manifest_floor_is_a_version_the_updater_can_parse() -> None:
    """The shipped floor is an artifact promise, so check the artifact -- not this box.

    `_version_parts` scrapes digits and defaults a non-numeric chunk to 0, so it reads
    `0.9.x` as (0, 9, 0) and every check that consults it stays green. The self-updater
    parses this same literal with `packaging.version.Version` and refuses the whole
    release when it does not parse, so an unparseable floor ships a bundle that no host
    can install and no host can roll back through.

    Deliberately NOT asserted: that the floor equals some value, or that it is at or
    below the locally installed Open WebUI. Both are properties of a developer's machine
    rather than of the artifact, and raising the floor is a normal, intended edit.
    """
    floor = _manifest_floor()
    assert floor, f"no `required_open_webui_version:` in {_MANIFEST}"
    try:
        Version(floor)
    except InvalidVersion:
        pytest.fail(
            f"`required_open_webui_version: {floor}` in {_MANIFEST} is not a version "
            "`packaging` can parse. The self-updater validates this literal before the "
            "compatibility gate and refuses the release outright, so shipping it would "
            "make the bundle uninstallable everywhere. Write a bare version, not a "
            "range or a wildcard."
        )


@pytest.mark.skipif(_BASE is None, reason="open_webui is not installed")
@pytest.mark.parametrize(("module", "symbol"), sorted(_NEWER_THAN_OUR_FLOOR))
def test_a_declared_absence_matches_the_installed_version(module: str, symbol: str) -> None:
    """The entry records WHICH version introduces the symbol, so check against that.

    Presence alone cannot decide. This entry is correct on 0.10.x, where the symbol is
    absent, AND on 0.11.x, where it is present -- the guard exists precisely because the
    manifest floor spans both. Asserting bare absence turned the supported upgrade into a
    red build the day Open WebUI shipped the symbol, and deleting the entry to make that
    green breaks the other direction: `test_owui_import_seam_resolves` then fails on every
    install below the introducing version, which is most of the supported range.

    Both directions are still checked, against the recorded version rather than against
    nothing: below it the symbol must be absent, from it the symbol must be present.
    """
    assert _BASE is not None
    introduced = _NEWER_THAN_OUR_FLOOR[(module, symbol)]
    installed = _installed_owui_version()
    assert installed is not None, (
        "open_webui resolved on disk but its distribution reports no version, so the "
        "declaration cannot be checked against anything"
    )
    resolves = _resolves(_BASE, module, symbol)
    if _version_parts(installed) >= _version_parts(introduced):
        assert resolves, (
            f"`{module}.{symbol}` is recorded as arriving in Open WebUI {introduced} and "
            f"{installed} is installed, but it does not resolve at {_BASE}. Either the "
            "recorded version is wrong, or Open WebUI removed it again and our guard now "
            "runs its fallback permanently."
        )
    else:
        assert not resolves, (
            f"`{module}.{symbol}` resolves in the installed open_webui {installed} at "
            f"{_BASE}, but the entry records it as arriving in {introduced}. The recorded "
            "version is wrong, and the entry is exempting a symbol that is present."
        )


def test_no_declared_absence_predates_the_manifest_floor() -> None:
    """The entry dies when the floor catches up, not when the developer's install does.

    An entry whose version is at or below `required_open_webui_version` describes nothing:
    every supported host has the symbol, so the guard's fallback is unreachable and both
    the guard and the entry should go. That is the only condition under which the entry is
    stale -- an installed Open WebUI being newer than the entry is the normal case.
    """
    floor = _manifest_floor()
    assert floor, f"no `required_open_webui_version:` in {_MANIFEST}"
    stale = sorted(
        (m, s)
        for (m, s), introduced in _NEWER_THAN_OUR_FLOOR.items()
        if _version_parts(introduced) <= _version_parts(floor)
    )
    assert not stale, (
        f"{stale} are recorded as arriving at or below the manifest floor {floor}, so "
        "every supported Open WebUI has them. Drop the guard and the entry."
    )


def test_every_declared_absence_is_actually_imported_under_a_guard() -> None:
    """The declaration cannot outlive the import it excuses."""
    guarded = {(m, sym) for m, sym, optional in _IMPORTS if optional and sym is not None}
    orphans = sorted(set(_NEWER_THAN_OUR_FLOOR) - guarded)
    assert not orphans, (
        f"_NEWER_THAN_OUR_FLOOR lists {orphans}, which this package no longer imports "
        "under a guard. Remove the entries."
    )


_EXPECTED_SEAM_IMPORTS = 64


def test_seam_checklist_covers_every_open_webui_import() -> None:
    """Pinned exactly, not floored.

    The floor was 30 against 61 real imports, so half the seam could stop being checked
    with the guard still green -- dropping the `plugins/` subtree alone takes it to 31,
    which passes while `update_service.py`'s fifteen Open WebUI imports go unverified.
    An exact count makes a change in either direction a deliberate edit.
    """
    assert not _UNPARSED, (
        "these files did not parse, so their Open WebUI imports are absent from the "
        f"checklist rather than checked: {_UNPARSED}"
    )
    assert len(_IMPORTS) == _EXPECTED_SEAM_IMPORTS, (
        f"the package now has {len(_IMPORTS)} open_webui imports, not "
        f"{_EXPECTED_SEAM_IMPORTS}. Every one of them is drift-checked against the "
        "installed Open WebUI, so update the number deliberately after confirming the "
        "new ones resolve."
    )
