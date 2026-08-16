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


@pytest.mark.skipif(_BASE is None, reason="open_webui is not installed")
@pytest.mark.parametrize(("module", "symbol"), sorted(_NEWER_THAN_OUR_FLOOR))
def test_a_declared_absence_is_still_absent(module: str, symbol: str) -> None:
    """A stale declaration is a hole, so the list is checked in both directions.

    Once the installed Open WebUI catches up, the entry stops describing anything and
    starts exempting a real symbol from drift detection.
    """
    assert _BASE is not None
    assert not _resolves(_BASE, module, symbol), (
        f"`{module}.{symbol}` now resolves in the installed open_webui at {_BASE}, so "
        "its _NEWER_THAN_OUR_FLOOR entry is stale and is exempting a symbol that is "
        "present. Remove the entry."
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
