"""Every external name the bundler dedups must resolve to the same object.

A bundle is one flat namespace, so when two third-party modules export the same name
the shared header can bind only one of them. The dedup is unavoidable; what makes it
dangerous is that the loser vanishes without a NameError. ``Request`` is dropped from
``starlette.requests`` today and the bundle is correct only because
``fastapi.Request is starlette.requests.Request``. The next such pair need not be.

The bundler itself only reports these, because proving identity requires importing the
dependency and the gh-pages in-browser builder has none installed. This is where the
dependencies exist, so this is where it is proven.
"""

from __future__ import annotations

from functools import lru_cache

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLER = PROJECT_ROOT / "scripts" / "bundle_v2.py"


def _bundler() -> Any:
    spec = importlib.util.spec_from_file_location("_bundle_v2_collisions", BUNDLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _resolve(origin: str, name: str) -> tuple[bool, Any]:
    """Resolve what a header line actually binds.

    ``import x.y as z`` / ``import x`` bind a module object; ``from x import y`` binds
    an attribute of one. Resolving both as attributes makes a real mismatch --
    ``from datetime import datetime`` versus ``import datetime`` -- look identical.
    """
    try:
        if origin.startswith("import "):
            token = origin[len("import ") :].strip()
            module_path = token.split(" as ")[0].strip()
            module = importlib.import_module(module_path)
            if " as " in token:
                return True, module
            return True, importlib.import_module(module_path.split(".")[0])
        head, _, member = origin.partition(" import ")
        module_path = head[len("from ") :].strip()
        member_name = member.split(" as ")[0].strip() or name
        return True, getattr(importlib.import_module(module_path), member_name)
    except Exception:
        return False, None


def test_deduped_header_names_are_the_same_object(tmp_path):
    bundler = _bundler()
    bundler.bundle(output_path=tmp_path / "built.py", compressed=False)

    unproven: list[str] = []
    for name, kept, dropped in bundler.NAME_COLLISIONS:
        kept_ok, kept_obj = _resolve(kept, name)
        dropped_ok, dropped_obj = _resolve(dropped, name)
        if not (kept_ok and dropped_ok):
            unproven.append(f"{name}: could not import {kept} or {dropped} to compare")
        elif kept_obj is not dropped_obj:
            unproven.append(
                f"{name}: {kept}.{name} is not {dropped}.{name} -- the bundle binds "
                f"the {kept} one and silently changes behaviour for every call site "
                f"that meant the {dropped} one. Alias one of them at the import."
            )

    assert not unproven, "unsafe header name collisions:\n  " + "\n  ".join(unproven)


def test_the_known_collision_is_still_detected(tmp_path):
    """Guards the detector itself: if this stops firing, the check has gone blind."""
    bundler = _bundler()
    bundler.bundle(output_path=tmp_path / "built.py", compressed=False)

    assert any(
        name == "Request" and "fastapi" in kept and "starlette.requests" in dropped
        for name, kept, dropped in bundler.NAME_COLLISIONS
    ), (
        "the fastapi/starlette Request collision is no longer reported. Either the "
        "source stopped importing both (fine -- drop this test) or collision "
        "detection broke (not fine)."
    )


def test_annotated_module_globals_are_visible_to_the_collision_report(tmp_path):
    """`x: T = ...` binds a module global exactly as `x = ...` does.

    Omitting ast.AnnAssign from the bundler's top-level-name collection made two
    modules' identically-named globals invisible: `_get_pipe: Any = None` exists in
    both dashboard_socket and http_routes, the flat bundle contained two copies merged
    into one variable, and the build reported no collision at all.

    Calls analyze_module directly rather than scanning its source. An earlier version
    asserted `"ast.AnnAssign" in inspect.getsource(...)`, which a comment mentioning
    the name was enough to satisfy.
    """
    import ast

    bundler = _bundler()

    src = "import typing\n_annotated: typing.Any = None\n_plain = 1\n"
    module = bundler.ModuleInfo(
        dotted_name="probe",
        file_path=tmp_path / "probe.py",
        raw_source=src,
        source_lines=src.splitlines(keepends=True),
        tree=ast.parse(src),
    )
    bundler.analyze_module(module, {"probe": module})

    assert "_plain" in module.top_level_names, "plain assignment collection broke"
    assert "_annotated" in module.top_level_names, (
        "annotated module globals are not collected, so two modules defining the same "
        "annotated name merge silently in the flat bundle with no collision reported"
    )


def test_no_annotated_global_is_defined_by_two_modules():
    """`x: T = ...` in two modules is one slot in the flat bundle.

    ``_get_pipe: Any = None`` was declared by both dashboard_socket and http_routes;
    the package keeps two namespaces, the bundle keeps one, so ``set_pipe_getter`` and
    ``register_socket_handler`` wrote to the same variable only in the shipped artifact.
    Renaming one fixed it. This fails the moment the shape comes back.

    Independent of the build gate on purpose: that gate only fires while
    ``_is_idempotent_definition`` keeps rejecting ``= None``, and this does not care.
    """
    duplicated = _package_annotated_collisions()
    assert not duplicated, (
        f"{sorted(duplicated)} are annotated module globals defined by more than one "
        "module. The flat bundle collapses them into a single variable, so whichever "
        "module's body runs last resets the other's -- with no NameError, and only in "
        "the bundle. Rename one."
    )


def test_annotated_globals_reach_the_collision_detector(tmp_path):
    """The detector must see annotated globals, or the check above cannot fire.

    An earlier version asserted `seen or reported`, and `reported` always contains the
    fastapi/starlette `Request` pair -- so the assertion could never fail.
    """
    import ast

    bundler = _bundler()
    src = "import typing\n_dup: typing.Any = None\n"
    mods = {}
    for short in ("alpha", "beta"):
        mod = bundler.ModuleInfo(
            dotted_name=short,
            file_path=tmp_path / f"{short}.py",
            raw_source=src,
            source_lines=src.splitlines(keepends=True),
            tree=ast.parse(src),
        )
        bundler.analyze_module(mod, {short: mod})
        mods[short] = mod

    assert all("_dup" in m.top_level_names for m in mods.values()), (
        "annotated globals are invisible to the bundler, so two modules declaring the "
        "same one merge silently in the flat bundle with no collision reported"
    )
    assert not bundler._definitions_textually_identical("_dup", ["alpha", "beta"], mods), (
        "`_dup: Any = None` was treated as a benign identical redefinition. It is not: "
        "the second module's body re-runs the assignment and resets whatever the first "
        "one was holding."
    )


@lru_cache(maxsize=None)
def _package_annotated_collisions() -> frozenset[str]:
    import ast
    from collections import Counter

    pkg = PROJECT_ROOT / "open_webui_openrouter_pipe"
    counts: Counter[str] = Counter()
    for path in pkg.rglob("*.py"):
        for node in ast.iter_child_nodes(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                counts[node.target.id] += 1
    return frozenset(n for n, c in counts.items() if c > 1)


def test_internal_collision_fails_the_build(tmp_path, monkeypatch):
    """A non-identical internal collision must abort, not print.

    The flat bundle collapses every module into one namespace, so two modules that
    each define ``x`` leave one definition standing and silently redirect every call
    site that meant the other. It only manifests in the shipped artifact -- the
    package the tests import keeps the two namespaces apart -- so a warning on stderr
    is read by nobody and shipped by everybody.
    """
    bundler = _bundler()
    monkeypatch.setattr(bundler, "_definitions_textually_identical", lambda *a, **k: False)

    with pytest.raises(RuntimeError) as excinfo:
        bundler.bundle(output_path=tmp_path / "built.py", compressed=False)

    assert "defined by more than one module" in str(excinfo.value)


def test_the_benign_exemption_is_not_vacuous(tmp_path):
    """The build passes because the collisions are provably identical, not because
    the detector found none. If nothing is ever exempted the exemption is dead code
    and the previous test proves nothing about the real build."""
    bundler = _bundler()
    bundler.bundle(output_path=tmp_path / "built.py", compressed=False)

    duplicated = {
        name: origins
        for name, origins in _module_name_origins(bundler).items()
        if len(origins) > 1
    }
    assert duplicated, (
        "no top-level name is defined by two modules any more -- the benign path is "
        "never exercised, so test_internal_collision_fails_the_build only proves the "
        "raise is syntactically reachable. Re-anchor this test or drop both."
    )
    mods = _mods_by_short(bundler)
    unexempted = [
        name
        for name, origins in duplicated.items()
        if not (
            bundler._definitions_textually_identical(
                name, [o.removeprefix(bundler.PACKAGE_NAME + ".") for o in origins], mods
            )
            or bundler._bindings_are_equivalent(
                name, [o.removeprefix(bundler.PACKAGE_NAME + ".") for o in origins], mods
            )
        )
    ]
    assert not unexempted, (
        f"the build succeeded but {unexempted} are not provably identical -- the gate "
        "is not running over the same set this test inspects"
    )

    by_binding = [
        name
        for name, origins in duplicated.items()
        if bundler._bindings_are_equivalent(
            name, [o.removeprefix(bundler.PACKAGE_NAME + ".") for o in origins], mods
        )
    ]
    assert by_binding, (
        "no duplicate name is exempted by binding equivalence, so that path is dead "
        "code here. It exists for the optional-import guards -- if those are all gone, "
        "drop the path rather than leaving an untested exemption in the gate."
    )


def _module_name_origins(bundler) -> dict[str, list[str]]:
    from collections import defaultdict

    origins: dict[str, list[str]] = defaultdict(list)
    for short, mod in _mods_by_short(bundler).items():
        for name in mod.top_level_names:
            origins[name].append(short)
    return origins


def _mods_by_short(bundler) -> dict[str, Any]:
    all_modules = bundler.discover_modules(bundler.PACKAGE_DIR, exclude_plugins=False)
    for mod in all_modules.values():
        bundler.analyze_module(mod, all_modules)
    return {
        mod.dotted_name.removeprefix(bundler.PACKAGE_NAME + "."): mod
        for mod in all_modules.values()
        if not mod.is_init
    }


def test_the_hoisted_import_guard_rejects_a_dropped_import():
    """Replacing its body with `pass` breaks nothing else.

    The bundler strips each module's top-level external imports and re-emits them in a
    shared header. A statement lost in between still parses, still imports, and raises
    NameError only when the affected path first runs -- which is why the build asserts
    it, and why the assertion needs its own guard.
    """
    bundler = _bundler()
    mod = bundler.ModuleInfo(
        dotted_name="probe",
        file_path=Path("probe.py"),
        raw_source="",
        source_lines=[],
        tree=None,
    )
    mod.external_import_lines = ["import base64", "from decimal import Decimal"]

    bundler._assert_hoisted_imports_rebound([mod], ["import base64", "from decimal import Decimal"])

    with pytest.raises(RuntimeError) as excinfo:
        bundler._assert_hoisted_imports_rebound([mod], ["import base64"])
    assert "Decimal" in str(excinfo.value)

    mod.external_import_lines = ["import zlib as _z"]
    with pytest.raises(RuntimeError):
        bundler._assert_hoisted_imports_rebound([mod], ["import zlib"])
    bundler._assert_hoisted_imports_rebound([mod], ["import zlib as _z"])


def test_a_module_that_does_not_parse_fails_the_compressed_build(tmp_path):
    """The compressed bundle embeds each module as a blob and imports it lazily.

    Without the parse gate a syntactically broken module ships, and the artifact only
    breaks when something first imports that module -- at runtime, in production, on
    whichever code path happens to touch it.
    """
    bundler = _bundler()
    pkg = tmp_path / "open_webui_openrouter_pipe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "broken.py").write_text("def f(:\n", encoding="utf-8")

    with pytest.raises(SyntaxError) as excinfo:
        bundler._collect_all_modules(pkg, exclude_plugins=False)
    assert "does not parse" in str(excinfo.value)

    (pkg / "broken.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    collected = bundler._collect_all_modules(pkg, exclude_plugins=False)
    assert any(name.endswith(".broken") for name in collected), (
        "the fixed module is not collected, so the gate above proves nothing"
    )


def test_trailing_comments_are_reported_from_any_physical_line():
    """A dropped `# type: ignore` on a hoisted import silently changes type checking.

    The bundle header cannot carry them, so the build reports them instead. Returning
    "" makes that report empty and the loss invisible.
    """
    bundler = _bundler()
    lines = [
        "from PIL import (\n",
        "    Image,  # type: ignore[import-untyped]\n",
        ")\n",
    ]
    found = bundler._trailing_comment(lines, 1, 3)
    assert "type: ignore" in found, (
        "a comment on a continuation line is not seen, so a multi-line hoisted import "
        f"loses it without a word; got {found!r}"
    )

    assert bundler._trailing_comment(["import os\n"], 1, 1) == "", (
        "an import with no comment reports one, so every build would print noise and "
        "the real losses would be unreadable"
    )


def test_a_bare_import_colliding_with_a_from_import_is_recorded():
    """`from datetime import datetime` + `import datetime` both reach the header.

    Unlike the from-import case -- where only one of the two names survives the dedup
    -- both lines are emitted, so the later one silently rebinds the name to a
    different object. There is no NameError, and `datetime.now()` becomes an
    AttributeError on the module far from where the two imports met.
    """
    bundler = _bundler()
    bundler.NAME_COLLISIONS.clear()

    def _module(name: str, imports: list[str]):
        mod = bundler.ModuleInfo(
            dotted_name=name,
            file_path=Path(f"{name}.py"),
            raw_source="",
            source_lines=[],
            tree=None,
        )
        mod.external_import_lines = imports
        return mod

    bundler.collect_and_dedup_external_imports(
        [_module("a", ["from datetime import datetime"]), _module("b", ["import datetime"])]
    )
    recorded = {name for name, _, _ in bundler.NAME_COLLISIONS}
    assert "datetime" in recorded, (
        "a bare import that rebinds a from-imported name is not recorded, so the "
        f"header ships both lines with no report at all. Recorded: {recorded}"
    )

    # A directive comment must not hide the collision. The token the local name is
    # derived from used to carry the trailing `# type: ignore`, so `bare_local` read
    # "datetime  # type: ignore[...]" and matched nothing -- the header emitted both
    # lines and the report stayed empty. imageio is imported exactly this way.
    bundler.NAME_COLLISIONS.clear()
    bundler.collect_and_dedup_external_imports(
        [
            _module("a", ["from datetime import datetime"]),
            _module("b", ["import datetime  # type: ignore[import-untyped]"]),
        ]
    )
    recorded = {name for name, _, _ in bundler.NAME_COLLISIONS}
    assert "datetime" in recorded, (
        "a bare import carrying a tool directive is not recorded as a collision, so the "
        f"directive comment silently disables the check. Recorded: {recorded}"
    )

    bundler.NAME_COLLISIONS.clear()
    bundler.collect_and_dedup_external_imports(
        [_module("a", ["from json import dumps as jsonlib"]), _module("b", ["import zlib as jsonlib"])]
    )
    assert "jsonlib" in {name for name, _, _ in bundler.NAME_COLLISIONS}, (
        "an aliased bare import colliding with an aliased from-import is not recorded"
    )

    bundler.NAME_COLLISIONS.clear()
    bundler.collect_and_dedup_external_imports(
        [_module("a", ["from datetime import datetime"]), _module("b", ["import zlib"])]
    )
    assert not bundler.NAME_COLLISIONS, (
        f"names that do not collide were reported as collisions: {bundler.NAME_COLLISIONS}"
    )


def test_the_blank_line_collapse_never_edits_a_string_literal():
    """Bundle tidying must not rewrite embedded template text.

    filter_manager renders the Open WebUI filters it installs from triple-quoted
    source held in this package. A raw `re.sub(r"\\n{4,}", ...)` over the module
    shortened a blank run *inside* one of those strings, so the flat bundle installed
    a filter whose source differed from the package's -- caught only because two
    copies are compared byte for byte. Any embedded template may depend on its own
    whitespace, so the transform must be literal-aware, not merely lucky.
    """
    bundler = _bundler()

    source = (
        'HEADER = """\n'
        "a\n\n\n\n"
        'b\n"""\n\n\n\n\n'
        "def after():\n    return HEADER\n"
    )
    out = bundler._collapse_blank_runs_outside_strings(source)

    assert "a\n\n\n\nb" in out, (
        "the four-newline run inside the triple-quoted template was collapsed; the "
        "bundled filter source no longer matches what the package renders"
    )
    assert "\n\n\n\n\ndef after():" not in out, (
        "the blank run in real code was left alone, so the transform now does nothing"
    )

    untouched = "x = 1\n\n\ny = 2\n"
    assert bundler._collapse_blank_runs_outside_strings(untouched) == untouched


def test_only_one_script_writes_the_shipped_bundle_paths():
    """A second bundler defaulting to the same filenames is a loaded gun.

    scripts/bundle.py was the superseded v1 bundler: referenced by no workflow, no
    doc and no test, but its own usage string said `python scripts/bundle.py`, and it
    wrote to exactly the four artifact paths bundle_v2 writes -- without the anyio
    #1111 workaround, the SKIP_FILES exclusion, the internal-collision gate, the
    hoisted-import assertion or the per-module parse gate. Anyone following it
    replaced the shipped artifacts with a quietly worse bundle.
    """
    scripts = PROJECT_ROOT / "scripts"
    artifacts = {
        "open_webui_openrouter_pipe_bundled.py",
        "open_webui_openrouter_pipe_bundled_compressed.py",
        "open_webui_openrouter_pipe_bundled_no_plugins.py",
        "open_webui_openrouter_pipe_bundled_compressed_no_plugins.py",
    }
    writers = sorted(
        path.name
        for path in scripts.glob("*.py")
        if any(a in path.read_text(encoding="utf-8") for a in artifacts)
    )
    assert writers == ["bundle_v2.py"], (
        f"these scripts name a shipped bundle path: {writers}. Exactly one may, or the "
        "artifact that ships depends on which script someone happened to run."
    )


def test_the_blank_line_collapse_protects_f_string_templates_too():
    """PEP 701 changed f-string tokenisation in 3.12, and the guard keyed on STRING.

    filter_manager renders filters from f''' templates. Before 3.12 those are a single
    STRING token; from 3.12 they are FSTRING_START/MIDDLE/END and never STRING, so the
    protection covered them on CI's 3.11 and not on a 3.12 developer machine -- the
    same bundler emitting two different artifacts depending on the interpreter.
    """
    bundler = _bundler()
    nl = chr(10)
    source = (
        f"TPL = f'''{nl}a{nl}{nl}{nl}{nl}b{nl}'''{nl}{nl}{nl}{nl}{nl}"
        f"def after():{nl}    return TPL{nl}"
    )
    out = bundler._collapse_blank_runs_outside_strings(source)

    assert "a" + nl * 4 + "b" in out, (
        "the blank run inside the f-string template was collapsed; a bundle built on "
        "this interpreter does not match one built on the other, and the installed "
        "filter source stops matching what the package renders"
    )
    assert nl * 4 + "def after" not in out, (
        "the blank run in real code was not collapsed, so the transform does nothing "
        "and this test proves nothing about the f-string protection above"
    )


def _guard(source_module: str, fallback: str, handler: str = "Exception") -> str:
    return (
        f"try:\n"
        f"    from {source_module} import Thing\n"
        f"except {handler}:\n"
        f"    Thing = {fallback}\n"
    )


def _collect(bundler, tmp_path, sources: dict[str, str]):
    import ast

    mods = {}
    for short, src in sources.items():
        mod = bundler.ModuleInfo(
            dotted_name=short,
            file_path=tmp_path / f"{short}.py",
            raw_source=src,
            source_lines=src.splitlines(keepends=True),
            tree=ast.parse(src),
        )
        bundler.analyze_module(mod, {short: mod})
        mods[short] = mod
    return mods


def test_a_name_bound_inside_a_module_level_try_reaches_the_detector(tmp_path):
    """The hole: optional-import guards bind names the detector never collected.

    `analyze_module` walked `ast.iter_child_nodes`, so a binding inside a module-level
    `try` is a grandchild and was invisible. It stayed invisible for as long as the
    guards were spelled `except ImportError`, because those blocks were lifted out into
    the bundle header and deduplicated on the way. Respelling the handlers as
    `except Exception` -- which changed no behaviour in the package -- left them inline
    and unhoisted, and took the detector's coverage of them from 19 blocks to 6 without
    a single test going red.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {"alpha": _guard("pkg.one", "None"), "beta": _guard("pkg.one", "None")},
    )
    assert all("Thing" in m.top_level_names for m in mods.values()), (
        "a name bound inside a module-level try/except is invisible to the collision "
        "detector, so two modules binding it merge silently in the flat bundle"
    )


def test_guards_that_bind_the_same_thing_are_benign(tmp_path):
    """Anti-vacuity for the test below: the detector must not reject every guard.

    The handlers differ, and so do the type-checker suppressions in the real package.
    Neither can change which object the name receives, so neither may change the
    verdict -- the previous mechanism keyed on exactly that kind of spelling.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {
            "alpha": _guard("pkg.one", "None", handler="ImportError"),
            "beta": _guard("pkg.one", "None  # type: ignore", handler="Exception"),
        },
    )
    assert bundler._bindings_are_equivalent("Thing", ["alpha", "beta"], mods), (
        "two guards importing the same symbol from the same module with the same "
        "fallback were reported as a collision; the flat bundle binds one object here "
        "and failing the build would be a false alarm"
    )


@pytest.mark.parametrize(
    ("alpha", "beta", "why"),
    [
        (
            _guard("pkg.one", "None"),
            _guard("pkg.two", "None"),
            "the same name imported from two different modules",
        ),
        (
            _guard("pkg.one", "None"),
            _guard("pkg.one", "object()"),
            "the same import with two different fallbacks",
        ),
        (
            _guard("pkg.one", "None"),
            "Thing = 1\n",
            "a guard in one module and a plain assignment in the other",
        ),
    ],
)
def test_guards_that_bind_different_things_are_a_collision(tmp_path, alpha, beta, why):
    """The property: one slot in the collapsed namespace, so one object.

    Whichever module runs later wins for both, with no NameError and no failing test --
    the package keeps two namespaces, so only the shipped flat bundle misbehaves.
    """
    bundler = _bundler()
    mods = _collect(bundler, tmp_path, {"alpha": alpha, "beta": beta})
    assert not bundler._bindings_are_equivalent("Thing", ["alpha", "beta"], mods), (
        f"{why} was accepted as benign. In the flat bundle both write one slot, so "
        "every call site that meant the other one silently changes target."
    )


def test_nothing_compared_is_not_the_same_as_definitions_agreeing(tmp_path):
    """An empty comparison set must score as a collision, not as agreement.

    `_definitions_textually_identical` walks `ast.iter_child_nodes`, so a name bound
    inside a module-level `try` contributes no text at all. The set is then empty, and
    `len(texts) == 1` returned False only by accident -- one character (`<=`) turned
    "I compared nothing" into "they agree", which exempts the collision without ever
    consulting `_bindings_are_equivalent`, and the flat bundle ships with two different
    objects written into one slot.

    The probe is a guard-bound name on purpose. `test_annotated_globals_reach_the_
    collision_detector` looks like it covers this and does not: its `_dup: Any = None`
    returns early at the idempotency gate, before `len(texts)` is ever evaluated.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {"alpha": _guard("pkg.one", "None"), "beta": _guard("pkg.two", "object()")},
    )

    assert all("Thing" in m.top_level_names for m in mods.values()), (
        "the detector no longer collects guard-bound names, so this probe never "
        "reaches the predicate under test and would pass for the wrong reason"
    )
    assert not bundler._definitions_textually_identical("Thing", ["alpha", "beta"], mods), (
        "two guards binding the same name to different objects were called textually "
        "identical on the strength of an empty comparison set"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "from typing import cast, Any  # type: ignore[attr-defined]",
        "from typing import Any, cast  # noqa: F401",
        "from typing import cast, Any, Final  # pragma: no cover",
    ],
)
def test_a_directive_comment_never_swallows_a_name(tmp_path, statement):
    """The header must bind every name the module bodies lost.

    The trailing directive was appended to the reconstructed statement string, which the
    collector then comma-split and re-sorted. Whenever the annotated name stopped sorting
    last, everything after it landed inside the comment: `from typing import cast, Any
    # type: ignore` rendered as `from typing import Any  # type: ignore, cast`, binding
    only `Any`. The artifact still parses, and `_assert_hoisted_imports_rebound` fails
    the build naming an unrelated module with no mention of the comment.

    Parametrised over orderings so the fix cannot be "sort the annotated name last".
    """
    import ast

    bundler = _bundler()
    src = f"{statement}\n\nx = 1\n"
    mod = bundler.ModuleInfo(
        dotted_name="probe",
        file_path=tmp_path / "probe.py",
        raw_source=src,
        source_lines=src.splitlines(keepends=True),
        tree=ast.parse(src),
    )
    bundler.analyze_module(mod, {"probe": mod})
    stdlib_lines, third_party, _names = bundler.collect_and_dedup_external_imports([mod])

    expected = sorted(
        a.asname or a.name for n in ast.parse(statement).body for a in n.names
    )
    lines = [ln for ln in stdlib_lines + third_party if ln.startswith("from typing ")]
    assert len(lines) == 1, f"expected one typing import line, got {lines}"
    bound = sorted(a.asname or a.name for n in ast.parse(lines[0]).body for a in n.names)
    assert bound == expected, (
        f"{lines[0]!r} binds {bound}, but the module lost {expected}. A name after the "
        "directive is now inside the comment, so the flat bundle raises NameError at "
        "the first call site that uses it."
    )
    assert "#" in lines[0], (
        f"{lines[0]!r} dropped the directive entirely; the suppression holds in package "
        "mode and would vanish from the artifact, which is what the dropped-comment "
        "report exists to prevent"
    )


def test_an_origin_the_scan_cannot_see_is_not_a_vote_for_benign(tmp_path):
    """One origin binding the name in a `try` must not leave the others voting alone.

    `_definitions_textually_identical` walks each origin for Class/Function/Assign
    definitions of the name. A module-level `try` is none of those, so an origin whose
    only binding lives in one contributed nothing and the surviving origins agreed with
    themselves. The build then reported `Identical redefinitions (benign)` and exit 0
    for a real collision: in the flat bundle the later assignment wins, so the module
    that expected the guarded import silently reads the other module's object -- no
    NameError, no warning, and only in the shipped artifact.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {
            "alpha": _guard("pkg.one", "None"),
            "beta": "import logging\n\nThing = logging.getLogger(__name__)\n",
        },
    )
    assert not bundler._definitions_textually_identical("Thing", ["alpha", "beta"], mods), (
        "one origin binds Thing in a try/except and the other assigns it a Logger, yet "
        "the collision was reported benign. The flat bundle collapses both into one "
        "name and the guarded import's value is silently discarded."
    )


def test_a_plain_module_global_is_never_idempotent(tmp_path):
    """`x = None` in two modules is a collision, however identical the text.

    It is the canonical shape for a global later reassigned through `global`, which is
    exactly where the flat bundle's single slot makes the last writer win for both
    modules. Accepting it because the two spellings match is accepting the defect.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {"alpha": "_state = None\n", "beta": "_state = None\n"},
    )
    assert not bundler._definitions_textually_identical("_state", ["alpha", "beta"], mods), (
        "two modules each declaring `_state = None` were reported benign; in the flat "
        "bundle they share one slot and whichever module assigns to it last wins for "
        "both."
    )


def test_a_rebinding_outside_the_guard_is_not_an_equivalent_binding(tmp_path):
    """Matching optional-import guards do not make two modules equivalent on their own.

    `_binding_signatures` inspects only top-level `try` nodes, so a plain `Thing = ...`
    after the guard was invisible to it. Two modules with identical guards then compared
    equal and the collision was printed benign -- while in the flat bundle the rebinding
    ran last and BOTH modules' call sites saw it. Measured on a built artifact: the
    package bound the rebound string, the bundle bound the imported object.

    The classifier must abstain rather than guess; `None` sends the caller to the
    stricter textual rule.
    """
    bundler = _bundler()
    mods = _collect(
        bundler,
        tmp_path,
        {
            "alpha": _guard("pkg.one", "None") + '\nThing = "rebound-by-alpha"\n',
            "beta": _guard("pkg.one", "None"),
        },
    )
    assert bundler._binding_signatures(mods["alpha"], "Thing") is None, (
        "alpha rebinds Thing at module scope after its guard, which this scan cannot "
        "see; it must abstain rather than report the guard's signature as the whole "
        "story"
    )
    assert not bundler._bindings_are_equivalent("Thing", ["alpha", "beta"], mods), (
        "two modules were called equivalent on their guards alone while one of them "
        "rebinds the name afterwards -- in the flat bundle that rebinding wins for both"
    )


def test_a_function_local_inside_a_guard_is_not_a_module_level_name(tmp_path):
    """`ast.walk` descended into nested defs and collected their locals.

    Since the collision report became a `raise`, a local variable name shared by two
    modules blocked the build outright, telling the author to "rename one of them" when
    nothing was wrong. Names bound in a nested scope are not module-level names.
    """
    bundler = _bundler()
    body = (
        "try:\n"
        "    import json\n"
        "\n"
        "    def encode(obj):\n"
        "        payload = json.dumps(obj)\n"
        "        return payload\n"
        "except Exception:\n"
        "    json = None\n"
        "\n"
        "    def encode(obj):\n"
        "        payload = repr(obj)\n"
        "        return payload\n"
    )
    mods = _collect(bundler, tmp_path, {"alpha": body, "beta": body})
    for short, mod in mods.items():
        assert "payload" not in mod.top_level_names, (
            f"{short} reports the function-local 'payload' as a module-level name, so "
            "two modules that both use it fail the build for no reason"
        )
        assert "json" in mod.top_level_names, (
            f"{short} lost the guarded import binding; the gate must still see names "
            "the try really does bind at module scope"
        )
        assert "encode" in mod.top_level_names, (
            f"{short} lost the function defined inside the guard, which IS a "
            "module-level binding"
        )
