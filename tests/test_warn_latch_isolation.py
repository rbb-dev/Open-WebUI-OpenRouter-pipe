"""The warn-once latch reset must actually work, and must cover every latch.

A warn-once latch is a module-level set that suppresses a warning for the life of the
process. Without a per-test reset, the first test to trip one silently disarms every
later assertion that the warning is emitted -- the suite stays green while the
assertions stop checking anything, and which test disarmed which is invisible.

conftest's ``_reset_warn_latches`` clears them. These tests exist because that fixture
is otherwise unfalsifiable: nothing else in the suite fails when it stops clearing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from open_webui_openrouter_pipe.core import config as _core_config

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"

EXPECTED_LATCHES = {
    "_warned_chat_chunk_parse",
    "_warned_collectors",
    "_warned_dropped_video_param",
    "_warned_image_catalog",
    "_warned_dropped_image_param",
    "_warned_image_cost_snapshot",
    "_warned_image_endpoints",
    "_warned_image_provider_keys",
    "_warned_forward_headers",
    "_warned_import_sites",
    "_warned_pinned_attachment",
    "_warned_pipes_maintenance",
    "_warned_plugin_dispatch",
    "_warned_provider_slug_guess",
    "_warned_queue_backlog",
    "_warned_responses_chunk_parse",
    "_warned_row_timestamps",
    "_warned_storage_provider",
    "_warned_system_resources",
    "_warned_timing_file",
    "_warned_user_valves",
    "_warned_video_catalog",
    "_warned_video_provider_keys",
}

_LATCH_RE = re.compile(
    r"^(_warned[A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*(?:set\(\)|dict\(\)|\{\}|\[\])", re.M
)


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_every_latch_in_the_source_is_accounted_for():
    located: list[tuple[str, str]] = []
    for path in PACKAGE_DIR.rglob("*.py"):
        rel = str(path.relative_to(PACKAGE_DIR)).replace("\\", "/")
        located += [(rel, n) for n in _LATCH_RE.findall(path.read_text(encoding="utf-8"))]

    duplicated = sorted({n for _, n in located if [x for _, x in located].count(n) > 1})
    assert not duplicated, (
        f"these latch names are declared by more than one module: {duplicated}. The "
        "reset sweep keys by module and name, so both are cleared -- but two modules "
        "sharing a latch name is the shape that silently merged into one variable in "
        "the flat bundle. Rename one."
    )
    found: set[str] = {n for _, n in located}

    assert found == EXPECTED_LATCHES, (
        "the set of warn-once latches changed.\n"
        f"  only in source:   {sorted(found - EXPECTED_LATCHES)}\n"
        f"  only in expected: {sorted(EXPECTED_LATCHES - found)}\n"
        "A new latch needs adding here so the reset is known to cover it; a renamed "
        "one silently stops being reset, which re-arms the bug this guards."
    )


def _live_latches() -> dict[str, object]:
    """Every inventoried latch, resolved to the live object on its module."""
    import importlib
    import sys

    found: dict[str, object] = {}
    for name, module in list(sys.modules.items()):
        root = name.split(".")[0]
        if root != PACKAGE_DIR.name and not name.startswith(PACKAGE_DIR.name + "."):
            continue
        for attr in EXPECTED_LATCHES:
            value = getattr(module, attr, None)
            if isinstance(value, (set, dict, list)):
                found[f"{name}.{attr}"] = value
    return found


@pytest.mark.parametrize("run", [1, 2, 3])
def test_every_latched_warning_is_reset_on_every_test(run):
    """Fails on run 2 if the reset fixture stops clearing ANY inventoried latch.

    The previous version armed one latch. Clearing only that one -- the shape a
    narrowed sweep produces -- left the other four permanently armed for the whole
    session with this file green, so the inventory above was pinned while the reset it
    exists to protect covered a fifth of it.
    """
    live = _live_latches()
    assert live, "no inventoried latch resolved to a live object; this checks nothing"

    still_held = {
        where: sorted(str(x) for x in value)
        for where, value in live.items()
        if isinstance(value, (set, dict, list)) and value
    }
    assert not still_held, (
        f"run {run}: these latches still held entries from an earlier test, so every "
        f"assertion that their warning is emitted is checking nothing: {still_held}"
    )

    for value in live.values():
        if isinstance(value, set):
            value.add("sentinel-from-this-test")
        elif isinstance(value, dict):
            value["sentinel-from-this-test"] = 0.0
        elif isinstance(value, list):
            value.append("sentinel-from-this-test")


def test_every_instance_held_latch_is_reset_by_construction():
    """A latch on `self` is safe only if `__init__` gives every instance a fresh one.

    The class census below sweeps `vars(cls)`, which cannot see an attribute assigned in
    `__init__`, and the module inventory's regex is `^`-anchored, so it cannot see an
    indented assignment either. Three latches live exactly there -- two of them spelled
    `_warned*`, so a reader checking "is mine covered?" sees the prefix in both guards
    and reasonably concludes yes.

    They are legitimately per-instance: `MultimodalHandler` and `SessionTracker` are
    per-Pipe, and a module-level latch would make one chat's warning silence every other
    worker's. What makes them safe is that each is assigned a NEW container in
    `__init__`, so a fresh Pipe starts clean. That is what this asserts -- discovered,
    not listed, so a fourth one arrives covered.
    """
    import ast
    import inspect
    import re
    import sys

    shape = re.compile(r"^(_warned\w*|\w*_warn_ts|\w*_warn_last_ts)$")
    unreset = []
    seen = 0
    # Deduped by module OBJECT, not by name. In the flat bundle every submodule name is
    # an alias for one module, so keying on the name re-parses the whole 2.6 MB artifact
    # once per alias -- 107 parses for 2 distinct modules, 56 s against CI's 60 s
    # per-test ceiling, and --timeout-method=thread kills the process rather than the
    # test.
    parsed: set[int] = set()
    for name, module in list(sys.modules.items()):
        if name != "open_webui_openrouter_pipe" and not name.startswith(
            "open_webui_openrouter_pipe."
        ):
            continue
        if id(module) in parsed:
            continue
        parsed.add(id(module))
        try:
            tree = ast.parse(inspect.getsource(module))
        except (OSError, TypeError, SyntaxError):
            continue
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            init = next(
                (
                    f
                    for f in cls.body
                    if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef)
                    and f.name == "__init__"
                ),
                None,
            )
            assigned = set()
            if init is not None:
                for node in ast.walk(init):
                    tgt = None
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        tgt = node.targets[0]
                    elif isinstance(node, ast.AnnAssign):
                        tgt = node.target
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and shape.match(tgt.attr)
                        # A fresh container: `set()`/`dict()` or a `{}`/`set()`/`[]`
                        # literal. Requiring a Call alone missed `= {}`.
                        and isinstance(
                            getattr(node, "value", None), ast.Call | ast.Dict | ast.Set | ast.List
                        )
                    ):
                        assigned.add(tgt.attr)
            for node in ast.walk(cls):
                tgt = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
                    node.target if isinstance(node, ast.AnnAssign) else None
                )
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and shape.match(tgt.attr)
                ):
                    seen += 1
                    if tgt.attr not in assigned:
                        unreset.append(f"{name}.{cls.name}.self.{tgt.attr}")

    # Two, not three: the third lives in `plugins/pipe_dashboard/session_tracker.py`,
    # which a no-plugins bundle does not contain, so a floor of three passed in three
    # build shapes and failed in the fourth. The floor exists only to catch a discovery
    # that has gone blind and returns nothing.
    assert seen >= 2, (
        f"only {seen} instance-held latches found; the discovery has gone blind and "
        "this guard is asserting nothing"
    )
    assert not unreset, (
        "these latches are held on an instance but never given a fresh container in "
        f"__init__, so they persist for as long as the object does: {sorted(set(unreset))}"
    )


def test_no_warn_latch_hides_on_a_class():
    """No warn-once latch is held on a CLASS, where the autouse sweep cannot reach it.

    Module-level latches are swept by `_warn_latches()`; instance-level ones are covered
    by the sibling above, which requires `__init__` to build a fresh container. A CLASS
    attribute is the one placement with neither protection.

    `Pipe._timing_file_warned_path` used to be a class attribute. `_warn_latches()`
    sweeps module attributes only, so it leaked between tests: four tests in
    test_pipe.py reset it by hand and the fifth did not, leaving its assertion silently
    order-dependent. It is now a module-level set like every other latch.

    This asserts the CATEGORY stays empty rather than testing the specific attribute
    that was moved -- a new class-held latch would reintroduce exactly the same leak,
    so the remedy is to move it to module level, where `_warn_latches()` reaches it.
    """
    import inspect
    import sys

    offenders = []
    inspected = 0
    for name, module in list(sys.modules.items()):
        if name != "open_webui_openrouter_pipe" and not name.startswith(
            "open_webui_openrouter_pipe."
        ):
            continue
        for cls_name, cls in vars(module).items():
            if not inspect.isclass(cls) or getattr(cls, "__module__", None) != name:
                continue
            inspected += 1
            for attr, value in vars(cls).items():
                if attr.startswith("_warned") or attr.endswith("_warned_path"):
                    offenders.append(f"{name}.{cls_name}.{attr} = {value!r}")

    if not inspected:
        pytest.skip(
            "in a flat bundle every submodule name aliases one module whose __name__ is "
            "the host module, so cls.__module__ never equals the alias and this guard "
            "sees no classes at all. The property is a source shape; package and "
            "compressed modes cover it."
        )
    assert not offenders, (
        "these warn-once latches are held on a CLASS, so the autouse module sweep in "
        f"conftest never resets them and they leak between tests: {sorted(offenders)}. "
        "Move each to a module-level set."
    )
