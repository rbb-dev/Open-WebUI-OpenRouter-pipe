"""The citation date stamp must render in the server's local frame.

Lives outside the dashboard test module on purpose: that module opens with
``pytest.importorskip`` on the plugin package, so a guard over ``streaming/`` source
placed there disappears entirely in both no-plugins bundle modes -- the guard is gone
exactly where nobody is looking.
"""

from __future__ import annotations

from typing import Any, cast

import os

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"

# Interpolated into the failure message, never retyped there: the message used to quote
# a stale literal while the assertion compared a different one.
_EXPECTED_OWUI_IMPORTS = (17, 76)

@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_every_citation_stamp_goes_through_the_shared_helper():
    """No site may rebuild the stamp inline.

    Five copies of the same expression existed, and the only thing a scan could check
    across them was that the text `.astimezone()` appeared. Appending a second
    `.astimezone(datetime.UTC)` keeps that text and re-stamps the UTC instant, which
    is the wrong calendar day for part of every day east of UTC -- and it is persisted
    into chat exports, so the wrong value is durable. One helper means one place to
    assert the actual behaviour, which the test below does.
    """
    import ast

    from tests.package_sources import parsed_sources

    helpers = {"citation_access_stamp"}
    found, offenders = 0, []
    for path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "date_accessed"
                ):
                    pairs_to_check = [(target.slice, node.value)]
                else:
                    continue
            elif isinstance(node, ast.Dict):
                pairs_to_check = list(zip(node.keys, node.values))
            else:
                continue
            for key, value in pairs_to_check:
                if not (isinstance(key, ast.Constant) and key.value == "date_accessed"):
                    continue
                found += 1
                is_helper = (
                    isinstance(value, ast.Call)
                    and (getattr(value.func, "id", None) or getattr(value.func, "attr", None))
                    in helpers
                )
                if not is_helper:
                    offenders.append(
                        f"{path.relative_to(PACKAGE)}:{value.lineno}: "
                        f"{ast.unparse(value)}"
                    )

    assert found == 5, (
        f"{found} date_accessed sites found, expected 5. Update this deliberately -- "
        "a floor with slack lets a site drop out of the scan unnoticed."
    )
    assert not offenders, (
        "these build the citation stamp inline instead of calling the shared helper, "
        "so their behaviour is unasserted:\n  " + "\n  ".join(offenders)
    )


def test_the_citation_helpers_render_in_the_servers_own_frame():
    """Assert the value under a timezone where UTC and local disagree TODAY.

    Comparing the helper's date against `datetime.now().astimezone().date()` in the
    ambient zone is not a test: for most of the day those agree even when the helper
    stamps UTC, so a re-normalising mutation passes. The zone below is chosen from the
    current UTC hour precisely so the two calendar days differ right now, which is the
    condition under which the bug is visible to a user.
    """
    import datetime as _dt
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation needs tzset (POSIX only)")


    from open_webui_openrouter_pipe.core import utils as _utils

    utc_now = _dt.datetime.now(_dt.UTC)
    zone = "Etc/GMT+12" if utc_now.hour < 12 else "Etc/GMT-14"

    saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        local_now = _dt.datetime.now(_dt.UTC).astimezone()
        assert local_now.date() != _dt.datetime.now(_dt.UTC).date(), (
            f"{zone} was expected to put local and UTC on different calendar days at "
            f"{utc_now.isoformat()}, but it did not -- this test can no longer detect "
            "a UTC-stamped date and must be re-anchored."
        )

        day = _utils.citation_access_stamp().split("T")[0]
        assert day == local_now.date().isoformat(), (
            f"under {zone} the citation date is {day}, but the server's local date is "
            f"{local_now.date().isoformat()} (UTC is "
            f"{_dt.datetime.now(_dt.UTC).date().isoformat()}). The stamp is rendering "
            "the UTC calendar day, which is the wrong day for every reader in this "
            "zone and is persisted into chat exports."
        )

        assert _dt.date.fromisoformat(day) == local_now.date(), (
            f"{day!r} does not parse back to the local calendar day; the stamp is the "
            "value Open WebUI renders as-is into the citation"
        )
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_a_broken_version_lookup_does_not_break_the_import():
    """__init__ is the package's first executable statement.

    importlib.metadata raises more than PackageNotFoundError -- a malformed METADATA
    file raises ValueError -- and narrowing the handler made any such failure take the
    whole pipe down rather than falling back to the pinned version.
    """
    import ast
    from pathlib import Path

    init = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe" / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))

    handlers = [
        h
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if any("_get_version" in ast.unparse(s) for s in node.body)
    ]
    assert handlers, "the version lookup is no longer wrapped; scan is stale"
    for h in handlers:
        assert h.type is None or ast.unparse(h.type) == "Exception", (
            f"the version lookup catches only {ast.unparse(h.type)}; anything else "
            "importlib.metadata raises makes the entire package unimportable"
        )


def _latch_name(target) -> str | None:
    """`x` or `self.x` -- a per-instance latch is still a latch."""
    import ast

    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_perf_counter_cooldowns_are_seeded_below_zero():
    """Every warn-once cooldown seed must be a NEGATIVE NUMERIC LITERAL.

    time.perf_counter() measures uptime, so seeding a "last warned at" to 0.0 mutes the
    first warning for the whole cooldown after a host boot -- and so does any other
    non-negative value.

    This requires a canonical SHAPE rather than detecting a forbidden VALUE, and that
    distinction is the point. An earlier version asked "does this evaluate to 0?",
    which has infinitely many spellings -- `float(0)`, `0.0 + 0.0`, `[0.0][0]`, a name
    bound to any of those -- and each review round found spellings the last one missed.
    "Is this `-<number>`?" is closed: a disguise is not a negative literal, so it FAILS
    this check rather than slipping past it. The failure direction is inverted, which
    is what makes it stable.
    """
    import ast

    from tests.package_sources import parsed_sources

    offenders, checked = [], 0

    for path, source, tree in parsed_sources("open_webui_openrouter_pipe"):
        if "perf_counter" not in source and "monotonic" not in source:
            continue
        for node in ast.walk(tree):
            target = value = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, (ast.Name, ast.Attribute)):
                target, value = _latch_name(node.target), node.value
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Name, ast.Attribute))
            ):
                target, value = _latch_name(node.targets[0]), node.value
            if target and target.endswith("_warn_last_ts"):
                if isinstance(value, (ast.Name, ast.Call, ast.Attribute)):
                    continue
            else:
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                ):
                    continue
                owner = _latch_name(node.func.value)
                if not (owner and owner.endswith("_warn_ts")):
                    continue
                target, value = f"{owner}.get(...)", node.args[1]
            checked += 1
            if not _is_negative_literal(value):
                offenders.append(
                    f"{path.relative_to(PACKAGE)}:{getattr(node, 'lineno', 0)} {target} = "
                    f"{ast.unparse(value) if value is not None else '<none>'}"
                )

    # No count assertion: every cooldown seed is gone, because every site now keys on
    # warn_level's dict latch where an absent key IS the never-warned state. A `== 0`
    # would be satisfied by a matcher that matches nothing, so the matcher is driven
    # against literal text instead -- the population being correctly zero is exactly
    # when a census stops being able to prove it still works.
    _probe_offenders, _probe_checked = [], 0
    for node in ast.walk(ast.parse("_last_warn_ts: float = 0.0\n")):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = _latch_name(node.target), node.value
            if name and ("warn" in name or "cooldown" in name):
                _probe_checked += 1
                if not _is_negative_literal(value):
                    _probe_offenders.append(name)
    assert _probe_checked == 1 and _probe_offenders == ["_last_warn_ts"], (
        "the cooldown-seed matcher no longer recognises `_last_warn_ts: float = 0.0`, "
        "so the package scan above reports green by matching nothing"
    )
    assert not offenders, (
        "these are not written as a negative numeric literal, so nothing shows they "
        "seed the cooldown in the past. perf_counter is uptime, so anything >= 0 mutes "
        "the first warning for the whole cooldown after a host boot:\n  "
        + "\n  ".join(offenders)
    )


def _is_negative_literal(value) -> bool:
    """`-1e9` and friends: unary minus applied to a positive numeric constant."""
    import ast

    if not isinstance(value, ast.UnaryOp) or not isinstance(value.op, ast.USub):
        return False
    operand = value.operand
    return (
        isinstance(operand, ast.Constant)
        and isinstance(operand.value, (int, float))
        and not isinstance(operand.value, bool)
        and operand.value > 0
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_open_webui_imports_are_guarded_broadly():
    """A module-scope Open WebUI import must catch broadly, or the pipe cannot load.

    open_webui.env alone does mkdir, shutil.copytree, make_archive and a dict index at
    import time, so a read-only DATA_DIR is enough to raise. At module scope that
    makes the whole package unimportable, which is why the guard exists. A lazy import
    inside a function is a different risk class -- it raises into a caller that can
    handle it -- so the requirement is scoped to where the rationale applies, and the
    lazy ones are counted rather than demanded.

    Walks EVERY Open WebUI import and classifies it, rather than inspecting only ones
    already inside a try: that earlier shape could not see an import whose guard had
    been deleted outright. It also matches the resolved module name instead of a
    concatenated string -- joining module and alias with no separator rendered
    `from open_webui import env` as "open_webuienv", which matched nothing, so the
    package's own motivating example was scanned only because a sibling dotted import
    happened to share its try block.
    """
    import ast

    from tests.package_sources import parsed_sources

    offenders, module_scope, lazy = [], 0, 0

    for path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent

        broadly_guarded: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.handlers:
                continue
            if not any(
                h.type is None or ast.unparse(h.type) in {"Exception", "BaseException"}
                for h in node.handlers
            ):
                continue
            for stmt in node.body:
                broadly_guarded.update(
                    id(n) for n in ast.walk(stmt)
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                )

        def _at_module_scope(node: ast.AST) -> bool:
            cur = parents.get(id(node))
            while cur is not None:
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    return False
                cur = parents.get(id(cur))
            return True

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets = [node.module or ""]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                continue
            if not any(t == "open_webui" or t.startswith("open_webui.") for t in targets):
                continue
            if not _at_module_scope(node):
                lazy += 1
                continue
            module_scope += 1
            if id(node) not in broadly_guarded:
                offenders.append(
                    f"{path.relative_to(PACKAGE)}:{node.lineno} imports {targets}"
                )

    assert (module_scope, lazy) == _EXPECTED_OWUI_IMPORTS, (
        f"found {module_scope} module-scope and {lazy} lazy Open WebUI imports, "
        f"expected {_EXPECTED_OWUI_IMPORTS}. Update these deliberately -- a floor with slack absorbs an "
        "import dropping out of the scan silently."
    )
    assert not offenders, (
        "these import Open WebUI at module scope without a broad guard, so anything "
        "its module body raises makes the package unimportable:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_the_bundle_heads_own_metadata_lookup_is_guarded_broadly():
    """The workaround block is the artifact's first executable statement.

    The sibling scan walks the package only, so narrowing this file's guard back to
    ImportError is invisible to it -- and anything importlib.metadata raises there
    (a malformed METADATA raises ValueError) makes the whole bundle unimportable
    before a single line of the pipe runs.
    """
    import ast
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "anyio_1111_workaround.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any("_pkg_version(" in ast.unparse(s) for s in node.body)
    ]
    assert guarded, "the anyio version lookup is no longer wrapped; this guard is stale"
    for node in guarded:
        handled = {ast.unparse(h.type) if h.type else "bare" for h in node.handlers}
        assert handled & {"Exception", "BaseException", "bare"}, (
            f"the anyio version lookup catches only {sorted(handled)}; anything else "
            "importlib.metadata raises kills the bundle at its first statement"
        )


@pytest.mark.asyncio
async def test_the_emitted_citation_stamp_is_the_local_calendar_day():
    """Assert the EMITTED value under a zone that is not UTC.

    Comparing against the ambient zone made this vacuous on a UTC host -- which is
    every CI runner: both sides are +00:00, so re-normalising the stamp back to UTC
    passed. The zone is forced here for the same reason the sibling forces it, and
    this one covers the emit path rather than the helper in isolation.
    """
    import datetime as _dt
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation needs tzset (POSIX only)")

    from open_webui_openrouter_pipe.streaming.event_emitter import EventEmitterHandler

    emitted: list[dict] = []

    async def _sink(event):
        emitted.append(event)

    # The zone is derived from the current UTC hour so that local and UTC ALWAYS land on
    # different calendar days. A fixed +10 only straddles midnight when the UTC hour is
    # 14 or later; for the other 14 hours the two dates are the same string and the
    # assertion below cannot tell a local stamp from a UTC one. CI runs in UTC.
    saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/GMT+12" if _dt.datetime.now(_dt.UTC).hour < 12 else "Etc/GMT-14"
    time.tzset()
    try:
        handler = EventEmitterHandler.__new__(EventEmitterHandler)
        await EventEmitterHandler._emit_citation(
            handler, _sink, {"document": "d", "source": {"url": "https://example.test"}}
        )

        assert emitted, "no source event was emitted"
        stamp = emitted[0]["data"]["metadata"][0]["date_accessed"]
        local_day = _dt.datetime.now(_dt.UTC).astimezone().date().isoformat()
        utc_day = _dt.datetime.now(_dt.UTC).date().isoformat()
        assert local_day != utc_day, (
            f"the forced zone put local ({local_day}) and UTC ({utc_day}) on the same "
            "calendar day, so the assertion below is satisfied by a UTC stamp too and "
            "this test proves nothing. Re-anchor it."
        )
        assert stamp.split("T")[0] == local_day, (
            f"{stamp!r} is not the server's local calendar day ({local_day}); Open "
            "WebUI renders the string as-is, so the citation shows a date the reader "
            "never experienced."
        )
        assert stamp.split("T")[0] != utc_day, (
            f"{stamp!r} carries the UTC calendar day, not the local one -- which is the "
            "whole point of the .astimezone() in citation_access_stamp"
        )
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()


_EXPECTED_RENDERERS = 10
"""How many filter renderers the package defines. A count, not a floor."""


def _image_model_filter_spec_for_stamp_check():
    """One real spec, so the per-model renderer is exercised like the fixed ones were."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    return build_image_model_filter_spec(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        {
            "provider_slug": "recraft",
            "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
        },
        dedicated_image_api=True,
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_the_installed_filters_guard_their_open_webui_import():
    """The filters the pipe installs must follow the rule the pipe follows.

    The package-wide scan walks open_webui_openrouter_pipe/ only, so the three filter
    sources -- which live in filters/ and inside string templates in filter_manager --
    were outside every guard. `open_webui.env` does mkdir and copytree at import, so
    on a read-only DATA_DIR an unguarded import makes the installed filter fail to
    load, and Open WebUI shows the operator a broken filter rather than a degraded one.
    """
    import ast
    import importlib
    from pathlib import Path

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    from open_webui_openrouter_pipe.pipe import Pipe

    manager = cast("Any", FilterManager.__new__(FilterManager))
    manager._pipe = None
    manager._valves = Pipe.Valves()

    # Discovered, not enumerated. The previous version listed five renderers by hand
    # while the package renders several, so the fusion and provider-routing templates
    # could lose their guard with the suite green.
    # The coverage assertion below is what makes renderer twelve a failure rather than
    # a silent omission.
    ARGS: dict[str, dict[str, object]] = {
        "render_openrouter_web_tools_filter_source": {
            "enable_web_search": True, "enable_web_fetch": True, "enable_datetime": True
        },
        "render_openrouter_image_gen_filter_source": {"dedicated_image_api": True},
        "render_direct_uploads_filter_source": {},
        "render_openrouter_image_filter_source": {
            "model_id": "recraft/recraft-v3",
            "image_model": {"id": "recraft/recraft-v3", "name": "Recraft V3"},
            "dedicated_image_api": True,
        },
        "render_openrouter_video_gen_filter_source": {
            "model_id": "google/veo-3",
            "video_model": {"id": "google/veo-3", "name": "Veo 3"},
        },
        "_render_provider_routing_filter_source": {
            "model_slug": "openai/gpt-4o", "providers": ["openai"],
            "quantizations": ["fp16"], "visibility": "both",
        },
        "render_openrouter_fusion_filter_source": {"marker": "fusion"},
        "render_image_model_filter_source": {
            "spec": _image_model_filter_spec_for_stamp_check(),
        },
        "render_image_gen_filter_source": {
            "spec": _image_model_filter_spec_for_stamp_check(),
            "catalog_match": True,
        },
        "render_video_filter_source": {
            "model_id": "google/veo-3",
            "video_model": {"id": "google/veo-3", "name": "Veo 3"},
        },
    }

    discovered: dict[str, Any] = {}
    for module_name in (
        "filter_manager", "fusion_filter_renderer",
        "image_filter_renderer", "video_filter_renderer",
    ):
        module = importlib.import_module(
            f"open_webui_openrouter_pipe.filters.{module_name}"
        )
        for attr, value in vars(module).items():
            if attr.startswith("render_") and callable(value):
                discovered[attr] = value
    for attr in dir(FilterManager):
        if ("render_" in attr) and callable(getattr(FilterManager, attr, None)):
            discovered[attr] = getattr(FilterManager, attr)

    uncovered = [a for a in sorted(discovered) if a not in ARGS]
    assert not uncovered, (
        f"these filter renderers are not in the argument table, so their sources are "
        f"never checked for a guarded open_webui import: {uncovered}. Add them rather "
        "than narrowing the sweep."
    )
    stale = [a for a in ARGS if a not in discovered]
    assert not stale, (
        f"the argument table names renderers that no longer exist: {stale}; the table "
        "would silently stop covering anything it names"
    )

    rendered: dict[str, str] = {}
    for name, kwargs in ARGS.items():
        fn = discovered[name]
        if name in {
            "render_openrouter_video_gen_filter_source",
        }:
            rendered[name] = fn(manager, **kwargs)
        else:
            rendered[name] = fn(**kwargs)

    assert len(discovered) == _EXPECTED_RENDERERS, (
        f"{len(discovered)} filter renderers found, expected {_EXPECTED_RENDERERS}. Update "
        "this deliberately -- the two membership assertions above already force ARGS and "
        "the discovered set to match each other, so they cannot notice a coordinated "
        "shrink where a module and its ARGS entries are dropped in one edit."
    )

    template_total, template_offenders = 0, []
    per_renderer: dict[str, int] = {}
    for label, source in rendered.items():
        rendered_tree = ast.parse(source)
        guarded_here: set[int] = set()
        for node in ast.walk(rendered_tree):
            if not isinstance(node, ast.Try) or not node.handlers:
                continue
            if not any(
                h.type is None or ast.unparse(h.type) in {"Exception", "BaseException"}
                for h in node.handlers
            ):
                continue
            for stmt in node.body:
                guarded_here.update(
                    id(n) for n in ast.walk(stmt)
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                )
        for node in ast.walk(rendered_tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            if not any(n == "open_webui" or n.startswith("open_webui.") for n in names):
                continue
            template_total += 1
            per_renderer[label] = per_renderer.get(label, 0) + 1
            if id(node) not in guarded_here:
                template_offenders.append(f"{label}: line {node.lineno} imports {names}")

    # Per renderer, not a sum. The docstring's property is "every installable filter
    # imports SRC_LOG_LEVELS", and a total only happens to express that while every
    # renderer contributes exactly one. The moment one legitimately gains a second
    # open_webui import the sum acquires slack, and a different renderer can lose its
    # import entirely with the total unchanged -- an absent import is not an unguarded
    # one, so the offender check below would then inspect nothing for it, silently.
    importless = sorted(label for label in rendered if not per_renderer.get(label))
    assert not importless, (
        "these rendered filters contain no Open WebUI import at all, so the guard check "
        f"below inspects nothing for them: {importless}"
    )
    assert not template_offenders, (
        "these generated filters import Open WebUI without a broad guard, so the "
        "installed filter fails to load if open_webui.env raises:\n  "
        + "\n  ".join(template_offenders)
    )

    filters_dir = Path(__file__).resolve().parents[1] / "filters"
    checked, offenders = 0, []
    for path in sorted(filters_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.handlers:
                continue
            if not any(
                h.type is None or ast.unparse(h.type) in {"Exception", "BaseException"}
                for h in node.handlers
            ):
                continue
            for stmt in node.body:
                guarded.update(
                    id(imp) for imp in ast.walk(stmt)
                    if isinstance(imp, (ast.Import, ast.ImportFrom))
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets = [node.module or ""]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                continue
            if not any(t == "open_webui" or t.startswith("open_webui.") for t in targets):
                continue
            checked += 1
            if id(node) not in guarded:
                offenders.append(f"{path.name}:{node.lineno} imports {targets}")

    assert checked == 3, (
        f"{checked} open_webui imports found across filters/, expected 3. Update this "
        "deliberately -- a floor with slack lets one drop out of the scan."
    )
    assert not offenders, (
        "these installed filters import Open WebUI without a broad guard, so anything "
        "its module body raises stops the filter loading:\n  " + "\n  ".join(offenders)
    )


def test_the_access_stamp_carries_a_time_not_just_a_date(monkeypatch):
    """Precision preserved, not merely uniformity.

    `_emit_citation` stamped a full local timestamp; the other four sites stamped a bare
    date. Consolidating them onto one helper adopted the date-only form and silently
    dropped the time from the one that had it -- in a value Open WebUI persists into the
    chat record and exports verbatim.

    Parametrised over two distinct instants via a frozen clock so a hardcoded return
    cannot satisfy both, and asserted on the PARSED value rather than on the presence of
    a "T", which a date-only string with a suffix would also satisfy.
    """
    import datetime as _dt

    from open_webui_openrouter_pipe.core import utils as _utils

    seen = []
    for hour, minute in ((3, 7), (21, 44)):
        class _Frozen(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime(2026, 3, 14, hour, minute, 5, tzinfo=tz or _dt.UTC)

        monkeypatch.setattr(_utils.datetime, "datetime", _Frozen)
        stamp = _utils.citation_access_stamp()
        parsed = _dt.datetime.fromisoformat(stamp)
        assert parsed.time() != _dt.time(0, 0), (
            f"citation_access_stamp() returned {stamp!r}, which has no time component. "
            "The value is persisted into chat exports; a bare date loses when the source "
            "was actually read."
        )
        seen.append(parsed)

    assert seen[0] != seen[1], (
        "both frozen clocks produced the same stamp, so the value does not come from the "
        "clock at all"
    )
