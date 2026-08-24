"""Config-tab valve editing: every valve survives edit -> save -> round-trip, and a save stores only the custom subset."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from open_webui_openrouter_pipe.core.config import EncryptedStr, Valves
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import config_service as cs

_ALL_FIELDS = list(Valves.model_fields.items())
_IDS = [name for name, _ in _ALL_FIELDS]


def _valid_new_value(name, field):
    annotation = field.annotation
    if cs.is_secret(annotation):
        return "brand-new-secret-value-1234567890"
    enum = cs._literal_options(annotation)
    default = field.get_default(call_default_factory=True)
    if enum:
        for option in enum:
            if option != default:
                return option
        return enum[0]
    base, _nullable = cs._base_type(annotation)
    bounds = cs._bounds(field) or {}
    if base is bool:
        return not bool(default)
    if base is int:
        lo = int(bounds.get("ge", bounds.get("gt", 0)))
        if "gt" in bounds and "ge" not in bounds:
            lo += 1
        hi = int(bounds.get("le", bounds.get("lt", lo + 1000)))
        if "lt" in bounds and "le" not in bounds:
            hi -= 1
        for cand in (lo, lo + 1, hi, (lo + hi) // 2):
            if lo <= cand <= hi and cand != default:
                return cand
        return lo
    if base is float:
        lo = float(bounds.get("ge", bounds.get("gt", 0.0)))
        hi = float(bounds.get("le", bounds.get("lt", lo + 10.0)))
        for cand in (lo, lo + 0.5, hi, (lo + hi) / 2):
            if lo <= cand <= hi and cand != default:
                return cand
        return lo
    if name.endswith("_TEMPLATE"):
        return "## Edited\nsecond line\n{{#if error_id}}- `{error_id}`{{/if}}"
    return "edited-config-value"


@pytest.mark.parametrize("name,field", _ALL_FIELDS, ids=_IDS)
def test_every_valve_edit_round_trips(name, field):
    new_value = _valid_new_value(name, field)
    dumped = cs.merge_for_save(Valves, current={}, edits={name: new_value})
    assert set(dumped) == {name}, f"expected only {name}, got {sorted(dumped)}"
    reconstructed = getattr(Valves(**dumped), name)
    if cs.is_secret(field.annotation):
        assert EncryptedStr.decrypt(str(reconstructed)) == new_value
    else:
        assert reconstructed == new_value


def test_secret_blank_edit_keeps_existing_key():
    dumped = cs.merge_for_save(Valves, current={"API_KEY": "existing-openrouter-key-abc123"}, edits={"API_KEY": ""})
    assert EncryptedStr.decrypt(dumped["API_KEY"]) == "existing-openrouter-key-abc123"


def test_secret_new_value_replaces():
    dumped = cs.merge_for_save(Valves, current={"API_KEY": "old"}, edits={"API_KEY": "new-key-9"})
    assert EncryptedStr.decrypt(dumped["API_KEY"]) == "new-key-9"


def test_secret_equal_to_env_default_is_dropped(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-123")
    dumped = cs.merge_for_save(Valves, current={"API_KEY": "env-key-123"}, edits={"MODEL_ID": "x"})
    assert "API_KEY" not in dumped
    assert dumped["MODEL_ID"] == "x"


def test_secret_existing_ciphertext_preserved(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-key-xyz")
    ct = EncryptedStr.encrypt("stored-real-key")
    assert ct.startswith("encrypted:")
    dumped = cs.merge_for_save(Valves, current={"API_KEY": ct}, edits={"MODEL_ID": "x"})
    assert dumped["API_KEY"] == ct
    assert EncryptedStr.decrypt(dumped["API_KEY"]) == "stored-real-key"


def test_secret_never_exposed_in_describe():
    specs = {s["name"]: s for s in cs.describe_valves(Valves)}
    for name in ("API_KEY", "ARTIFACT_ENCRYPTION_KEY", "SESSION_LOG_ZIP_PASSWORD"):
        assert specs[name]["secret"] is True
        assert specs[name]["default"] is None
        assert "value" not in specs[name]


def test_editing_one_valve_persists_only_custom_subset():
    current = {"MODEL_ID": "anthropic/*", "MAX_CONCURRENT_REQUESTS": 500}
    dumped = cs.merge_for_save(Valves, current=current, edits={"MAX_CONCURRENT_REQUESTS": 300})
    assert set(dumped) == {"MODEL_ID", "MAX_CONCURRENT_REQUESTS"}
    assert dumped["MAX_CONCURRENT_REQUESTS"] == 300
    assert dumped["MODEL_ID"] == "anthropic/*"


def test_nullable_field_none_edit_reconstructs_to_none():
    dumped = cs.merge_for_save(Valves, current={}, edits={"HTTP_TOTAL_TIMEOUT_SECONDS": None})
    assert "HTTP_TOTAL_TIMEOUT_SECONDS" not in dumped
    assert Valves(**dumped).HTTP_TOTAL_TIMEOUT_SECONDS is None


def test_enum_rejects_invalid_value():
    with pytest.raises(Exception):
        cs.merge_for_save(Valves, current={}, edits={"REASONING_EFFORT": "not-a-real-option"})


def test_numeric_bound_rejects_out_of_range():
    with pytest.raises(Exception):
        cs.merge_for_save(Valves, current={}, edits={"TOOL_TIMEOUT_SECONDS": 99999})


def test_multiline_template_round_trips_intact():
    tpl = "# Error\n{{#if error_id}}\n- **ID**: `{error_id}`\n{{/if}}\nContact support."
    dumped = cs.merge_for_save(Valves, current={}, edits={"INTERNAL_ERROR_TEMPLATE": tpl})
    assert dumped["INTERNAL_ERROR_TEMPLATE"] == tpl


_RESTORABLE = ["RATE_LIMIT_TEMPLATE", "CONNECTION_ERROR_TEMPLATE"]
_BLANKS = ["", "   ", "\n\t \n"]


def _factory_text(name):
    return Valves.model_fields[name].get_default(call_default_factory=True)


def _effective(stored):
    """Mirror of the Config tab's read path (actions._effective_valves)."""
    return Valves(**{k: v for k, v in stored.items() if v is not None})


@pytest.mark.parametrize("name", _RESTORABLE)
@pytest.mark.parametrize("blank", _BLANKS, ids=["empty", "spaces", "newlines"])
def test_clearing_a_template_restores_its_factory_text(name, blank):
    """An admin who mangles an error template recovers it by clearing the box and saving.

    The built-in text lives only in Python source an admin cannot read, so without this there is
    no way back from a bad edit. Asserts the value an admin would then SEE in the box, not the
    shape of the stored subset, so the persistence strategy stays free to change. Two valves with
    two different factory texts, so no single hardcoded string satisfies both, and three spellings
    of "cleared" including whitespace-only.
    """
    mangled = cs.merge_for_save(Valves, current={}, edits={name: "MANGLED {oops"})
    assert getattr(_effective(mangled), name) == "MANGLED {oops"

    cleared = cs.merge_for_save(Valves, current=mangled, edits={name: blank})
    assert getattr(_effective(cleared), name) == _factory_text(name)


# Open WebUI's own valve-update route needs nothing from this plugin, so its guard lives
# in tests/test_template_valve_restore.py and runs in the plugin-free bundles too.


@pytest.mark.parametrize("name", _RESTORABLE)
def test_a_real_template_edit_survives_the_save(name):
    """Restoring on blank must not degrade into restoring always.

    A validator that returned the default unconditionally would pass every blank-input assertion
    while silently discarding an admin's customisation, so a non-blank edit is checked for each
    valve the restore covers.
    """
    edited = "## House style\n{{#if error_id}}- `{error_id}`{{/if}}\n"
    stored = cs.merge_for_save(Valves, current={}, edits={name: edited})
    assert getattr(_effective(stored), name) == edited


def test_clearing_one_template_leaves_every_other_valve_alone():
    """Clearing a box is a per-valve reset, not a reset to factory defaults.

    Two unrelated valves and a SECOND customised template are all in the stored subset when one
    template is cleared; every one of them must come back unchanged.
    """
    current = {
        "MODEL_ID": "anthropic/*",
        "MAX_CONCURRENT_REQUESTS": 300,
        "CONNECTION_ERROR_TEMPLATE": "custom connection text",
    }
    stored = cs.merge_for_save(Valves, current=current, edits={"RATE_LIMIT_TEMPLATE": ""})
    effective = _effective(stored)
    assert effective.MODEL_ID == "anthropic/*"
    assert effective.MAX_CONCURRENT_REQUESTS == 300
    assert effective.CONNECTION_ERROR_TEMPLATE == "custom connection text"
    assert effective.RATE_LIMIT_TEMPLATE == _factory_text("RATE_LIMIT_TEMPLATE")


@pytest.mark.parametrize("name", ["BASE_URL", "TIMING_LOG_FILE"])
def test_a_cleared_non_template_string_valve_stays_cleared(name):
    """The restore is scoped to templates; for every other string valve empty still means empty.

    Two non-template string valves that both have a non-empty built-in default, so a rule widened
    to "any string valve with a default" would fail here.
    """
    stored = cs.merge_for_save(Valves, current={}, edits={name: ""})
    assert getattr(_effective(stored), name) == ""


def test_every_valve_the_config_tab_shows_as_a_template_is_restorable():
    """The Config tab's is_template flag and the blank-restore rule are the same predicate.

    Adding a new *_TEMPLATE valve must not produce a box the UI renders with the template editor
    and diff view but which cannot be recovered once cleared. Checked against every field, so the
    guard grows with the valve set rather than against a copied list.
    """
    specs = {spec["name"]: spec for spec in cs.describe_valves(Valves)}
    templates = sorted(name for name, spec in specs.items() if spec["is_template"])
    assert templates, "describe_valves reported no template valves"
    for name in templates:
        assert getattr(Valves.model_validate({name: "   "}), name) == _factory_text(name), name


def test_enrichment_covers_core_and_plugin_valves():
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.plugin import PipeDashboardPlugin

    live = set(Valves.model_fields) | set(PipeDashboardPlugin.plugin_valves)
    mapped = set(CONFIG_META)
    assert live - mapped == set(), f"unenriched: {sorted(live - mapped)}"
    assert mapped - live == set(), f"orphaned: {sorted(mapped - live)}"


_LABEL_SHAPED = re.compile(r"^[A-Z][A-Za-z-]*(?: [A-Za-z-]+)+$")
_BACKTICKED = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_TITLE_KWARG = re.compile(r"""title=(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")""")
_SHELL_TEXT = re.compile(r">([^<>{}]{2,60})<")

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Names a detail cites that no screen of this pipe's own renders as a label. Each one is
# real and each one is somebody else's, so nothing here can be resolved from a title the
# pipe owns. They are named rather than pattern-matched, and an entry no detail cites any
# more is a failure below, so the map cannot quietly grow into a way of silencing a
# genuinely stale reference.
_NOT_A_SETTING = {
    "Built-in tools": "Open WebUI's own checkbox on the model editor; the pipe writes it but never draws it",
    "File context": "Open WebUI's own checkbox on the model editor; the pipe writes it but never draws it",
    "Video too large": "the opening words of the rejection the request transformer raises, not a control",
}


def _filter_control_titles() -> set[str]:
    """Every control title the generated per-model filters draw, rendered for real.

    These are the per-chat controls an admin is sent to. They exist only in generated
    source, so they are read out of that source rather than listed here: renaming one
    moves the label here in the same commit that moves it on screen.
    """
    import json

    from open_webui_openrouter_pipe.filters.fusion_filter_renderer import (
        render_openrouter_fusion_filter_source,
    )
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_gen_filter_source,
        render_image_model_filter_source,
    )
    from open_webui_openrouter_pipe.filters.video_filter_renderer import render_video_filter_source

    def _titles(source: str) -> set[str]:
        return {single or double for single, double in _TITLE_KWARG.findall(source)}

    found: set[str] = _titles(render_openrouter_fusion_filter_source(marker="xref-probe"))

    contracts = sorted(_FIXTURES.glob("openrouter_image_endpoints_*.json"))
    assert len(contracts) > 30, f"only {len(contracts)} image contracts; this sweep went hollow"
    for path in contracts:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("endpoints") or [raw]
        model_id = raw["id"]
        spec = build_image_model_filter_spec(
            model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True
        )
        found |= _titles(render_image_model_filter_source(spec))
        found |= _titles(
            render_image_gen_filter_source(spec, catalog_match=True, selected_model=model_id)
        )

    catalog = json.loads((_FIXTURES / "video_models_catalog.json").read_text(encoding="utf-8"))
    videos = [m for m in catalog.get("data", []) if isinstance(m, dict) and m.get("id")]
    assert len(videos) > 15, f"only {len(videos)} video models; this sweep went hollow"
    for model in videos:
        found |= _titles(render_video_filter_source(model_id=model["id"], video_model=model))

    return found


def _card_headings() -> set[str]:
    """The heading each built-in error card prints, which details quote by name."""
    from open_webui_openrouter_pipe.core import config as _config

    found: set[str] = set()
    for name in dir(_config):
        if not (name.startswith("DEFAULT_") and name.endswith("_TEMPLATE")):
            continue
        for line in str(getattr(_config, name)).splitlines():
            heading = _HEADING.match(line.strip())
            if heading:
                found.add(re.sub(r"[^\w\s'&-]", " ", heading.group(1)).strip())
    return found


def _visible_labels() -> dict[str, set[str]]:
    """Every name this artifact puts on a screen, by the screen that puts it there."""
    from open_webui_openrouter_pipe.core.config import UserValves
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
        _build_dashboard_shell,
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.plugin import PipeDashboardPlugin

    shell = _build_dashboard_shell("dash-v2")
    sources = {
        "the Config tab": set(CONFIG_META) | {meta["title"] for meta in CONFIG_META.values()},
        "the per-chat controls": {
            f.title for f in UserValves.model_fields.values() if getattr(f, "title", None)
        },
        "the generated model filters": _filter_control_titles(),
        "the error cards": _card_headings(),
        "the dashboard shell": (
            {text.strip() for text in _SHELL_TEXT.findall(shell) if text.strip()}
            | {PipeDashboardPlugin.plugin_name}
        ),
    }
    floors = {
        "the Config tab": 200,
        "the per-chat controls": 5,
        "the generated model filters": 40,
        "the error cards": 5,
        "the dashboard shell": 100,
    }
    for name, labels in sources.items():
        assert len(labels) >= floors[name], (
            f"only {len(labels)} labels came out of {name}; that source went hollow and "
            "every reference it used to answer for now resolves against nothing"
        )
    return sources


def test_no_cross_ref_targets_a_name_no_screen_shows():
    """Every backticked cross-reference in a detail must name something on a screen.

    config_service resolves titles from CONFIG_META and never from the config.py Field
    title, so a ref matching only a stale admin title points at a name the tab never
    renders -- that much was already checked. What was not: a setting renamed on BOTH
    surfaces leaves its old name matching neither, and references to it stayed green.
    A ref is now measured against every screen this artifact puts a name on, so a rename
    has to reach the details that cite it whichever way it was made.

    The screens are read at runtime -- rendered filters, rendered cards, the rendered
    shell -- rather than transcribed, so renaming a control moves the label in the same
    commit that moves it on screen.
    """
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    surfaces = _visible_labels()
    admin_cfg_titles = {
        f.title: name
        for name, f in Valves.model_fields.items()
        if getattr(f, "title", None) and name in CONFIG_META
    }

    broken: list[str] = []
    borrowed: set[str] = set()
    scanned = 0
    for name, meta in CONFIG_META.items():
        for span in sorted(set(_BACKTICKED.findall(meta["detail"]))):
            if any(span in labels for labels in surfaces.values()):
                continue
            if not _LABEL_SHAPED.match(span):
                continue
            scanned += 1
            if span in _NOT_A_SETTING:
                borrowed.add(span)
                continue
            hint = admin_cfg_titles.get(span)
            broken.append(
                f"[{name}] `{span}` -> use `{hint}`"
                if hint
                else f"[{name}] `{span}` -> no screen shows this name"
            )
    assert scanned, (
        "no detail cites a name at all, so this guard is checking nothing; either the "
        f"cross-references were removed or {_LABEL_SHAPED.pattern!r} stopped matching them"
    )
    assert not broken, "Cross-refs point at names no screen shows: " + "; ".join(sorted(broken))

    unused = sorted(set(_NOT_A_SETTING) - borrowed)
    assert not unused, (
        f"no detail cites {unused} any more; drop the entry rather than leaving it "
        "pre-authorising a name nothing uses"
    )

def test_no_detail_has_literal_backslash_n():
    """Config-tab detail blurbs must separate paragraphs with real newlines, never the 2-char
    literal ``\\n`` sequence. The renderer splits on a real newline, so a literal backslash-n
    renders as a visible run-on paragraph with the characters ``\\n`` on screen and collapses
    ``**Tip:**``/``**Warning:**`` callouts. This guards against an apply script re-``json.dumps``-ing
    an already-escaped detail string (double-escaping the separator)."""
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    broken = [name for name, meta in CONFIG_META.items() if "\\n" in meta["detail"]]
    assert not broken, "detail has a literal backslash-n (double-escaped separator): " + ", ".join(broken)
