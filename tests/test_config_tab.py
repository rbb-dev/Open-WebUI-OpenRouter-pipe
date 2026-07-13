"""Config-tab valve editing: every valve survives edit -> save -> round-trip, and a save stores only the custom subset."""

from __future__ import annotations

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


def test_enrichment_covers_core_and_plugin_valves():
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.plugin import PipeDashboardPlugin

    live = set(Valves.model_fields) | set(PipeDashboardPlugin.plugin_valves)
    mapped = set(CONFIG_META)
    assert live - mapped == set(), f"unenriched: {sorted(live - mapped)}"
    assert mapped - live == set(), f"orphaned: {sorted(mapped - live)}"


def test_no_cross_ref_targets_a_non_displayed_title():
    """Every backticked valve cross-reference must resolve to something the admin can see in
    the Config tab: a valve code-name, a displayed config_meta title, or a UserValves title.
    config_service resolves titles from CONFIG_META (never the config.py Field title), so a
    ref that matches only a stale admin Field title points at a name the tab never renders."""
    import re

    from open_webui_openrouter_pipe.core.config import UserValves
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    code_names = set(CONFIG_META)
    displayed_titles = {meta["title"] for meta in CONFIG_META.values()}
    user_titles = {f.title for f in UserValves.model_fields.values() if getattr(f, "title", None)}
    findable = code_names | displayed_titles | user_titles
    admin_cfg_titles = {
        f.title: name
        for name, f in Valves.model_fields.items()
        if getattr(f, "title", None) and name in code_names
    }
    broken = []
    for name, meta in CONFIG_META.items():
        for span in set(re.findall(r"`([^`]+)`", meta["detail"])):
            if span in findable:
                continue
            if span in admin_cfg_titles:
                broken.append(f"[{name}] `{span}` -> use `{admin_cfg_titles[span]}`")
    assert not broken, "Cross-refs point at non-displayed config.py titles: " + "; ".join(broken)


def test_no_detail_has_literal_backslash_n():
    """Config-tab detail blurbs must separate paragraphs with real newlines, never the 2-char
    literal ``\\n`` sequence. The renderer splits on a real newline, so a literal backslash-n
    renders as a visible run-on paragraph with the characters ``\\n`` on screen and collapses
    ``**Tip:**``/``**Warning:**`` callouts. This guards against an apply script re-``json.dumps``-ing
    an already-escaped detail string (double-escaping the separator)."""
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    broken = [name for name, meta in CONFIG_META.items() if "\\n" in meta["detail"]]
    assert not broken, "detail has a literal backslash-n (double-escaped separator): " + ", ".join(broken)
