"""Clearing a template valve box puts the built-in text back, in every artifact that ships.

The restore lives on core `Valves`, so it ships in the package, both plugin-carrying
bundles AND both plugin-free ones. Every guard for it used to live in a file that opens
with `pytest.importorskip(...pipe_dashboard)`, which is one skip in the two plugin-free
bundles -- so neutering the validator there was caught by nothing, and that is precisely
the pair of artifacts where Open WebUI's own valve panel is the ONLY settings surface an
admin has.

Nothing here imports the dashboard plugin, its config service, or its save helper. The
door exercised is Open WebUI's own `POST /functions/id/{id}/valves/update`, reproduced
literally: drop None, build `Valves(**form_data)`, persist `model_dump(exclude_unset=True)`.
The plugin-path variants (merge_for_save, the CONFIG_META cross-check, persistence through
the Config tab) stay in tests/test_config_tab.py, where they belong.
"""
from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.core.config import Valves

_RESTORABLE = ["RATE_LIMIT_TEMPLATE", "CONNECTION_ERROR_TEMPLATE"]
_BLANKS = ["", "   ", "\n\t \n"]


def _factory_text(name):
    return Valves.model_fields[name].get_default(call_default_factory=True)


def _template_valves():
    return sorted(
        name
        for name, field in Valves.model_fields.items()
        if name.endswith("_TEMPLATE")
        and isinstance(field.get_default(call_default_factory=True), str)
    )


def _owui_valves_update(form_data):
    """Open WebUI's own valve-update route, not the pipe's save helper."""
    return Valves(**{k: v for k, v in form_data.items() if v is not None}).model_dump(
        exclude_unset=True
    )


@pytest.mark.parametrize("name", _RESTORABLE)
@pytest.mark.parametrize("blank", _BLANKS, ids=["empty", "spaces", "newlines"])
def test_the_open_webui_valve_panel_restores_a_cleared_template(name, blank):
    """The restore survives the door that does not go through the pipe's save helper.

    An admin who mangles an error template has no other way back: the built-in text lives
    in Python source they cannot read. Two valves with two different factory texts, so no
    single hardcoded string satisfies both, and three spellings of "cleared". An unrelated
    valve in the same payload must come through untouched, so a validator that rebuilt the
    whole model from defaults fails here.
    """
    persisted = _owui_valves_update({name: blank, "MODEL_ID": "anthropic/*"})

    assert persisted[name] == _factory_text(name)
    assert persisted["MODEL_ID"] == "anthropic/*"


@pytest.mark.parametrize("name", _RESTORABLE)
def test_a_real_template_edit_survives_the_open_webui_valve_panel(name):
    """Restoring on blank must not degrade into restoring always.

    A validator that returned the default unconditionally satisfies every blank-input
    assertion above while silently discarding the admin's customisation, so a non-blank
    edit is checked for each valve the restore covers.
    """
    edited = f"## House style for {name}\n{{{{#if error_id}}}}- `{{error_id}}`{{{{/if}}}}\n"

    persisted = _owui_valves_update({name: edited})

    assert persisted[name] == edited


@pytest.mark.parametrize("name", ["BASE_URL", "TIMING_LOG_FILE"])
def test_a_cleared_non_template_string_valve_stays_cleared(name):
    """The restore is scoped to templates; for every other string valve empty means empty.

    Two non-template string valves that both carry a non-empty built-in default, so a rule
    widened to "any string valve with a default" fails here rather than silently ignoring
    an admin who deliberately blanked one.
    """
    persisted = _owui_valves_update({name: ""})

    assert persisted[name] == ""
    assert getattr(Valves(**persisted), name) == ""


def test_every_template_valve_restores_from_whitespace_only_input():
    """The guard grows with the valve set rather than against a copied list.

    Adding a new `*_TEMPLATE` valve must not produce a box an admin can empty and never
    recover. Swept over every field on core Valves, so this needs nothing from the Config
    tab to notice a new one.
    """
    templates = _template_valves()
    assert len(templates) > 1, f"expected several template valves, found {templates}"

    for name in templates:
        restored = Valves.model_validate({name: "   "})
        assert getattr(restored, name) == _factory_text(name), name


def test_a_template_valve_left_out_entirely_is_not_written_back():
    """Blank means "put it back"; absent means "do not store this valve at all".

    Open WebUI persists `exclude_unset=True`, so a validator that injected defaults for
    keys nobody sent would freeze today's built-in text into every deployment's stored
    subset -- and a later release changing that text would never reach them.
    """
    persisted = _owui_valves_update({"MODEL_ID": "anthropic/*"})

    assert persisted == {"MODEL_ID": "anthropic/*"}
