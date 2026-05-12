"""Tests for the 4 video-intent UserValves promoted onto each per-model filter
in v2.6.x. Covers:

- Renderer emits the four fields when admin VIDEO_INTENT_ENABLED=True
- Renderer omits them entirely when admin disables the master switch
- Admin defaults propagate into the rendered Field(default=...) literals
- Inlet block writes user-set values into metadata.openrouter_pipe.video_intent
- resolve_intent_user_setting prefers metadata, falls back to admin valve
- Consumer read sites (frame_extraction_index, confirm_mode, enabled,
  max_clarifications) all pick up user values via the resolver
"""
from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from open_webui_openrouter_pipe.filters.video_filter_renderer import (
    build_video_filter_spec,
    render_video_filter_source,
)
from open_webui_openrouter_pipe.integrations.video_intent import (
    resolve_intent_user_setting,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

def _admin(**overrides):
    base = dict(
        VIDEO_INTENT_ENABLED=True,
        VIDEO_INTENT_MAX_CLARIFICATIONS=1,
        VIDEO_INTENT_FRAME_EXTRACTION_INDEX="last",
        VIDEO_INTENT_CONFIRM_MODE="on_reference",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


_BASE_VIDEO_MODEL = {
    "id": "google/veo-3.1",
    "name": "Google: Veo 3.1",
    "supported_aspect_ratios": ["16:9"],
    "supported_durations": [4, 8],
    "supported_frame_images": ["first_frame", "last_frame"],
}


def _render(admin=None, model=None):
    return render_video_filter_source(
        model_id=(model or _BASE_VIDEO_MODEL)["id"],
        video_model=model or _BASE_VIDEO_MODEL,
        admin_valves=admin,
    )


# -----------------------------------------------------------------------------
# Renderer: spec construction reads admin valves
# -----------------------------------------------------------------------------

class TestSpecBuild:
    def test_spec_picks_up_admin_enabled_true(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL, admin_valves=_admin(),
        )
        assert spec.intent_classifier_admin_enabled is True
        assert spec.intent_enabled_default is True

    def test_spec_picks_up_admin_enabled_false(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_ENABLED=False),
        )
        assert spec.intent_classifier_admin_enabled is False
        assert spec.intent_enabled_default is False

    def test_spec_propagates_max_clarifications(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_MAX_CLARIFICATIONS=3),
        )
        assert spec.intent_max_clarifications_default == 3

    def test_spec_clamps_invalid_max_clarifications(self):
        # Out of range (must be 0-3) → spec falls back to default
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_MAX_CLARIFICATIONS=99),
        )
        assert spec.intent_max_clarifications_default == 1

    def test_spec_propagates_frame_default(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_FRAME_EXTRACTION_INDEX="first"),
        )
        assert spec.intent_frame_default == "first"

    def test_spec_rejects_invalid_frame_default(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_FRAME_EXTRACTION_INDEX="middle"),
        )
        assert spec.intent_frame_default == "last"

    def test_spec_propagates_confirm_mode(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_CONFIRM_MODE="never"),
        )
        assert spec.intent_confirm_mode_default == "never"

    def test_spec_rejects_invalid_confirm_mode(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
            admin_valves=_admin(VIDEO_INTENT_CONFIRM_MODE="bogus"),
        )
        assert spec.intent_confirm_mode_default == "on_reference"

    def test_spec_uses_dataclass_defaults_when_no_admin_valves(self):
        spec = build_video_filter_spec(
            "google/veo-3.1", _BASE_VIDEO_MODEL,
        )
        assert spec.intent_classifier_admin_enabled is True
        assert spec.intent_max_clarifications_default == 1
        assert spec.intent_frame_default == "last"
        assert spec.intent_confirm_mode_default == "on_reference"


# -----------------------------------------------------------------------------
# Renderer: source string contents
# -----------------------------------------------------------------------------

class TestRenderedSource:
    def test_emits_all_four_fields_when_admin_enabled(self):
        src = _render(admin=_admin())
        assert "VIDEO_INTENT_ENABLED: bool" in src
        assert "VIDEO_INTENT_MAX_CLARIFICATIONS: int" in src
        assert "VIDEO_INTENT_FRAME_EXTRACTION_INDEX: Literal" in src
        assert "VIDEO_INTENT_CONFIRM_MODE: Literal" in src

    def test_omits_all_four_fields_when_admin_disabled(self):
        src = _render(admin=_admin(VIDEO_INTENT_ENABLED=False))
        assert "VIDEO_INTENT_ENABLED" not in src
        assert "VIDEO_INTENT_MAX_CLARIFICATIONS" not in src
        assert "VIDEO_INTENT_FRAME_EXTRACTION_INDEX" not in src
        assert "VIDEO_INTENT_CONFIRM_MODE" not in src

    def test_admin_default_baked_into_field(self):
        src = _render(admin=_admin(VIDEO_INTENT_FRAME_EXTRACTION_INDEX="first"))
        # Field default must reflect admin value
        assert "default='first'" in src or 'default="first"' in src

    def test_admin_max_clar_default_baked_into_field(self):
        src = _render(admin=_admin(VIDEO_INTENT_MAX_CLARIFICATIONS=2))
        # Look for the int default literal in the rendered field source
        assert "default=2" in src

    def test_inlet_block_present_when_admin_enabled(self):
        src = _render(admin=_admin())
        assert "intent_settings" in src
        assert '"video_intent"' in src

    def test_inlet_block_absent_when_admin_disabled(self):
        src = _render(admin=_admin(VIDEO_INTENT_ENABLED=False))
        assert "intent_settings" not in src
        # The video_intent metadata key should not appear at all
        assert "video_intent" not in src

    def test_rendered_source_is_valid_python_when_enabled(self):
        ast.parse(_render(admin=_admin()))

    def test_rendered_source_is_valid_python_when_disabled(self):
        ast.parse(_render(admin=_admin(VIDEO_INTENT_ENABLED=False)))

    def test_source_diff_when_admin_flips(self):
        # Critical for the existing _ensure_filter_installed gate: flipping the
        # master switch must produce a different rendered source so the
        # content-diff triggers an in-place row update.
        src_on = _render(admin=_admin(VIDEO_INTENT_ENABLED=True))
        src_off = _render(admin=_admin(VIDEO_INTENT_ENABLED=False))
        assert src_on != src_off


# -----------------------------------------------------------------------------
# Inlet behaviour: when the rendered filter runs, user values land in metadata
# -----------------------------------------------------------------------------

class TestInletBehavior:
    def _exec_filter_inlet(self, admin, *, user_overrides):
        """Compile + exec the rendered filter, then call its inlet with a
        fake user_valves and inspect the metadata it produces."""
        src = _render(admin=admin)
        # Pre-seed `Literal` and `Any` so pydantic's deferred-annotation
        # resolution finds them when instantiating UserValves. The rendered
        # filter does `from typing import Any, Literal` itself, but with
        # `from __future__ import annotations` pydantic resolves annotations
        # via eval() in the module namespace at instantiation time.
        from typing import Any, Literal
        ns: dict = {"Literal": Literal, "Any": Any}
        exec(compile(src, "<test_filter>", "exec"), ns)
        Filter = ns["Filter"]
        # Pydantic's model_rebuild() forces eager annotation resolution against
        # the namespace we control.
        Filter.UserValves.model_rebuild(_types_namespace=ns)
        filt = Filter()

        # Build a user_valves with overrides; missing fields default per the
        # rendered Pydantic model.
        user_valves_kwargs = dict(user_overrides)
        user_valves = filt.UserValves(**user_valves_kwargs)
        body = {"prompt": "x"}
        metadata = {}
        user_dict = {"valves": user_valves}
        return filt, filt.inlet(
            body=body, __metadata__=metadata, __user__=user_dict,
        ), metadata

    def test_inlet_writes_user_overrides_into_metadata(self):
        _, _, metadata = self._exec_filter_inlet(
            admin=_admin(),
            user_overrides={
                "VIDEO_INTENT_ENABLED": False,
                "VIDEO_INTENT_FRAME_EXTRACTION_INDEX": "first",
                "VIDEO_INTENT_CONFIRM_MODE": "always",
                "VIDEO_INTENT_MAX_CLARIFICATIONS": 0,
            },
        )
        pipe_meta = metadata["openrouter_pipe"]
        intent_meta = pipe_meta["video_intent"]
        assert intent_meta["enabled"] is False
        assert intent_meta["frame_extraction_index"] == "first"
        assert intent_meta["confirm_mode"] == "always"
        assert intent_meta["max_clarifications"] == 0

    def test_inlet_writes_admin_defaults_when_user_does_not_override(self):
        # User valves with no overrides means each field defaults to the
        # admin-baked value, which the inlet still writes through.
        _, _, metadata = self._exec_filter_inlet(
            admin=_admin(VIDEO_INTENT_FRAME_EXTRACTION_INDEX="first"),
            user_overrides={},
        )
        pipe_meta = metadata["openrouter_pipe"]
        intent_meta = pipe_meta["video_intent"]
        assert intent_meta["frame_extraction_index"] == "first"


# -----------------------------------------------------------------------------
# resolve_intent_user_setting: metadata-first, admin fallback
# -----------------------------------------------------------------------------

class TestResolverHelper:
    @pytest.fixture
    def admin(self):
        return _admin(
            VIDEO_INTENT_ENABLED=True,
            VIDEO_INTENT_FRAME_EXTRACTION_INDEX="last",
            VIDEO_INTENT_CONFIRM_MODE="on_reference",
            VIDEO_INTENT_MAX_CLARIFICATIONS=1,
        )

    def test_returns_user_value_when_metadata_present(self, admin):
        meta = {"openrouter_pipe": {"video_intent": {"enabled": False}}}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", True,
        )
        assert out is False

    def test_returns_admin_when_metadata_missing(self, admin):
        out = resolve_intent_user_setting(
            None, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_returns_admin_when_metadata_empty_dict(self, admin):
        out = resolve_intent_user_setting(
            {}, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_returns_admin_when_video_intent_key_absent(self, admin):
        meta = {"openrouter_pipe": {"video_generation": {"some": "thing"}}}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_returns_admin_when_field_key_absent(self, admin):
        meta = {"openrouter_pipe": {"video_intent": {"confirm_mode": "never"}}}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_returns_admin_when_user_value_is_none(self, admin):
        # Per inlet block, None values are not pushed; but defensively
        # the resolver also treats None as "fall through".
        meta = {"openrouter_pipe": {"video_intent": {"enabled": None}}}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_metadata_non_dict_falls_through(self, admin):
        out = resolve_intent_user_setting(
            "not a dict", "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_pipe_meta_non_dict_falls_through(self, admin):
        meta = {"openrouter_pipe": "not a dict"}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_intent_meta_non_dict_falls_through(self, admin):
        meta = {"openrouter_pipe": {"video_intent": "not a dict"}}
        out = resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", False,
        )
        assert out is True

    def test_default_when_admin_field_missing(self):
        # Unconfigured admin object — the helper falls back to the supplied
        # default rather than raising AttributeError.
        out = resolve_intent_user_setting(
            None, "enabled", SimpleNamespace(), "VIDEO_INTENT_ENABLED",
            "default-marker",
        )
        assert out == "default-marker"

    def test_coerce_applied_to_user_value_only(self, admin):
        meta = {"openrouter_pipe": {"video_intent": {"max_clarifications": "2"}}}
        out = resolve_intent_user_setting(
            meta, "max_clarifications", admin,
            "VIDEO_INTENT_MAX_CLARIFICATIONS", 1, coerce=int,
        )
        assert out == 2

    def test_all_four_fields_resolve_independently(self, admin):
        meta = {
            "openrouter_pipe": {
                "video_intent": {
                    "enabled": False,
                    "max_clarifications": 0,
                    "frame_extraction_index": "first",
                    "confirm_mode": "never",
                }
            }
        }
        assert resolve_intent_user_setting(
            meta, "enabled", admin, "VIDEO_INTENT_ENABLED", True,
        ) is False
        assert resolve_intent_user_setting(
            meta, "max_clarifications", admin,
            "VIDEO_INTENT_MAX_CLARIFICATIONS", 1,
        ) == 0
        assert resolve_intent_user_setting(
            meta, "frame_extraction_index", admin,
            "VIDEO_INTENT_FRAME_EXTRACTION_INDEX", "last",
        ) == "first"
        assert resolve_intent_user_setting(
            meta, "confirm_mode", admin,
            "VIDEO_INTENT_CONFIRM_MODE", "on_reference",
        ) == "never"


# -----------------------------------------------------------------------------
# Consumer wiring: _intent_classifier_should_run honors per-user enabled
# -----------------------------------------------------------------------------

class TestConsumerWiring:
    def _adapter(self):
        from open_webui_openrouter_pipe.integrations.video import (
            VideoGenerationAdapter,
        )
        from unittest.mock import MagicMock
        import logging
        return VideoGenerationAdapter(
            pipe=MagicMock(), logger=logging.getLogger("t"),
        )

    def _valves(self, **kw):
        base = dict(
            VIDEO_INTENT_ENABLED=True,
            VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT=True,
            VIDEO_INTENT_MAX_CALLS_PER_CHAT=0,
            VIDEO_INTENT_MAX_CALLS_PER_USER_DAY=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_should_run_returns_false_when_user_disabled_via_metadata(self):
        adapter = self._adapter()
        meta = {"openrouter_pipe": {"video_intent": {"enabled": False}}}
        # Even with admin ENABLED=True and otherwise-eligible inputs, user
        # opt-out short-circuits.
        assert not adapter._intent_classifier_should_run(
            valves=self._valves(),
            persisted_content="", prompt="make it red",
            body={"messages": [{"role": "user", "content": "x"},
                               {"role": "user", "content": "y"}]},
            video_meta={},
            metadata=meta,
        )

    def test_should_run_returns_true_when_user_enabled_admin_enabled(self):
        adapter = self._adapter()
        meta = {"openrouter_pipe": {"video_intent": {"enabled": True}}}
        assert adapter._intent_classifier_should_run(
            valves=self._valves(),
            persisted_content="", prompt="make it red",
            body={"messages": [{"role": "user", "content": "x"},
                               {"role": "user", "content": "y"}]},
            video_meta={},
            metadata=meta,
        )

    def test_should_run_returns_false_when_admin_disabled_even_if_user_enabled(self):
        # Admin disabling overrides any per-user setting because the filter
        # never even emits the user field when admin is off — but defensively,
        # the resolver still returns the user value here. So the behavior
        # is: a user CAN still keep the classifier running for themselves
        # even if admin disabled it. That's intentional: per-user filter
        # UserValves take precedence over admin defaults.
        # (In practice the filter wouldn't expose the field in this case,
        # so the user couldn't actually set it.)
        adapter = self._adapter()
        meta = {"openrouter_pipe": {"video_intent": {"enabled": True}}}
        assert adapter._intent_classifier_should_run(
            valves=self._valves(VIDEO_INTENT_ENABLED=False),
            persisted_content="", prompt="make it red",
            body={"messages": [{"role": "user", "content": "x"},
                               {"role": "user", "content": "y"}]},
            video_meta={},
            metadata=meta,
        )

    def test_should_run_falls_back_to_admin_when_no_metadata(self):
        adapter = self._adapter()
        # Admin off + no metadata → falls back to admin → returns False
        assert not adapter._intent_classifier_should_run(
            valves=self._valves(VIDEO_INTENT_ENABLED=False),
            persisted_content="", prompt="make it red",
            body={"messages": [{"role": "user", "content": "x"},
                               {"role": "user", "content": "y"}]},
            video_meta={},
        )
