"""Security regression tests for the video intent classifier."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_webui_openrouter_pipe.core.errors import RequiredInternalFileError
from open_webui_openrouter_pipe.core.utils import (
    _safe_marker_body,
    _serialize_kind_marker,
)
from open_webui_openrouter_pipe.integrations.video_intent import (
    ClarificationPayload,
    FramePlanEntry,
    VideoIntentResult,
    _user_facing_downgrade_message,
    neutralise_control_tokens,
    render_intent_disclosure_block,
)


# -----------------------------------------------------------------------------
# B.4 marker injection — _serialize_kind_marker forbidden chars
# -----------------------------------------------------------------------------

class TestMarkerInjectionGuard:
    def test_rejects_newline_in_body(self):
        with pytest.raises(ValueError, match="forbidden"):
            _serialize_kind_marker("intent_mode", "x\nattack")

    def test_rejects_carriage_return(self):
        with pytest.raises(ValueError, match="forbidden"):
            _serialize_kind_marker("intent_mode", "x\rattack")

    def test_rejects_closing_bracket(self):
        with pytest.raises(ValueError, match="forbidden"):
            _serialize_kind_marker("intent_mode", "x]: #fake")

    def test_rejects_unicode_line_separators(self):
        for ch in (" ", " ", "\x85"):
            with pytest.raises(ValueError, match="forbidden"):
                _serialize_kind_marker("intent_mode", f"x{ch}y")

    def test_rejects_invalid_kind_format(self):
        with pytest.raises(ValueError, match="invalid kind"):
            _serialize_kind_marker("Intent.Mode", "x")  # uppercase + dot
        with pytest.raises(ValueError, match="invalid kind"):
            _serialize_kind_marker("0bad", "x")  # starts with digit

    def test_safe_marker_body_strips_forbidden(self):
        result = _safe_marker_body("hello\nworld]: #attack")
        assert "\n" not in result
        assert "]" not in result

    def test_safe_marker_body_returns_underscore_for_empty(self):
        assert _safe_marker_body("") == "_"
        assert _safe_marker_body("   ") == "_"
        assert _safe_marker_body(None) == "_"  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# B.5 prompt-injection bypass via Unicode lookalikes
# -----------------------------------------------------------------------------

class TestUnicodePromptInjection:
    def test_neutralises_fullwidth_chatml(self):
        # NFKC normalisation should fold fullwidth to ASCII, then regex matches
        text = "normal ＜｜im_start｜＞attack＜｜im_end｜＞"
        result = neutralise_control_tokens(text)
        assert "＜｜im_start｜＞" not in result and "<|im_start|>" not in result

    def test_strips_zero_width_joiners(self):
        text = "<​|im_start|>attack<​|im_end|>"
        result = neutralise_control_tokens(text)
        assert "<|im_start|>" not in result

    def test_neutralises_alt_fence_tildes(self):
        result = neutralise_control_tokens("normal ~~~python\nimport os\n~~~ done")
        assert "~~~" not in result

    def test_preserves_innocuous_content(self):
        text = "what is the meaning of life?"
        assert neutralise_control_tokens(text) == text


# -----------------------------------------------------------------------------
# B.6 information disclosure — downgrade messages are user-friendly
# -----------------------------------------------------------------------------

class TestDowngradeUserFacingMessages:
    def test_known_codes_have_user_messages(self):
        msg = _user_facing_downgrade_message("frame_extract_failed_idx_0")
        assert "exception" not in msg.lower()
        assert "could not" in msg.lower() or "failed" in msg.lower() or "previous" in msg.lower()

    def test_unknown_code_falls_back_to_generic(self):
        msg = _user_facing_downgrade_message("totally_unknown_code_xyz")
        assert msg == "A non-critical step was skipped."

    def test_empty_code_falls_back(self):
        msg = _user_facing_downgrade_message("")
        assert "non-critical" in msg.lower()

    def test_middle_embedded_index_maps_to_curated_message(self):
        """Codes with the index in the MIDDLE (e.g. prior_video_index_0_unresolvable)
        must resolve to their curated message, not fall back to the generic one."""
        msg = _user_facing_downgrade_message("prior_video_index_0_unresolvable")
        assert msg == "Referenced previous video not found."
        msg2 = _user_facing_downgrade_message("prior_video_index_12_unresolvable")
        assert msg2 == "Referenced previous video not found."

    def test_frame_past_eof_code_maps_to_curated_message(self):
        """The frame-extraction end-seek fallback emits the coded note
        frame_past_eof_used_last_frame; it must map to a specific message,
        not the generic fallback."""
        msg = _user_facing_downgrade_message("frame_past_eof_used_last_frame")
        assert msg != "A non-critical step was skipped."
        assert "last frame" in msg.lower()

    def test_disclosure_block_uses_user_facing_messages(self):
        intent = VideoIntentResult(
            intent="modify_prior_video",
            frame_plan=[FramePlanEntry(
                source="prior_video_first_frame", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            prompt="x", use_user_prompt=False, language="en", confidence="high",
            clarification=None, reason="x",
            downgrades=["frame_extract_failed_idx_0"],
        )
        out = render_intent_disclosure_block(intent, thumb_urls=["/api/v1/files/T/content"])
        # The raw code should NOT appear in the user-visible block
        assert "frame_extract_failed_idx_0" not in out.split("[openrouter:v1:intent_block_end")[0].split(">")[-1]


# -----------------------------------------------------------------------------
# A.1 / A.2 path-and-auth checks (smoke tests via mock)
# -----------------------------------------------------------------------------

class TestResolveOwuiFilePath:
    """Gateway-backed `_resolve_owui_file_path`.

    The method now looks the record up via the module-level `get_file_by_id`,
    routes the authorised read through `materialize_owui_file_to_temp` (any
    Storage backend), and returns a PRIVATE TEMP Path the caller owns — never
    the persistent UPLOAD_DIR path. Auth/size/containment failures surface as
    `RequiredInternalFileError`, which the method swallows and degrades to None.
    """

    def _make_adapter(self):
        from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
        pipe = MagicMock()
        pipe.valves.VIDEO_MAX_SIZE_MB = 500
        pipe.valves.ALLOW_UNKNOWN_SIZE_CLOUD_READS = False
        pipe.logger = logging.getLogger("test")
        return VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("test"))

    @pytest.mark.asyncio
    async def test_returns_temp_path_not_persistent_path_on_success(self):
        """Success returns the materialised TEMP path, not the persistent
        `file_obj.path` under UPLOAD_DIR."""
        adapter = self._make_adapter()
        persistent = "/srv/owui/uploads/user_A/legit.mp4"
        temp_path = Path("/tmp/orpipe-read-deadbeef.mp4")
        file_obj = SimpleNamespace(id="file_A", user_id="user_A", path=persistent)
        with patch(
            "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
            AsyncMock(return_value=file_obj),
        ), patch(
            "open_webui_openrouter_pipe.integrations.video.materialize_owui_file_to_temp",
            AsyncMock(return_value=temp_path),
        ) as mat:
            result = await adapter._resolve_owui_file_path(
                file_id="file_A", request=None,
                user_obj=SimpleNamespace(id="user_A"),
            )
        assert result == temp_path
        assert str(result) != persistent
        mat.assert_awaited_once()
        assert mat.await_args is not None
        kwargs = mat.await_args.kwargs
        assert kwargs["user"] == SimpleNamespace(id="user_A")
        assert kwargs["max_bytes"] == 500 * 1024 * 1024
        assert {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"} == kwargs["allowed_suffixes"]

    @pytest.mark.asyncio
    async def test_returns_none_on_required_internal_file_error(self):
        """An auth/size/containment denial from the gateway degrades to None
        (best-effort frame extraction), not a crash."""
        adapter = self._make_adapter()
        file_obj = SimpleNamespace(id="file_A", user_id="user_A", path="/srv/up/x.mp4")
        with patch(
            "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
            AsyncMock(return_value=file_obj),
        ), patch(
            "open_webui_openrouter_pipe.integrations.video.materialize_owui_file_to_temp",
            AsyncMock(side_effect=RequiredInternalFileError("denied", denied=True)),
        ):
            result = await adapter._resolve_owui_file_path(
                file_id="file_A", request=None,
                user_obj=SimpleNamespace(id="user_B"),
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_file_obj_none(self):
        adapter = self._make_adapter()
        with patch(
            "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
            AsyncMock(return_value=None),
        ), patch(
            "open_webui_openrouter_pipe.integrations.video.materialize_owui_file_to_temp",
            AsyncMock(),
        ) as mat:
            result = await adapter._resolve_owui_file_path(
                file_id="x", request=None,
                user_obj=SimpleNamespace(id="user_A"),
            )
        assert result is None
        mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_error(self):
        """Any non-cancellation error is swallowed and degrades to None."""
        adapter = self._make_adapter()
        with patch(
            "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await adapter._resolve_owui_file_path(
                file_id="x", request=None,
                user_obj=SimpleNamespace(id="user_A"),
            )
        assert result is None


# -----------------------------------------------------------------------------
# B.1 ffmpeg argv injection guard
# -----------------------------------------------------------------------------

class TestFfmpegArgvGuard:
    @pytest.mark.asyncio
    async def test_rejects_path_starting_with_dash(self):
        from open_webui_openrouter_pipe.media.frame_extraction import (
            FrameExtractionError, _extract_frame_ffmpeg,
        )
        with pytest.raises(FrameExtractionError, match="argv injection"):
            await _extract_frame_ffmpeg(
                Path("-malicious-flag"),
                timestamp_seconds=0.0,
                logger=logging.getLogger("test"),
            )


# -----------------------------------------------------------------------------
# B.2 decompression bomb / size cap
# -----------------------------------------------------------------------------

class TestDecompressionBomb:
    def test_make_thumbnail_rejects_oversized_input(self):
        from open_webui_openrouter_pipe.media.thumbnail import make_thumbnail
        big = b"x" * (51 * 1024 * 1024)
        with pytest.raises(ValueError, match="too large"):
            make_thumbnail(big)

    def test_make_thumbnail_rejects_invalid_quality(self):
        from open_webui_openrouter_pipe.media.thumbnail import make_thumbnail
        from PIL import Image
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        with pytest.raises(ValueError, match="quality"):
            make_thumbnail(buf.getvalue(), quality=0)
        with pytest.raises(ValueError, match="quality"):
            make_thumbnail(buf.getvalue(), quality=100)
