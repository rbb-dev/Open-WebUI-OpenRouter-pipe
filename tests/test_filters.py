"""Integration tests for filter template strings from FilterManager.

These tests load filters from FilterManager.render_*_filter_source() methods
to ensure we test the ACTUAL code that gets deployed, not static backup copies.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest
from aioresponses import aioresponses
from pydantic import BaseModel

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.filters import FilterManager


def _load_filter_from_source(source: str, module_name: str) -> ModuleType:
    """Load filter source string as a module.

    This tests the ACTUAL code that FilterManager deploys to Open WebUI,
    not static backup copies in /filters/.
    """
    import sys

    # Mock open_webui.env if not available (test environment)
    if "open_webui" not in sys.modules:
        open_webui_mock = ModuleType("open_webui")
        sys.modules["open_webui"] = open_webui_mock

    if "open_webui.env" not in sys.modules:
        env_mock = ModuleType("open_webui.env")
        env_mock.SRC_LOG_LEVELS = {}  # type: ignore
        sys.modules["open_webui.env"] = env_mock

    # Create a module namespace
    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}_rendered_source>"

    # Add to sys.modules so nested imports can resolve
    sys.modules[module_name] = module

    # Execute the source in the module namespace
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)

    # Rebuild Pydantic models to resolve forward references (Literal types)
    if hasattr(module, "Filter"):
        if hasattr(module.Filter, "UserValves"):
            module.Filter.UserValves.model_rebuild()
        if hasattr(module.Filter, "Valves"):
            module.Filter.Valves.model_rebuild()

    return module


# Load the Direct Uploads filter from rendered template
_direct_uploads_source = FilterManager.render_direct_uploads_filter_source()
_direct_uploads_module = _load_filter_from_source(_direct_uploads_source, "direct_uploads_filter")
Filter = _direct_uploads_module.Filter


# ============================================================================
# Filter Initialization Tests
# ============================================================================


def test_filter_initializes_with_default_valves():
    """Test Filter initializes with proper default valves."""
    filt = Filter()

    assert filt.toggle is True
    assert hasattr(filt, 'valves')
    assert isinstance(filt.valves, Filter.Valves)
    assert filt.valves.priority == 0
    assert filt.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB == 50
    assert filt.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB == 50
    assert filt.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB == 25
    assert filt.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB == 20


def test_filter_user_valves_defaults():
    """Test Filter UserValves has proper defaults."""
    user_valves = Filter.UserValves()

    assert user_valves.DIRECT_FILES is False
    assert user_valves.DIRECT_AUDIO is False
    assert user_valves.DIRECT_VIDEO is False


def test_filter_user_valves_has_pdf_parser():
    """Test Filter UserValves has DIRECT_PDF_PARSER field."""
    user_valves = Filter.UserValves()

    assert hasattr(user_valves, 'DIRECT_PDF_PARSER')
    assert user_valves.DIRECT_PDF_PARSER == "Native"

    # Test other values
    user_valves_pdf = Filter.UserValves(DIRECT_PDF_PARSER="PDF Text")
    assert user_valves_pdf.DIRECT_PDF_PARSER == "PDF Text"

    user_valves_ocr = Filter.UserValves(DIRECT_PDF_PARSER="Mistral OCR")
    assert user_valves_ocr.DIRECT_PDF_PARSER == "Mistral OCR"


# ============================================================================
# Helper Method Tests (_to_int)
# ============================================================================


def test_to_int_with_none():
    """Test _to_int returns None for None input."""
    assert Filter._to_int(None) is None


def test_to_int_with_bool():
    """Test _to_int returns None for boolean input."""
    assert Filter._to_int(True) is None
    assert Filter._to_int(False) is None


def test_to_int_with_int():
    """Test _to_int passes through integers."""
    assert Filter._to_int(42) == 42
    assert Filter._to_int(0) == 0
    assert Filter._to_int(-5) == -5


def test_to_int_with_float():
    """Test _to_int converts floats to int."""
    assert Filter._to_int(3.7) == 3
    assert Filter._to_int(10.0) == 10


def test_to_int_with_valid_string():
    """Test _to_int converts valid numeric strings."""
    assert Filter._to_int("123") == 123
    assert Filter._to_int("  456  ") == 456
    assert Filter._to_int("-789") == -789


def test_to_int_with_empty_string():
    """Test _to_int returns None for empty string."""
    assert Filter._to_int("") is None
    assert Filter._to_int("   ") is None


def test_to_int_with_invalid_string():
    """Test _to_int returns None for non-numeric strings."""
    assert Filter._to_int("abc") is None
    assert Filter._to_int("12.34") is None  # Not pure int format


def test_to_int_with_other_types():
    """Test _to_int returns None for unsupported types."""
    assert Filter._to_int([1, 2, 3]) is None
    assert Filter._to_int({"value": 42}) is None


# ============================================================================
# Helper Method Tests (_csv_set)
# ============================================================================


def test_csv_set_with_valid_csv():
    """Test _csv_set parses comma-separated values."""
    result = Filter._csv_set("pdf,txt,md")
    assert result == {"pdf", "txt", "md"}


def test_csv_set_with_whitespace():
    """Test _csv_set strips whitespace from values."""
    result = Filter._csv_set("  pdf , txt ,  md  ")
    assert result == {"pdf", "txt", "md"}


def test_csv_set_with_empty_values():
    """Test _csv_set ignores empty values."""
    result = Filter._csv_set("pdf,,txt,,,md")
    assert result == {"pdf", "txt", "md"}


def test_csv_set_lowercases():
    """Test _csv_set lowercases all values."""
    result = Filter._csv_set("PDF,TXT,Md")
    assert result == {"pdf", "txt", "md"}


def test_csv_set_with_empty_string():
    """Test _csv_set returns empty set for empty string."""
    result = Filter._csv_set("")
    assert result == set()


def test_csv_set_with_non_string():
    """Test _csv_set returns empty set for non-string input."""
    assert Filter._csv_set(None) == set()
    assert Filter._csv_set(123) == set()
    assert Filter._csv_set(["a", "b"]) == set()


# ============================================================================
# Helper Method Tests (_mime_allowed)
# ============================================================================


def test_mime_allowed_exact_match():
    """Test _mime_allowed with exact MIME type match."""
    assert Filter._mime_allowed("application/pdf", "application/pdf,text/plain") is True


def test_mime_allowed_with_wildcard():
    """Test _mime_allowed with wildcard patterns."""
    assert Filter._mime_allowed("audio/mp3", "audio/*") is True
    assert Filter._mime_allowed("audio/wav", "audio/*") is True
    assert Filter._mime_allowed("video/mp4", "audio/*") is False


def test_mime_allowed_case_insensitive():
    """Test _mime_allowed is case insensitive."""
    assert Filter._mime_allowed("APPLICATION/PDF", "application/pdf") is True
    assert Filter._mime_allowed("Audio/MP3", "audio/*") is True


def test_mime_allowed_with_empty_mime():
    """Test _mime_allowed returns False for empty MIME."""
    assert Filter._mime_allowed("", "application/pdf") is False
    assert Filter._mime_allowed("   ", "application/pdf") is False


def test_mime_allowed_with_empty_allowlist():
    """Test _mime_allowed returns False for empty allowlist."""
    assert Filter._mime_allowed("application/pdf", "") is False


def test_mime_allowed_no_match():
    """Test _mime_allowed returns False when no match."""
    assert Filter._mime_allowed("video/mp4", "audio/*,application/pdf") is False


# ============================================================================
# Helper Method Tests (_infer_audio_format)
# ============================================================================


def test_infer_audio_format_from_mime_wav():
    """Test _infer_audio_format detects wav from MIME."""
    assert Filter._infer_audio_format("file.xyz", "audio/wav") == "wav"
    assert Filter._infer_audio_format("file.xyz", "audio/wave") == "wav"
    assert Filter._infer_audio_format("file.xyz", "audio/x-wav") == "wav"


def test_infer_audio_format_from_mime_mp3():
    """Test _infer_audio_format detects mp3 from MIME."""
    assert Filter._infer_audio_format("file.xyz", "audio/mpeg") == "mp3"
    assert Filter._infer_audio_format("file.xyz", "audio/mp3") == "mp3"


def test_infer_audio_format_from_extension():
    """Test _infer_audio_format falls back to filename extension."""
    assert Filter._infer_audio_format("song.flac", "audio/unknown") == "flac"
    assert Filter._infer_audio_format("voice.m4a", "audio/x-m4a") == "m4a"
    assert Filter._infer_audio_format("music.ogg", "") == "ogg"


def test_infer_audio_format_no_extension():
    """Test _infer_audio_format with no extension returns empty."""
    assert Filter._infer_audio_format("noextension", "") == ""
    assert Filter._infer_audio_format("", "") == ""


def test_infer_audio_format_case_insensitive():
    """Test _infer_audio_format is case insensitive."""
    assert Filter._infer_audio_format("file.MP3", "") == "mp3"
    assert Filter._infer_audio_format("file.xyz", "AUDIO/WAV") == "wav"


def test_infer_audio_format_with_non_string():
    """Test _infer_audio_format handles non-string inputs."""
    assert Filter._infer_audio_format(None, None) == ""
    assert Filter._infer_audio_format(123, "audio/mp3") == "mp3"


# ============================================================================
# Helper Method Tests (_model_caps)
# ============================================================================


def test_model_caps_with_full_structure():
    """Test _model_caps extracts capabilities from proper structure."""
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {
                        "file_input": True,
                        "audio_input": False,
                        "video_input": True,
                    }
                }
            }
        }
    }
    caps = Filter._model_caps(model)
    assert caps == {"file_input": True, "audio_input": False, "video_input": True}


def test_model_caps_with_missing_structure():
    """Test _model_caps returns empty dict for missing structure."""
    assert Filter._model_caps(None) == {}
    assert Filter._model_caps({}) == {}
    assert Filter._model_caps({"info": {}}) == {}
    assert Filter._model_caps({"info": {"meta": {}}}) == {}
    assert Filter._model_caps({"info": {"meta": {"openrouter_pipe": {}}}}) == {}


def test_model_caps_with_non_dict():
    """Test _model_caps returns empty dict for non-dict inputs."""
    assert Filter._model_caps("string") == {}
    assert Filter._model_caps([1, 2, 3]) == {}
    assert Filter._model_caps({"info": {"meta": {"openrouter_pipe": "not a dict"}}}) == {}


# ============================================================================
# Inlet Tests - Basic Validation
# ============================================================================


def test_inlet_returns_body_if_not_dict():
    """Test inlet returns body unchanged if not a dict."""
    filt = Filter()
    result = filt.inlet("not a dict")
    assert result == "not a dict"


def test_inlet_returns_body_if_metadata_not_dict():
    """Test inlet returns body unchanged if metadata is not dict."""
    filt = Filter()
    body = {"files": []}
    result = filt.inlet(body, __metadata__="not a dict")
    assert result == body


def test_inlet_returns_body_if_no_files():
    """Test inlet returns body unchanged if no files."""
    filt = Filter()
    body = {"messages": []}
    metadata = {}
    result = filt.inlet(body, __metadata__=metadata)
    assert result == body


def test_inlet_returns_body_if_files_empty():
    """Test inlet returns body unchanged if files list is empty."""
    filt = Filter()
    body = {"files": []}
    metadata = {}
    result = filt.inlet(body, __metadata__=metadata)
    assert result == body


# ============================================================================
# Inlet Tests - File Diversion
# ============================================================================


def test_inlet_diverts_files_when_enabled():
    """Test inlet diverts files to direct uploads when enabled."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Files should be removed from body
    assert result["files"] == []

    # Diverted files should be in metadata
    pipe_meta = metadata.get("openrouter_pipe", {})
    direct_uploads = pipe_meta.get("direct_uploads", {})
    assert len(direct_uploads.get("files", [])) == 1
    assert direct_uploads["files"][0]["id"] == "file_1"


def test_inlet_records_pdf_parser_for_diverted_pdfs():
    """Test inlet records selected PDF parser when diverting PDFs."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True, DIRECT_PDF_PARSER="Mistral OCR")}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    pipe_meta = metadata.get("openrouter_pipe", {})
    direct_uploads = pipe_meta.get("direct_uploads", {})
    assert direct_uploads.get("pdf_parser") == "Mistral OCR"


def test_inlet_retains_files_when_user_valve_disabled():
    """Test inlet retains files when DIRECT_FILES is disabled."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=False)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Files should remain in body
    assert len(result["files"]) == 1


def test_inlet_retains_files_when_model_lacks_capability():
    """Test inlet retains files when model doesn't support file input."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": False}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Files should remain in body
    assert len(result["files"]) == 1

    # Warning should be added
    pipe_meta = metadata.get("openrouter_pipe", {})
    warnings = pipe_meta.get("direct_uploads_warnings", [])
    assert any("not supported" in str(w) for w in warnings)


def test_inlet_retains_files_with_unsupported_mime():
    """Test inlet retains files with unsupported MIME types."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.docx",
            "size": 1024,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Files should remain (fail open for unsupported MIME)
    assert len(result["files"]) == 1


# ============================================================================
# Inlet Tests - Audio Diversion
# ============================================================================


def test_inlet_diverts_audio_when_enabled():
    """Test inlet diverts audio files when enabled."""
    filt = Filter()

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.mp3",
            "size": 2048,
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Audio should be removed from body
    assert result["files"] == []

    # Diverted audio should be in metadata
    pipe_meta = metadata.get("openrouter_pipe", {})
    direct_uploads = pipe_meta.get("direct_uploads", {})
    assert len(direct_uploads.get("audio", [])) == 1
    assert direct_uploads["audio"][0]["id"] == "audio_1"
    assert direct_uploads["audio"][0]["format"] == "mp3"


def test_inlet_retains_audio_when_format_not_allowed():
    """Test inlet retains audio when format is not in allowlist."""
    filt = Filter()
    # Set a restrictive allowlist
    filt.valves.DIRECT_AUDIO_FORMAT_ALLOWLIST = "wav"

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.xyz",  # Unknown format
            "size": 2048,
            "content_type": "audio/xyz",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Audio should remain in body (unsupported format)
    assert len(result["files"]) == 1


# ============================================================================
# Inlet Tests - Video Diversion
# ============================================================================


def test_inlet_diverts_video_when_enabled():
    """Test inlet diverts video files when enabled."""
    filt = Filter()

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "clip.mp4",
            "size": 4096,
            "content_type": "video/mp4",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Video should be removed from body
    assert result["files"] == []

    # Diverted video should be in metadata
    pipe_meta = metadata.get("openrouter_pipe", {})
    direct_uploads = pipe_meta.get("direct_uploads", {})
    assert len(direct_uploads.get("video", [])) == 1
    assert direct_uploads["video"][0]["id"] == "video_1"


def test_inlet_retains_video_with_unsupported_mime():
    """Test inlet retains video with unsupported MIME type."""
    filt = Filter()

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "clip.avi",
            "size": 4096,
            "content_type": "video/avi",  # Not in default allowlist
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Video should remain (unsupported MIME)
    assert len(result["files"]) == 1


# ============================================================================
# Inlet Tests - Size Limits
# ============================================================================


def test_inlet_raises_on_file_too_large():
    """Test inlet raises exception when file exceeds size limit."""
    filt = Filter()
    filt.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB = 1  # 1MB limit

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "large.pdf",
            "size": 2 * 1024 * 1024,  # 2MB
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="too large"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_raises_on_audio_too_large():
    """Test inlet raises exception when audio exceeds size limit."""
    filt = Filter()
    filt.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB = 1  # 1MB limit

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "large.mp3",
            "size": 2 * 1024 * 1024,  # 2MB
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="too large"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_raises_on_video_too_large():
    """Test inlet raises exception when video exceeds size limit."""
    filt = Filter()
    filt.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB = 1  # 1MB limit

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "large.mp4",
            "size": 2 * 1024 * 1024,  # 2MB
            "content_type": "video/mp4",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="too large"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_raises_on_total_payload_exceeded():
    """Test inlet raises when total payload exceeds limit."""
    filt = Filter()
    filt.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB = 1  # 1MB total limit
    filt.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB = 10  # Individual limit is higher

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc1.pdf",
            "size": 600 * 1024,  # 600KB
            "content_type": "application/pdf",
        },
        {
            "id": "file_2",
            "type": "file",
            "name": "doc2.pdf",
            "size": 600 * 1024,  # 600KB (total 1.2MB)
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="exceed total limit"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_raises_on_missing_file_size():
    """Test inlet raises when file is missing size."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc.pdf",
            # No size field
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="missing a valid size"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


# ============================================================================
# Inlet Tests - Edge Cases
# ============================================================================


def test_inlet_skips_legacy_files():
    """Test inlet skips files with legacy=True."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "legacy.pdf",
            "size": 1024,
            "content_type": "application/pdf",
            "legacy": True,
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Legacy files should remain
    assert len(result["files"]) == 1


def test_inlet_skips_non_file_type():
    """Test inlet skips items with type != 'file'."""
    filt = Filter()

    files = [
        {
            "id": "item_1",
            "type": "image",  # Not 'file'
            "name": "photo.jpg",
            "size": 1024,
            "content_type": "image/jpeg",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Non-file type should remain
    assert len(result["files"]) == 1


def test_inlet_skips_items_without_valid_id():
    """Test inlet skips items without valid string id."""
    filt = Filter()

    files = [
        {
            "id": "",  # Empty id
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        },
        {
            "id": 123,  # Non-string id
            "type": "file",
            "name": "doc2.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Invalid id items should remain unchanged
    assert len(result["files"]) == 2


def test_inlet_handles_non_dict_items_in_files():
    """Test inlet handles non-dict items in files list."""
    filt = Filter()

    files = ["not a dict", 123, None]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {}

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Non-dict items should remain
    assert len(result["files"]) == 3


def test_inlet_handles_user_not_dict():
    """Test inlet handles non-dict __user__."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}

    # __user__ is not a dict
    result = filt.inlet(body, __metadata__=metadata, __user__="not a dict", __model__={})

    # Should still process (with default user valves)
    assert "files" in result


def test_inlet_handles_user_valves_not_basemodel():
    """Test inlet handles user valves that's not a BaseModel."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": "not a BaseModel"}  # Invalid valves

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__={})

    # Should use default UserValves
    assert "files" in result


def test_inlet_updates_metadata_files():
    """Test inlet updates metadata['files'] to match body['files']."""
    filt = Filter()

    diverted = {
        "id": "file_1",
        "type": "file",
        "name": "doc.pdf",
        "size": 1024,
        "content_type": "application/pdf",
    }
    retained = {
        "id": "file_2",
        "type": "file",
        "name": "doc.docx",
        "size": 1024,
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    files = [diverted, retained]
    metadata = {"files": files}  # Same reference
    body = {"files": files}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Both body and metadata should have the retained file
    assert len(result["files"]) == 1
    assert len(metadata["files"]) == 1


def test_inlet_merges_existing_direct_uploads():
    """Test inlet merges with existing direct_uploads in metadata."""
    filt = Filter()

    # Pre-existing direct upload
    existing_file = {"id": "existing_1", "name": "old.pdf", "size": 500, "content_type": "application/pdf"}

    new_file = {
        "id": "file_1",
        "type": "file",
        "name": "new.pdf",
        "size": 1024,
        "content_type": "application/pdf",
    }

    body = {"files": [new_file]}
    metadata = {
        "openrouter_pipe": {
            "direct_uploads": {
                "files": [existing_file]
            }
        }
    }
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Should have both existing and new files
    direct_uploads = metadata["openrouter_pipe"]["direct_uploads"]
    assert len(direct_uploads["files"]) == 2


def test_inlet_deduplicates_by_id():
    """Test inlet deduplicates files by id."""
    filt = Filter()

    # Pre-existing direct upload with same id
    existing_file = {"id": "file_1", "name": "old.pdf", "size": 500, "content_type": "application/pdf"}

    new_file = {
        "id": "file_1",  # Same id
        "type": "file",
        "name": "new.pdf",
        "size": 1024,
        "content_type": "application/pdf",
    }

    body = {"files": [new_file]}
    metadata = {
        "openrouter_pipe": {
            "direct_uploads": {
                "files": [existing_file]
            }
        }
    }
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Should have only one file (deduplicated)
    direct_uploads = metadata["openrouter_pipe"]["direct_uploads"]
    assert len(direct_uploads["files"]) == 1


def test_inlet_persists_responses_audio_format_allowlist():
    """Test inlet persists responses audio format allowlist in metadata."""
    filt = Filter()
    filt.valves.DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST = "wav,mp3"

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.mp3",
            "size": 1024,
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Should persist the allowlist
    direct_uploads = metadata["openrouter_pipe"]["direct_uploads"]
    assert direct_uploads["responses_audio_format_allowlist"] == "wav,mp3"


# ============================================================================
# Additional Coverage Tests
# ============================================================================


def test_model_caps_with_meta_not_dict():
    """Test _model_caps returns empty when info.meta is not a dict."""
    model = {
        "info": {
            "meta": "not a dict"
        }
    }
    caps = Filter._model_caps(model)
    assert caps == {}


def test_inlet_retains_audio_when_user_valve_disabled():
    """Test inlet retains audio when DIRECT_AUDIO is disabled (lines 295-296)."""
    filt = Filter()

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.mp3",
            "size": 2048,
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=False)}  # Disabled
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Audio should remain in body
    assert len(result["files"]) == 1


def test_inlet_retains_audio_when_model_lacks_capability():
    """Test inlet retains audio when model doesn't support audio (lines 298-300)."""
    filt = Filter()

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.mp3",
            "size": 2048,
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": False}  # No audio support
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Audio should remain
    assert len(result["files"]) == 1

    # Warning should be added
    pipe_meta = metadata.get("openrouter_pipe", {})
    warnings = pipe_meta.get("direct_uploads_warnings", [])
    assert any("audio uploads not supported" in str(w).lower() for w in warnings)


def test_inlet_retains_audio_with_unsupported_mime():
    """Test inlet retains audio with unsupported MIME type (lines 302-303)."""
    filt = Filter()
    # Default audio MIME allowlist is "audio/*" so let's change it
    filt.valves.DIRECT_AUDIO_MIME_ALLOWLIST = "audio/wav"  # Only wav

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "recording.mp3",
            "size": 2048,
            "content_type": "audio/mp3",  # Not in allowlist
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Audio should remain (MIME not allowed)
    assert len(result["files"]) == 1


def test_inlet_raises_on_audio_total_payload_exceeded():
    """Test inlet raises when audio total payload exceeds limit (line 314)."""
    filt = Filter()
    filt.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB = 1  # 1MB total limit
    filt.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB = 10  # Individual limit higher

    files = [
        {
            "id": "audio_1",
            "type": "file",
            "name": "song1.mp3",
            "size": 600 * 1024,  # 600KB
            "content_type": "audio/mp3",
        },
        {
            "id": "audio_2",
            "type": "file",
            "name": "song2.mp3",
            "size": 600 * 1024,  # 600KB (total 1.2MB)
            "content_type": "audio/mp3",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"audio_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="exceed total limit"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_retains_video_when_user_valve_disabled():
    """Test inlet retains video when DIRECT_VIDEO is disabled (lines 329-331)."""
    filt = Filter()

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "clip.mp4",
            "size": 4096,
            "content_type": "video/mp4",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=False)}  # Disabled
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": True}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Video should remain in body
    assert len(result["files"]) == 1


def test_inlet_retains_video_when_model_lacks_capability():
    """Test inlet retains video when model doesn't support video (lines 333-335)."""
    filt = Filter()

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "clip.mp4",
            "size": 4096,
            "content_type": "video/mp4",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": False}  # No video support
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Video should remain
    assert len(result["files"]) == 1

    # Warning should be added
    pipe_meta = metadata.get("openrouter_pipe", {})
    warnings = pipe_meta.get("direct_uploads_warnings", [])
    assert any("video uploads not supported" in str(w).lower() for w in warnings)


def test_inlet_raises_on_video_total_payload_exceeded():
    """Test inlet raises when video total payload exceeds limit (line 345)."""
    filt = Filter()
    filt.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB = 1  # 1MB total limit
    filt.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB = 10  # Individual limit higher

    files = [
        {
            "id": "video_1",
            "type": "file",
            "name": "clip1.mp4",
            "size": 600 * 1024,  # 600KB
            "content_type": "video/mp4",
        },
        {
            "id": "video_2",
            "type": "file",
            "name": "clip2.mp4",
            "size": 600 * 1024,  # 600KB (total 1.2MB)
            "content_type": "video/mp4",
        }
    ]
    body = {"files": list(files)}
    metadata = {}
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"video_input": True}
                }
            }
        }
    }

    with pytest.raises(Exception, match="exceed total limit"):
        filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)


def test_inlet_merges_existing_warnings():
    """Test inlet merges with existing warnings in metadata (lines 379-382)."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    # Pre-existing warning
    metadata = {
        "openrouter_pipe": {
            "direct_uploads_warnings": ["Previous warning"]
        }
    }
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    # Model lacks capability - generates new warning
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": False}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Both warnings should be present
    warnings = metadata["openrouter_pipe"]["direct_uploads_warnings"]
    assert "Previous warning" in warnings
    assert any("not supported" in str(w) for w in warnings)


def test_inlet_deduplicates_warnings():
    """Test inlet deduplicates duplicate warnings."""
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    # Pre-existing warning that matches the one that will be generated
    metadata = {
        "openrouter_pipe": {
            "direct_uploads_warnings": ["Direct file uploads not supported by the selected model; falling back to Open WebUI."]
        }
    }
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": False}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Should only have one warning (deduplicated)
    warnings = metadata["openrouter_pipe"]["direct_uploads_warnings"]
    assert len(warnings) == 1


# ============================================================================
# Full Integration Tests - Filter + Pipe
# ============================================================================


@pytest.mark.asyncio
async def test_filter_integration_with_pipe_direct_uploads(pipe_instance_async):
    """Test filter output integrates correctly with Pipe processing."""
    pipe = pipe_instance_async

    # Prepare filter inputs
    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Analyze this document"}],
        "stream": False,
        "files": list(files),
    }
    metadata = {"model": {"id": "openai/gpt-4o"}}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": True}
                }
            }
        }
    }

    # Create filter and apply
    filt = Filter()
    filtered_body = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Verify filter worked
    assert filtered_body["files"] == []
    assert "direct_uploads" in metadata.get("openrouter_pipe", {})

    # Now pass to pipe with aioresponses mock
    with aioresponses() as mock_http:
        # Mock catalog
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [
                {"id": "openai/gpt-4o", "name": "GPT-4o", "context_length": 128000}
            ]},
        )

        # Mock chat completion response
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I see the document."}],
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

        valves = pipe.valves
        session = pipe._create_http_session(valves)

        try:
            result = await pipe._handle_pipe_call(
                filtered_body,
                __user__=user,
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=metadata,
                __tools__=None,
                __task__=None,
                __task_body__=None,
                valves=valves,
                session=session,
            )

            assert isinstance(result, dict)
            # Verify pipe processed the request
            choices = result.get("choices", [])
            assert len(choices) > 0
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_filter_warnings_passed_to_pipe(pipe_instance_async):
    """Test filter warnings are available to pipe processing."""
    pipe = pipe_instance_async

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "document.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        }
    ]
    body = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Analyze this"}],
        "stream": False,
        "files": list(files),
    }
    metadata = {"model": {"id": "openai/gpt-4o"}}
    user = {"valves": Filter.UserValves(DIRECT_FILES=True)}
    # Model lacks file capability - should generate warning
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": False}
                }
            }
        }
    }

    # Create filter and apply
    filt = Filter()
    filtered_body = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    # Verify warning was added
    pipe_meta = metadata.get("openrouter_pipe", {})
    warnings = pipe_meta.get("direct_uploads_warnings", [])
    assert any("not supported" in str(w) for w in warnings)

    # Files should remain (fail-open)
    assert len(filtered_body["files"]) == 1


# ============================================================================
# OpenRouter Search (ORS) Filter Tests
# ============================================================================


class TestORSFilter:
    """Tests for the OpenRouter Search filter rendered from template."""

    @pytest.fixture
    def ors_filter_class(self):
        """Load ORS filter from rendered template."""
        source = FilterManager.render_ors_filter_source()
        module = _load_filter_from_source(source, "ors_filter")
        return module.Filter

    def test_ors_filter_initializes(self, ors_filter_class):
        """Test ORS filter initializes with proper defaults."""
        filt = ors_filter_class()
        assert filt.toggle is True
        # ORS filter doesn't have valves - just toggle

    def test_ors_filter_has_inlet(self, ors_filter_class):
        """Test ORS filter has inlet method."""
        filt = ors_filter_class()
        assert hasattr(filt, 'inlet')
        assert callable(filt.inlet)

    def test_ors_filter_inlet_sets_web_search_false(self, ors_filter_class):
        """Test ORS filter inlet disables web_search in body."""
        filt = ors_filter_class()
        body = {"messages": []}
        metadata = {}
        result = filt.inlet(body, __metadata__=metadata)
        # ORS filter sets web_search to False
        assert result["features"]["web_search"] is False

    def test_ors_filter_inlet_sets_metadata_flag(self, ors_filter_class):
        """Test ORS filter inlet sets feature flag in metadata."""
        filt = ors_filter_class()
        body = {"messages": []}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)
        # ORS filter sets feature flag in metadata
        assert "features" in metadata
        assert metadata["features"].get("openrouter_web_search") is True


# ============================================================================
# Provider Routing Filter Tests
# ============================================================================


class TestProviderRoutingFilter:
    """Tests for Provider Routing filters rendered from template."""

    @pytest.fixture
    def provider_filter_admin(self):
        """Load provider routing filter with admin visibility."""
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="openai/gpt-4o",
            providers=["openai", "azure"],
            quantizations=["fp16", "int8"],
            visibility="admin",
            short_name="GPT-4o",
            provider_names={"openai": "OpenAI", "azure": "Azure"},
        )
        module = _load_filter_from_source(source, "provider_filter_admin")
        return module.Filter

    @pytest.fixture
    def provider_filter_user(self):
        """Load provider routing filter with user visibility."""
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="anthropic/claude-3-opus",
            providers=["anthropic", "amazon-bedrock"],
            quantizations=[],
            visibility="user",
            short_name="Claude 3 Opus",
            provider_names={"anthropic": "Anthropic", "amazon-bedrock": "Amazon Bedrock"},
        )
        module = _load_filter_from_source(source, "provider_filter_user")
        return module.Filter

    @pytest.fixture
    def provider_filter_both(self):
        """Load provider routing filter with both visibility."""
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="meta-llama/llama-3-70b",
            providers=["together", "fireworks", "deepinfra"],
            quantizations=["fp16", "int4", "int8"],
            visibility="both",
            short_name="Llama 3 70B",
        )
        module = _load_filter_from_source(source, "provider_filter_both")
        return module.Filter

    def test_provider_filter_admin_initializes(self, provider_filter_admin):
        """Test admin provider filter initializes correctly."""
        filt = provider_filter_admin()
        # Admin visibility means toggle=False (always runs)
        assert filt.toggle is False
        # Should have Valves but not UserValves
        assert hasattr(filt, 'valves')
        assert not hasattr(filt, 'user_valves')

    def test_provider_filter_admin_valves_has_fields(self, provider_filter_admin):
        """Test admin Valves has all expected fields."""
        filt = provider_filter_admin()
        valves = filt.valves

        # Check field existence
        assert hasattr(valves, 'ORDER')
        assert hasattr(valves, 'ALLOW_FALLBACKS')
        assert hasattr(valves, 'REQUIRE_PARAMETERS')
        assert hasattr(valves, 'DATA_COLLECTION')
        assert hasattr(valves, 'ZDR')
        assert hasattr(valves, 'ONLY')
        assert hasattr(valves, 'IGNORE')
        assert hasattr(valves, 'QUANTIZATION')
        assert hasattr(valves, 'SORT')

        # Check defaults
        assert valves.ORDER == "(no preference)"
        assert valves.ALLOW_FALLBACKS is True
        assert valves.ZDR is False

    def test_provider_filter_user_initializes(self, provider_filter_user):
        """Test user provider filter initializes correctly."""
        filt = provider_filter_user()
        # User visibility means toggle=True (user can disable)
        assert filt.toggle is True
        # Should have UserValves but not Valves
        assert hasattr(filt, 'user_valves')
        assert not hasattr(filt, 'valves')

    def test_provider_filter_user_valves_has_fields(self, provider_filter_user):
        """Test user UserValves has all expected fields."""
        filt = provider_filter_user()
        user_valves = filt.user_valves

        assert hasattr(user_valves, 'ORDER')
        assert hasattr(user_valves, 'ONLY')
        assert hasattr(user_valves, 'IGNORE')
        assert user_valves.ORDER == "(no preference)"

    def test_provider_filter_both_initializes(self, provider_filter_both):
        """Test both-visibility provider filter has both valve types."""
        filt = provider_filter_both()
        # Both visibility means toggle=True
        assert filt.toggle is True
        # Should have both Valves and UserValves
        assert hasattr(filt, 'valves')
        assert hasattr(filt, 'user_valves')

    def test_provider_filter_has_inlet(self, provider_filter_admin):
        """Test provider filter has inlet method."""
        filt = provider_filter_admin()
        assert hasattr(filt, 'inlet')
        assert callable(filt.inlet)

    def test_provider_filter_inlet_returns_body(self, provider_filter_admin):
        """Test provider filter inlet returns body."""
        filt = provider_filter_admin()
        body = {"messages": [], "model": "openai/gpt-4o"}
        metadata = {}
        result = filt.inlet(body, __metadata__=metadata)
        assert result is body

    def test_provider_filter_inlet_sets_provider_routing(self, provider_filter_admin):
        """Test provider filter inlet sets provider routing in metadata."""
        filt = provider_filter_admin()
        # Create new Valves instance with ONLY explicitly set (triggers model_fields_set)
        filt.valves = filt.Valves(ONLY="OpenAI")

        body = {"messages": [], "model": "openai/gpt-4o"}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        # Should have provider routing in metadata (key is "provider", not "provider_routing")
        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
        assert routing.get("only") == ["openai"]

    def test_provider_filter_with_quantization(self, provider_filter_admin):
        """Test provider filter with quantization option."""
        filt = provider_filter_admin()
        # Create new Valves instance with QUANTIZATION explicitly set
        filt.valves = filt.Valves(QUANTIZATION="fp16")

        body = {"messages": []}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
        assert routing.get("quantizations") == ["fp16"]

    def test_provider_filter_with_zdr(self, provider_filter_admin):
        """Test provider filter with ZDR enabled."""
        filt = provider_filter_admin()
        # Create new Valves instance with ZDR explicitly set
        filt.valves = filt.Valves(ZDR=True)

        body = {"messages": []}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
        assert routing.get("zdr") is True

    def test_provider_filter_order_mapping(self, provider_filter_both):
        """Test provider filter ORDER field maps to slug list."""
        filt = provider_filter_both()
        # Create new Valves instance with ORDER explicitly set
        filt.valves = filt.Valves(ORDER="Together > Fireworks > Deepinfra")

        body = {"messages": []}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
        # ORDER should be mapped to list of provider slugs
        order = routing.get("order", [])
        assert order == ["together", "fireworks", "deepinfra"]

    def test_provider_filter_marker_present(self, provider_filter_admin):
        """Test provider filter source contains marker."""
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="openai/gpt-4o",
            providers=["openai"],
            quantizations=[],
            visibility="admin",
        )
        assert "OWUI_OPENROUTER_PIPE_MARKER" in source
        assert "openrouter_pipe:provider_routing:" in source

    def test_provider_filter_escapes_model_slug(self):
        """Test provider filter properly escapes model slugs."""
        # Test with a slug that might need escaping
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="vendor/model-with-dashes",
            providers=["provider"],
            quantizations=[],
            visibility="admin",
        )
        assert "vendor/model-with-dashes" in source
        # Should not cause syntax errors when compiled
        module = _load_filter_from_source(source, "provider_filter_escape_test")
        filt = module.Filter()
        assert filt is not None
