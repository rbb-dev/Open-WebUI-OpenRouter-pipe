"""Integration tests for filter template strings from FilterManager.



These tests load filters from FilterManager.render_*_filter_source() methods
to ensure we test the ACTUAL code that gets deployed, not static backup copies.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

from tests.media_slots import inlined_file
import contextlib
import inspect
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from aioresponses import aioresponses
from pydantic import BaseModel

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.filters import FilterManager
from open_webui_openrouter_pipe.filters.filter_manager import (
    _WriteOutcome,
    _PIPE_INSTALLED_META_KEY,
    _PIPE_OFF_META_KEY,
    _meta_dict,
)
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
from open_webui.models import functions as _owui_functions

def _use_sized_file_records(size: int = 4096) -> None:
    """Stand in for the record Open WebUI hands back, carrying the size its upload measured.

    The pipe sizes every direct upload against `declared_file_size(file_obj)`, which reads
    `meta.size`; a stub without it is a record the pipe cannot size. The autouse
    `_restore_rebound_file_accessors` fixture puts the real accessor back afterwards.
    """
    import open_webui_openrouter_pipe.requests.orchestrator as orch

    async def _get_file_by_id(file_id, *_args, **_kwargs):
        return MagicMock(id=str(file_id), meta={"size": size})

    orch.get_file_by_id = _get_file_by_id


def _load_filter_from_source(source: str, module_name: str) -> ModuleType:
    """Load filter source string as a module.

    This tests the ACTUAL code that FilterManager deploys to Open WebUI,
    not static backup copies in /filters/.
    """
    import sys

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

    sys.modules[module_name] = module

    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)

    if hasattr(module, "Filter"):
        if hasattr(module.Filter, "UserValves"):
            module.Filter.UserValves.model_rebuild()
        if hasattr(module.Filter, "Valves"):
            module.Filter.Valves.model_rebuild()

    return module


_direct_uploads_source = FilterManager.render_direct_uploads_filter_source()
_direct_uploads_module = _load_filter_from_source(_direct_uploads_source, "direct_uploads_filter")
Filter = _direct_uploads_module.Filter


# Filter Initialization Tests


def test_filter_initializes_with_default_valves():
    """Test Filter initializes with proper default valves."""
    filt = Filter()

    assert filt.toggle is True
    assert hasattr(filt, 'valves')
    assert isinstance(filt.valves, Filter.Valves)
    assert filt.valves.priority == 0


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


# Inlet Tests - Basic Validation


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


# Inlet Tests - File Diversion


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

    assert len(result["files"]) == 1


# Inlet Tests - Audio Diversion


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
            "name": "recording.xyz",
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

    assert len(result["files"]) == 1


# Inlet Tests - Video Diversion


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
            "content_type": "video/avi",
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

    assert len(result["files"]) == 1


# Inlet Tests - Edge Cases


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
            "type": "image",
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
            "id": "",
            "type": "file",
            "name": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        },
        {
            "id": 123,
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
    user = {"valves": "not a BaseModel"}

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
    metadata = {"files": files}
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

    direct_uploads = metadata["openrouter_pipe"]["direct_uploads"]
    assert len(direct_uploads["files"]) == 2


def test_inlet_deduplicates_by_id():
    """Test inlet deduplicates files by id."""
    filt = Filter()

    existing_file = {"id": "file_1", "name": "old.pdf", "size": 500, "content_type": "application/pdf"}

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

    direct_uploads = metadata["openrouter_pipe"]["direct_uploads"]
    assert len(direct_uploads["files"]) == 1


def test_inlet_persists_responses_audio_format_allowlist():
    """Test inlet persists both audio format allowlists in metadata.

    Both valves are set to non-default values: the shipped defaults happen to equal the
    strings the test would otherwise assert, so a deleted persistence line would still
    leave a value on both sides and the assertion would hold. The pipe reads the
    operator allowlist as its gate's second term, so a hand-off that silently stops
    happening looks, from the pipe's side, like an operator who never configured anything.
    """
    filt = Filter()
    filt.valves.DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST = "wav,mp3"
    filt.valves.DIRECT_AUDIO_FORMAT_ALLOWLIST = "wav,mp3,opus"

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
    assert direct_uploads["audio_format_allowlist"] == "wav,mp3,opus"


# Additional Coverage Tests


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
    user = {"valves": Filter.UserValves(DIRECT_AUDIO=False)}
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
                    "capabilities": {"audio_input": False}
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
    filt.valves.DIRECT_AUDIO_MIME_ALLOWLIST = "audio/wav"

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

    assert len(result["files"]) == 1


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
    user = {"valves": Filter.UserValves(DIRECT_VIDEO=False)}
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
                    "capabilities": {"video_input": False}
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

    warnings = metadata["openrouter_pipe"]["direct_uploads_warnings"]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_filter_integration_with_pipe_direct_uploads(pipe_instance_async, monkeypatch):
    """Test filter output integrates correctly with Pipe processing."""
    pipe = pipe_instance_async
    _use_sized_file_records()

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

    # The body carries a `files` entry, so the loop resolves it before the request goes
    # out. That resolution is stubbed here rather than left to reach Open WebUI's storage:
    # without a key the request never went out, so the loader never ran and this node
    # passed while asserting nothing about it.
    from unittest.mock import AsyncMock

    from open_webui_openrouter_pipe.storage.owui_files import InlinedFile

    pipe._file_gateway.inline_owui_file_id = AsyncMock(
        return_value=inlined_file("data:application/pdf;base64,JVBERi0xLjQK", "document.pdf")
    )

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
                session=session,
            valves=valves,
            )

            # `_handle_pipe_call` hands a chat turn's non-streamed answer back as the
            # text itself. The `dict` this asserted on was the adapter's internal shape,
            # which the pipe never returned: the node only passed because a keyless pipe
            # refused before the request was sent. The point of this node is that the
            # filter's own edits survive the call, so what is asserted is the answer.
            assert isinstance(result, str), f"a chat turn returned {type(result).__name__}"
            assert result == "I see the document.", f"the filter's turn answered {result!r}"
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

    assert len(filtered_body["files"]) == 1


# Provider Routing Filter Tests


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
        assert filt.toggle is False
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
        """Test user provider filter initializes correctly.

        The instance carries an admin `valves` on every visibility: Open WebUI's
        `apply_filter_valves` (utils/filter.py:121) returns early unless the instance
        has both the attribute and the class, so a user-visibility row without them can
        never have a stored value applied. What a user-visibility row must NOT publish
        is an admin routing *control* -- those are what the admin valves are for, and
        `priority` is the one field on that class (the sort key Open WebUI reads at
        utils/filter.py:88-97, which has no per-user meaning).
        """
        filt = provider_filter_user()
        assert filt.toggle is True
        assert hasattr(filt, 'user_valves')
        assert hasattr(filt, 'valves')
        assert not hasattr(filt.valves, 'ORDER')

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
        assert filt.toggle is True
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
        filt.valves = filt.Valves(ONLY="OpenAI")

        body = {"messages": [], "model": "openai/gpt-4o"}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
        assert routing.get("only") == ["openai"]

    def test_provider_filter_with_quantization(self, provider_filter_admin):
        """Test provider filter with quantization option."""
        filt = provider_filter_admin()
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
        filt.valves = filt.Valves(ORDER="Together > Fireworks > Deepinfra")

        body = {"messages": []}
        metadata = {}
        filt.inlet(body, __metadata__=metadata)

        pipe_meta = metadata.get("openrouter_pipe", {})
        routing = pipe_meta.get("provider", {})
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

    # --- H133-2: presence, not value. A user who explicitly clears a boolean or a
    # number the admin set must have that clear reach the request. ---

    _BOOL_CONTROLS = [
        ("ZDR", "zdr"),
        ("REQUIRE_PARAMETERS", "require_parameters"),
        ("ENFORCE_DISTILLABLE_TEXT", "enforce_distillable_text"),
        ("ALLOW_FALLBACKS", "allow_fallbacks"),
    ]

    _NUMERIC_CONTROLS = [
        ("MIN_THROUGHPUT", "preferred_min_throughput"),
        ("MAX_LATENCY", "preferred_max_latency"),
        ("MAX_PRICE_PROMPT", ("max_price", "prompt")),
        ("MAX_PRICE_COMPLETION", ("max_price", "completion")),
        ("MAX_PRICE_IMAGE", ("max_price", "image")),
        ("MAX_PRICE_AUDIO", ("max_price", "audio")),
        ("MAX_PRICE_REQUEST", ("max_price", "request")),
    ]

    @staticmethod
    def _emitted(provider, spec):
        """Read a control out of the emitted provider dict, None when absent."""
        if isinstance(spec, tuple):
            return provider.get(spec[0], {}).get(spec[1]) if spec[0] in provider else None
        return provider.get(spec)

    def _drive(self, provider_filter_both, *, admin=None, user=None):
        filt = provider_filter_both()
        if admin is not None:
            filt.valves = filt.Valves(**admin)
        user_valves = filt.UserValves(**user) if user is not None else None
        metadata = {}
        kwargs = {"__metadata__": metadata}
        if user_valves is not None:
            kwargs["__user__"] = {"valves": user_valves}
        filt.inlet({"messages": []}, **kwargs)
        return metadata.get("openrouter_pipe", {}).get("provider", {})

    def test_provider_filter_escapes_model_slug(self):
        """Test provider filter properly escapes model slugs."""
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="vendor/model-with-dashes",
            providers=["provider"],
            quantizations=[],
            visibility="admin",
        )
        assert "vendor/model-with-dashes" in source
        module = _load_filter_from_source(source, "provider_filter_escape_test")
        filt = module.Filter()
        assert filt is not None


def _order_options_of(module) -> dict[str, list[str]]:
    return dict(module._ORDER_MAP)


def _literal_options(valves_cls, field: str) -> tuple:
    from typing import get_args

    return get_args(valves_cls.model_fields[field].annotation)


def _render_with_n_providers(count: int, visibility: str = "admin") -> str:
    providers = [f"prov-{i:02d}" for i in range(count)]
    names = {p: f"Provider {i:02d}" for i, p in enumerate(providers)}
    return FilterManager._render_provider_routing_filter_source(
        model_slug="example/scale-model",
        providers=providers,
        quantizations=["fp8"],
        visibility=visibility,
        short_name="Scale Model",
        provider_names=names,
    )


class TestProviderRoutingOrderBounds:
    """ORDER dropdown generation must stay bounded (uncapped n! froze the loop)."""

    def test_order_full_permutations_at_three(self):
        module = _load_filter_from_source(
            _render_with_n_providers(3), "order_bounds_three"
        )
        order_map = _order_options_of(module)
        assert len(order_map) == 6
        assert all(len(slugs) == 3 for slugs in order_map.values())

    def test_order_full_permutations_at_four(self):
        module = _load_filter_from_source(
            _render_with_n_providers(4), "order_bounds_four"
        )
        order_map = _order_options_of(module)
        assert len(order_map) == 24
        assert all(len(slugs) == 4 for slugs in order_map.values())

    def test_order_linear_at_five(self):
        module = _load_filter_from_source(
            _render_with_n_providers(5), "order_bounds_five"
        )
        order_map = _order_options_of(module)
        assert len(order_map) == 5
        assert all(len(slugs) == 1 for slugs in order_map.values())
        assert all(label.endswith(" first") for label in order_map)

    def test_order_linear_at_live_scale_fourteen(self):
        module = _load_filter_from_source(
            _render_with_n_providers(14), "order_bounds_fourteen"
        )
        order_map = _order_options_of(module)
        assert len(order_map) == 14
        slugs_seen = {slugs[0] for slugs in order_map.values()}
        assert len(slugs_seen) == 14
        options = _literal_options(module.Filter.Valves, "ORDER")
        assert len(options) == 15
        assert "(no preference)" in options

    def test_only_ignore_keep_full_list_above_threshold(self):
        module = _load_filter_from_source(
            _render_with_n_providers(14), "order_bounds_only_full"
        )
        only_options = _literal_options(module.Filter.Valves, "ONLY")
        assert len(only_options) == 15


class TestProviderRoutingStaleValueHealing:
    """Saved valve values from a previous filter generation must never raise."""

    @pytest.fixture
    def healing_module(self):
        source = FilterManager._render_provider_routing_filter_source(
            model_slug="example/heal-model",
            providers=["baidu", "streamlake"],
            quantizations=["fp8"],
            visibility="both",
            short_name="Heal Model",
            provider_names={"baidu": "Baidu", "streamlake": "StreamLake"},
        )
        return _load_filter_from_source(source, "provider_filter_healing")

    def test_admin_stale_order_coerced(self, healing_module):
        valves = healing_module.Filter.Valves(ORDER="StreamLake")
        assert valves.ORDER == "(no preference)"

    def test_admin_stale_only_coerced(self, healing_module):
        valves = healing_module.Filter.Valves(ONLY="Vanished Provider")
        assert valves.ONLY == "(no preference)"

    def test_user_stale_ignore_coerced(self, healing_module):
        user_valves = healing_module.Filter.UserValves(IGNORE="Vanished Provider")
        assert user_valves.IGNORE == "(no preference)"

    def test_stale_quantization_coerced(self, healing_module):
        valves = healing_module.Filter.Valves(QUANTIZATION="int4")
        assert valves.QUANTIZATION == "(no preference)"

    def test_non_string_value_coerced(self, healing_module):
        valves = healing_module.Filter.Valves(ORDER=None)
        assert valves.ORDER == "(no preference)"

    def test_valid_values_pass_through(self, healing_module):
        valves = healing_module.Filter.Valves(ONLY="StreamLake")
        assert valves.ONLY == "StreamLake"

    def test_static_literals_stay_strict(self, healing_module):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            healing_module.Filter.Valves(DATA_COLLECTION="bogus")

    def test_inlet_emits_nothing_for_coerced_value(self, healing_module):
        filt = healing_module.Filter()
        filt.valves = healing_module.Filter.Valves(ORDER="StreamLake")
        metadata = {}
        filt.inlet({"messages": []}, __metadata__=metadata)
        routing = metadata.get("openrouter_pipe", {}).get("provider", {})
        assert "order" not in routing

    def test_both_mode_user_stale_shadows_admin(self, healing_module):
        filt = healing_module.Filter()
        filt.valves = healing_module.Filter.Valves(ONLY="StreamLake")
        stale_user = healing_module.Filter.UserValves(ONLY="Vanished Provider")
        metadata = {}
        filt.inlet(
            {"messages": []},
            __metadata__=metadata,
            __user__={"valves": stale_user},
        )
        routing = metadata.get("openrouter_pipe", {}).get("provider", {})
        assert "only" not in routing

from types import SimpleNamespace


_STORED_USER_VALVES_PREFIX = "enc:"


class _FakeFunctionsTable:
    store: dict = {}
    user_valves: dict = {}
    refuse_user_writes = False

    @classmethod
    def reset(cls):
        cls.store = {}
        cls.update_count = 0
        cls.user_valves = {}
        cls.refuse_user_writes = False

    @classmethod
    async def get_functions_by_type(cls, function_type, active_only=False):
        return list(cls.store.values())

    @classmethod
    async def get_function_by_id(cls, function_id):
        return cls.store.get(function_id)

    @classmethod
    async def insert_new_function(cls, user_id, function_type, form):
        obj = SimpleNamespace(
            id=form.id,
            name=form.name,
            content=form.content,
            meta=form.meta,
            is_active=False,
            is_global=False,
        )
        cls.store[form.id] = obj
        return obj

    update_count = 0

    @classmethod
    async def update_function_by_id(cls, function_id, updates):
        cls.update_count += 1
        obj = cls.store.get(function_id)
        if obj is None:
            return None
        payload = dict(updates)
        try:
            json.dumps(payload)
        except TypeError:
            return None
        for key, value in payload.items():
            setattr(obj, key, value)
        return obj

    @classmethod
    async def delete_function_by_id(cls, function_id):
        return cls.store.pop(function_id, None) is not None

    @classmethod
    async def get_function_valves_by_ids(cls, ids, db=None):
        return {i: (getattr(cls.store[i], "valves", None) or {}) for i in ids if i in cls.store}

    @classmethod
    async def get_user_valves_by_id_and_user_id(cls, function_id, user_id, db=None):
        stored = cls.user_valves.get(user_id, {}).get(function_id)
        if isinstance(stored, str):
            return json.loads(stored[len(_STORED_USER_VALVES_PREFIX):])
        return {}

    @classmethod
    async def update_user_valves_by_id_and_user_id(cls, function_id, user_id, valves, db=None):
        if cls.refuse_user_writes or user_id not in cls.user_valves:
            return None
        cls.user_valves[user_id][function_id] = _STORED_USER_VALVES_PREFIX + json.dumps(valves, sort_keys=True)
        return valves


class _FakeUsersTable:
    @classmethod
    async def get_users(cls, filter=None, sort=None, skip=None, limit=None, db=None):
        ordered = sorted(_FakeFunctionsTable.user_valves)
        chosen = ordered[skip or 0 :][: len(ordered) if limit is None else limit]
        return {
            "users": [
                SimpleNamespace(
                    id=user_id,
                    settings=SimpleNamespace(
                        model_dump=lambda user_id=user_id: {
                            "ui": {},
                            "functions": {"valves": dict(_FakeFunctionsTable.user_valves[user_id])},
                        }
                    ),
                )
                for user_id in chosen
            ],
            "total": len(ordered),
        }


class TestProviderRoutingEndToEndOverlay:
    """Valve-listed model absent from the frontend map still gets a full filter."""

    @pytest.fixture(autouse=True)
    def _isolate_state(self, monkeypatch):
        import open_webui.models.functions as functions_module

        _FakeFunctionsTable.reset()
        monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)


    @pytest.mark.asyncio
    async def test_null_endpoint_model_gets_full_filter(self, pipe_instance_async):
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        frontend_data = {
            "data": [
                {"slug": "example/null-model", "short_name": "Null Model", "endpoint": None},
                {
                    "slug": "example/variant-model",
                    "short_name": "Variant Model",
                    "endpoint": {
                        "model_variant_slug": "example/variant-model:free",
                        "provider_info": {"slug": "freeprov", "displayName": "FreeProv"},
                    },
                },
            ]
        }
        endpoints_payload = {
            "data": {
                "id": "example/null-model",
                "name": "Example: Null Model",
                "endpoints": [
                    {"provider_name": "Alpha", "tag": "alpha", "quantization": "fp8"},
                    {"provider_name": "Beta", "tag": "beta/fp4", "quantization": "fp4"},
                    {"provider_name": "Gamma", "tag": "gamma", "quantization": "unknown"},
                ],
            }
        }
        from aioresponses import aioresponses

        with aioresponses() as mocked:
            mocked.get(
                "https://openrouter.ai/api/v1/models/example/null-model/endpoints",
                payload=endpoints_payload,
            )
            session = pipe._create_http_session()
            try:
                provider_map = await manager._build_provider_map_with_overlay(
                    session, frontend_data, "example/null-model", ""
                )
            finally:
                await session.close()

        assert provider_map["example/null-model"]["providers"] == ["alpha", "beta", "gamma"]

        filter_manager = pipe._ensure_filter_manager()
        mapping, _ = await filter_manager.ensure_provider_routing_filters(
            "example/null-model",
            "",
            provider_map,
            [],
            "openrouter",
        )
        assert "example/null-model" in mapping
        stored = _FakeFunctionsTable.store[mapping["example/null-model"]]
        assert "Alpha" in stored.content
        assert "Beta" in stored.content
        assert "Gamma" in stored.content
        assert stored.is_active is True

    def test_hash_differs_between_collapsed_and_enriched_map(self):
        collapsed = {
            "example/null-model": {
                "providers": ["alpha"],
                "quantizations": [],
                "short_name": "Null Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        enriched = {
            "example/null-model": {
                "providers": ["alpha", "beta", "gamma"],
                "quantizations": ["fp4", "fp8", "unknown"],
                "short_name": "Null Model",
                "provider_names": {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
            }
        }
        hash_collapsed = FilterManager.compute_provider_routing_hash(
            "example/null-model", "", collapsed
        )
        hash_enriched = FilterManager.compute_provider_routing_hash(
            "example/null-model", "", enriched
        )
        assert hash_collapsed != hash_enriched


class TestProviderRoutingHashPersistence:
    """The state hash must persist after a no-change pass so the early-return
    optimization actually engages (it never persisted in steady state before)."""

    @pytest.fixture(autouse=True)
    def _isolate_state(self, monkeypatch):
        import open_webui.models.functions as functions_module

        _FakeFunctionsTable.reset()
        monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
        yield

    @pytest.mark.asyncio
    async def test_listing_failure_aborts_without_creating_duplicates(self, pipe_instance_async, monkeypatch):
        """A failed filter listing must abort the sync, not create duplicate filters.

        Falling through with an empty list makes every routed slug look "missing",
        so the create branch runs, the id suffix bumps to _1/_2/..., and the model is
        repointed at the duplicate on every refresh.
        """
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        provider_map = {
            "example/steady-model": {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": "Steady Model",
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
        }
        first, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model", "", provider_map, [], "openrouter"
        )
        assert "example/steady-model" in first
        rows_before = dict(_FakeFunctionsTable.store)
        hash_before = filter_manager._provider_routing_state_hash

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(_FakeFunctionsTable, "get_functions_by_type", _boom)
        filter_manager._provider_routing_state_hash = ""

        result, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model", "", provider_map, [], "openrouter"
        )
        assert result == {}
        assert _FakeFunctionsTable.store == rows_before
        assert not any(k.endswith("_1") for k in _FakeFunctionsTable.store)
        assert filter_manager._provider_routing_state_hash != hash_before or hash_before == ""

    @pytest.mark.asyncio
    async def test_create_loop_lookup_error_propagates(self, pipe_instance_async, monkeypatch):
        """The id-collision loop no longer nets get_function_by_id.

        OWUI self-guards that call, so a raise means something genuinely unexpected
        and must not be silently treated as "id is free".
        """
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        provider_map = {
            "example/collide-model": {
                "providers": ["alpha"],
                "quantizations": ["fp8"],
                "short_name": "Collide Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("lookup exploded")

        monkeypatch.setattr(_FakeFunctionsTable, "get_function_by_id", _boom)
        with pytest.raises(RuntimeError, match="lookup exploded"):
            await filter_manager.ensure_provider_routing_filters(
                "example/collide-model", "", provider_map, [], "openrouter"
            )

    @pytest.mark.asyncio
    async def test_duplicate_filters_are_reaped(self, pipe_instance_async):
        """Duplicates left by an earlier failed sync must be disabled, not ignored.

        They carry the same marker, so a slug->filter dict silently kept only one
        and the rest stayed active forever.
        """
        from open_webui_openrouter_pipe.core.config import (
            _PROVIDER_ROUTING_FILTER_MARKER_PREFIX,
        )

        slug = "example/dup-model"
        marker = f'OWUI_OPENROUTER_PIPE_MARKER = "{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{slug}:v1"\n'
        for fid in ("openrouter_provider_example_dup_model", "openrouter_provider_example_dup_model_1"):
            _FakeFunctionsTable.store[fid] = SimpleNamespace(
                id=fid, name=fid, content=marker, meta=None, is_active=True, is_global=False,
            )

        provider_map = {
            slug: {
                "providers": ["alpha"],
                "quantizations": ["fp8"],
                "short_name": "Dup Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        filter_manager = pipe_instance_async._ensure_filter_manager()
        mapping, _ = await filter_manager.ensure_provider_routing_filters(
            slug, "", provider_map, [], "openrouter"
        )
        assert mapping[slug] == "openrouter_provider_example_dup_model"
        assert _FakeFunctionsTable.store["openrouter_provider_example_dup_model_1"].is_active is False

    @pytest.mark.asyncio
    async def test_slugless_marker_is_ignored(self, pipe_instance_async):
        """A marker carrying only a version must not register the version as a slug."""
        from open_webui_openrouter_pipe.core.config import (
            _PROVIDER_ROUTING_FILTER_MARKER_PREFIX,
        )

        _FakeFunctionsTable.store["stray_filter"] = SimpleNamespace(
            id="stray_filter",
            name="stray",
            content=f'OWUI_OPENROUTER_PIPE_MARKER = "{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}v1"\n',
            meta=None,
            is_active=True,
            is_global=False,
        )
        provider_map = {
            "example/plain-model": {
                "providers": ["alpha"],
                "quantizations": ["fp8"],
                "short_name": "Plain Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        filter_manager = pipe_instance_async._ensure_filter_manager()
        mapping, _ = await filter_manager.ensure_provider_routing_filters(
            "example/plain-model", "", provider_map, [], "openrouter"
        )
        assert "v1" not in mapping
        assert "example/plain-model" in mapping

    @pytest.mark.asyncio
    async def test_variant_slug_marker_round_trips_without_duplicating(self, pipe_instance_async):
        """A slug containing a colon (":free", ":nitro") must survive marker recovery.

        Splitting the marker on every colon truncated the slug, so the lookup missed
        and a fresh _1/_2/... filter was created on every single refresh.
        """
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        slug = "example/variant-model:free"
        provider_map = {
            slug: {
                "providers": ["alpha"],
                "quantizations": ["fp8"],
                "short_name": "Variant Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        first, _ = await filter_manager.ensure_provider_routing_filters(slug, "", provider_map, [], "openrouter")
        assert slug in first
        created = len(_FakeFunctionsTable.store)

        filter_manager._provider_routing_state_hash = ""
        second, _ = await filter_manager.ensure_provider_routing_filters(slug, "", provider_map, [], "openrouter")
        assert second == first
        assert len(_FakeFunctionsTable.store) == created
        assert not any(k.endswith("_1") for k in _FakeFunctionsTable.store)

    @pytest.mark.asyncio
    async def test_hash_persists_after_noop_pass(self, pipe_instance_async):
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        provider_map = {
            "example/steady-model": {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": "Steady Model",
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
        }
        first, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model", "", provider_map, [], "openrouter"
        )
        assert "example/steady-model" in first
        hash_after_create = filter_manager._provider_routing_state_hash
        assert hash_after_create

        second, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model", "", provider_map, [], "openrouter"
        )
        assert second == first
        assert filter_manager._provider_routing_state_hash == hash_after_create

        filter_manager._provider_routing_state_hash = ""
        third, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model", "", provider_map, [], "openrouter"
        )
        assert third == first
        assert filter_manager._provider_routing_state_hash == hash_after_create

    @pytest.mark.asyncio
    async def test_unresolvable_slug_does_not_defeat_early_return(self, pipe_instance_async):
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        provider_map = {
            "example/steady-model": {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": "Steady Model",
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
        }
        first, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model, example/typo-model", "", provider_map, [], "openrouter"
        )
        assert "example/steady-model" in first
        assert "example/typo-model" not in first
        updates_after_first = _FakeFunctionsTable.update_count

        second, _ = await filter_manager.ensure_provider_routing_filters(
            "example/steady-model, example/typo-model", "", provider_map, [], "openrouter"
        )
        assert second == first
        assert _FakeFunctionsTable.update_count == updates_after_first

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("routable", "undeliverable"),
        [
            ("example/steady-model", "acme/reelmaker-1"),
        ],
    )
    async def test_a_transport_without_routing_controls_gets_no_filter(
        self, pipe_instance_async, routable, undeliverable
    ):
        """A model whose transport carries none of the routing keys is skipped.

Video requests carry a provider block of `options` alone, so every routing control a filter offered would be
        accepted from the user and then dropped before the request left. Installing the filter anyway puts controls
        on screen that cannot do anything.
        """
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        OpenRouterModelRegistry.register_video_models(
            [{"id": undeliverable, "name": undeliverable, "pricing": {},
              "allowed_passthrough_parameters": []}]
        )
        assert filter_manager.model_transport(undeliverable) == "video"
        assert filter_manager.model_transport(routable) == "chat"

        provider_map = {
            slug: {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": slug,
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
            for slug in (routable, undeliverable)
        }

        mapping, _ = await filter_manager.ensure_provider_routing_filters(
            f"{routable},{undeliverable}", "", provider_map, [], "openrouter"
        )

        assert routable in mapping
        assert undeliverable not in mapping, (
            f"{undeliverable} was handed a provider routing filter, but its request "
            "format carries none of those settings"
        )
        installed = {
            row_id
            for row_id, row in _FakeFunctionsTable.store.items()
            if getattr(row, "is_active", False)
        }
        assert mapping[routable] in installed
        assert not any(
            filter_manager.sanitize_model_for_filter_id(undeliverable) in row_id
            for row_id in installed
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("routable", "undeliverable"),
        [
            ("example/steady-model", "acme/reelmaker-1"),
            ("vendor/other-model", "zeta/clipsmith-9"),
        ],
    )
    async def test_an_already_installed_filter_for_such_a_transport_is_switched_off(
        self, pipe_instance_async, routable, undeliverable
    ):
        """A filter installed before the model's transport was known is deactivated.

        The routing valves are unchanged across the two passes, so the second pass takes
        the state-hash early return. That return must still drop the model whose
        transport cannot carry the controls, otherwise a filter installed by an older
        build stays attached and keeps offering settings the request format discards.
        """
        pipe = pipe_instance_async
        filter_manager = pipe._ensure_filter_manager()
        provider_map = {
            slug: {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": slug,
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
            for slug in (routable, undeliverable)
        }
        csv = f"{routable},{undeliverable}"

        first, _ = await filter_manager.ensure_provider_routing_filters(
            csv, "", provider_map, [], "openrouter"
        )
        assert undeliverable in first
        stale_id = first[undeliverable]
        assert _FakeFunctionsTable.store[stale_id].is_active is True

        OpenRouterModelRegistry.register_video_models(
            [{"id": undeliverable, "name": undeliverable, "pricing": {},
              "allowed_passthrough_parameters": []}]
        )

        second, _ = await filter_manager.ensure_provider_routing_filters(
            csv, "", provider_map, [], "openrouter"
        )

        assert undeliverable not in second
        assert routable in second
        assert _FakeFunctionsTable.store[stale_id].is_active is False, (
            f"the provider routing filter for {undeliverable} is still active after its "
            "transport was found to carry none of those settings"
        )

        third, _ = await filter_manager.ensure_provider_routing_filters(
            csv, "", provider_map, [], "openrouter"
        )
        assert undeliverable not in third
        assert third == second

    @staticmethod
    def _routing_map(*slugs: str) -> dict:
        return {
            slug: {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": slug,
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
            for slug in slugs
        }

@pytest.mark.parametrize("auto_install", [True, False])
@pytest.mark.asyncio
async def test_the_fusion_filter_install_is_wired_to_its_own_marker_valve_and_id(
    monkeypatch, auto_install
):
    """The fusion install composes a renderer, a valve and an id; all three must arrive.

    Nothing else asserts them, so stubbing the whole method to a constant id used to pass
    the suite. Both valve states are driven, because a wiring that ignores the valve
    returns the same id in each.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import (
        _OPENROUTER_FUSION_FILTER_MARKER,
        _OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID,
    )

    rows: dict[str, Any] = {}

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return list(rows.values())

        @staticmethod
        async def get_function_by_id(function_id):
            return rows.get(function_id)

        @staticmethod
        async def insert_new_function(user_id, function_type, form):
            rows[form.id] = SimpleNamespace(
                id=form.id, name=form.name, content=form.content, meta=form.meta,
                is_active=False, is_global=False, updated_at=1,
            )
            return rows[form.id]

        @staticmethod
        async def update_function_by_id(function_id, updates):
            row = rows.get(function_id)
            if row is None:
                return None
            for key, value in dict(updates).items():
                setattr(row, key, value)
            return row

        @staticmethod
        async def delete_function_by_id(function_id):
            return rows.pop(function_id, None) is not None

    monkeypatch.setattr(functions_module, "Functions", _Table)
    pipe = MagicMock()
    pipe.valves.AUTO_INSTALL_FUSION_FILTER = auto_install
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    result, _unresolved = await manager.ensure_openrouter_fusion_filter_function_id()

    if not auto_install:
        assert result is None, "AUTO_INSTALL_FUSION_FILTER off must install nothing"
        assert rows == {}
        return

    assert result == _OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID
    installed = rows[result]
    assert _OPENROUTER_FUSION_FILTER_MARKER in installed.content, (
        "the installed source carries no fusion marker, so the next pass cannot "
        "recognise its own row and installs a duplicate"
    )
    assert "class Filter" in installed.content


def test_filter_id_sanitiser_handles_every_real_catalog_id():
    """Enumerated .replace() chains only cover the separators someone thought of.

    The catalog contains tilde aliases (~anthropic/claude-sonnet-latest) which are
    real, supported ids; a colon-only fix leaves them producing a non-identifier.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    slugs = [
        "openai/gpt-4o",
        "deepseek/deepseek-v3.2:free",
        "~anthropic/claude-sonnet-latest",
        "~google/gemini-pro-latest",
        "meta-llama/llama-3.3-70b",
        "qwen/qwen-2.5-72b-instruct:nitro",
        "x-ai/grok-beta",
    ]
    bad = [
        f"{s} -> {FilterManager.sanitize_model_for_filter_id(s)}"
        for s in slugs
        if not FilterManager.sanitize_model_for_filter_id(s).isidentifier()
    ]
    assert not bad, "these produce ids that are not valid identifiers:\n  " + "\n  ".join(bad)

    mapped = {s: FilterManager.sanitize_model_for_filter_id(s) for s in slugs}
    collisions = {
        v: [k for k in mapped if mapped[k] == v]
        for v in set(mapped.values())
        if list(mapped.values()).count(v) > 1
    }
    assert not collisions, (
        f"distinct model slugs map to the same filter id: {collisions}"
    )


def test_filter_id_sanitiser_is_injective():
    """Distinct catalog ids must produce distinct OWUI function ids.

    Injective over the alphabet OpenRouter actually uses, which is what matters and
    what the probes below cover. It is NOT injective over arbitrary strings -- `a/b`
    and `a_b` both map to `a_b`, and the two lower-casing sanitisers additionally
    merge case -- so the alphabet assertion at the end is load-bearing rather than
    decorative: it is what makes the claim true.

    A run-collapsing `+` maps `a/b-c` and `a/b--c` to the same id, so the second
    model silently reuses -- and overwrites -- the first model's installed filter.
    The ids are persisted OWUI function rows, so a collision is not recoverable by
    re-running discovery.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    probes = [
        "a/b-c",
        "a/b--c",
        "x.y",
        "x..y",
        "openai/gpt-4o",
        "openai/gpt--4o",
        "~anthropic/claude-sonnet-latest",
        "anthropic/claude-sonnet-latest",
        "x/y:free",
        "x/y::free",
        "google/veo-3",
        "~google/veo-3",
    ]
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        sanitize_image_filter_id,
    )
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        sanitize_video_filter_id,
    )

    for label, fn in (
        ("provider_routing", FilterManager.sanitize_model_for_filter_id),
        ("video", sanitize_video_filter_id),
        ("image", sanitize_image_filter_id),
    ):
        mapped = {p: fn(p) for p in probes}
        collisions = {
            v: sorted(k for k in mapped if mapped[k] == v)
            for v in set(mapped.values())
            if list(mapped.values()).count(v) > 1
        }
        assert not collisions, f"{label}: distinct slugs share a filter id: {collisions}"
        bad = sorted(v for v in mapped.values() if not v.isidentifier())
        assert not bad, (
            f"{label}: these filter ids are not Python identifiers: {bad}. Open WebUI "
            "rejects the row (`routers/functions.py` checks `id.isidentifier()`), so the "
            "filter is silently never installed and the model loses its controls."
        )

    import json
    from pathlib import Path as _Path

    # Vendored, because `.external/` is gitignored: reading the alphabet only from there
    # meant this assertion -- the one the docstring calls load-bearing -- skipped on
    # every CI run and every fresh checkout, leaving twelve hand-written probes that
    # contain no "_" and no uppercase letter to stand in for the real population.
    # Local dumps are still read when present, so a fresh one widens the alphabet.
    fixture = _Path(__file__).resolve().parent / "fixtures" / "catalog_model_ids.json"
    ids: set[str] = set(json.loads(fixture.read_text(encoding="utf-8")))
    for dump in sorted(
        (_Path(__file__).resolve().parents[1] / ".external").glob("models-openrouter-*-pretty.json")
    ):
        payload = json.loads(dump.read_text(encoding="utf-8"))
        ids.update(m.get("id", "") for m in payload.get("data", []) if m.get("id"))
    assert len(ids) > 100, f"only {len(ids)} catalog ids; the alphabet check is inert"
    offenders = sorted(i for i in ids if "_" in i or any(c.isupper() for c in i))
    assert not offenders, (
        f"{len(offenders)} catalog ids contain '_' or an uppercase letter, e.g. "
        f"{offenders[:5]}. The sanitisers map those onto the same id as their "
        "separator-punctuated twin, so injectivity no longer holds for real input."
    )


class TestInstalledRowMatchesTheRenderedModule:
    """The row Open WebUI is handed must agree with the module stored in it."""

    @pytest.fixture(autouse=True)
    def _isolate_state(self, monkeypatch):
        import open_webui.models.functions as functions_module

        _FakeFunctionsTable.reset()
        monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
        yield

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("admin_csv", "user_csv", "visibility", "expected"),
        [
            ("example/toggle-model", "", "admin", False),
            ("", "example/toggle-model", "user", True),
        ],
    )
    async def test_the_row_advertises_the_toggle_its_module_declares(
        self, pipe_instance_async, admin_csv, user_csv, visibility, expected
    ):
        """`meta.toggle` is what Open WebUI reads to decide whether users see a switch.

        It was unobservable: the `FunctionMeta` stub in conftest declared only
        `description` and `manifest` and had no `model_config`, so pydantic's default
        `extra='ignore'` silently discarded `toggle` on the way in. The real class is
        `extra='allow'`. Both `"toggle": False` and an inverted `visibility == "admin"`
        left the whole suite green.

        `test_provider_filter_*_initializes` asserts `filt.toggle` on the *rendered
        module*, which is a different fact from the row metadata -- and the pairing of
        the two is exactly what can silently disagree. This reads the row the install
        path actually wrote.
        """
        provider_map = {
            "example/toggle-model": {
                "providers": ["alpha"],
                "quantizations": ["fp16"],
                "short_name": "Toggle Model",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        manager = pipe_instance_async._ensure_filter_manager()
        mapping, _ = await manager.ensure_provider_routing_filters(
            admin_csv, user_csv, provider_map, [], "openrouter"
        )
        assert "example/toggle-model" in mapping, "the filter was never installed"
        stored = _FakeFunctionsTable.store[mapping["example/toggle-model"]]

        row_toggle = _meta_dict(stored).get("toggle")
        assert row_toggle is not None, (
            "meta.toggle did not survive into the stored row at all. The FunctionMeta "
            "stub is dropping extra keys again, which makes every meta field the pipe "
            "sets beyond description/manifest untestable."
        )
        assert row_toggle is expected, (
            f"visibility={visibility!r} wrote meta.toggle={row_toggle!r}, expected "
            f"{expected!r}. Open WebUI renders the user-facing switch from this."
        )

        module = _load_filter_from_source(stored.content, f"toggle_probe_{visibility}")
        module_toggle = bool(getattr(module.Filter(), "toggle", False))
        assert module_toggle is expected, (
            f"the rendered module declares toggle={module_toggle!r} while the row says "
            f"{row_toggle!r}; the two describe the same filter and must agree"
        )



# ============================================================================
# REGFIX: infrastructure fixes 2, 3, 6
# ============================================================================


_INSTALLED_FILTER_RENDERER_ARGS: dict[str, dict[str, Any]] = {
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
    "render_router_filter_source": {},
    "render_image_model_filter_source": {},
    "render_image_gen_filter_source": {"catalog_match": True},
    "render_video_filter_source": {
        "model_id": "google/veo-3",
        "video_model": {"id": "google/veo-3", "name": "Veo 3"},
    },
}


def _installed_filter_renderers() -> dict[str, Any]:
    """The renderer callables, resolved by name rather than by scanning a namespace.

    A scan is wrong here: in the flat bundle every module shares one namespace, so a
    `render_` prefix sweep also collects `render_video_help` and `render_image_help`,
    which produce chat prose and are never installed as filters. The completeness of this
    table is enforced separately, in package mode, where the namespaces are still distinct.
    """
    import importlib

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    found: dict[str, Any] = {}
    for module_name in (
        "filter_manager", "fusion_filter_renderer",
        "image_filter_renderer", "router_filter_renderer", "video_filter_renderer",
    ):
        module = importlib.import_module(
            f"open_webui_openrouter_pipe.filters.{module_name}"
        )
        for name in _INSTALLED_FILTER_RENDERER_ARGS:
            candidate = getattr(module, name, None)
            if callable(candidate):
                found.setdefault(name, candidate)
    for name in _INSTALLED_FILTER_RENDERER_ARGS:
        candidate = getattr(FilterManager, name, None)
        if callable(candidate):
            found.setdefault(name, candidate)
    return found


def _render_installed_filter(name: str) -> str:
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )
    from open_webui_openrouter_pipe.pipe import Pipe as _Pipe

    manager = cast("Any", FilterManager.__new__(FilterManager))
    manager._pipe = None
    manager._valves = _Pipe.Valves()

    kwargs = dict(_INSTALLED_FILTER_RENDERER_ARGS[name])
    if name in ("render_image_model_filter_source", "render_image_gen_filter_source"):
        kwargs["spec"] = build_image_model_filter_spec(
            "recraft/recraft-v3",
            {"id": "recraft/recraft-v3", "name": "Recraft V3"},
            [{
                "provider_slug": "recraft",
                "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1"]}},
                "allowed_passthrough_parameters": ["style"],
            }],
            dedicated_image_api=True,
        )

    if name == "render_router_filter_source":
        from open_webui_openrouter_pipe.filters.router_filter_renderer import (
            build_router_filter_spec,
        )
        from open_webui_openrouter_pipe.integrations.router_catalog import RouterOption

        kwargs["spec"] = build_router_filter_spec(
            "openrouter/auto",
            "auto-router",
            "Auto Router",
            (
                RouterOption(
                    name="cost_tier", field="COST_TIER", kind="enum", title="Cost tier",
                    description="How much it may spend.", choices=("low", "high"),
                ),
            ),
        )

    fn = _installed_filter_renderers()[name]
    if name == "render_openrouter_video_gen_filter_source":
        return fn(manager, **kwargs)
    return fn(**kwargs)


@pytest.mark.parametrize("renderer", sorted(_INSTALLED_FILTER_RENDERER_ARGS))
def test_every_template_the_pipe_installs_survives_open_webuis_import_rewrite(renderer):
    """The property that decides whether this guard ships or bricks every install.

    `replace_imports` is unanchored and Open WebUI writes its result back to the DB, so a
    template whose PROSE happens to contain 'from config' would be refused by the new
    guard -- and the pipe would then install nothing for that model. Checking the guard
    only against synthetic strings proves the guard works and says nothing about whether
    the real templates pass it.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    is_valid, error = FilterManager.validate_filter_source(_render_installed_filter(renderer))

    assert is_valid, (
        f"{renderer} renders a source Open WebUI would rewrite on load, so the pipe would "
        f"refuse to install it and that model loses its filter entirely: {error}"
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="the flat bundle shares one namespace, so a per-module scan cannot tell a "
    "filter renderer from a help renderer; the table's completeness is a source property",
)
def test_the_installed_filter_table_still_names_every_filter_renderer():
    """A renderer added without a table entry stops being checked against the guard."""
    import importlib

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    discovered: set[str] = set()
    for module_name in (
        "filter_manager", "fusion_filter_renderer",
        "image_filter_renderer", "router_filter_renderer", "video_filter_renderer",
    ):
        module = importlib.import_module(
            f"open_webui_openrouter_pipe.filters.{module_name}"
        )
        discovered.update(
            attr for attr, value in vars(module).items()
            if attr.startswith("render_") and callable(value)
        )
    discovered.update(
        attr for attr in dir(FilterManager)
        if "render_" in attr and callable(getattr(FilterManager, attr, None))
    )

    assert discovered == set(_INSTALLED_FILTER_RENDERER_ARGS), (
        "only in source: "
        f"{sorted(discovered - set(_INSTALLED_FILTER_RENDERER_ARGS))}; only in the table: "
        f"{sorted(set(_INSTALLED_FILTER_RENDERER_ARGS) - discovered)}. A renderer missing "
        "from the table is never checked against Open WebUI's import rewrite."
    )


@pytest.mark.parametrize(
    ("prose", "accepted"),
    [
        ("Read the value from main before the request goes out.", False),
        ("Pick the ratio from the model default, or leave this empty.", True),
    ],
)
def test_generated_prose_open_webui_would_rewrite_is_refused(prose, accepted):
    """OWUI's replace_imports is unanchored and writes its result back to the DB.

    A generated filter whose text is not a fixed point of that transform enters a
    permanent two-writer loop: OWUI rewrites it on load, the pipe sees a content
    mismatch on the next refresh and writes it back.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    source = f'X = 1\nDESCRIPTION = "{prose}"\n'
    is_valid, error = FilterManager.validate_filter_source(source)

    assert is_valid is accepted, (
        f"{prose!r} was {'accepted' if is_valid else 'refused'}; prose Open WebUI would "
        f"rewrite must be refused and prose it leaves alone must be accepted. {error}"
    )
    if not accepted:
        assert error and "rewrite" in error, (
            "the refusal must say why, or the author cannot tell it from a syntax error"
        )


def test_the_replace_imports_guard_runs_against_open_webui_s_own_rule():
    """Not a local copy of the four word-pairs: the check goes through OWUI's function."""
    import open_webui.utils.plugin as owp

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    calls: list[str] = []
    real = owp.replace_imports

    def _recording(content: str) -> str:
        calls.append(content)
        return real(content)

    owp.replace_imports = _recording
    try:
        FilterManager.validate_filter_source("X = 1\n")
    finally:
        owp.replace_imports = real

    assert calls == ["X = 1\n"], (
        "validate_filter_source did not call open_webui.utils.plugin.replace_imports, so "
        "it is checking its own copy of the rule and will not follow OWUI when it changes"
    )


@pytest.mark.parametrize(
    ("valve", "label"),
    [
        ("AUTO_INSTALL_IMAGE_FILTERS", "OpenRouter image filter for a/b"),
    ],
)
@pytest.mark.asyncio
async def test_a_row_that_will_never_be_updated_says_so_once(
    valve, label, caplog, monkeypatch
):
    """INSTALL off + ATTACH on finds the row, attaches it, and never rewrites it.

Without a signal every fix to that filter stays undelivered forever.
    """
    import logging
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.filters import filter_manager as fm_module
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    stale = SimpleNamespace(id="fn_1", content="STALE = 1", updated_at=1)

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [stale]

        @staticmethod
        async def get_function_by_id(function_id):
            return stale if function_id == "fn_1" else None

        @staticmethod
        async def update_function_by_id(function_id, updates):
            raise AssertionError(
                "the row was rewritten with the install valve off; this test would then "
                "be asserting a warning about a state that does not exist"
            )

    monkeypatch.setattr(functions_module, "Functions", _Table)
    fm_module._warned_stale_filter_rows.clear()

    logger = logging.getLogger(f"stale-row-test.{valve}")
    pipe = MagicMock()
    pipe.valves = SimpleNamespace(**{valve: False})
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=logger)

    async def _install():
        function_id, _write_not_installed = await manager._ensure_filter_installed(
            desired_source="FRESH = 1\n",
            desired_name="N",
            desired_meta={},
            preferred_id="fn_1",
            auto_install_valve=valve,
            log_label=label,
            matches_candidate=lambda content: True,
        )
        return function_id

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        first = await _install()
        second = await _install()

    assert first == "fn_1" and second == "fn_1", (
        "the row must still be returned and attached; the warning is about content, not "
        "about reachability"
    )
    emitted = [r for r in caplog.records if r.name == logger.name]
    warnings = [r for r in emitted if r.levelno >= logging.WARNING]
    repeats = [r for r in emitted if r.levelno == logging.DEBUG]

    assert len(warnings) == 1, (
        f"the stale row produced {len(warnings)} warnings across two refreshes; one is "
        "the contract -- none means the operator never learns, two means a line per refresh"
    )
    assert len(repeats) == 1, (
        "the repeat did not land at DEBUG, so an operator who raises the log level to "
        "diagnose a recurring fault sees strictly less than before"
    )
    message = warnings[0].getMessage()
    assert valve in message and "fn_1" in message and label in message, (
        f"the warning must name the blocking valve, the function id and the filter: {message}"
    )


@pytest.mark.asyncio
async def test_a_row_that_is_already_current_says_nothing(caplog, monkeypatch):
    """No warning when the stored source already matches: that row is not stale."""
    import logging
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.filters import filter_manager as fm_module
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    current = SimpleNamespace(id="fn_1", content="FRESH = 1", updated_at=1)

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [current]

    monkeypatch.setattr(functions_module, "Functions", _Table)
    fm_module._warned_stale_filter_rows.clear()

    logger = logging.getLogger("stale-row-test.current")
    pipe = MagicMock()
    pipe.valves = SimpleNamespace(AUTO_INSTALL_IMAGE_FILTERS=False)
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=logger)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        await manager._ensure_filter_installed(
            desired_source="FRESH = 1\n",
            desired_name="N",
            desired_meta={},
            preferred_id="fn_1",
            auto_install_valve="AUTO_INSTALL_IMAGE_FILTERS",
            log_label="OpenRouter image filter for a/b",
            matches_candidate=lambda content: True,
        )

    assert not [r for r in caplog.records if r.name == logger.name], (
        "a current row warned about being stale, so the signal is noise and gets muted"
    )


@pytest.mark.asyncio
async def test_a_valve_read_that_failed_leaves_the_installed_filter_alone():
    """A read that failed and a read that returned nothing are different answers.

    A read that raised already came back as `None`; a read that *returned* `None` came
    back as `""`, and `""` means "use the default model", so one transient database
    error during a catalog refresh rewrote the installed row for a model nobody had
    chosen -- while the valve still named the real one, so requests kept going to it
    with the wrong controls on screen. A dict that was read and simply holds no
    `IMAGE_GENERATION_MODEL` is the genuine unset case and must still install for the
    default model.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import pytest as _pytest

    from open_webui.models.functions import Functions

    from open_webui_openrouter_pipe.filters.filter_manager import (
        _OPENROUTER_IMAGE_GEN_FILTER_MARKER,
        FilterManager,
    )

    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    manager._ensure_filter_installed = AsyncMock(return_value=("openrouter_image_gen", _WriteOutcome()))

    installed = SimpleNamespace(id="openrouter_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)

    with _pytest.MonkeyPatch.context() as patched:
        patched.setattr(Functions, "get_functions_by_type", AsyncMock(return_value=[installed]))
        patched.setattr(
            Functions, "get_function_valves_by_id",
            AsyncMock(side_effect=RuntimeError("the database is locked")),
        )
        assert await manager.image_gen_filter_selected_model() is None, (
            "a read that raised must be distinguishable from a valve that is unset"
        )
        assert await manager.image_gen_filter_inputs() == (None, None, None, False)
        mapping, _ = await manager.ensure_openrouter_image_gen_filter_function_id()
        assert mapping is None
        assert manager._ensure_filter_installed.await_count == 0, (
            "a read that failed must not rewrite the installed row"
        )

    with _pytest.MonkeyPatch.context() as patched:
        patched.setattr(Functions, "get_functions_by_type", AsyncMock(return_value=[installed]))
        patched.setattr(Functions, "get_function_valves_by_id", AsyncMock(return_value={}))
        assert await manager.image_gen_filter_selected_model() == ""
        mapping, _ = await manager.ensure_openrouter_image_gen_filter_function_id()
        assert mapping == "openrouter_image_gen"
        assert manager._ensure_filter_installed.await_count == 1, (
            "a valve that is genuinely unset must still install for the default model"
        )


# REGFIX: "Frames: none" must mean none


@pytest.mark.parametrize("model_id", ["google/veo-3.1", "bytedance/seedance-2.0"])
@pytest.mark.parametrize(
    ("frame_mode", "any_picture_sent"), [("none", False)]
)
@pytest.mark.asyncio
async def test_setting_frames_to_none_stops_every_picture_reaching_the_model(
    model_id, frame_mode, any_picture_sent
):
    """The control's own text is "none ignores them", and billing follows the payload.

    Selecting no frames stopped pictures anchoring the shot but a later sweep collected
    every unclaimed attachment as a soft reference, so the pictures still steered the
    result and were still charged for. Two modes, because a filter that sends nothing
    ever satisfies the "none" row alone.
    """
    import json as _json
    from pathlib import Path as _Path

    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    catalog = {
        item["id"]: item
        for item in _json.loads(
            (_Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text()
        )["data"]
    }
    source = render_video_filter_source(model_id=model_id, video_model=catalog[model_id])
    module = _load_filter_from_source(source, f"frames_none_{model_id.replace('/', '_')}")
    instance = module.Filter()

    files = [
        {"id": f"img-{tag}", "name": f"{tag}.png", "content_type": "image/png"}
        for tag in ("A", "B", "C")
    ]
    body = {"model": model_id, "files": list(files), "messages": [{"role": "user", "content": "go"}]}
    signature = inspect.signature(instance.inlet)
    metadata: dict = {}
    supplied = {
        "__user__": {"valves": instance.UserValves(VIDEO_FRAME_MODE=frame_mode)},
        "__metadata__": metadata,
        "__model__": {},
        "__request__": None,
    }
    result = instance.inlet(
        body, **{k: v for k, v in supplied.items() if k in signature.parameters}
    )
    if inspect.isawaitable(result):
        await result

    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

    video_meta = (metadata.get(_PIPE_METADATA_KEY) or {}).get("video_generation") or {}
    reached = [
        item["id"]
        for key in ("frame_images", "input_references")
        for item in (video_meta.get(key) or [])
    ]
    assert bool(reached) is any_picture_sent, (
        f"{model_id} with frames={frame_mode!r} sent {reached}"
    )


from unittest.mock import AsyncMock, patch  # noqa: E402


@pytest.mark.asyncio
async def test_only_copies_installed_under_a_per_load_id_are_removed(monkeypatch):
    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.filters.filter_manager import _PIPE_INSTALLED_META_KEY, _FilterRows

    _FakeFunctionsTable.reset()
    monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
    pipe = Pipe()
    owner = pipe.id
    per_load = f"{owner}_0123456789abcdef0123456789abcdef"

    def _row(function_id: str, name: str, stamp: str) -> SimpleNamespace:
        row = SimpleNamespace(
            id=function_id,
            name=name,
            content="",
            meta={_PIPE_INSTALLED_META_KEY: stamp},
            is_active=True,
            is_global=False,
        )
        _FakeFunctionsTable.store[function_id] = row
        return row

    rows = [
        _row("openrouter_fusion", "OpenRouter Fusion", owner),
        _row("openrouter_fusion_1", "OpenRouter Fusion", per_load),
        _row("openrouter_fusion_2", "OpenRouter Fusion", per_load),
        _row("openrouter_image_filter_recraft_recraft_v4", "Recraft V4", owner),
        _row("openrouter_image_filter_recraft_recraft_v4_1", "Recraft V4.1", per_load),
        _row("someone_elses_filter", "Theirs", "another_pipe"),
        _row("someone_elses_filter_1", "Theirs", per_load),
        _row("openrouter_web_tools", "OR Web Tools", per_load),
        _row("openrouter_web_tools_1", "OR Web Tools", owner),
        _row("openrouter_direct_uploads", "OR Direct Uploads", per_load),
        _row("openrouter_direct_uploads_1", "OR Direct Uploads", per_load),
        _row("openrouter_image_gen", "OR Image Gen", per_load),
        _row("openrouter_image_gen_1", "OR Image Gen", owner),
        _row("openrouter_router_openrouter_auto", "OR Router", owner),
        _row("openrouter_router_openrouter_auto_1", "OR Router", per_load),
        _row("openrouter_video_gen", "OR Video Gen", per_load),
        _row("openrouter_video_gen_1", "OR Video Gen", owner),
        _row("openrouter_video_vendor_clip_3", "Clip", per_load),
        _row("openrouter_video_vendor_clip_3_1", "Clip", per_load),
    ]
    store = _FakeFunctionsTable.store
    store["openrouter_video_vendor_clip_3"].content = "VIDEO_MODEL_ID = 'vendor/clip-3'\n"
    store["openrouter_video_vendor_clip_3_1"].content = "VIDEO_MODEL_ID = 'vendor/clip-3-1'\n"
    store["openrouter_fusion_2"].is_active = False
    store["openrouter_direct_uploads_1"].valves = {"DIRECT_FILE_MIME_ALLOWLIST": "text/plain"}
    store["openrouter_image_gen"].is_active = False
    store["openrouter_router_openrouter_auto_1"].is_global = True
    store["openrouter_video_gen"].is_active = False
    store["openrouter_video_gen"].meta[_PIPE_OFF_META_KEY] = True
    try:
        removed = await pipe._ensure_filter_manager().remove_per_load_duplicates(_FilterRows(rows, None, True))
    finally:
        await pipe.close()

    assert removed == 4
    assert sorted(store) == [
        "openrouter_direct_uploads",
        "openrouter_direct_uploads_1",
        "openrouter_fusion",
        "openrouter_image_filter_recraft_recraft_v4",
        "openrouter_image_filter_recraft_recraft_v4_1",
        "openrouter_image_gen",
        "openrouter_image_gen_1",
        "openrouter_router_openrouter_auto",
        "openrouter_router_openrouter_auto_1",
        "openrouter_video_gen",
        "openrouter_video_vendor_clip_3",
        "openrouter_video_vendor_clip_3_1",
        "openrouter_web_tools",
        "someone_elses_filter",
        "someone_elses_filter_1",
    ]


@pytest.mark.asyncio
async def test_a_filter_stamped_with_the_pipes_id_and_a_per_load_tail_is_taken_over_in_place(monkeypatch):
    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import (
        _OPENROUTER_FUSION_FILTER_MARKER,
        _OPENROUTER_WEB_TOOLS_FILTER_MARKER,
    )

    _FakeFunctionsTable.reset()
    monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
    pipe = Pipe()
    owner = pipe.id
    tail = "0123456789abcdef0123456789abcdef"

    def _stale_row(function_id: str, name: str, marker: str, stamp: str) -> None:
        _FakeFunctionsTable.store[function_id] = SimpleNamespace(
            id=function_id,
            name=name,
            content=f"{marker}\nclass Filter:\n    pass\n",
            meta={_PIPE_INSTALLED_META_KEY: stamp},
            is_active=True,
            is_global=False,
            updated_at=1,
        )

    _stale_row("openrouter_fusion", "OpenRouter Fusion", _OPENROUTER_FUSION_FILTER_MARKER, f"{owner}_{tail}")
    _stale_row("openrouter_web_tools", "OR Web Tools", _OPENROUTER_WEB_TOOLS_FILTER_MARKER, f"another_pipe_{tail}")
    manager = pipe._ensure_filter_manager()
    try:
        fusion_id, _unresolved = await manager.ensure_openrouter_fusion_filter_function_id()
        web_tools_id = await manager.ensure_openrouter_web_tools_filter_function_id()
    finally:
        await pipe.close()

    store = _FakeFunctionsTable.store
    assert fusion_id == "openrouter_fusion"
    assert "def inlet" in store["openrouter_fusion"].content
    assert store["openrouter_fusion"].meta[_PIPE_INSTALLED_META_KEY] == owner
    assert "openrouter_fusion_1" not in store
    assert web_tools_id == "openrouter_web_tools_1"
    assert store["openrouter_web_tools"].meta[_PIPE_INSTALLED_META_KEY] == f"another_pipe_{tail}"
    assert "def inlet" not in store["openrouter_web_tools"].content


@pytest.mark.asyncio
async def test_the_settings_each_user_saved_on_a_copy_move_to_the_original_before_the_copy_is_deleted(monkeypatch):
    import open_webui.models.functions as functions_module
    import open_webui.models.models as models_module
    import open_webui.models.users as users_module
    from open_webui.models.models import ModelMeta

    from open_webui_openrouter_pipe.filters import filter_manager as filter_manager_module
    from open_webui_openrouter_pipe.filters.filter_manager import _FilterRows

    _FakeFunctionsTable.reset()
    monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
    monkeypatch.setattr(users_module, "Users", _FakeUsersTable)
    monkeypatch.setattr(filter_manager_module, "_USER_SCAN_PAGE_SIZE", 2)
    carrying = SimpleNamespace(
        id="carries_the_copy",
        base_model_id=None,
        name="Carries the copy",
        meta=ModelMeta(filterIds=["openrouter_fusion_2"], defaultFilterIds=["openrouter_fusion_2"]),
        params={},
        is_active=True,
    )
    re_pointed: list[str] = []

    async def _read_model(model_id, db=None):
        return carrying

    async def _save_model(model_id, form):
        re_pointed.append(model_id)
        carrying.meta = form.meta
        return SimpleNamespace(id=model_id)

    monkeypatch.setattr(models_module.Models, "get_all_models", AsyncMock(return_value=[carrying]))
    monkeypatch.setattr(models_module.Models, "get_model_by_id", _read_model)
    monkeypatch.setattr(models_module.Models, "update_model_by_id", _save_model)
    pipe = Pipe()
    owner = pipe.id

    for function_id, stamp in (
        ("openrouter_fusion", f"{owner}_0123456789abcdef0123456789abcdef"),
        ("openrouter_fusion_1", owner),
        ("openrouter_fusion_2", owner),
    ):
        _FakeFunctionsTable.store[function_id] = SimpleNamespace(
            id=function_id,
            name="OpenRouter Fusion",
            content="",
            meta={_PIPE_INSTALLED_META_KEY: stamp},
            is_active=True,
            is_global=False,
        )

    def _stored(values: dict) -> str:
        return _STORED_USER_VALVES_PREFIX + json.dumps(values, sort_keys=True)

    already_on_the_original = _stored({"FUSION_PRESET": "mine"})
    _FakeFunctionsTable.user_valves = {
        "u1": {
            "openrouter_fusion_1": _stored({"FUSION_PRESET": "older"}),
            "openrouter_fusion_2": _stored({"FUSION_PRESET": "newer"}),
        },
        "u2": {
            "openrouter_fusion_1": _stored({"FUSION_PRESET": "copy"}),
            "openrouter_fusion": already_on_the_original,
        },
        "u3": {"openrouter_fusion_1": _stored({"FUSION_PRESET": "only"})},
        "u4": {"some_other_filter": _stored({"unrelated": 1})},
    }
    manager = pipe._ensure_filter_manager()

    def _table() -> _FilterRows:
        return _FilterRows(list(_FakeFunctionsTable.store.values()), None, True)

    try:
        _FakeFunctionsTable.refuse_user_writes = True
        unsettled_while_refused: set[str] = set()
        removed_while_refused = await manager.remove_per_load_duplicates(_table(), unsettled_while_refused)
        kept_while_refused = sorted(_FakeFunctionsTable.store)
        re_pointed_while_refused = list(re_pointed)
        filter_ids_while_refused = list(dict(carrying.meta)["filterIds"])
        _FakeFunctionsTable.refuse_user_writes = False
        unsettled: set[str] = set()
        removed = await manager.remove_per_load_duplicates(_table(), unsettled)
    finally:
        await pipe.close()

    saved = _FakeFunctionsTable.user_valves
    assert removed_while_refused == 0
    assert unsettled_while_refused == {"openrouter_fusion_1", "openrouter_fusion_2"}
    assert kept_while_refused == ["openrouter_fusion", "openrouter_fusion_1", "openrouter_fusion_2"]
    assert re_pointed_while_refused == []
    assert filter_ids_while_refused == ["openrouter_fusion_2"]
    assert removed == 2
    assert not unsettled
    assert re_pointed == ["carries_the_copy"]
    assert dict(carrying.meta)["filterIds"] == ["openrouter_fusion"]
    assert dict(carrying.meta)["defaultFilterIds"] == ["openrouter_fusion"]
    assert sorted(_FakeFunctionsTable.store) == ["openrouter_fusion"]
    assert await _FakeFunctionsTable.get_user_valves_by_id_and_user_id("openrouter_fusion", "u1") == {
        "FUSION_PRESET": "newer"
    }
    assert saved["u2"]["openrouter_fusion"] == already_on_the_original
    assert await _FakeFunctionsTable.get_user_valves_by_id_and_user_id("openrouter_fusion", "u3") == {
        "FUSION_PRESET": "only"
    }
    assert "openrouter_fusion" not in saved["u4"]
    assert "openrouter_fusion_1" in saved["u1"]


def _stage_the_first_refresh_after_an_upgrade(monkeypatch, pipe):
    import open_webui.models.functions as functions_module
    import open_webui.models.models as models_module
    from open_webui.models.models import ModelMeta

    from open_webui_openrouter_pipe import pipe as pipe_module
    from open_webui_openrouter_pipe.core.config import (
        _OPENROUTER_FUSION_FILTER_MARKER,
        _OPENROUTER_WEB_TOOLS_FILTER_MARKER,
    )

    owner = pipe.id
    _FakeFunctionsTable.reset()
    for function_id, name, marker, stamp, updated_at in (
        ("openrouter_fusion", "OpenRouter Fusion", _OPENROUTER_FUSION_FILTER_MARKER, f"{owner}_0123456789abcdef0123456789abcdef", 1),
        ("openrouter_fusion_1", "OpenRouter Fusion", _OPENROUTER_FUSION_FILTER_MARKER, owner, 2),
        ("openrouter_web_tools", "OR Web Tools", _OPENROUTER_WEB_TOOLS_FILTER_MARKER, f"{owner}_0123456789abcdef0123456789abcdef", 1),
        ("openrouter_web_tools_1", "OR Web Tools", _OPENROUTER_WEB_TOOLS_FILTER_MARKER, owner, 2),
    ):
        _FakeFunctionsTable.store[function_id] = SimpleNamespace(
            id=function_id,
            name=name,
            content=f"{marker}\nclass Filter:\n    pass\n",
            meta={_PIPE_INSTALLED_META_KEY: stamp},
            is_active=True,
            is_global=False,
            updated_at=updated_at,
        )
    carrying = SimpleNamespace(
        id="carries_the_copies",
        base_model_id=None,
        name="Carries the copies",
        meta=ModelMeta(
            filterIds=["openrouter_fusion_1", "openrouter_web_tools_1"],
            defaultFilterIds=["openrouter_fusion_1"],
        ),
        params={},
        is_active=True,
    )
    saves = {"refused": False}

    async def _read(model_id, db=None):
        return carrying

    async def _save(model_id, form):
        if saves["refused"]:
            return None
        carrying.meta = form.meta
        return SimpleNamespace(id=model_id)

    async def _no_stored_settings(function_id, db=None):
        return {}

    monkeypatch.setattr(functions_module, "Functions", _FakeFunctionsTable)
    monkeypatch.setattr(pipe_module, "_PER_LOAD_LEFTOVERS_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(_FakeFunctionsTable, "get_function_valves_by_id", _no_stored_settings, raising=False)
    monkeypatch.setattr(models_module.Models, "get_all_models", AsyncMock(return_value=[carrying]))
    monkeypatch.setattr(models_module.Models, "get_model_by_id", _read)
    monkeypatch.setattr(models_module.Models, "update_model_by_id", _save)
    monkeypatch.setattr(pipe, "_maybe_start_startup_checks", lambda: None)
    monkeypatch.setattr(pipe, "_maybe_start_redis", lambda: None)
    monkeypatch.setattr(pipe, "_maybe_start_cleanup", lambda: None)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: (None, None))
    monkeypatch.setattr(
        pipe_module.OpenRouterModelRegistry, "list_models", lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}]
    )
    monkeypatch.setattr(pipe, "_select_models", lambda _model_id, models: models)
    monkeypatch.setattr(pipe, "_apply_model_filters", lambda models, _valves: models)
    monkeypatch.setattr(pipe._ensure_catalog_manager(), "maybe_schedule_model_metadata_sync", lambda *_a, **_k: None)
    return carrying, saves


@pytest.mark.asyncio
async def test_the_first_model_list_refresh_settles_the_copies_before_it_installs_any_filter(monkeypatch):
    pipe = Pipe()
    carrying, _saves = _stage_the_first_refresh_after_an_upgrade(monkeypatch, pipe)
    try:
        await pipe.pipes()
    finally:
        await pipe.close()

    store = _FakeFunctionsTable.store
    assert "openrouter_fusion_1" not in store
    assert "openrouter_web_tools_1" not in store
    assert store["openrouter_fusion"].meta[_PIPE_INSTALLED_META_KEY] == pipe.id
    assert store["openrouter_web_tools"].meta[_PIPE_INSTALLED_META_KEY] == pipe.id
    assert dict(carrying.meta)["filterIds"] == ["openrouter_fusion", "openrouter_web_tools"]
    assert dict(carrying.meta)["defaultFilterIds"] == ["openrouter_fusion"]


@pytest.mark.asyncio
async def test_a_refresh_that_could_not_read_or_save_what_it_needs_removes_no_copy_and_a_later_one_removes_them(
    monkeypatch,
):
    pipe = Pipe()
    carrying, saves = _stage_the_first_refresh_after_an_upgrade(monkeypatch, pipe)
    real_rows_by_type = _FakeFunctionsTable.get_functions_by_type
    pipes_readable = {"now": False}

    async def _rows_by_type(function_type, active_only=False):
        if function_type == "pipe" and not pipes_readable["now"]:
            raise RuntimeError("function table unreadable")
        return await real_rows_by_type(function_type, active_only)

    monkeypatch.setattr(_FakeFunctionsTable, "get_functions_by_type", _rows_by_type)

    def _copies_left() -> list[str]:
        return sorted(function_id for function_id in _FakeFunctionsTable.store if function_id.endswith("_1"))

    try:
        await pipe.pipes()
        left_while_the_loaded_pipes_were_unreadable = _copies_left()
        pipes_readable["now"] = True
        saves["refused"] = True
        await pipe.pipes()
        left_after_the_refusal = _copies_left()
        saves["refused"] = False
        await pipe.pipes()
    finally:
        await pipe.close()

    store = _FakeFunctionsTable.store
    assert left_while_the_loaded_pipes_were_unreadable == ["openrouter_fusion_1", "openrouter_web_tools_1"]
    assert left_after_the_refusal == ["openrouter_fusion_1", "openrouter_web_tools_1"]
    assert "openrouter_fusion_1" not in store
    assert "openrouter_web_tools_1" not in store
    assert store["openrouter_web_tools"].meta[_PIPE_INSTALLED_META_KEY] == pipe.id
    assert dict(carrying.meta)["filterIds"] == ["openrouter_fusion", "openrouter_web_tools"]


async def _send_pictures_to_a_video_model(
    monkeypatch,
    *,
    frame_types,
    description,
    pictures,
    frame_mode="auto",
    model_id="acme/glimmer-video-1",
    helper=None,
    sound_files=(),
    earlier_turns=(),
):
    import base64

    import aiohttp
    from aioresponses import CallbackResult
    from open_webui.models.config import Config as _OwuiConfig

    from open_webui_openrouter_pipe import EncryptedStr
    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY
    from open_webui_openrouter_pipe.filters.video_filter_renderer import render_video_filter_source
    from open_webui_openrouter_pipe.integrations import video as video_module
    from open_webui_openrouter_pipe.integrations import video_intent as video_intent_module
    from tests.media_slots import bare_slot
    from tests.posted_body import posted_json

    video_model = {
        "id": model_id,
        "name": "Acme: Glimmer Video 1",
        "supported_frame_images": list(frame_types),
        "supported_durations": [4, 8],
        "supported_aspect_ratios": ["16:9"],
        "supported_resolutions": ["720p"],
        "allowed_passthrough_parameters": [],
    }
    row = {
        "slug": model_id,
        "description": description,
        "endpoint": {"model_variant_slug": model_id, "provider_info": {"slug": "acme"}},
    }
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 1
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 1
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    manager = pipe._ensure_catalog_manager()
    manager._update_or_insert_model_with_metadata = AsyncMock()
    monkeypatch.setattr(
        manager, "_fetch_frontend_model_catalog", AsyncMock(return_value={"data": [row]})
    )
    try:
        await manager._sync_model_metadata_to_owui(
            [{"id": model_id.replace("/", "."), "name": "Glimmer", "original_id": model_id}],
            pipe_identifier="test_pipe",
        )
        OpenRouterModelRegistry.register_video_models([video_model])

        module = _load_filter_from_source(
            render_video_filter_source(model_id=model_id, video_model=video_model),
            "glimmer_video_filter",
        )
        instance = module.Filter()
        metadata: dict = {"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"}
        instance.inlet(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "go"}],
                "files": [
                    {"id": f"file-{name}", "name": name, "content_type": "image/png"}
                    for name in pictures
                ]
                + [
                    {"id": f"file-{name}", "name": name, "content_type": "audio/mpeg"}
                    for name in sound_files
                ],
            },
            __metadata__=metadata,
            __user__={"valves": instance.UserValves(VIDEO_FRAME_MODE=frame_mode)},
            __model__={},
        )
        assert metadata[_PIPE_METADATA_KEY]["video_generation"], "the filter wrote no request"

        async def _file(file_id, _logger):
            return SimpleNamespace(id=file_id, filename=file_id, user_id="user-1")

        async def _bytes(file_obj, *_args, **_kwargs):
            return bare_slot(base64.b64encode(str(file_obj.id).encode()).decode())

        sound_ids = {f"file-{name}" for name in sound_files}
        monkeypatch.setattr(video_module, "get_file_by_id", _file)
        monkeypatch.setattr(
            video_module,
            "infer_file_mime_type",
            lambda file_obj: "audio/mpeg" if file_obj.id in sound_ids else "image/png",
        )
        monkeypatch.setattr(pipe._file_gateway, "read_file_record_media", _bytes)

        class _Chat:
            async def load_message_content(self, **_kwargs):
                return ""

        cast(Any, pipe._ensure_video_generation_adapter())._persistence = _Chat()
        asked: list[dict] = []
        if helper is not None:
            _OwuiConfig._rows["task.model.default"] = "task-llm"
            _OwuiConfig._rows["task.model.external"] = "task-llm"
            reply = {
                "attachments_words": "",
                "attachments_as": "not_said",
                "intent": "image_to_video",
                "frame_plan": [],
                "prompt": "go",
                "use_user_prompt": False,
                "language": "en",
                "confidence": "high",
                "clarification": {"needs": False, "question": "", "options": None, "reason": ""},
                "reason": "the test's helper",
                **helper,
            }

            async def _answer(form_data):
                asked.append(form_data)
                return {"choices": [{"message": {"content": json.dumps(reply)}}]}

            monkeypatch.setattr(
                video_intent_module, "_make_default_invoke", lambda _request, _user: _answer
            )
        posted: list[dict] = []

        def _accept(_url, **kwargs):
            posted.append(posted_json(kwargs))
            return CallbackResult(status=200, payload={"id": "job-1", "status": "pending"})

        base = pipe.valves.BASE_URL
        with aioresponses() as http:
            http.post(f"{base}/videos", callback=_accept)
            http.get(
                f"{base}/videos/job-1",
                payload={"status": "failed", "error": "stopped for this test"},
                repeat=True,
            )
            async with aiohttp.ClientSession() as session:
                card = await pipe._ensure_video_generation_adapter().generate(
                    body={"messages": [*earlier_turns, {"role": "user", "content": "go"}]},
                    responses_body=SimpleNamespace(provider={}),
                    valves=pipe.valves,
                    session=session,
                    event_emitter=None,
                    metadata=metadata,
                    user={"id": "user-1"},
                    request=None,
                    user_obj=SimpleNamespace(id="user-1", email="u@example.com", name="u"),
                    normalized_model_id=model_id.replace("/", "."),
                    api_model_id=model_id,
                )
    finally:
        await pipe.close()
    assert len(posted) == 1, f"OpenRouter was sent {len(posted)} requests"
    assert bool(asked) is (helper is not None), f"the helper was asked {len(asked)} time(s)"
    return posted[0], card


def _data_url_of(name: str) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(f"file-{name}".encode()).decode()


@pytest.mark.asyncio
async def test_two_pictures_for_a_model_that_mentions_references_all_go_in_as_references(
    monkeypatch,
):
    sent, _card = await _send_pictures_to_a_video_model(
        monkeypatch,
        frame_types=["first_frame"],
        description="Glimmer makes short clips from a prompt, a first frame or reference images.",
        pictures=["environment.png", "character.png"],
    )

    assert "frame_images" not in sent, sent
    assert [entry["image_url"]["url"] for entry in sent["input_references"]] == [
        _data_url_of("environment.png"),
        _data_url_of("character.png"),
    ]


@pytest.mark.asyncio
async def test_pictures_beyond_the_frames_are_named_not_sent_for_a_model_silent_on_references(
    monkeypatch,
):
    sent, card = await _send_pictures_to_a_video_model(
        monkeypatch,
        frame_types=["first_frame", "last_frame"],
        description="Glimmer 2 makes short clips from a prompt or from a first and a last frame.",
        pictures=["start.png", "middle.png", "end.png"],
    )

    assert [(f["frame_type"], f["image_url"]["url"]) for f in sent["frame_images"]] == [
        ("first_frame", _data_url_of("start.png")),
        ("last_frame", _data_url_of("end.png")),
    ]
    assert "input_references" not in sent, sent
    assert "Not sent with this video" in card and "middle.png" in card, card


@pytest.mark.asyncio
async def test_a_frames_choice_keeps_the_frames_on_a_model_that_mentions_references(monkeypatch):
    sent, card = await _send_pictures_to_a_video_model(
        monkeypatch,
        frame_types=["first_frame", "last_frame"],
        description="Glimmer 3 makes short clips from a prompt, frames or reference images.",
        pictures=["start.png", "middle.png", "end.png"],
        frame_mode="first_only",
    )

    assert [f["image_url"]["url"] for f in sent["frame_images"]] == [_data_url_of("start.png")]
    assert "input_references" not in sent, sent
    assert "middle.png" in card and "end.png" in card, card


@pytest.mark.asyncio
async def test_a_model_measured_to_ignore_reference_pictures_gets_frames_though_its_description_mentions_them(
    monkeypatch,
):
    sent, card = await _send_pictures_to_a_video_model(
        monkeypatch,
        model_id="minimax/hailuo-2.3",
        frame_types=["first_frame"],
        description="A model that makes short clips from a prompt, a first frame or reference images.",
        pictures=["environment.png", "character.png"],
    )

    assert "input_references" not in sent, sent
    assert [f["image_url"]["url"] for f in sent["frame_images"]] == [_data_url_of("environment.png")]
    assert "character.png" in card, card


@pytest.mark.asyncio
async def test_the_helper_saying_references_sends_every_picture_as_a_reference_though_the_frames_fit(
    monkeypatch,
):
    sent, _card = await _send_pictures_to_a_video_model(
        monkeypatch,
        frame_types=["first_frame", "last_frame"],
        description="Glimmer 4 makes short clips from a prompt, frames or reference images.",
        pictures=["environment.png", "character.png"],
        helper={
            "attachments_words": "use the first picture as the environment and the second as the character",
            "attachments_as": "references",
        },
    )

    assert "frame_images" not in sent, sent
    assert [entry["image_url"]["url"] for entry in sent["input_references"]] == [
        _data_url_of("environment.png"),
        _data_url_of("character.png"),
    ]


@pytest.mark.asyncio
async def test_the_helper_saying_frames_keeps_the_frames_and_names_the_rest_on_a_model_that_mentions_references(
    monkeypatch,
):
    sent, card = await _send_pictures_to_a_video_model(
        monkeypatch,
        frame_types=["first_frame", "last_frame"],
        description="Glimmer 5 makes short clips from a prompt, frames or reference images.",
        pictures=["start.png", "middle.png", "end.png"],
        helper={
            "attachments_words": "start from the first picture and end on the last one",
            "attachments_as": "frames",
            "frame_plan": [
                {"source": "uploaded_attachment", "source_index": 0, "timestamp_seconds": None, "target": "first_frame"},
                {"source": "uploaded_attachment", "source_index": 2, "timestamp_seconds": None, "target": "last_frame"},
            ],
        },
    )

    assert "input_references" not in sent, sent
    assert [(f["frame_type"], f["image_url"]["url"]) for f in sent["frame_images"]] == [
        ("first_frame", _data_url_of("start.png")),
        ("last_frame", _data_url_of("end.png")),
    ]
    assert "middle.png" in card, card


def _footer_heading(card: str) -> str:
    quoted = [line for line in card.splitlines() if line.startswith("> ")]
    assert quoted, card
    return quoted[0]


@pytest.mark.asyncio
async def test_the_footer_heading_follows_whether_the_request_sends_a_picture(monkeypatch):
    chat = {
        "frame_types": ["first_frame", "last_frame"],
        "description": "Glimmer 6 makes short clips from a prompt or from a first and a last frame.",
        "earlier_turns": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }

    held_back, held_back_card = await _send_pictures_to_a_video_model(
        monkeypatch,
        **chat,
        pictures=["start.png"],
        sound_files=["voice.mp3"],
        frame_mode="none",
        helper={"intent": "image_to_video", "prompt": "a short clip"},
    )
    with_frame, with_frame_card = await _send_pictures_to_a_video_model(
        monkeypatch,
        **chat,
        pictures=["start.png"],
        helper={"intent": "image_to_video", "prompt": "a short clip"},
    )
    _, text_only_card = await _send_pictures_to_a_video_model(
        monkeypatch,
        **chat,
        pictures=[],
        helper={"intent": "text_to_video", "prompt": "a short clip"},
    )

    assert "frame_images" not in held_back and "input_references" not in held_back, held_back
    assert [f["image_url"]["url"] for f in with_frame["frame_images"]] == [_data_url_of("start.png")]
    assert _footer_heading(held_back_card) == _footer_heading(text_only_card), held_back_card
    assert _footer_heading(with_frame_card) != _footer_heading(text_only_card), with_frame_card
