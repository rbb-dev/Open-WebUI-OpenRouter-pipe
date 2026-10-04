from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import re
import tempfile
import time
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.doc_truth_anchors import config_meta_detail, doc, valve_field_description
from tests.pipe_limits import slot, set_slot
from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY
from open_webui_openrouter_pipe.core.errors import (
    OpenRouterAPIError,
    RequiredInternalFileError,
    _build_openrouter_api_error,
)
from open_webui_openrouter_pipe.filters import FilterManager
from open_webui_openrouter_pipe.filters.filter_manager import _WriteOutcome
from open_webui_openrouter_pipe.filters.video_filter_renderer import (
    _CONTROL_TEXT,
    _PASSTHROUGH_CONTROLS,
    build_video_filter_spec,
    render_video_filter_source,
)
from open_webui_openrouter_pipe.integrations.provider_options import (
    payload_addresses,
    requested_provider_block,
    requested_provider_options,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_intent import (
    FramePlanEntry,
    FrameTargetLiteral,
    VideoIntentResult,
    collect_attachments_from_video_meta,
)
from open_webui_openrouter_pipe.integrations.video_client import (
    OpenRouterVideoClient,
    extension_for_video_mime,
)
from open_webui_openrouter_pipe.storage.multimodal import _sniff_mime_from_prefix
from open_webui_openrouter_pipe.integrations.video_help import VIDEO_HELP_BY_MODEL, render_video_help
from open_webui_openrouter_pipe.integrations.video_types import (
    DownloadedVideo,
    VideoGenerationError,
    VideoGenerationStalled,
    VideoLifecycleResult,
    VideoStatusUnavailable,
)
from open_webui_openrouter_pipe.models.registry import ModelFamily, OpenRouterModelRegistry
from open_webui_openrouter_pipe.storage.video_persistence import VideoPersistence
from open_webui_openrouter_pipe.integrations.video_intent import VideoIntentResult
from open_webui_openrouter_pipe.integrations import video as video_module
from open_webui_openrouter_pipe.storage import owui_files as owui_files_module
from tests.test_filters import _load_filter_from_source as _compile_rendered_filter
import pytest_asyncio

_CALLER = SimpleNamespace(id="caller-1", role="user", email="caller@example.com")



async def _a_listening_chat(_event):
    """A chat whose socket is attached, which is the precondition for any upload.

    `_encode_input_references` refuses to publish a user's file when it cannot say so
    first, so a test about anything else has to supply a channel that works.
    """
    return None

NEEDS_A_PROMPT_REASON = (
    "Video generation needs a prompt in your message. Add words describing the video you want "
    "\u2014 an attachment alone is not enough."
)
_VIDEO_CATALOG_FIXTURE = Path(__file__).parent / "fixtures" / "video_models_catalog.json"
VIDEO_MODELS = json.loads(_VIDEO_CATALOG_FIXTURE.read_text())["data"]
VIDEO_BY_ID = {item["id"]: item for item in VIDEO_MODELS}
def _load_filter_from_source(source: str, module_name: str) -> ModuleType:
    if "open_webui.env" not in sys.modules:
        env_mock = ModuleType("open_webui.env")
        env_mock.SRC_LOG_LEVELS = {}  # type: ignore[attr-defined]
        sys.modules["open_webui.env"] = env_mock

    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}_rendered_source>"
    sys.modules[module_name] = module
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)
    module.Filter.UserValves.model_rebuild()
    module.Filter.Valves.model_rebuild()
    return module


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, status: int = 200) -> None:
        self.status = status
        self.content = _FakeContent(chunks)
        self.headers: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return {}

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def get(self, *_args, **_kwargs):
        return _FakeResponse(self.chunks)

    async def close(self) -> None:
        self.closed = True


class _MemoryPersistence:

    def __init__(self, initial: str = "") -> None:
        self.content = initial
        self.persisted: list[str] = []
        self.stored: list[str] = []

    async def load_message_content(self, *, chat_id: str, message_id: str, user: Any = None) -> str:
        return self.content

    async def store_video_file_from_path(self, **kwargs) -> str:
        self.stored.append(str(kwargs["source_path"]))
        return "file-1"


class _StubCatalogManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_cached_provider_map(self):
        return self._mapping


"""Sentinel for a capability key the catalogue does not carry at all.

`dict.get` answers None for both a published null and an absent key, and those two are
opposite decisions: a published null is a third state the model leaves to its own default,
an absent key is a model that never mentioned the capability.
"""


class _Row:
    """Open WebUI's row for one assistant message."""

    def __init__(self) -> None:
        self.content = ""

    async def load(self, *, chat_id: str, message_id: str, user: Any = None) -> str:
        if chat_id.startswith(("temporary:", "local:", "channel:")):
            return ""
        return self.content

    def append(self, delta: str) -> None:
        self.content += delta


def _sentences(text: str) -> list[str]:
    """The statements a card actually makes, as the card itself wrote them."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped or stripped.startswith("[openrouter:"):
            continue
        for part in re.split(r"(?<=[.!?])\s+", stripped):
            candidate = part.strip()
            if len(candidate) >= 15:
                out.append(candidate)
    return out


_ATLAS = doc("valves_and_configuration_atlas.md")


"""Published parameters still rendered by the generic fallback rather than a descriptor.

A tripwire, not a description. The fallback gives a control titled with the raw wire name and
a description that can say nothing about values, because there is nothing to say it from. That
is the honest rendering for a parameter nobody has looked up yet -- but it should be a decision,
not a default, so a new one has to be argued for here.

Empty today: all thirty-nine parameters the catalogue publishes carry a descriptor in
`_PASSTHROUGH_CONTROLS`, with a vendor citation wherever a closed domain is offered.
"""


"""Every value the video filter offers from a closed domain, written out in full.

A tripwire, not a description. Unlike the image side -- where the values come from
OpenRouter's own contract and cannot drift from it -- these are read off a vendor page by
hand. Nothing else notices if one is quietly deleted, and a user then loses a choice the
model still accepts, silently. Naming them here makes a change two deliberate edits.
"""


def _collect(video_meta):
    return collect_attachments_from_video_meta(video_meta)
