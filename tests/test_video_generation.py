from __future__ import annotations

import asyncio
import base64
import contextlib
import time
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.core.errors import RequiredInternalFileError
from open_webui_openrouter_pipe.filters import FilterManager
from open_webui_openrouter_pipe.filters.video_filter_renderer import (
    build_video_filter_spec,
    render_video_filter_source,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_client import (
    OpenRouterVideoClient,
    extension_for_video_mime,
)
from open_webui_openrouter_pipe.storage.multimodal import _sniff_mime_from_prefix
from open_webui_openrouter_pipe.integrations.video_help import VIDEO_HELP_BY_MODEL, render_video_help
from open_webui_openrouter_pipe.integrations.video_types import (
    DownloadedVideo,
    VideoGenerationError,
    VideoLifecycleResult,
)
from open_webui_openrouter_pipe.models.registry import ModelFamily, OpenRouterModelRegistry
from open_webui_openrouter_pipe.storage.video_persistence import VideoPersistence


_VIDEO_CATALOG_FIXTURE = Path(__file__).parent / "fixtures" / "video_models_catalog.json"
VIDEO_MODELS = json.loads(_VIDEO_CATALOG_FIXTURE.read_text())["data"]
VIDEO_BY_ID = {item["id"]: item for item in VIDEO_MODELS}
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


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


def _test_logger() -> logging.Logger:
    return logging.getLogger("tests.video_generation")


def _async_return(value: Any):
    """Build an async callable that ignores its args and returns ``value``."""
    async def _inner(*_args, **_kwargs):
        return value
    return _inner


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

    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        return self.content

    async def store_video_file_from_path(self, **kwargs) -> str:
        self.stored.append(str(kwargs["source_path"]))
        return "file-1"


async def _consume_pipe_result(result: Any) -> Any:
    if hasattr(result, "__aiter__"):
        items: list[Any] = []
        async for item in result:
            items.append(item)
        return items
    return result



class _StubCatalogManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_cached_provider_map(self):
        return self._mapping


def _pipe_with_provider_map(mapping) -> Pipe:
    pipe = Pipe()
    pipe._catalog_manager = cast(Any, _StubCatalogManager(mapping))
    return pipe


def _provider_params(payload, slug):
    return ((payload.get("provider") or {}).get("options") or {}).get(slug) or {}


_ROUTED_SLUGS = {
    "alibaba/wan-2.6": "atlas-cloud",
    "alibaba/wan-2.7": "atlas-cloud",
    "google/veo-3.1": "google-vertex",
    "kwaivgi/kling-v3.0-std": "atlas-cloud",
    "bytedance/seedance-2.0": "seed",
}

_ROUTED_PROVIDER_MAP = {
    model_id: {"providers": [slug]} for model_id, slug in _ROUTED_SLUGS.items()
}

_NOT_PUBLISHED = object()
"""Sentinel for a capability key the catalogue does not carry at all.

`dict.get` answers None for both a published null and an absent key, and those two are
opposite decisions: a published null is a third state the model leaves to its own default,
an absent key is a model that never mentioned the capability.
"""


@pytest.mark.parametrize(
    ("flag", "declared", "promoted"),
    [
        ("seed", True, True),
        ("seed", False, False),
        ("seed", 1, False),
        ("seed", {"min": 0, "max": 9}, False),
        ("generate_audio", True, True),
        ("generate_audio", 1, False),
        ("generate_audio", "yes", False),
        ("seed", None, True),
        ("generate_audio", None, True),
    ],
)
def test_a_capability_flag_promotes_only_on_a_published_boolean(flag, declared, promoted):
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    top_level, _ = VideoGenerationAdapter._split_allowed_parameters(
        object.__new__(VideoGenerationAdapter), {flag: declared}
    )

    assert (flag in top_level) is promoted, (
        "the image half of this changeset reads capabilities as descriptor objects, so a "
        "video catalog adopting that shape is exactly the drift this conjunct decides; "
        "a truthy read would ship a range descriptor as if it were the flag"
    )


@pytest.mark.parametrize("declared", [[], {}, "16:9", None, 0])
def test_a_non_list_capability_declaration_does_not_promote_the_knob(declared):
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    top_level, _ = VideoGenerationAdapter._split_allowed_parameters(
        object.__new__(VideoGenerationAdapter), {"supported_aspect_ratios": declared}
    )

    assert ("aspect_ratio" in top_level) is isinstance(declared, list), (
        "only a published list of ratios makes aspect_ratio a documented top-level field; "
        "anything else must leave it in provider passthrough where OpenRouter can reject it"
    )


def test_video_registry_marks_models_non_zdr_and_filterable():
    OpenRouterModelRegistry._zdr_model_ids = {"openai.sora-2-pro"}
    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID["openai/sora-2-pro"]])

    assert ModelFamily.supports("video_generation", "openai.sora-2-pro") is True
    assert OpenRouterModelRegistry.is_zdr_capable("openai.sora-2-pro") is False

    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    filtered = pipe._apply_model_filters(OpenRouterModelRegistry.list_models(), pipe.valves)
    assert filtered == []


@pytest.mark.parametrize("listed_as_zdr", [True, False])
def test_a_transport_that_cannot_carry_retention_outranks_the_zdr_roster(listed_as_zdr):
    """The exemption has to beat set membership, not merely fill in when the set is silent.

    A model reachable only over the image API cannot carry a retention key at all, so an
    upstream roster listing it as ZDR-capable is answering about a transport this model
    never uses. Enforcement must refuse it rather than generate with the control stripped.
    """
    OpenRouterModelRegistry._specs["vendor.image-only"] = {
        "architecture": {"output_modalities": ["image"]},
        "features": {"image_output"},
    }
    OpenRouterModelRegistry._zdr_model_ids = (
        {"vendor.image-only"} if listed_as_zdr else set()
    )

    assert OpenRouterModelRegistry.is_zdr_capable("vendor.image-only") is False, (
        "an image-only model has no transport that defines a retention key; being named on "
        f"the ZDR roster cannot change that. listed_as_zdr={listed_as_zdr}"
    )


def test_model_specific_filters_hide_unsupported_controls():
    sora_source = render_video_filter_source(model_id="openai/sora-2-pro", video_model=VIDEO_BY_ID["openai/sora-2-pro"])
    hailuo_source = render_video_filter_source(model_id="minimax/hailuo-2.3", video_model=VIDEO_BY_ID["minimax/hailuo-2.3"])
    kling_source = render_video_filter_source(
        model_id="kwaivgi/kling-video-o1",
        video_model=VIDEO_BY_ID["kwaivgi/kling-video-o1"],
    )
    wan_source = render_video_filter_source(model_id="alibaba/wan-2.7", video_model=VIDEO_BY_ID["alibaba/wan-2.7"])

    for source in (sora_source, hailuo_source, kling_source, wan_source):
        valid, error = FilterManager.validate_filter_source(source)
        assert valid, error

    assert "VIDEO_FRAME_MODE" not in sora_source
    assert "VIDEO_AUDIO_URL" not in hailuo_source
    assert "VIDEO_GENERATE_AUDIO" not in hailuo_source
    assert "VIDEO_SEED" not in kling_source
    assert "VIDEO_REFERENCE_IMAGES_JSON" in wan_source
    assert "VIDEO_REFERENCE_VIDEOS_JSON" in wan_source
    assert "VIDEO_AUDIO_URL" in wan_source


def test_build_success_content_ends_with_newline():
    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    content = adapter._build_success_content(
        job_id="job-abc",
        model_id="google/veo-3.1-lite",
        file_ids=["file-xyz"],
        elapsed=12.3,
        usage={"cost": 0.4},
    )
    assert content.endswith("\n"), "success content must end with \\n so concatenation cannot smash markers into inline text"
    doubled = content + content
    assert "\n[openrouter:v1:videojob:job-abc]: #" in doubled
    assert "Generated in" not in content, "success content must not embed time/cost text — those go through the status emitter"


@pytest.mark.asyncio
async def test_video_lifecycle_bg_task_does_not_emit_chat_completion(monkeypatch, tmp_path):
    from open_webui_openrouter_pipe.integrations.video_types import VideoLifecycleResult

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def status(self, _job_id, polling_url=None):
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id, index=0):
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self):
            return {"Authorization": "Bearer test"}

    async def fake_streaming_download(url, dest_path, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4_BYTES)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": len(MP4_BYTES)}

    async def fake_upload_from_path(*_args, **_kwargs):
        return "file-1"

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient)
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_args, **_kwargs: _FakeSession([]))
    monkeypatch.setattr(pipe._multimodal_handler, "_download_remote_url_streaming", fake_streaming_download)
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path)

    captured: list[dict[str, Any]] = []

    async def emitter(event):
        captured.append(event)

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    message_lock = asyncio.Lock()
    await message_lock.acquire()

    result = await adapter._run_lifecycle_after_submit(
        key=("chat-1", "msg-1"),
        job_id="job-1",
        api_model_id="openai/sora-2-pro",
        normalized_model_id="openai.sora-2-pro",
        valves=pipe.valves,
        event_emitter=emitter,
        user={"id": "user-1"},
        user_obj={"id": "user-1"},
        chat_id="chat-1",
        message_id="msg-1",
        request=None,
        user_id="user-1",
        global_semaphore=semaphore,
        message_lock=message_lock,
        started_at=time.monotonic(),
    )

    assert isinstance(result, VideoLifecycleResult)
    completion_events = [evt for evt in captured if evt.get("type") in {"chat:message:delta", "chat:completion"}]
    assert completion_events == [], (
        f"Lifecycle bg task must not emit chat:message:delta or chat:completion — saw {completion_events}"
    )


@pytest.mark.asyncio
async def test_video_lifecycle_removes_temp_directory(monkeypatch):
    """The per-job mkdtemp directory must be removed after the lifecycle so it
    doesn't leak one empty dir per generation (inode exhaustion over time)."""
    from open_webui_openrouter_pipe.integrations.video_types import VideoLifecycleResult

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def status(self, _job_id, polling_url=None):
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id, index=0):
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self):
            return {"Authorization": "Bearer test"}

    captured_dirs: list = []

    async def fake_streaming_download(url, dest_path, **_kwargs):
        captured_dirs.append(dest_path.parent)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4_BYTES)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": len(MP4_BYTES)}

    async def fake_upload_from_path(*_args, **_kwargs):
        return "file-1"

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient)
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_args, **_kwargs: _FakeSession([]))
    monkeypatch.setattr(pipe._multimodal_handler, "_download_remote_url_streaming", fake_streaming_download)
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path)

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    message_lock = asyncio.Lock()
    await message_lock.acquire()

    result = await adapter._run_lifecycle_after_submit(
        key=("chat-1", "msg-1"), job_id="job-1", api_model_id="openai/sora-2-pro",
        normalized_model_id="openai.sora-2-pro", valves=pipe.valves,
        event_emitter=lambda _e: asyncio.sleep(0), user={"id": "u"}, user_obj={"id": "u"},
        chat_id="chat-1", message_id="msg-1", request=None, user_id="u",
        global_semaphore=semaphore, message_lock=message_lock, started_at=time.monotonic(),
    )

    assert isinstance(result, VideoLifecycleResult)
    assert captured_dirs, "download was not invoked"
    for d in captured_dirs:
        assert not d.exists(), f"temp dir leaked: {d}"


def test_video_filter_spec_strips_vendor_prefix_from_display_name():
    model = {
        "id": "google/veo-3.1-lite",
        "name": "Google: Veo 3.1 Lite",
        "supported_aspect_ratios": ["16:9"],
        "supported_durations": [4],
    }
    spec = build_video_filter_spec("google/veo-3.1-lite", model)
    assert spec.display_name == "Veo 3.1 Lite"


def test_video_filter_spec_no_colon_keeps_raw_name():
    model = {"id": "openrouter/video", "name": "OpenRouter Video"}
    spec = build_video_filter_spec("openrouter/video", model)
    assert spec.display_name == "OpenRouter Video"


def test_video_filter_spec_handles_multiple_colons():
    model = {"id": "kling/video-o1", "name": "Kling: Video: O1"}
    spec = build_video_filter_spec("kling/video-o1", model)
    assert spec.display_name == "Video: O1"


def test_video_filter_renderer_escapes_untrusted_model_metadata():
    malicious_model = {
        "id": "evil/model\"\"\"\nINJECTED = True\n\"\"\"",
        "name": "Bad Model\"\"\"\nINJECTED = True\n\"\"\"",
        "supported_aspect_ratios": ["16:9"],
        "supported_durations": [4],
        "supported_resolutions": ["720p"],
        "supported_frame_images": ["first_frame"],
        "allowed_passthrough_parameters": ["negative_prompt"],
    }
    source = render_video_filter_source(model_id=malicious_model["id"], video_model=malicious_model)

    valid, error = FilterManager.validate_filter_source(source)
    assert valid, error
    module = _load_filter_from_source(source, "video_gen_filter_malicious")
    assert not hasattr(module, "INJECTED")
    assert module.VIDEO_MODEL_ID == malicious_model["id"]


def test_video_filter_deep_merges_provider_options_and_diverts_frames():
    source = render_video_filter_source(model_id="google/veo-3.1", video_model=VIDEO_BY_ID["google/veo-3.1"])
    module = _load_filter_from_source(source, "video_gen_filter_veo")
    filt = module.Filter()
    user_valves = module.Filter.UserValves(
        VIDEO_FRAME_MODE="first_last",
        VIDEO_ASPECT_RATIO="16:9",
        VIDEO_PROVIDER_OPTIONS_JSON='{"google": {"parameters": {"enhancePrompt": true}}}',
    )
    body = {
        "files": [
            {"id": "first", "name": "first.png", "content_type": "image/png", "size": 10},
            {"id": "doc", "name": "doc.pdf", "content_type": "application/pdf", "size": 20},
            {"id": "last", "name": "last.webp", "content_type": "image/webp", "size": 30},
        ]
    }
    metadata = {
        "openrouter_pipe": {
            "provider": {
                "order": ["google"],
                "options": {"existing": {"parameters": {"keep": True}}},
            }
        }
    }

    returned = filt.inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    assert returned is body
    assert body["files"] == [{"id": "doc", "name": "doc.pdf", "content_type": "application/pdf", "size": 20}]
    pipe_meta = metadata["openrouter_pipe"]
    assert pipe_meta["provider"]["order"] == ["google"]
    assert pipe_meta["provider"]["options"]["existing"]["parameters"]["keep"] is True
    assert pipe_meta["provider"]["options"]["google"]["parameters"]["enhancePrompt"] is True
    assert pipe_meta["video_generation"]["params"]["aspect_ratio"] == "16:9"
    assert [frame["frame_type"] for frame in pipe_meta["video_generation"]["frame_images"]] == [
        "first_frame",
        "last_frame",
    ]
    assert metadata["files"] == body["files"]


def test_first_image_becomes_first_frame_and_second_becomes_last_frame():
    source = render_video_filter_source(model_id="google/veo-3.1", video_model=VIDEO_BY_ID["google/veo-3.1"])
    module = _load_filter_from_source(source, "video_gen_filter_veo_frameorder_two")
    user_valves = module.Filter.UserValves(VIDEO_FRAME_MODE="first_last")
    body = {
        "files": [
            {"id": "img-A", "name": "a.png", "content_type": "image/png", "size": 10},
            {"id": "img-B", "name": "b.png", "content_type": "image/png", "size": 20},
        ]
    }
    metadata: dict[str, Any] = {}

    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    assert frames[0]["frame_type"] == "first_frame"
    assert frames[0]["id"] == "img-A"
    assert frames[1]["frame_type"] == "last_frame"
    assert frames[1]["id"] == "img-B"


def test_three_images_keeps_first_and_last_only():
    source = render_video_filter_source(model_id="google/veo-3.1", video_model=VIDEO_BY_ID["google/veo-3.1"])
    module = _load_filter_from_source(source, "video_gen_filter_veo_frameorder_three")
    user_valves = module.Filter.UserValves(VIDEO_FRAME_MODE="first_last")
    body = {
        "files": [
            {"id": "img-A", "name": "a.png", "content_type": "image/png", "size": 10},
            {"id": "img-B", "name": "b.png", "content_type": "image/png", "size": 20},
            {"id": "img-C", "name": "c.png", "content_type": "image/png", "size": 30},
        ]
    }
    metadata: dict[str, Any] = {}

    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    frame_ids = {frame["frame_type"]: frame["id"] for frame in frames}
    assert frame_ids == {"first_frame": "img-A", "last_frame": "img-C"}
    assert "img-B" not in {frame["id"] for frame in frames}


def test_filter_reads_files_from_metadata_user_message_when_body_files_empty():
    source = render_video_filter_source(model_id="google/veo-3.1", video_model=VIDEO_BY_ID["google/veo-3.1"])
    module = _load_filter_from_source(source, "video_gen_filter_veo_owui_shape")
    user_valves = module.Filter.UserValves(VIDEO_FRAME_MODE="first_last")
    body: dict[str, Any] = {"files": None, "messages": [{"role": "user", "content": "make a video"}]}
    metadata: dict[str, Any] = {
        "user_message": {
            "files": [
                {
                    "type": "file",
                    "file": {"id": "48850b5f", "filename": "first.jpg"},
                    "id": "48850b5f",
                    "name": "first.jpg",
                    "content_type": "image/jpeg",
                    "size": 600059,
                },
                {
                    "type": "file",
                    "file": {"id": "00461432", "filename": "last.jpg"},
                    "id": "00461432",
                    "name": "last.jpg",
                    "content_type": "image/jpeg",
                    "size": 722005,
                },
            ]
        }
    }

    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    assert len(frames) == 2
    assert frames[0]["id"] == "48850b5f"
    assert frames[0]["frame_type"] == "first_frame"
    assert frames[1]["id"] == "00461432"
    assert frames[1]["frame_type"] == "last_frame"


def _file_item(file_id: str, name: str, content_type: str, size: int = 1024) -> dict[str, Any]:
    return {
        "type": "file",
        "file": {"id": file_id, "filename": name},
        "id": file_id,
        "name": name,
        "content_type": content_type,
        "size": size,
    }


def _run_inlet_via_metadata(model_id: str, files: list[dict[str, Any]], frame_mode: str = "auto") -> tuple[dict, dict]:
    source = render_video_filter_source(model_id=model_id, video_model=VIDEO_BY_ID[model_id])
    module = _load_filter_from_source(source, f"video_filter_inlet_{model_id.replace('/', '_').replace('.', '_').replace('-', '_')}_{frame_mode}")
    valves_kwargs: dict[str, Any] = {}
    if hasattr(module.Filter.UserValves, "model_fields") and "VIDEO_FRAME_MODE" in module.Filter.UserValves.model_fields:
        valves_kwargs["VIDEO_FRAME_MODE"] = frame_mode
    user_valves = module.Filter.UserValves(**valves_kwargs)
    body: dict[str, Any] = {"files": None}
    metadata: dict[str, Any] = {"user_message": {"files": files}}
    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})
    return body, metadata


def test_auto_mode_adapts_to_image_count_on_first_last_capable_models():
    """Auto + 2 images on Veo 3.1 (supports first_last) → first + last automatically."""
    files = [
        _file_item("img-A", "a.jpg", "image/jpeg"),
        _file_item("img-B", "b.jpg", "image/jpeg"),
    ]
    body, metadata = _run_inlet_via_metadata("google/veo-3.1", files, frame_mode="auto")
    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    assert [f["frame_type"] for f in frames] == ["first_frame", "last_frame"]
    assert frames[0]["id"] == "img-A"
    assert frames[1]["id"] == "img-B"


def test_auto_mode_caps_at_first_only_for_single_frame_models():
    """Auto + 2 images on Hailuo (only supports first_frame) → only first; second dropped."""
    files = [
        _file_item("img-A", "a.jpg", "image/jpeg"),
        _file_item("img-B", "b.jpg", "image/jpeg"),
    ]
    body, metadata = _run_inlet_via_metadata("minimax/hailuo-2.3", files, frame_mode="auto")
    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    assert len(frames) == 1
    assert frames[0]["id"] == "img-A"
    assert frames[0]["frame_type"] == "first_frame"


def test_unselected_images_dropped_not_routed_to_rag():
    """RAG bypass: 3 images in first_only mode → 1 frame, the other 2 must NOT appear in
    body['files'] or metadata['files'] (so OWUI's chat_completion_files_handler skips RAG).
    """
    files = [
        _file_item("img-A", "a.jpg", "image/jpeg"),
        _file_item("img-B", "b.jpg", "image/jpeg"),
        _file_item("img-C", "c.jpg", "image/jpeg"),
    ]
    body, metadata = _run_inlet_via_metadata("google/veo-3.1", files, frame_mode="first_only")
    assert body["files"] == []
    assert metadata["files"] == []
    frames = metadata["openrouter_pipe"]["video_generation"]["frame_images"]
    assert len(frames) == 1
    assert frames[0]["id"] == "img-A"


def test_video_chat_attachment_routed_for_wan_2_7():
    """Wan 2.7 supports `video` and `videos` passthrough — single video chat attachment
    should land in metadata.video_attachments."""
    files = [_file_item("vid-X", "clip.mp4", "video/mp4")]
    body, metadata = _run_inlet_via_metadata("alibaba/wan-2.7", files)
    video_attachments = metadata["openrouter_pipe"]["video_generation"].get("video_attachments", [])
    assert len(video_attachments) == 1
    assert video_attachments[0]["id"] == "vid-X"
    assert video_attachments[0]["content_type"] == "video/mp4"
    assert body["files"] == []
    assert metadata["files"] == []


def test_multiple_video_chat_attachments_routed_for_wan_2_7():
    """Two video chat attachments + Wan 2.7 → both go to video_attachments."""
    files = [
        _file_item("vid-1", "a.mp4", "video/mp4"),
        _file_item("vid-2", "b.mp4", "video/mp4"),
    ]
    body, metadata = _run_inlet_via_metadata("alibaba/wan-2.7", files)
    video_attachments = metadata["openrouter_pipe"]["video_generation"].get("video_attachments", [])
    assert len(video_attachments) == 2
    assert {v["id"] for v in video_attachments} == {"vid-1", "vid-2"}


def test_audio_chat_attachment_routed_for_wan_2_6():
    """Wan 2.6 supports `audio` passthrough — audio chat attachment should land in metadata.audio_attachments."""
    files = [_file_item("aud-1", "voice.mp3", "audio/mpeg")]
    body, metadata = _run_inlet_via_metadata("alibaba/wan-2.6", files)
    audio_attachments = metadata["openrouter_pipe"]["video_generation"].get("audio_attachments", [])
    assert len(audio_attachments) == 1
    assert audio_attachments[0]["id"] == "aud-1"
    assert body["files"] == []
    assert metadata["files"] == []


def test_audio_chat_attachment_routed_for_wan_2_7():
    """Wan 2.7 also supports `audio` passthrough — audio chat attachment must land in metadata.audio_attachments."""
    files = [_file_item("aud-1", "voice.mp3", "audio/mpeg")]
    body, metadata = _run_inlet_via_metadata("alibaba/wan-2.7", files)
    audio_attachments = metadata["openrouter_pipe"]["video_generation"].get("audio_attachments", [])
    assert len(audio_attachments) == 1
    assert audio_attachments[0]["id"] == "aud-1"
    assert body["files"] == []
    assert metadata["files"] == []


def test_video_attachment_dropped_for_models_that_do_not_accept_video():
    """Veo doesn't accept video passthrough — video chat attachment is dropped, NOT sent to RAG."""
    files = [_file_item("vid-X", "clip.mp4", "video/mp4")]
    body, metadata = _run_inlet_via_metadata("google/veo-3.1", files)
    assert metadata["openrouter_pipe"]["video_generation"].get("video_attachments") is None
    assert body["files"] == []
    assert metadata["files"] == []


def test_audio_attachment_dropped_for_models_that_do_not_accept_audio():
    """Veo doesn't accept audio passthrough — audio chat attachment is dropped."""
    files = [_file_item("aud-1", "voice.mp3", "audio/mpeg")]
    body, metadata = _run_inlet_via_metadata("google/veo-3.1", files)
    assert metadata["openrouter_pipe"]["video_generation"].get("audio_attachments") is None
    assert body["files"] == []
    assert metadata["files"] == []


def test_non_media_files_kept_in_retained():
    """A PDF chat attachment is not media and SHOULD remain in retained → body.files / metadata.files
    so OWUI's normal RAG path can process it."""
    files = [_file_item("doc-1", "report.pdf", "application/pdf")]
    body, metadata = _run_inlet_via_metadata("google/veo-3.1", files)
    assert len(body["files"]) == 1
    assert body["files"][0]["id"] == "doc-1"
    assert metadata["files"] == body["files"]


def test_sora_routes_every_attachment_to_input_references():
    """Sora 2 Pro publishes no frame, video or audio slot, so nothing is anchored — but the
    attachment is a reference the documented `input_references` route carries on every
    model, and it must leave `body["files"]` so Open WebUI never RAGs it."""
    files = [
        _file_item("img-A", "a.jpg", "image/jpeg"),
        _file_item("vid-X", "clip.mp4", "video/mp4"),
        _file_item("aud-1", "voice.mp3", "audio/mpeg"),
    ]
    body, metadata = _run_inlet_via_metadata("openai/sora-2-pro", files)
    assert body["files"] == []
    assert metadata["files"] == []
    pipe_meta = metadata.get("openrouter_pipe") or {}
    video_meta = pipe_meta.get("video_generation") or {}
    assert "frame_images" not in video_meta
    assert "video_attachments" not in video_meta
    assert "audio_attachments" not in video_meta
    assert [r["id"] for r in video_meta["input_references"]] == ["img-A", "vid-X", "aud-1"]


@pytest.mark.asyncio
async def test_payload_routes_video_attachment_url_to_video_field_for_wan_2_7():
    """`_build_payload` routes a single video data URL to params.video for Wan 2.7."""
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:video/mp4;base64,AAAA"
    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=[data_url],
    )
    assert _provider_params(payload, "atlas-cloud")["video"] == data_url
    assert set(payload["provider"]["options"]) == {"atlas-cloud"}

@pytest.mark.asyncio
async def test_payload_routes_multiple_video_urls_to_videos_array_for_wan_2_7():
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    urls = ["data:video/mp4;base64,AAAA", "data:video/mp4;base64,BBBB"]
    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=urls,
    )
    assert _provider_params(payload, "atlas-cloud")["videos"] == [{"url": urls[0]}, {"url": urls[1]}]
    assert set(payload["provider"]["options"]) == {"atlas-cloud"}

@pytest.mark.asyncio
async def test_payload_routes_audio_attachment_for_wan_2_6():
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:audio/mpeg;base64,AAAA"
    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.6",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.6"],
        frame_images=[],
        provider_options={},
        audio_attachment_url=data_url,
    )
    assert _provider_params(payload, "atlas-cloud")["audio"] == data_url
    assert set(payload["provider"]["options"]) == {"atlas-cloud"}

@pytest.mark.asyncio
async def test_payload_routes_audio_attachment_for_wan_2_7():
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:audio/mpeg;base64,AAAA"
    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        audio_attachment_url=data_url,
    )
    assert _provider_params(payload, "atlas-cloud")["audio"] == data_url
    assert set(payload["provider"]["options"]) == {"atlas-cloud"}

@pytest.mark.asyncio
async def test_data_urls_bypass_ssrf_validator():
    """Data URLs are inline content with no network fetch — must be allowed through
    `_validate_passthrough_urls` without DNS resolution."""
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {"audio": "data:audio/mpeg;base64,AAAA"}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
    )
    assert _provider_params(payload, "atlas-cloud")["audio"].startswith("data:audio/")


def test_sniff_video_mime_and_extension_match_container():
    webm = b"\x1A\x45\xDF\xA3" + b"\x00" * 8

    assert _sniff_mime_from_prefix(MP4_BYTES) == "video/mp4"
    assert extension_for_video_mime("video/mp4") == ".mp4"
    assert _sniff_mime_from_prefix(webm) == "video/webm"
    assert extension_for_video_mime("video/webm") == ".webm"
    assert _sniff_mime_from_prefix(b"not a video") is None
    assert _sniff_mime_from_prefix(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == "image/png"
    assert _sniff_mime_from_prefix(b"\xff\xd8\xff\xe0" + b"\x00" * 8) == "image/jpeg"


@pytest.mark.asyncio
async def test_video_persistence_local_chats_skip_message_load(monkeypatch):
    persistence = VideoPersistence(logger=_test_logger())
    assert await persistence.load_message_content(chat_id="local:chat", message_id="msg-1") == ""


@pytest.mark.asyncio
async def test_video_adapter_completed_marker_returns_cached_content(monkeypatch):
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    adapter = pipe._ensure_video_generation_adapter()
    final = "[openrouter:v1:videojob:job-1]: #\n\n<video>/api/v1/files/file-1/content</video>"
    cast(Any, adapter)._persistence = _MemoryPersistence(final)

    def fail_client(*_args, **_kwargs):
        raise AssertionError("client must not be constructed for cached final content")

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", fail_client)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
        user={"id": "user-1"},
        request=None,
        user_obj=None,
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert result == final


@pytest.mark.asyncio
async def test_video_adapter_pending_marker_resumes_without_submit(monkeypatch, tmp_path):
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 1
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 1
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence(
        "[openrouter:v1:videojob:job-resume]: #\n\nVideo generation is running..."
    )
    submit_calls = 0

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, _payload):
            nonlocal submit_calls
            submit_calls += 1
            raise AssertionError("resume path must not submit")

        async def status(self, job_id, polling_url=None):
            assert job_id == "job-resume"
            return {"status": "completed", "usage": {"cost": "0.25"}}

        def content_url(self, job_id: str, index: int = 0) -> str:
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer test"}

    async def fake_streaming_download(url: str, dest_path, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4_BYTES)
        return {
            "path": dest_path,
            "mime_type": "video/mp4",
            "url": url,
            "size_bytes": len(MP4_BYTES),
        }

    async def fake_upload_from_path(*_args, **_kwargs):
        return "file-1"

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient)
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_args, **_kwargs: _FakeSession([]))
    monkeypatch.setattr(pipe._multimodal_handler, "_download_remote_url_streaming", fake_streaming_download)
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert submit_calls == 0
    assert "<video>\n/api/v1/files/file-1/content\n</video>" in result
    assert pipe._video_user_active_counts == {}
    assert pipe._video_user_active_jobs == {}
    assert pipe._video_active_tasks == {}


@pytest.mark.asyncio
async def test_video_zdr_enforce_rejects_before_video_dispatch(monkeypatch):
    from aioresponses import aioresponses

    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID["openai/sora-2-pro"]])
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
    pipe.valves.ZDR_ENFORCE = True
    events: list[dict[str, Any]] = []

    def fail_video_adapter():
        raise AssertionError("video adapter must not run when ZDR enforcement rejects the model")

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    monkeypatch.setattr(pipe, "_ensure_video_generation_adapter", fail_video_adapter)
    try:
        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": []},
                repeat=True,
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/videos/models",
                payload={"data": [VIDEO_BY_ID["openai/sora-2-pro"]]},
                repeat=True,
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": []},
                repeat=True,
            )

            result = await pipe.pipe(
                body={
                    "model": "openai.sora-2-pro",
                    "messages": [{"role": "user", "content": "make a video"}],
                    "stream": False,
                },
                __user__={"id": "user-1", "valves": {}},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={"model": {"id": "openai.sora-2-pro"}, "chat_id": "chat-1", "message_id": "msg-1"},
                __tools__={},
            )

        assert await _consume_pipe_result(result) == ""
        assert any("ZDR_ENFORCE" in str(event) for event in events)
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "cancelled", "expired"])
async def test_video_adapter_terminal_failures_persist_visible_failure(monkeypatch, terminal_status):
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()
    persistence = _MemoryPersistence("[openrouter:v1:videojob:job-resume]: #\n\nVideo generation is running...")
    cast(Any, adapter)._persistence = persistence

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def status(self, _job_id, polling_url=None):
            return {"status": terminal_status, "error": {"message": "provider stopped"}}

        async def download_content_to_temp(self, *_args, **_kwargs):
            raise AssertionError("failed jobs must not download")

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient)
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_args, **_kwargs: _FakeSession([]))

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=emitter,
        metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert "### Video generation failed" in result
    assert "provider stopped" in result
    delta_events = [e for e in events if e.get("type") == "chat:message:delta"]
    assert any(e.get("data", {}).get("content") == result for e in delta_events), (
        f"expected a chat:message:delta with the failure content; got {events!r}"
    )
    assert pipe._video_message_locks == {}


@pytest.mark.asyncio
async def test_video_adapter_releases_semaphore_when_submit_fails(monkeypatch):
    Pipe._video_global_semaphore = None
    Pipe._video_global_limit = 0
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.MAX_CONCURRENT_VIDEO_GENS = 1
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence("")

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, _payload):
            raise RuntimeError("submit exploded")

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=object(),
        event_emitter=None,
        metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert "submit exploded" in result
    assert pipe._video_user_active_counts == {}
    semaphore = adapter._ensure_global_semaphore(pipe.valves)
    await asyncio.wait_for(semaphore.acquire(), timeout=0.2)
    semaphore.release()


@pytest.mark.asyncio
async def test_a_failing_cleanup_step_does_not_strand_the_releases_after_it():
    """One contract for both release paths, driven rather than counted.

    Two byte-near-identical closures used to encode this 250 lines apart, differing
    only in one word of the log message, so changing which exceptions propagate in one
    left the other on the old contract. Both now call `_cleanup_step`.

    Two arms, opposite directions: an ordinary failure is logged and the NEXT step still
    runs; a cancellation propagates instead of being swallowed. A version that swallows
    everything passes the first arm and fails the second.
    """
    import asyncio as _asyncio
    import logging

    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    adapter.logger = logging.getLogger("video-cleanup-test")

    ran: list[str] = []

    async def _boom():
        raise RuntimeError("release failed")

    async def _later():
        ran.append("later")

    await adapter._cleanup_step("lifecycle", ("c1", "m1"), "user slot", _boom())
    await adapter._cleanup_step("lifecycle", ("c1", "m1"), "message lock", _later())
    assert ran == ["later"], (
        "a failing cleanup step stopped the releases after it, which strands every "
        "resource below it for the life of the process"
    )

    async def _cancelled():
        raise _asyncio.CancelledError()

    with pytest.raises(_asyncio.CancelledError):
        await adapter._cleanup_step("lifecycle", ("c1", "m1"), "cancelled", _cancelled())


@pytest.mark.asyncio
async def test_video_adapter_does_not_release_an_unacquired_global_slot(monkeypatch):
    """Cancelling while queued on the global semaphore must not mint a extra permit."""
    Pipe._video_global_semaphore = None
    Pipe._video_global_limit = 0
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.MAX_CONCURRENT_VIDEO_GENS = 1
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence("")

    semaphore = adapter._ensure_global_semaphore(pipe.valves)
    await semaphore.acquire()

    try:
        await _assert_no_phantom_permit(adapter, pipe, semaphore)
    finally:
        Pipe._video_global_semaphore = None
        Pipe._video_global_limit = 0


async def _assert_no_phantom_permit(adapter, pipe, semaphore):
    task = asyncio.create_task(
        adapter.generate(
            body={"messages": [{"role": "user", "content": "make a video"}]},
            responses_body=SimpleNamespace(provider={}),
            valves=pipe.valves,
            session=object(),
            event_emitter=None,
            metadata={"chat_id": "chat-9", "message_id": "msg-9", "user_id": "user-9"},
            user={"id": "user-9"},
            request=None,
            user_obj={"id": "user-9"},
            normalized_model_id="openai.sora-2-pro",
            api_model_id="openai/sora-2-pro",
        )
    )
    for _ in range(200):
        await asyncio.sleep(0)
        if semaphore._waiters:
            break
    assert semaphore._waiters, "request never queued on the global semaphore"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    semaphore.release()

    await asyncio.wait_for(semaphore.acquire(), timeout=0.5)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
    semaphore.release()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persisted", "path"),
    [
        ("", "fresh-submit"),
        ("[openrouter:v1:videojob:job-mid-handoff]: #\n\nVideo generation is running...", "resume"),
    ],
)
async def test_video_adapter_does_not_double_release_when_cancelled_mid_handoff(
    monkeypatch, persisted, path
):
    """A cancel after the lifecycle task exists must not release its permit twice.

    Parametrised because the handoff is written twice -- once for a fresh submit and
    once for resuming a job found in the persisted message -- and only the first was
    covered. Moving `lifecycle_transferred = True` back below the `async with` on the
    resume branch left the whole suite green.

    When the cancel lands while suspended on the contended dict lock, the flag is still
    False, so generate()'s finally releases the global semaphore, the user slot and the
    message lock -- all still owned by the lifecycle task, which releases them again.
    Over-releasing an asyncio.Semaphore mints a permanent extra permit, so the
    configured concurrency cap drifts upward for the worker's lifetime.
    """
    Pipe._video_global_semaphore = None
    Pipe._video_global_limit = 0
    pipe = Pipe()
    created: list[Any] = []
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.MAX_CONCURRENT_VIDEO_GENS = 1
        pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
        adapter = pipe._ensure_video_generation_adapter()
        cast(Any, adapter)._persistence = _MemoryPersistence(persisted)
        semaphore = adapter._ensure_global_semaphore(pipe.valves)
        assert semaphore._value == 1

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def submit(self, _payload):
                return {"id": "job-mid-handoff", "status": "queued"}

            async def status(self, _job_id, polling_url=None):
                await asyncio.sleep(3600)

            def content_url(self, job_id, index=0):
                return f"https://example.test/videos/{job_id}/content"

            def bearer_header(self):
                return {"Authorization": "Bearer test"}

        monkeypatch.setattr(
            "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient",
            FakeClient,
        )

        reached_handoff = asyncio.Event()
        real_create = adapter._create_lifecycle_task

        def spy_create(**kwargs):
            bg = real_create(**kwargs)
            created.append(bg)
            return bg

        monkeypatch.setattr(adapter, "_create_lifecycle_task", spy_create)

        real_lock = pipe._video_active_tasks_dict_lock

        class _StallAfterHandoff:
            async def __aenter__(self):
                if created:
                    reached_handoff.set()
                    await asyncio.Event().wait()
                return await real_lock.__aenter__()

            async def __aexit__(self, *exc):
                return await real_lock.__aexit__(*exc)

        monkeypatch.setattr(
            pipe, "_video_active_tasks_dict_lock", _StallAfterHandoff()
        )

        task = asyncio.create_task(
            adapter.generate(
                body={"messages": [{"role": "user", "content": "make a video"}]},
                responses_body=SimpleNamespace(provider={}),
                valves=pipe.valves,
                session=object(),
                event_emitter=None,
                metadata={"chat_id": "chat-h", "message_id": "msg-h", "user_id": "user-h"},
                user={"id": "user-h"},
                request=None,
                user_obj={"id": "user-h"},
                normalized_model_id="openai.sora-2-pro",
                api_model_id="openai/sora-2-pro",
            )
        )
        await asyncio.wait_for(reached_handoff.wait(), timeout=5)
        assert created, "lifecycle task was never created"
        assert semaphore._value == 0, "the permit should be held at this point"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(50):
            await asyncio.sleep(0)

        assert semaphore._value == 0, (
            f"global video semaphore inflated to {semaphore._value} on the {path} path: "
            "generate() released a permit already owned by the still-running lifecycle "
            "task, so the configured cap is now permanently wrong for this worker"
        )
    finally:
        for bg in created:
            bg.cancel()
        for _ in range(50):
            await asyncio.sleep(0)
        Pipe._video_global_semaphore = None
        Pipe._video_global_limit = 0
        await pipe.close()


@pytest.mark.asyncio
async def test_video_adapter_waiter_uses_active_task_before_user_cap():
    pipe = Pipe()
    pipe.valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER = 1
    pipe._video_user_active_counts["user-1"] = 1
    adapter = pipe._ensure_video_generation_adapter()
    events: list[dict[str, Any]] = []

    async def active_result():
        await asyncio.sleep(0)
        return VideoLifecycleResult(
            content="done",
            status_description="complete",
            job_id="job-1",
        )

    key = ("chat-1", "msg-1")
    task = asyncio.create_task(active_result())
    pipe._video_active_tasks[key] = task

    async def emitter(event: dict[str, Any]):
        events.append(event)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=emitter,
        metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
        user={"id": "user-1"},
        request=None,
        user_obj=None,
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert result == "done"
    assert pipe._video_user_active_counts["user-1"] == 1
    assert events[-1]["type"] == "chat:completion"


@pytest.mark.asyncio
async def test_job_tracking_stores_real_job_ids_and_cleans_one_at_a_time():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    pipe._video_user_active_counts["user-1"] = 2

    await adapter._add_user_active_job("user-1", "job-a")
    await adapter._add_user_active_job("user-1", "job-b")
    await adapter._release_user_slot("user-1", "job-a")

    assert pipe._video_user_active_jobs["user-1"] == {"job-b"}
    assert pipe._video_user_active_counts["user-1"] == 1

    await adapter._release_user_slot("user-1", "job-b")
    assert pipe._video_user_active_jobs == {}
    assert pipe._video_user_active_counts == {}


@pytest.mark.asyncio
async def test_message_lock_refcount_cleanup_is_awaited():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    key = ("chat-1", "msg-1")

    lock = await adapter._acquire_message_lock(key)
    await adapter._release_message_lock(key, lock)

    assert pipe._video_message_locks == {}
    assert pipe._video_message_lock_refs == {}


@pytest.mark.asyncio
async def test_message_lock_three_waiter_race_cancels_cleanly():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    key = ("chat-race", "msg-race")

    lock_a = await adapter._acquire_message_lock(key)
    assert pipe._video_message_lock_refs[key] == 1

    task_b = asyncio.create_task(adapter._acquire_message_lock(key))
    task_c = asyncio.create_task(adapter._acquire_message_lock(key))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pipe._video_message_lock_refs[key] == 3
    assert not task_b.done()
    assert not task_c.done()

    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_b
    assert pipe._video_message_lock_refs[key] == 2

    await adapter._release_message_lock(key, lock_a)
    lock_c = await asyncio.wait_for(task_c, timeout=1.0)
    assert pipe._video_message_lock_refs[key] == 1

    await adapter._release_message_lock(key, lock_c)
    assert pipe._video_message_locks == {}
    assert pipe._video_message_lock_refs == {}


@pytest.mark.asyncio
async def test_message_lock_cancellation_after_release_yields_lock_to_next_waiter():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    key = ("chat-postrelease", "msg-postrelease")

    lock_a = await adapter._acquire_message_lock(key)
    task_b = asyncio.create_task(adapter._acquire_message_lock(key))
    task_c = asyncio.create_task(adapter._acquire_message_lock(key))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pipe._video_message_lock_refs[key] == 3

    await adapter._release_message_lock(key, lock_a)
    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_b

    lock_c = await asyncio.wait_for(task_c, timeout=1.0)
    assert pipe._video_message_lock_refs[key] == 1

    await adapter._release_message_lock(key, lock_c)
    assert pipe._video_message_locks == {}
    assert pipe._video_message_lock_refs == {}


def test_provider_options_are_emitted_flat_and_legacy_wrappers_are_unwrapped():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    flattened = adapter._normalise_provider_options(
        {
            "google-vertex": {"negativePrompt": "blur"},
            "fal": {"parameters": {"motion": "slow"}},
        }
    )

    assert flattened["google-vertex"] == {"negativePrompt": "blur"}
    assert flattened["fal"] == {"motion": "slow"}


@pytest.mark.asyncio
async def test_video_payload_drops_unsupported_passthrough_and_sends_documented_aspect_ratio():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    unsupported = await adapter._build_payload(
        api_model_id="unknown/video",
        prompt="make a video",
        video_meta={"params": {"surprise": "value"}},
        video_model={"allowed_passthrough_parameters": []},
        frame_images=[],
        provider_options={},
    )
    veo = await adapter._build_payload(
        api_model_id="google/veo-3.1",
        prompt="make a video",
        video_meta={"params": {"aspect_ratio": "16:9"}},
        video_model=VIDEO_BY_ID["google/veo-3.1"],
        frame_images=[],
        provider_options={},
    )

    assert "surprise" not in unsupported
    assert veo["aspect_ratio"] == "16:9"
    assert "aspectRatio" not in veo
    assert "aspectRatio" not in _provider_params(veo, "google-vertex")


@pytest.mark.asyncio
async def test_status_events_are_throttled_to_meaningful_progress(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    events: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self):
            self.statuses = iter(
                [
                    {"status": "pending"},
                    {"status": "pending"},
                    {"status": "in_progress"},
                    {"status": "completed"},
                ]
            )

        async def status(self, _job_id, polling_url=None):
            return next(self.statuses)

    async def no_sleep(_seconds):
        return None

    async def emitter(event):
        events.append(event)

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.asyncio.sleep", no_sleep)
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 1
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 1

    payload = await adapter._poll_until_terminal(cast(Any, FakeClient()), "job-1", pipe.valves, emitter)

    progress = [event["data"].get("progress") for event in events if event["type"] == "status"]
    assert payload["status"] == "completed"
    assert progress == [5, 50, 100]


def test_video_help_is_model_specific_for_all_catalog_models():
    assert set(VIDEO_BY_ID) <= set(VIDEO_HELP_BY_MODEL)
    rendered = {model_id: render_video_help(model_id, VIDEO_BY_ID[model_id]) for model_id in VIDEO_BY_ID}

    for model_id, model in VIDEO_BY_ID.items():
        assert model["name"] in rendered[model_id], f"{model_id} help missing display name"

    assert len(set(rendered.values())) == len(rendered)


def test_video_passthrough_naming_consistency_across_renderer_help_and_catalog():
    """Drift guard for the three-name spread (`<Human Label>` → `passthrough_param` → `VIDEO_VALVE_NAME`).

    Two invariants:
      (1) Every passthrough param in any catalog model's
          `allowed_passthrough_parameters` MUST be renderable, i.e. its name can
          become a form field. A name that cannot is the only kind the renderer
          genuinely drops; anything else it offers, typed where a purpose-built
          control exists and free text otherwise.
      (2) Every `_KNOB_GATE` value that names a passthrough param (i.e. not a
          special top-level marker) MUST be a param the renderer offers —
          otherwise help advertises a knob that the renderer cannot wire.

    The reverse direction (catalog → `_KNOB_GATE`) is intentionally NOT
    asserted: aliases like `negative_prompt`/`negativePrompt` and top-level
    fields like `aspectRatio`/`size` map via marker gates or `None`-gate
    entries that don't share literal string identity with the catalog name.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        _HANDLED_PASSTHROUGH_PARAMS,
        _unhandled_params,
        build_video_filter_spec,
    )
    from open_webui_openrouter_pipe.integrations.image_types import RENDERABLE_FIELD_NAME_RE
    from open_webui_openrouter_pipe.integrations.video_help import _KNOB_GATE

    catalog_passthrough_params: set[str] = set()
    for model in VIDEO_BY_ID.values():
        for param in model.get("allowed_passthrough_parameters") or []:
            if isinstance(param, str) and param:
                catalog_passthrough_params.add(param)

    unrenderable = {
        param for param in catalog_passthrough_params
        if not RENDERABLE_FIELD_NAME_RE.fullmatch(param)
    }
    assert not unrenderable, (
        f"Catalog passthrough params whose names cannot become form fields: "
        f"{sorted(unrenderable)} — a user cannot reach them at all. Either the name is "
        "wrong in the fixture, or the renderer needs a way to express it."
    )

    knob_gate_passthrough_values = {v for v in _KNOB_GATE.values() if isinstance(v, str)}
    top_level_markers = {
        "seed_top_level",
        "generate_audio_top_level",
        "negative_prompt_or_camelcase",
    }
    knob_gate_passthrough_only = knob_gate_passthrough_values - top_level_markers
    offered = set(_HANDLED_PASSTHROUGH_PARAMS)
    for model_id, model in VIDEO_BY_ID.items():
        offered |= set(_unhandled_params(build_video_filter_spec(model_id, model)))
    knob_gate_only_unrendered = knob_gate_passthrough_only - offered
    assert not knob_gate_only_unrendered, (
        f"`_KNOB_GATE` advertises knob(s) the renderer cannot wire: {sorted(knob_gate_only_unrendered)} — "
        "either remove the gate entry or add the renderer branch."
    )


def test_video_help_renders_pricing_live_from_pricing_skus():
    veo_help = render_video_help("google/veo-3.1-fast", VIDEO_BY_ID["google/veo-3.1-fast"])
    assert "**Cost** (as OpenRouter publishes it for this model)" in veo_help
    assert "$0.12" in veo_help or "$0.10" in veo_help

    kling_help = render_video_help("kwaivgi/kling-video-o1", VIDEO_BY_ID["kwaivgi/kling-video-o1"])
    assert "$0.0896" in kling_help


def test_video_help_pricing_section_omitted_when_no_skus():
    minimal_model = {
        "id": "test/synthetic",
        "name": "Test: Synthetic",
        "supported_durations": [4],
        "supported_aspect_ratios": ["16:9"],
        "supported_resolutions": ["720p"],
        "supported_frame_images": [],
        "generate_audio": False,
        "seed": False,
        "allowed_passthrough_parameters": [],
        "pricing_skus": {},
    }
    rendered = render_video_help("test/synthetic", minimal_model)
    assert "**Cost**" not in rendered


def test_video_help_includes_typed_valve_descriptions_per_model():
    veo_help = render_video_help("google/veo-3.1", VIDEO_BY_ID["google/veo-3.1"])
    for label in ("`Person generation`", "`Conditioning scale`", "`Enhance prompt`", "`Seed`", "`Audio`"):
        assert label in veo_help, f"Veo 3.1 help missing {label}"

    hailuo_help = render_video_help("minimax/hailuo-2.3", VIDEO_BY_ID["minimax/hailuo-2.3"])
    for label in ("`Prompt optimizer`", "`Fast pretreatment`"):
        assert label in hailuo_help, f"Hailuo help missing {label}"
    for label in ("`Seed`", "`Audio`", "`Negative prompt`"):
        assert label not in hailuo_help, f"Hailuo help should not list {label}"

    sora_help = render_video_help("openai/sora-2-pro", VIDEO_BY_ID["openai/sora-2-pro"])
    for label in ("`Quality`", "`Style`"):
        assert label in sora_help, f"Sora help missing {label}"
    assert "`Frames`" not in sora_help
    assert "`Seed`" not in sora_help


def test_video_help_sku_unit_formatter_known_keys():
    from open_webui_openrouter_pipe.integrations.video_help import _sku_unit
    assert _sku_unit("duration_seconds").label == "per second"
    assert _sku_unit("duration_seconds_with_audio").label == "per second (with audio)"
    assert _sku_unit("duration_seconds_with_audio_4k").label == "per second (with audio, 4K)"
    assert _sku_unit("video_tokens").label == "per video token"
    assert _sku_unit("video_tokens_without_audio").label == "per video token (without audio)"
    assert _sku_unit("text_to_video_duration_seconds_720p").label == "per second (text-to-video, 720p)"
    assert _sku_unit("image_to_video_duration_seconds_1080p").label == "per second (image-to-video, 1080p)"
    assert _sku_unit("cents_per_second_output").label == "per output second"
    assert _sku_unit("cents_per_second_video_continuation_720p").label == (
        "per second of continued video (720p)"
    )
    assert _sku_unit("reference_images").label == "per reference image"
    assert _sku_unit("video_tokens_4k_with_video_input").label == (
        "per video token (4K with video input)"
    ), "one model lists both, so the two must spell the tier the same way"


def _video_model(**published: Any) -> dict[str, Any]:
    """A catalogue row carrying only what the panel reads."""
    model = {
        "id": "test/priced",
        "name": "Test: Priced",
        "supported_durations": [4],
        "supported_aspect_ratios": ["16:9"],
        "supported_resolutions": ["720p"],
        "supported_frame_images": [],
        "generate_audio": False,
        "seed": False,
        "allowed_passthrough_parameters": [],
    }
    model.update(published)
    return model


@pytest.mark.parametrize(
    ("published_cents", "dollars"),
    [("56", "$0.56"), ("125", "$1.25")],
)
def test_a_minimum_charge_is_read_in_cents_and_kept_out_of_the_rate_list(published_cents, dollars):
    """`runway/aleph-2` publishes `minimum_cents_per_generation`, and both halves bit.

    The cents-to-dollars conversion was gated on the key *starting* `cents_per`, so a
    floor whose key carries the token in the middle rendered at a hundred times its real
    figure; and a floor is not a rate, so bulleting it beside per-second rates invites
    the reader to add it to them.
    """
    rendered = render_video_help(
        "test/priced",
        _video_model(
            pricing_skus={
                "cents_per_second_output": "28",
                "minimum_cents_per_generation": published_cents,
            }
        ),
    )
    assert f"Minimum charge per generation: {dollars}" in rendered, rendered
    assert f"- per generation: {dollars}" not in rendered, "a floor is not a rate bullet"
    assert f"${published_cents}" not in rendered, "the value is published in cents"
    assert "- per output second: $0.28" in rendered, "the rate beside it still renders"
    assert "per minimum cents per generation" not in rendered


def test_a_charge_this_panel_cannot_name_is_marked_rather_than_invented():
    """The vocabulary is OpenRouter's and it grows.

    Turning any unrecognised key into "per <the rest of the key>" is what produced "per
    minimum cents per generation", and it would produce the next one too.
    """
    rendered = render_video_help(
        "test/priced",
        _video_model(pricing_skus={"duration_seconds": "0.10", "storage_gigabyte_month": "0.02"}),
    )
    assert "- per second: $0.10" in rendered
    assert '"storage_gigabyte_month" at $0.02' in rendered, rendered
    assert "per storage gigabyte month" not in rendered
    assert "- per storage" not in rendered


@pytest.mark.parametrize(
    ("skus", "cheapest_looking"),
    [
        ({"video_tokens": "0.000007", "video_tokens_4k": "0.000004"}, "$0.000004"),
        ({"video_tokens": "0.0000024", "video_tokens_4k": "0.0000012"}, "$0.0000012"),
    ],
)
def test_a_per_token_model_says_a_clip_price_cannot_be_derived(skus, cheapest_looking):
    """4K carries the smallest per-token number and the largest bill.

    The count of tokens a clip uses is not published anywhere in the catalogue, so the
    panel cannot convert seconds to money, and the rates do not rank the settings. Saying
    that is the only honest thing available; printing the numbers alone reads backwards.
    """
    rendered = render_video_help("test/priced", _video_model(pricing_skus=skus))
    assert cheapest_looking in rendered, "the published rates are still shown"
    assert "does not publish how many tokens a clip uses" in rendered, rendered
    assert "do not compare with each other" in rendered

    seconds = render_video_help("test/priced", _video_model(pricing_skus={"duration_seconds": "0.10"}))
    assert "does not publish how many tokens a clip uses" not in seconds, (
        "a per-second model has no such caveat and must not carry it"
    )


def test_the_base_unit_table_resolves_the_longest_token_first():
    """Ordering is the whole mechanism: the table is scanned in source order.

    An alphabetiser or a merge that reordered these pairs would silently relabel keys,
    which no rendering test would catch on today's vocabulary.
    """
    from open_webui_openrouter_pipe.integrations.video_help import _SKU_BASE_LABELS

    assert isinstance(_SKU_BASE_LABELS, tuple), "a dict would let a formatter reorder it"
    tokens = [token for token, _ in _SKU_BASE_LABELS]
    for position, token in enumerate(tokens):
        for later in tokens[position + 1:]:
            assert not later.startswith(token) or later == token, (
                f"{later!r} can never match: {token!r} precedes it and is a prefix of it"
            )


def test_video_filter_spec_seed_and_audio_gates_use_top_level_fields():
    veo = build_video_filter_spec("google/veo-3.1", VIDEO_BY_ID["google/veo-3.1"])
    assert veo.seed_capable is True
    assert veo.audio_capable is True
    assert veo.supports_seed is True
    assert veo.supports_generate_audio_toggle is True

    kling = build_video_filter_spec("kwaivgi/kling-video-o1", VIDEO_BY_ID["kwaivgi/kling-video-o1"])
    assert kling.seed_capable is False
    assert kling.audio_capable is True
    assert kling.supports_seed is False
    assert kling.supports_generate_audio_toggle is True

    hailuo = build_video_filter_spec("minimax/hailuo-2.3", VIDEO_BY_ID["minimax/hailuo-2.3"])
    assert hailuo.seed_capable is True
    assert hailuo.audio_capable is False

    sora = build_video_filter_spec("openai/sora-2-pro", VIDEO_BY_ID["openai/sora-2-pro"])
    assert sora.seed_capable is False
    assert sora.audio_capable is True


def test_happyhorse_video_filter_spec_covers_wide_ratios_and_undeclared_audio():
    for model_id in ("alibaba/happyhorse-1.0", "alibaba/happyhorse-1.1"):
        spec = build_video_filter_spec(model_id, VIDEO_BY_ID[model_id])
        assert set(spec.aspect_ratios) == {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"}
        assert "21:9" in spec.aspect_ratios
        assert "9:21" in spec.aspect_ratios
        assert set(spec.durations) == set(range(3, 16))
        assert set(spec.resolutions) == {"720p", "1080p"}
        assert spec.frame_types == ("first_frame",)
        assert spec.seed_capable is True
        assert spec.audio_capable is True
        assert spec.allowed_params == ()


@pytest.mark.parametrize("model_id", list(_VIDEO_MODEL_IDS_FOR_AUDIT := [
    "google/veo-3.1-fast",
    "google/veo-3.1-lite",
    "google/veo-3.1",
    "kwaivgi/kling-video-o1",
    "kwaivgi/kling-v3.0-pro",
    "kwaivgi/kling-v3.0-std",
    "minimax/hailuo-2.3",
    "alibaba/wan-2.7",
    "bytedance/seedance-2.0-fast",
    "bytedance/seedance-2.0",
    "alibaba/wan-2.6",
    "bytedance/seedance-1-5-pro",
    "openai/sora-2-pro",
    "alibaba/happyhorse-1.0",
    "alibaba/happyhorse-1.1",
]))
def test_video_filter_renderer_per_model_audit(model_id: str):
    model = VIDEO_BY_ID[model_id]
    source = render_video_filter_source(model_id=model_id, video_model=model)

    seed = model.get("seed", _NOT_PUBLISHED)
    assert ("VIDEO_SEED" in source) is (seed is None or seed is True), (
        f"{model_id} publishes seed={seed!r}; a control is offered when the key is present "
        "and its value is True or null, and withheld otherwise"
    )

    audio = model.get("generate_audio", _NOT_PUBLISHED)
    assert ("VIDEO_GENERATE_AUDIO" in source) is (audio is None or audio is True), (
        f"{model_id} publishes generate_audio={audio!r}; a control is offered when the key "
        "is present and its value is True or null, and withheld otherwise"
    )

    allowed = set(model.get("allowed_passthrough_parameters") or [])
    valve_for = {
        "personGeneration": "VIDEO_PERSON_GENERATION",
        "conditioningScale": "VIDEO_CONDITIONING_SCALE",
        "cfg_scale": "VIDEO_CFG_SCALE",
        "enhancePrompt": "VIDEO_ENHANCE_PROMPT",
        "prompt_optimizer": "VIDEO_PROMPT_OPTIMIZER",
        "fast_pretreatment": "VIDEO_FAST_PRETREATMENT",
        "prompt_extend": "VIDEO_PROMPT_EXTEND",
        "ratio": "VIDEO_RATIO",
        "enable_prompt_expansion": "VIDEO_ENABLE_PROMPT_EXPANSION",
        "shot_type": "VIDEO_SHOT_TYPE",
        "watermark": "VIDEO_WATERMARK",
        "req_key": "VIDEO_REQ_KEY",
        "quality": "VIDEO_QUALITY",
        "style": "VIDEO_STYLE",
    }
    for param, valve in valve_for.items():
        if param in allowed:
            assert valve in source, f"{model_id} missing typed valve {valve} for allowed param {param}"
        else:
            assert valve not in source, f"{model_id} should not expose typed valve {valve} (param {param} not allowed)"


def test_video_filter_typed_valves_route_into_metadata_params():
    veo = VIDEO_BY_ID["google/veo-3.1"]
    source = render_video_filter_source(model_id="google/veo-3.1", video_model=veo)
    module = _load_filter_from_source(source, "video_gen_filter_veo_routing")

    user_valves = module.Filter.UserValves(
        VIDEO_PERSON_GENERATION="allow_adult",
        VIDEO_CONDITIONING_SCALE=0.7,
        VIDEO_ENHANCE_PROMPT="on",
        VIDEO_SEED=42,
    )
    body: dict[str, Any] = {"files": []}
    metadata: dict[str, Any] = {}
    user_dict = {"valves": user_valves}

    module.Filter().inlet(body, __metadata__=metadata, __user__=user_dict)

    params = metadata["openrouter_pipe"]["video_generation"]["params"]
    assert params["personGeneration"] == "allow_adult"
    assert params["conditioningScale"] == 0.7
    assert params["enhancePrompt"] is True
    assert params["seed"] == 42


@pytest.mark.parametrize("model_id", ["kwaivgi/kling-v3.0-pro", "kwaivgi/kling-v3.0-std"])
def test_video_filter_routes_cfg_scale_into_params_for_kling_v3(model_id: str):
    kling = VIDEO_BY_ID[model_id]
    source = render_video_filter_source(model_id=model_id, video_model=kling)
    module = _load_filter_from_source(source, f"video_gen_filter_cfg_scale_{model_id.replace('/', '_').replace('.', '_').replace('-', '_')}")

    user_valves = module.Filter.UserValves(VIDEO_CFG_SCALE=0.6)
    body: dict[str, Any] = {"files": []}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    params = metadata["openrouter_pipe"]["video_generation"]["params"]
    assert params["cfg_scale"] == 0.6


@pytest.mark.parametrize("model_id", ["kwaivgi/kling-v3.0-pro", "kwaivgi/kling-v3.0-std"])
def test_video_filter_drops_cfg_scale_when_zero_for_kling_v3(model_id: str):
    """`VIDEO_CFG_SCALE=0.0` must NOT emit a cfg_scale param — the > 0.0 sentinel
    preserves the "0 means provider default" semantic shared with conditioningScale."""
    kling = VIDEO_BY_ID[model_id]
    source = render_video_filter_source(model_id=model_id, video_model=kling)
    module = _load_filter_from_source(source, f"video_gen_filter_cfg_scale_zero_{model_id.replace('/', '_').replace('.', '_').replace('-', '_')}")

    user_valves = module.Filter.UserValves(VIDEO_CFG_SCALE=0.0)
    body: dict[str, Any] = {"files": []}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(body, __metadata__=metadata, __user__={"valves": user_valves})

    params = metadata["openrouter_pipe"]["video_generation"]["params"]
    assert "cfg_scale" not in params


def test_video_filter_does_not_expose_cfg_scale_for_kling_video_o1():
    """`kwaivgi/kling-video-o1` does NOT list cfg_scale in `allowed_passthrough_parameters`,
    so neither the typed valve nor the param-emit branch should appear in the rendered source."""
    o1 = VIDEO_BY_ID["kwaivgi/kling-video-o1"]
    source = render_video_filter_source(model_id="kwaivgi/kling-video-o1", video_model=o1)
    assert "VIDEO_CFG_SCALE" not in source
    assert 'params["cfg_scale"]' not in source


def test_video_help_includes_cfg_scale_for_kling_v3_only():
    """`CFG scale` must appear in the help output for `kwaivgi/kling-v3.0-pro` and
    `kwaivgi/kling-v3.0-std` (whose `allowed_passthrough_parameters` includes
    `cfg_scale`), and must NOT appear for `kwaivgi/kling-video-o1`."""
    for model_id in ("kwaivgi/kling-v3.0-pro", "kwaivgi/kling-v3.0-std"):
        rendered = render_video_help(model_id, VIDEO_BY_ID[model_id])
        assert "`CFG scale`" in rendered, f"{model_id} help missing `CFG scale` knob description"
        assert VIDEO_BY_ID[model_id]["name"] in rendered, f"{model_id} help missing display name"
        assert "Classifier-free guidance" in rendered, f"{model_id} help missing cfg_scale description body"

    o1_rendered = render_video_help("kwaivgi/kling-video-o1", VIDEO_BY_ID["kwaivgi/kling-video-o1"])
    assert "`CFG scale`" not in o1_rendered, "kwaivgi/kling-video-o1 should not show CFG scale (cfg_scale not in passthrough)"


@pytest.mark.asyncio
async def test_payload_includes_seed_and_generate_audio_for_capable_models(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(
        pipe._multimodal_handler, "_request_ips_blocking", lambda url: ["203.0.113.9"]
    )

    payload = await adapter._build_payload(
        api_model_id="google/veo-3.1",
        prompt="a sunset over mountains",
        video_meta={"params": {"seed": 7, "generate_audio": False}},
        video_model=VIDEO_BY_ID["google/veo-3.1"],
        frame_images=[],
        provider_options={},
    )

    assert payload["seed"] == 7
    assert payload["generate_audio"] is False


@pytest.mark.asyncio
async def test_payload_drops_seed_for_seed_incapable_models(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(
        pipe._multimodal_handler, "_request_ips_blocking", lambda url: ["203.0.113.9"]
    )

    payload = await adapter._build_payload(
        api_model_id="kwaivgi/kling-video-o1",
        prompt="a quiet room",
        video_meta={"params": {"seed": 7}},
        video_model=VIDEO_BY_ID["kwaivgi/kling-video-o1"],
        frame_images=[],
        provider_options={},
    )

    assert "seed" not in payload


@pytest.mark.asyncio
async def test_generate_audio_does_not_clobber_audio_url_on_wan(monkeypatch):
    pipe = _pipe_with_provider_map(_ROUTED_PROVIDER_MAP)
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(
        pipe._multimodal_handler, "_request_ips_blocking", lambda url: ["203.0.113.9"]
    )

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="a campfire",
        video_meta={"params": {"audio": "https://example.com/voice.mp3", "generate_audio": True}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
    )

    assert _provider_params(payload, "atlas-cloud")["audio"] == "https://example.com/voice.mp3"
    assert payload["generate_audio"] is True


@pytest.mark.asyncio
async def test_unsafe_url_in_passthrough_raises():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(_ROUTED_PROVIDER_MAP), logger=_test_logger()
    )

    with pytest.raises(VideoGenerationError, match="Refusing to forward unsafe URL"):
        await adapter._build_payload(
            api_model_id="alibaba/wan-2.7",
            prompt="a campfire",
            video_meta={"params": {"audio": "file:///etc/passwd"}},
            video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
            frame_images=[],
            provider_options={},
        )


@pytest.mark.parametrize("supplied", [4, 25])
@pytest.mark.asyncio
async def test_the_passthrough_url_budget_bounds_entries_awaits_and_reporting(supplied):
    from open_webui_openrouter_pipe.integrations.video import _MAX_PASSTHROUGH_URLS

    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(_ROUTED_PROVIDER_MAP), logger=_test_logger()
    )
    checked: list[str] = []

    async def _is_safe(url):
        checked.append(url)
        return True

    cast(Any, adapter._pipe)._multimodal_handler._is_safe_url = _is_safe
    withheld: list[tuple[str, str]] = []
    urls = [f"https://example.invalid/{index}.mp4" for index in range(supplied)]

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {"videos": urls}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    expected = min(supplied, _MAX_PASSTHROUGH_URLS)
    options = (payload.get("provider") or {}).get("options") or {}
    kept = next((entry["videos"] for entry in options.values() if "videos" in entry), [])

    assert len(kept) == expected, f"kept {len(kept)}, expected {expected}"
    assert len(checked) == expected, (
        "each entry costs a name resolution awaited one at a time while the deployment-wide "
        f"video semaphore is held; {len(checked)} lookups for {supplied} supplied URLs"
    )
    assert bool(withheld) is (supplied > expected), (
        f"a drop must be reported and a non-drop must not. withheld={withheld!r}"
    )


@pytest.mark.asyncio
async def test_a_passthrough_url_is_withheld_when_no_provider_slug_is_known():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    withheld: list[tuple[str, str]] = []

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="a campfire",
        video_meta={"params": {"audio": "https://example.invalid/a.mp3"}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    assert "provider" not in payload, (
        "with no catalog slug there is no key OpenRouter would match, so the parameter must "
        f"not be written under a guessed one. got {payload.get('provider')!r}"
    )
    notice = VideoGenerationAdapter._withheld_notice(withheld)
    assert "audio" in notice, (
        "the requester set this and it is not being sent; silence here is what made the "
        f"vendor-prefix guess look like it worked for years. got {notice!r}"
    )
    assert "publishes no provider slug" in notice, (
        "this was withheld for want of a slug, not because the API rejects the field; "
        f"telling the user to set it on a chat model instead is unfollowable. got {notice!r}"
    )


@pytest.mark.asyncio
async def test_unsafe_url_in_the_operator_provider_options_hatch_raises():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(_ROUTED_PROVIDER_MAP), logger=_test_logger()
    )

    with pytest.raises(VideoGenerationError, match="Refusing to forward unsafe URL"):
        await adapter._build_payload(
            api_model_id="alibaba/wan-2.7",
            prompt="a campfire",
            video_meta={"params": {}},
            video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
            frame_images=[],
            provider_options={"atlas-cloud": {"audio": "file:///etc/passwd"}},
        )


@pytest.mark.asyncio
async def test_encode_frame_images_threads_user_into_gateway_read(monkeypatch):
    """`_encode_frame_images` must pass the requester `user=user_obj` into the
    gateway `read_file_record_base64` so the file read is authorised against the
    requester rather than read unconditionally."""
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    file_obj = SimpleNamespace(id="f1", user_id="u1", path="/srv/up/f1.png")
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
        _async_return(file_obj),
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: "image/png",
    )
    captured: dict[str, Any] = {}

    async def fake_read(file_obj_arg, chunk_size, max_bytes, *, user=None, **_kwargs):
        captured["user"] = user
        return base64.b64encode(b"png-bytes").decode("ascii")

    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", fake_read)

    user_obj = SimpleNamespace(id="u1")
    encoded = await adapter._encode_frame_images(
        {"frame_images": [{"id": "f1", "frame_type": "first_frame"}]},
        VIDEO_BY_ID["google/veo-3.1"],
        pipe.valves,
        user_obj=user_obj,
    )
    assert len(encoded) == 1
    assert encoded[0]["frame_type"] == "first_frame"
    assert captured["user"] is user_obj


@pytest.mark.asyncio
async def test_encode_frame_images_converts_required_file_error_to_video_error(monkeypatch):
    """A RequiredInternalFileError raised by the gateway for a required frame
    attachment must surface as a VideoGenerationError (with the user message),
    not leak the internal error type."""
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    file_obj = SimpleNamespace(id="f1", user_id="owner", path="/srv/up/f1.png")
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id",
        _async_return(file_obj),
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: "image/png",
    )

    async def fake_read(*_args, **_kwargs):
        raise RequiredInternalFileError("You do not have access to a referenced file.", denied=True)

    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", fake_read)

    with pytest.raises(VideoGenerationError, match="do not have access"):
        await adapter._encode_frame_images(
            {"frame_images": [{"id": "f1", "frame_type": "first_frame"}]},
            VIDEO_BY_ID["google/veo-3.1"],
            pipe.valves,
            user_obj=SimpleNamespace(id="not-the-owner"),
        )


def test_serialize_kind_marker_rejects_empty_body():
    from open_webui_openrouter_pipe.core.utils import _serialize_kind_marker

    with pytest.raises(ValueError, match="kind"):
        _serialize_kind_marker("", "abc")
    with pytest.raises(ValueError, match="body"):
        _serialize_kind_marker("videojob", "")


def test_serialize_kind_marker_round_trips_through_extractors():
    from open_webui_openrouter_pipe.core.utils import (
        _extract_kind_marker,
        _find_first_kind_marker_body,
        _iter_kind_marker_spans,
        _serialize_kind_marker,
    )

    line = _serialize_kind_marker("videojob", "job-abc")
    assert line == "[openrouter:v1:videojob:job-abc]: #"
    assert _extract_kind_marker(line) == ("videojob", "job-abc")

    text = (
        f"{_serialize_kind_marker('videojob', 'j-1')}\n"
        f"{_serialize_kind_marker('videomodel', 'google/veo-3.1')}\n\nhello"
    )
    spans = _iter_kind_marker_spans(text)
    assert [(s["kind"], s["body"]) for s in spans] == [
        ("videojob", "j-1"),
        ("videomodel", "google/veo-3.1"),
    ]
    assert _find_first_kind_marker_body(text, kind="videojob") == "j-1"
    assert _find_first_kind_marker_body(text, kind="missing") == ""


def test_coerce_video_usage_pads_zero_tokens():
    raw = {"cost": 0.42}
    out = VideoGenerationAdapter._coerce_video_usage(raw)
    assert out["cost"] == 0.42
    assert out["total_tokens"] == 0
    assert out["input_tokens"] == 0
    assert out["output_tokens"] == 0


def test_coerce_video_usage_zeros_default_when_input_is_none():
    out = VideoGenerationAdapter._coerce_video_usage(None)
    assert out["total_tokens"] == 0
    assert out["input_tokens"] == 0
    assert out["output_tokens"] == 0
    assert "cost" not in out


def test_format_final_status_renders_chat_style_token_bracket():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    usage = VideoGenerationAdapter._coerce_video_usage({"cost": 0.20})
    description = adapter._format_final_status(elapsed=4.0, usage=usage, valves=pipe.valves)
    assert "Total tokens: 0" in description
    assert "Input: 0" in description
    assert "Output: 0" in description


def test_build_success_content_has_no_time_or_cost_footer():
    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    content = adapter._build_success_content(
        job_id="j-1",
        model_id="google/veo-3.1-lite",
        file_ids=["file-1"],
        elapsed=4.2,
        usage={"cost": 0.20, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
    )
    assert "Generated in" not in content
    assert "· $" not in content
    assert "$0." not in content
    assert "</video>" in content
    assert content.endswith("\n")


@pytest.mark.asyncio
async def test_emit_completion_includes_usage_in_chat_completion_event():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    captured: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        captured.append(event)

    usage = {"cost": 0.20, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    await adapter._emit_completion(emitter, "hello", usage=usage)

    completion_events = [e for e in captured if e.get("type") == "chat:completion"]
    assert len(completion_events) == 1
    assert completion_events[0]["data"]["usage"] == usage
    delta_events = [e for e in captured if e.get("type") == "chat:message:delta"]
    assert len(delta_events) == 1
    assert "usage" not in delta_events[0]["data"]


@pytest.mark.asyncio
async def test_emit_completion_omits_usage_field_when_unset():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    captured: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        captured.append(event)

    await adapter._emit_completion(emitter, "hello")

    completion_events = [e for e in captured if e.get("type") == "chat:completion"]
    assert len(completion_events) == 1
    assert "usage" not in completion_events[0]["data"]


@pytest.mark.asyncio
async def test_pipe_close_cancels_in_process_video_lifecycles():
    pipe = Pipe()
    task_started = asyncio.Event()

    async def never_finishes():
        task_started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(never_finishes())
    pipe._video_active_tasks[("chat-1", "msg-1")] = task
    await task_started.wait()

    await pipe.close()

    assert task.cancelled()
    assert pipe._video_active_tasks == {}


def test_video_defaults_match_plan():
    valves = Pipe.Valves()

    assert valves.MAX_CONCURRENT_VIDEO_GENS == 2
    assert valves.MAX_CONCURRENT_VIDEO_GENS_PER_USER == 2
    assert valves.REMOTE_VIDEO_MAX_SIZE_MB == 500
    assert valves.VIDEO_MAX_POLL_TIME_SECONDS == 600
    assert valves.VIDEO_FRAME_TOTAL_MAX_BYTES == 50 * 1024 * 1024


def _ttl_test_valves():
    return SimpleNamespace(
        ENABLE_VIDEO_GENERATION=True,
        BASE_URL="https://openrouter.ai/api/v1",
        HTTP_REFERER_OVERRIDE="",
    )


@pytest.mark.asyncio
async def test_ensure_video_catalog_loaded_respects_ttl_within_window():
    from open_webui_openrouter_pipe.integrations.video_catalog import ensure_video_catalog_loaded

    OpenRouterModelRegistry._last_video_attempt = time.time()
    before_fetch = OpenRouterModelRegistry._last_video_fetch

    await ensure_video_catalog_loaded(
        session=cast(Any, None),
        valves=_ttl_test_valves(),
        api_key="unused",
        logger=logging.getLogger("test"),
        cache_seconds=3600,
    )

    assert OpenRouterModelRegistry._last_video_fetch == before_fetch


@pytest.mark.asyncio
async def test_ensure_video_catalog_loaded_refetches_after_ttl_expires(monkeypatch):
    from open_webui_openrouter_pipe.integrations import video_catalog as _vc

    OpenRouterModelRegistry._last_video_attempt = time.time() - 3601

    list_models_called = False

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            nonlocal list_models_called
            list_models_called = True
            return []

    monkeypatch.setattr(_vc, "OpenRouterVideoClient", _StubClient)

    await _vc.ensure_video_catalog_loaded(
        session=cast(Any, object()),
        valves=_ttl_test_valves(),
        api_key="test",
        logger=logging.getLogger("test"),
        cache_seconds=3600,
    )

    assert list_models_called


@pytest.mark.asyncio
async def test_ensure_video_catalog_loaded_empty_result_engages_attempt_ttl_without_bumping_fetch_timestamp(monkeypatch):
    from open_webui_openrouter_pipe.integrations import video_catalog as _vc

    OpenRouterModelRegistry._last_video_fetch = 100.0
    OpenRouterModelRegistry._last_video_attempt = 0.0

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            return []

    monkeypatch.setattr(_vc, "OpenRouterVideoClient", _StubClient)

    before_attempt = time.time()
    await _vc.ensure_video_catalog_loaded(
        session=cast(Any, object()),
        valves=_ttl_test_valves(),
        api_key="test",
        logger=logging.getLogger("test"),
        cache_seconds=3600,
    )

    assert OpenRouterModelRegistry._last_video_fetch == 100.0, (
        "_last_video_fetch (sync_key invalidator) must NOT bump on empty result"
    )
    assert OpenRouterModelRegistry._last_video_attempt >= before_attempt, (
        "_last_video_attempt (TTL gate) must bump on every fetch attempt — including empty"
    )


@pytest.mark.asyncio
async def test_ensure_video_catalog_loaded_network_failure_engages_attempt_ttl(monkeypatch):
    from open_webui_openrouter_pipe.integrations import video_catalog as _vc

    OpenRouterModelRegistry._last_video_fetch = 100.0
    OpenRouterModelRegistry._last_video_attempt = 0.0

    class _RaisingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            raise OSError("connection refused")

    monkeypatch.setattr(_vc, "OpenRouterVideoClient", _RaisingClient)

    before_attempt = time.time()
    await _vc.ensure_video_catalog_loaded(
        session=cast(Any, object()),
        valves=_ttl_test_valves(),
        api_key="test",
        logger=logging.getLogger("test"),
        cache_seconds=3600,
    )

    assert OpenRouterModelRegistry._last_video_fetch == 100.0, (
        "_last_video_fetch must NOT bump on network failure"
    )
    assert OpenRouterModelRegistry._last_video_attempt >= before_attempt, (
        "_last_video_attempt must bump even on network failure to prevent refetch storm"
    )


@pytest.mark.asyncio
async def test_ensure_video_catalog_loaded_propagates_cancelled_error_without_bumping_attempt(monkeypatch):
    from open_webui_openrouter_pipe.integrations import video_catalog as _vc

    OpenRouterModelRegistry._last_video_attempt = 0.0

    class _CancellingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            raise asyncio.CancelledError()

    monkeypatch.setattr(_vc, "OpenRouterVideoClient", _CancellingClient)

    with pytest.raises(asyncio.CancelledError):
        await _vc.ensure_video_catalog_loaded(
            session=cast(Any, object()),
            valves=_ttl_test_valves(),
            api_key="test",
            logger=logging.getLogger("test"),
            cache_seconds=3600,
        )

    assert OpenRouterModelRegistry._last_video_attempt == 0.0, (
        "CancelledError must NOT bump _last_video_attempt — user cancellation "
        "should not engage the TTL gate and block the next legitimate retry."
    )


def test_register_video_models_with_empty_list_does_not_bump_last_video_fetch():
    OpenRouterModelRegistry._last_video_fetch = 0.0

    OpenRouterModelRegistry.register_video_models([])

    assert OpenRouterModelRegistry._last_video_fetch == 0.0


def test_register_video_models_with_models_does_bump_last_video_fetch():
    OpenRouterModelRegistry._last_video_fetch = 0.0
    before = time.time()

    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID["openai/sora-2-pro"]])

    assert OpenRouterModelRegistry._last_video_fetch >= before


@pytest.mark.asyncio
async def test_safe_emit_always_reraises_cancelled_error():
    """Regression: _safe_emit must NOT swallow CancelledError under any condition.

    Bug class: silently consuming CancelledError leaves anyio cancel scopes in
    a confused state where a task is marked done but still tracked, triggering
    the upstream anyio bug #1111 (100% CPU spin in _deliver_cancellation).
    """
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())

    async def cancelling_emitter(_event):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await adapter._safe_emit(cancelling_emitter, {"type": "status", "data": {}})


@pytest.mark.asyncio
async def test_safe_emit_swallows_non_cancelled_exceptions():
    """Non-CancelledError exceptions are still suppressed (debug-logged) so
    transient emit failures don't take down the lifecycle."""
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())

    async def failing_emitter(_event):
        raise RuntimeError("emit failed")

    await adapter._safe_emit(failing_emitter, {"type": "status", "data": {}})


@pytest.mark.asyncio
async def test_passthrough_validation_uses_async_safe_url(monkeypatch):
    """Passthrough URL validation must use the async _is_safe_url (DNS offloaded
    to a thread), not _is_safe_url_blocking which stalls the shared event loop."""
    pipe = Pipe()
    adapter = pipe._ensure_video_generation_adapter()
    handler = pipe._multimodal_handler

    import threading

    loop_thread = threading.get_ident()
    resolved_on: list = []

    def recording_blocking(url):
        resolved_on.append((url, threading.get_ident()))
        return ["203.0.113.9"]

    monkeypatch.setattr(handler, "_request_ips_blocking", recording_blocking)

    payload = {"video": "https://example.test/clip.mp4"}
    await adapter._validate_passthrough_urls(payload)

    assert [url for url, _ in resolved_on] == ["https://example.test/clip.mp4"], (
        "the passthrough URL was not validated at all"
    )
    assert all(tid != loop_thread for _, tid in resolved_on), (
        "the SSRF gate resolved DNS on the event loop's own thread. getaddrinfo blocks, "
        "so every other request on this worker stalls for the length of the lookup -- "
        "which is why the async wrapper offloads it."
    )


@pytest.mark.asyncio
async def test_cancellation_during_cleanup_still_releases_every_resource(monkeypatch):
    """A generation cancelled while finalising must not strand its locks or slots.

    The cleanup tail awaits `_video_active_tasks_dict_lock`, which every video request
    touches, so suspending there is routine rather than exotic. Cancellation delivered
    at that point used to abandon the two releases below it: the per-user slot stayed
    spent (that user locked out until restart) and the per-message lock stayed held,
    which deadlocks every later request for the same message because
    `_acquire_message_lock` has no timeout.
    """
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
        key = ("chat-1", "msg-1")
        user_id = "user-1"
        job_id = "job-1"

        async def never_finishes(*_args, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(
            "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient",
            lambda *a, **k: SimpleNamespace(),
        )
        monkeypatch.setattr(pipe, "_create_http_session", lambda *a, **k: _FakeSession([]))
        monkeypatch.setattr(adapter, "_poll_until_terminal", never_finishes)

        message_lock = await adapter._acquire_message_lock(key)
        assert message_lock.locked()
        pipe._video_user_active_counts[user_id] = 1
        pipe._video_user_active_jobs[user_id] = {job_id}
        pipe._video_user_locks[user_id] = asyncio.Lock()
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()

        task = asyncio.create_task(
            adapter._run_lifecycle_after_submit(
                key=key,
                job_id=job_id,
                api_model_id="openai/sora-2-pro",
                normalized_model_id="openai.sora-2-pro",
                valves=pipe.valves,
                event_emitter=None,
                user={"id": user_id},
                user_obj={"id": user_id},
                chat_id="chat-1",
                message_id="msg-1",
                request=None,
                user_id=user_id,
                global_semaphore=semaphore,
                message_lock=message_lock,
                started_at=time.monotonic(),
            )
        )
        for _ in range(50):
            await asyncio.sleep(0)

        await pipe._video_active_tasks_dict_lock.acquire()
        task.cancel()
        for _ in range(50):
            await asyncio.sleep(0)
            if pipe._video_active_tasks_dict_lock._waiters:
                break
        assert pipe._video_active_tasks_dict_lock._waiters, (
            "cleanup never reached the dict-lock await; the test is not exercising "
            "the cancellation point it exists for"
        )
        task.cancel()
        for _ in range(50):
            await asyncio.sleep(0)
        pipe._video_active_tasks_dict_lock.release()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        for _ in range(200):
            await asyncio.sleep(0)
            if not message_lock.locked():
                break

        assert not message_lock.locked(), (
            "per-message lock stranded: the next video request for this message "
            "would hang forever"
        )
        assert pipe._video_user_active_counts.get(user_id, 0) == 0, (
            "per-user slot stranded: this user is locked out of video generation"
        )
        assert semaphore._value == 1, "global video permit lost for the process lifetime"
    finally:
        await pipe.close()


def test_every_video_cleanup_await_is_shielded():
    """Both `finally` blocks that release video resources must shield their awaits.

    The cancellation fix was applied to the lifecycle tail but not to generate()'s own
    finally, which has the identical shape. A behavioural test of the helper cannot
    catch that -- it shields the call itself, so removing production's shield changes
    nothing. This asserts the structure instead: any `await` inside a video cleanup
    `finally` must be wrapped in asyncio.shield, or a cancellation delivered there
    abandons the releases below it (the per-user slot stays spent and the per-message
    lock stays held, and _acquire_message_lock has no timeout).
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe" / "integrations" / "video.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    checked, unshielded = 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for stmt in node.finalbody:
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.Await):
                    continue
                call = inner.value
                checked += 1
                shielded = (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "shield"
                )
                if not shielded:
                    unshielded.append(f"video.py:{inner.lineno}: {ast.unparse(inner)[:80]}")

    # No name filter and no floor. The filter matched `_release_`/`_finalize_generation`
    # and so skipped `await session.close()` -- which was genuinely unshielded -- while
    # the floor of 2 was exactly the population it matched, so a new cleanup helper
    # under any other name moved neither number. Every await in a `finally` is checked;
    # a legitimate exception goes in the dict below and is verified to still match.
    assert checked >= 3, (
        f"only {checked} awaits found in any finally block; the scan has gone blind"
    )
    assert not unshielded, (
        "these awaits sit in a cleanup `finally` without asyncio.shield, so a "
        "cancellation delivered there strands every release below them:\n  "
        + "\n  ".join(unshielded)
    )


@pytest.mark.asyncio
async def test_a_failing_cleanup_step_does_not_skip_the_others():
    """One release failing must not strand the rest, and must not vanish.

    Both cleanup helpers run inside asyncio.shield. Without per-step isolation the
    first failure propagates out of the shielded coroutine, skipping every release
    below it -- and because nothing awaits that coroutine, the exception surfaces
    only as asyncio's default handler output, if at all.
    """
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
        key = ("chat-s", "msg-s")
        user_id = "user-s"

        message_lock = await adapter._acquire_message_lock(key)
        assert message_lock.locked()

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("user-slot release exploded")

        adapter._release_user_slot = _boom

        await adapter._release_presubmit_slots(
            key, user_id, "job-s", message_lock, release_user_slot=True
        )

        assert not message_lock.locked(), (
            "a failure in the user-slot release skipped the message-lock release, "
            "stranding it for the lifetime of the process"
        )
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_lifecycle_cleanup_step_failure_does_not_skip_the_others():
    """Same guarantee as the pre-submit twin, on the long-lived generation path.

    _finalize_generation has three steps rather than two, and only the twin was
    covered -- so its isolation could be removed with the whole suite green.
    """
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
        key = ("chat-l", "msg-l")
        user_id = "user-l"

        message_lock = await adapter._acquire_message_lock(key)
        assert message_lock.locked()

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("user-slot release exploded")

        adapter._release_user_slot = _boom

        await adapter._finalize_generation(key, user_id, "job-l", message_lock)

        assert not message_lock.locked(), (
            "a failure in the user-slot release skipped the message-lock release, "
            "stranding it for the lifetime of the process"
        )
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "should_emit"),
    [
        ("chat-ok", True),
        ("temporary:abc", False),
        ("channel:abc", False),
        ("local:abc", False),
    ],
)
async def test_the_pending_placeholder_is_only_emitted_for_a_storable_chat(
    monkeypatch, chat_id, should_emit
):
    """The one is_linkable_chat call site that no test reached.

    Every other consumer of the predicate is covered -- removing the prefix check
    inside is_linkable_chat itself kills a dozen tests -- but deleting
    `and is_linkable_chat(chat_id)` here left the whole suite green.

    Without it the pending placeholder, carrying the resume marker, is emitted into
    Temporary Chats and channel invocations, which have no chat row. If the turn is
    interrupted the user holds a marker for a conversation that cannot be looked up:
    VideoPersistence.load_message refuses those ids by design.

    Asserted on the emitted event rather than on the predicate, because the predicate
    is already covered three times over; what was missing is that this site consults it.
    """
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence("")

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, _payload):
            return {"id": "job-gate", "status": "queued"}

        async def status(self, _job_id, polling_url=None):
            await asyncio.sleep(3600)

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    task = asyncio.create_task(
        adapter.generate(
            body={"messages": [{"role": "user", "content": "make a video"}]},
            responses_body=SimpleNamespace(provider={}),
            valves=pipe.valves,
            session=object(),
            event_emitter=emitter,
            metadata={"chat_id": chat_id, "message_id": "msg-g", "user_id": "user-g"},
            user={"id": "user-g"},
            request=None,
            user_obj={"id": "user-g"},
            normalized_model_id="openai.sora-2-pro",
            api_model_id="openai/sora-2-pro",
        )
    )
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            if any(e.get("type") == "message" for e in emitted):
                break
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await pipe.close()

    markers = [
        e
        for e in emitted
        if e.get("type") == "message"
        and "openrouter:v1:videojob:" in str(e.get("data", {}).get("content", ""))
    ]
    if should_emit:
        assert markers, (
            f"no resume marker was emitted for chat_id={chat_id!r}; an interrupted turn "
            "would leave the user with no way to resume the job"
        )
    else:
        assert not markers, (
            f"a resume marker was emitted for chat_id={chat_id!r}, which has no chat "
            "row. VideoPersistence.load_message refuses that id, so the marker points "
            "at a conversation that cannot be looked up."
        )


@pytest.mark.asyncio
async def test_a_temp_directory_that_cannot_be_removed_is_reported(monkeypatch, tmp_path, caplog):
    """The leak warning must be reachable for the failure its own message names.

    It was not: the call kept `ignore_errors=True`, which swallows every OSError inside
    rmtree, so the `except` around it could never fire and the leak stayed exactly as
    silent as before the "surface swallowed failures" work.

    The failure is produced by pointing mkdtemp's result at a REGULAR FILE, so rmtree
    raises NotADirectoryError on every platform and regardless of uid. Monkeypatching
    rmtree to raise would prove nothing -- a stub raises whether or not ignore_errors
    is present, so that test passes against the unfixed code. chmod would be a no-op
    for uid 0, which is common in container CI.
    """
    import logging

    from open_webui_openrouter_pipe.integrations.video_types import VideoLifecycleResult

    not_a_directory = tmp_path / "definitely-a-file"
    not_a_directory.write_text("x", encoding="utf-8")

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def status(self, _job_id, polling_url=None):
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id, index=0):
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self):
            return {"Authorization": "Bearer test"}

    async def fake_streaming_download(url, dest_path, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4_BYTES)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": len(MP4_BYTES)}

    async def fake_upload_from_path(*_args, **_kwargs):
        return "file-1"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.tempfile.mkdtemp",
        lambda **_kw: str(not_a_directory),
    )
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_a, **_k: _FakeSession([]))
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url_streaming", fake_streaming_download
    )
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path)

    async def emitter(_event):
        return None

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    message_lock = asyncio.Lock()
    await message_lock.acquire()

    try:
        with caplog.at_level(logging.WARNING, logger=pipe.logger.name):
            result = await adapter._run_lifecycle_after_submit(
                key=("chat-1", "msg-1"),
                job_id="job-leak",
                api_model_id="openai/sora-2-pro",
                normalized_model_id="openai.sora-2-pro",
                valves=pipe.valves,
                event_emitter=emitter,
                user={"id": "user-1"},
                user_obj={"id": "user-1"},
                chat_id="chat-1",
                message_id="msg-1",
                request=None,
                user_id="user-1",
                global_semaphore=semaphore,
                message_lock=message_lock,
                started_at=time.monotonic(),
            )
    finally:
        await pipe.close()

    assert isinstance(result, VideoLifecycleResult)
    assert any("leak one" in m for m in caplog.messages), (
        f"rmtree failed and nothing was reported; messages were {caplog.messages!r}. "
        "The directory leaks once per generation and the operator never finds out."
    )


@pytest.mark.parametrize(
    "written",
    [
        {"negativePrompt": "blur", "steps": 40},
        {"parameters": {"negativePrompt": "blur", "steps": 40}},
        {"parameters": {"negativePrompt": "blur"}, "steps": 40},
    ],
)
def test_provider_options_emit_one_flat_shape_however_the_operator_nested_them(written):
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    result = adapter._normalise_provider_options({"google-vertex": written})

    assert result == {"google-vertex": {"negativePrompt": "blur", "steps": 40}}, (
        "OpenRouter reads provider.options.<slug> flat — the ByteDance watermark A/B proved a "
        "`parameters` wrapper is inert. Emitting a different shape because of an unrelated "
        "sibling key means the same operator config silently works or does nothing."
    )


def test_derived_provider_params_beat_a_legacy_parameters_block():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    result = adapter._normalise_provider_options(
        {"seed": {"parameters": {"watermark": True}, "watermark": False}}
    )

    assert result["seed"]["watermark"] is False, (
        "_build_payload merges derived filter params over the manual hatch as siblings; "
        "flattening must not invert that precedence."
    )


@pytest.mark.parametrize(
    ("model_id", "routed_slug"),
    [
        ("google/veo-3.1", "google-vertex"),
        ("kwaivgi/kling-v3.0-std", "atlas-cloud"),
        ("bytedance/seedance-2.0", "seed"),
    ],
)
def test_provider_slug_comes_from_the_catalog_not_the_model_id_prefix(model_id, routed_slug):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({model_id: {"providers": [routed_slug]}}),
        logger=_test_logger(),
    )

    candidates = adapter._provider_slug_candidates({"id": model_id}, model_id)

    assert candidates == [routed_slug], (
        "the vendor prefix is a fallback for the empty-catalog case, not an addition: writing "
        f"the same params under a second, wrong slug is a silent drop. got {candidates!r}"
    )


@pytest.mark.parametrize(
    "model_id", ["openai/sora-2-pro", "google/veo-3.1", "kwaivgi/kling-v2.5-turbo-pro"]
)
def test_the_vendor_prefix_is_never_used_as_a_provider_slug(caplog, model_id):
    from open_webui_openrouter_pipe.integrations import video as video_module

    video_module._warned_provider_slug_guess.clear()
    adapter = VideoGenerationAdapter(pipe=_pipe_with_provider_map({}), logger=_test_logger())

    with caplog.at_level(logging.DEBUG):
        candidates = adapter._provider_slug_candidates({"id": model_id}, model_id)

    assert candidates == [], (
        "google/veo is served by google-vertex and kwaivgi/kling by atlas-cloud, so a vendor "
        "prefix is not a provider slug. Writing parameters under one keys them to a provider "
        f"OpenRouter never matches and drops them silently. got {candidates!r}"
    )


def test_a_missing_provider_slug_warns_once_per_model(caplog):
    from open_webui_openrouter_pipe.integrations import video as video_module

    video_module._warned_provider_slug_guess.clear()
    adapter = VideoGenerationAdapter(pipe=_pipe_with_provider_map({}), logger=_test_logger())

    with caplog.at_level(logging.DEBUG):
        adapter._provider_slug_candidates({"id": "openai/sora-2-pro"}, "openai/sora-2-pro")
        adapter._provider_slug_candidates({"id": "openai/sora-2-pro"}, "openai/sora-2-pro")
        adapter._provider_slug_candidates({"id": "meta/movie-gen"}, "meta/movie-gen")

    levels = [
        record.levelno
        for record in caplog.records
        if "No catalog provider slug" in record.getMessage()
    ]
    assert levels == [logging.WARNING, logging.DEBUG, logging.WARNING], (
        "warn-once means the first miss for a model warns, repeats drop to DEBUG, and a "
        f"different model warns again; got {levels!r}"
    )


def test_the_catalog_slug_is_found_through_a_tilde_alias():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(_ROUTED_PROVIDER_MAP), logger=_test_logger()
    )

    assert adapter._provider_slug_candidates(
        {"id": "~alibaba/wan-2.7"}, "~alibaba/wan-2.7"
    ) == ["atlas-cloud"]


@pytest.mark.asyncio
async def test_a_documented_top_level_field_is_never_shipped_inside_provider_options():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={
            "params": {
                "duration": 5,
                "resolution": "1080p",
                "size": "1920x1080",
                "provider": "not-a-provider",
                "callback_url": "https://evil.example/hook",
                "negativePrompt": "blur",
            }
        },
        video_model={
            "id": "vendor/model",
            "allowed_passthrough_parameters": [
                "duration",
                "resolution",
                "size",
                "provider",
                "callback_url",
                "negativePrompt",
            ],
            "supported_durations": [5, 10],
            "supported_resolutions": ["1080p"],
            "supported_sizes": ["1920x1080"],
        },
        frame_images=[],
        provider_options={},
    )

    assert payload["duration"] == 5
    assert payload["resolution"] == "1080p"
    assert payload["size"] == "1920x1080"
    assert "resolution" not in _provider_params(payload, "slug-x")
    assert "size" not in _provider_params(payload, "slug-x")
    assert _provider_params(payload, "slug-x") == {"negativePrompt": "blur"}, (
        "the subtraction must be selective: a documented top-level name is removed from the "
        "passthrough set, everything else still goes through"
    )


@pytest.mark.asyncio
async def test_an_inline_attachment_is_not_duplicated_once_per_provider_slug():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(
            {"alibaba/wan-2.7": {"providers": ["atlas-cloud", "alibaba", "fal"]}}
        ),
        logger=_test_logger(),
    )
    data_url = "data:video/mp4;base64," + ("A" * 200_000)

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {"negative_prompt": "blur"}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=[data_url],
    )

    serialized = len(json.dumps(payload))
    assert serialized < int(len(data_url) * 1.2), (
        "the attachment must appear once however many providers the catalog lists; "
        f"payload is {serialized} bytes for a {len(data_url)}-byte attachment"
    )
    options = payload["provider"]["options"]
    assert set(options) == {"atlas-cloud", "alibaba", "fal"}
    assert all(entry.get("negative_prompt") == "blur" for entry in options.values()), (
        "the scalar knobs must still reach every candidate; only the bulky values are pinned"
    )
    carriers = [slug for slug, entry in options.items() if "video" in entry]
    assert carriers == ["atlas-cloud"]


@pytest.mark.asyncio
async def test_a_parameter_that_reaches_neither_destination_is_reported(caplog):
    from open_webui_openrouter_pipe.integrations import video as video_module

    video_module._warned_dropped_video_param.clear()
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    with caplog.at_level(logging.DEBUG):
        payload = await adapter._build_payload(
            api_model_id="vendor/model",
            prompt="x",
            video_meta={
                "params": {"resolution": "1080p", "surprise": "x", "negativePrompt": "blur"}
            },
            video_model={
                "id": "vendor/model",
                "allowed_passthrough_parameters": ["resolution", "negativePrompt"],
            },
            frame_images=[],
            provider_options={},
        )

    assert "resolution" not in payload
    assert "resolution" not in _provider_params(payload, "slug-x")
    dropped = [r for r in caplog.records if "resolution" in r.getMessage() and "Dropping" in r.getMessage()]
    other = [r for r in caplog.records if "surprise" in r.getMessage() and "Dropping" in r.getMessage()]
    assert other and other[0].levelno == logging.WARNING, (
        "the latch keys on model+parameter: the operator must learn about every dropped knob, "
        "not just the first one for that model"
    )
    assert dropped, (
        "a supplied parameter that reaches neither the top level nor provider.options is a "
        "silent drop unless the operator can find it in the log"
    )
    assert dropped[0].levelno == logging.WARNING, (
        "the first drop for a model must be visible to an operator running at the default "
        f"level, not buried at DEBUG; got {logging.getLevelName(dropped[0].levelno)}"
    )


def test_the_catalog_id_comes_from_the_model_record_not_the_request_id():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(_ROUTED_PROVIDER_MAP), logger=_test_logger()
    )

    assert adapter._provider_slug_candidates(
        {"id": "google/veo-3.1"}, "some/other-id"
    ) == ["google-vertex"], "video_model['id'] is the catalog key when it is present"
    assert adapter._provider_slug_candidates({}, "google/veo-3.1") == ["google-vertex"], (
        "api_model_id is the fallback when the record carries no id"
    )


@pytest.mark.asyncio
async def test_derived_params_reach_every_provider_the_catalog_lists():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["a", "b"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {"negativePrompt": "blur"}},
        video_model={
            "id": "vendor/model",
            "allowed_passthrough_parameters": ["negativePrompt"],
        },
        frame_images=[],
        provider_options={},
    )

    assert set(payload["provider"]["options"]) == {"a", "b"}, (
        "OpenRouter picks the serving provider; options under only one of the candidates are "
        "silently ignored if it routes to the other"
    )


@pytest.mark.parametrize("value", [{"k": 1}, {"k": 2}])
def test_the_operator_provider_options_reach_the_extractor_from_the_request(value):
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    assert adapter._extract_provider_options({"options": {"slug-x": value}}, {}) == {
        "slug-x": value
    }


@pytest.mark.parametrize("value", [{"k": 1}, {"k": 2}])
def test_the_operator_provider_options_fall_back_to_pipe_metadata(value):
    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    assert adapter._extract_provider_options(
        None, {_PIPE_METADATA_KEY: {"provider": {"options": {"slug-x": value}}}}
    ) == {"slug-x": value}


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_the_requester_is_told_which_video_preferences_were_withheld():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )
    withheld: list[tuple[str, str]] = []

    await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {}},
        video_model={"id": "vendor/model"},
        frame_images=[],
        provider_options={},
        provider_block={"zdr": True, "data_collection": "deny", "only": ["fal"]},
        withheld=withheld,
    )

    notice = VideoGenerationAdapter._withheld_notice(withheld)
    for key in ("zdr", "data_collection", "only"):
        assert key in notice, (
            "an admin who set a retention control and never hears otherwise believes it is "
            f"in force; a log line the requester cannot see is not telling them. got {notice!r}"
        )
    assert "does not define this key" in notice, (
        f"these were withheld because the schema has no such field. got {notice!r}"
    )


@pytest.mark.asyncio
async def test_the_video_payload_carries_only_what_the_video_schema_defines(caplog):
    from open_webui_openrouter_pipe.integrations import video as video_module

    video_module._warned_video_provider_keys.clear()
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    with caplog.at_level(logging.WARNING):
        payload = await adapter._build_payload(
            api_model_id="vendor/model",
            prompt="x",
            video_meta={"params": {"negativePrompt": "blur"}},
            video_model={"id": "vendor/model", "allowed_passthrough_parameters": ["negativePrompt"]},
            frame_images=[],
            provider_options={},
            provider_block={"only": ["fal"], "sort": "price", "zdr": True},
        )

    provider = payload["provider"]
    assert set(provider) == {"options"}, (
        "VideoGenerationRequestProvider defines exactly one property, options. zdr and the "
        "routing keys are undefined there and no additionalProperties:false rejects them, so "
        f"sending them buys nothing and reads as a control in force. sent={sorted(provider)}"
    )
    assert provider["options"]["slug-x"] == {"negativePrompt": "blur"}
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    for key in ("only", "sort", "zdr"):
        assert key in warned, (
            f"{key} was withheld but the operator was never told; a silently dropped privacy "
            "preference is indistinguishable from an honoured one"
        )


@pytest.mark.asyncio
async def test_the_drop_reason_distinguishes_its_two_causes(caplog):
    from open_webui_openrouter_pipe.integrations import video as video_module

    video_module._warned_dropped_video_param.clear()
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    with caplog.at_level(logging.DEBUG):
        await adapter._build_payload(
            api_model_id="vendor/model",
            prompt="x",
            video_meta={"params": {"resolution": "1080p", "surprise": "x"}},
            video_model={
                "id": "vendor/model",
                "allowed_passthrough_parameters": ["resolution"],
            },
            frame_images=[],
            provider_options={},
        )

    drops = [m for m in caplog.messages if "Dropping video parameter" in m]
    assert len(drops) == 2, f"expected two drop records, got {drops!r}"
    documented = next(m for m in drops if "resolution" in m)
    unknown = next(m for m in drops if "surprise" in m)
    assert documented != unknown
    assert "documented top-level field" in documented, (
        "the catalog DOES list resolution as an allowed passthrough; saying otherwise sends the "
        f"operator to check a claim that is false. got {documented!r}"
    )
    assert "does not list it as an allowed passthrough" in unknown


@pytest.mark.parametrize("pin", ["fal", "atlas-cloud"])
@pytest.mark.asyncio
async def test_a_pin_the_video_api_never_receives_does_not_decide_the_carrier(pin):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(
            {"alibaba/wan-2.7": {"providers": ["alibaba", "atlas-cloud", "fal"]}}
        ),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {"negative_prompt": "blur"}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        provider_block={"only": [pin]},
        video_attachment_urls=["data:video/mp4;base64,AAAA"],
    )

    options = payload["provider"]["options"]
    carriers = [slug for slug, entry in options.items() if "video" in entry or "videos" in entry]
    assert carriers == ["alibaba"], (
        "VideoGenerationRequestProvider carries no `only`, so routing never sees the pin. "
        "Keying the attachment to it would guarantee the provider that does serve the "
        f"request receives nothing. got {options!r}"
    )
    assert set(payload["provider"]) == {"options"}


@pytest.mark.asyncio
async def test_without_a_pin_the_attachment_uses_the_first_candidate():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(
            {"alibaba/wan-2.7": {"providers": ["alibaba", "atlas-cloud"]}}
        ),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=["data:video/mp4;base64,AAAA"],
    )

    options = payload["provider"]["options"]
    carriers = [slug for slug, entry in options.items() if "video" in entry]
    assert carriers == ["alibaba"]


@pytest.mark.parametrize("order", [["fal", "atlas-cloud"], ["atlas-cloud", "fal"]])
@pytest.mark.asyncio
async def test_provider_order_decides_the_attachment_carrier_without_an_only(order):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(
            {"alibaba/wan-2.7": {"providers": ["alibaba", "atlas-cloud", "fal"]}}
        ),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        provider_block={"order": order},
        video_attachment_urls=["data:video/mp4;base64,AAAA"],
    )

    options = payload["provider"]["options"]
    carriers = [slug for slug, entry in options.items() if "video" in entry]
    assert carriers == [order[0]], (
        "provider.order is a routing preference; the attachment must follow the provider "
        f"routing will try first. got {options!r}"
    )


@pytest.mark.parametrize("derived", ["blur", "grain"])
@pytest.mark.asyncio
async def test_a_request_param_overrides_a_stale_operator_hatch_entry(derived):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {"negativePrompt": derived}},
        video_model={"id": "vendor/model", "allowed_passthrough_parameters": ["negativePrompt"]},
        frame_images=[],
        provider_options={"slug-x": {"negativePrompt": "stale"}},
    )

    assert _provider_params(payload, "slug-x")["negativePrompt"] == derived, (
        "a leftover entry in the operator's JSON hatch must not override what the user just "
        "set in the video filter"
    )


@pytest.mark.parametrize("field", ["video", "videos", "images", "last_image", "audio"])
@pytest.mark.asyncio
async def test_no_bulky_attachment_is_duplicated_across_provider_slugs(field):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map(
            {"alibaba/wan-2.7": {"providers": ["alibaba", "atlas-cloud", "fal"]}}
        ),
        logger=_test_logger(),
    )
    blob = "data:video/mp4;base64," + ("A" * 200_000)
    value = [{"url": blob}] if field in ("videos", "images") else blob

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {field: value}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
    )

    options = (payload.get("provider") or {}).get("options") or {}
    carriers = [slug for slug, entry in options.items() if field in entry]
    assert len(carriers) <= 1, (
        f"{field} is bounded by VIDEO_MAX_SIZE_MB per copy; writing it under every candidate "
        f"multiplies the request by the provider count. carried by {carriers}"
    )
    if carriers:
        assert len(json.dumps(payload)) < int(len(blob) * 1.3)


@pytest.mark.asyncio
async def test_a_non_dict_provider_options_entry_is_ignored_not_fatal():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {}},
        video_model={"id": "vendor/model", "allowed_passthrough_parameters": []},
        frame_images=[],
        provider_options={"slug-x": "not-a-dict"},
    )

    assert "slug-x" not in ((payload.get("provider") or {}).get("options") or {}), (
        "an operator typo in VIDEO_PROVIDER_OPTIONS_JSON must not raise inside payload building"
    )


@pytest.mark.asyncio
async def test_a_plain_video_request_emits_no_empty_provider_block():
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {}},
        video_model={"id": "vendor/model", "allowed_passthrough_parameters": []},
        frame_images=[],
        provider_options={},
    )

    assert "provider" not in payload


@pytest.mark.parametrize("documented", ["model", "prompt", "frame_images", "input_references"])
@pytest.mark.asyncio
async def test_every_documented_top_level_name_is_kept_out_of_provider_options(documented):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {documented: "value-x", "negativePrompt": "blur"}},
        video_model={
            "id": "vendor/model",
            "allowed_passthrough_parameters": [documented, "negativePrompt"],
        },
        frame_images=[],
        provider_options={},
    )

    assert documented not in _provider_params(payload, "slug-x"), (
        f"{documented} is a documented top-level field; smuggling it into provider.options is "
        "what the subtraction exists to prevent"
    )
    assert _provider_params(payload, "slug-x")["negativePrompt"] == "blur"


def test_a_provider_options_slug_is_matched_after_trimming():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    assert adapter._normalise_provider_options({"  slug-x  ": {"k": 1}}) == {"slug-x": {"k": 1}}


@pytest.mark.parametrize("value", ["blur", "grain"])
@pytest.mark.asyncio
async def test_every_serving_candidate_receives_the_knobs(value):
    adapter = VideoGenerationAdapter(
        pipe=_pipe_with_provider_map({"vendor/model": {"providers": ["slug-x"]}}),
        logger=_test_logger(),
    )

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="x",
        video_meta={"params": {"negativePrompt": value}},
        video_model={"id": "vendor/model", "allowed_passthrough_parameters": ["negativePrompt"]},
        frame_images=[],
        provider_options={},
        provider_block={"only": ["fal"]},
    )

    options = (payload.get("provider") or {}).get("options") or {}
    assert options.get("slug-x", {}).get("negativePrompt") == value, (
        "a scalar knob can be duplicated across slugs, so every provider routing might "
        f"select must carry it. got {options!r}"
    )
    assert "fal" not in options, (
        "the video API carries no `only`, so a slug drawn from an unsent pin can never be "
        f"selected; writing knobs there is dead weight on the request. got {options!r}"
    )


@pytest.mark.parametrize(
    ("published", "expected_fields"),
    [
        (["contentModeration", "keyframes"], ["VIDEO_CONTENTMODERATION", "VIDEO_KEYFRAMES"]),
        (["safety_tolerance", "version"], ["VIDEO_SAFETY_TOLERANCE", "VIDEO_VERSION"]),
        (["aigc_watermark"], ["VIDEO_AIGC_WATERMARK"]),
        ([], []),
    ],
)
def test_a_published_setting_with_no_purpose_built_control_is_still_offered(
    published, expected_fields
):
    """OpenRouter names these and publishes nothing about their values.

    Dropping them means a capability the model accepts that the user cannot reach; the
    previous behaviour was to log a warning and silently omit them. Free text is the only
    honest rendering when there is no domain to render from.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    model = {
        "id": "vendor/model",
        "name": "Model",
        "allowed_passthrough_parameters": published,
    }
    module = _load_filter_from_source(
        render_video_filter_source(model_id="vendor/model", video_model=model),
        f"generic_passthrough_{len(published)}_{published[0] if published else 'none'}",
    )
    for field in expected_fields:
        assert field in module.Filter.UserValves.model_fields, (
            f"{field} is published by the model and must be offered"
        )


def test_a_published_setting_travels_and_a_broken_container_is_named():
    """Plain text goes as text; JSON goes as JSON; a broken container names its field."""
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    model = {
        "id": "vendor/model",
        "name": "Model",
        "allowed_passthrough_parameters": ["contentModeration", "keyframes"],
    }
    module = _load_filter_from_source(
        render_video_filter_source(model_id="vendor/model", video_model=model),
        "generic_passthrough_travel",
    )

    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        {"files": []},
        __metadata__=metadata,
        __user__={
            "valves": module.Filter.UserValves(
                VIDEO_CONTENTMODERATION="low",
                VIDEO_KEYFRAMES='[{"at": 0}]',
            )
        },
    )
    params = metadata["openrouter_pipe"]["video_generation"]["params"]
    assert params.get("contentModeration") == "low", f"plain text must travel as text; got {params}"
    assert params.get("keyframes") == [{"at": 0}], f"JSON must travel parsed; got {params}"

    with pytest.raises(Exception) as caught:
        module.Filter().inlet(
            {"files": []},
            __metadata__={},
            __user__={"valves": module.Filter.UserValves(VIDEO_KEYFRAMES="[{broken")},
        )
    assert "keyframes" in str(caught.value), f"the message must name the field; got {caught.value}"


@pytest.mark.parametrize(
    ("published", "expect_typed"),
    [
        (["duration"], "VIDEO_DURATION"),
        (["resolution"], "VIDEO_RESOLUTION"),
        (["seed"], "VIDEO_SEED"),
    ],
)
def test_a_published_name_never_replaces_the_control_built_for_it(published, expect_typed):
    """A model can publish a name a purpose-built control already covers.

    Rendering a free-text field for it too defines the same attribute twice; pydantic
    keeps the last, so the dropdown built from the model's own published values silently
    becomes an unvalidated text box and the value reaches the API as a string.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    model = {
        "id": "vendor/model",
        "name": "Model",
        "supported_durations": [5, 10],
        "supported_resolutions": ["720p", "1080p"],
        "seed": True,
        "allowed_passthrough_parameters": published,
    }
    source = render_video_filter_source(model_id="vendor/model", video_model=model)
    assert source.count(f"{expect_typed}:") == 1, (
        f"{expect_typed} must be defined once; the published name must not add a second"
    )

    module = _load_filter_from_source(source, f"no_shadow_{published[0]}")
    annotation = module.Filter.UserValves.model_fields[expect_typed].annotation
    assert annotation is not str, (
        f"{expect_typed} must keep the type built from the model's published values, "
        f"got {annotation}"
    )


def test_the_unreachable_warning_names_only_what_cannot_be_offered(caplog):
    """A name that is offered must not be reported as dropped.

    The warning told operators to add a branch for parameters that are already rendered,
    and the branch it asked for is what creates the collision above.
    """
    import logging

    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    with caplog.at_level(logging.WARNING):
        render_video_filter_source(
            model_id="vendor/ok",
            video_model={
                "id": "vendor/ok",
                "name": "M",
                "allowed_passthrough_parameters": ["contentModeration"],
            },
        )
    assert "contentModeration" not in caplog.text, (
        "this parameter is rendered and sent, so it must not be reported as dropped"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        render_video_filter_source(
            model_id="vendor/odd",
            video_model={
                "id": "vendor/odd",
                "name": "M",
                "allowed_passthrough_parameters": ["cfg-scale"],
            },
        )
    assert "cfg-scale" in caplog.text, (
        "a name that cannot become a form field is genuinely unreachable and must be named"
    )


def test_video_help_names_the_free_text_controls_the_filter_draws():
    """Help and the filter must not disagree about what a model offers.

    The curated knob table cannot know about a setting OpenRouter adds, so a control the
    chat UI draws would go unmentioned -- the inverse of the defect that made the image
    help read from the contract instead of a hand-written table.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )
    from open_webui_openrouter_pipe.integrations.video_help import render_video_help

    base = dict(VIDEO_BY_ID["google/veo-3.1-fast"])
    base["allowed_passthrough_parameters"] = list(
        base.get("allowed_passthrough_parameters") or []
    ) + ["contentModeration"]

    module = _load_filter_from_source(
        render_video_filter_source(model_id=base["id"], video_model=base),
        "help_matches_filter",
    )
    assert "VIDEO_CONTENTMODERATION" in module.Filter.UserValves.model_fields, (
        "the filter must draw a control for the published setting"
    )

    rendered = render_video_help(base["id"], base)
    assert "contentModeration" in rendered, (
        "help must name every control the filter draws, including the free-text ones"
    )


def test_a_case_variant_published_name_does_not_render_twice():
    """Two published names differing only in case produce one VIDEO_ field.

    Rendering both defines the attribute twice; pydantic keeps the last, so one control
    would write two provider parameters and the user would never see the second.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        _unhandled_params,
        build_video_filter_spec,
        render_video_filter_source,
    )

    model = {
        "id": "vendor/model",
        "name": "Model",
        "allowed_passthrough_parameters": ["contentModeration", "contentmoderation", "keyframes"],
    }
    offered = _unhandled_params(build_video_filter_spec("vendor/model", model))
    assert offered == ("contentModeration", "keyframes"), (
        f"the case-variant must be dropped, not rendered twice; got {offered}"
    )

    source = render_video_filter_source(model_id="vendor/model", video_model=model)
    assert source.count("VIDEO_CONTENTMODERATION:") == 1
    module = _load_filter_from_source(source, "video_case_variant")
    assert "VIDEO_KEYFRAMES" in module.Filter.UserValves.model_fields



# ============================================================================
# REGFIX: video fixes 1-7
# ============================================================================


@pytest.mark.parametrize(
    ("system_text", "user_text"),
    [
        ("HOUSE STYLE: always cel-shaded, teal background", "a red mug"),
        ("STUDIO RULE: hand-held camera, 35mm grain", "a blue kettle"),
    ],
)
def test_a_system_prompt_is_prepended_to_the_video_prompt(system_text, user_text):
    """Two distinct pairs, so a constant cannot pass, and the join pins the order.

    Open WebUI pops a Workspace model's system prompt and injects it via
    `add_or_update_system_message`; the video API has one free-text field, so dropping the
    system block loses the whole house style with no note and no log.
    """
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    composed = adapter._extract_prompt(
        {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        }
    )

    assert composed == f"{system_text}\n\n{user_text}"
    assert composed.index(system_text) < composed.index(user_text)


@pytest.mark.parametrize(
    ("system_text", "user_text"),
    [
        ("HOUSE STYLE: always cel-shaded", "a red mug"),
        ("STUDIO RULE: hand-held camera", "a blue kettle"),
    ],
)
@pytest.mark.asyncio
async def test_the_system_prompt_reaches_the_video_payload(system_text, user_text):
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    prompt = adapter._extract_prompt(
        {
            "messages": [
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ]
        }
    )
    payload = await adapter._build_payload(
        api_model_id="google/veo-3.1",
        prompt=prompt,
        video_meta={},
        video_model=VIDEO_BY_ID["google/veo-3.1"],
        frame_images=[],
        provider_options={},
    )

    assert payload["prompt"] == f"{system_text}\n\n{user_text}"


@pytest.mark.parametrize(
    "system_text",
    ["HOUSE STYLE: always cel-shaded", "STUDIO RULE: hand-held camera"],
)
def test_a_system_prompt_alone_is_not_a_request(system_text):
    """A house style is a modifier, not a request.

    Composing it into the prompt when the user typed nothing would disarm the
    "needs a prompt" guard, so pressing send on an empty box would spend money on the
    system prompt instead of erroring. Two distinct system strings, so a composer that
    returned a constant empty string for the wrong reason still has to satisfy the
    positive test above.
    """
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    assert adapter._extract_prompt({"messages": [{"role": "system", "content": system_text}]}) == ""
    assert (
        adapter._extract_prompt(
            {"messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": "   "},
            ]}
        ).strip()
        == ""
    )


@pytest.mark.parametrize(
    ("frame_file_id", "frame_bytes"),
    [("frame-A", b"\x89PNG\r\n\x1a\n" + b"A" * 64), ("frame-B", b"\x89PNG\r\n\x1a\n" + b"B" * 96)],
)
@pytest.mark.asyncio
async def test_an_empty_prompt_with_a_frame_image_still_submits(
    monkeypatch, frame_file_id, frame_bytes
):
    """Image-to-video: two distinct frames so a constant payload cannot pass.

    The submitted payload must carry the caller's own frame, and the prompt key must be
    present-but-empty -- `prompt` has no minLength in either OpenAPI copy, so "" is valid,
    but omitting the key is not.
    """
    submitted: list[dict[str, Any]] = []
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID["google/veo-3.1"]])
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence("")

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, payload):
            submitted.append(payload)
            raise VideoGenerationError("stop after submit")

    async def fake_get_file_by_id(file_id, _logger):
        return SimpleNamespace(id=file_id, meta={"content_type": "image/png"}, filename="f.png")

    async def fake_read_b64(*_args, **_kwargs):
        return base64.b64encode(frame_bytes).decode()

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id", fake_get_file_by_id
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: "image/png",
    )
    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", fake_read_b64)

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": ""}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={
            "chat_id": "chat-1",
            "message_id": "msg-1",
            "openrouter_pipe": {
                "video_generation": {
                    "params": {},
                    "frame_images": [
                        {"id": frame_file_id, "frame_type": "first_frame",
                         "content_type": "image/png", "name": "f.png"}
                    ],
                }
            },
        },
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id="google.veo-3.1",
        api_model_id="google/veo-3.1",
    )

    assert submitted, f"the request never reached submit: {result!r}"
    assert submitted[0]["prompt"] == ""
    assert submitted[0]["frame_images"][0]["image_url"]["url"].endswith(
        base64.b64encode(frame_bytes).decode()
    )
    assert pipe._video_user_active_counts == {}


@pytest.mark.parametrize("model_id", ["google/veo-3.1", "openai/sora-2-pro"])
@pytest.mark.asyncio
async def test_an_empty_prompt_with_nothing_attached_is_still_refused(monkeypatch, model_id):
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID[model_id]])
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence("")

    submitted: list[Any] = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, payload):
            submitted.append(payload)
            raise AssertionError("a request with nothing to generate from must not submit")

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )

    result = await adapter.generate(
        body={"messages": [
            {"role": "system", "content": "HOUSE STYLE: cel-shaded"},
            {"role": "user", "content": "   "},
        ]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={"chat_id": "chat-1", "message_id": "msg-1"},
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id=model_id.replace("/", "."),
        api_model_id=model_id,
    )

    assert not submitted, (
        "the request reached submit, so it was billed; 'failed' in the answer is satisfied "
        "by any failure and cannot tell a refusal from a blow-up at the transport"
    )
    assert "### Video generation failed" in result
    assert "generate from" in result, (
        f"the refusal must say what is missing rather than surfacing a transport error: {result!r}"
    )
    assert pipe._video_user_active_counts == {}
    assert pipe._video_user_active_jobs == {}


@pytest.mark.parametrize(
    ("model_id", "file_id", "content_type"),
    [
        ("google/veo-3.1", "aud-1", "audio/mpeg"),
        ("openai/sora-2-pro", "vid-9", "video/mp4"),
    ],
)
def test_user_attached_audio_and_video_reach_input_references(model_id, file_id, content_type):
    """Two families on two models, so neither a constant id nor a constant type can pass.

    Measured before the fix: video reached the payload on 1 of 22 models and audio on 2 of
    22; on the rest the file left `body["files"]` and was sent nowhere -- no note, no log,
    no `withheld` entry.
    """
    files = [_file_item(file_id, "asset.bin", content_type)]
    _body, metadata = _run_inlet_via_metadata(model_id, files)

    references = metadata["openrouter_pipe"]["video_generation"].get("input_references", [])

    assert [r["id"] for r in references] == [file_id]
    assert references[0]["content_type"] == content_type


@pytest.mark.parametrize(
    ("content_type", "payload_bytes", "expected_kind"),
    [
        ("audio/mpeg", b"ID3\x04audio-one", "audio_url"),
        ("video/mp4", b"\x00\x00\x00\x18ftypmp42vid", "video_url"),
    ],
)
@pytest.mark.asyncio
async def test_the_reference_kind_follows_the_media_family(
    monkeypatch, content_type, payload_bytes, expected_kind
):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    encoded = base64.b64encode(payload_bytes).decode()

    async def fake_get_file_by_id(file_id, _logger):
        return SimpleNamespace(id=file_id)

    async def fake_read_b64(*_args, **_kwargs):
        return encoded

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id", fake_get_file_by_id
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: content_type,
    )
    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", fake_read_b64)

    refs = await adapter._encode_input_references(
        {"input_references": [{"id": "ref-1", "content_type": content_type}]},
        pipe.valves,
    )

    assert len(refs) == 1
    assert refs[0]["type"] == expected_kind
    assert refs[0][expected_kind]["url"] == f"data:{content_type};base64,{encoded}"
    assert "image_url" not in refs[0]


@pytest.mark.parametrize(
    ("model_id", "file_id", "content_type", "payload_bytes", "expected_kind"),
    [
        ("google/veo-3.1", "aud-7", "audio/mpeg", b"ID3\x04score", "audio_url"),
        ("openai/sora-2-pro", "vid-3", "video/mp4", b"\x00\x00\x00\x18ftypmp42clip", "video_url"),
    ],
)
@pytest.mark.asyncio
async def test_a_chat_attachment_reaches_the_wire_as_a_typed_reference(
    monkeypatch, model_id, file_id, content_type, payload_bytes, expected_kind
):
    """The whole pipeline, because neither end alone is the property.

    The plan's own stated predicate is DELIVERY: run `inlet`, run the encoder, run
    `_build_payload`, and assert the discriminator is on the request. A renderer that
    writes the metadata and a payload builder that drops it are each green in isolation.
    """
    files = [_file_item(file_id, "asset.bin", content_type)]
    _body, metadata = _run_inlet_via_metadata(model_id, files)
    video_meta = metadata["openrouter_pipe"]["video_generation"]

    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    encoded = base64.b64encode(payload_bytes).decode()

    async def fake_get_file_by_id(fid, _logger):
        return SimpleNamespace(id=fid)

    async def fake_read_b64(*_args, **_kwargs):
        return encoded

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id", fake_get_file_by_id
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: content_type,
    )
    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", fake_read_b64)

    references = await adapter._encode_input_references(video_meta, pipe.valves)
    payload = await adapter._build_payload(
        api_model_id=model_id,
        prompt="a red mug",
        video_meta=video_meta,
        video_model=VIDEO_BY_ID[model_id],
        frame_images=[],
        provider_options={},
        input_references=references,
    )

    delivered = payload["input_references"]
    assert [r["type"] for r in delivered] == [expected_kind]
    assert delivered[0][expected_kind]["url"] == f"data:{content_type};base64,{encoded}"


@pytest.mark.parametrize(
    ("model_id", "prompt_text"),
    [("google/veo-3.1", "a red mug spinning"), ("openai/sora-2-pro", "a blue kettle boiling")],
)
def test_a_file_free_request_keeps_its_body_untouched(model_id, prompt_text):
    """Two models and two prompts, so a constant body cannot pass.

    Before the fix the filter wrote `body["files"] = []` and `__metadata__["files"] = []`
    on every request, attachments or not -- a key Open WebUI then persists onto the chat
    row.
    """
    source = render_video_filter_source(model_id=model_id, video_model=VIDEO_BY_ID[model_id])
    module = _load_filter_from_source(
        source,
        f"video_filter_nofiles_{model_id.replace('/', '_').replace('.', '_').replace('-', '_')}",
    )
    body: dict[str, Any] = {"messages": [{"role": "user", "content": prompt_text}]}
    metadata: dict[str, Any] = {}

    module.Filter().inlet(
        body, __metadata__=metadata, __user__={"valves": module.Filter.UserValves()}
    )

    assert "files" not in body
    assert "files" not in metadata
    assert body["messages"][0]["content"] == prompt_text


@pytest.mark.parametrize(
    ("model_arg", "expected"),
    [
        ({"info": {"meta": {"capabilities": {"file_context": False}}}}, True),
        ({"info": {"meta": {"capabilities": {"file_context": True}}}}, False),
        ({"info": {"meta": {"capabilities": {}}}}, False),
        ({"info": {"meta": {}}}, False),
        ({}, False),
        (None, False),
        ({"info": {"meta": {"capabilities": {"file_context": "false"}}}}, False),
        ({"info": {"meta": {"capabilities": {"file_context": 0}}}}, False),
        ({"info": {"meta": {"capabilities": {"file_context": None}}}}, False),
    ],
)
def test_the_filter_only_hands_files_back_when_owui_will_not_rag_them(model_arg, expected):
    """Nine shapes of `__model__`, so neither `return True` nor `return False` passes.

    Open WebUI reads this capability with a default of True, and on True it answers an
    attachment-bearing request with a `generate_queries` LLM round-trip plus RAG injection
    -- into a video prompt. Three rows separate identity from truthiness: the string
    "false" is truthy, `0` is falsy, `None` is falsy, and none of the three is the
    published boolean `False`. `not caps.get("file_context", True)` accepts two of them.
    """
    source = render_video_filter_source(
        model_id="google/veo-3.1", video_model=VIDEO_BY_ID["google/veo-3.1"]
    )
    module = _load_filter_from_source(source, "video_filter_filectx_probe")

    assert module.Filter._owui_skips_file_context(model_arg) is expected


@pytest.mark.parametrize(
    ("model_id", "file_id", "content_type"),
    [
        ("google/veo-3.1", "img-ref", "image/jpeg"),
        ("openai/sora-2-pro", "vid-ref", "video/mp4"),
    ],
)
@pytest.mark.parametrize("file_context", [False, True])
def test_a_consumed_reference_is_handed_back_only_when_owui_will_not_rag_it(
    model_id, file_id, content_type, file_context
):
    """`file_context: False` returns the attachment to the list Open WebUI persists.

    Today the reference vanishes from the conversation the moment it is consumed, because
    `body["files"]` is the list Open WebUI writes onto the chat row. Two models, two media
    families and both capability values, so neither `kept = retained` nor
    `kept = retained + claimed` unconditionally satisfies the table.
    """
    source = render_video_filter_source(model_id=model_id, video_model=VIDEO_BY_ID[model_id])
    module = _load_filter_from_source(
        source,
        f"video_filter_handback_{model_id.replace('/', '_').replace('.', '_').replace('-', '_')}",
    )
    body: dict[str, Any] = {"files": None}
    metadata: dict[str, Any] = {"user_message": {"files": [_file_item(file_id, "a.bin", content_type)]}}

    module.Filter().inlet(
        body,
        __metadata__=metadata,
        __user__={"valves": module.Filter.UserValves()},
        __model__={"info": {"meta": {"capabilities": {"file_context": file_context}}}},
    )

    handed_back = [item["id"] for item in body["files"]]
    assert handed_back == ([file_id] if file_context is False else [])
    assert metadata["files"] == body["files"]


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ({"video_generation"}, False),
        ({"image_output"}, False),
        ({"vision"}, None),
    ],
)
@pytest.mark.parametrize("builtin_valve", [True, False])
def test_file_context_is_unticked_regardless_of_the_builtin_tools_valve(
    features, expected, builtin_valve
):
    """The RAG round-trip must not be re-armed by unticking a valve about tools."""
    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.models.catalog_manager import media_capability_defaults

    valves = Valves()
    valves.UPDATE_MODEL_CAPABILITIES = True
    valves.DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS = builtin_valve

    defaults = media_capability_defaults(
        valves,
        {
            "image_output": "image_output" in features,
            "video_generation": "video_generation" in features,
            "vision": "vision" in features,
        },
    )

    assert defaults.get("file_context") is expected
    assert (defaults.get("builtin_tools") is False) is (expected is False and builtin_valve)


@pytest.mark.parametrize(
    ("field", "valve", "params_key", "supplied", "expected"),
    [
        ("seed", "VIDEO_SEED", "seed", 7, 7),
        ("seed", "VIDEO_SEED", "seed", 4321, 4321),
        ("generate_audio", "VIDEO_GENERATE_AUDIO", "generate_audio", "on", True),
        ("generate_audio", "VIDEO_GENERATE_AUDIO", "generate_audio", "off", False),
    ],
)
@pytest.mark.asyncio
async def test_an_undeclared_boolean_renders_a_control_and_reaches_the_payload(
    field, valve, params_key, supplied, expected
):
    """A published null is 'not declared', not 'declared false'.

    Two values per field, so neither a hardcoded payload nor a hardcoded valve default can
    pass. Both modules are exercised in one pipeline: the renderer must emit the control
    and the adapter must promote the name to a top-level field.
    """
    model = dict(VIDEO_BY_ID["minimax/hailuo-2.3"])
    model[field] = None

    source = render_video_filter_source(model_id="minimax/hailuo-2.3", video_model=model)
    assert valve in source

    module = _load_filter_from_source(source, f"video_tristate_{field}_{supplied!r}")
    body: dict[str, Any] = {"files": []}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        body,
        __metadata__=metadata,
        __user__={"valves": module.Filter.UserValves(**{valve: supplied})},
    )

    video_meta = metadata["openrouter_pipe"]["video_generation"]
    assert video_meta["params"][params_key] == expected

    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    payload = await adapter._build_payload(
        api_model_id="minimax/hailuo-2.3",
        prompt="a red mug",
        video_meta=video_meta,
        video_model=model,
        frame_images=[],
        provider_options={},
    )

    assert payload[params_key] == expected


@pytest.mark.parametrize("field", ["seed", "generate_audio"])
def test_an_explicit_false_still_hides_the_control(field):
    model = dict(VIDEO_BY_ID["minimax/hailuo-2.3"])
    model[field] = False
    valve = "VIDEO_SEED" if field == "seed" else "VIDEO_GENERATE_AUDIO"

    source = render_video_filter_source(model_id="minimax/hailuo-2.3", video_model=model)

    assert valve not in source


@pytest.mark.parametrize("field", ["seed", "generate_audio"])
def test_a_capability_the_catalog_never_mentions_renders_no_control(field):
    """Absent is not null. `dict.get` answers None for both and they decide oppositely."""
    model = {k: v for k, v in VIDEO_BY_ID["minimax/hailuo-2.3"].items() if k != field}
    valve = "VIDEO_SEED" if field == "seed" else "VIDEO_GENERATE_AUDIO"

    source = render_video_filter_source(model_id="minimax/hailuo-2.3", video_model=model)

    assert valve not in source
    top_level, _ = VideoGenerationAdapter._split_allowed_parameters(
        object.__new__(VideoGenerationAdapter), model
    )
    assert field not in top_level


@pytest.mark.parametrize(
    "list_field",
    ["supported_sizes", "supported_frame_images", "supported_durations", "supported_resolutions"],
)
def test_a_null_list_field_still_renders_no_control(list_field):
    """`null` on a list means no published values; a control there would invent them."""
    model = dict(VIDEO_BY_ID["alibaba/wan-2.7"])
    model[list_field] = None
    valve = {
        "supported_sizes": "VIDEO_SIZE",
        "supported_frame_images": "VIDEO_FRAME_MODE",
        "supported_durations": "VIDEO_DURATION",
        "supported_resolutions": "VIDEO_RESOLUTION",
    }[list_field]

    source = render_video_filter_source(model_id="alibaba/wan-2.7", video_model=model)

    assert valve not in source


@pytest.mark.parametrize("clip_count", [1, 3])
@pytest.mark.asyncio
async def test_every_output_of_a_multi_clip_job_is_downloaded_and_rendered(
    monkeypatch, clip_count
):
    """Two counts, so a loop that always fetches one and a loop that always fetches three
    both fail. Before the fix the content URL carried no `index`, so a job with three
    outputs delivered clip #1 three times.

    The double SUBCLASSES the real client and stubs only `status` and `bearer_header`, so
    `content_url` and `output_count` are the production implementations and a mutation of
    either is visible here. A double carrying its own copies mutates code the test never
    runs.
    """
    requested: list[str] = []
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence(
        "[openrouter:v1:videojob:job-multi]: #\n\nVideo generation is running..."
    )

    class FakeClient(OpenRouterVideoClient):
        async def status(self, job_id, polling_url=None):
            assert job_id == "job-multi"
            return {
                "status": "completed",
                "generation_id": "gen-xyz789",
                "unsigned_urls": [f"https://storage.test/{i}.mp4" for i in range(clip_count)],
            }

        def bearer_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer test"}

    async def fake_streaming_download(url: str, dest_path, **_kwargs):
        requested.append(url)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4_BYTES)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url,
                "size_bytes": len(MP4_BYTES)}

    uploads: list[str] = []

    async def fake_upload_from_path(*_args, **kwargs):
        uploads.append(str(kwargs["source_path"]))
        return f"file-{len(uploads)}"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_a, **_k: _FakeSession([]))
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url_streaming", fake_streaming_download
    )
    monkeypatch.setattr(
        pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path
    )

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "make a video"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={"chat_id": "chat-1", "message_id": "msg-1"},
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id="openai.sora-2-pro",
        api_model_id="openai/sora-2-pro",
    )

    assert len(requested) == clip_count
    assert len(set(requested)) == clip_count
    assert result.count("<video>") == clip_count
    for i in range(1, clip_count + 1):
        assert f"/api/v1/files/file-{i}/content" in result


@pytest.mark.parametrize(
    ("size", "ratio", "ratio_survives"),
    [
        ("1280x720", "16:9", True),
        ("720x1280", "9:16", True),
        ("1280x720", "9:16", False),
        ("720x1280", "16:9", False),
        ("854x480", "16:9", True),
        ("1120x480", "21:9", True),
        ("960x720", "4:3", True),
        ("960x720", "3:2", False),
    ],
)
@pytest.mark.asyncio
async def test_an_exact_size_keeps_only_an_aspect_ratio_that_agrees_with_it(
    size, ratio, ratio_survives
):
    """Eight rows spanning four decades of error, so no threshold except a correct one
    passes them all. Relative errors:

        1280x720 / 16:9   0.00000   keep
        854x480  / 16:9   0.00078   keep   (widest divergence in the 15-row fixture)
        1120x480 / 21:9   0.00000   keep   (a 21:9 cell that gcd-reduces to 7:3)
        960x720  / 4:3    0.00000   keep
        960x720  / 3:2    0.11111   drop   (the near miss -- kills a slack tolerance)
        720x1280 / 16:9   0.68359   drop
        1280x720 / 9:16   2.16049   drop

    The 0.111 row is the one that matters: it sits above the 0.025 tolerance and below the
    0.125 gap between the two closest published ratios, so a tolerance widened to
    "obviously safe" values like 0.2 lets a genuine 400 through.
    """
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    model = dict(VIDEO_BY_ID["bytedance/seedance-2.0"])
    withheld: list[tuple[str, str]] = []

    payload = await adapter._build_payload(
        api_model_id="bytedance/seedance-2.0",
        prompt="a red mug",
        video_meta={"params": {"size": size, "aspect_ratio": ratio}},
        video_model=model,
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    assert payload["size"] == size
    assert ("aspect_ratio" in payload) is ratio_survives
    assert (payload.get("aspect_ratio") == ratio) is ratio_survives
    assert any(name == "aspect_ratio" for name, _ in withheld) is not ratio_survives


@pytest.mark.parametrize(
    ("size", "resolution"),
    [("1280x720", "1080p"), ("1920x1080", "720p")],
)
@pytest.mark.asyncio
async def test_an_exact_size_supersedes_an_ambiguous_resolution_tier(size, resolution):
    """`alibaba/wan-2.7` publishes two tiers, so a size fixes one of them and the contract
    does not say which. The tier cannot be shown to agree, so it is dropped."""
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    withheld: list[tuple[str, str]] = []

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="a red mug",
        video_meta={"params": {"size": size, "resolution": resolution}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    assert payload["size"] == size
    assert "resolution" not in payload
    assert [name for name, _ in withheld] == ["resolution"]


@pytest.mark.parametrize(
    ("published_tiers", "declared", "survives"),
    [
        (["1080p"], "1080p", True),
        (["720p", "1080p"], "1080p", False),
        (["720p"], "720p", True),
        (["480p", "720p"], "720p", False),
        (["720p"], "1080p", False),
        (["1080p"], "720p", False),
    ],
)
@pytest.mark.asyncio
async def test_a_single_published_tier_cannot_contradict_an_exact_size(
    published_tiers, declared, survives
):
    """One size, opposite outcomes, decided only by what the contract publishes.

    A model publishing one tier puts every size it publishes in that tier, so the pair is
    the same statement twice and dropping it would attach a note about a rejection that
    cannot happen. This is the row an unconditional pop reddens.

    The last two rows are the other half of the same property: one published tier and a
    DIFFERENT declared one is a provable mismatch, so "there is only one tier" alone is
    not the test -- the declared tier has to be that tier.
    """
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    withheld: list[tuple[str, str]] = []

    payload = await adapter._build_payload(
        api_model_id="vendor/model",
        prompt="a red mug",
        video_meta={"params": {"size": "1920x1080", "resolution": declared}},
        video_model={
            "id": "vendor/model",
            "supported_resolutions": published_tiers,
            "supported_sizes": ["1920x1080"],
        },
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    assert payload["size"] == "1920x1080"
    assert ("resolution" in payload) is survives
    assert (payload.get("resolution") == declared) is survives
    assert any(name == "resolution" for name, _ in withheld) is not survives


@pytest.mark.parametrize(
    ("resolution", "ratio"),
    [("720p", "16:9"), ("1080p", "9:16")],
)
@pytest.mark.asyncio
async def test_resolution_and_aspect_ratio_still_combine_when_no_size_is_set(resolution, ratio):
    """The rule is not a three-way exclusion: without `size` both survive untouched."""
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())
    withheld: list[tuple[str, str]] = []

    payload = await adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="a red mug",
        video_meta={"params": {"resolution": resolution, "aspect_ratio": ratio}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        withheld=withheld,
    )

    assert payload["resolution"] == resolution
    assert payload["aspect_ratio"] == ratio
    assert withheld == []


def test_every_published_size_agrees_with_a_published_aspect_ratio():
    """Fleet guard: the tolerance must admit every cell OpenRouter actually publishes.

    Runs over the catalogue rather than a sample, so a future model whose sizes sit
    further from their ratios than the tolerance allows fails here instead of silently
    losing the user's ratio at request time.
    """
    for model_id, model in VIDEO_BY_ID.items():
        sizes = model.get("supported_sizes")
        ratios = model.get("supported_aspect_ratios")
        if not (isinstance(sizes, list) and isinstance(ratios, list) and ratios):
            continue
        for size in sizes:
            pixels = VideoGenerationAdapter._parse_pixel_size(size)
            assert pixels is not None, f"{model_id} publishes an unparsable size {size!r}"
            width, height = pixels
            values = [
                value
                for r in ratios
                if (value := VideoGenerationAdapter._aspect_ratio_value(r)) is not None
            ]
            assert values, f"{model_id} publishes unparsable aspect ratios {ratios!r}"
            best = min(abs(width / height - value) / value for value in values)
            assert best <= 0.025, (
                f"{model_id} size {size} is {best:.5f} from its nearest published ratio; "
                "the consistency rule would drop a ratio the model accepts"
            )


@pytest.mark.parametrize("model_id", ["bytedance/seedance\u00ad2", "runway/gen'4"])
@pytest.mark.asyncio
async def test_a_video_filter_is_re_identified_whatever_its_id_needs_escaping(model_id):
    """The matcher and the renderer must agree on the literal, character for character.

    Two ids whose reprs escape differently: one the old hand-built double-quoted form
    happened to catch and one it did not, so a matcher that always returns True fails the
    negative half below.
    """
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    captured: dict = {}
    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    manager._ensure_filter_installed = AsyncMock(
        side_effect=lambda **kw: (captured.update(kw), kw["preferred_id"])[1]
    )

    await manager._ensure_single_video_gen_filter_function_id(
        model_id=model_id, video_model={"id": model_id, "name": "N"}
    )

    matches = captured["matches_candidate"]
    own = captured["desired_source"]
    other = render_video_filter_source(
        model_id="other/model",
        video_model={"id": "other/model", "name": "O"},
        admin_valves=None,
    )

    assert matches(own), (
        f"the filter this run just rendered for {model_id!r} does not re-identify itself, "
        "so every refresh installs another copy under a _N suffix until the 50 cap"
    )
    assert not matches(other), (
        "the matcher accepts a different model's filter, so it would overwrite it"
    )
