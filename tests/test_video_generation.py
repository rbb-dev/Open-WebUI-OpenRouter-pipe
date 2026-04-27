from __future__ import annotations

import asyncio
import time
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import EncryptedStr, Pipe
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


def test_video_registry_marks_models_non_zdr_and_filterable():
    OpenRouterModelRegistry._zdr_model_ids = {"openai.sora-2-pro"}
    OpenRouterModelRegistry.register_video_models([VIDEO_BY_ID["openai/sora-2-pro"]])

    assert ModelFamily.supports("video_generation", "openai.sora-2-pro") is True
    assert OpenRouterModelRegistry.is_zdr_capable("openai.sora-2-pro") is False

    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    filtered = pipe._apply_model_filters(OpenRouterModelRegistry.list_models(), pipe.valves)
    assert filtered == []


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
        file_id="file-xyz",
        elapsed=12.3,
        usage={"cost": 0.4},
    )
    assert content.endswith("\n"), "success content must end with \\n so concatenation cannot smash markers into inline text"
    doubled = content + content
    assert "\n[openrouter:v1:videojob:job-abc]: #" in doubled
    assert "*Generated in 12.3s · $0.4000*[openrouter" not in doubled, "marker must NOT be glued onto the footer line"


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

        async def status(self, _job_id):
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id):
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
    monkeypatch.setattr(pipe._multimodal_handler, "_upload_to_owui_storage_from_path", fake_upload_from_path)

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


def test_sora_drops_all_attachments():
    """Sora 2 Pro is text-only (no frames, no media refs) — any chat attachment must be dropped
    from retained so RAG never sees it. Required for the chromadb-bypass invariant."""
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


def test_payload_routes_video_attachment_url_to_video_field_for_wan_2_7():
    """`_build_payload` routes a single video data URL to params.video for Wan 2.7."""
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:video/mp4;base64,AAAA"
    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=[data_url],
    )
    assert payload["video"] == data_url


def test_payload_routes_multiple_video_urls_to_videos_array_for_wan_2_7():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    urls = ["data:video/mp4;base64,AAAA", "data:video/mp4;base64,BBBB"]
    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        video_attachment_urls=urls,
    )
    assert payload["videos"] == [{"url": urls[0]}, {"url": urls[1]}]


def test_payload_routes_audio_attachment_for_wan_2_6():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:audio/mpeg;base64,AAAA"
    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.6",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.6"],
        frame_images=[],
        provider_options={},
        audio_attachment_url=data_url,
    )
    assert payload["audio"] == data_url


def test_payload_routes_audio_attachment_for_wan_2_7():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    data_url = "data:audio/mpeg;base64,AAAA"
    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
        audio_attachment_url=data_url,
    )
    assert payload["audio"] == data_url


def test_data_urls_bypass_ssrf_validator():
    """Data URLs are inline content with no network fetch — must be allowed through
    `_validate_passthrough_urls` without DNS resolution."""
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="x",
        video_meta={"params": {"audio": "data:audio/mpeg;base64,AAAA"}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
    )
    assert payload["audio"].startswith("data:audio/")


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

        async def status(self, job_id):
            assert job_id == "job-resume"
            return {"status": "completed", "usage": {"cost": "0.25"}}

        # T0-A refactor: download/upload no longer go through the client. The adapter calls
        # the canonical _download_remote_url_streaming + _upload_to_owui_storage_from_path on
        # the multimodal handler. content_url/bearer_header just return the URL/headers the
        # canonical helper consumes.
        def content_url(self, job_id: str) -> str:
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
    monkeypatch.setattr(pipe._multimodal_handler, "_upload_to_owui_storage_from_path", fake_upload_from_path)

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
    # Multi-line <video>...</video> with URL on its own line is the canonical format that
    # marked tokenizes as a single CommonMark "type 7" html block (verified empirically).
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

        async def status(self, _job_id):
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
    # Failure content reaches OWUI via chat:message:delta (which feeds the stream accumulator
    # so the end-of-stream finalizer writes it to DB) — NOT via direct DB upsert.
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


def test_provider_options_normalise_to_parameters_wrapper():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    wrapped = adapter._normalise_provider_options(
        {
            "google": {"negativePrompt": "blur"},
            "fal": {"parameters": {"motion": "slow"}},
        }
    )

    assert wrapped["google"] == {"parameters": {"negativePrompt": "blur"}}
    assert wrapped["fal"] == {"parameters": {"motion": "slow"}}


def test_video_payload_drops_unsupported_passthrough_and_preserves_provider_casing():
    adapter = VideoGenerationAdapter(pipe=Pipe(), logger=_test_logger())

    unsupported = adapter._build_payload(
        api_model_id="unknown/video",
        prompt="make a video",
        video_meta={"params": {"surprise": "value"}},
        video_model={"allowed_passthrough_parameters": []},
        frame_images=[],
        provider_options={},
    )
    veo = adapter._build_payload(
        api_model_id="google/veo-3.1",
        prompt="make a video",
        video_meta={"params": {"aspect_ratio": "16:9"}},
        video_model=VIDEO_BY_ID["google/veo-3.1"],
        frame_images=[],
        provider_options={},
    )

    assert "surprise" not in unsupported
    assert veo["aspectRatio"] == "16:9"
    assert "aspect_ratio" not in veo


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

        async def status(self, _job_id):
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


def test_video_help_renders_pricing_live_from_pricing_skus():
    veo_help = render_video_help("google/veo-3.1-fast", VIDEO_BY_ID["google/veo-3.1-fast"])
    assert "**Cost** (live from OpenRouter catalog `pricing_skus`)" in veo_help
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
    from open_webui_openrouter_pipe.integrations.video_help import _format_sku_unit
    assert _format_sku_unit("duration_seconds") == "per second"
    assert _format_sku_unit("duration_seconds_with_audio") == "per second (with audio)"
    assert _format_sku_unit("duration_seconds_with_audio_4k") == "per second (with audio, 4K)"
    assert _format_sku_unit("video_tokens") == "per video token"
    assert _format_sku_unit("video_tokens_without_audio") == "per video token (without audio)"
    assert _format_sku_unit("text_to_video_duration_seconds_720p") == "per second (text-to-video, 720p)"
    assert _format_sku_unit("image_to_video_duration_seconds_1080p") == "per second (image-to-video, 1080p)"


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
    assert hailuo.seed_capable is False
    assert hailuo.audio_capable is False

    sora = build_video_filter_spec("openai/sora-2-pro", VIDEO_BY_ID["openai/sora-2-pro"])
    assert sora.seed_capable is False
    assert sora.audio_capable is True


@pytest.mark.parametrize("model_id", list(_VIDEO_MODEL_IDS_FOR_AUDIT := [
    "google/veo-3.1-fast",
    "google/veo-3.1-lite",
    "google/veo-3.1",
    "kwaivgi/kling-video-o1",
    "minimax/hailuo-2.3",
    "alibaba/wan-2.7",
    "bytedance/seedance-2.0-fast",
    "bytedance/seedance-2.0",
    "alibaba/wan-2.6",
    "bytedance/seedance-1-5-pro",
    "openai/sora-2-pro",
]))
def test_video_filter_renderer_per_model_audit(model_id: str):
    model = VIDEO_BY_ID[model_id]
    source = render_video_filter_source(model_id=model_id, video_model=model)

    if model.get("seed") is True:
        assert "VIDEO_SEED" in source
    else:
        assert "VIDEO_SEED" not in source

    if model.get("generate_audio") is True:
        assert "VIDEO_GENERATE_AUDIO" in source
    else:
        assert "VIDEO_GENERATE_AUDIO" not in source

    allowed = set(model.get("allowed_passthrough_parameters") or [])
    valve_for = {
        "personGeneration": "VIDEO_PERSON_GENERATION",
        "conditioningScale": "VIDEO_CONDITIONING_SCALE",
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


def test_payload_includes_seed_and_generate_audio_for_capable_models(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url_blocking", lambda url: True)

    payload = adapter._build_payload(
        api_model_id="google/veo-3.1",
        prompt="a sunset over mountains",
        video_meta={"params": {"seed": 7, "generate_audio": False}},
        video_model=VIDEO_BY_ID["google/veo-3.1"],
        frame_images=[],
        provider_options={},
    )

    assert payload["seed"] == 7
    assert payload["generate_audio"] is False


def test_payload_drops_seed_for_seed_incapable_models(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url_blocking", lambda url: True)

    payload = adapter._build_payload(
        api_model_id="kwaivgi/kling-video-o1",
        prompt="a quiet room",
        video_meta={"params": {"seed": 7}},
        video_model=VIDEO_BY_ID["kwaivgi/kling-video-o1"],
        frame_images=[],
        provider_options={},
    )

    assert "seed" not in payload


def test_generate_audio_does_not_clobber_audio_url_on_wan(monkeypatch):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())
    monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url_blocking", lambda url: True)

    payload = adapter._build_payload(
        api_model_id="alibaba/wan-2.7",
        prompt="a campfire",
        video_meta={"params": {"audio": "https://example.com/voice.mp3", "generate_audio": True}},
        video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
        frame_images=[],
        provider_options={},
    )

    assert payload["audio"] == "https://example.com/voice.mp3"
    assert payload["generate_audio"] is True


def test_unsafe_url_in_passthrough_raises():
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=_test_logger())

    with pytest.raises(VideoGenerationError, match="Refusing to forward unsafe URL"):
        adapter._build_payload(
            api_model_id="alibaba/wan-2.7",
            prompt="a campfire",
            video_meta={"params": {"audio": "file:///etc/passwd"}},
            video_model=VIDEO_BY_ID["alibaba/wan-2.7"],
            frame_images=[],
            provider_options={},
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
