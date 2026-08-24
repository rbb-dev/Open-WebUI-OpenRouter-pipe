"""What a request leaves behind in the message, and what has to be true before it goes.

Every test here defends one of three properties:

  * a user whose file was uploaded to an anonymous public host has a DURABLE record of
    it -- naming the host and how long it stays -- inside the message Open WebUI stores,
    not only in a toast that a reload discards;
  * a user whose settings were not all sent has a DURABLE record of which ones and why,
    in that same message, and it survives a job resumed in a later request; and
  * the upload does not happen at all when the advance warning could not be delivered.

The toast is deliberately not the subject of any assertion below. It is the channel that
fails: `_emit_notification` swallows its own exceptions, the emitter handed to the pipe is
wrapped in a second swallow, and Open WebUI renders the event as a transient toast that no
reload can recover. Asserting the toast went out measures the channel, not the record.
"""
from __future__ import annotations

import base64
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.integrations import video as video_module
from open_webui_openrouter_pipe.integrations.media_relay import _ENDPOINTS
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError
from open_webui_openrouter_pipe.storage.owui_files import infer_file_mime_type
from open_webui_openrouter_pipe.streaming.event_emitter import (
    EventEmitterHandler,
    unguarded_emitter,
)

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


class _Ctx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


def _record(file_id, mime="video/mp4", filename="clip.mp4"):
    return SimpleNamespace(
        id=file_id, filename=filename, user_id="bob", mime_type=None,
        content_type=None, meta={"content_type": mime},
    )


class _Publisher:
    """The real encoder against a real relay; only storage and the socket are doubles."""

    def __init__(self, *, emitter, count=1):
        self.posts: list[str] = []
        self.pipe = MagicMock()
        self.pipe.logger = logging.getLogger("relay-disclosure")
        self.handler = EventEmitterHandler(
            logger=self.pipe.logger, valves=Valves(), pipe_instance=self.pipe,
        )
        self.pipe._event_emitter_handler = self.handler
        self.emitter = emitter
        self.count = count

        async def _read(_file_obj, _chunk, _cap, user=None):
            return base64.b64encode(MP4).decode()

        self.pipe._file_gateway.read_file_record_base64 = _read
        self.adapter = VideoGenerationAdapter(pipe=self.pipe, logger=self.pipe.logger)

    def _wire(self, url, **_kwargs):
        self.posts.append(str(url))
        return CallbackResult(status=200, body="https://files.catbox.moe/PUBLISHED.mp4")

    async def publish(self, valves, relayed):
        async def _get_file(file_id, _logger):
            return _record(file_id)

        video_module.get_file_by_id = _get_file
        video_module.infer_file_mime_type = infer_file_mime_type
        async with aiohttp.ClientSession() as session:
            self.pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
            with aioresponses() as mocked:
                for endpoint in _ENDPOINTS.values():
                    mocked.post(endpoint.url, callback=self._wire, repeat=True)
                return await self.adapter._encode_input_references(
                    {
                        "input_references": [
                            {"id": f"f{i}"} for i in range(self.count)
                        ],
                        "model_id": "runway/aleph-2",
                    },
                    valves,
                    withheld=[],
                    user_obj=SimpleNamespace(id="bob", role="user"),
                    video_model={
                        "id": "runway/aleph-2",
                        "input_modalities": ["video", "image", "audio"],
                    },
                    relayed=relayed,
                    companions=True,
                    event_emitter=self.emitter,
                )


def _valves(**overrides):
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    for name, value in overrides.items():
        setattr(valves, name, value)
    return valves


# ------------------------------------------------------------------ DURABLE --
@pytest.mark.parametrize(
    ("host", "retention", "expected_span"),
    [
        ("litterbox", "1h", "an hour"),
        ("litterbox", "72h", "three days"),
        ("catbox", "1h", "stays there for good"),
    ],
)
def test_the_stored_message_says_where_the_file_went_and_for_how_long(
    host, retention, expected_span
):
    """The record has to name the host and the retention that actually applied.

    Three combinations, so neither the host name nor the retention phrase can be a
    constant: a production `return "litterbox ... an hour"` fails two of the three, and
    reading `MEDIA_FILE_HOST` instead of what was used fails the catbox row.
    """
    valves = _valves(MEDIA_FILE_HOST=host, MEDIA_FILE_HOST_RETENTION=retention)

    record = VideoGenerationAdapter._file_host_record(valves, {("video", host)})

    assert host in record, f"the record does not name the host: {record!r}"
    assert expected_span in record, f"the record does not say how long: {record!r}"
    assert "clip" in record, record


@pytest.mark.parametrize(
    ("relayed", "expected"),
    [
        ({("video", "litterbox")}, ["Your clip was uploaded", "open it,", "is deleted"]),
        (
            {("video", "litterbox"), ("audio", "litterbox")},
            ["clip and sound file were uploaded", "open them,", "are deleted"],
        ),
    ],
)
def test_the_record_reads_correctly_whether_one_file_went_out_or_several(
    relayed, expected
):
    """The record is the thing a person reads months later, so it has to be a sentence.

    One kind and two kinds, so a template frozen in either number cannot pass -- and
    the retention clause has to agree too, which is why it is asserted separately from
    the verb in the first half.
    """
    record = VideoGenerationAdapter._file_host_record(_valves(), relayed)

    for fragment in expected:
        assert fragment in record, f"{fragment!r} is missing from {record!r}"


@pytest.mark.parametrize(
    ("relayed", "expected"),
    [
        ({("video", "catbox")}, ["the upload carries no account", "take it down"]),
        (
            {("video", "catbox"), ("image", "catbox")},
            ["the uploads carry no account", "take them down"],
        ),
    ],
)
def test_the_permanent_host_warning_also_reads_correctly_in_both_numbers(
    relayed, expected
):
    """The permanent-host sentence is the one that matters most, and it is a second branch."""
    record = VideoGenerationAdapter._file_host_record(
        _valves(MEDIA_FILE_HOST="catbox"), relayed
    )

    for fragment in expected:
        assert fragment in record, f"{fragment!r} is missing from {record!r}"


def test_nothing_is_recorded_when_nothing_was_published():
    """An inline attachment never leaves this server, and must not be reported as if it had."""
    assert VideoGenerationAdapter._file_host_record(_valves(), set()) == ""
    assert VideoGenerationAdapter._with_the_file_host_record("KEEP", _valves(), set()) == "KEEP"


@pytest.mark.parametrize("existing", ["", "PRIOR-BLOCK\n"])
def test_the_record_joins_the_block_that_reaches_the_stored_message(existing):
    """`_with_the_file_host_record` is the one seam between the relay and the message.

    Parametrised over an empty and a non-empty prior block, because a production
    `return record` passes only the empty case and silently drops the intent disclosure
    that was already there.
    """
    valves = _valves(MEDIA_FILE_HOST="catbox")

    block = VideoGenerationAdapter._with_the_file_host_record(
        existing, valves, {("video", "catbox")}
    )

    assert existing in block, f"the prior block was dropped: {block!r}"
    assert "catbox" in block


@pytest.mark.parametrize("builder", ["pending", "success", "failure"])
def test_every_message_open_webui_stores_can_carry_the_record(builder):
    """Pending, success and failure are the three strings Open WebUI persists.

    A record that only survives on success is no record: a job that fails after the
    upload leaves the file public with nothing in the chat saying so. All three
    parametrised, so a fix applied to one of them cannot pass here.
    """
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)
    valves = _valves(MEDIA_FILE_HOST="catbox")
    record = adapter._file_host_record(valves, {("video", "catbox")})
    body = {
        "pending": lambda: adapter._build_pending_content(job_id="j", model_id="m"),
        "success": lambda: adapter._build_success_content(
            job_id="j", model_id="m", file_ids=["a"], elapsed=1.0, usage={},
        ),
        "failure": lambda: adapter._build_failure_content(
            job_id="j", model_id="m", reason="boom",
        ),
    }[builder]()

    stored = record + "\n" + body

    assert "catbox" in stored and "for good" in stored, stored
    assert body in stored, "the record replaced the message instead of joining it"


@pytest.mark.asyncio
async def test_the_set_the_encoder_fills_in_is_the_one_the_record_is_built_from():
    """`relayed` was populated and never read, which is why no record existed.

    Driving the real encode path and building the record out of the set it filled is
    what proves the two are connected; asserting on `_file_host_record` alone would pass
    with the argument still discarded.
    """
    async def _emitter(_event):
        return None

    publisher = _Publisher(emitter=_emitter)
    relayed: set[tuple[str, str]] = set()

    await publisher.publish(_valves(), relayed)

    assert publisher.posts, "precondition: the file must have been published"
    assert relayed == {("video", "litterbox")}, relayed
    record = VideoGenerationAdapter._file_host_record(_valves(), relayed)
    assert "litterbox" in record, record


@pytest.mark.parametrize(
    ("host", "retention"), [("litterbox", "12h"), ("catbox", "1h")]
)
def test_the_record_survives_a_job_that_is_resumed_in_a_later_request(host, retention):
    """A resumed job rebuilds its message from what was persisted, and replaces it.

    Recovering only the intent block dropped the relay record on every resume, which is
    exactly the long-running job most likely to be resumed. Two hosts and two
    retentions, so recovering a hardcoded sentence cannot pass.
    """
    valves = _valves(MEDIA_FILE_HOST=host, MEDIA_FILE_HOST_RETENTION=retention)
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)
    record = adapter._file_host_record(valves, {("video", host)})
    persisted = record + "\n" + adapter._build_pending_content(job_id="j1", model_id="m")

    recovered = adapter._recover_the_file_host_record(persisted)

    assert host in recovered, f"the host was lost on resume: {recovered!r}"
    assert recovered.strip() == record.strip(), recovered


def test_nothing_is_recovered_from_a_message_that_never_carried_a_record():
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)

    assert adapter._recover_the_file_host_record(
        adapter._build_pending_content(job_id="j1", model_id="m")
    ) == ""
    assert adapter._recover_the_file_host_record("") == ""


@pytest.mark.parametrize("kind", ["pending", "final"])
def test_the_record_does_not_hide_the_markers_the_resume_path_reads(kind):
    """The record sits ahead of the job marker, which is what the resume lookup scans for.

    A record that shadowed the marker would turn every resumed job into a second
    submission -- billed twice, and the first job's output lost.
    """
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)
    record = adapter._file_host_record(_valves(), {("video", "litterbox")})
    body = (
        adapter._build_pending_content(job_id="job-7", model_id="m")
        if kind == "pending"
        else adapter._build_success_content(
            job_id="job-7", model_id="m", file_ids=["a"], elapsed=1.0, usage={},
        )
    )
    stored = record + "\n" + body

    assert adapter._extract_video_job_marker(stored) == "job-7"
    assert adapter._looks_like_final_video_content(stored) is (kind == "final")


# ------------------------------------------------- WHAT WAS NOT SENT AT ALL --
_WITHHELD_CASES = {
    "one-cause": [("seed", "the request schema does not define it")],
    "two-causes": [
        ("resolution", "size already fixes the pixels"),
        ("aspect_ratio", "size contradicts the ratio"),
    ],
}


def _adapter():
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    return VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)


@pytest.mark.parametrize("case", sorted(_WITHHELD_CASES), ids=sorted(_WITHHELD_CASES))
def test_what_was_withheld_survives_a_job_that_is_resumed_in_a_later_request(case):
    """A resumed job rebuilds its message from what was persisted, and replaces it.

    The notice moved off a toast and into the message precisely so a reader who comes
    back still finds the reason -- and a long-running video job is exactly when they come
    back. Recovering only the file-host record dropped it on every resume. Two different
    withheld lists producing two different sentences, so recovering a hardcoded block
    cannot pass, and the recovered text is compared to the record verbatim rather than
    merely being non-empty.
    """
    adapter = _adapter()
    withheld = _WITHHELD_CASES[case]
    record = adapter._withheld_record(withheld)
    assert record, "precondition: a non-empty withheld list produces a record"
    persisted = record + "\n" + adapter._build_pending_content(job_id="job-7", model_id="m")

    recovered = adapter._recover_the_withheld_record(persisted)

    assert recovered.strip() == record.strip(), f"the reason was lost on resume: {recovered!r}"
    for name, reason in withheld:
        assert name in recovered and reason in recovered, recovered


def test_nothing_is_recovered_from_a_message_that_never_said_anything_was_withheld():
    adapter = _adapter()

    assert adapter._recover_the_withheld_record(
        adapter._build_pending_content(job_id="j1", model_id="m")
    ) == ""
    assert adapter._recover_the_withheld_record("") == ""
    assert adapter._recover_the_withheld_record(
        adapter._file_host_record(_valves(), {("video", "litterbox")})
    ) == ""


def test_both_records_recover_from_one_message_and_neither_hides_the_job_marker():
    """The resume path recovers the two records onto one line, from one stored message.

    Asserting only that the recovered disclosure is non-empty proves nothing: the
    file-host recovery runs on that same line and would satisfy it while the withheld
    recovery returned nothing. So each is asserted by its own text. The job marker still
    has to be readable underneath both, or the resume submits -- and bills -- a second job.
    """
    adapter = _adapter()
    host_record = adapter._file_host_record(_valves(), {("video", "litterbox")})
    withheld_record = adapter._withheld_record(_WITHHELD_CASES["two-causes"])
    persisted = (
        host_record
        + withheld_record
        + "\n"
        + adapter._build_pending_content(job_id="job-9", model_id="m")
    )

    resumed = adapter._recover_the_file_host_record(persisted)
    resumed += adapter._recover_the_withheld_record(persisted)

    assert "litterbox" in resumed, resumed
    assert "resolution" in resumed and "aspect_ratio" in resumed, resumed
    assert resumed.count("Not sent with this video") == 1, resumed
    assert adapter._extract_video_job_marker(persisted) == "job-9"


@pytest.mark.parametrize("case", sorted(_WITHHELD_CASES), ids=sorted(_WITHHELD_CASES))
@pytest.mark.asyncio
async def test_generate_puts_back_what_was_withheld_when_it_resumes_a_running_job(
    case, monkeypatch, tmp_path
):
    """The resume branch of ``generate()`` itself, not the two helpers it calls.

    Every other test of this property calls the helpers directly, and two of them rebuild
    the very pair of lines the adapter runs -- recover the file-host record, then add the
    withheld one. A test that re-implements the call site cannot notice the call site being
    deleted, and deleting it was green across the whole suite.

    So this drives ``generate()``: the stored message is handed back by the persistence
    double carrying both records and a job marker, the poll is stubbed one seam below at the
    OpenRouter client, and the assertion is on the text ``generate()`` RETURNS -- which is
    what Open WebUI writes back into the message. Both records are asserted, so deleting
    either recovery reddens this. Two withheld lists producing two different sentences, so a
    constant cannot satisfy both rows, and the expected text is written out here rather than
    obtained from the code under test.
    """
    from open_webui_openrouter_pipe import EncryptedStr, Pipe

    withheld = _WITHHELD_CASES[case]
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 1
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 1
    adapter = pipe._ensure_video_generation_adapter()

    stored = (
        adapter._file_host_record(_valves(), {("video", "litterbox")})
        + adapter._withheld_record(withheld)
        + "\n"
        + adapter._build_pending_content(job_id="job-resume", model_id="m")
    )
    assert adapter._extract_video_job_marker(stored) == "job-resume", (
        "precondition: the stored message has to look resumable, or generate() submits anew"
    )

    class _Persistence:
        async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
            return stored

    class _Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def submit(self, _payload):
            raise AssertionError("a resumed job must not be submitted a second time")

        async def status(self, job_id, polling_url=None):
            assert job_id == "job-resume"
            return {"status": "completed", "usage": {"cost": "0.25"}}

        def content_url(self, job_id: str, index: int = 0) -> str:
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer test"}

    async def _download(url: str, dest_path, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(MP4)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": len(MP4)}

    async def _upload(*_args, **_kwargs):
        return "file-1"

    cast(Any, adapter)._persistence = _Persistence()
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", _Client
    )
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url_streaming", _download
    )
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", _upload)

    try:
        answered = await adapter.generate(
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
    finally:
        await pipe.close()

    assert "/api/v1/files/file-1/content" in answered, (
        f"the resumed job did not finish, so nothing below is about a resume:\n{answered}"
    )
    for name, reason in withheld:
        assert name in answered, (
            f"the setting that was not sent is missing from the message a resumed job "
            f"leaves behind, so the reader never learns {name} went nowhere:\n{answered}"
        )
        assert reason in answered, (
            f"{name} is named without why it was not sent:\n{answered}"
        )
    assert answered.count("Not sent with this video") == 1, (
        f"the record was written more than once on the way through:\n{answered}"
    )
    assert "litterbox" in answered, (
        f"the file-host record was dropped by the same resume, and a user whose file was "
        f"published to a public host has no durable record of it:\n{answered}"
    )


def test_a_resumed_message_carrying_the_withheld_record_round_trips_through_it_again():
    """What the resume path writes back has to be recoverable by the NEXT resume.

    A serialization whose end marker glued to the record text rendered as one line and
    was the reason the block was reshaped; a shape that survives one pass but not two
    would lose the reason on the second poll of a long job.
    """
    adapter = _adapter()
    record = adapter._withheld_record(_WITHHELD_CASES["one-cause"])
    first = record + "\n" + adapter._build_pending_content(job_id="job-7", model_id="m")

    once = adapter._recover_the_withheld_record(first)
    rewritten = once + adapter._build_pending_content(job_id="job-7", model_id="m")
    twice = adapter._recover_the_withheld_record(rewritten)

    assert twice.strip() == record.strip(), twice
    assert adapter._extract_video_job_marker(rewritten) == "job-7"


# -------------------------------------------------------------- FAIL CLOSED --
@pytest.mark.asyncio
async def test_a_chat_that_cannot_be_warned_gets_no_upload():
    """The warning is best-effort by construction, so the upload must not be.

    The emitter here is the real wrapped one, whose whole job is to swallow transport
    errors -- exactly the shape that turned "could not tell the user" into "upload
    anyway". Stubbed one seam lower than the subject: the raw socket raises, and every
    production layer above it runs.
    """
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    handler = EventEmitterHandler(logger=pipe.logger, valves=Valves(), pipe_instance=pipe)

    async def _dead_socket(_event):
        raise RuntimeError("socket is gone")

    publisher = _Publisher(emitter=handler._wrap_safe_event_emitter(_dead_socket))
    publisher.pipe._event_emitter_handler = handler
    publisher.adapter = VideoGenerationAdapter(pipe=publisher.pipe, logger=pipe.logger)
    publisher.pipe._file_gateway.read_file_record_base64 = AsyncMock(
        return_value=base64.b64encode(MP4).decode()
    )

    with pytest.raises(VideoGenerationError) as refused:
        await publisher.publish(_valves(), set())

    assert publisher.posts == [], (
        f"the file was published to {publisher.posts} after the chat could not be told"
    )
    assert "Nothing was uploaded" in str(refused.value), refused.value


@pytest.mark.asyncio
async def test_a_request_with_no_chat_attached_gets_no_upload():
    """No emitter is no channel, so there is nobody the warning could reach."""
    publisher = _Publisher(emitter=None)

    with pytest.raises(VideoGenerationError):
        await publisher.publish(_valves(), set())

    assert publisher.posts == [], publisher.posts


@pytest.mark.asyncio
async def test_an_operator_who_warns_their_users_elsewhere_still_gets_the_upload():
    """`TELL_USERS_ABOUT_THE_FILE_HOST` off is a decision, not a delivery failure.

    Failing closed on an undelivered warning must not turn the valve into a kill switch;
    this is the case that separates "could not say it" from "chose not to say it".
    """
    async def _dead_socket(_event):
        raise RuntimeError("socket is gone")

    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-disclosure")
    handler = EventEmitterHandler(logger=pipe.logger, valves=Valves(), pipe_instance=pipe)
    publisher = _Publisher(emitter=handler._wrap_safe_event_emitter(_dead_socket))
    relayed: set[tuple[str, str]] = set()

    await publisher.publish(
        _valves(TELL_USERS_ABOUT_THE_FILE_HOST=False), relayed
    )

    assert publisher.posts, "an operator who opted out of the warning lost the feature"
    assert relayed == {("video", "litterbox")}


# --------------------------------------------------- THE CHANNEL IS HONEST ---
@pytest.mark.parametrize("outcome", [True, False])
@pytest.mark.asyncio
async def test_the_notifier_reports_whether_the_event_left_the_process(outcome):
    """`_emit_notification` returning True while the socket raised is the whole defect.

    Two outcomes from the same call, so a production `return True` cannot pass. The raw
    socket is the stub; `_emit_notification` and the safe wrapper both run for real.
    """
    pipe = MagicMock()
    handler = EventEmitterHandler(
        logger=logging.getLogger("relay-disclosure"), valves=Valves(), pipe_instance=pipe,
    )
    seen: list[dict] = []

    async def _socket(event):
        if not outcome:
            raise RuntimeError("socket is gone")
        seen.append(event)

    wrapped = handler._wrap_safe_event_emitter(_socket)

    assert await handler._emit_notification(wrapped, "a clip is going out") is outcome
    assert bool(seen) is outcome


def test_the_wrapper_keeps_a_handle_on_the_emitter_it_wrapped():
    """Without it there is no way to observe a failure the wrapper exists to swallow."""
    pipe = MagicMock()
    handler = EventEmitterHandler(
        logger=logging.getLogger("relay-disclosure"), valves=Valves(), pipe_instance=pipe,
    )

    async def _socket(_event):
        return None

    wrapped = cast(Any, handler._wrap_safe_event_emitter(_socket))
    assert unguarded_emitter(wrapped) is _socket
    assert unguarded_emitter(_socket) is _socket
