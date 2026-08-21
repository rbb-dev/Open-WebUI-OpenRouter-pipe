"""What a published file leaves behind, and what has to be true before it is published.

Every test here defends one of two properties:

  * a user whose file was uploaded to an anonymous public host has a DURABLE record of
    it -- naming the host and how long it stays -- inside the message Open WebUI stores,
    not only in a toast that a reload discards; and
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
