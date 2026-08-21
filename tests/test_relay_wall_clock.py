"""How long one request may spend uploading a user's files, and to how many places.

The relay carries no credential, so nothing it publishes can ever be deleted. That makes
"post it again" a permanent, unbounded cost rather than a retry, and makes wall clock a
resource the requester must not be able to choose.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

import open_webui_openrouter_pipe.integrations.media_relay as media_relay
from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.integrations.media_relay import (
    _ENDPOINTS,
    MAX_RELAY_SECONDS_PER_REQUEST,
    MediaRelayError,
    relay_to_public_url,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


class _Ctx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.parametrize("budget", [11.0, 47.0])
@pytest.mark.asyncio
async def test_an_upload_is_given_no_more_time_than_the_budget_it_was_handed(budget):
    """Three attempts at a five-minute timeout each is fifteen minutes per host.

    The assertion is on the deadline the transport is handed, because that is the thing
    that bounds a real socket; a stalled `aioresponses` callback is not subject to it.
    Two budgets, both under the 300s per-POST ceiling, so a production
    `ClientTimeout(total=_MAX_UPLOAD_SECONDS)` that ignores the argument fails both.
    """
    asked: list[float] = []

    def _capture(_url, **kwargs):
        asked.append(kwargs["timeout"].total)
        return CallbackResult(status=200, body="https://files.catbox.moe/a.mp4")

    async with aiohttp.ClientSession() as session:
        with aioresponses() as http:
            http.post(_ENDPOINTS["litterbox"][0], callback=_capture, repeat=True)
            await relay_to_public_url(
                session, MP4, filename="c.mp4", mime="video/mp4",
                host="litterbox", retention="1h", max_bytes=0,
                seconds_left=budget,
            )

    assert asked and asked[0] <= budget, (
        f"a {budget}s budget asked the transport for {asked}"
    )


@pytest.mark.asyncio
async def test_an_upload_handed_no_time_at_all_never_reaches_the_host():
    """An exhausted request budget must refuse rather than start one more transfer."""
    async with aiohttp.ClientSession() as session:
        with aioresponses() as http:
            http.post(_ENDPOINTS["litterbox"][0], status=200, body="https://x/a.mp4",
                      repeat=True)
            with pytest.raises(MediaRelayError) as refused:
                await relay_to_public_url(
                    session, MP4, filename="c.mp4", mime="video/mp4",
                    host="litterbox", retention="1h", max_bytes=0, seconds_left=0.0,
                )
            assert list(http.requests) == [], "an out-of-time upload still reached a host"
    assert refused.value.may_have_stored_it is False
    assert "ran out" in str(refused.value)


def test_the_default_budget_is_finite_and_shorter_than_the_old_worst_case():
    """906 seconds per reference per host was the loop's own worst case."""
    assert 0 < MAX_RELAY_SECONDS_PER_REQUEST < 906


@pytest.mark.parametrize("references", [2, 3])
@pytest.mark.asyncio
async def test_one_request_shares_a_single_upload_budget_across_its_references(
    references, monkeypatch
):
    """A budget re-derived per reference is a budget multiplied by the attachment count.

    Sixteen references would restore the unbounded wall clock while every single-file
    test still passed. Two counts, so a production rule keyed to one cannot pass.
    """
    handed: list[float] = []

    async def _relay(_session, _blob, **kwargs):
        handed.append(kwargs["seconds_left"])
        await asyncio.sleep(0.05)
        return "https://files.catbox.moe/a.mp4"

    from open_webui_openrouter_pipe.integrations import video as video_module
    from open_webui_openrouter_pipe.storage.owui_files import infer_file_mime_type
    from types import SimpleNamespace

    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-clock")
    pipe._event_emitter_handler._emit_notification = _AlwaysHeard()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)

    async def _read(_file_obj, _chunk, _cap, user=None):
        return base64.b64encode(MP4).decode()

    async def _file(file_id, _logger):
        return SimpleNamespace(
            id=file_id, filename="clip.mp4", user_id="bob", mime_type=None,
            content_type=None, meta={"content_type": "video/mp4"},
        )

    pipe._file_gateway.read_file_record_base64 = _read
    monkeypatch.setattr(video_module, "get_file_by_id", _file)
    monkeypatch.setattr(video_module, "infer_file_mime_type", infer_file_mime_type)
    monkeypatch.setattr(video_module, "relay_to_public_url", _relay)

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True

    async def _emitter(_event):
        return None

    async with aiohttp.ClientSession() as session:
        pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
        await adapter._encode_input_references(
            {
                "input_references": [{"id": f"f{i}"} for i in range(references)],
                "model_id": "runway/aleph-2",
            },
            valves,
            withheld=[],
            user_obj=SimpleNamespace(id="bob", role="user"),
            video_model={"id": "runway/aleph-2",
                         "input_modalities": ["video", "image", "audio"]},
            relayed=set(),
            companions=True,
            event_emitter=_emitter,
        )

    assert len(handed) == references, handed
    assert all(later < earlier for earlier, later in zip(handed, handed[1:])), (
        f"each reference was handed its own fresh budget: {handed}"
    )
    assert handed[0] <= MAX_RELAY_SECONDS_PER_REQUEST


class _AlwaysHeard:
    async def __call__(self, *_args, **_kwargs):
        return True


@pytest.mark.parametrize("budget", [8.0, 30.0])
@pytest.mark.asyncio
async def test_the_second_host_draws_on_what_the_first_one_already_spent(
    budget, monkeypatch
):
    """A budget re-derived per host doubles the wall clock the moment a fallback fires.

    The first host is unreachable, which is the one failure that both retries and frees
    the fallback -- three attempts there and one at the second host, all four drawing on
    one deadline. Two budgets, so a production per-host constant cannot pass either.
    """
    asked: list[float] = []

    async def _tick(_url, **kwargs):
        asked.append(kwargs["timeout"].total)
        await asyncio.sleep(0.05)
        return CallbackResult(status=200, body="https://files.catbox.moe/second.mp4")

    unreachable = aiohttp.ClientConnectorError(MagicMock(ssl=None), OSError(111, "refused"))
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-clock")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)
    valves = Valves()
    valves.MEDIA_FILE_HOST = "litterbox"
    valves.USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN = True

    real_sleep = asyncio.sleep

    async def _pause(seconds):
        await real_sleep(min(seconds, 0.01))

    monkeypatch.setattr(media_relay, "_RETRY_PAUSE_SECONDS", 0.01)
    monkeypatch.setattr(media_relay.asyncio, "sleep", _pause)

    async with aiohttp.ClientSession() as session:
        pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
        with aioresponses() as http:
            http.post(_ENDPOINTS["litterbox"][0], exception=unreachable, repeat=True)
            http.post(_ENDPOINTS["catbox"][0], callback=_tick, repeat=True)
            link, host = await adapter._relay_reference(
                valves, base64.b64encode(MP4).decode(), filename="c.mp4",
                mime="video/mp4", family="video",
                deadline=time.monotonic() + budget,
            )

    assert host == "catbox" and link.endswith("second.mp4")
    assert asked and asked[-1] < budget, (
        f"the second host was handed {asked[-1]}s out of a {budget}s request budget"
    )


@pytest.mark.parametrize("pause", [2.0, 0.0])
def test_the_retry_pause_never_outlives_the_budget(pause, monkeypatch):
    """A fixed pause between attempts can itself exceed what is left.

    Two pauses, so a production `sleep(_RETRY_PAUSE_SECONDS * attempt)` that ignores the
    remaining budget fails the first and a hardcoded zero fails nothing but proves
    nothing either -- the assertion is on what was requested, not on what elapsed.
    """
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    failure = aiohttp.ClientConnectorError(MagicMock(ssl=None), OSError(111, "refused"))

    async def _run():
        async with aiohttp.ClientSession() as session:
            with aioresponses() as http:
                http.post(_ENDPOINTS["litterbox"][0], exception=failure, repeat=True)
                monkeypatch.setattr(media_relay, "_RETRY_PAUSE_SECONDS", pause)
                monkeypatch.setattr(media_relay.asyncio, "sleep", _record)
                with pytest.raises(MediaRelayError):
                    await relay_to_public_url(
                        session, MP4, filename="c.mp4", mime="video/mp4",
                        host="litterbox", retention="1h", max_bytes=0, seconds_left=0.5,
                    )

    asyncio.run(_run())

    assert slept, "no retry happened, so the pause was never exercised"
    assert all(s <= 0.5 for s in slept), f"a pause outran the whole budget: {slept}"
