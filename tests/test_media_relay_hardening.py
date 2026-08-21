"""What the relay may publish, on whose say-so, and what it tells the user first.

Uploading to litterbox or catbox needs no account, which is the whole reason the feature
works -- and the reason nothing on this side can ever delete what it sent. Every property
here follows from that: the user is told before the bytes go, only the owner of a file may
send it, the type comes from the stored record rather than from the browser, one request
has a ceiling, and the link that comes back is only believed if the host that took the
file is the one serving it.

The seam stubbed throughout is the storage read and the HTTP session -- both a level below
the encoder under test, so its own gates really run.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.integrations import video as video_module
from open_webui_openrouter_pipe.integrations.media_relay import (
    _ENDPOINTS,
    MediaRelayError,
    _as_the_host_put_it,
    _extract_url,
    _validate_endpoints,
    host_keeps_forever,
    relay_to_public_url,
    usable_media_type,
)
from open_webui_openrouter_pipe.integrations.video import (
    _MAX_INPUT_REFERENCES,
    VideoGenerationAdapter,
)
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError
from open_webui_openrouter_pipe.storage.owui_files import (
    authorize_file_publication,
    infer_file_mime_type,
)

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
OWNER = SimpleNamespace(id="bob", role="user")


def _png(width: int, height: int) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 34, 56)).save(buffer, "PNG")
    return buffer.getvalue()


class _Ctx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


def _record(file_id, stored_mime, *, owner="bob", filename="clip.mp4", size=None):
    meta = {"content_type": stored_mime}
    if size is not None:
        meta["size"] = size
    return SimpleNamespace(
        id=file_id, filename=filename, user_id=owner, mime_type=None,
        content_type=None, meta=meta,
    )


class _Relay:
    """Drives the real encoder against a real aiohttp session and a mocked host."""

    def __init__(self, blobs, records, *, answer="https://files.catbox.moe/PUBLISHED.mp4"):
        self.blobs, self.records, self.answer = blobs, records, answer
        self.posts: list[str] = []
        self.notices: list[str] = []
        self.timeline: list[str] = []
        self.pipe = MagicMock()
        self.pipe.logger = logging.getLogger("relay-hardening")

        async def _read(file_obj, _chunk, _cap, user=None):
            return base64.b64encode(self.blobs[file_obj.id]).decode()

        async def _notify(_emitter, content, *, level="info"):
            self.notices.append(content)
            self.timeline.append("told")

        self.pipe._file_gateway.read_file_record_base64 = _read
        self.pipe._event_emitter_handler._emit_notification = _notify
        self.adapter = VideoGenerationAdapter(pipe=self.pipe, logger=self.pipe.logger)

    def _wire(self, url, **_kwargs):
        self.posts.append(str(url))
        self.timeline.append("published")
        return CallbackResult(status=200, body=self.answer)

    async def encode(self, refs, valves, *, user=OWNER, withheld=None, emitter=object()):
        async def _get_file(file_id, _logger):
            return self.records.get(file_id)

        video_module.get_file_by_id = _get_file
        video_module.infer_file_mime_type = infer_file_mime_type
        async with aiohttp.ClientSession() as session:
            self.pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
            with aioresponses() as mocked:
                for endpoint in _ENDPOINTS.values():
                    mocked.post(endpoint.url, callback=self._wire, repeat=True)
                return await self.adapter._encode_input_references(
                    {"input_references": refs, "model_id": "runway/aleph-2"},
                    valves,
                    withheld=withheld if withheld is not None else [],
                    user_obj=user,
                    video_model={
                        "id": "runway/aleph-2",
                        "input_modalities": ["video", "image", "audio"],
                    },
                    relayed=set(),
                    companions=True,
                    event_emitter=emitter,
                )


def _relaying_valves(**overrides):
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    for name, value in overrides.items():
        setattr(valves, name, value)
    return valves


# ------------------------------------------------------- told before it goes --
@pytest.mark.parametrize(
    ("host", "must_say"), [("litterbox", "hour"), ("catbox", "for good")]
)
@pytest.mark.asyncio
async def test_the_chat_is_told_which_host_and_for_how_long_before_the_upload(host, must_say):
    """Publication is irreversible, so the disclosure cannot wait for the request to end.

    Two hosts because they make different promises: one deletes itself, one never does,
    and a notice that says the same thing for both is not a disclosure.
    """
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", "video/mp4")})

    await relay.encode([{"id": "f1"}], _relaying_valves(MEDIA_FILE_HOST=host))

    assert relay.timeline == ["told", "published"], relay.timeline
    assert host in relay.notices[0], relay.notices
    assert must_say in relay.notices[0], relay.notices


@pytest.mark.asyncio
async def test_the_fallback_host_is_named_up_front_and_again_once_it_took_the_file():
    """With the fallback on, either host may end up holding the file, so both are named.

    Naming only the configured host would be a disclosure that is wrong exactly when it
    matters -- the fallback swaps an hour's retention for a permanent one.
    """
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", "video/mp4")})
    valves = _relaying_valves(
        MEDIA_FILE_HOST="litterbox", USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN=True
    )

    await relay.encode([{"id": "f1"}], valves)

    assert relay.timeline[0] == "told", relay.timeline
    assert "litterbox" in relay.notices[0] and "catbox" in relay.notices[0], relay.notices
    assert relay.notices[-1] != relay.notices[0], (
        "the host that actually took the file was never narrowed down"
    )
    assert "catbox" not in relay.notices[-1], relay.notices


@pytest.mark.asyncio
async def test_an_operator_who_turned_the_notice_off_gets_no_notice():
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", "video/mp4")})

    await relay.encode(
        [{"id": "f1"}], _relaying_valves(TELL_USERS_ABOUT_THE_FILE_HOST=False)
    )

    assert relay.notices == []
    assert relay.posts, "the upload itself is the operator's decision, not the notice's"


# ---------------------------------------------------------------- who may publish --
@pytest.mark.parametrize(
    ("owner", "published"), [("bob", True), ("alice", False)]
)
@pytest.mark.asyncio
async def test_only_the_person_who_owns_a_file_can_put_it_on_a_public_host(owner, published):
    """Open WebUI grants read through shared chats, channels and knowledge bases.

    Being able to watch a colleague's clip in a chat they shared is not permission to
    publish it to an anonymous host that nobody here can delete from. The storage read is
    stubbed to succeed either way, which is exactly the situation being defended: the read
    is allowed, the publication is not.
    """
    withheld: list[tuple[str, str]] = []
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", "video/mp4", owner=owner)})

    encoded = await relay.encode([{"id": "f1"}], _relaying_valves(), withheld=withheld)

    assert bool(relay.posts) is published, relay.posts
    assert bool(encoded) is published, encoded
    if not published:
        assert withheld and "your own" in withheld[0][1], withheld


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (SimpleNamespace(id="bob", role="user"), True),
        (SimpleNamespace(id="alice", role="user"), False),
        (SimpleNamespace(id="alice", role="admin"), True),
        ({"id": "bob", "role": "user"}, True),
        ({"id": "alice", "role": "user"}, False),
        (None, False),
        (SimpleNamespace(role="user"), False),
    ],
)
def test_publication_is_authorised_for_the_owner_and_the_administrator_only(user, expected):
    """Fail closed: a requester the pipe cannot identify owns nothing."""
    assert authorize_file_publication(_record("f1", "video/mp4", owner="bob"), user) is expected


# ------------------------------------------------------------- which type governs --
@pytest.mark.parametrize("declared", ["video/mp4", "audio/mpeg"])
@pytest.mark.asyncio
async def test_a_type_only_the_browser_declared_never_puts_a_file_on_a_host(declared):
    """`body["files"][i]["content_type"]` is request JSON, and Open WebUI stores a blank
    type for any upload whose part header it could not parse.

    The declared type decides the family, which decides which of the per-kind valves
    applies -- so trusting it lets a picture be published under the video switch while the
    picture switch is off. Two declared types, so a rule that only knows about video fails
    the audio row.
    """
    withheld: list[tuple[str, str]] = []
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", ";")})

    encoded = await relay.encode(
        [{"id": "f1", "content_type": declared}], _relaying_valves(), withheld=withheld
    )

    assert relay.posts == [], f"{declared!r} from the request published a file"
    assert encoded == []
    assert withheld and "no media type" in withheld[0][1], withheld


@pytest.mark.parametrize(
    "stored", ["video/mp4", "video/mp4; codecs=avc1", "VIDEO/MP4"]
)
@pytest.mark.asyncio
async def test_the_stored_type_is_what_the_file_is_published_as(stored):
    """The record decides, and the record's own spelling variants all mean the same clip."""
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", stored)})

    encoded = await relay.encode([{"id": "f1", "content_type": "image/png"}], _relaying_valves())

    assert len(relay.posts) == 1
    assert encoded and encoded[0]["type"] == "video_url", encoded


@pytest.mark.parametrize(
    "value",
    [
        "video/mp4\r\nX-Injected: yes",
        "video/mp4\nX: y",
        ";",
        "",
        "video",
        "video/",
        "/mp4",
        None,
        12,
    ],
)
def test_a_value_that_is_not_a_media_type_is_not_treated_as_one(value):
    assert usable_media_type(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("video/mp4", "video/mp4"),
        ("Audio/MPEG", "audio/mpeg"),
        (" image/png ; charset=binary", "image/png"),
    ],
)
def test_an_ordinary_media_type_still_passes(value, expected):
    assert usable_media_type(value) == expected


@pytest.mark.parametrize("hostile", ["video/mp4\r\nX-Injected: yes", "audio/mpeg\nX: y"])
def test_a_media_type_carrying_a_header_break_is_refused_not_raised_on(hostile):
    """aiohttp raises ValueError on CRLF in a header value, and nothing here catches it.

    A refusal is a failure card naming the file; the ValueError is a 500 with the request
    already half-published.
    """
    with pytest.raises(MediaRelayError, match="not a media type"):
        asyncio.run(
            relay_to_public_url(
                MagicMock(), b"CLIP", filename="c.mp4", mime=hostile,
                host="litterbox", retention="1h", max_bytes=0,
            )
        )


# --------------------------------------------------------------- what one request may do --
@pytest.mark.parametrize("offered", [17, 200])
@pytest.mark.asyncio
async def test_one_request_publishes_at_most_the_reference_ceiling(offered):
    """The list arrives from the request, and every accepted entry is held in memory as
    base64 until the uploads start.

    Two absolute lengths rather than lengths derived from the ceiling, so raising the
    ceiling cannot quietly satisfy both rows: seventeen is one over the documented
    sixteen, and two hundred is what an ordinary user can send by hand.
    """
    assert _MAX_INPUT_REFERENCES == 16, (
        "the ceiling is published in the valve atlas and the video document as sixteen; "
        "changing it is a documentation change too"
    )
    withheld: list[tuple[str, str]] = []
    blobs = {f"f{i}": MP4 for i in range(offered)}
    records = {f"f{i}": _record(f"f{i}", "video/mp4") for i in range(offered)}
    relay = _Relay(blobs, records)

    encoded = await relay.encode(
        [{"id": f"f{i}"} for i in range(offered)], _relaying_valves(), withheld=withheld
    )

    assert len(relay.posts) == _MAX_INPUT_REFERENCES, len(relay.posts)
    assert len(encoded) == _MAX_INPUT_REFERENCES
    assert withheld and str(offered - _MAX_INPUT_REFERENCES) in withheld[-1][0], withheld


@pytest.mark.parametrize("cap_mb", [1, 2])
@pytest.mark.asyncio
async def test_one_request_cannot_publish_more_bytes_than_the_operator_allows(cap_mb):
    """The cap the operator set is per file AND per request; three files of the same size
    otherwise walk straight past it. Two caps, so a constant cannot satisfy both."""
    each = (cap_mb * 1024 * 1024) // 2
    blobs = {f"f{i}": MP4 + b"X" * each for i in range(3)}
    records = {f"f{i}": _record(f"f{i}", "video/mp4") for i in range(3)}
    relay = _Relay(blobs, records)

    with pytest.raises(VideoGenerationError, match="sends at most"):
        await relay.encode(
            [{"id": f"f{i}"} for i in range(3)],
            _relaying_valves(MEDIA_FILE_HOST_MAX_SIZE_MB=cap_mb),
        )

    assert relay.posts == [], "the request was already publishing before it counted"


@pytest.mark.parametrize("declared_mb", [3, 9])
@pytest.mark.asyncio
async def test_a_file_the_record_already_says_is_too_big_is_never_read(declared_mb):
    """The relay cap used to be enforced inside the uploader -- after a full read and a
    base64 encode of a file that was always going to be refused."""
    reads: list[str] = []
    relay = _Relay({"f1": MP4}, {"f1": _record("f1", "video/mp4", size=declared_mb * 1024 * 1024)})
    inner = relay.pipe._file_gateway.read_file_record_base64

    async def _record_read(file_obj, *args, **kwargs):
        reads.append(file_obj.id)
        return await inner(file_obj, *args, **kwargs)

    relay.pipe._file_gateway.read_file_record_base64 = _record_read

    with pytest.raises(VideoGenerationError, match="limit for sending media"):
        await relay.encode([{"id": "f1"}], _relaying_valves(MEDIA_FILE_HOST_MAX_SIZE_MB=2))

    assert reads == [], "the file was read out of storage before the cap was consulted"
    assert relay.posts == []


# ------------------------------------------------------------- what comes back --
@pytest.mark.parametrize("host", ["litterbox", "catbox"])
def test_a_link_the_chosen_host_serves_is_accepted(host):
    """The refusals below must not be a blanket one: the ordinary answer still works."""
    assert (
        _extract_url("https://files.catbox.moe/ok.mp4", _ENDPOINTS[host].origins)
        == "https://files.catbox.moe/ok.mp4"
    )


@pytest.mark.parametrize(
    "answer",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:8080/admin",
        "https://attacker.example/beacon.mp4",
        "https://catbox.moe.attacker.example/x.mp4",
        "http://files.catbox.moe/ok.mp4",
    ],
)
def test_an_address_the_chosen_host_does_not_serve_is_not_a_link(answer):
    """Whatever the body says becomes a URL handed to OpenRouter, so a 200 carrying an
    address somewhere else is a redirection, not an answer."""
    assert _extract_url(answer, _ENDPOINTS["litterbox"].origins) == ""


def test_every_known_host_declares_origins_that_cover_its_own_endpoint():
    """The allowlist and the upload address are the same fact stated twice."""
    _validate_endpoints(_ENDPOINTS)
    with pytest.raises(ValueError, match="do not cover"):
        _validate_endpoints(
            {"catbox": _ENDPOINTS["catbox"]._replace(origins=("example.test",))}
        )
    with pytest.raises(ValueError, match="names no origin"):
        _validate_endpoints({"catbox": _ENDPOINTS["catbox"]._replace(origins=())})


def test_the_upload_does_not_follow_a_redirect_to_somewhere_else():
    """A 307 would re-post the user's file wherever it pointed, with no second check."""
    posted: list[bool] = []

    def _seen(_url, **kwargs):
        posted.append(kwargs.get("allow_redirects", True))
        return CallbackResult(status=200, body="https://files.catbox.moe/ok.mp4")

    async def _run():
        async with aiohttp.ClientSession() as session:
            with aioresponses() as http:
                http.post(_ENDPOINTS["catbox"].url, callback=_seen, repeat=True)
                return await relay_to_public_url(
                    session, b"CLIP", filename="c.mp4", mime="video/mp4",
                    host="catbox", retention="1h", max_bytes=0,
                )

    assert asyncio.run(_run()) == "https://files.catbox.moe/ok.mp4"
    assert posted == [False], "the upload followed redirects"


@pytest.mark.parametrize(
    "body",
    [
        "![](https://attacker.example/px.gif?u=1)[Click to fix](https://evil.example)",
        "<img src=x onerror=alert(1)>",
    ],
)
def test_what_the_host_said_reaches_the_chat_as_text_not_as_markup(body):
    """The failure line is rendered as assistant markdown, so an anonymous host was
    writing images and links into a user's conversation."""
    quoted = _as_the_host_put_it(body)

    assert "](" not in quoted and "://" not in quoted, quoted
    assert "<" not in quoted and ">" not in quoted, quoted
    assert quoted.startswith("`") and quoted.endswith("`"), quoted


def test_an_ordinary_complaint_from_the_host_still_reads():
    assert "are you a robot" in _as_the_host_put_it("<html>are you a robot</html>")


# ------------------------------------------------------------------ what is true --
@pytest.mark.parametrize("host", ["litterbox", "catbox"])
def test_the_notice_never_promises_a_deletion_nobody_here_can_perform(host):
    """The upload carries no account, so no credential exists on this side to delete with.

    Both hosts, because the sentence differs: litterbox deletes itself and may say so.
    """
    valves = Valves()
    valves.MEDIA_FILE_HOST = host
    line = VideoGenerationAdapter._file_host_notice(valves, {("video", host)})

    if host_keeps_forever(host):
        assert "nothing here can take it down" in line, line
        assert "someone deletes it" not in line, line
    else:
        assert "deleted again" in line, line


# ------------------------------------------------------------------- the latch --
@pytest.mark.parametrize("count", [3, 12])
@pytest.mark.asyncio
async def test_one_cause_warns_once_however_many_files_it_drops(count, caplog):
    """The warn-once latch is a process-global set, and two of these reasons carry the
    user's own image dimensions -- so keying it on the whole sentence let a request write
    an unbounded number of permanent entries into it.

    Two request sizes, so a rule that warns once per request rather than once per cause
    fails the larger row.
    """
    from open_webui_openrouter_pipe.integrations.video import _warned_dropped_video_param

    _warned_dropped_video_param.clear()
    blobs = {f"f{i}": MP4 for i in range(count)}
    records = {f"f{i}": _record(f"f{i}", "video/mp4", owner="alice") for i in range(count)}
    relay = _Relay(blobs, records)

    with caplog.at_level(logging.DEBUG, logger=relay.adapter.logger.name):
        await relay.encode([{"id": f"f{i}"} for i in range(count)], _relaying_valves())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"{len(warnings)} WARNINGs for one cause"
    assert len(_warned_dropped_video_param) == 1, _warned_dropped_video_param
    assert len([r for r in caplog.records if r.levelno == logging.DEBUG]) >= count - 1


@pytest.mark.parametrize(("width", "height"), [(64, 64), (100, 100)])
@pytest.mark.asyncio
async def test_a_reason_carrying_the_users_own_numbers_does_not_widen_the_latch(
    width, height, caplog
):
    """The picture-size reason embeds the size the user sent. Keyed on the sentence, every
    distinct size armed its own permanent latch entry; keyed on the cause, one does."""
    from open_webui_openrouter_pipe.integrations.video import _warned_dropped_video_param

    _warned_dropped_video_param.clear()
    blobs = {"f1": _png(width, height), "f2": _png(width * 2, height * 2)}
    records = {
        "f1": _record("f1", "image/png", filename="a.png"),
        "f2": _record("f2", "image/png", filename="b.png"),
    }
    relay = _Relay(blobs, records)
    withheld: list[tuple[str, str]] = []

    with caplog.at_level(logging.DEBUG, logger=relay.adapter.logger.name):
        await relay.encode(
            [{"id": "f1"}, {"id": "f2"}], _relaying_valves(), withheld=withheld
        )

    assert len(withheld) == 2, withheld
    assert withheld[0][1] != withheld[1][1], "the two sizes produced the same sentence"
    assert len(_warned_dropped_video_param) == 1, _warned_dropped_video_param
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
