"""The file host is the only way an attachment can reach a video model.

OpenRouter takes reference media as a link its providers fetch, and refuses audio or video
sent any other way -- 38 request shapes were tried against the live API and every one was
rejected. So an attachment either goes through a public host or it does not go at all, and
the rules below are what keep that from happening by accident.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.integrations import media_relay
from open_webui_openrouter_pipe.integrations.media_relay import _ENDPOINTS
from open_webui_openrouter_pipe.integrations.media_relay import (
    RELAY_HOSTS,
    RELAY_RETENTIONS,
    MediaRelayError,
    host_keeps_forever,
    relay_to_public_url,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError

# Every OWUI file row carries the id of the person who uploaded it, and every request
# carries the person making it. Relaying is gated on those two being the same, so a
# record without an owner is not a record this code path can ever see.
_OWNER = SimpleNamespace(id="bob", role="user")


def test_nothing_is_uploaded_until_an_operator_asks_for_it():
    """A user's private media must not leave the server on a default install."""
    valves = Valves()
    assert valves.SEND_MEDIA_VIA_FILE_HOST is False
    for family in ("video", "audio", "image"):
        assert VideoGenerationAdapter._file_host_wanted(valves, family) is False, (
            f"{family} would be uploaded with the master switch off"
        )


@pytest.mark.parametrize(
    ("family", "expected"), [("video", True), ("audio", True), ("image", False)]
)
def test_each_kind_is_relayed_only_when_its_own_switch_is_on(family, expected):
    """Pictures already travel inside the request, so they are not sent away by default."""
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    assert VideoGenerationAdapter._file_host_wanted(valves, family) is expected

    for name in ("SEND_VIDEO_VIA_FILE_HOST", "SEND_AUDIO_VIA_FILE_HOST", "SEND_IMAGES_VIA_FILE_HOST"):
        setattr(valves, name, False)
    assert VideoGenerationAdapter._file_host_wanted(valves, family) is False


@pytest.mark.parametrize("host", RELAY_HOSTS)
@pytest.mark.parametrize("retention", RELAY_RETENTIONS)
def test_the_notice_reads_as_a_sentence_on_every_host_and_every_span(host, retention):
    """Whatever the operator picked, the line a user sees has to make sense."""
    valves = Valves()
    valves.MEDIA_FILE_HOST = host
    valves.MEDIA_FILE_HOST_RETENTION = retention
    line = VideoGenerationAdapter._file_host_notice(valves, {("video", host)})

    assert host in line
    assert "{" not in line and "}" not in line, f"a placeholder was left unfilled: {line}"
    assert line.endswith("."), line
    if host_keeps_forever(host):
        assert "stays there for good" in line, line
        assert "nothing here can take it down" in line, (
            f"the notice implies somebody at this deployment could delete it: {line}"
        )
    else:
        assert "deleted again" in line, line


def test_an_operator_can_write_the_notice_in_their_own_language():
    valves = Valves()
    valves.FILE_HOST_NOTICE = "Ihre Datei geht an {host}."
    assert VideoGenerationAdapter._file_host_notice(valves, {("video", "litterbox")}) == (
        "Ihre Datei geht an litterbox."
    )


def test_a_notice_that_names_no_placeholders_is_left_alone():
    valves = Valves()
    valves.FILE_HOST_NOTICE = "Uploading your file so the model can read it."
    assert VideoGenerationAdapter._file_host_notice(valves, {("video", "litterbox")}) == (
        "Uploading your file so the model can read it."
    )


@pytest.mark.parametrize(("size_mb", "cap_mb"), [(4, 1), (300, 200)])
def test_a_file_over_the_cap_is_refused_before_anything_is_uploaded(size_mb, cap_mb):
    """The refusal names both numbers, because 'too big' alone tells an operator nothing."""
    session = MagicMock()
    session.post = MagicMock(side_effect=AssertionError("an oversize file must not be posted"))

    with pytest.raises(MediaRelayError) as raised:
        asyncio.run(
            relay_to_public_url(
                session,
                b"x" * (size_mb * 1024 * 1024),
                filename="clip.mp4",
                mime="video/mp4",
                host="litterbox",
                retention="1h",
                max_bytes=cap_mb * 1024 * 1024,
            )
        )
    message = str(raised.value)
    assert f"{size_mb}.0 MB" in message and f"{cap_mb}.0 MB" in message, message


def test_a_host_this_pipe_does_not_know_is_refused_by_name():
    with pytest.raises(MediaRelayError, match="not a file host"):
        asyncio.run(
            relay_to_public_url(
                MagicMock(), b"x", filename="a.mp4", mime="video/mp4",
                host="somewhere-else", retention="1h", max_bytes=0,
            )
        )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("https://files.catbox.moe/abc.mp4", "https://files.catbox.moe/abc.mp4"),
        ('{"status":"success","data":{"url":"https://litter.catbox.moe/x.mp4"}}',
         "https://litter.catbox.moe/x.mp4"),
    ],
)
def test_a_link_is_read_out_of_either_answer_shape(answer, expected):
    """One host answers with a bare URL, the other with JSON around it."""
    from open_webui_openrouter_pipe.integrations.media_relay import _ENDPOINTS, _extract_url

    assert _extract_url(answer, _ENDPOINTS["catbox"].origins) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "ERROR: something went wrong",
        "{}",
        "{not json",
        "https://attacker.example/beacon.mp4",
        "http://files.catbox.moe/plaintext.mp4",
        "https://catbox.moe.attacker.example/x.mp4",
        '{"status":"success","data":{"url":"http://169.254.169.254/latest/meta-data/"}}',
    ],
)
def test_an_answer_carrying_no_link_is_not_mistaken_for_one(answer):
    """A link is only a link when the host that took the file is the one serving it.

    The last four are 200 answers carrying a perfectly well-formed URL somewhere else:
    the pipe hands whatever comes back to OpenRouter as the user's media, so an address
    the chosen host does not serve is not an answer, it is a redirection.
    """
    from open_webui_openrouter_pipe.integrations.media_relay import _ENDPOINTS, _extract_url

    assert _extract_url(answer, _ENDPOINTS["litterbox"].origins) == ""


@pytest.mark.asyncio
async def test_a_clip_reaches_the_request_as_a_link_rather_than_as_bytes(monkeypatch):
    """The whole point: what leaves the pipe is a URL, and the bytes went elsewhere.

    OpenRouter answers a base64 data URL under `video_url` with
    `400 Only HTTPS URLs are allowed`, so a request carrying one is a lost generation.
    """
    from open_webui_openrouter_pipe.integrations import video as module

    uploaded: list[tuple[str, str, int]] = []

    async def fake_relay(_session, blob, *, filename, mime, host, retention, max_bytes):
        uploaded.append((host, retention, len(blob)))
        return "https://litter.catbox.moe/abc123.mp4"

    monkeypatch.setattr(module, "relay_to_public_url", fake_relay)

    async def fake_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename="holiday.mp4", user_id=_OWNER.id)

    monkeypatch.setattr(module, "get_file_by_id", fake_file)
    monkeypatch.setattr(module, "infer_file_mime_type", lambda _obj: "video/mp4")

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    adapter.logger = logging.getLogger("media-relay-test")
    adapter._pipe = MagicMock()
    adapter._pipe._create_http_session = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)
        )
    )
    adapter._pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value="AAAAGGZ0eXBtcDQy")

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    valves.MEDIA_FILE_HOST_RETENTION = "24h"

    withheld: list[tuple[str, str]] = []
    refs = await adapter._encode_input_references(
        {"input_references": [{"id": "clip-1", "content_type": "video/mp4"}]},
        valves,
        withheld=withheld,
        user_obj=_OWNER,
    )

    assert withheld == [], withheld
    assert refs == [
        {"type": "video_url", "video_url": {"url": "https://litter.catbox.moe/abc123.mp4"}}
    ]
    assert "base64" not in str(refs), "the bytes travelled in the request after all"
    assert uploaded == [("litterbox", "24h", 12)], uploaded


_DECLARED = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_video_input_modalities.json").read_text()
)["input_modalities"]


@pytest.mark.parametrize(("model_id", "declared"), sorted(_DECLARED.items()))
@pytest.mark.parametrize("family", ["video", "audio"])
@pytest.mark.asyncio
async def test_a_reference_kind_is_offered_only_where_the_model_declares_it(
    monkeypatch, model_id, declared, family
):
    """OpenRouter answers `does not accept video input references` for 16 of the 23.

    Which models take a clip is published at `GET /models/{id}/endpoints` as
    `architecture.input_modalities`, and it is not in the video catalog the pipe lists
    from. Guessing it from `allowed_passthrough_parameters` is wrong in both directions:
    `alibaba/wan-2.7` publishes `video` and `videos` and is refused, while `runway/aleph-2`
    publishes neither and is accepted. Every model is driven so a rule that answers the
    same way for all of them fails.
    """
    from open_webui_openrouter_pipe.integrations import video as module

    mime = {"video": "video/mp4", "audio": "audio/mpeg"}[family]
    payload = {"video": b"\x00\x00\x00\x18ftypmp42", "audio": b"ID3\x04tone"}[family]

    async def fake_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename=file_id, user_id=_OWNER.id)

    monkeypatch.setattr(module, "get_file_by_id", fake_file)
    monkeypatch.setattr(
        module,
        "infer_file_mime_type",
        lambda obj: "image/png" if str(obj.id).startswith("pair") else mime,
    )

    async def fake_relay(_session, _blob, **_kwargs):
        return "https://files.example/asset"

    monkeypatch.setattr(module, "relay_to_public_url", fake_relay)

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    adapter.logger = logging.getLogger("declared-modalities")
    adapter._pipe = MagicMock()
    adapter._pipe._create_http_session = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)
        )
    )
    import base64

    PAIR = base64.b64encode(
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=red:s=512x512", "-frames:v", "1", "-f", "image2", "-c:v", "png", "-"],
            capture_output=True, check=True,
        ).stdout
    ).decode()

    async def _bytes(file_obj, *_args, **_kwargs):
        return PAIR if str(file_obj.id).startswith("pair") else base64.b64encode(payload).decode()

    adapter._pipe._file_gateway.read_file_record_base64 = AsyncMock(side_effect=_bytes)

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True

    items = [{"id": "ref-1", "content_type": mime}]
    if family == "audio":
        items.append({"id": "pair-1", "content_type": "image/png"})

    withheld: list[tuple[str, str]] = []
    refs = await adapter._encode_input_references(
        {"model_id": model_id, "input_references": items},
        valves,
        withheld=withheld,
        video_model={"id": model_id, "input_modalities": declared},
        user_obj=_OWNER,
    )

    accepted = family in declared
    carried = [item["type"] for item in refs]
    assert (f"{family}_url" in carried) is accepted, (
        f"{model_id} declares {declared}; the request carried {carried} and withheld {withheld}"
    )
    if not accepted:
        assert any("does not take" in reason for _name, reason in withheld), withheld


@pytest.mark.parametrize(("model_id", "declared"), sorted(_DECLARED.items()))
def test_no_panel_draws_a_reference_control_its_model_refuses(model_id, declared):
    """A control the model will not accept must not appear on the panel at all.

    Withholding the value at send time is the second line; the first is never offering it.
    `alibaba/wan-2.7` publishes `video`, `videos` and `audio` in its passthrough list and
    OpenRouter refuses all three, so a panel built from that list alone drew three boxes
    that could only ever produce a rejection.
    """
    import json as _json
    from pathlib import Path as _Path

    from open_webui_openrouter_pipe.filters.video_filter_renderer import build_video_filter_spec

    catalog = {
        item["id"]: item
        for item in _json.loads(
            (_Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text()
        )["data"]
    }
    model = catalog.get(model_id)
    if model is None:
        pytest.skip(f"{model_id} is not in the recorded video catalog")

    by_kind = {"video": ("video", "videos"), "audio": ("audio",), "image": ("images", "last_image")}
    published = set(model.get("allowed_passthrough_parameters") or [])

    with_declaration = dict(model, input_modalities=declared)
    drawn = set(build_video_filter_spec(model_id, with_declaration).allowed_params)

    for kind, names in by_kind.items():
        refused = published & set(names)
        if kind in declared or not refused:
            continue
        assert not (drawn & refused), (
            f"{model_id} declares {sorted(declared)} yet the panel still draws "
            f"{sorted(drawn & refused)}"
        )

    assert drawn <= published, (
        f"{model_id}: the panel invented {sorted(drawn - published)}"
    )
    keepable = {
        name
        for kind, names in by_kind.items()
        if kind in declared
        for name in names
    } | {n for n in published if not any(n in names for names in by_kind.values())}
    assert drawn == published & keepable, (
        f"{model_id}: expected {sorted(published & keepable)}, drew {sorted(drawn)}"
    )


@pytest.mark.asyncio
async def test_the_catalog_loader_puts_the_declared_kinds_where_the_gates_read_them(monkeypatch):
    """The gates were correct and inert: nothing ever gave them the declaration.

    `/videos/models` carries no `architecture` block, so the entry the registry stores has
    no `input_modalities` and both gates fell through to "nothing declared, allow
    everything". Every test fed the field in by hand, so the wiring was never exercised
    and deleting it changed no result. This drives the real loader.
    """
    import json as _json
    from pathlib import Path as _Path
    from unittest.mock import AsyncMock as _AsyncMock

    from open_webui_openrouter_pipe.integrations import video_catalog
    from open_webui_openrouter_pipe.integrations.video import _declared_input_kinds
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    fixtures = _Path(__file__).parent / "fixtures"
    catalog = _json.loads((fixtures / "video_models_catalog.json").read_text())["data"]
    declared = _json.loads(
        (fixtures / "openrouter_video_input_modalities.json").read_text()
    )["input_modalities"]

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def list_models(self):
            return [dict(item) for item in catalog]

        async def model_modalities(self, model_id):
            return list(declared.get(model_id, []))

    monkeypatch.setattr(video_catalog, "OpenRouterVideoClient", _Client)

    monkeypatch.setattr(OpenRouterModelRegistry, "_specs", {})
    monkeypatch.setattr(OpenRouterModelRegistry, "_id_map", {})
    monkeypatch.setattr(OpenRouterModelRegistry, "_models", [])
    monkeypatch.setattr(OpenRouterModelRegistry, "_last_video_attempt", 0.0)

    await video_catalog.ensure_video_catalog_loaded(
        MagicMock(),
        valves=SimpleNamespace(
            ENABLE_VIDEO_GENERATION=True,
            BASE_URL="https://openrouter.ai/api/v1",
            HTTP_REFERER_OVERRIDE="",
        ),
        api_key="sk-test",
        logger=logging.getLogger("catalog-wiring"),
        cache_seconds=0,
    )

    checked = 0
    for model_id, kinds in sorted(declared.items()):
        spec = OpenRouterModelRegistry.spec(model_id)
        if not isinstance(spec, dict) or not spec:
            continue
        stored = spec.get("video_model") or {}
        checked += 1
        assert stored.get("input_modalities") == list(kinds), (
            f"{model_id}: the registry stored {stored.get('input_modalities')!r}"
        )
        takes = _declared_input_kinds(stored)
        for family in ("video", "audio", "image"):
            assert takes(family) is (family in kinds), (
                f"{model_id}: the gate says {family}={takes(family)} for a model declaring {kinds}"
            )
    assert checked >= 20, f"only {checked} models were reached, so this gate asserts little"


@pytest.mark.asyncio
async def test_a_model_whose_declaration_cannot_be_read_keeps_every_control():
    """A read that failed is not a declaration of nothing.

    Refusing every reference kind because one HTTP call failed would silently disable
    attachments fleet-wide on a bad afternoon.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    from open_webui_openrouter_pipe.integrations import video_catalog
    from open_webui_openrouter_pipe.integrations.video import _declared_input_kinds

    client = MagicMock()
    client.model_modalities = _AsyncMock(return_value=[])
    models = [{"id": "vendor/model", "allowed_passthrough_parameters": ["video"]}]

    await video_catalog._attach_declared_input_modalities(client, models, logging.getLogger("t"))

    assert "input_modalities" not in models[0]
    takes = _declared_input_kinds(models[0])
    assert all(takes(kind) for kind in ("video", "audio", "image"))


def _serialised(form) -> bytes:
    payload = form()
    captured = bytearray()

    class _Sink:
        async def write(self, chunk):
            captured.extend(chunk)

    asyncio.run(payload.write(_Sink()))
    return bytes(captured)


class _Wire:
    """Records every POST aiohttp actually makes, and answers from a scripted queue.

    Registered with aioresponses as a callback, so the subject builds a real request
    through a real ClientSession and the multipart bytes below are the bytes that
    would have gone out.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": str(url), **kwargs})
        status, text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return CallbackResult(status=status, body=text)

    @property
    def bodies(self):
        return [_serialised(call["data"]) for call in self.calls]

    @property
    def content_types(self):
        return [call["data"]().headers["Content-Type"] for call in self.calls]


async def _relayed(answers, *, host="litterbox", retention="1h", blob=b"\x00\x01clip", sleeps=None):
    wire = _Wire(answers)
    async with aiohttp.ClientSession() as session:
        with aioresponses() as mocked:
            mocked.post(_ENDPOINTS[host][0], callback=wire, repeat=True)
            paused = []

            async def _record(seconds):
                paused.append(seconds)

            with patch.object(media_relay.asyncio, "sleep", _record):
                try:
                    link = await relay_to_public_url(
                        session, blob, filename="clip.mp4", mime="video/mp4",
                        host=host, retention=retention, max_bytes=0,
                    )
                except MediaRelayError as exc:
                    link = exc
            if sleeps is not None:
                sleeps.extend(paused)
    return link, wire


@pytest.mark.parametrize("retention", ["12h", "72h"])
def test_the_retention_an_operator_chose_is_the_one_on_the_wire(retention):
    """The valve is the whole feature: a link that outlives the job, and no longer.

    Parametrised over two values because a hardcoded '1h' in production satisfies neither.
    """
    link, wire = asyncio.run(
        _relayed([(200, "https://litter.catbox.moe/abc.mp4")], retention=retention)
    )
    assert link == "https://litter.catbox.moe/abc.mp4"
    sent = wire.bodies[0]
    assert b'name="time"' in sent
    assert retention.encode() in sent.split(b'name="time"')[1][:64]


def test_a_retention_the_host_does_not_publish_falls_back_to_the_shortest():
    """An unknown value must not be sent verbatim -- the host would reject the upload."""
    _, wire = asyncio.run(_relayed([(200, "https://l.moe/a.mp4")], retention="forever"))
    after = wire.bodies[0].split(b'name="time"')[1][:64]
    assert b"forever" not in after
    assert RELAY_RETENTIONS[0].encode() in after


def test_the_permanent_host_is_never_asked_for_a_retention():
    """catbox has no expiry field; sending one is an error on its API, not a no-op."""
    _, wire = asyncio.run(_relayed([(200, "https://files.catbox.moe/a.mp4")], host="catbox"))
    assert b'name="time"' not in wire.bodies[0]
    assert b'name="reqtype"' in wire.bodies[0]


def test_the_file_reaches_the_host_intact_inside_a_declared_multipart_part():
    """Framing is the whole upload: a mangled boundary is a silent corrupt file."""
    blob = bytes(range(256)) * 4
    _, wire = asyncio.run(_relayed([(200, "https://l.moe/a.mp4")], blob=blob))
    sent = wire.bodies[0]
    boundary = wire.content_types[0].split("boundary=")[1]
    assert sent.startswith(f"--{boundary}\r\n".encode())
    assert sent.rstrip(b"\r\n").endswith(f"--{boundary}--".encode())
    assert b'filename="clip.mp4"' in sent
    assert b"Content-Type: video/mp4" in sent
    assert blob in sent


@pytest.mark.parametrize("status", [500, 502, 429])
def test_a_host_having_a_bad_moment_is_retried_and_can_still_succeed(status):
    """429 and 5xx are the transient answers; giving up on them loses a working upload."""
    sleeps = []
    link, wire = asyncio.run(
        _relayed([(status, "busy"), (200, "https://litter.catbox.moe/ok.mp4")], sleeps=sleeps)
    )
    assert link == "https://litter.catbox.moe/ok.mp4"
    assert len(wire.calls) == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize("status", [400, 413])
def test_a_refusal_the_host_will_repeat_is_not_retried(status):
    """Re-posting a 200MB file to earn the same 413 wastes minutes of the user's wait."""
    link, wire = asyncio.run(_relayed([(status, "no")]))
    assert isinstance(link, MediaRelayError)
    assert str(status) in str(link)
    assert len(wire.calls) == 1


def test_a_host_that_stays_down_is_given_three_tries_with_a_widening_pause():
    sleeps = []
    link, wire = asyncio.run(_relayed([(503, "down")], sleeps=sleeps))
    assert isinstance(link, MediaRelayError)
    assert len(wire.calls) == 3
    assert sleeps == [2.0, 4.0]


def test_a_success_carrying_no_link_is_reported_with_what_the_host_said():
    """A 200 whose body is an HTML interstitial is exactly how tmpfiles failed."""
    link, wire = asyncio.run(_relayed([(200, "<html>are you a robot</html>")]))
    assert isinstance(link, MediaRelayError)
    assert "without a link" in str(link)
    assert "are you a robot" in str(link)
    assert len(wire.calls) == 1


@pytest.mark.parametrize(
    "hostile",
    [
        'clip.mp4"\r\nContent-Disposition: form-data; name="reqtype"\r\n\r\ndeletefiles\r\n--x\r\nX: y',
        'a\r\nContent-Type: text/html\r\n\r\n<script>alert(1)</script>\r\n--x--\r\nb.mp4',
    ],
)
def test_a_filename_cannot_write_its_own_headers_into_the_upload(hostile):
    """Open WebUI lets any verified user rename a file to anything at all.

    `POST /api/v1/files/{id}/rename` assigns the string straight onto the row with no
    validation, so the filename is attacker-chosen. Interpolating it into a header
    position let it close the quote and append fields -- a second `reqtype` reaching a
    file host is a different API call than the one this pipe intended to make.
    """
    form = media_relay._form(
        {"reqtype": "fileupload"}, hostile, b"CLIP", "video/mp4", "fileToUpload"
    )
    body = _serialised(form)
    boundary = form().headers["Content-Type"].split("boundary=")[1].encode()

    parts = [chunk for chunk in body.split(b"--" + boundary) if chunk.strip() not in (b"", b"--")]
    assert len(parts) == 2, f"the filename opened a third part: {body!r}"
    for part in parts:
        block, _, _ = part.lstrip(b"\r\n").partition(b"\r\n\r\n")
        headers = [line for line in block.split(b"\r\n") if line]
        assert sum(1 for h in headers if h.startswith(b"Content-Disposition:")) == 1, headers
        assert not any(h.startswith(b"X:") for h in headers), headers
        assert not any(h == b"Content-Type: text/html" for h in headers), headers
    assert body.count(b"CLIP") == 1


def test_an_ordinary_filename_still_arrives_intact():
    body = _serialised(
        media_relay._form({"reqtype": "fileupload"}, "holiday clip.mp4", b"CLIP", "video/mp4", "fileToUpload")
    )

    assert b"holiday" in body and b".mp4\"" in body, body
    assert b"Content-Type: video/mp4" in body
    assert b"\r\n" not in body.split(b'filename="')[1].split(b'"')[0]


@pytest.mark.parametrize(
    ("used_host", "expected_span"),
    [("litterbox", "an hour later"), ("catbox", "stays there for good")],
)
def test_the_notice_names_the_host_that_actually_took_the_file(used_host, expected_span):
    """The fallback is exactly when the privacy statement changes.

    An operator who picked litterbox and allowed a fallback gets catbox when litterbox is
    down -- a file that would have deleted itself within the hour instead stays up until a
    human removes it. Reading the valve rather than the outcome told the user the opposite.
    """
    valves = Valves()
    valves.MEDIA_FILE_HOST = "litterbox"
    valves.MEDIA_FILE_HOST_RETENTION = "1h"

    line = VideoGenerationAdapter._file_host_notice(valves, {("video", used_host)})

    assert used_host in line, line
    assert expected_span in line, line
    if used_host != "litterbox":
        assert "litterbox" not in line, line


def _adapter_with_files(records, reads, uploads):
    """An adapter whose storage reader and relay both record every call.

    Both are seams below `_encode_input_references`, so the subject's own body runs.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    from open_webui_openrouter_pipe.integrations import video as video_module

    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-order")

    async def _read(file_obj, _chunk, _cap, user=None):
        reads.append(file_obj.id)
        return records[file_obj.id]["b64"]

    pipe._file_gateway.read_file_record_base64 = _read
    adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("relay-order"))

    async def _relay(self, valves, b64, *, filename, mime, family):
        uploads.append((filename, mime, family))
        return f"https://files.example/{family}.bin", "litterbox"

    return adapter, video_module, _relay, _AsyncMock


@pytest.mark.asyncio
async def test_a_sound_file_with_nothing_to_pair_with_is_never_uploaded(monkeypatch):
    """OpenRouter refuses a lone sound reference, so uploading one publishes it for nothing.

    Both hosts take the file anonymously and neither offers the caller a delete, so on
    catbox a user's private audio would stay public permanently in exchange for a request
    that was never going to be sent.
    """
    import base64 as _base64

    from open_webui_openrouter_pipe.integrations import video as video_module

    records = {"aud-1": {"b64": _base64.b64encode(b"ID3sound").decode()}}
    reads: list[str] = []
    uploads: list[tuple[str, str, str]] = []
    adapter, module, relay, _AsyncMock = _adapter_with_files(records, reads, uploads)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename="voice.mp3", user_id=_OWNER.id)

    monkeypatch.setattr(module, "get_file_by_id", _get_file)
    monkeypatch.setattr(module, "infer_file_mime_type", lambda _f: "audio/mpeg")
    monkeypatch.setattr(
        VideoGenerationAdapter, "_relay_reference", relay, raising=True
    )

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    withheld: list[tuple[str, str]] = []

    encoded = await adapter._encode_input_references(
        {"input_references": [{"id": "aud-1"}], "model_id": "bytedance/seedance-2.0"},
        valves,
        withheld=withheld,
        video_model={"id": "bytedance/seedance-2.0", "input_modalities": ["audio", "image"]},
        relayed=set(),
        companions=False,
        user_obj=_OWNER,
    )

    assert encoded == []
    assert uploads == [], f"the file was published before it was discarded: {uploads}"
    assert withheld and "alongside" in withheld[0][1]


@pytest.mark.asyncio
async def test_a_sound_file_is_kept_when_a_picture_was_attached_as_a_frame(monkeypatch):
    """The picture the user attached is the companion, wherever the filter routed it.

    Counting only the reference list meant a first-frame picture did not count, so the
    user was told their sound file had nothing to pair with while it sat in the same
    request.
    """
    import base64 as _base64

    from open_webui_openrouter_pipe.integrations import video as video_module

    records = {"aud-1": {"b64": _base64.b64encode(b"ID3sound").decode()}}
    reads: list[str] = []
    uploads: list[tuple[str, str, str]] = []
    adapter, module, relay, _AsyncMock = _adapter_with_files(records, reads, uploads)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename="voice.mp3", user_id=_OWNER.id)

    monkeypatch.setattr(module, "get_file_by_id", _get_file)
    monkeypatch.setattr(module, "infer_file_mime_type", lambda _f: "audio/mpeg")
    monkeypatch.setattr(VideoGenerationAdapter, "_relay_reference", relay, raising=True)

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    withheld: list[tuple[str, str]] = []

    encoded = await adapter._encode_input_references(
        {"input_references": [{"id": "aud-1"}], "model_id": "bytedance/seedance-2.0"},
        valves,
        withheld=withheld,
        video_model={"id": "bytedance/seedance-2.0", "input_modalities": ["audio", "image"]},
        relayed=set(),
        companions=True,
        user_obj=_OWNER,
    )

    assert [entry["type"] for entry in encoded] == ["audio_url"]
    assert uploads == [("voice.mp3", "audio/mpeg", "audio")]
    assert withheld == []


@pytest.mark.parametrize(
    ("relay_on", "expected_reads"), [(False, []), (True, ["clip-1"])]
)
@pytest.mark.asyncio
async def test_a_clip_that_cannot_be_sent_is_never_read_out_of_storage(
    monkeypatch, relay_on, expected_reads
):
    """A clip may only travel as a link, so with no file host there is nothing to read.

    The default cap is hundreds of megabytes; reading and base64-encoding one only to
    discard it costs that much memory per request, and the concurrency limit is taken
    after this point, so nothing bounds how many do it at once.
    """
    import base64 as _base64

    from open_webui_openrouter_pipe.integrations import video as video_module

    records = {"clip-1": {"b64": _base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()}}
    reads: list[str] = []
    uploads: list[tuple[str, str, str]] = []
    adapter, module, relay, _AsyncMock = _adapter_with_files(records, reads, uploads)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename="clip.mp4", user_id=_OWNER.id)

    monkeypatch.setattr(module, "get_file_by_id", _get_file)
    monkeypatch.setattr(module, "infer_file_mime_type", lambda _f: "video/mp4")
    monkeypatch.setattr(VideoGenerationAdapter, "_relay_reference", relay, raising=True)

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = relay_on
    withheld: list[tuple[str, str]] = []

    await adapter._encode_input_references(
        {"input_references": [{"id": "clip-1"}], "model_id": "runway/aleph-2"},
        valves,
        withheld=withheld,
        video_model={"id": "runway/aleph-2", "input_modalities": ["video", "image"]},
        relayed=set(),
        companions=True,
        user_obj=_OWNER,
    )

    assert reads == expected_reads


class _RelaySession:
    """A real aiohttp session wrapped as the pipe's `_create_http_session` context."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


def _relay_adapter(session):
    pipe = MagicMock()
    pipe.logger = logging.getLogger("relay-failover")
    pipe._create_http_session = lambda *_a, **_k: _RelaySession(session)
    return VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("relay-failover"))


@pytest.mark.parametrize("chosen", ["litterbox", "catbox"])
@pytest.mark.asyncio
async def test_the_second_file_host_is_tried_only_when_the_operator_turned_it_on(chosen):
    """`USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN` is the difference between a clip and an error.

    Parametrised over both hosts as the operator's first choice, so the fallback cannot be
    a hardcoded "try catbox": whichever host was chosen, the OTHER one is the second try
    and neither is tried twice.

    With the valve off, the same outage must end as a refusal that names what went wrong,
    because silently reaching a host the operator did not pick publishes a user's file
    somewhere they did not agree to.
    """
    import base64

    other = next(name for name in RELAY_HOSTS if name != chosen)
    blob = base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()

    async with aiohttp.ClientSession() as session:
        adapter = _relay_adapter(session)
        valves = Valves()
        valves.MEDIA_FILE_HOST = chosen
        valves.USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN = True
        valves.MEDIA_FILE_HOST_MAX_SIZE_MB = 200

        with aioresponses() as http:
            http.post(_ENDPOINTS[chosen][0], status=500, body="down", repeat=True)
            http.post(
                _ENDPOINTS[other][0], status=200, body=f"https://files.catbox.moe/{other}.mp4"
            )
            with patch.object(media_relay.asyncio, "sleep", AsyncMock()):
                link, used = await adapter._relay_reference(
                    valves, blob, filename="clip.mp4", mime="video/mp4", family="video"
                )

        assert used == other, f"the operator's fallback did not reach {other}"
        assert link == f"https://files.catbox.moe/{other}.mp4"

        valves.USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN = False
        with aioresponses() as http:
            http.post(_ENDPOINTS[chosen][0], status=500, body="down", repeat=True)
            http.post(
                _ENDPOINTS[other][0], status=200, body=f"https://files.catbox.moe/{other}.mp4",
                repeat=True
            )
            with patch.object(media_relay.asyncio, "sleep", AsyncMock()):
                with pytest.raises(VideoGenerationError) as failed:
                    await adapter._relay_reference(
                        valves, blob, filename="clip.mp4", mime="video/mp4", family="video"
                    )
            reached = sorted({str(url) for (_method, url) in http.requests})

        assert reached == [_ENDPOINTS[chosen][0]], (
            f"with the valve off the upload still reached {reached}"
        )
        assert "video" in str(failed.value) and chosen in str(failed.value), (
            f"the refusal does not say what failed: {failed.value}"
        )


@pytest.mark.asyncio
async def test_a_clip_that_is_not_readable_is_refused_before_any_host_sees_it():
    """Bytes that are not base64 would upload as whatever `b64decode` salvaged."""
    async with aiohttp.ClientSession() as session:
        adapter = _relay_adapter(session)
        valves = Valves()
        valves.MEDIA_FILE_HOST = "litterbox"

        with aioresponses() as http:
            with pytest.raises(VideoGenerationError) as failed:
                await adapter._relay_reference(
                    valves, "AAAAA", filename="clip.mp4", mime="video/mp4", family="video"
                )
            assert list(http.requests) == [], "an unreadable clip still reached a host"
        assert "video" in str(failed.value)


@pytest.mark.parametrize("host", list(RELAY_HOSTS))
def test_an_empty_file_is_refused_rather_than_published_as_a_zero_byte_link(host):
    """Both hosts accept a zero-byte upload and hand back a permanent link to nothing.

    The provider then fetches it, fails, and the user is billed for a refusal caused by a
    file the pipe could have declined for free. Checked on both hosts because the guard
    sits above the host split.
    """
    async def _run():
        async with aiohttp.ClientSession() as session:
            with aioresponses() as http:
                http.post(_ENDPOINTS[host][0], status=200, body="https://x.test/nothing")
                with pytest.raises(MediaRelayError) as refused:
                    await relay_to_public_url(
                        session, b"", filename="clip.mp4", mime="video/mp4",
                        host=host, retention="1h", max_bytes=200 * 1024 * 1024,
                    )
                assert list(http.requests) == [], "an empty file still reached the host"
        return str(refused.value)

    assert "empty" in asyncio.run(_run())


@pytest.mark.parametrize(
    "failure",
    [aiohttp.ClientConnectionError("connection refused"), TimeoutError("timed out")],
)
def test_a_host_that_cannot_be_reached_at_all_is_reported_by_name(failure):
    """A DNS or TCP failure is the common file-host outage, and it raises rather than answers.

    The retry loop only ever saw HTTP statuses in the suite; the arm that turns a
    transport failure into a sentence could be deleted and the user would get
    "litterbox did not accept the file" for a host that was never contacted.

    Two failure types, so an arm that catches only one is not enough.
    """
    async def _run():
        async with aiohttp.ClientSession() as session:
            with aioresponses() as http:
                http.post(_ENDPOINTS["litterbox"][0], exception=failure, repeat=True)
                with patch.object(media_relay.asyncio, "sleep", AsyncMock()):
                    with pytest.raises(MediaRelayError) as refused:
                        await relay_to_public_url(
                            session, b"\x00\x01clip", filename="clip.mp4", mime="video/mp4",
                            host="litterbox", retention="1h", max_bytes=0,
                        )
                attempts = len(next(iter(http.requests.values())))
        return str(refused.value), attempts

    said, attempts = asyncio.run(_run())
    assert "litterbox could not be reached" in said, said
    assert str(failure) in said, f"the reason the host gave is missing: {said}"
    assert attempts == 3, f"a transport failure was retried {attempts} time(s), not three"


@pytest.mark.parametrize("template", ["Your file goes to {hosst}.", "Your file goes to {0}."])
def test_a_notice_whose_placeholder_the_pipe_cannot_fill_is_shown_as_written(template):
    """An operator's typo in a chat notice must not take the video request down with it.

    `str.format` raises on an unknown name and on a positional field, and this notice is
    built while a job is being submitted. Showing the template as typed is a visible
    prompt to fix it; raising loses the generation.
    """
    valves = Valves()
    valves.FILE_HOST_NOTICE = template

    assert VideoGenerationAdapter._file_host_notice(
        valves, {("video", "litterbox")}
    ) == template


@pytest.mark.parametrize(
    ("width", "height", "sent"), [(320, 240, False), (1280, 720, True)]
)
@pytest.mark.asyncio
async def test_a_clip_below_the_models_pixel_floor_is_dropped_before_it_is_published(
    monkeypatch, width, height, sent
):
    """The floor decides in the encoder, and a dropped clip must never reach a host.

    Both hosts publish anonymously and offer no delete, so uploading a clip the request
    was always going to leave out puts a user's video on the public internet for nothing.
    The two frame sizes straddle the floor, so a rule that always drops -- or one that
    never does -- fails one row.
    """
    import base64 as _base64

    from open_webui_openrouter_pipe.integrations import video as video_module
    from open_webui_openrouter_pipe.integrations.video import _INPUT_PIXEL_FLOORS
    from open_webui_openrouter_pipe.media.frame_extraction import VideoMetadata

    model_id = "bytedance/seedance-2.0"
    floor = _INPUT_PIXEL_FLOORS[model_id]
    assert (width * height >= floor.pixels) is sent, "this row is on the wrong side of the floor"

    records = {"clip-1": {"b64": _base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()}}
    reads: list[str] = []
    uploads: list[tuple[str, str, str]] = []
    adapter, module, relay, _AsyncMock = _adapter_with_files(records, reads, uploads)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename="clip.mp4", user_id=_OWNER.id)

    async def _probe(_path):
        return VideoMetadata(
            duration_seconds=2.0, width=width, height=height, fps=24.0, has_audio=False
        )

    monkeypatch.setattr(module, "get_file_by_id", _get_file)
    monkeypatch.setattr(module, "infer_file_mime_type", lambda _f: "video/mp4")
    monkeypatch.setattr(video_module, "probe_video", _probe)
    monkeypatch.setattr(VideoGenerationAdapter, "_relay_reference", relay, raising=True)

    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    valves.SEND_VIDEO_VIA_FILE_HOST = True
    withheld: list[tuple[str, str]] = []

    encoded = await adapter._encode_input_references(
        {"input_references": [{"id": "clip-1"}], "model_id": model_id},
        valves,
        withheld=withheld,
        video_model={"id": model_id, "input_modalities": ["video", "image"]},
        relayed=set(),
        companions=True,
        user_obj=_OWNER,
    )

    assert (encoded != []) is sent, f"the clip produced {encoded!r}"
    assert (uploads != []) is sent, f"a clip that was never sent was published: {uploads}"
    assert (withheld == []) is sent, withheld
    if not sent:
        assert f"{width} by {height}" in withheld[0][1], withheld[0][1]
        assert f"{floor.pixels:,}" in withheld[0][1], withheld[0][1]
