# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportIndexIssue=false, reportCallIssue=false
"""URL schemes are case-insensitive (RFC 3986 3.1); the pipe decided them on raw prefixes.

Measured against the real ``transform_messages_to_input`` before the fix, with
``ALLOW_INSECURE_HTTP=False`` and ``_download_remote_url`` spied::

    'http://insecure.example.com/x.png'   downloads=[]  out=[]
    'HTTP://insecure.example.com/x.png'   downloads=[]  out=[{'image_url': 'HTTP://...'}]
    'https://secure.example.com/y.png'    downloads=['https://secure.example.com/y.png']
    'HTTPS://secure.example.com/y.png'    downloads=[]

Two invariants broke at once. ``ALLOW_INSECURE_HTTP``'s "disabled by default" never
consulted its own gate for ``HTTP://``, and the download of remote images -- which is
what keeps a third-party URL from being handed to OpenRouter to fetch itself -- never
fired for ``HTTPS://``, so the URL went upstream verbatim and OpenRouter fetched it.

Every case-varying test here parametrises over at least two spellings whose expected
answers differ, or over both a blocked and an allowed row, so no constant return value
in production satisfies the whole parametrisation.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import logging
import re

import pytest
from unittest.mock import AsyncMock
from yarl import URL

from open_webui_openrouter_pipe import ModelFamily, Pipe
from open_webui_openrouter_pipe.core.config import _select_openrouter_http_referer
from open_webui_openrouter_pipe.core.errors import StatusMessages
from open_webui_openrouter_pipe.core.url_scheme import (
    HTTP_SCHEMES,
    is_cleartext_http_url,
    is_http_or_https_url,
    is_inline_data_url,
    split_base64_data_url,
    url_scheme,
)
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from open_webui_openrouter_pipe.storage import multimodal as mm
from open_webui_openrouter_pipe.storage.owui_files import InlinedFile, is_internal_file_url

INSECURE_HOST = "insecure.example.test"
@pytest.fixture(autouse=True)
def _reset_model_specs():
    ModelFamily.set_dynamic_specs({})
    yield
    ModelFamily.set_dynamic_specs({})


# ── the predicate, held to the parser that actually dials ─────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "http://h/x",
        "HTTP://h/x",
        "Http://h/x",
        "hTtP://h/x",
        "https://h/x",
        "HTTPS://h/x",
        "hTTps://h/x",
        "data:image/png;base64,AAAA",
        "DATA:image/png;base64,AAAA",
        "/api/v1/files/abc/content",
        "ftp://h/x",
        "FTP://h/x",
        "nocolon",
        "",
        " http://h/x",
        "\thttp://h/x",
        "ht\ttp://h/x",
        "http\n://h/x",
        "\x00http://h/x",
        "http:/onlyoneslash",
    ],
)
def test_url_scheme_returns_what_yarl_would_dial(raw):
    """``_is_insecure_http_allowed`` takes its scheme from yarl, so the callers that
    decide whether to CALL it have to reach the same answer or the gate is bypassable.

    A hand-rolled ``.lower().startswith(...)`` does not: ``urlsplit`` (which yarl is
    built on) strips ASCII tab/CR/LF from anywhere in the URL and lstrips C0 controls
    and spaces, so ``'ht<TAB>tp://h/x'`` dials as http while every prefix test on earth
    reads it as something else.
    """
    assert url_scheme(raw) == URL(raw).scheme, (
        f"{raw!r} is dialled as {URL(raw).scheme!r} but classified as "
        f"{url_scheme(raw)!r}, so the gate that refuses it is never reached"
    )


@pytest.mark.parametrize(
    ("raw", "scheme", "cleartext"),
    [
        ("http://[::1", "http", True),
        ("HTTP://[::1", "http", True),
        ("https://[::1", "https", False),
        ("//[::1", "", False),
    ],
)
def test_an_unparseable_url_never_reads_as_harmless(raw, scheme, cleartext):
    """yarl raises on these, and ``_is_insecure_http_allowed`` turns that into a refusal.

    The predicate must therefore stay total AND still report ``http`` on the cleartext
    branch: answering "" for everything unparseable would route it past the gate and
    emit it verbatim, which is strictly worse than the raw prefix test it replaced. The
    expected values are written out rather than derived, because a formula here is a
    second implementation of the thing under test.
    """
    with pytest.raises(ValueError):
        URL(raw)
    assert url_scheme(raw) == scheme
    assert is_cleartext_http_url(raw) is cleartext


def test_the_two_scheme_questions_stay_distinct():
    """https is a web URL and is not cleartext; collapsing the two predicates would
    either send every https URL through the insecure-http gate or exempt every http one.
    """
    assert (is_cleartext_http_url("HTTP://h/x"), is_http_or_https_url("HTTP://h/x")) == (True, True)
    assert (is_cleartext_http_url("HTTPS://h/x"), is_http_or_https_url("HTTPS://h/x")) == (False, True)
    assert (is_cleartext_http_url("FTP://h/x"), is_http_or_https_url("FTP://h/x")) == (False, False)


def test_the_predicate_and_the_fetch_gate_name_the_same_schemes():
    """``is_http_or_https_url`` decides whether ``_download_remote_url`` is entered;
    ``_FETCHABLE_SCHEMES`` decides whether ``_is_insecure_http_allowed`` permits it.
    Two tables, one decision -- if they drift, a URL is accepted by one and refused by
    the other and the pipe silently falls back to handing OpenRouter the raw URL.
    """
    assert HTTP_SCHEMES == mm._FETCHABLE_SCHEMES


# ── the OWUI-internal classifier ──────────────────────────────────────────────


@pytest.mark.parametrize("scheme", ["http", "HTTP", "Http", "https", "HTTPS", "hTTps"])
def test_an_absolute_url_is_external_whatever_the_scheme_is_typed_as(scheme):
    """``is_internal_file_url`` was ``not url.startswith(("http://", "https://"))``, so
    ``HTTP://host/api/v1/files/<id>/content`` scored as an OWUI file reference.

    Measured consequence on the real transformer before the fix: the id was pulled out
    of the FOREIGN url and handed to OWUI storage, which raised
    ``RequiredInternalFileError: A referenced image (abc) could not be retrieved`` and
    killed the turn. It also exempted the url from the insecure-http gate at the image,
    file and video sites, all three of which read ``and not is_internal_file_url(...)``.
    """
    absolute = f"{scheme}://cdn.example.test/api/v1/files/abc/content"
    assert is_internal_file_url(absolute) is False, (
        f"{absolute!r} is a foreign host, not Open WebUI storage"
    )


@pytest.mark.parametrize(
    "relative",
    ["/api/v1/files/abc/content", "/api/v1/files/abc/content?token=Az09", "./api/v1/files/abc"],
)
def test_a_relative_owui_path_is_still_internal(relative):
    """The must-pass half: narrowing the classifier must not un-classify the paths it
    exists for, or every stored attachment stops being inlined.
    """
    assert is_internal_file_url(relative) is True


# ── the image path, end to end ────────────────────────────────────────────────


def _vision_pipe(pipe, *, allow=False, hosts=""):
    pipe.valves.ALLOW_INSECURE_HTTP = allow
    pipe.valves.ALLOW_INSECURE_HTTP_HOSTS = hosts
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision", "video"}}})
    return pipe


async def _transform_block(pipe, block, *, user=None):
    async def _emitter(_event):
        return None

    return await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": [block]}],
        model_id="vision-model",
        event_emitter=_emitter,
        user_obj=user,
    valves=pipe.valves,
    )


def _image_block(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _images(transformed):
    blocks = transformed[0]["content"] if transformed else []
    return [
        b for b in blocks
        if isinstance(b, dict) and b.get("type") in {"input_image", "image_url"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "HTTP", "Http", "hTtP"])
@pytest.mark.parametrize(
    ("allow", "hosts", "kept"),
    [(False, "", False), (True, INSECURE_HOST, True)],
)
async def test_cleartext_http_images_are_gated_however_the_scheme_is_typed(
    pipe_instance_async, monkeypatch, scheme, allow, hosts, kept
):
    """The gate is on the VALVE, not on the spelling of four letters.

    Both rows run for every spelling: the blocked row alone is satisfied by production
    that drops every image, and the allowlisted row alone by production that keeps
    every image. Neither constant survives the pair.
    """
    pipe = _vision_pipe(pipe_instance_async, allow=allow, hosts=hosts)
    # The allowlisted row is kept only if the download SUCCEEDS; a download the pipe could
    # not make is now refused in its own right, so stubbing it to None would leave this
    # asserting the fetch refusal rather than the cleartext gate.
    downloaded = AsyncMock(
        return_value={"data": b"\x89PNG", "mime_type": "image/png"}
    )
    monkeypatch.setattr(pipe._multimodal_handler, "_download_remote_url", downloaded)

    url = f"{scheme}://{INSECURE_HOST}/a.png"
    images = _images(await _transform_block(pipe, _image_block(url)))

    assert bool(images) is kept, (
        f"url={url} ALLOW_INSECURE_HTTP={allow}: expected the image to be "
        f"{'kept' if kept else 'dropped'}, got {images!r}. A cleartext URL reaching the "
        "outbound payload is fetched by OpenRouter over the network the operator "
        "disabled."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "HTTP", "Http"])
@pytest.mark.parametrize(
    ("allow", "hosts", "kept"),
    [(False, "", False), (True, INSECURE_HOST, True)],
)
async def test_a_foreign_url_carrying_the_owui_file_path_is_still_a_foreign_url(
    pipe_instance_async, monkeypatch, scheme, allow, hosts, kept
):
    """The two halves of the defect meet here, and fixing only one leaves it open.

    All three cleartext gates read ``and not is_internal_file_url(url)``. With the
    scheme test fixed but the classifier still case-sensitive, ``HTTP://host/api/v1/
    files/abc/content`` scores as internal, skips the gate AND the download, and is then
    fed to Open WebUI storage as file id ``abc`` -- which is where the measured
    ``RequiredInternalFileError`` came from. The allowlisted row is the one that catches
    it: an allowed URL must come back as a normal remote image, not as a lookup of
    somebody's stored file.
    """
    pipe = _vision_pipe(pipe_instance_async, allow=allow, hosts=hosts)
    inline = AsyncMock(return_value=None)
    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", inline)
    # As above: the allowlisted row is kept only because the download succeeded.
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url",
        AsyncMock(return_value={"data": b"\x89PNG", "mime_type": "image/png"}),
    )
    url = f"{scheme}://{INSECURE_HOST}/api/v1/files/abc/content"

    images = _images(await _transform_block(pipe, _image_block(url)))

    assert bool(images) is kept, f"url={url} ALLOW_INSECURE_HTTP={allow}: got {images!r}"
    inline.assert_not_awaited()


# ── the audio path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "data"),
    [
        ("http://h/a.mp3", ""),
        ("HTTP://h/a.mp3", ""),
        ("https://h/a.mp3", ""),
        ("HTTPS://h/a.mp3", ""),
        ("QUJDRA==", "QUJDRA=="),
    ],
)
async def test_a_remote_audio_url_is_refused_rather_than_read_as_base64(
    pipe_instance_async, payload, data
):
    """Audio takes base64 only. A URL that slipped the scheme test was handed to the
    base64 normaliser, which strips the characters it does not recognise and ships
    whatever is left as audio bytes. The last row is the real base64 that must survive.

    For the four URL rows there is now nothing to inspect: the refusal yields an audio
    block with no payload, T433 stops that block from being shipped, and the turn carries
    the one line that says so. The base64 row is the control -- it must still arrive as
    audio, which is what keeps "refuse everything" from satisfying this row.
    """
    pipe = _vision_pipe(pipe_instance_async)
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision", "audio"}}})

    transformed = await _transform_block(
        pipe, {"type": "input_audio", "input_audio": payload}
    )
    block = transformed[0]["content"][0]

    if data:
        assert block["input_audio"]["data"] == data, f"payload={payload!r} produced {block!r}"
        return
    assert block["type"] == "input_text", f"payload={payload!r} produced {block!r}"
    assert block["text"] == "[The user sent an empty message.]"


# ── the video path, end to end ────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheme", "host", "allow", "hosts", "forwarded"),
    [
        ("http", INSECURE_HOST, False, "", False),
        ("HTTP", INSECURE_HOST, False, "", False),
        ("http", INSECURE_HOST, True, INSECURE_HOST, True),
        ("HTTP", INSECURE_HOST, True, INSECURE_HOST, True),
    ],
)
async def test_cleartext_video_urls_are_gated_however_the_scheme_is_typed(
    pipe_instance_async, monkeypatch, scheme, host, allow, hosts, forwarded
):
    """The video branch reads its own copy of the cleartext test and emits an EMPTY
    ``video_url`` when the gate refuses, so the tell was the forwarded url, not a missing
    block. ``_is_safe_url`` is stubbed because it resolves DNS; the decision under test
    sits above it.

    The refused case no longer ships that empty ``video_url`` either -- a block with an
    empty url is a void attachment, and T433 stops void attachments from reaching the
    provider. The gate's decision is still what is under test: a refused URL produces no
    video block at all, and an allowed one forwards the URL unchanged. An allowed row
    that stopped forwarding, or a refused row that started, would both show up here.
    """
    pipe = _vision_pipe(pipe_instance_async, allow=allow, hosts=hosts)
    monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url", AsyncMock(return_value=True))
    url = f"{scheme}://{host}/clip.mp4"

    transformed = await _transform_block(pipe, {"type": "video_url", "video_url": {"url": url}})
    videos = [b for b in transformed[0]["content"] if b.get("type") == "video_url"]

    if forwarded:
        assert videos and videos[0]["video_url"]["url"] == url, (
            f"url={url} ALLOW_INSECURE_HTTP={allow}: got {videos!r}"
        )
        return
    assert not videos, (
        f"url={url} ALLOW_INSECURE_HTTP={allow}: a refused cleartext URL was sent as "
        f"{videos!r}"
    )
    assert transformed[0]["content"][0]["text"] == "[The user sent an empty message.]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "vetted"),
    [
        ("https://cdn.example.test/clip.mp4", True),
        ("HTTPS://cdn.example.test/clip.mp4", True),
        ("hTTps://cdn.example.test/clip.mp4", True),
        ("data:video/mp4;base64,AAAA", False),
    ],
)
async def test_a_remote_video_url_is_ssrf_checked_however_the_scheme_is_typed(
    pipe_instance_async, monkeypatch, url, vetted
):
    """The SSRF check sits behind its own copy of the is-this-remote test, and the
    fall-through below it forwards the URL unchecked.

    So an uppercase scheme skipped ``_is_safe_url`` entirely and a URL pointing at a
    private address was handed to the provider. The data-URL row is what stops a
    production edit that simply vets everything from satisfying this.
    """
    pipe = _vision_pipe(pipe_instance_async)
    safe = AsyncMock(return_value=True)
    monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url", safe)

    await _transform_block(pipe, {"type": "video_url", "video_url": {"url": url}})

    assert [c.args[0] for c in safe.await_args_list] == ([url] if vetted else []), (
        f"{url!r} bypassed the private-network check"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry", ["_download_remote_url", "_download_remote_url_streaming"]
)
@pytest.mark.parametrize(
    ("url", "fetched"),
    [
        ("https://cdn.example.test/a.png", True),
        ("HTTPS://cdn.example.test/a.png", True),
        ("http://cdn.example.test/a.png", True),
        ("HTTP://cdn.example.test/a.png", True),
        ("ftp://cdn.example.test/a.png", False),
        ("/api/v1/files/abc/content", False),
        ("data:image/png;base64,AAAA", False),
    ],
)
async def test_both_downloaders_admit_the_same_schemes(
    pipe_instance_async, monkeypatch, tmp_path, entry, url, fetched
):
    """Two entry points, one scheme question, and each had its own hand-written copy.

    The refusal is the whole reason ``_prepare_pinned_request`` -- which is where the
    SSRF pin and the insecure-http gate live -- is never reached for a non-web scheme,
    so the spy sits on that boundary rather than on the socket. The refused rows are
    what keep the admitted rows from being satisfied by "always fetch".
    """
    pipe = pipe_instance_async
    reached: list[str] = []

    async def _spy(requested, *args, **kwargs):
        reached.append(requested)
        return None

    monkeypatch.setattr(pipe._multimodal_handler, "_prepare_pinned_request", _spy)
    call = getattr(pipe._multimodal_handler, entry)
    if entry == "_download_remote_url_streaming":
        assert await call(url, tmp_path / "out.bin") is None
    else:
        assert await call(url) is None

    assert reached == ([url] if fetched else []), (
        f"{entry}({url!r}) reached the pinned-request boundary: {reached!r}"
    )


# ── obfuscated schemes ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "ht\ttp://{host}/a.png",
        "http\n://{host}/a.png",
        " http://{host}/a.png",
        "\x00http://{host}/a.png",
    ],
)
async def test_a_scheme_split_by_control_characters_still_reaches_the_gate(
    pipe_instance_async, monkeypatch, raw
):
    """``urlsplit`` -- and therefore yarl, and therefore aiohttp -- deletes ASCII
    tab/CR/LF from anywhere in a URL and lstrips C0 controls and spaces before reading
    the scheme. A case-insensitive PREFIX test would still have read every one of these
    as scheme-less and passed it straight through.
    """
    pipe = _vision_pipe(pipe_instance_async)
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url", AsyncMock(return_value=None)
    )
    url = raw.format(host=INSECURE_HOST)

    assert URL(url).scheme == "http", "this input is not the http URL the test claims"
    assert _images(await _transform_block(pipe, _image_block(url))) == [], (
        f"{url!r} dials as cleartext http and was forwarded anyway"
    )


# ── the referer valve ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("override", "honoured"),
    [
        ("https://site.example.test", True),
        ("HTTPS://site.example.test", True),
        ("http://site.example.test", True),
        ("HTTP://site.example.test", True),
        ("site.example.test", False),
        ("", False),
    ],
)
def test_the_referer_override_is_accepted_whatever_the_scheme_is_typed_as(override, honoured):
    """Two copies of this predicate exist -- ``_select_openrouter_http_referer`` decides
    whether to USE the override, ``Pipe.pipe`` decides whether to WARN about it -- and
    they have to agree. An operator typing ``HTTPS://`` got warned and silently fell
    back to the pipe's default referer.
    """
    valves = Pipe.Valves(HTTP_REFERER_OVERRIDE=override)
    selected = _select_openrouter_http_referer(valves)
    assert (selected == override) is honoured, (
        f"HTTP_REFERER_OVERRIDE={override!r} selected {selected!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "warns"),
    [
        ("https://site.example.test", False),
        ("HTTPS://site.example.test", False),
        ("http://site.example.test", False),
        ("HTTP://site.example.test", False),
        ("not-a-url", True),
    ],
)
async def test_the_referer_warning_agrees_with_the_referer_selection(
    pipe_instance_async, monkeypatch, override, warns
):
    """The second copy: ``Pipe.pipe`` decides whether to TELL the operator the override
    is unusable, and ``_select_openrouter_http_referer`` decides whether to use it.

    When these disagreed, an operator typing ``HTTPS://`` was told the value was
    rejected. Driving the real entry point is what proves the call site consults the
    predicate, rather than that the predicate exists.
    """
    pipe = pipe_instance_async
    pipe.valves.HTTP_REFERER_OVERRIDE = override
    notifications: list[tuple[str, str]] = []

    async def _spy(_emitter, message, *args, **kwargs):
        notifications.append((kwargs.get("level", "info"), str(message)))

    monkeypatch.setattr(pipe._event_emitter_handler, "_emit_notification", _spy)

    async def _emitter(_event):
        return None

    with contextlib.suppress(Exception):
        await pipe.pipe(
            body={"stream": False, "messages": [{"role": "user", "content": "hi"}]},
            __user__={},
            __request__=None,
            __event_emitter__=_emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
        )

    complaints = [n for n in notifications
                  if n[0] == "warning" and "HTTP_REFERER_OVERRIDE" in n[1]]
    assert bool(complaints) is warns, (
        f"HTTP_REFERER_OVERRIDE={override!r} produced {notifications!r}"
    )
