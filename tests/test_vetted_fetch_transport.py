"""What the pipe connects to, as opposed to what it checked.

The address gate used to be applied to a STRING and the request then handed to a client
that resolves the name again and follows `Location` on its own. Two things follow, and
both were reproduced against real sockets before this file existed:

- A catalog-supplied `https://cdn.example.com/icon.png` that answers `302` to
  `http://127.0.0.1:PORT/secret.png` returned the loopback body as a data URL. The check
  saw the CDN; the process fetched the internal service.
- With no redirect at all, a name that resolves public for `getaddrinfo` and loopback
  for the connector reached the loopback server. The check and the connection asked two
  different resolvers and got two different answers.

So the property here is not "the URL was checked". It is: every address the process
actually dials was validated, at every hop, and the socket went to the address that was
validated rather than to whatever the name resolves to at connect time.

Both callers are driven -- the model-icon download and the release-asset download --
because the point of a single transport is that neither can drift from the other.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import time
import tracemalloc
from typing import Any

import aiohttp
import pytest
import yarl
from aiohttp import web

from open_webui_openrouter_pipe.storage import multimodal as mm
from open_webui_openrouter_pipe.storage.multimodal import (
    UnfetchableAddress,
    _MAX_VETTED_REDIRECTS,
)
from tests.vetting_helpers import (
    STUB_PUBLIC_IP,
    LocalServer,
    address_routed_to_loopback,
    counted_dns,
    dns_answering,
    single_san_cert,
    public_ip_routed_to_loopback,
    rebinding_dns,
    stalled_dns,
    vetting,
    vetting_handler,
)

CDN = "cdn.example.com"
ORIGIN = "origin.example.com"


def _us():
    """The dashboard plugin's updater, or a skip when this artifact has no plugins.

    Importing it at module scope aborts collection under the two `--no-plugins` bundles,
    which takes the whole suite with it. A module-level `importorskip` fixes the abort by
    skipping every test in the file -- including the fifteen that drive `_vetted_get` and
    the icon path, which live in `storage/multimodal.py` and ARE present in those
    artifacts. Deferring the import to the four tests that need it skips only those four.
    """
    return pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard.update_service"
    )


def _png(side: int = 6) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


SECRET_PNG = _png(6)


async def _png_response(_request: web.Request) -> web.Response:
    return web.Response(body=SECRET_PNG, content_type="image/png")


async def _bytes_response(_request: web.Request) -> web.Response:
    return web.Response(body=b"payload-from-the-origin", content_type="text/plain")


def _allowlist(*entries: str) -> str:
    return ",".join(entries)


class _RecordingRefusal:
    """The transport's own session, standing in, recording every URL it is handed.

    It RECORDS and then raises. Signalling by raise alone is invisible here: every
    subject that reads this session wraps the call in `except Exception` and turns it
    into `None` or `UpdateError("offline", ...)` -- byte-identical to a real refusal. With
    `_hop_is_refused` neutered to `return False`, the whole file stayed green while the
    process issued a GET to `169.254.169.254`. The list is a record the subject cannot
    destroy, so `requested == []` is the assertion that means something.
    """

    closed = False

    def __init__(self) -> None:
        self.requested: list[str] = []

    def get(self, url, **_kwargs):
        self.requested.append(str(url))
        raise AssertionError(f"an outbound request was issued for {url!r}")

    async def close(self) -> None:
        self.closed = True


def _refusing_session() -> Any:
    return _RecordingRefusal()


def _decoded(data_url: str | None) -> bytes | None:
    if data_url is None:
        return None
    return base64.b64decode(data_url.split(",", 1)[1])


@pytest.fixture()
def routed():
    """The stubbed public address answers, so a pinned request can actually land."""
    with public_ip_routed_to_loopback():
        yield


# ── the redirect a client would follow on its own ────────────────────────────


@pytest.mark.asyncio
async def test_a_redirect_into_the_private_network_is_refused(routed, vetting):
    """The hop the address check never saw.

    Both hosts are on the insecure-HTTP allowlist, so the HTTP policy cannot be what
    refuses the second hop -- only the address can be. `internal.requests` is the
    assertion that matters: a refusal that still made the request is not a refusal.
    """
    internal = LocalServer()
    internal.route("/secret.png", _png_response)
    await internal.start()

    cdn = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://127.0.0.1:{internal.port}/secret.png")

    cdn.route("/icon.png", _redirect)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{cdn.port}", f"127.0.0.1:{internal.port}"
        ),
    )
    session = aiohttp.ClientSession()
    try:
        result = await handler._fetch_image_as_data_url(
            f"http://{CDN}:{cdn.port}/icon.png"
        )
    finally:
        await session.close()
        await cdn.stop()
        await internal.stop()

    assert result is None
    assert [path for path, _host in cdn.requests] == ["/icon.png"]
    assert internal.requests == [], (
        "the redirect was followed into the private network; the address check only "
        "ever saw the first URL"
    )


@pytest.mark.asyncio
async def test_a_redirect_to_a_vettable_address_is_followed(routed, vetting):
    """Refusing redirects outright would break the updater.

    A GitHub `browser_download_url` answers `302` to a signed object host on every real
    download, so the hops have to be followed and vetted, not refused. This is the
    must-pass half of the pair above and it fails on any fix that just switches
    redirects off.
    """
    origin = LocalServer()
    origin.route("/asset.png", _png_response)
    await origin.start()

    cdn = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://{ORIGIN}:{origin.port}/asset.png")

    cdn.route("/icon.png", _redirect)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{cdn.port}", f"{ORIGIN}:{origin.port}"
        ),
    )
    session = aiohttp.ClientSession()
    try:
        result = await handler._fetch_image_as_data_url(
            f"http://{CDN}:{cdn.port}/icon.png"
        )
    finally:
        await session.close()
        await cdn.stop()
        await origin.stop()

    assert _decoded(result) is not None
    assert [path for path, _host in origin.requests] == ["/asset.png"]


@pytest.mark.asyncio
async def test_every_hop_carries_the_host_it_was_vetted_for(routed, vetting):
    """The pin rewrites the URL host to the IP, so the name has to travel separately.

    A virtual-hosted origin routes on `Host`. If the pin dropped it, every pinned
    request would land on whatever that IP serves by default, which is a functional
    break rather than a security one -- and invisible to a test that only asserts a
    refusal.
    """
    origin = LocalServer()
    origin.route("/asset.png", _png_response)
    await origin.start()

    cdn = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://{ORIGIN}:{origin.port}/asset.png")

    cdn.route("/icon.png", _redirect)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{cdn.port}", f"{ORIGIN}:{origin.port}"
        ),
    )
    session = aiohttp.ClientSession()
    try:
        await handler._fetch_image_as_data_url(f"http://{CDN}:{cdn.port}/icon.png")
    finally:
        await session.close()
        await cdn.stop()
        await origin.stop()

    assert cdn.requests == [("/icon.png", f"{CDN}:{cdn.port}")]
    assert origin.requests == [("/asset.png", f"{ORIGIN}:{origin.port}")]


@pytest.mark.asyncio
async def test_a_redirect_chain_stops_at_the_hop_ceiling(routed, vetting):
    """A server that redirects to itself must not be followed for ever."""
    server = LocalServer()
    hops: list[int] = []

    async def _loop(_request: web.Request):
        hops.append(1)
        raise web.HTTPFound(f"http://{CDN}:{server.port}/loop")

    server.route("/loop", _loop)
    await server.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{server.port}",
    )
    try:
        result = await handler._fetch_image_as_data_url(
            f"http://{CDN}:{server.port}/loop"
        )
    finally:
        await server.stop()

    assert result is None
    # Absolute first: a ceiling raised to a thousand would satisfy the second assertion
    # on its own, and "bounded" is the property, not "bounded by whatever it says".
    assert len(hops) <= 8, len(hops)
    assert len(hops) == _MAX_VETTED_REDIRECTS + 1


# ── the rebind a client would follow with no redirect at all ─────────────────


@pytest.mark.asyncio
async def test_a_rebound_name_is_dialled_at_the_address_that_was_validated(vetting):
    """No redirect, no allowlist trickery: the name simply answers differently twice.

    The rebind is staged in `getaddrinfo`, which is where a real one happens: the first
    lookup answers public, every one after that answers loopback. An unvetted client
    resolves once for the check and again for the connection and reaches the private
    address. The transport resolves ONCE, in the connector, through the gate -- so there
    is no second answer to prefer, and that is what the lookup counts pin.

    The control is a stock `aiohttp.ClientSession`, which really does re-resolve. It used
    to be a session carrying `MapResolver`, which hardcodes name-to-address and calls
    `getaddrinfo` zero times: staging the rebind or not made no difference to it, so the
    "the rebind really was staged" guard it was supposed to provide was vacuous.
    """
    internal = LocalServer()
    internal.route("/icon.png", _png_response)
    await internal.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{internal.port}",
    )
    url = f"http://{CDN}:{internal.port}/icon.png"
    try:
        with rebinding_dns(CDN, [STUB_PUBLIC_IP, "127.0.0.1"]), counted_dns() as vetted:
            result = await handler._fetch_image_as_data_url(url)

        vetted_requests = list(internal.requests)
        vetted_lookups = [host for host in vetted if host == CDN]

        with rebinding_dns(CDN, [STUB_PUBLIC_IP, "127.0.0.1"]), counted_dns() as control:
            plain = aiohttp.ClientSession()
            try:
                assert handler._request_ips_blocking(url) == [STUB_PUBLIC_IP]
                async with plain.get(url) as resp:
                    assert resp.status == 200
            finally:
                await plain.close()
        control_lookups = [host for host in control if host == CDN]
    finally:
        await internal.stop()

    assert result is None
    assert vetted_requests == [], (
        "a second resolution was preferred over the validated one, so a name that "
        "answers differently the next time reaches a private host"
    )
    assert internal.requests, (
        "the control never reached the private server either, so the rebind was not "
        "staged and the assertion above passes for the wrong reason"
    )
    assert len(vetted_lookups) == 1, (
        f"the transport resolved {CDN} {len(vetted_lookups)} times; the check and the "
        "connection have to be one lookup, or the second answer is what gets dialled"
    )
    assert len(control_lookups) == 2, (
        f"the control resolved {CDN} {len(control_lookups)} times; it is only evidence "
        "of a rebind if it really asked twice"
    )


@pytest.mark.asyncio
async def test_the_one_resolution_the_transport_makes_is_the_gated_one(vetting):
    """One lookup is necessary and not sufficient: it has to be the gate's.

    A connector carrying a stock `ThreadedResolver` also resolves exactly once, and in
    the test above it also fails to reach the server -- because conftest refuses outbound
    connections to the stubbed public address, not because anything was validated. So the
    name answers PRIVATE here, and there is nothing else to stop the connection: the
    control reaches the internal server on the same answer, and the transport must not.
    """
    internal = LocalServer()
    internal.route("/icon.png", _png_response)
    await internal.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{internal.port}",
    )
    url = f"http://{CDN}:{internal.port}/icon.png"
    try:
        with dns_answering(CDN, ["127.0.0.1"]), counted_dns() as vetted:
            result = await handler._fetch_image_as_data_url(url)
        vetted_requests = list(internal.requests)
        vetted_lookups = [host for host in vetted if host == CDN]

        with dns_answering(CDN, ["127.0.0.1"]):
            plain = aiohttp.ClientSession()
            try:
                async with plain.get(url) as resp:
                    assert resp.status == 200
            finally:
                await plain.close()
    finally:
        await internal.stop()

    assert result is None
    assert vetted_requests == [], (
        "the connector dialled an address the gate never approved"
    )
    assert internal.requests, (
        "the control did not reach the server either, so the answer was not private and "
        "the assertion above passes for the wrong reason"
    )
    assert len(vetted_lookups) == 1, len(vetted_lookups)


# ── the same transport, driven through the release-asset download ────────────


@pytest.mark.asyncio
async def test_the_release_download_refuses_a_redirect_into_the_private_network(routed, vetting):
    us = _us()
    internal = LocalServer()
    internal.route("/secret.py", _bytes_response)
    await internal.start()

    github = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://127.0.0.1:{internal.port}/secret.py")

    github.route("/asset.py", _redirect)
    await github.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{github.port}", f"127.0.0.1:{internal.port}"
        ),
    )
    try:
        with pytest.raises(us.UpdateError) as raised:
            await us._http_get_bytes(
                f"http://{CDN}:{github.port}/asset.py", vetting=handler
            )
    finally:
        await github.stop()
        await internal.stop()

    assert raised.value.code == "offline"
    assert internal.requests == []


@pytest.mark.asyncio
async def test_the_release_download_follows_a_redirect_to_a_vettable_address(routed, vetting):
    """`browser_download_url` always redirects; the updater has to survive it."""
    us = _us()
    origin = LocalServer()
    origin.route("/asset.py", _bytes_response)
    await origin.start()

    github = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://{ORIGIN}:{origin.port}/asset.py")

    github.route("/asset.py", _redirect)
    await github.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{github.port}", f"{ORIGIN}:{origin.port}"
        ),
    )
    try:
        data = await us._http_get_bytes(
            f"http://{CDN}:{github.port}/asset.py", vetting=handler
        )
    finally:
        await github.stop()
        await origin.stop()

    assert data == b"payload-from-the-origin"
    assert [path for path, _host in origin.requests] == ["/asset.py"]


@pytest.mark.asyncio
async def test_the_release_download_refuses_a_private_address_outright(vetting):
    us = _us()
    internal = LocalServer()
    internal.route("/secret.py", _bytes_response)
    await internal.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"127.0.0.1:{internal.port}",
    )
    try:
        with pytest.raises(us.UpdateError) as raised:
            await us._http_get_bytes(
                f"http://127.0.0.1:{internal.port}/secret.py", vetting=handler
            )
    finally:
        await internal.stop()

    assert raised.value.code == "offline"
    assert internal.requests == []


# ── a URL that cannot be parsed at all ───────────────────────────────────────


MALFORMED = [
    "https://[::1/icon.png",
    "http://[bad",
    "https://[::ffff:127.0.0.1/x",
]


@pytest.mark.parametrize("url", MALFORMED)
def test_a_url_the_parser_rejects_is_refused_rather_than_raised(url):
    """`urlparse('https://[::1/x')` raises ValueError.

    The gate is reached from `transformer`, `image` and `video` with URLs taken straight
    out of a user message, so the raise is not hypothetical: it escapes as a ValueError
    from a helper whose contract is a boolean.
    """
    handler = vetting_handler()
    assert handler._request_ips_blocking(url) is None
    assert handler._is_insecure_http_allowed(url) is False


@pytest.mark.parametrize("url", MALFORMED)
@pytest.mark.asyncio
async def test_the_address_check_answers_false_for_an_unparseable_url(url):
    handler = vetting_handler()
    assert await handler._is_safe_url(url) is False


@pytest.mark.asyncio
async def test_an_unparseable_icon_url_returns_none_without_raising(vetting):
    """`catalog_manager` gathers icon fetches with `return_exceptions=True`, so a raise
    here is swallowed with no log at all."""
    handler = vetting()
    session = _refusing_session()
    handler._vetted_http_session = session

    assert await handler._fetch_image_as_data_url("https://[::1/icon.png") is None
    assert session.requested == [], session.requested


@pytest.mark.asyncio
async def test_an_unparseable_release_url_stays_an_update_error(vetting):
    """`actions.py` catches `UpdateError` and nothing else, so a ValueError is a 500."""
    us = _us()
    handler = vetting()
    session = _refusing_session()
    handler._vetted_http_session = session

    with pytest.raises(us.UpdateError) as raised:
        await us._http_get_bytes("https://[::1/asset.py", vetting=handler)
    assert raised.value.code == "offline"
    assert session.requested == [], session.requested


# ── one resolver outage, one warning ─────────────────────────────────────────


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_one_dead_host_produces_one_warning_across_the_icon_fan_out(vetting):
    """The catalog fans out over every distinct icon URL -- 513 for the current
    catalog. One dead host used to mean one WARNING per URL.

    Driven through the real transport rather than a session double, because the name is
    now resolved inside the connector: a double with no connector never reaches the
    branch that warns.
    """
    handler = vetting()
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)

    try:
        for index in range(50):
            await handler._fetch_image_as_data_url(
                f"https://gone.invalid/icon-{index}.png"
            )
        first_host = [r for r in records.records if r.levelno >= logging.WARNING]
        for index in range(50):
            await handler._fetch_image_as_data_url(
                f"https://other.invalid/icon-{index}.png"
            )
        both_hosts = [r for r in records.records if r.levelno >= logging.WARNING]
    finally:
        handler.logger.removeHandler(records)

    assert len(first_host) == 1, [r.getMessage() for r in first_host]
    assert len(both_hosts) == 2, [r.getMessage() for r in both_hosts]
    repeats = [
        r
        for r in records.records
        if r.levelno == logging.DEBUG and "DNS resolution failed" in r.getMessage()
    ]
    assert len(repeats) == 98, (
        "the repeats have to stay visible at DEBUG; a latch that goes silent hides a "
        "recurring outage from an operator who raised the level to find it"
    )


# ── the deadline ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_dribbling_server_cannot_hold_the_transport_open(vetting):
    """`sock_read` restarts on every byte and `HTTP_TOTAL_TIMEOUT_SECONDS` defaults to
    disabled, so one byte every half second held the call open indefinitely and the
    byte cap never fired. Ten such URLs fill the catalog sync's semaphore."""
    handler = vetting()

    class _Body:
        def __init__(self) -> None:
            self.served = 0

        async def iter_chunked(self, _size: int):
            while True:
                await asyncio.sleep(0.05)
                self.served += 1
                yield b"\x00"

    class _Response:
        status = 200
        headers = {"Content-Type": "image/png"}

        def __init__(self) -> None:
            self.content = _Body()

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    response = _Response()

    class _Session:
        closed = False

        def get(self, _url, **_kwargs):
            return response

        async def close(self) -> None:
            self.closed = True

    dribbling: Any = _Session()

    handler._vetted_http_session = dribbling

    async def _drive() -> None:
        async with handler._vetted_get(
            "https://cdn.example.com/dribble.png", total_seconds=0.3
        ) as resp:
            async for _chunk in resp.content.iter_chunked(64 * 1024):
                pass

    # The outer wait_for is a backstop, never the mechanism: without it a regression
    # here hangs the suite instead of failing it. `elapsed` is what tells the two apart.
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_drive(), timeout=5.0)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 3.0, (
        f"the exchange ran for {elapsed:.1f}s, so it was the test's backstop that "
        "stopped it and not the transport's own deadline"
    )
    assert response.content.served < 60, response.content.served


@pytest.mark.asyncio
async def test_a_refused_address_raises_the_shared_transport_error(vetting):
    handler = vetting()
    session = _refusing_session()
    handler._vetted_http_session = session

    with pytest.raises(UnfetchableAddress):
        async with handler._vetted_get("https://127.0.0.1/x", total_seconds=1.0):
            pass

    assert session.requested == [], session.requested


# ── what pinning the IP into the URL cost, one finding at a time ─────────────


@pytest.mark.asyncio
async def test_two_names_on_one_address_do_not_share_an_authenticated_connection(
    vetting, tmp_path, monkeypatch
):
    """The connection pool keys on the URL's host, and TLS identity follows it.

    `ConnectionKey` is `(host, port, is_ssl, ssl, proxy, proxy_auth, proxy_headers_hash)`.
    It carries no `server_hostname`, so rewriting the URL host to the validated IP made
    the key an ADDRESS: two hostnames resolving to the same machine shared one pooled TLS
    connection, and only the first was ever authenticated. Measured against this server,
    whose leaf's only SAN is the first name: `b` alone was refused, `a` then `b` on one
    session both succeeded, and the pool held one connection keyed on the IP.

    Reproduced end to end rather than by reading the key, because the key is aiohttp's
    and could change; what must not change is that the second name is refused.
    """
    import ssl

    certs = single_san_cert(ORIGIN, tmp_path)
    server = LocalServer()
    server.route("/icon.png", _png_response)
    server.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.ssl_context.load_cert_chain(certs["cert"], certs["key"])
    await server.start()

    handler = vetting()
    trusting = ssl.create_default_context(cafile=str(certs["cert"]))
    installed = aiohttp.ClientSession._request

    async def _trusting(self, method, url, **kwargs):
        kwargs.setdefault("ssl", trusting)
        return await installed(self, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", _trusting)
    results = []
    try:
        with public_ip_routed_to_loopback():
            for name in (ORIGIN, CDN):
                results.append(
                    await handler._fetch_image_as_data_url(
                        f"https://{name}:{server.port}/icon.png",
                    )
                    is not None
                )
    finally:
        await server.stop()

    assert results[0] is True, (
        "the name the certificate is actually for was refused, so the second assertion "
        "would pass for the wrong reason"
    )
    assert results[1] is False, (
        "a second hostname on the same address reused the first one's authenticated "
        "connection; nothing checked that the certificate covers it"
    )


@pytest.mark.asyncio
async def test_every_validated_address_is_offered_to_the_connector(vetting):
    """A host with several A records must not hang on the first one being down.

    The gate rejects the host outright if ANY address is private, so every address it
    returns is equally safe and there is no reason to discard all but one. Pinning kept
    `ips[0]` and had no fallback, so one blackholed anycast node or one down address in
    a rotation failed the fetch. Asserted as what the connector is handed.
    """
    handler = vetting()
    resolver = (await handler._vetted_session()).connector._resolver
    rotation = ["93.184.216.34", "93.184.216.35", "93.184.216.36"]

    assert [a["host"] for a in await resolver.resolve(CDN, 443)] == [STUB_PUBLIC_IP], (
        "the single-address case has to work, or the multi-address assertion below "
        "passes for the wrong reason"
    )

    with dns_answering("rotation.example.com", rotation):
        answers = await resolver.resolve("rotation.example.com", 443)

    assert [a["host"] for a in answers] == rotation, (
        "the connector was handed one address out of three, with no fallback"
    )


@pytest.mark.asyncio
async def test_a_fan_out_over_one_host_resolves_it_once(vetting, routed):
    """`ttl_dns_cache` sits in front of the gate, so the fan-out pays for one lookup.

    A host that is already an address never reaches a resolver, so pinning the IP into
    the URL bypassed the connector's cache entirely: 20 icon URLs on one host meant 20
    blocking `getaddrinfo` calls. The real fan-out is up to 513 URLs.

    The server closes each connection, so the pool cannot be what supplies the answer:
    with keep-alive, one pooled connection serves all twenty and the count is 1 whether
    the cache exists or not, which is a test that cannot see the thing it names.
    """
    origin = LocalServer()

    async def _closing(_request: web.Request):
        return web.Response(
            body=SECRET_PNG, content_type="image/png", headers={"Connection": "close"}
        )

    origin.route("/icon.png", _closing)
    await origin.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{origin.port}",
    )
    try:
        with counted_dns() as calls:
            for _ in range(20):
                await handler._fetch_image_as_data_url(
                    f"http://{CDN}:{origin.port}/icon.png"
                )
    finally:
        await origin.stop()

    for_cdn = [host for host in calls if host == CDN]
    assert len(origin.requests) == 20, len(origin.requests)
    assert len(for_cdn) <= 2, for_cdn


@pytest.mark.asyncio
async def test_the_request_carries_the_hostname_the_caller_asked_for(
    vetting, routed, monkeypatch
):
    """URL host, `Host` header and SNI, asserted together.

    Pinning carried the hostname out of band -- URL host became the IP, `Host` was
    overridden, and `server_hostname` was passed for SNI. Deleting the three lines that
    forwarded SNI was invisible across the whole suite. Nothing is hand-carried now, so
    the property is that the hostname never leaves the URL in the first place.
    """
    origin = LocalServer()
    origin.route("/icon.png", _png_response)
    await origin.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{origin.port}",
    )
    dialled: list[str] = []
    installed = aiohttp.ClientSession._request

    async def _record(self, method, url, **kwargs):
        dialled.append(str(url))
        assert "server_hostname" not in kwargs, (
            "SNI is being hand-carried again; aiohttp derives it from the URL host, and "
            "a hostname carried beside the URL is one the connection pool cannot see"
        )
        assert "Host" not in (kwargs.get("headers") or {}), kwargs.get("headers")
        return await installed(self, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", _record)
    try:
        await handler._fetch_image_as_data_url(f"http://{CDN}:{origin.port}/icon.png")
    finally:
        await origin.stop()

    assert dialled == [f"http://{CDN}:{origin.port}/icon.png"], dialled
    assert origin.requests == [("/icon.png", f"{CDN}:{origin.port}")]


@pytest.mark.asyncio
async def test_the_transport_never_hands_a_name_to_an_env_proxy(vetting, monkeypatch):
    """A proxy resolves the name itself, so an address this package vetted is not the
    address that gets dialled.

    Pinning made this worse in the other direction as well: `proxy_bypass` was handed an
    IP literal, so a `NO_PROXY` entry naming the host stopped matching and the request
    was proxied by a rule the operator had written to exclude it.
    """
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    handler = vetting()

    assert (await handler._vetted_session())._trust_env is False  # noqa: SLF001


def test_the_resolver_bypass_predicate_matches_the_one_aiohttp_uses():
    """`_skips_the_resolver` has to be a SUPERSET of what aiohttp skips.

    aiohttp answers an address-shaped host from the URL without consulting any resolver,
    so anything it treats that way is a host the connector's gate never sees and which
    has to be decided before the request is built. The predicate is replicated rather
    than imported because `aiohttp.helpers` is not public API -- and pinned here, against
    the real one, so the two cannot drift apart quietly.

    This checks the PREDICATE and nothing else. It stayed green through an SSRF bypass
    that reached loopback with a completed TLS handshake, because the predicate was never
    wrong -- the ARGUMENT was. What the pre-check must be handed is pinned below.
    """
    from aiohttp.helpers import is_ip_address

    from open_webui_openrouter_pipe.storage.multimodal import _skips_the_resolver

    hosts = [
        "127.0.0.1", "10.0.0.5", "::1", "fe80::1", "0.0.0.0", "1.2.3.4.5", "123",
        "example.com", "cdn.example.com", "localhost", "xn--80ak6aa92e.com", "",
        "1.2.3", "a::b", "999.999.999.999",
    ]
    mine = {host: _skips_the_resolver(host) for host in hosts}
    theirs = {host: is_ip_address(host) for host in hosts}

    assert mine == theirs, {h: (mine[h], theirs[h]) for h in hosts if mine[h] != theirs[h]}


# ── the host the pre-check reads has to be the host that gets dialled ─────────

# UTS-46, which `idna` applies inside yarl and therefore inside aiohttp, maps five
# codepoints to '.' on top of the ASCII one. `urlparse` maps none of them, so a host
# written with any of the five parsed as a NAME for the gate and as an IP LITERAL for
# the connector -- and aiohttp never calls a resolver for a literal, so the connector's
# gate was never reached either. Measured end to end on the shipped defaults: a completed
# TCP connect to 127.0.0.1 and a TLS ClientHello, with the gate consulted zero times.
DOT_LOOKALIKES = ["\u002e", "\u2024", "\u3002", "\ufe52", "\uff0e", "\uff61"]

DOT_IDS = ["full-stop", "one-dot-leader", "ideographic", "small-full-stop",
           "fullwidth", "halfwidth-ideographic"]


@pytest.mark.parametrize("separator", DOT_LOOKALIKES, ids=DOT_IDS)
def test_the_gate_reads_the_host_the_connector_will_dial(separator):
    """One URL, one host. Two parsers is the whole defect.

    Asserted against `yarl.URL(...).raw_host` because that is literally what aiohttp
    builds the request from, so this cannot drift with a library update the way a
    hand-written normalisation table of these six codepoints would -- that table would be
    a denylist frozen at one Unicode version, and the next `idna` release reopens it.
    """
    from yarl import URL

    handler = vetting_handler()
    url = f"https://127{separator}0{separator}0{separator}1/x"

    target = handler._parsed_target(url)

    assert target is not None
    assert target[0] == URL(url).raw_host


@pytest.mark.parametrize("separator", DOT_LOOKALIKES, ids=DOT_IDS)
@pytest.mark.parametrize(
    ("quad", "refused"), [("127.0.0.1", True), ("93.184.216.34", False)]
)
def test_a_dot_lookalike_cannot_smuggle_an_address_past_the_pre_check(
    separator, quad, refused
):
    """Both directions, so neither `return True` nor `return False` satisfies this.

    The loopback row is the exploit: refusing it is the fix. The public row is the guard
    on the fix -- refusing every host carrying an unusual codepoint would also make the
    loopback row pass, and would break every internationalised host at the same time.
    """
    handler = vetting_handler()
    url = f"https://{quad.replace('.', separator)}/x"

    assert handler._hop_is_refused(url) is refused


@pytest.mark.asyncio
async def test_a_dot_lookalike_loopback_host_never_reaches_a_socket(vetting):
    """The predicate, driven to the socket, on the valves the pipe actually ships.

    `ENABLE_SSRF_PROTECTION` and `ALLOW_INSECURE_HTTP` are left at their defaults, so
    this is the configuration a deployment runs. The server records connections rather
    than requests: the vulnerable version completed the TCP connect and sent a TLS
    ClientHello, which a request-level double cannot see.
    """
    connections: list[Any] = []

    async def _record(reader, writer):
        connections.append(writer.get_extra_info("peername"))
        writer.close()

    server = await asyncio.start_server(_record, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    handler = vetting()
    url = f"https://127\uff0e0\uff0e0\uff0e1:{port}/latest/meta-data/"
    try:
        with pytest.raises(UnfetchableAddress):
            async with handler._vetted_get(url, total_seconds=5.0):
                pass
    finally:
        server.close()
        await server.wait_closed()

    assert connections == [], connections


def test_the_insecure_http_allowlist_matches_the_host_that_gets_dialled():
    """The allowlist is the other consumer of the host, and it has to see the same one.

    An operator allowlisting `cdn.example.com` and a URL written with a fullwidth dot
    used to be two different hosts: the allowlist compared the raw text and the connector
    dialled the normalised name. Both rows here are the SAME dialled host, so an
    implementation that simply refused the odd one fails the second.
    """
    handler = vetting_handler(
        ALLOW_INSECURE_HTTP=True, ALLOW_INSECURE_HTTP_HOSTS="cdn.example.com:8080"
    )

    assert handler._is_insecure_http_allowed("http://cdn.example.com:8080/x") is True
    assert handler._is_insecure_http_allowed("http://cdn\uff0eexample\uff0ecom:8080/x") is True
    assert handler._is_insecure_http_allowed("http://other.example.com:8080/x") is False


# ── the scheme the connector will dial, as opposed to the one that was checked ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme, lands",
    [("http", True), ("ws", False), ("tcp", False)],
    ids=["http-lands", "ws-refused", "tcp-refused"],
)
async def test_a_redirect_that_changes_the_scheme_reaches_no_socket_unless_it_is_http(
    routed, vetting, scheme, lands
):
    """`Location` chooses the scheme, and aiohttp dials more than two of them.

    `TCPConnector.allowed_protocol_schema_set` is `{'', http, https, ws, wss, tcp}`, and
    `is_ssl()` is true for exactly `https` and `wss`. So `ws://` and `tcp://` leave as
    cleartext HTTP on the port in the URL -- and the gate tested `scheme != "http"`,
    which answered "allowed" for both. `urljoin` returns a differing-scheme `Location`
    verbatim, so the hostile origin here does not need to have chosen the first URL.

    Both hosts are allowlisted, so the HTTP policy cannot be what separates the rows:
    the only difference between them is the four letters in `Location`. The evidence is
    what the far end RECORDED, because an `UnfetchableAddress` is also what an
    unresolvable host raises -- the exception type cannot tell a refusal from a typo.
    The `http` row is the must-pass half: a gate that refuses every redirect fails it.
    """
    origin = LocalServer()
    origin.route("/secret.png", _png_response)
    await origin.start()

    cdn = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"{scheme}://{ORIGIN}:{origin.port}/secret.png")

    cdn.route("/icon.png", _redirect)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{cdn.port}", f"{ORIGIN}:{origin.port}"
        ),
    )
    try:
        result = await handler._fetch_image_as_data_url(
            f"http://{CDN}:{cdn.port}/icon.png"
        )
    finally:
        await cdn.stop()
        await origin.stop()

    assert [path for path, _host in cdn.requests] == ["/icon.png"]
    assert [path for path, _host in origin.requests] == (
        ["/secret.png"] if lands else []
    ), (
        f"a {scheme}:// redirect was dialled; the scheme test was a negative one and "
        f"everything that was not literally 'http' counted as allowed"
    )
    assert (_decoded(result) is not None) is lands


def test_every_scheme_the_connector_will_dial_is_refused_unless_the_gate_names_it():
    """The set of dialable schemes belongs to aiohttp, and it has already grown once.

    `tcp` is on `TCPConnector`'s set and on no other connector's, so a denylist naming
    `ws`/`wss` would have shipped with `tcp` open. Read from aiohttp at run time rather
    than copied, so the next scheme it adds arrives here as a failure the day the
    dependency is bumped -- and the fix for that failure is a deliberate edit to
    `_FETCHABLE_SCHEMES`, not to a list of refusals.

    Both verdicts are asserted over distinct inputs, so neither a constant `True` nor a
    constant `False` satisfies the test.
    """
    dialable = set(aiohttp.TCPConnector.allowed_protocol_schema_set)
    handler = vetting_handler(
        ALLOW_INSECURE_HTTP=True, ALLOW_INSECURE_HTTP_HOSTS=_allowlist(CDN)
    )

    def _verdict(scheme: str) -> bool:
        url = f"{scheme}://{CDN}/x" if scheme else f"//{CDN}/x"
        return handler._is_insecure_http_allowed(url)

    assert mm._FETCHABLE_SCHEMES <= dialable, (
        "the allow-list names a scheme this connector will not dial, so one of the two "
        "is stale"
    )
    assert {s: _verdict(s) for s in sorted(mm._FETCHABLE_SCHEMES)} == {
        "http": True,
        "https": True,
    }, "an allow-listed scheme was refused, so the refusals below prove nothing"
    assert {s: _verdict(s) for s in sorted(dialable - mm._FETCHABLE_SCHEMES)} == {
        s: False for s in sorted(dialable - mm._FETCHABLE_SCHEMES)
    }, "aiohttp will dial a scheme the gate has no opinion about"


@pytest.mark.parametrize("prefix", ["http", "HTTP", "Http"])
def test_an_owui_file_path_inside_an_absolute_url_earns_no_exemption(prefix):
    """`is_internal_file_url` is a raw-string test; the scheme above it came from yarl.

    So `HTTP://` normalised to `http`, reached an exemption whose own guard is
    `url.startswith("http://")`, and was allowed on the spelling of four letters. For
    the branch's stated purpose -- a relative OWUI path -- it was already dead: that had
    no scheme and was answered two lines earlier, and is refused downstream anyway
    because it has no host to dial.

    The allowlisted row is the must-pass half: this is about the substring earning an
    exemption, not about http being blocked.
    """
    handler = vetting_handler(
        ALLOW_INSECURE_HTTP=True, ALLOW_INSECURE_HTTP_HOSTS=_allowlist(CDN)
    )
    path = "/api/v1/files/5c9d.png"

    assert handler._is_insecure_http_allowed(f"{prefix}://{ORIGIN}{path}") is False
    assert handler._is_insecure_http_allowed(f"{prefix}://{CDN}{path}") is True
    assert handler._hop_is_refused(path) is True, (
        "a relative OWUI path is refused for having no host, which is what makes "
        "deleting the exemption a no-op for the paths it was written for"
    )


# ── the latch, and what stops it growing without bound ───────────────────────


@pytest.mark.asyncio
async def test_one_blocked_host_produces_one_warning_across_the_icon_fan_out(vetting):
    """The sibling of the resolver-outage latch, on the branch that refuses an address.

    A single ULA or link-local AAAA anywhere in a DNS answer blocks the whole host, which
    is an ordinary Docker or corporate condition -- and the icon path fans out over every
    distinct URL in the catalog, up to 513 of them. Unlatched that was one WARNING per
    URL; measured at 50 URLs on one host it was 50 records against the sibling branch's 1.
    """
    handler = vetting()
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)

    try:
        with dns_answering("blocked.example.com", ["fd00::1"]):
            for index in range(50):
                handler._validated_ips_for_host("blocked.example.com")
                del index
        warnings = [r for r in records.records if r.levelno >= logging.WARNING]
        repeats = [r for r in records.records if r.levelno == logging.DEBUG]
    finally:
        handler.logger.removeHandler(records)

    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert len(repeats) == 49, len(repeats)


@pytest.mark.parametrize(
    ("url", "port"),
    [
        ("https://blocked.example.com/x", 443),
        ("http://blocked.example.com/x", 80),
        ("https://blocked.example.com:8443/x", 8443),
    ],
    ids=["default-https", "default-http", "explicit"],
)
def test_the_block_record_names_the_port_that_was_targeted(vetting, url, port):
    """Host alone does not say what was being probed.

    The record used to carry the whole URL, which wrote `?token=...` into the log; the
    repair dropped the query AND the port with it. Scheme, path and query are noise, the
    port is not: `10.0.0.5:22` and `10.0.0.5:8080` are different reports, and this gate
    is reached from `requests/transformer.py` for any image, file or video URL pasted
    into a chat, so an operator watching for internal probing needs it.

    Three rows, and the default ports are read off the scheme rather than the text, so a
    record that prints only what the URL literally spelled fails two of them.
    """
    handler = vetting(
        ALLOW_INSECURE_HTTP=True, ALLOW_INSECURE_HTTP_HOSTS="blocked.example.com"
    )
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)
    try:
        with dns_answering("blocked.example.com", ["10.0.0.5"]):
            assert handler._request_ips_blocking(url) is None
        blocked = [r for r in records.records if "Blocked SSRF" in r.getMessage()]
    finally:
        handler.logger.removeHandler(records)

    assert len(blocked) == 1, [r.getMessage() for r in blocked]
    message = blocked[0].getMessage()
    assert str(port) in message, message
    assert "blocked.example.com" in message, message
    assert "?" not in message, message


def test_the_block_record_re_warns_after_its_cooldown(vetting):
    """A latch with no cooldown is silence for the life of the worker.

    Once per host, WARNING, then DEBUG for ever: an operator who raises the log level to
    investigate a recurring probe sees nothing, because the one record was emitted hours
    earlier at a level nobody was capturing. The latch itself is not the problem -- this
    branch is driven by a URL in a chat message and unlatched it floods -- so it stays,
    with a window.

    The clock is moved rather than waited on, and the cooldown is read from the module so
    this cannot pass by the constant being lowered to nothing.
    """
    from open_webui_openrouter_pipe.storage.multimodal import (
        _ADDRESS_WARN_COOLDOWN_SECONDS,
    )

    handler = vetting()
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)
    try:
        with dns_answering("blocked.example.com", ["10.0.0.5"]):
            for _ in range(3):
                handler._validated_ips_for_host("blocked.example.com", 443)
            within = [r for r in records.records if r.levelno >= logging.WARNING]
            for key in list(handler._warned_blocked_hosts):
                handler._warned_blocked_hosts[key] -= _ADDRESS_WARN_COOLDOWN_SECONDS + 1
            handler._validated_ips_for_host("blocked.example.com", 443)
            after = [r for r in records.records if r.levelno >= logging.WARNING]
    finally:
        handler.logger.removeHandler(records)

    assert len(within) == 1, [r.getMessage() for r in within]
    assert len(after) == 2, [r.getMessage() for r in after]
    assert _ADDRESS_WARN_COOLDOWN_SECONDS > 0


def test_two_ports_on_one_host_are_two_records(vetting):
    """Keying the latch on the host alone silences the second port for ever."""
    handler = vetting()
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)
    try:
        with dns_answering("blocked.example.com", ["10.0.0.5"]):
            handler._validated_ips_for_host("blocked.example.com", 22)
            handler._validated_ips_for_host("blocked.example.com", 8080)
        warnings = [r.getMessage() for r in records.records if r.levelno >= logging.WARNING]
    finally:
        handler.logger.removeHandler(records)

    assert len(warnings) == 2, warnings
    assert any("22" in m for m in warnings) and any("8080" in m for m in warnings), warnings


def test_the_warning_latch_cannot_grow_without_bound(vetting):
    """A latch keyed on the host is a set that a hostile catalog can fill.

    The bound is asserted as an ABSOLUTE number first: a check written only against
    `_DNS_WARN_LATCH_MAX_HOSTS` is satisfied by raising the constant, which is the thing
    it is supposed to prevent. The constant is checked second, so the two together say
    "bounded, and bounded by the number that is written down".
    """
    from open_webui_openrouter_pipe.storage.multimodal import _DNS_WARN_LATCH_MAX_HOSTS

    handler = vetting()
    latch: dict[str, float] = {}

    for index in range(5000):
        handler._latched_warn_level(latch, f"host-{index}.example.com")

    assert len(latch) <= 1024, len(latch)
    assert len(latch) <= _DNS_WARN_LATCH_MAX_HOSTS, (
        f"{len(latch)} hosts retained against a cap of {_DNS_WARN_LATCH_MAX_HOSTS}"
    )


def _blocked_literal(index: int) -> str:
    return f"10.{index // 65536 % 256}.{index // 256 % 256}.{index % 256}"


def _unresolvable_name(index: int) -> str:
    return f"host-{index}.invalid"


@pytest.mark.parametrize(
    ("latch_name", "other_latch", "host_for"),
    [
        ("_warned_blocked_hosts", "_warned_dns_failures", _blocked_literal),
        ("_warned_dns_failures", "_warned_blocked_hosts", _unresolvable_name),
    ],
)
def test_both_address_latches_are_bounded_by_what_production_does(
    vetting, latch_name, other_latch, host_for
):
    """Round 1 bounded one branch and left its sibling unlatched 43 lines away.

    Driven through `_validated_ips_for_host`, which is what a hostile catalog reaches.
    The previous version read the latch dicts off the handler and then called
    `_latched_warn_level` ITSELF, so neither production branch ran: making the
    blocked-host branch call `warn_level` directly and skip eviction left the whole file
    green while 5000 distinct blocked hosts grew the dict from 136 entries to 5000.

    Identity of the dict is deliberately NOT asserted -- it holds in the mutated version
    too. What is asserted is the population after production has been driven 5000 times,
    and that the sibling latch stayed empty, so a helper wired to one branch fails here
    on the other.
    """
    from open_webui_openrouter_pipe.storage.multimodal import _DNS_WARN_LATCH_MAX_HOSTS

    handler = vetting()
    records = _Records()
    handler.logger.addHandler(records)
    handler.logger.setLevel(logging.DEBUG)
    try:
        for _ in range(3):
            assert handler._validated_ips_for_host(host_for(0), 443) is None
        repeated = [r.levelno for r in records.records]
        for index in range(1, 5000):
            assert handler._validated_ips_for_host(host_for(index), 443) is None
    finally:
        handler.logger.removeHandler(records)

    assert repeated == [logging.WARNING, logging.DEBUG, logging.DEBUG], repeated

    latch = getattr(handler, latch_name)
    assert len(latch) <= 1024, len(latch)
    assert len(latch) <= _DNS_WARN_LATCH_MAX_HOSTS, (
        f"{len(latch)} hosts retained against a cap of {_DNS_WARN_LATCH_MAX_HOSTS}"
    )
    assert getattr(handler, other_latch) == {}, (
        f"{latch_name} and {other_latch} are not separate branches"
    )


# ── the deadline, and the threads it does not bound ──────────────────────────


@pytest.mark.asyncio
async def test_each_resolution_is_bounded_even_when_the_whole_call_is(
    vetting, monkeypatch
):
    """A deadline on the coroutine does not bound the thread the resolution runs in.

    `_is_safe_url` wraps its `to_thread` in `wait_for(ADDRESS_CHECK_SECONDS)` and its own
    docstring says the budget frees the caller, not the thread. The pinned-download path
    re-derived a deadline and dropped that, so a slow nameserver held an executor worker
    for as long as it wanted -- multiplied by the redirect hops, on the path that fans out
    over the whole catalog, sharing an executor with image decoding.
    """
    import threading

    monkeypatch.setattr(mm, "ADDRESS_CHECK_SECONDS", 0.3)
    handler = vetting()
    released = threading.Event()

    def _hang(_url):
        released.wait(30)
        return None

    handler._request_ips_blocking = _hang
    started = asyncio.get_running_loop().time()
    try:
        result = await asyncio.wait_for(
            handler._prepare_pinned_request("https://cdn.example.com/x"), timeout=20.0
        )
        elapsed = asyncio.get_running_loop().time() - started
    finally:
        released.set()

    assert result is None, result
    assert elapsed < 3.0, elapsed
    assert elapsed >= 0.25, (
        f"returned after {elapsed:.2f}s, so the bound asserted above was not the one "
        "that fired"
    )


# ── a port the parser accepts and every consumer rejects ─────────────────────


OUT_OF_RANGE = [
    "https://cdn.example.com:99999/icon.png",
    "http://cdn.example.com:70000/icon.png",
    "https://cdn.example.com:-1/icon.png",
]


@pytest.mark.parametrize("url", OUT_OF_RANGE)
def test_a_port_the_parser_accepts_and_rejects_is_refused_by_the_gate(url):
    """`urlparse` succeeds and `.port` raises, so the gate and the pin disagreed.

    The gate read only `.hostname` and called the address safe; `_build_pinned_request`
    then read `.port` and raised `ValueError` out of `_download_remote_url`, whose
    docstring promises None. It is reached from `streaming/streaming_core.py` with a URL
    taken out of the model's own output and no enclosing `try` in the chain.
    """
    handler = vetting_handler()

    assert handler._request_ips_blocking(url) is None
    assert handler._hop_is_refused(url) is True


@pytest.mark.parametrize("url", OUT_OF_RANGE)
@pytest.mark.asyncio
async def test_a_port_the_parser_rejects_never_reaches_the_pin(url):
    """The gate says no, so `None` still means 'blocked by policy' to both its callers.

    A `try/except` inside `_build_pinned_request` returning None would have made the pin
    disagree with the gate in a NEW direction: two callers log that None as an SSRF
    refusal, so an unparseable port would have been reported as an attack.
    """
    handler = vetting_handler()

    assert await handler._is_safe_url(url) is False
    assert await handler._prepare_pinned_request(url) is None
    assert await handler._download_remote_url(url) is None


@pytest.mark.asyncio
async def test_a_redirect_to_an_unparseable_location_stays_an_unfetchable_address(
    routed, vetting
):
    """`Location` is a header from a remote server, and `urljoin` raises on some of them.

    `urljoin('https://cdn/a', '//[::1')` raises ValueError. Every current caller happens
    to catch it, but the transport's documented failure is `UnfetchableAddress`, and a
    caller that catches only that is a reasonable thing to write.
    """
    cdn = LocalServer()

    async def _bad_location(_request: web.Request):
        return web.Response(status=302, headers={"Location": "//[::1"})

    cdn.route("/icon.png", _bad_location)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{cdn.port}",
    )
    try:
        with pytest.raises(UnfetchableAddress):
            async with handler._vetted_get(
                f"http://{CDN}:{cdn.port}/icon.png", total_seconds=5.0
            ):
                pass
    finally:
        await cdn.stop()

    assert len(cdn.requests) == 1, (
        f"{len(cdn.requests)} requests were issued; an unparseable Location that is "
        "treated as 'stay here' also ends in UnfetchableAddress -- from running the hop "
        "budget out -- so the exception alone cannot tell the two apart. Asserting "
        f"<= {_MAX_VETTED_REDIRECTS + 1} would not either: that bound is what the defect "
        "produces."
    )


# ── the session the transport owns ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_transport_closes_the_session_it_opened(vetting, recwarn):
    """Nobody else owns it, so nothing else can close it.

    `_vetted_get` opens a `ClientSession` on the handler lazily, because the connector
    that session carries is what does the vetting -- a caller handing one in could hand
    in an unvetted one. Ownership means a close hook, and `Pipe._do_close` calls it.
    """
    import gc

    handler = vetting()
    with contextlib.suppress(Exception):
        await handler._fetch_image_as_data_url("https://cdn.example.com/icon.png")

    session = handler._vetted_http_session
    assert session is not None and not session.closed

    await handler.aclose()
    assert session.closed
    assert handler._vetted_http_session is None

    await handler.aclose()

    gc.collect()
    unclosed = [w for w in recwarn.list if "Unclosed client session" in str(w.message)]
    assert unclosed == [], [str(w.message) for w in unclosed]


@pytest.mark.asyncio
async def test_a_closed_transport_stays_closed(vetting):
    """`aclose` has to be terminal, because nothing can close a second generation.

    Both resources were built lazily and consulted no closed state, and `aclose` set both
    handles to `None` -- which is exactly the state that makes the next call build a
    fresh one. `Pipe.close()` is one-shot (`_closing` latches, and `_do_close` runs once),
    so a session opened after it can never be closed by anything. Measured across three
    hot-reload generations before this: three OPEN `ClientSession` objects and
    `or-icon-decode` threads going 0 -> 3, each retaining that generation's handler,
    valves and logger through the resolver's strong reference.

    Reached in production without a race: `_do_close` caps plugin shutdown at
    `wait_for(..., 5.0)`, so an update tick still running when that expires calls
    `_require_vetting` on a handler that has already been closed.
    """
    handler = vetting()
    with contextlib.suppress(Exception):
        await handler._fetch_image_as_data_url("https://cdn.example.com/icon.png")
    assert await handler._run_decode(len, b"xyz") == 3
    session = handler._vetted_http_session
    pool = handler._decode_pool
    assert session is not None and pool is not None

    await handler.aclose()

    with pytest.raises(UnfetchableAddress):
        async with handler._vetted_get("https://cdn.example.com/icon.png", total_seconds=1.0):
            pass
    with pytest.raises(RuntimeError):
        await handler._run_decode(len, b"x")

    assert handler._vetted_http_session is None, handler._vetted_http_session
    assert handler._decode_pool is None, handler._decode_pool


@pytest.mark.asyncio
async def test_two_callers_racing_a_mode_change_end_up_on_one_session(vetting):
    """Retiring the transport nulls the slot and THEN awaits the close.

    A second caller arriving during that await saw `None`, built its own session and
    installed it; the first resumed and overwrote the slot. `aclose()` closes what is in
    the slot, so the loser stayed open for the life of the worker -- measured as one
    orphan, `closed=False` after `aclose()`, and an "Unclosed client session" error at
    collection. The catalog refresh fans out concurrently, so two simultaneous callers
    is the ordinary case, and taking the session per HOP rather than per chain raises
    the rate further.

    `a is b` is the assertion that says which fix this is. Keeping a set of every session
    ever built and closing them all in `aclose()` stops the leak while leaving two
    connection pools serving traffic, which quietly doubles the connection limit -- it
    passes the closed-ness assertions and fails this one.
    """
    handler = vetting(ENABLE_SSRF_PROTECTION=False)
    url = "https://cdn.example.com/icon.png"

    before = await handler._vetted_session(url)
    handler.valves.ENABLE_SSRF_PROTECTION = True
    a, b = await asyncio.gather(
        handler._vetted_session(url), handler._vetted_session(url)
    )
    await handler.aclose()

    assert a is not before, (
        "the valve change did not rebuild the transport, so nothing raced here"
    )
    assert a is b, (
        "the two callers were handed different sessions, so the pipe is running two "
        "connection pools and only one of them is reachable through the handler"
    )
    assert before.closed, "the session the mode change retired was left open"
    assert a.closed and b.closed, "a session survived aclose()"


def test_a_lock_held_when_a_loop_dies_does_not_deadlock_the_next_one():
    """The transport outlives loops, so whatever serialises it has to be rebuilt too.

    An `asyncio.Lock` created in `__init__` is one object for the life of the handler.
    Nothing binds it to a loop until someone waits on it -- and then the waiter's future
    belongs to the loop that is running, so a lock left acquired by a coroutine that
    died with its loop is never released and the next loop's caller waits for ever. The
    same shape `_vetted_loop_is_stale` already exists for, applied to the lock.

    Driven by ABANDONING the lock on the first loop rather than by asserting object
    identity: a check that the lock is a different object is satisfied by rebuilding it
    on every call, which would serialise nothing.
    """
    handler = vetting_handler(ENABLE_SSRF_PROTECTION=False)

    async def _strand_the_lock() -> None:
        await handler._vetted_transport_lock().acquire()

    async def _use_the_transport() -> Any:
        try:
            return await asyncio.wait_for(
                handler._vetted_session("https://cdn.example.com/i.png"), timeout=5
            )
        except TimeoutError:
            return None
        finally:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(handler.aclose(), timeout=5)

    asyncio.run(_strand_the_lock())
    session = asyncio.run(_use_the_transport())

    assert session is not None, (
        "the transport never came back: this loop is waiting on a lock the previous "
        "loop's caller left acquired, and nothing on this loop can ever release it"
    )
    assert session.closed


@pytest.mark.asyncio
async def test_an_icon_fetched_after_shutdown_is_refused_rather_than_reopening_one(
    vetting,
):
    """The refusal reaches the caller as the outcome the caller already handles.

    `_fetch_image_as_data_url` returns None and `_http_get_bytes` raises
    `UpdateError("offline")`, which is TRANSIENT -- so a late update tick backs off
    rather than retiring the release. What must not happen is a fourth session.
    """
    us = _us()
    handler = vetting()
    await handler.aclose()

    assert await handler._fetch_image_as_data_url("https://cdn.example.com/i.png") is None
    with pytest.raises(us.UpdateError) as raised:
        await us._http_get_bytes("https://cdn.example.com/a.py", vetting=handler)

    assert raised.value.code in us._TRANSIENT_CODES
    assert handler._vetted_http_session is None


# ── ranges no ipaddress flag fires for ─────────────────────────────────────

# Measured on 3.11.14 and 3.12.2: none of is_private / is_loopback / is_link_local /
# is_multicast / is_reserved / is_unspecified is True for any of these, and is_global is
# True for the last two -- so neither the six-flag ladder NOR is_global alone refuses
# them. 100.64.0.0/10 is Tailscale's default range.
UNFLAGGED = (
    "100.64.0.1",       # RFC 6598 carrier-grade NAT
    "100.100.100.100",  # the Tailscale MagicDNS resolver
    "2002:7f00:1::",    # 6to4 encoding of 127.0.0.1
    "fec0::1",          # IPv6 site-local
)

# Refused today by is_reserved / is_private, and refused for the RIGHT reason after the
# inversion: the embedded IPv4 is what decides, so the verdict does not move with the
# interpreter's opinion of the wrapper.
TUNNELLED_TO_PRIVATE = ("64:ff9b::7f00:1", "::ffff:127.0.0.1", "2002:a00:1::")

# A Teredo address carries TWO IPv4 addresses -- the tunnel server in bits 32..63 and the
# client in the low 32 -- and both are chosen by whoever writes the address. Keeping only
# the client turned 2001::/32, which `is_private` refuses outright before any unwrapping,
# into a range that reported "no reason to refuse". Measured identical on 3.11.14 and
# 3.12.2. Each row here names a server nobody should be able to steer a fetch at, behind
# a client that is 8.8.8.8.
TEREDO_SERVER_IS_INTERNAL = (
    "2001:0:7f00:1::f7f7:f7f7",     # server 127.0.0.1
    "2001:0:a9fe:a9fe::f7f7:f7f7",  # server 169.254.169.254, the cloud metadata address
    "2001:0:a00:1::f7f7:f7f7",      # server 10.0.0.1
)

# is_global disagrees between 3.11 and 3.12 for 6to4, so a gate keyed on it alone would
# refuse a globally routable address on one interpreter and allow it on the other.
# `2001:0:808:404::f7f7:f7f7` is Teredo through 8.8.4.4 to 8.8.8.8: every address it
# carries is global, so it is the row that separates "judge all of them" from "refuse
# the whole range", which is what deleting the Teredo branch would do.
GLOBAL = (
    "8.8.8.8",
    "2606:4700:4700::1111",
    "::ffff:8.8.8.8",
    "2002:808:808::",
    "2001:0:808:404::f7f7:f7f7",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address", [*UNFLAGGED, *TUNNELLED_TO_PRIVATE, *TEREDO_SERVER_IS_INTERNAL]
)
async def test_an_address_the_flag_ladder_misses_is_refused_every_way_in(vetting, address):
    """The classifier was allow-unless-flagged, and three ranges tripped no flag.

    Reachable three ways, all driven here: a catalog icon URL whose host resolves into
    the range, a literal address in a chat message reaching `_download_remote_url`, and a
    redirect hop, which is pre-checked without DNS because an address-shaped host never
    reaches a resolver.

    On a Tailscale- or CGNAT-connected host the first row is the whole tailnet.
    """
    handler = vetting()
    literal = f"[{address}]" if ":" in address else address

    with dns_answering("tailnet.example.com", [address]):
        by_name = handler._validated_ips_for_host("tailnet.example.com", 443)
        through_the_gate = await handler._is_safe_url("https://tailnet.example.com/x")

    assert by_name is None, by_name
    assert through_the_gate is False
    assert handler._validated_ips_for_host(address, 443) is None
    assert handler._hop_is_refused(f"https://{literal}/next.png") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("address", GLOBAL)
async def test_a_globally_routable_address_is_still_fetchable(vetting, address):
    """The inversion must not be satisfied by refusing everything.

    `::ffff:8.8.8.8` and `2002:808:808::` wrap a global IPv4 and are the cases that
    separate "unwrap and judge the embedded address" from "refuse all tunnelled forms".
    """
    import ipaddress

    handler = vetting()
    expected = [ipaddress.ip_address(address).compressed]

    with dns_answering("public.example.com", [address]):
        assert handler._validated_ips_for_host("public.example.com", 443) == expected
        assert await handler._is_safe_url("https://public.example.com/x") is True


@pytest.mark.parametrize(
    ("address", "refused"),
    [
        ("2001:0:7f00:1::f7f7:f7f7", True),
        ("2001:0:808:404::80ff:fffe", True),
        ("2001:0:808:404::f7f7:f7f7", False),
    ],
    ids=["server-is-loopback", "client-is-loopback", "both-are-global"],
)
def test_a_wrapper_is_allowed_only_when_every_address_it_carries_is(address, refused):
    """The rule is a fold over what the wrapper carries, not a pick of one field.

    Teredo carries two. The classifier read the second, so the first -- the tunnel
    server, equally attacker-chosen -- was never judged, and a range `is_private`
    already refused became reachable. Both single-field rules are wrong in one direction
    each, and the three rows here are what separates them from the fold: a rule that
    reads only the server passes rows one and three and fails row two, a rule that reads
    only the client fails row one, and only judging both passes all three.

    Stated on `_address_refusal` rather than through a fetch because the next wrapper
    format with three embedded addresses has to be handled by the same fold, and this is
    where that is decided.
    """
    import ipaddress

    from open_webui_openrouter_pipe.storage.multimodal import _address_refusal

    reason = _address_refusal(ipaddress.ip_address(address))

    assert (reason is not None) is refused, reason


def test_one_unflagged_address_among_public_ones_blocks_the_whole_host(vetting):
    """The gate is all-or-nothing per host, and the new ranges join that rule."""
    handler = vetting()
    with dns_answering("mixed.example.com", [STUB_PUBLIC_IP, "100.64.0.1"]):
        assert handler._validated_ips_for_host("mixed.example.com", 443) is None


# ── one parser decides the host and the port ───────────────────────────────

# yarl parses a port with int(), which accepts a sign, surrounding space, an underscore
# separator and non-ASCII digits. urlparse.port rejects all four with ValueError. The
# gate reads yarl, so all four passed it and then exploded in the pinned-request builder
# -- out of `_download_remote_url`, whose docstring promises it never raises, on URLs
# `streaming_core.py` takes from model output.
PORTS_ONLY_YARL_ACCEPTS = (
    ("https://cdn.example.com:+443/i.png", 443),
    ("https://cdn.example.com: 443/i.png", 443),
    ("https://cdn.example.com:8_0/i.png", 80),
    ("https://cdn.example.com:\u0661\u0662/i.png", 12),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("url", "port"), PORTS_ONLY_YARL_ACCEPTS)
async def test_a_port_only_yarl_accepts_reaches_a_pinned_request(vetting, url, port):
    """Two ports are asserted across the rows, so a constant satisfies neither."""
    from urllib.parse import urlparse

    with pytest.raises(ValueError):
        _ = urlparse(url).port

    handler = vetting()
    request_url, headers, extensions = await handler._prepare_pinned_request(url)

    assert headers == {"Host": f"cdn.example.com:{port}"}
    assert extensions == {"sni_hostname": "cdn.example.com"}
    assert STUB_PUBLIC_IP in request_url, request_url


@pytest.mark.asyncio
async def test_a_download_of_a_url_urlparse_rejects_returns_rather_than_raises(
    pipe_instance_async,
):
    """`_download_remote_url` says it never raises; the second parser made it.

    The retries are turned off so the assertion is about the return value rather than
    about how long a connection refusal takes to give up.
    """
    pipe_instance_async.valves.REMOTE_DOWNLOAD_MAX_RETRIES = 0
    pipe_instance_async.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS = 1
    handler = pipe_instance_async._multimodal_handler
    assert await handler._download_remote_url("https://cdn.example.com:+443/f.bin") is None


@pytest.mark.asyncio
async def test_an_idn_host_pins_the_name_the_gate_validated(vetting):
    """`Host:` and SNI came from `urlparse.hostname`, which is not IDNA-encoded.

    The address was validated for yarl's `raw_host`, so for an international name the two
    disagreed and the header could not even be encoded -- a fetch that fails with nothing
    in the log to explain it.
    """
    handler = vetting()
    request_url, headers, extensions = await handler._prepare_pinned_request(
        "https://b\u00fccher.example/i.png"
    )

    assert headers == {"Host": "xn--bcher-kva.example"}
    assert extensions == {"sni_hostname": "xn--bcher-kva.example"}
    headers["Host"].encode("ascii")
    assert STUB_PUBLIC_IP in request_url, request_url


@pytest.mark.asyncio
async def test_the_pinned_url_carries_the_query_exactly_as_it_was_given(vetting):
    """Rewriting the host must not re-encode anything else.

    A presigned URL carries `%2F` and `%3A` in its query and its signature covers them
    verbatim, so a builder that hands the URL through yarl's default (decoding) parse
    breaks every signed download while looking equivalent.
    """
    handler = vetting()
    query = "X-Amz-Credential=AKIA%2F20260101%2Fus-east-1%2Fs3%2Faws4_request&n=a%20b"
    request_url, _headers, _extensions = await handler._prepare_pinned_request(
        f"https://cdn.example.com/o.bin?{query}"
    )

    assert request_url.endswith(f"?{query}"), request_url


@pytest.mark.asyncio
async def test_a_userinfo_url_keeps_its_credentials_when_the_host_is_pinned(vetting):
    handler = vetting()
    request_url, headers, _extensions = await handler._prepare_pinned_request(
        "https://user:pa%40ss@cdn.example.com:8443/o.bin"
    )

    assert request_url.startswith(f"https://user:pa%40ss@{STUB_PUBLIC_IP}:8443/"), request_url
    assert headers == {"Host": "cdn.example.com:8443"}


@pytest.mark.asyncio
async def test_an_ipv6_origin_is_told_the_host_it_was_asked_for(pipe_instance_async):
    """The Host header is what a virtual-hosted origin routes on, and it has to parse.

    The pin puts the validated IP in the URL and moves the caller's host into `Host`,
    where an IPv6 literal needs brackets (RFC 7230 s5.4). Unbracketed, the far end reads
    `2606:4700:4700::1111:41234`, which parses as neither a host nor a host and a port --
    yarl, which is the parser this package reads hosts with, raises on it.

    Driven through `_download_remote_url`, which is the caller that pins: the icon and
    release fetches go through the vetted CONNECTOR instead, where aiohttp writes its own
    Host from the URL and this expression is never reached. Measured: with the brackets
    removed, an icon fetch to the same address still passes, so a test written against
    that path proves nothing about this.

    What is asserted is what the ORIGIN received, parsed back to a host and a port
    rather than compared as a string: brackets are a means, and a request naming some
    other host would satisfy a literal comparison written to match it.
    """
    v6 = "2606:4700:4700::1111"
    origin = LocalServer(bind="::1")
    origin.route("/asset.bin", _png_response)
    await origin.start()

    pipe = pipe_instance_async
    pipe.valves.ALLOW_INSECURE_HTTP = True
    pipe.valves.ALLOW_INSECURE_HTTP_HOSTS = f"[{v6}]:{origin.port}"
    pipe.valves.REMOTE_DOWNLOAD_MAX_RETRIES = 0
    try:
        with address_routed_to_loopback(v6, "::1"):
            result = await pipe._multimodal_handler._download_remote_url(
                f"http://[{v6}]:{origin.port}/asset.bin"
            )
    finally:
        await origin.stop()

    assert result is not None and result["data"] == SECRET_PNG, (
        "the request never landed, so the header this test is about was never sent"
    )
    [(path, host_header)] = origin.requests
    assert path == "/asset.bin"
    parsed = yarl.URL(f"//{host_header}")
    assert (parsed.host, parsed.explicit_port) == (v6, origin.port), (
        f"the origin was sent Host: {host_header!r}, which does not name the host and "
        "port the caller asked for"
    )


@pytest.mark.parametrize(
    ("url", "ip", "host", "port"),
    [
        ("https://example.com/o.bin", STUB_PUBLIC_IP, "example.com", None),
        ("https://example.com:8443/o.bin", STUB_PUBLIC_IP, "example.com", 8443),
        ("https://example.com:443/o.bin", STUB_PUBLIC_IP, "example.com", 443),
        (
            "https://[2606:4700:4700::1111]/o.bin",
            "2606:4700:4700::1111",
            "2606:4700:4700::1111",
            None,
        ),
        (
            "https://[2606:4700:4700::1111]:8443/o.bin",
            "2606:4700:4700::1111",
            "2606:4700:4700::1111",
            8443,
        ),
    ],
    ids=["name", "name-with-a-port", "name-with-the-default-port", "v6", "v6-with-a-port"],
)
def test_the_pin_carries_a_host_header_that_parses_back_to_the_caller_url(
    url, ip, host, port
):
    """Both branches of the header, over both address families.

    The end-to-end test above can only reach the branch that appends a port, because a
    server on the default port is not something a test can bind. Five rows over two
    families and three port shapes is also what stops a constant satisfying this: no
    single string is right for more than one of them.

    `:443` is here for the reason `origin().raw_authority` is not used to build this --
    yarl drops a port equal to the scheme default, so that spelling would send a
    different Host than the caller wrote.
    """
    handler = vetting_handler()

    request_url, headers, extensions = handler._build_pinned_request(url, ip)

    parsed = yarl.URL(f"//{headers['Host']}")
    assert (parsed.host, parsed.explicit_port) == (host, port)
    assert yarl.URL(request_url).host == ip
    assert extensions == {"sni_hostname": host}, (
        "SNI takes the bare literal: CPython suppresses the extension for an IP and "
        "verifies against IP SANs, and a bracketed string is neither a name nor an IP"
    )


@pytest.mark.asyncio
async def test_an_ipv6_pin_is_bracketed_in_the_request_url(vetting):
    handler = vetting()
    with dns_answering("v6.example.com", ["2606:4700:4700::1111"]):
        request_url, headers, _extensions = await handler._prepare_pinned_request(
            "https://v6.example.com/o.bin"
        )

    assert request_url == "https://[2606:4700:4700::1111]/o.bin", request_url
    assert headers == {"Host": "v6.example.com"}


# ── a valve the operator flips while the transport is warm ────────────────────


def _pooled(handler) -> int:
    """Keep-alive connections the transport is currently holding."""
    connector = getattr(handler._vetted_http_session, "_connector", None)
    if connector is None:
        return 0
    return sum(len(conns) for conns in connector._conns.values())


async def _keep_alive_icon(_request: web.Request):
    return web.Response(body=SECRET_PNG, content_type="image/png")


async def _closing_icon(_request: web.Request):
    return web.Response(
        body=SECRET_PNG, content_type="image/png", headers={"Connection": "close"}
    )


@pytest.mark.asyncio
async def test_turning_protection_back_on_severs_a_warm_keep_alive_transport(vetting):
    """The valve change has to sever the transport, not scrub one cache inside it.

    `clear_dns_cache()` empties `_cached_hosts` and does not touch `connector._conns`,
    and aiohttp never consults a resolver for a pooled connection. So a connection opened
    while the gate was off carried the request AFTER the flip: same handler, same host
    resolving only to 127.0.0.1, protection on -- reached, with `getaddrinfo` never
    called again.

    The predecessor served `Connection: close` and then asserted `pooled == 0`, which
    guards the TEST's validity rather than the product's: it engineered out the one
    condition under which the defect appears. Here the server keeps the connection alive
    and the pool is asserted NON-empty before the flip, so the confound is present on
    purpose.
    """
    origin = LocalServer()
    origin.route("/icon.png", _keep_alive_icon)
    await origin.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{origin.port}",
        ENABLE_SSRF_PROTECTION=False,
    )
    url = f"http://{CDN}:{origin.port}/icon.png"
    try:
        with dns_answering(CDN, ["127.0.0.1"]):
            while_off = await handler._fetch_image_as_data_url(url)
            warm_session = handler._vetted_http_session
            pooled_before = _pooled(handler)
            handler.valves.ENABLE_SSRF_PROTECTION = True
            with counted_dns() as lookups:
                after_flip = await handler._fetch_image_as_data_url(url)
            fresh_session = handler._vetted_http_session
    finally:
        await origin.stop()

    assert while_off is not None, (
        "the fetch did not succeed with protection off, so the flip below has nothing "
        "to change and this test proves nothing"
    )
    assert pooled_before >= 1, (
        "no connection was pooled before the flip, so the reuse this test exists to "
        "catch could not have happened and a pass means nothing"
    )
    assert after_flip is None, (
        "the request after the flip still reached the private address, over a "
        "connection opened while the gate was off"
    )
    assert fresh_session is not warm_session, (
        "the same session object survived the flip, so its pool and its DNS cache did too"
    )
    assert warm_session.closed, "the transport built without the gate was left open"
    assert lookups.count(CDN) >= 1, (
        f"the host was never resolved after the flip ({lookups}), so the refusal did "
        "not come from the gate"
    )


@pytest.mark.asyncio
async def test_a_lookup_in_flight_at_the_flip_cannot_answer_a_later_request(vetting):
    """The second mechanism: an answer that lands in the cache AFTER it was cleared.

    `TCPConnector._resolve_host_with_throttle` writes into `_cached_hosts` once the
    resolver returns, so a lookup that began while the gate was off, and finished after
    the flip had already run `clear_dns_cache()`, poisons the cache for the next
    `_VETTED_DNS_CACHE_SECONDS`. That reaches FRESH connections, not only pooled ones,
    which is why a pool sweep next to `clear_dns_cache()` would not have closed it.

    The server sends `Connection: close`, deliberately: with the pool empty, the only
    thing that can carry the last request is the cache entry.
    """
    origin = LocalServer()
    origin.route("/icon.png", _closing_icon)
    await origin.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=f"{CDN}:{origin.port}",
        ENABLE_SSRF_PROTECTION=False,
    )
    url = f"http://{CDN}:{origin.port}/icon.png"
    try:
        with stalled_dns(CDN, ["127.0.0.1"]) as (began, release):
            in_flight = asyncio.create_task(handler._fetch_image_as_data_url(url))
            deadline = time.monotonic() + 10
            while not began.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            started_before_the_flip = began.is_set()

            handler.valves.ENABLE_SSRF_PROTECTION = True
            concurrent = await handler._fetch_image_as_data_url(
                "https://b-side.invalid/x.png"
            )

            release.set()
            await asyncio.gather(in_flight, return_exceptions=True)
            after = await handler._fetch_image_as_data_url(url)
            pooled_after = _pooled(handler)
    finally:
        await origin.stop()

    assert started_before_the_flip, (
        "the lookup never started, so no answer was in flight when the valve changed "
        "and this test staged nothing"
    )
    assert concurrent is None
    assert pooled_after == 0, (
        f"{pooled_after} connections were pooled, so a keep-alive connection rather "
        "than a cached address could be what carried the last request"
    )
    assert after is None, (
        "the request after the flip reached the private address from a cache entry "
        "written by a lookup that began before it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("starts_on", "flips", "reaches_the_second_hop"),
    [(True, False, False), (False, False, True), (False, True, False)],
    ids=["on-throughout", "off-throughout", "flipped-on-mid-chain"],
)
async def test_a_hop_taken_after_the_valve_flips_is_gated_by_the_new_setting(
    routed, vetting, starts_on, flips, reaches_the_second_hop
):
    """The two halves of the gate must not hold different opinions of the mode.

    The session was acquired once per CHAIN, and the resolver inside its connector froze
    `protection` when it was built. `_hop_is_refused` reads the valve live, but for a
    host that is not address-shaped it returns False and defers to that resolver -- so
    the deciding half was the stale one. An operator turning protection back on while a
    chain was open left every remaining hop validated by nothing, and the origin chooses
    how long to hold hop one open, so the window is as wide as the total budget.

    Three arms, because one refusal proves nothing on its own: `on-throughout` is the
    control that this exact hop IS refusable, `off-throughout` is the control that it is
    otherwise reachable, and only the contrast between them makes the third arm's
    verdict about the flip rather than about the address.

    The second host is a NAME, deliberately. An address-shaped hop is refused by the
    live pre-check and never reaches the resolver at all, so a test written with a
    literal would pass against the defect.
    """
    internal = LocalServer()
    internal.route("/loot", _png_response)
    await internal.start()

    cdn = LocalServer()
    began = asyncio.Event()
    release = asyncio.Event()

    async def _stalls_then_redirects(_request: web.Request):
        began.set()
        await release.wait()
        raise web.HTTPFound(f"http://{ORIGIN}:{internal.port}/loot")

    cdn.route("/start", _stalls_then_redirects)
    await cdn.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{cdn.port}", f"{ORIGIN}:{internal.port}"
        ),
        ENABLE_SSRF_PROTECTION=starts_on,
    )
    try:
        with dns_answering(ORIGIN, ["127.0.0.1"]):
            chain = asyncio.create_task(
                handler._fetch_image_as_data_url(f"http://{CDN}:{cdn.port}/start")
            )
            await asyncio.wait_for(began.wait(), timeout=10)
            if flips:
                handler.valves.ENABLE_SSRF_PROTECTION = True
            valve_at_the_second_hop = bool(handler.valves.ENABLE_SSRF_PROTECTION)
            release.set()
            result = await asyncio.wait_for(chain, timeout=10)
    finally:
        release.set()
        await cdn.stop()
        await internal.stop()

    assert [path for path, _host in cdn.requests] == ["/start"], (
        "the first hop never completed, so nothing was in flight when the valve was "
        "read and this arm staged nothing"
    )
    served = [path for path, _host in internal.requests]
    if reaches_the_second_hop:
        assert served == ["/loot"], served
        assert _decoded(result) == SECRET_PNG
    else:
        assert served == [], (
            f"the second hop reached the private address with the valve reading "
            f"{valve_at_the_second_hop} at the time it was taken"
        )
        assert result is None


def test_a_second_event_loop_does_not_inherit_the_first_loops_transport():
    """A session whose loop has closed is not `closed`, so nothing rebuilt it.

    Measured on the shipped guard: loop A's fetch returns data, loop B's returns None for
    ever off the same session object, and `transport_session_state()` reports "active"
    for a transport that cannot complete a request. `pipe.py` enforces exactly this
    invariant in five places for the queue, the lock and the worker.

    Driven through the real fetch on both loops rather than through `_vetted_session`, and
    `sessions[0].closed` is asserted False so the reason the shipped guard never fired is
    part of the record.
    """
    handler = vetting_handler(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=CDN,
        ENABLE_SSRF_PROTECTION=False,
    )
    results: list[str | None] = []
    sessions: list[Any] = []

    async def _round(shut_down: bool) -> None:
        origin = LocalServer()
        origin.route("/icon.png", _keep_alive_icon)
        await origin.start()
        try:
            with dns_answering(CDN, ["127.0.0.1"]):
                results.append(
                    await handler._fetch_image_as_data_url(
                        f"http://{CDN}:{origin.port}/icon.png"
                    )
                )
            sessions.append(handler._vetted_http_session)
        finally:
            if shut_down:
                await handler.aclose()
            await origin.stop()

    asyncio.run(_round(False))
    state_after_the_loop_closed = handler.transport_session_state()
    closed_flag_when_the_loop_died = sessions[0].closed
    asyncio.run(_round(True))

    assert results[0] is not None, "the first loop never fetched anything"
    assert results[1] is not None, (
        "the second loop reused a transport bound to a loop that had closed, so every "
        "fetch on it fails for the life of the process"
    )
    assert sessions[1] is not sessions[0], "the dead loop's session was handed out again"
    assert closed_flag_when_the_loop_died is False, (
        "the shipped guard keys on `session.closed`, and this is why it never fired"
    )
    assert state_after_the_loop_closed == "none", (
        f"the dashboard reported {state_after_the_loop_closed!r} for a transport that "
        "cannot complete a request"
    )


@pytest.mark.asyncio
async def test_the_pipes_shutdown_closes_the_transport(pipe_instance_async):
    """The hook, driven through the real shutdown rather than asserted to exist."""
    pipe = pipe_instance_async
    handler = pipe._multimodal_handler
    with contextlib.suppress(Exception):
        await handler._fetch_image_as_data_url("https://cdn.example.com/icon.png")
    session = handler._vetted_http_session
    assert session is not None and not session.closed

    await pipe.close()

    assert session.closed, (
        "the pipe shut down without closing the transport's session, so a hot reload "
        "leaks one per generation"
    )


@pytest.mark.asyncio
async def test_the_release_metadata_fetch_refuses_a_redirect_into_the_private_network(
    routed, vetting
):
    """The other half of the same self-update flow.

    `_http_get_json` reads the release metadata, and the release DIGEST comes out of that
    response -- so an attacker who can answer it can name both the bytes and the hash they
    must match. Its exemption said the URL is built by its one caller, which is true of the
    first hop and false of hops two through ten: it followed redirects on its own.
    """
    us = _us()
    internal = LocalServer()

    async def _secret(_request: web.Request):
        return web.json_response({"tag_name": "v9.9.9", "assets": []})

    internal.route("/releases/latest", _secret)
    await internal.start()

    github = LocalServer()

    async def _redirect(_request: web.Request):
        raise web.HTTPFound(f"http://127.0.0.1:{internal.port}/releases/latest")

    github.route("/releases/latest", _redirect)
    await github.start()

    handler = vetting(
        ALLOW_INSECURE_HTTP=True,
        ALLOW_INSECURE_HTTP_HOSTS=_allowlist(
            f"{CDN}:{github.port}", f"127.0.0.1:{internal.port}"
        ),
    )
    try:
        with pytest.raises(us.UpdateError) as raised:
            await us._http_get_json(
                f"http://{CDN}:{github.port}/releases/latest", vetting=handler
            )
    finally:
        await github.stop()
        await internal.stop()

    assert raised.value.code == "offline"
    assert internal.requests == [], (
        "the metadata fetch followed a redirect into the private network; the release "
        "digest would have come from whatever answered"
    )


class _CountingPage:
    """A maker page generated on demand, reporting what was actually pulled off it."""

    def __init__(self, total: int, prefix: bytes = b"") -> None:
        self.total = total
        self.served = 0
        self._prefix = prefix

    async def iter_chunked(self, size: int):
        if self._prefix:
            self.served += len(self._prefix)
            yield self._prefix
        while self.served < self.total:
            take = min(size, self.total - self.served)
            self.served += take
            yield b"x" * take


class _PageResponse:
    status = 200

    def __init__(self, body: _CountingPage, headers: dict[str, str] | None = None) -> None:
        self.content = body
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _PageSession:
    closed = False

    def __init__(self, response: _PageResponse) -> None:
        self._response = response
        self.requested: list[str] = []

    def get(self, url, **_kwargs):
        self.requested.append(str(url))
        return self._response

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("total", [64 * 1024 * 1024, 256 * 1024 * 1024])
async def test_an_endless_maker_page_is_cut_off_at_the_byte_cap(vetting, total):
    """`await resp.text()` buffered whatever the far end sent.

    The only bound was the 15s deadline, and a 200 MiB page arrives inside it: measured
    at 0.5s and a 421 MiB peak. `_build_maker_profile_image_mapping` runs these ten at a
    time, and `_vetted_get` follows `Location`, so the body is chosen by whatever the
    redirect lands on rather than by openrouter.ai.

    Both numbers are of the process rather than of a log line, and both body sizes are
    driven, because the defect is that they track the body size.
    """
    page = _CountingPage(total)
    session = _PageSession(_PageResponse(page, {"Content-Type": "text/html"}))
    handler = vetting()
    handler._vetted_http_session = session

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = await handler._fetch_maker_profile_image_url("acme")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result is None
    assert session.requested == ["https://openrouter.ai/acme"], session.requested
    assert page.served <= mm._MAKER_PAGE_MAX_BYTES + 256 * 1024, (
        f"pulled {page.served} bytes off a {total}-byte page"
    )
    assert peak <= mm._MAKER_PAGE_MAX_BYTES + 1024 * 1024, (
        f"peak {peak} bytes for a {total}-byte page"
    )


@pytest.mark.asyncio
async def test_a_declared_oversize_maker_page_is_refused_before_a_byte_is_read(vetting):
    """A truthful Content-Length is a cheap early-out; the running total is the guard."""
    page = _CountingPage(64 * 1024 * 1024)
    session = _PageSession(
        _PageResponse(
            page,
            {
                "Content-Type": "text/html",
                "Content-Length": str(64 * 1024 * 1024),
            },
        )
    )
    handler = vetting()
    handler._vetted_http_session = session

    assert await handler._fetch_maker_profile_image_url("acme") is None
    assert page.served == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image", ["https://cdn.example.com/a.png", "https://cdn.example.com/b.png"]
)
async def test_a_maker_page_under_the_cap_still_yields_its_og_image(vetting, image):
    """The cap must not be a refusal of every page: a real one still parses.

    Two images, so a hardcoded return value satisfies neither row.
    """
    markup = f'<html><head><meta property="og:image" content="{image}"></head></html>'
    page = _CountingPage(len(markup.encode()), markup.encode())
    session = _PageSession(_PageResponse(page, {"Content-Type": "text/html"}))
    handler = vetting()
    handler._vetted_http_session = session

    assert await handler._fetch_maker_profile_image_url("acme") == image


@pytest.mark.asyncio
async def test_a_maker_page_that_is_not_utf8_still_yields_its_og_image(vetting):
    """A page carrying invalid UTF-8 must not lose the ASCII URL sitting next to it."""
    markup = (
        b'<html><head><meta name="x" content="\xff\xfe">'
        b'<meta property="og:image" content="https://cdn.example.com/c.png">'
        b"</head></html>"
    )
    page = _CountingPage(len(markup), markup)
    session = _PageSession(_PageResponse(page, {"Content-Type": "text/html"}))
    handler = vetting()
    handler._vetted_http_session = session

    assert (
        await handler._fetch_maker_profile_image_url("acme")
        == "https://cdn.example.com/c.png"
    )
