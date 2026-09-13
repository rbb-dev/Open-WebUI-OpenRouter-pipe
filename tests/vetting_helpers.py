"""Fixtures for driving the pipe's vetted HTTP transport against real sockets.

`MultimodalHandler._vetted_get` is the single place the package connects to an address
somebody else chose. Proving what it does needs three things a response double cannot
supply: a server that really redirects, a name that resolves to one address for the
vetting and another for the connector, and a Host header the far end can report back.

`conftest` already supplies half of that. Its `_stub_getaddrinfo` answers every name
from a rule -- loopback names resolve to 127.0.0.1, `.test`/`.invalid` raise NXDOMAIN,
everything else is one fixed public address -- and `_no_real_network` refuses any
connect that is not loopback.

The transport owns its own session, whose connector resolves through the SSRF gate, so
a test cannot hand it a second resolver that disagrees. That is the point: there is no
longer a second answer for the connection to prefer. `rebinding_dns` therefore stages
the disagreement one layer lower, in `getaddrinfo` itself, which is where a real rebind
happens -- and the control arm a test contrasts it with has to be a client that really
re-resolves. A connector carrying a hardcoded name-to-address map calls `getaddrinfo`
zero times, so staging a rebind around it changes nothing and the contrast is with the
map rather than with the gate.
"""

from __future__ import annotations

import logging
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest_asyncio
from aiohttp import web

from open_webui_openrouter_pipe.storage.multimodal import MultimodalHandler

# conftest._STUB_PUBLIC_IP: the address every non-loopback name resolves to in tests.
STUB_PUBLIC_IP = "93.184.216.34"


def vetting_handler(**valve_over: Any) -> MultimodalHandler:
    """A real handler with only the valves the address gate reads."""
    valves = SimpleNamespace(
        ALLOW_INSECURE_HTTP=False,
        ALLOW_INSECURE_HTTP_HOSTS="",
        ENABLE_SSRF_PROTECTION=True,
    )
    for key, value in valve_over.items():
        setattr(valves, key, value)
    return MultimodalHandler(logging.getLogger("tests.vetting"), valves)


@pytest_asyncio.fixture()
async def vetting():
    """Build handlers, and close whatever transport they opened when the test ends.

    `_vetted_get` lazily opens a `ClientSession` on the handler, because the connector
    that session carries is the thing doing the vetting. Nothing else owns it, so a test
    that drives the transport and walks away leaks it and prints "Unclosed client
    session" against some later, unrelated test.
    """
    made: list[MultimodalHandler] = []

    def _make(**valve_over: Any) -> MultimodalHandler:
        handler = vetting_handler(**valve_over)
        made.append(handler)
        return handler

    yield _make
    for handler in made:
        await handler.aclose()


class LocalServer:
    """An aiohttp app on a loopback address with an ephemeral port, plus what it was asked for.

    `bind` exists so a test can drive the IPv6 path end to end: the Host header a pinned
    request carries is built from the URL's host, and only a URL with an IPv6 literal in
    it produces the form that has to be bracketed.
    """

    def __init__(self, bind: str = "127.0.0.1") -> None:
        self.app = web.Application()
        self.requests: list[tuple[str, str | None]] = []
        self._runner: web.AppRunner | None = None
        self.bind = bind
        self.port = 0
        self.ssl_context: Any | None = None

    def route(self, path: str, handler) -> None:
        async def _wrapped(request: web.Request):
            self.requests.append((request.path, request.headers.get("Host")))
            return await handler(request)

        self.app.router.add_get(path, _wrapped)

    async def start(self) -> LocalServer:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.bind, 0, ssl_context=self.ssl_context)
        await site.start()
        self.port = self._runner.addresses[0][1]
        return self

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


def single_san_cert(name: str, into: Any) -> dict[str, Any]:
    """A self-signed leaf whose ONLY subjectAltName is `name`.

    So a second hostname on the same address has no certificate covering it, and any
    connection that serves it anyway did so without checking.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = into / "cert.pem"
    key_path = into / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return {"cert": cert_path, "key": key_path}


@contextmanager
def _socket_resolution_only():
    """Make aiohttp resolve through `socket.getaddrinfo`, so staged DNS is seen.

    With the production dependency set installed, `aiodns` is present and aiohttp's
    `DefaultResolver` is `AsyncResolver`, which asks c-ares directly and never calls
    `socket.getaddrinfo`. Every DNS staging helper below patches `getaddrinfo`, so under
    production dependencies the staging silently does nothing and the test performs a real
    lookup of a name that does not exist. That is how two rebind guards passed for months
    in a lean environment and failed the moment CI installed the full set.

    Forcing the threaded resolver does not weaken what these tests prove: the control
    session still re-resolves, which is the behaviour being contrasted with the vetted
    transport's single gated lookup.
    """
    import aiohttp.connector as _connector
    import aiohttp.resolver as _resolver

    threaded = _resolver.ThreadedResolver
    saved = [
        (module, getattr(module, "DefaultResolver", None))
        for module in (_resolver, _connector)
    ]
    for module, current in saved:
        if current is not None:
            module.DefaultResolver = threaded
    try:
        yield
    finally:
        for module, current in saved:
            if current is not None:
                module.DefaultResolver = current


@contextmanager
def rebinding_dns(name: str, answers: list[str]):
    """`name` resolves to a different address on each successive lookup.

    A real DNS rebind, staged where one happens: in the resolver, not in a redirect and
    not in a second client-side resolver. Everything else keeps conftest's answers.

    Each entry is consumed once and the last one repeats, so `[public, private]` is the
    classic shape -- vets clean, then answers privately for whoever asks next.
    """
    import socket as _socket

    installed = _socket.getaddrinfo
    remaining = list(answers)

    def _resolve(host, port, *args, **kwargs):
        if str(host or "").strip().strip("[]").lower() != name.lower():
            return installed(host, port, *args, **kwargs)
        ip = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        try:
            resolved_port = int(port) if port is not None else 0
        except (TypeError, ValueError):
            resolved_port = 0
        return [(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, resolved_port))]

    _socket.getaddrinfo = _resolve
    try:
        with _socket_resolution_only():
            yield
    finally:
        _socket.getaddrinfo = installed


@contextmanager
def dns_answering(name: str, addresses: list[str]):
    """`name` resolves to ALL of `addresses` at once, as a multi-A record does."""
    import socket as _socket

    installed = _socket.getaddrinfo

    def _resolve(host, port, *args, **kwargs):
        if str(host or "").strip().strip("[]").lower() != name.lower():
            return installed(host, port, *args, **kwargs)
        try:
            resolved_port = int(port) if port is not None else 0
        except (TypeError, ValueError):
            resolved_port = 0
        return [
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, resolved_port))
            for ip in addresses
        ]

    _socket.getaddrinfo = _resolve
    try:
        with _socket_resolution_only():
            yield
    finally:
        _socket.getaddrinfo = installed


@contextmanager
def stalled_dns(name: str, addresses: list[str], wait_seconds: float = 10.0):
    """`name` blocks inside `getaddrinfo` until the caller releases it.

    aiohttp writes a resolution into the connector's DNS cache AFTER the resolver
    returns, so staging "a lookup that was already in flight when the valve changed"
    needs the block to happen there and nowhere else. Yields the two events: one set as
    soon as the lookup starts, one the caller sets to let it finish.

    Restored in `finally`, like every other rebind in this module -- a test module that
    assigns `socket.getaddrinfo` directly hands real DNS back to everything after it, and
    `test_suite_is_hermetic.py` fails on exactly that.
    """
    import socket as _socket
    import threading

    installed = _socket.getaddrinfo
    began = threading.Event()
    release = threading.Event()

    def _resolve(host, port, *args, **kwargs):
        if str(host or "").strip().strip("[]").lower() != name.lower():
            return installed(host, port, *args, **kwargs)
        began.set()
        release.wait(wait_seconds)
        try:
            resolved_port = int(port) if port is not None else 0
        except (TypeError, ValueError):
            resolved_port = 0
        return [
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, resolved_port))
            for ip in addresses
        ]

    _socket.getaddrinfo = _resolve
    try:
        yield began, release
    finally:
        release.set()
        _socket.getaddrinfo = installed


@contextmanager
def counted_dns():
    """Every `getaddrinfo` call this block makes, by host."""
    import socket as _socket

    installed = _socket.getaddrinfo
    calls: list[str] = []

    def _counting(host, port, *args, **kwargs):
        calls.append(str(host))
        return installed(host, port, *args, **kwargs)

    _socket.getaddrinfo = _counting
    try:
        yield calls
    finally:
        _socket.getaddrinfo = installed


@contextmanager
def address_routed_to_loopback(literal: str, loopback: str = "127.0.0.1"):
    """Make one global address reachable, as the real internet would.

    A pinned request carries an IP literal in the host position, and aiohttp does not
    consult its resolver for those -- it dials the literal. So a test where the request
    has to LAND cannot fake it at the resolver; it has to be faked one layer down, at
    the socket. Translating here is also the stronger evidence: the process really did
    dial the address the gate validated.

    `loopback` has to match the family of the socket being connected, because an
    AF_INET6 socket cannot be handed an IPv4 address: an IPv6 literal translates to
    `::1` and the server binds there.
    """
    installed_connect = socket.socket.connect
    installed_connect_ex = socket.socket.connect_ex

    def _translate(address):
        if isinstance(address, tuple) and address and address[0] == literal:
            return (loopback, *address[1:])
        return address

    def _connect(self, address):
        return installed_connect(self, _translate(address))

    def _connect_ex(self, address):
        return installed_connect_ex(self, _translate(address))

    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    try:
        yield
    finally:
        socket.socket.connect = installed_connect
        socket.socket.connect_ex = installed_connect_ex


@contextmanager
def public_ip_routed_to_loopback():
    """The stubbed public address every non-loopback name resolves to, routed home."""
    with address_routed_to_loopback(STUB_PUBLIC_IP):
        yield
