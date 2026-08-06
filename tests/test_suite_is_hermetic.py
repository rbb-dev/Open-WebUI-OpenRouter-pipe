"""The suite must not depend on the network.

Twenty-three test/host pairs were resolving real names -- `openrouter.ai`,
`example.com`, `localhost` -- because the SSRF gate resolves a hostname before fetching
it. One of them went further and issued a real HTTPS request to
`https://openrouter.ai/api/v1/models` on every run, because `pipes()` loads the catalog
and the test set an API key without mocking the endpoint.

Two costs, both observed. A DNS outage during a five-mode battery turned into five red
tests whose failure messages were about video blocks and web-tools filters, with nothing
pointing at the real cause. And the unmocked test's negative arm passed for the wrong
reason: a failed fetch installs nothing, so "every tool is disabled" and "the request
never completed" produced the same answer.

conftest answers every name from a rule instead. This file is the guard that the rule
stays in force -- an autouse fixture is easy to delete or shadow, and the failure mode
when it goes is a suite that looks green until the network is slow.
"""

from __future__ import annotations

import socket

import pytest


def test_name_resolution_is_stubbed():
    """The real resolver must not be reachable from a test.

    Checked by name rather than by object identity: pytest loads the root conftest as
    `conftest`, so importing `tests.conftest` here yields a SECOND module object whose
    functions are not the ones installed. An identity assertion fails while the stub is
    correctly in place, which is a guard that cries wolf until it is deleted.
    """
    assert socket.getaddrinfo.__name__ == "_stub_getaddrinfo", (
        f"socket.getaddrinfo is {socket.getaddrinfo!r}, not the conftest stub, so tests "
        "can reach real DNS and a network outage becomes a red build with a misleading "
        "failure"
    )


def test_a_public_name_resolves_to_the_fixed_stub_address():
    """Deterministic, so no test depends on what a name resolves to today."""
    infos = socket.getaddrinfo("openrouter.ai", 443)
    assert [i[4][0] for i in infos] == ["93.184.216.34"]
    assert infos[0][4][1] == 443, "the requested port was not carried through"


def test_loopback_stays_loopback():
    """The SSRF gate must still see a private address for localhost.

    Mapping everything to one public IP would quietly disarm every test that checks a
    loopback URL is blocked.
    """
    assert [i[4][0] for i in socket.getaddrinfo("localhost", 80)] == ["127.0.0.1"]


@pytest.mark.parametrize(
    "host",
    [
        "nonexistent.example.com",
        "no-such-host.example.com",
        "does-not-exist.example.com",
        "images.example.test",
        "something.invalid",
    ],
    ids=["marker-nonexistent", "marker-no-such-host", "marker-does-not-exist", "tld-test", "tld-invalid"],
)
def test_an_unresolvable_name_still_fails(host):
    """Negative paths need a real NXDOMAIN, not a successful answer.

    Both halves of the rule: the explicit markers, and RFC 2606's reserved TLDs. The
    TLDs matter for speed as much as correctness -- answering `images.example.test`
    turned an instant NXDOMAIN into a refused connection that the retry policy backed
    off over for 35 seconds, past CI's per-test ceiling.
    """
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo(host, 443)


def test_outbound_connections_are_refused_instantly():
    """An unmocked request must fail fast and say so, not hang until a timeout."""
    import time

    sock = socket.socket()
    sock.settimeout(30)
    started = time.monotonic()
    try:
        with pytest.raises(ConnectionRefusedError) as excinfo:
            sock.connect(("93.184.216.34", 443))
    finally:
        sock.close()
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, (
        f"the blocked connection took {elapsed:.1f}s; it must be refused immediately, "
        "or an unmocked request becomes a slow test rather than a loud one"
    )
    assert "not mocked" in str(excinfo.value), (
        "the refusal does not tell the reader that the test made an unmocked request"
    )


def test_loopback_connections_are_still_allowed():
    """Tests that bind a local server or a Redis stub must keep working."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.socket()
        client.settimeout(5)
        client.connect(server.getsockname())
        client.close()
    finally:
        server.close()


def test_no_test_module_opts_out_of_the_stub():
    """A test that restores the real resolver reopens the hole for everything after it."""
    import pathlib
    import re

    offenders = []
    here = pathlib.Path(__file__).resolve().parent
    for path in sorted(here.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^.*getaddrinfo.*$", text, re.M):
            line = match.group(0).strip()
            if line.startswith("#"):
                continue
            if "monkeypatch." in line or "mock.patch" in line or "with patch" in line:
                continue  # reverted for us at teardown
            if re.search(r"socket\.getaddrinfo\s*=", line):
                offenders.append(f"{path.name}: {line}")
    assert not offenders, (
        "these modules rebind socket.getaddrinfo without a mechanism that restores it, "
        "so they either hand real DNS back or shadow the conftest stub for every test "
        f"that follows:\n  " + "\n  ".join(offenders)
    )
