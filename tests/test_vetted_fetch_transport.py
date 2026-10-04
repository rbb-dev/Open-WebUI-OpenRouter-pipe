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
    open_live_listener,
    single_san_cert,
    public_ip_routed_to_loopback,
    rebinding_dns,
    stalled_dns,
    vetting,
    vetting_handler,
)

CDN = "cdn.example.com"
ORIGIN = "origin.example.com"


def _png(side: int = 6) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def routed():
    """The stubbed public address answers, so a pinned request can actually land."""
    with public_ip_routed_to_loopback():
        yield


# ── one resolver outage, one warning ─────────────────────────────────────────


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
