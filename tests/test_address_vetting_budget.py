"""Bounds on the work a request can make this process do before it is sent.

Three properties, one file:

  * a single address check finishes inside a wall-clock budget, and a budget that
    expires means "unsafe" -- exactly what a failed resolution means;
  * the deployment-wide video slot is not held while those checks run; and
  * the depth bound the refusal message advertises is the bound that actually applies,
    on both walkers rather than on the one that happens to run second.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from open_webui_openrouter_pipe.integrations.provider_options import (
    MAX_URL_SCAN_DEPTH,
    UnvettableRequest,
    payload_addresses,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.storage.multimodal import (
    ADDRESS_CHECK_BUDGET_SECONDS,
    ADDRESS_CHECK_SECONDS,
    MultimodalHandler,
)


def _handler(blocking):
    handler = MultimodalHandler.__new__(MultimodalHandler)
    handler.logger = logging.getLogger("address-budget")
    handler._request_ips_blocking = blocking
    return handler


def _adapter(is_safe):
    pipe = MagicMock()
    pipe.logger = logging.getLogger("address-budget")
    pipe._multimodal_handler._is_safe_url = is_safe
    return VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)


def _nested(levels: int) -> dict:
    node: dict = {"video": "https://ok.example/a.mp4"}
    for _ in range(levels):
        node = {"provider": {"options": {"p": node}}}
    return node


# ------------------------------------------------------------- PER LOOKUP ----
@pytest.mark.parametrize("budget", [0.2, 0.5])
@pytest.mark.asyncio
async def test_a_stalled_lookup_gives_up_inside_the_budget_it_was_given(budget):
    """`socket.getaddrinfo` takes as long as the requester's nameserver wants.

    The host comes out of a free-text request field, so the caller chooses the resolver
    and therefore the duration. Two budgets, so a production `wait_for(..., 5)` that
    ignores the argument cannot pass both, and a hardcoded sleep cannot either.
    """
    released = threading.Event()

    def _blocking(_url):
        released.wait(30)
        return ["1.2.3.4"]

    handler = _handler(_blocking)
    started = time.monotonic()
    try:
        verdict = await handler._is_safe_url("https://slow.example/a.mp4", seconds=budget)
        elapsed = time.monotonic() - started
    finally:
        released.set()

    assert verdict is False, "a check that never finished was reported as safe"
    assert budget <= elapsed < budget + 1.0, (
        f"a {budget}s budget took {elapsed:.2f}s"
    )


@pytest.mark.parametrize(
    ("resolved", "expected"), [(["93.184.216.34"], True), (None, False)]
)
@pytest.mark.asyncio
async def test_a_lookup_that_finishes_still_decides_the_answer(resolved, expected):
    """The budget must not become the verdict: an allowed address stays allowed.

    Parametrised over an allowed and a blocked resolution, so a production
    `return False` that "fails closed" everywhere cannot pass.
    """
    assert await _handler(lambda _url: resolved)._is_safe_url("https://x.example/") is expected


@pytest.mark.asyncio
async def test_a_request_full_of_stalled_addresses_stops_at_the_request_budget():
    """Per-lookup alone still lets sixteen addresses cost sixteen budgets in a row.

    The refusal is the point: an expired request budget is a check that did not pass, so
    the request is refused rather than sent unchecked.
    """
    slow = 0.15
    checked: list[str] = []

    async def _is_safe(url, *, seconds=ADDRESS_CHECK_SECONDS):
        checked.append(url)
        await asyncio.sleep(min(slow, max(0.0, seconds)))
        return seconds > 0

    adapter = _adapter(_is_safe)
    payload = {
        "provider": {"options": {"p": {
            "videos": [{"url": f"https://slow{i}.example/c.mp4"} for i in range(16)]
        }}}
    }
    started = time.monotonic()
    with pytest.raises(Exception) as refused:
        await adapter._validate_passthrough_urls(payload, deadline=time.monotonic() + 0.45)
    elapsed = time.monotonic() - started

    assert elapsed < 16 * slow, f"the request budget did not bind: {elapsed:.2f}s"
    assert "unsafe" in str(refused.value).lower(), refused.value
    assert len(checked) < 16, checked


@pytest.mark.asyncio
async def test_the_request_budget_is_shared_rather_than_restarted_per_branch():
    """A budget re-derived inside the recursion is a budget per branch, not per request.

    Two provider slugs each carrying an address: the second check must inherit what the
    first one spent. Asserting on the seconds handed down is what shows it, because both
    calls succeed either way.
    """
    handed: list[float] = []

    async def _is_safe(_url, *, seconds=ADDRESS_CHECK_SECONDS):
        handed.append(seconds)
        await asyncio.sleep(0.05)
        return True

    payload = {"provider": {"options": {
        "alpha": {"video": "https://a.example/a.mp4"},
        "beta": {"video": "https://b.example/b.mp4"},
    }}}

    await _adapter(_is_safe)._validate_passthrough_urls(
        payload, deadline=time.monotonic() + 0.30
    )

    assert len(handed) == 2, handed
    assert handed[0] < 0.30 and handed[1] < handed[0] - 0.04, (
        f"the second branch was handed a fresh budget: {handed}"
    )


def test_the_budget_constants_leave_room_for_more_than_one_lookup():
    """A request budget below a single lookup would refuse every request that has one."""
    assert ADDRESS_CHECK_BUDGET_SECONDS > ADDRESS_CHECK_SECONDS > 0


# ----------------------------------------------------------- THE SLOT ---------
@pytest.mark.parametrize("address", [
    "https://one.example/clip.mp4", "https://two.example/other.mp4",
])
@pytest.mark.asyncio
async def test_the_deployment_wide_slot_is_taken_after_the_addresses_are_resolved(
    address, monkeypatch
):
    """One user's chosen nameservers must not be able to pin every video slot.

    `getaddrinfo` blocks for as long as the host in the request wants it to, and the
    whole request used to sit inside the deployment-wide semaphore while it did. Two
    slots by default, so two in-flight requests from one person were the whole pipe.

    Observed by recording when each thing happened rather than by racing a stall: the
    semaphore records the order it was acquired in, the real resolver seam records the
    order it was asked in, and the sequence has to read resolve-then-acquire. Two
    addresses, so a production `if url == ...` cannot satisfy both.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe

    order: list[str] = []

    class _Recording(asyncio.Semaphore):
        async def acquire(self):
            order.append("acquired the slot")
            return await super().acquire()

    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, _payload):
            raise RuntimeError("stop here; the ordering is already recorded")

    Pipe._video_global_semaphore = None
    Pipe._video_global_limit = 0
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        adapter = pipe._ensure_video_generation_adapter()
        cast(Any, adapter)._persistence = _NoPersistence()
        cast(Any, adapter)._ensure_global_semaphore = lambda valves: _Recording(2)

        async def _is_safe(url, *, seconds=ADDRESS_CHECK_SECONDS):
            order.append(f"resolved {url}")
            return True

        monkeypatch.setattr(pipe._multimodal_handler, "_is_safe_url", _is_safe)
        monkeypatch.setattr(
            "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient",
            _FakeClient,
        )
        monkeypatch.setattr(pipe, "_create_http_session", lambda *_a, **_k: None)

        await adapter.generate(
            body={"messages": [{"role": "user", "content": "make a video"}]},
            responses_body=SimpleNamespace(provider={"options": {"p": {"video": address}}}),
            valves=pipe.valves,
            session=object(),
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

    assert f"resolved {address}" in order, f"the address was never checked: {order}"
    assert "acquired the slot" in order, f"the slot was never taken: {order}"
    assert order.index(f"resolved {address}") < order.index("acquired the slot"), (
        f"the deployment-wide slot was held while addresses were resolved: {order}"
    )


class _NoPersistence:
    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        return ""


# ------------------------------------------------------------- DEPTH ---------
@pytest.mark.parametrize("levels", [1, 3, 4, 5, 40, 2000])
@pytest.mark.asyncio
async def test_the_two_walkers_refuse_exactly_the_same_requests(levels):
    """One constant governs both, so the advertised bound is the operative one.

    The video walker descends `provider.options.<slug>` on its own before the shared
    scanner ever runs. Without a counter it accepted nesting the scanner refuses, and at
    real depth it died with `RecursionError` -- a bound of "whatever CPython allows"
    rather than the twelve the refusal names. Six depths spanning the boundary and well
    past the interpreter's stack, so neither an off-by-one nor "it happens to not crash
    here" can pass.
    """
    async def _is_safe(_url, *, seconds=ADDRESS_CHECK_SECONDS):
        return True

    payload = _nested(levels)
    try:
        list(payload_addresses(_nested(levels)))
        scanner_refused = False
    except UnvettableRequest:
        scanner_refused = True

    try:
        await _adapter(_is_safe)._validate_passthrough_urls(payload)
        walker_refused = False
        message = ""
    except Exception as exc:
        walker_refused = True
        message = str(exc)
        assert not isinstance(exc, RecursionError), (
            f"{levels} levels blew the stack instead of meeting the declared bound"
        )

    assert walker_refused is scanner_refused, (
        f"{levels} levels: scanner refused={scanner_refused}, walker refused={walker_refused}"
    )
    if walker_refused:
        assert str(MAX_URL_SCAN_DEPTH) in message, message
        assert "levels deep" in message, message


@pytest.mark.asyncio
async def test_a_request_within_the_bound_still_reaches_every_address():
    """The depth guard must refuse deep requests, not merely refuse."""
    seen: list[str] = []

    async def _is_safe(url, *, seconds=ADDRESS_CHECK_SECONDS):
        seen.append(url)
        return True

    await _adapter(_is_safe)._validate_passthrough_urls(_nested(3))

    assert seen == ["https://ok.example/a.mp4"], seen


@pytest.mark.parametrize("moved_to", [6, 21])
@pytest.mark.asyncio
async def test_moving_the_one_constant_moves_where_the_video_walker_refuses(
    moved_to, monkeypatch
):
    """A second literal twelve inside video.py passes every behavioural test above.

    So move the declared constant and watch: if the video walker consults the shared
    guard it follows, and if it carries its own copy it does not. Two directions --
    tighter and looser than the shipped twelve -- because a walker frozen at any single
    number fails one of them, and a walker that refuses everything fails the other.
    """
    import open_webui_openrouter_pipe.integrations.provider_options as options_module

    async def _is_safe(_url, *, seconds=ADDRESS_CHECK_SECONDS):
        return True

    monkeypatch.setattr(options_module, "MAX_URL_SCAN_DEPTH", moved_to)
    just_inside = (moved_to // 3) - 1
    just_outside = (moved_to // 3) + 1
    adapter = _adapter(_is_safe)

    await adapter._validate_passthrough_urls(_nested(just_inside))

    with pytest.raises(Exception) as refused:
        await adapter._validate_passthrough_urls(_nested(just_outside))
    assert "levels deep" in str(refused.value), refused.value
