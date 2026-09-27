"""The chat-id prefixes with no `chat` row are Open WebUI's list, not ours.

`_UNLINKABLE_CHAT_PREFIXES` is a hand-copy of upstream's `NON_SAVED_CHAT_ID_PREFIXES`,
and it has already drifted from it twice: it named `local:` after upstream renamed the
concept to `temporary:`, and `channel:` was added here separately, reaching exactly one
of the gates that needed it. A prefix Open WebUI adds next used to be uncovered; the
resolver now unions upstream's list, so the symptom stays quiet only while that union
holds -- an upload against an unsaved chat reaches an INSERT whose foreign key has
nothing to point at.

So the resolver prefers upstream's list where the deployment publishes one. It cannot
just import it unguarded: the floor is 0.11.4, where the module is present, so the guard
is a safety net for a deployment that does not publish it -- a hand-edited or vendored
Open WebUI, or a broken import -- rather than a compatibility requirement, and the
import stays guarded today. That deployment is a live path, not a hypothetical one: the
`ImportError` arm is exercised by
`test_the_local_list_is_used_when_open_webui_has_no_such_module` below. Both halves of
that need a test, and the upstream half is unreachable on this Open WebUI without
injecting the module.

Two resolvers now read one shared publisher, and every resolver is cached -- including
`_channel_chat_prefix`, which reads the same publisher for the channel prefix -- so every
cache the module owns has to be cleared between tests: a module injected by one test would
otherwise still be the answer the next one sees. The reset seam is conftest's autouse
`_reset_package_caches`, which clears every package cache at both ends of every test;
`_CACHED_RESOLVERS` below records this module's whole set for
`test_every_cached_resolver_in_the_module_is_cleared_between_tests`.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from open_webui_openrouter_pipe.storage import owui_files
from open_webui_openrouter_pipe.storage.owui_files import (
    _UNLINKABLE_CHAT_PREFIXES,
    _unlinkable_chat_prefixes,
    is_linkable_chat,
    is_temporary_chat,
    temporary_chat_prefixes,
)
from tests.test_a_temporary_chats_cost_snapshot_names_no_chat import USAGE, _FakeRedis, _through_the_pipe

_CACHED_RESOLVERS = (
    "_channel_chat_prefix",
    "_published_chat_id_values",
    "_unlinkable_chat_prefixes",
    "temporary_chat_prefixes",
)


def _publish(monkeypatch, non_saved, temporary, channel):
    """Publish the three names the one guarded import reads, as upstream's module does.

    All three, always: the import is a single `try` over all three names, so a fake
    publishing only one of them raises ImportError and sends BOTH resolvers down the
    floor. That would leave the two tests below passing vacuously and every other test
    in this file measuring the fallback instead of the published values.
    """
    module = ModuleType("open_webui.utils.chat_id")
    setattr(module, "NON_SAVED_CHAT_ID_PREFIXES", non_saved)
    setattr(module, "TEMPORARY_CHAT_ID_PREFIXES", temporary)
    setattr(module, "CHANNEL_CHAT_ID_PREFIX", channel)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat_id", module)


def test_the_upstream_list_wins_when_open_webui_publishes_one(monkeypatch):
    """A prefix upstream knows about and we do not must still be recognised."""
    _publish(
        monkeypatch,
        ["temporary:", "local:", "channel:", "draft:"],
        ("temporary:", "local:"),
        "channel:",
    )
    assert _unlinkable_chat_prefixes() == (
        "temporary:",
        "local:",
        "channel:",
        "draft:",
    )
    assert is_linkable_chat("draft:abc") is False, (
        "a prefix Open WebUI publishes was not honoured, so the hand-copied tuple is "
        "still the only list and the next upstream addition goes uncovered"
    )


def test_the_local_list_is_used_when_open_webui_has_no_such_module(monkeypatch):
    """The module missing entirely: the resolver's ImportError path must still keep the gate armed.

    Absence is simulated explicitly. This test used to rely on `open_webui.utils.chat_id` being absent from the
    test stubs, but that module has shipped since Open WebUI v0.11.0 -- below the 0.11.4 floor -- so it is present
    wherever the pipe runs, and the stubs now carry a verbatim copy of it. A `None` entry in `sys.modules` makes
    the import raise, which is the path the resolver guards. (Retiring the hand-copied list, and with it this
    fallback, is tracked as T226.)
    """
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat_id", None)
    assert _unlinkable_chat_prefixes() == _UNLINKABLE_CHAT_PREFIXES
    assert is_linkable_chat("temporary:abc") is False
    assert is_linkable_chat("chat-1") is True


def test_upstream_cannot_narrow_the_local_floor(monkeypatch):
    """Union, not replace: a prefix upstream retires must still be refused here.

    `_UNLINKABLE_CHAT_PREFIXES` is this pipe's own record of which chat ids have no
    `chat` row, not an approximation of upstream's list. Upstream calls `local:`
    LEGACY_TEMPORARY_CHAT_ID_PREFIX, so it is a deprecation candidate -- and adopting
    their tuple wholesale meant the day they drop it, every upload in a temporary chat
    starts reaching the chat_files INSERT and orphaning a row.
    """
    _publish(monkeypatch, ["temporary:", "channel:"], ("temporary:",), "channel:")
    resolved = _unlinkable_chat_prefixes()
    assert "local:" in resolved, (
        f"upstream retiring a prefix removed it from this gate too: {resolved}"
    )
    assert is_linkable_chat("local:abc") is False, (
        "a legacy temporary chat id is now treated as linkable, so its uploads reach "
        "the chat_files INSERT with no chat row to point at"
    )
    assert is_linkable_chat("real-chat-id") is True, (
        "the union swallowed ordinary chat ids as well"
    )


def test_the_resolution_is_cached(monkeypatch):
    """Called on every upload; a miss re-walks sys.path each time."""
    _publish(monkeypatch, ["draft:"], ("draft:",), "channel:")
    first = _unlinkable_chat_prefixes()
    assert "draft:" in first, f"the published prefix was not adopted: {first}"
    monkeypatch.delitem(sys.modules, "open_webui.utils.chat_id")
    assert _unlinkable_chat_prefixes() == first, (
        "the resolver re-imported after the module went away, so it pays a failed "
        "import per upload"
    )
