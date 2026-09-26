"""The chat-id prefixes with no `chat` row are Open WebUI's list, not ours.

`_UNLINKABLE_CHAT_PREFIXES` is a hand-copy of upstream's `NON_SAVED_CHAT_ID_PREFIXES`,
and it has already drifted from it twice: it named `local:` after upstream renamed the
concept to `temporary:`, and `channel:` was added here separately, reaching exactly one
of the gates that needed it. A prefix Open WebUI adds next is uncovered by construction,
and the symptom is quiet -- an upload against an unsaved chat reaches an INSERT whose
foreign key has nothing to point at.

So the resolver prefers upstream's list where the deployment publishes one. It cannot
just import it: `open_webui.utils.chat_id` does not exist in the 0.10.x line, which the
manifest still supports, so an unguarded import would break a deployment that works
today. Both halves of that need a test, and the upstream half is unreachable on this
Open WebUI without injecting the module.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from open_webui_openrouter_pipe.storage.owui_files import (
    _UNLINKABLE_CHAT_PREFIXES,
    _unlinkable_chat_prefixes,
    is_linkable_chat,
    is_temporary_chat,
    temporary_chat_prefixes,
)
from tests.test_a_temporary_chats_cost_snapshot_names_no_chat import USAGE, _FakeRedis, _through_the_pipe

@pytest.fixture(autouse=True)
def _unresolved():
    _unlinkable_chat_prefixes.cache_clear()
    yield
    _unlinkable_chat_prefixes.cache_clear()


def _publish(monkeypatch, prefixes):
    module = ModuleType("open_webui.utils.chat_id")
    setattr(module, "NON_SAVED_CHAT_ID_PREFIXES", prefixes)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat_id", module)


def test_the_upstream_list_wins_when_open_webui_publishes_one(monkeypatch):
    """A prefix upstream knows about and we do not must still be recognised."""
    _publish(monkeypatch, ["temporary:", "local:", "channel:", "draft:"])
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


@pytest.mark.parametrize(
    "published",
    [[], ["", "  "], "temporary:", None, [b"temporary:"]],
    ids=["empty", "blank-strings", "a-bare-string", "none", "bytes"],
)
def test_an_unusable_upstream_value_falls_back_instead_of_disarming_the_gate(
    monkeypatch, published
):
    """The failure mode this guards is silence, not a crash.

    If upstream ever ships the name with a value that yields no usable prefixes, taking
    it at face value makes every unsaved chat look linkable and the gate stops existing.
    Note `"temporary:"` -- a bare string is iterable, and iterating it yields characters,
    so a `startswith` built from it would match any chat id beginning with `t`.
    """
    _publish(monkeypatch, published)
    assert _unlinkable_chat_prefixes() == _UNLINKABLE_CHAT_PREFIXES
    assert is_linkable_chat("temporary:abc") is False
    assert is_linkable_chat("this-is-a-real-chat") is True


def test_upstream_cannot_narrow_the_local_floor(monkeypatch):
    """Union, not replace: a prefix upstream retires must still be refused here.

    `_UNLINKABLE_CHAT_PREFIXES` is this pipe's own record of which chat ids have no
    `chat` row, not an approximation of upstream's list. Upstream calls `local:`
    LEGACY_TEMPORARY_CHAT_ID_PREFIX, so it is a deprecation candidate -- and adopting
    their tuple wholesale meant the day they drop it, every upload in a temporary chat
    starts reaching the chat_files INSERT and orphaning a row.
    """
    _publish(monkeypatch, ["temporary:", "channel:"])
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
    _publish(monkeypatch, ["draft:"])
    first = _unlinkable_chat_prefixes()
    assert "draft:" in first, f"the published prefix was not adopted: {first}"
    monkeypatch.delitem(sys.modules, "open_webui.utils.chat_id")
    assert _unlinkable_chat_prefixes() == first, (
        "the resolver re-imported after the module went away, so it pays a failed "
        "import per upload"
    )
