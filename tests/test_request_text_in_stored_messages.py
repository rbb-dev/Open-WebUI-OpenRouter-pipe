"""Request-chosen text that ends up in a message Open WebUI stores, and can share.

Two properties:

  * a path label built out of a request's own dict keys is bounded in length and inert
    as markdown/HTML by the time the walker yields it -- so every consumer inherits it
    rather than each raise site remembering to clamp; and
  * the link handed to OpenRouter is byte-for-byte the string the origin check
    validated, not a different string that happened to parse the same way.
"""
from __future__ import annotations

import json

import pytest

from open_webui_openrouter_pipe.integrations.media_relay import (
    _ENDPOINTS,
    _extract_url,
    _served_by,
)
from open_webui_openrouter_pipe.integrations.provider_options import (
    MAX_LABEL,
    MAX_LABEL_SEGMENT,
    payload_addresses,
)

CATBOX = _ENDPOINTS["catbox"].origins


def _labels(payload):
    return [where for _url, where in payload_addresses(payload)]


# --------------------------------------------------------------- CLAMPED ----
@pytest.mark.parametrize("length", [200, 20_000])
def test_a_key_cannot_size_the_label_it_ends_up_in(length):
    """The key is chosen by whoever wrote the request and reaches a stored message.

    Two lengths well apart, so a production clamp keyed to one of them cannot pass, and
    an unclamped label fails both.
    """
    payload = {"provider": {"options": {"k" * length: {"video": "https://o.example/a.mp4"}}}}

    label = _labels(payload)[0]

    assert len(label) <= MAX_LABEL, f"{len(label)} characters reached the message"


@pytest.mark.parametrize(
    "hostile",
    [
        "<img src=x onerror=alert(1)>",
        "](https://evil.example)[",
        "a\nb\n\n### Injected heading",
        "`code` **bold** <script>x</script>",
        "\r\n\r\nX-Injected: yes",
    ],
)
def test_a_key_cannot_change_how_the_message_renders(hostile):
    """The chat these errors land in can be shared, so the reader may be a third party.

    Five shapes covering HTML, markdown links, headings, emphasis and bare control
    characters -- an escape that handles angle brackets only fails the rest.
    """
    payload = {"provider": {"options": {hostile: {"video": "https://o.example/a.mp4"}}}}

    label = _labels(payload)[0]

    for active in ("<", ">", "`", "*", "#", "\n", "\r", "[", "]", "(", ")"):
        assert active not in label, f"{active!r} survived into {label!r}"


def test_a_top_level_key_is_bounded_by_the_same_rule_as_a_nested_one():
    """`payload_addresses` labels the top level itself, which is a second code path."""
    payload = {"z" * 5_000: "https://o.example/a.mp4"}

    label = _labels(payload)[0]

    assert len(label) <= MAX_LABEL_SEGMENT + 1, label
    assert "<" not in label


def test_a_label_still_says_where_the_address_was():
    """A clamp that erases the path leaves an error nobody can act on.

    The separators the walker writes itself are not request text and must survive, or
    the user is told an address was refused without being told which one.
    """
    payload = {"provider": {"options": {"runway": {
        "videos": [{"url": "https://o.example/a.mp4"}]
    }}}}

    assert _labels(payload) == ["provider.options.runway.videos[0].url"]


def test_an_empty_key_still_produces_something_to_point_at():
    payload = {"provider": {"options": {"   ": {"video": "https://o.example/a.mp4"}}}}

    assert _labels(payload) == ["provider.options.?.video"]


# ----------------------------------------------------------- ROUND TRIP -----
@pytest.mark.parametrize(
    "injected",
    [
        "https://catbox.moe/a.mp4\r\nX-Injected: yes",
        "https://catbox.moe/\tb.mp4",
        "https://cat\nbox.moe/c.mp4",
        "https://catbox.moe/d.mp4\n",
    ],
)
def test_a_link_that_parses_as_one_string_and_reads_as_another_is_refused(injected):
    """`urlsplit` deletes every tab, CR and LF anywhere in the string before parsing.

    So the origin check ran against a string the pipe then did not forward. Two distinct
    consequences, both here: header injection into whatever fetches the link, and -- the
    third case -- an origin bypass, since `cat\\nbox.moe` is checked as `catbox.moe`.
    The link is marked vetted the moment it comes back, so nothing downstream looks again.
    """
    body = json.dumps({"data": {"url": injected}})

    assert _extract_url(body, CATBOX) == "", f"{injected!r} was forwarded"
    assert _served_by(injected, CATBOX) is False


@pytest.mark.parametrize(
    "link",
    [
        "https://files.catbox.moe/abc123.mp4",
        "https://litter.catbox.moe/xyz.mp3?v=2",
        "https://catbox.moe/a.mp4#t=3",
    ],
)
def test_an_ordinary_link_the_host_serves_is_still_accepted(link):
    """Refusing on round-trip must not refuse the links the hosts actually answer with.

    Three real shapes -- plain, query, fragment -- because a check that rejected any URL
    carrying punctuation would break the feature while passing every test above.
    """
    assert _extract_url(json.dumps({"data": {"url": link}}), CATBOX) == link
    assert _extract_url(link, CATBOX) == link


@pytest.mark.parametrize(
    "link", ["https://attacker.example/beacon.mp4", "http://catbox.moe/a.mp4"]
)
def test_the_origin_and_scheme_checks_still_apply(link):
    """Round-tripping is an extra condition, not a replacement for the origin pin."""
    assert _extract_url(json.dumps({"data": {"url": link}}), CATBOX) == ""
