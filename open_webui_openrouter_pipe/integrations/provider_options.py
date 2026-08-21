"""The one reader of the provider block an operator asked for on a request."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from ..core.config import _PIPE_METADATA_KEY
from ..core.utils import clamp_text


def requested_provider_block(
    responses_body: Any, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    provider = getattr(responses_body, "provider", None)
    if not isinstance(provider, dict):
        pipe_meta = metadata.get(_PIPE_METADATA_KEY) if isinstance(metadata, dict) else None
        candidate = pipe_meta.get("provider") if isinstance(pipe_meta, dict) else None
        provider = candidate if isinstance(candidate, dict) else {}
    return provider


def requested_provider_options(
    responses_body: Any, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    options = requested_provider_block(responses_body, metadata).get("options")
    return dict(options) if isinstance(options, dict) else {}


def merge_provider_options(
    requested: dict[str, Any], slug: Any, params: dict[str, Any]
) -> dict[str, Any]:
    block = {key: value for key, value in requested.items() if key != "options"}
    raw_options = requested.get("options")
    options = {
        name: dict(value) if isinstance(value, dict) else value
        for name, value in raw_options.items()
    } if isinstance(raw_options, dict) else {}
    if params and isinstance(slug, str) and slug:
        merged = options.get(slug)
        merged = dict(merged) if isinstance(merged, dict) else {}
        merged.update(params)
        options[slug] = merged
    if options:
        block["options"] = options
    return block


_ROUTING_PIN_KEYS = ("only", "order", "ignore")


def options_key(slug: Any) -> str:
    return slug.split("/", 1)[0].strip() if isinstance(slug, str) else ""


def bare_pins(requested: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(requested)
    for name in _ROUTING_PIN_KEYS:
        value = normalised.get(name)
        if isinstance(value, list):
            normalised[name] = [key for slug in value if (key := options_key(slug))]
    return normalised


def fan_provider_options(
    requested: dict[str, Any], slugs: list[str], params: dict[str, Any]
) -> dict[str, Any]:
    block = requested
    for slug in slugs or [""]:
        block = merge_provider_options(block, slug, params)
    return block


def carrier_slug(
    requested: dict[str, Any], candidates: list[str], *, pin_routes: bool = True
) -> str:
    """Pick the slug that carries a value which cannot be duplicated across providers.

    A value written under a slug routing will not select is silently ignored by OpenRouter.
    Where the transport accepts ``only`` the pin decides which provider serves the request,
    so the pin is the carrier even if the cached catalog has not caught up. Where the
    transport does not accept it, routing cannot see the pin, so a carrier drawn from it
    would key the value to a provider that will never be selected.

    ``only`` is an allow-set, not a precedence order -- ``order`` is the ordered field -- so
    when several allowed providers serve the model the carrier is the allowed one that has a
    published record, not merely the first named. A pin naming nothing the catalog knows
    still carries, because the catalog may simply not have caught up. ``ignore`` removes a
    provider from consideration entirely: keying options to an excluded provider guarantees
    they are dropped.
    """
    if not candidates:
        return ""
    ignored = requested.get("ignore")
    reachable = [
        slug
        for slug in candidates
        if not (isinstance(ignored, list) and slug in ignored)
    ] or candidates
    if pin_routes:
        only = requested.get("only")
        if isinstance(only, list):
            allowed = [slug for slug in only if isinstance(slug, str) and slug]
            for slug in allowed:
                if slug in reachable:
                    return slug
            if allowed:
                return allowed[0]
    order = requested.get("order")
    if isinstance(order, list):
        for slug in order:
            if isinstance(slug, str) and slug in reachable:
                return slug
    return reachable[0]


IMAGE_PROVIDER_KEYS = frozenset(
    {"allow_fallbacks", "ignore", "only", "options", "order", "sort"}
)
VIDEO_PROVIDER_KEYS = frozenset({"options"})
CHAT_PROVIDER_KEYS = frozenset(
    {
        "allow_fallbacks",
        "data_collection",
        "enforce_distillable_text",
        "ignore",
        "max_price",
        "only",
        "order",
        "preferred_max_latency",
        "preferred_min_throughput",
        "quantizations",
        "require_parameters",
        "sort",
        "zdr",
    }
)

TRANSPORT_PROVIDER_KEYS: dict[str, frozenset[str]] = {
    "chat": CHAT_PROVIDER_KEYS,
    "image": IMAGE_PROVIDER_KEYS,
    "video": VIDEO_PROVIDER_KEYS,
}


def restrict_provider_block(
    block: dict[str, Any], accepted: frozenset[str]
) -> tuple[dict[str, Any], list[str]]:
    """Reduce a provider block to what the transport documents, reporting what was cut.

    OpenRouter's image and video schemas define fewer provider keys than chat completions
    and set no ``additionalProperties: false``, so an undefined key is accepted and
    ignored. Sending ``zdr`` or ``data_collection`` there would read as a privacy control
    in force while nothing enforces it.
    """
    kept = {key: value for key, value in block.items() if key in accepted}
    dropped = sorted(key for key in block if key not in accepted)
    return kept, dropped


PROSE_PAYLOAD_FIELDS = frozenset({"prompt"})

MAX_URL_SCAN_DEPTH = 12

MAX_URL_SCAN_NODES = 4096

_ABSOLUTE_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")

MAX_LABEL_SEGMENT = 48

MAX_LABEL = 200

_INERT_LABEL_SEGMENT = re.compile(r"[^A-Za-z0-9 ._/:@+-]")

_TOO_DEEP_TO_VET = (
    f"This request nests values more than {MAX_URL_SCAN_DEPTH} levels deep. Every address "
    "in a request is checked before OpenRouter is asked to fetch it, nothing that deep can "
    "be checked, so the request was not sent."
)

_TOO_MANY_VALUES_TO_VET = (
    f"This request carries more than {MAX_URL_SCAN_NODES} values. Every address in a "
    "request is checked before OpenRouter is asked to fetch it, a request that large "
    "cannot be checked, so it was not sent."
)


class UnvettableRequest(Exception):
    pass


def refuse_past_the_scan_depth(depth: int) -> None:
    if depth >= MAX_URL_SCAN_DEPTH:
        raise UnvettableRequest(_TOO_DEEP_TO_VET)


def label_segment(key: Any) -> str:
    text = " ".join(str(key).split())
    inert = _INERT_LABEL_SEGMENT.sub(" ", text).strip()
    return clamp_text(inert, MAX_LABEL_SEGMENT) if inert else "?"


def _addresses_in(
    value: Any, path: str, depth: int, remaining: list[int]
) -> Iterator[tuple[str, str]]:
    if remaining[0] <= 0:
        raise UnvettableRequest(_TOO_MANY_VALUES_TO_VET)
    remaining[0] -= 1
    if isinstance(value, str):
        cleaned = value.strip()
        if _ABSOLUTE_URL_RE.match(cleaned):
            yield cleaned, path
        return
    if isinstance(value, dict):
        refuse_past_the_scan_depth(depth)
        for key, item in value.items():
            yield from _addresses_in(
                item, clamp_text(f"{path}.{label_segment(key)}", MAX_LABEL), depth + 1, remaining
            )
        return
    if isinstance(value, list):
        refuse_past_the_scan_depth(depth)
        for index, item in enumerate(value):
            yield from _addresses_in(
                item, clamp_text(f"{path}[{index}]", MAX_LABEL), depth + 1, remaining
            )


def payload_addresses(
    payload: dict[str, Any], prose_fields: frozenset[str] = PROSE_PAYLOAD_FIELDS
) -> Iterator[tuple[str, str]]:
    remaining = [MAX_URL_SCAN_NODES]
    for key, value in payload.items():
        if key in prose_fields:
            continue
        yield from _addresses_in(value, label_segment(key), 1, remaining)
