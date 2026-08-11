"""The one reader of the provider block an operator asked for on a request."""

from __future__ import annotations

from typing import Any

from ..core.config import _PIPE_METADATA_KEY


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
