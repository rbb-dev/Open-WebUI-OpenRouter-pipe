"""Image generation via OpenRouter's dedicated Image API."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, NamedTuple

from ..core.config import _PIPE_METADATA_KEY, _select_openrouter_http_referer
from ..core.costs import maybe_dump_costs_snapshot
from ..core.errors import OpenRouterAPIError
from ..core.logging_system import SessionLogger
from ..core.warn_latch import warn_level
from .image_client import OpenRouterImageClient
from .image_types import (
    SCHEMA_ONLY_PARAMS,
    TOP_LEVEL_PARAMS,
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationResult,
    clamp_text,
    prompt_with_system,
    summarise_names,
    supersede_size_conflicts,
)
from .provider_options import (
    IMAGE_PROVIDER_KEYS,
    bare_pins,
    carrier_slug,
    fan_provider_options,
    options_key,
    requested_provider_block,
    restrict_provider_block,
)

_NOTE_NAME_LIMIT = 40
_NOTE_VALUE_LIMIT = 80

_PROVIDER_KEY_REPORT_LIMIT = 16

_NOTE_TEXT_LIMIT = 160


def _clamp(text: Any, limit: int = _NOTE_NAME_LIMIT) -> str:
    """Bound a span of request-supplied text before it reaches a log line or the browser.

    ``image_config`` arrives from the client verbatim, so a key or value interpolated at
    full length would size a WARNING record and a notification event to whatever the
    request carried.
    """
    return clamp_text(text, limit)

_TOP_LEVEL_PARAMS = TOP_LEVEL_PARAMS

_SCHEMA_ONLY_PARAMS = SCHEMA_ONLY_PARAMS

_REFERENCE_MODES = frozenset({"auto", "latest-only", "none"})

_SCHEMA_REFERENCE_CAP = 16

_BILLING_MULTIPLIERS = frozenset({"n"})

_LEGACY_PARAM_NAMES = {
    "image_size": "resolution",
}

_warned_image_endpoints: set[str] = set()

_warned_dropped_image_param: set[str] = set()

_warned_image_cost_snapshot: set[str] = set()

_warned_image_provider_keys: set[str] = set()

_NO_CONTRACT = (
    "provider passthrough parameters will be dropped and top-level values sent unvalidated."
)

_STALE_CONTRACT = (
    "the previously cached contract is being reused and may be stale, so a knob may be "
    "gated against limits this model no longer publishes."
)

if TYPE_CHECKING:
    from ..pipe import Pipe


_Outcome = dict[str, Any]


class _Note(NamedTuple):
    """A rejected knob, split so the latch key cannot be widened by request content."""

    kind: str
    name: str
    text: str


def _superseded(name: str, value: Any, reason: str) -> tuple[str, str, str]:
    return (
        "superseded",
        name,
        f"{name}={_clamp(repr(value), _NOTE_VALUE_LIMIT)} was not sent ({reason})",
    )


def _size_consistency_notes(top_level: dict[str, Any]) -> list[tuple[str, str, str]]:
    dropped = supersede_size_conflicts(top_level)
    if not dropped:
        return []
    shown_size = _clamp(repr(top_level.get("size")), _NOTE_VALUE_LIMIT)
    return [
        _superseded(name, value, f"size={shown_size} {reason}") for name, value, reason in dropped
    ]


class ImageGenerationAdapter:

    def __init__(self, *, pipe: Pipe, logger: logging.Logger) -> None:
        self._pipe = pipe
        self._logger = logger
        self._endpoint_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def _resolve_api_key(self, valves: Any) -> str:
        api_key, api_key_error = self._pipe._resolve_openrouter_api_key(valves)
        if api_key_error or not api_key:
            raise ImageGenerationError(api_key_error or "OpenRouter API key is not configured.")
        return api_key

    def _client(
        self,
        session: Any,
        valves: Any,
        *,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> OpenRouterImageClient:
        return OpenRouterImageClient(
            session,
            base_url=getattr(valves, "BASE_URL", "") or "https://openrouter.ai/api/v1",
            api_key=self._resolve_api_key(valves),
            logger=self._logger,
            http_referer=_select_openrouter_http_referer(valves),
            user=user,
            owui_chat_id=owui_chat_id,
        )

    @staticmethod
    def _fit_descriptor(descriptor: Any, value: Any) -> tuple[Any, str]:
        if not isinstance(descriptor, dict):
            return value, ""
        kind = descriptor.get("type")
        if kind == "enum":
            allowed = descriptor.get("values")
            if isinstance(allowed, list) and value not in allowed:
                return None, f"accepts {', '.join(str(item) for item in allowed)}"
            return value, ""
        if kind == "range":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, "expects a number"
            low, high = descriptor.get("min"), descriptor.get("max")
            if isinstance(high, (int, float)) and not isinstance(high, bool) and value > high:
                return high, f"capped at {high}"
            if isinstance(low, (int, float)) and not isinstance(low, bool) and value < low:
                return low, f"raised to {low}"
            return value, ""
        if kind == "boolean":
            # OpenRouter uses this to say the model *supports* the parameter, not that
            # its value is true or false -- their own model schema words it "whether the
            # model supports deterministic generation via seed parameter", and no such
            # descriptor ever carries a domain. The request itself takes a number, so
            # that is what is checked here.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, "expects a number"
            return value, ""
        return value, ""

    @staticmethod
    def _reachable_records(
        records: list[dict[str, Any]], requested: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        keyed = [
            (options_key(slug), record)
            for record in records
            if isinstance(slug := record.get("provider_slug"), str) and options_key(slug)
        ]
        chosen = carrier_slug(bare_pins(requested or {}), [key for key, _ in keyed])
        if not chosen:
            return [record for _key, record in keyed]
        return [record for key, record in keyed if key == chosen]

    @staticmethod
    def _union_declared(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        for record in records:
            supported = record.get("supported_parameters")
            if not isinstance(supported, dict):
                continue
            for name, descriptor in supported.items():
                if not isinstance(descriptor, dict):
                    continue
                current = merged.get(name)
                if current is None:
                    merged[name] = dict(descriptor)
                    continue
                if descriptor.get("type") == "enum" and current.get("type") == "enum":
                    values = list(current.get("values") or [])
                    for value in descriptor.get("values") or []:
                        if value not in values:
                            values.append(value)
                    current["values"] = values
                low, high = descriptor.get("min"), descriptor.get("max")
                if isinstance(low, (int, float)) and isinstance(current.get("min"), (int, float)):
                    current["min"] = min(current["min"], low)
                if isinstance(high, (int, float)) and isinstance(current.get("max"), (int, float)):
                    current["max"] = max(current["max"], high)
        return merged or None

    @staticmethod
    def _records_accepting(records: list[dict[str, Any]], chosen: dict[str, Any]) -> list[str]:
        keep: list[str] = []
        for record in records:
            supported = record.get("supported_parameters")
            declared = supported if isinstance(supported, dict) else {}
            slug = record.get("provider_slug")
            if not isinstance(slug, str) or not slug:
                continue
            ok = True
            for name, value in chosen.items():
                if name not in declared:
                    continue
                fitted, _note = ImageGenerationAdapter._fit_descriptor(declared.get(name), value)
                if fitted is None:
                    ok = False
                    break
            if ok:
                keep.append(slug)
        return keep

    @staticmethod
    def _split_image_config(
        body: dict[str, Any],
        *,
        allowed_passthrough: tuple[str, ...] | frozenset[str],
        record: dict[str, Any] | None,
        records: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[_Note]]:
        raw = body.get("image_config") if isinstance(body, dict) else None
        if not isinstance(raw, dict):
            return {}, {}, []
        supported = (record or {}).get("supported_parameters")
        declared = (
            ImageGenerationAdapter._union_declared(records)
            if records
            else (supported if isinstance(supported, dict) else None)
        )
        top_level: dict[str, Any] = {}
        provider: dict[str, Any] = {}
        notes: list[_Note] = []
        unvalidated: list[str] = []
        budget = len(_TOP_LEVEL_PARAMS) + _PROVIDER_KEY_REPORT_LIMIT
        unreported = 0

        def _note(kind: str, name: str, text: str) -> None:
            nonlocal unreported
            if len(notes) >= budget:
                unreported += 1
                return
            notes.append(
                _Note(kind, name if name in _TOP_LEVEL_PARAMS else "*", _clamp(text, _NOTE_TEXT_LIMIT))
            )

        for key, value in raw.items():
            # `image_config` arrives from the client, so a key need not be a string.
            if not isinstance(key, str) or value is None or value == "":
                continue
            shown = _clamp(key)
            # The alias is a compatibility spelling for a retired filter. A key the
            # model's own record claims as a provider option keeps that spelling, or the
            # rename would route it into a different published parameter and silently
            # overwrite whatever the user chose there.
            name = key if key in allowed_passthrough else _LEGACY_PARAM_NAMES.get(key, key)
            if name != key and raw.get(name) not in (None, ""):
                # Both spellings of one parameter were supplied -- the compatibility one
                # from the older filter and the published one from this model's own. They
                # would write the same destination, and whichever came later in the dict
                # would win silently. The spelling the model publishes is the one that
                # means something, so it wins and the user is told the other was ignored.
                _note(
                    "superseded",
                    name,
                    f"{shown} was ignored because {name} was set explicitly",
                )
                continue
            if name in _SCHEMA_ONLY_PARAMS and (declared is None or name not in declared):
                top_level[name] = value
                continue
            if name in _TOP_LEVEL_PARAMS:
                if declared is None:
                    if name in _BILLING_MULTIPLIERS:
                        _note(
                            "unbounded-multiplier",
                            name,
                            f"{name} was not sent (it multiplies what the request costs and "
                            "this model's published limit could not be read)",
                        )
                        continue
                    unvalidated.append(name)
                    top_level[name] = value
                    continue
                if name not in declared:
                    _note("not-offered", name, f"{name} is not offered by this model")
                    continue
                fitted, note = ImageGenerationAdapter._fit_descriptor(declared.get(name), value)
                if fitted is None:
                    _note(
                        "outside-contract",
                        name,
                        f"{name}={_clamp(repr(value), _NOTE_VALUE_LIMIT)} was not sent ({note})",
                    )
                    continue
                if note:
                    _note("clamped", name, f"{name} {note}")
                top_level[name] = fitted
            elif key in allowed_passthrough:
                provider[key] = value
            elif record is None:
                _note(
                    "contract-unreadable",
                    key,
                    f"{shown} was not sent (this model's options could not be read)",
                )
            else:
                _note("not-offered", key, f"{shown} is not offered by this model")
        for kind, superseded, note_text in _size_consistency_notes(top_level):
            _note(kind, superseded, note_text)
        unvalidated = [name for name in unvalidated if name in top_level]
        if unvalidated:
            notes.append(
                _Note(
                    "unvalidated",
                    "*",
                    f"{', '.join(sorted(unvalidated))} went out unchecked (this model's "
                    "published limits could not be read, so OpenRouter may reject them)",
                )
            )
        if unreported:
            notes.append(
                _Note("overflow", "*", f"{unreported} further setting(s) were not sent")
            )
        return top_level, provider, notes

    async def _published_records(
        self,
        session: Any,
        valves: Any,
        api_model_id: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Every record this model publishes, not the one a request would route to.

        Describing what a model offers is a different question from choosing who serves
        one request, and answering it with a single provider's record lists controls the
        model's filter -- built from the intersection -- does not draw.
        """
        await self._endpoint_record(session, valves, api_model_id, **kwargs)
        cached = self._endpoint_cache.get(api_model_id)
        return list(cached[1]) if cached else []

    async def _endpoint_record(
        self,
        session: Any,
        valves: Any,
        api_model_id: str,
        *,
        user: Any = None,
        owui_chat_id: str | None = None,
        requested: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        raw_ttl = getattr(valves, "MODEL_CATALOG_REFRESH_SECONDS", 0)
        ttl = float(raw_ttl) if isinstance(raw_ttl, (int, float)) and not isinstance(raw_ttl, bool) else 0.0
        ttl = ttl or 3600.0
        cached = self._endpoint_cache.get(api_model_id)
        if cached is not None and (time.monotonic() - cached[0]) < ttl:
            return self._select_endpoint(cached[1], requested)
        try:
            records = await self._client(
                session, valves, user=user, owui_chat_id=owui_chat_id
            ).endpoints(api_model_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fallback, unserved = (
                self._select_endpoint(cached[1], requested) if cached is not None else (None, "")
            )
            self._logger.log(
                warn_level(_warned_image_endpoints, f"{api_model_id}:{type(exc).__name__}"),
                "Image endpoint lookup failed for %r (%s); %s",
                api_model_id,
                type(exc).__name__,
                _STALE_CONTRACT if fallback is not None else _NO_CONTRACT,
                exc_info=True,
            )
            return fallback, unserved
        if not records:
            fallback, unserved = (
                self._select_endpoint(cached[1], requested) if cached is not None else (None, "")
            )
            self._logger.log(
                warn_level(_warned_image_endpoints, f"{api_model_id}:empty"),
                "OpenRouter returned no endpoint record for %r; %s",
                api_model_id,
                _STALE_CONTRACT if fallback is not None else _NO_CONTRACT,
            )
            return fallback, unserved
        self._endpoint_cache[api_model_id] = (time.monotonic(), records)
        return self._select_endpoint(records, requested)

    @staticmethod
    def _select_endpoint(
        records: list[dict[str, Any]], requested: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str]:
        """Pick the record for the provider that will serve this request.

        The second element names an operator pin that no record carries. Collapsing that
        onto a bare ``None`` would make a readable contract indistinguishable from an
        unreadable one, and the caller would blame an outage that did not happen.
        """
        if not records:
            return None, ""
        keyed = [
            (options_key(slug), slug, record)
            for record in records
            if isinstance(slug := record.get("provider_slug"), str) and options_key(slug)
        ]
        known = [key for key, _slug, _record in keyed]
        chosen = carrier_slug(bare_pins(requested or {}), known)
        matched = sorted(
            ((slug, record) for key, slug, record in keyed if key == chosen),
            key=lambda pair: pair[0],
        )
        if matched:
            return matched[0][1], ""
        if chosen and chosen not in known:
            return None, chosen
        return records[0], ""

    @staticmethod
    def _option_carriers(records: list[dict[str, Any]]) -> list[str]:
        keys = {options_key(record.get("provider_slug")) for record in records}
        return sorted(key for key in keys if key)

    @staticmethod
    def _every_endpoint_publishes_streaming(records: list[dict[str, Any]]) -> bool:
        return bool(records) and all(
            record.get("supports_streaming") is True for record in records
        )

    def _unserved_pin_note(self, unserved_pin: str, api_model_id: str) -> _Note:
        self._logger.log(
            warn_level(_warned_image_endpoints, f"{api_model_id}:pin"),
            "Provider %r was pinned for %r but carries no endpoint record; the contract "
            "was read, so this is a pin that does not match rather than a lookup failure.",
            _clamp(unserved_pin),
            api_model_id,
        )
        return _Note(
            "unserved-pin",
            "*",
            f"{_clamp(unserved_pin)} does not serve this model; its published limits "
            "were not applied",
        )

    def _status_reporter(self, event_emitter: Any) -> Any:
        if not event_emitter:
            return None

        async def report(message: str) -> None:
            await self._pipe._event_emitter_handler._emit_status(
                event_emitter, message, done=False
            )

        return report

    async def _report_notes(
        self, notes: list[_Note], *, api_model_id: str, event_emitter: Any
    ) -> None:
        for note in notes:
            self._logger.log(
                warn_level(
                    _warned_dropped_image_param, f"{api_model_id}:{note.kind}:{note.name}"
                ),
                "Image parameter not sent for %r: %s",
                api_model_id,
                note.text,
            )
        if notes and event_emitter:
            await self._pipe._event_emitter_handler._emit_notification(
                event_emitter,
                summarise_names(
                    [note.text for note in notes], _PROVIDER_KEY_REPORT_LIMIT, _NOTE_TEXT_LIMIT
                ),
                level="warning",
            )

    async def fit_chat_image_config(
        self,
        *,
        responses_body: Any,
        published: list[dict[str, Any]] | None,
        metadata: dict[str, Any] | None,
        event_emitter: Any,
        api_model_id: str,
    ) -> None:
        raw = getattr(responses_body, "image_config", None)
        records = [record for record in (published or []) if isinstance(record, dict)]
        if not isinstance(raw, dict) or not raw or not records:
            return
        record, unserved_pin = self._select_endpoint(
            records, requested_provider_block(responses_body, metadata)
        )
        if record is None:
            if unserved_pin:
                await self._report_notes(
                    [self._unserved_pin_note(unserved_pin, api_model_id)],
                    api_model_id=api_model_id,
                    event_emitter=event_emitter,
                )
            return
        allowed = frozenset(
            item
            for item in (record.get("allowed_passthrough_parameters") or [])
            if isinstance(item, str)
        )
        fitted, passthrough, notes = self._split_image_config(
            {"image_config": raw}, allowed_passthrough=allowed, record=record
        )
        merged = {**passthrough, **fitted}
        responses_body.image_config = merged or None
        await self._report_notes(
            notes, api_model_id=api_model_id, event_emitter=event_emitter
        )

    @staticmethod
    def _reference_limit(record: dict[str, Any] | None) -> int | None:
        supported = (record or {}).get("supported_parameters")
        if not isinstance(supported, dict):
            return None
        descriptor = supported.get("input_references")
        if not isinstance(descriptor, dict):
            return 0
        maximum = descriptor.get("max")
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            return None
        return int(maximum) if maximum >= 0 else None

    @staticmethod
    def _input_references(responses_body: Any) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        items = getattr(responses_body, "input", None)
        if not isinstance(items, list):
            return refs
        for item in items:
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if part.get("type") in ("input_image", "image_url") and isinstance(url, str) and url:
                    refs.append({"type": "image_url", "image_url": {"url": url}})
        return refs

    @staticmethod
    def _reference_settings(metadata: dict[str, Any] | None) -> tuple[str, list[str]]:
        pipe_meta = metadata.get(_PIPE_METADATA_KEY) if isinstance(metadata, dict) else None
        chosen = pipe_meta.get("image_generation") if isinstance(pipe_meta, dict) else None
        if not isinstance(chosen, dict):
            return "auto", []
        mode = chosen.get("reference_mode")
        raw = chosen.get("reference_urls")
        urls = (
            [item.strip() for item in raw if isinstance(item, str) and item.strip()]
            if isinstance(raw, list)
            else []
        )
        return (mode if mode in _REFERENCE_MODES else "auto"), urls

    async def _vetted_reference_urls(self, urls: list[str]) -> list[str]:
        if not urls:
            return []
        handler = self._pipe._multimodal_handler
        vetted: list[str] = []
        for url in urls[:_SCHEMA_REFERENCE_CAP]:
            if url.startswith("data:"):
                vetted.append(url)
                continue
            if not await handler._is_safe_url(url):
                raise ImageGenerationError(
                    f"Refusing to send the reference image link {_clamp(url)}. Use https, "
                    "or a plain http address this deployment allows."
                )
            vetted.append(url)
        return vetted

    async def _reference_payload(
        self,
        responses_body: Any,
        metadata: dict[str, Any] | None,
        *,
        record: dict[str, Any] | None,
        notes: list[_Note],
    ) -> list[dict[str, Any]]:
        mode, chosen = self._reference_settings(metadata)
        attached = self._input_references(responses_body)
        if mode == "none":
            attached = []
        elif mode == "latest-only":
            attached = attached[-1:]
        refs = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in await self._vetted_reference_urls(chosen)
        ]
        refs.extend(attached)
        published = self._reference_limit(record)
        limit = (
            _SCHEMA_REFERENCE_CAP
            if published is None
            else min(published, _SCHEMA_REFERENCE_CAP)
        )
        if len(refs) > limit:
            reason = (
                f"this model accepts {limit}"
                if published is not None
                else f"a request carries at most {limit}"
            )
            notes.append(
                _Note(
                    "refs-dropped",
                    "input_references",
                    f"dropped {len(refs) - limit} reference image(s); {reason}",
                )
            )
            refs = refs[:limit]
        return refs

    @staticmethod
    def _requester_id(user: Any, metadata: dict[str, Any] | None) -> str:
        for candidate in (
            (user or {}).get("id") if isinstance(user, dict) else getattr(user, "id", None),
            (metadata or {}).get("user_id") if isinstance(metadata, dict) else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    async def _persist(
        self,
        image: GeneratedImage,
        *,
        request: Any,
        user_obj: Any,
        metadata: dict[str, Any] | None,
        requester_id: str,
    ) -> str | None:
        upload_request, upload_user = await self._pipe._file_gateway.resolve_storage_context(
            request, user_obj
        )
        if not upload_request or not upload_user:
            return None
        meta = metadata if isinstance(metadata, dict) else {}
        file_id = await self._pipe._file_gateway.upload_to_owui_storage(
            request=upload_request,
            user=upload_user,
            file_data=image.data,
            filename=f"generated-image-{uuid.uuid4().hex}.{image.extension}",
            mime_type=image.mime_type,
            chat_id=meta.get("chat_id"),
            message_id=meta.get("message_id"),
            owui_user_id=requester_id or None,
        )
        return file_id

    def _final_status(self, *, elapsed: float, usage: dict[str, Any], valves: Any) -> str:
        try:
            return self._pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=elapsed,
                total_usage=dict(usage or {}),
                valves=valves,
                stream_duration=elapsed,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.warning(
                "Could not render the final image usage status; falling back to a plain "
                "duration line",
                exc_info=True,
            )
            return f"Image generated in {elapsed:.1f}s"

    async def _record_cost(
        self,
        valves: Any,
        usage: Any,
        *,
        user: Any,
        metadata: dict[str, Any] | None,
        user_obj: Any,
        api_model_id: str,
    ) -> None:
        try:
            await maybe_dump_costs_snapshot(
                self._pipe,
                valves,
                user_id=self._requester_id(user, metadata),
                model_id=api_model_id,
                usage=usage,
                user_obj=user_obj,
                pipe_id=self._pipe.id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.log(
                warn_level(_warned_image_cost_snapshot, type(exc).__name__),
                "Could not record the cost snapshot for %r; this image generation is billed "
                "but will be missing from the cost export.",
                api_model_id,
                exc_info=True,
            )

    async def _report_generation(
        self, usage: Any, status: str, metadata: dict[str, Any] | None
    ) -> None:
        await self._pipe._dispatch_plugin_event(
            "dispatch_on_generation_complete",
            usage if isinstance(usage, dict) and usage else None,
            status,
            request_id=SessionLogger.request_id.get() or "",
            metadata=metadata,
            task=None,
        )

    async def _close_status(self, event_emitter: Any) -> None:
        if event_emitter:
            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)

    async def _emit_failure(self, event_emitter: Any, reason: str) -> str:
        content = f"### Image generation failed\n\n{reason}"
        if event_emitter:
            await self._pipe._event_emitter_handler._emit_completion(
                event_emitter, content=content, done=True
            )
        return content

    async def generate(
        self,
        *,
        body: dict[str, Any],
        responses_body: Any,
        valves: Any,
        session: Any,
        event_emitter: Any,
        metadata: dict[str, Any] | None,
        user: Any,
        request: Any,
        user_obj: Any,
        normalized_model_id: str,
        api_model_id: str,
    ) -> str:
        outcome: _Outcome = {"usage": None, "reported": False, "costed": False}
        try:
            return await self._generate(
                body=body,
                user=user,
                responses_body=responses_body,
                valves=valves,
                session=session,
                event_emitter=event_emitter,
                metadata=metadata,
                request=request,
                user_obj=user_obj,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
                outcome=outcome,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._settle(outcome, valves, user, metadata, user_obj, api_model_id)
            )
            raise
        except OpenRouterAPIError as exc:
            await self._close_status(event_emitter)
            await self._settle(outcome, valves, user, metadata, user_obj, api_model_id)
            await self._pipe._ensure_error_formatter()._report_openrouter_error(
                exc,
                event_emitter=event_emitter,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
            )
            return ""
        except ImageGenerationError as exc:
            self._logger.warning("Image generation failed for %r: %s", api_model_id, exc)
            await self._close_status(event_emitter)
            outcome["usage"] = outcome["usage"] or getattr(exc, "usage", None) or None
            await self._settle(outcome, valves, user, metadata, user_obj, api_model_id)
            return await self._emit_failure(
                event_emitter, str(exc).strip() or type(exc).__name__
            )
        except Exception as exc:
            self._logger.exception("Image generation failed for %r", api_model_id)
            await self._close_status(event_emitter)
            await self._settle(outcome, valves, user, metadata, user_obj, api_model_id)
            return await self._emit_failure(
                event_emitter, str(exc).strip() or type(exc).__name__
            )

    async def _settle(
        self,
        outcome: _Outcome,
        valves: Any,
        user: Any,
        metadata: dict[str, Any] | None,
        user_obj: Any,
        api_model_id: str,
    ) -> None:
        """Close out a failed request without losing or duplicating what it already cost.

        The upstream call is billed the moment it returns 200, so a failure after that
        point still has to be reported with the usage it incurred, and it must not raise a
        second, contradictory completion event for a request already reported as ok.
        """
        billed = outcome["usage"] or None
        if not outcome["reported"]:
            outcome["reported"] = True
            await self._report_generation(billed, "failed", metadata)
        if billed and not outcome["costed"]:
            outcome["costed"] = True
            await self._record_cost(
                valves, billed, user=user, metadata=metadata, user_obj=user_obj,
                api_model_id=api_model_id,
            )

    async def _generate(
        self,
        *,
        body: dict[str, Any],
        responses_body: Any,
        valves: Any,
        session: Any,
        event_emitter: Any,
        metadata: dict[str, Any] | None,
        user: Any,
        request: Any,
        user_obj: Any,
        normalized_model_id: str,
        api_model_id: str,
        outcome: _Outcome,
    ) -> str:
        prompt = prompt_with_system(getattr(responses_body, "input", None))
        if not prompt.strip():
            prompt = prompt_with_system(body.get("messages") if isinstance(body, dict) else None)
        if not prompt.strip():
            raise ImageGenerationError(
                "An image prompt is required. Describe the image you want, or say what to "
                "change about the one you attached."
            )

        meta = metadata if isinstance(metadata, dict) else {}
        chat_id = meta.get("chat_id") if isinstance(meta.get("chat_id"), str) else None
        requested_provider = requested_provider_block(responses_body, metadata)
        record, unserved_pin = await self._endpoint_record(
            session,
            valves,
            api_model_id,
            user=user_obj,
            owui_chat_id=chat_id,
            requested=requested_provider,
        )
        allowed = frozenset(
            item
            for item in ((record or {}).get("allowed_passthrough_parameters") or [])
            if isinstance(item, str)
        )
        cached = self._endpoint_cache.get(api_model_id)
        records = list(cached[1]) if cached else []
        reachable_records = (
            self._reachable_records(records, requested_provider) if record is not None else []
        )
        top_level, provider_params, notes = self._split_image_config(
            body, allowed_passthrough=allowed, record=record, records=reachable_records
        )

        payload: dict[str, Any] = {"model": api_model_id, "prompt": prompt}
        payload.update(top_level)

        refs = await self._reference_payload(
            responses_body, metadata, record=record, notes=notes
        )
        if refs:
            payload["input_references"] = refs

        carriers = self._option_carriers(records)
        if provider_params and not carriers:
            names = summarise_names(sorted(provider_params), _PROVIDER_KEY_REPORT_LIMIT)
            notes.append(
                _Note(
                    "unkeyable",
                    "*",
                    f"{names} was not sent (OpenRouter does not name the company running this model)",
                )
            )
            self._logger.log(
                warn_level(_warned_image_endpoints, f"{api_model_id}:noslug"),
                "Endpoint record for %r advertises passthrough parameters but no provider slug; "
                "%s cannot be keyed and will not be sent.",
                api_model_id,
                names,
            )
        provider, unsupported = restrict_provider_block(
            fan_provider_options(requested_provider, carriers, provider_params),
            IMAGE_PROVIDER_KEYS,
        )
        if unserved_pin:
            notes.append(self._unserved_pin_note(unserved_pin, api_model_id))
        if unsupported:
            names = summarise_names(unsupported, _PROVIDER_KEY_REPORT_LIMIT)
            notes.append(
                _Note(
                    "provider-unsupported",
                    "*",
                    f"{names} was not sent (OpenRouter's image API does not accept it; "
                    "set it on a chat model instead)",
                )
            )
            self._logger.log(
                warn_level(_warned_image_provider_keys, f"{api_model_id}:unsupported"),
                "Provider preferences not accepted by the image API were not sent for %r: %s. "
                "Sending them would read as a control in force while nothing enforces it.",
                api_model_id,
                names,
            )
        reachable = self._records_accepting(reachable_records, top_level)
        if reachable and len(reachable) < len(reachable_records) and "only" not in provider:
            provider["only"] = sorted(reachable)

        if provider:
            payload["provider"] = provider

        await self._report_notes(
            notes, api_model_id=api_model_id, event_emitter=event_emitter
        )

        requested = payload.get("n")
        single = not (
            isinstance(requested, int)
            and not isinstance(requested, bool)
            and requested > 1
        )
        if single and self._every_endpoint_publishes_streaming(records):
            payload["stream"] = True

        if event_emitter:
            await self._pipe._event_emitter_handler._emit_status(
                event_emitter, "Generating image…", done=False
            )

        started_at = time.monotonic()
        result: ImageGenerationResult = await self._client(
            session, valves, user=user_obj, owui_chat_id=chat_id
        ).generate(
            payload,
            max_decoded_bytes=int(getattr(valves, "BASE64_MAX_SIZE_MB", 0) or 0) * 1024 * 1024,
            on_progress=self._status_reporter(event_emitter),
        )

        outcome["usage"] = result.usage
        await asyncio.shield(
            self._record_cost(
                valves, result.usage, user=user, metadata=metadata, user_obj=user_obj,
                api_model_id=api_model_id,
            )
        )
        outcome["costed"] = True

        snippets: list[str] = []
        unsaved = 0
        for index, image in enumerate(result.images):
            file_id = await self._persist(
                image,
                request=request,
                user_obj=user_obj,
                metadata=metadata,
                requester_id=self._requester_id(user, metadata),
            )
            if not file_id:
                unsaved += 1
                continue
            label = "Generated image" if len(result.images) == 1 else f"Generated image {index + 1}"
            snippets.append(f"![{label}](/api/v1/files/{file_id}/content)")
        if result.rejected:
            self._logger.warning(
                "%d generated image(s) for %r were discarded before storage: %s",
                len(result.rejected),
                api_model_id,
                summarise_names(result.rejected),
            )
        if unsaved:
            self._logger.warning(
                "%d of %d generated image(s) could not be saved to Open WebUI storage for %r",
                unsaved,
                len(result.images),
                api_model_id,
            )
        if result.rejected and snippets:
            snippets.append(
                f"_{len(result.rejected)} of {len(result.images) + len(result.rejected)} "
                "generated image(s) arrived in a form the pipe could not read._"
            )
        if unsaved and snippets:
            snippets.append(f"_{unsaved} generated image(s) could not be saved to storage._")

        if not snippets:
            await self._close_status(event_emitter)
            outcome["reported"] = True
            await self._report_generation(result.usage, "failed", metadata)
            return await self._emit_failure(
                event_emitter, "The generated image could not be saved to Open WebUI storage."
            )

        content = "\n\n".join(snippets)
        outcome["reported"] = True
        await self._report_generation(result.usage, "ok", metadata)
        if event_emitter:
            await self._pipe._event_emitter_handler._emit_status(
                event_emitter,
                self._final_status(
                    elapsed=time.monotonic() - started_at, usage=result.usage, valves=valves
                ),
                done=True,
            )
            await self._pipe._event_emitter_handler._emit_completion(
                event_emitter, content=content, done=True, usage=result.usage or None
            )
        return content
