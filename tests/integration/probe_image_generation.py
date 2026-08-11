"""Live OpenRouter image-generation probe. Run manually; see README-dev.md."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import aiohttp

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
RUN_PROBE = os.getenv("OPENROUTER_RUN_IMAGE_PROBE", "").strip() == "1"
SWEEP_LIMIT = int(os.getenv("OPENROUTER_IMAGE_PROBE_SWEEP_LIMIT", "60"))

RATIO_MODEL = "qwen/qwen-image-3"
PASSTHROUGH_MODEL = "recraft/recraft-v3"
PASSTHROUGH_SLUG = "recraft"
PASSTHROUGH_KEY = "style"
BAD_STYLE = "not_a_real_style_xyz"
PROMPT = "a single red maple leaf on white paper, flat lay, studio light"

EXCLUSION_MARKER = "cannot be used with the chat/completions endpoint"


class ProbeFailure(AssertionError):
    """A contract assumption is broken."""


class Inconclusive(AssertionError):
    """The probe could not reach the question it exists to answer."""


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/",
        "X-OpenRouter-Title": "Open WebUI OpenRouter pipe image probe",
    }


async def _json_or_text(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        payload = await resp.json()
    except Exception:
        payload = {"text": await resp.text()}
    if isinstance(payload, dict):
        return payload
    return {"payload": payload}


async def _get(session: aiohttp.ClientSession, path: str) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    async with session.get(url, headers=_headers()) as resp:
        payload = await _json_or_text(resp)
        if resp.status >= 400:
            raise ProbeFailure(f"GET {url} failed HTTP {resp.status}: {payload}")
        return payload


async def _post(
    session: aiohttp.ClientSession, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    async with session.post(url, headers=_headers(), json=payload) as resp:
        body = await _json_or_text(resp)
    return {"url": url, "method": "POST", "status": resp.status, "request": payload, "response": body}


def _write_fixture(name: str, payload: Any) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / name).write_text(
        json.dumps(_redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _container(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return f"webp/{raw[12:16].decode('ascii', 'replace').strip()}"
    if raw[:2] == b"\xff\xd8":
        return "jpeg"
    stripped = raw.lstrip()[:5].lower()
    if stripped.startswith(b"<svg") or stripped.startswith(b"<?xml"):
        return "svg"
    return f"unknown:{raw[:8]!r}"


def _dimensions(raw: bytes) -> tuple[int, int] | None:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        chunk = raw[12:16]
        if chunk == b"VP8X":
            return (
                int.from_bytes(raw[24:27], "little") + 1,
                int.from_bytes(raw[27:30], "little") + 1,
            )
        if chunk == b"VP8 ":
            return (
                struct.unpack("<H", raw[26:28])[0] & 0x3FFF,
                struct.unpack("<H", raw[28:30])[0] & 0x3FFF,
            )
        if chunk == b"VP8L":
            bits = int.from_bytes(raw[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if raw[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(raw):
            while offset < len(raw) and raw[offset] == 0xFF and raw[offset + 1] == 0xFF:
                offset += 1
            if raw[offset] != 0xFF:
                return None
            marker = raw[offset + 1]
            length = int.from_bytes(raw[offset + 2 : offset + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return (
                    int.from_bytes(raw[offset + 7 : offset + 9], "big"),
                    int.from_bytes(raw[offset + 5 : offset + 7], "big"),
                )
            if length < 2:
                return None
            offset += 2 + length
    return None


def _describe_image(blob: str) -> dict[str, Any]:
    raw = base64.b64decode(blob)
    dims = _dimensions(raw)
    return {
        "container": _container(raw),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": dims[0] if dims else None,
        "height": dims[1] if dims else None,
    }


def _redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "b64_json" and isinstance(value, str) and value:
                out[key] = f"<redacted {len(value)} base64 chars>"
                out["b64_json_description"] = _describe_image(value)
            else:
                out[key] = _redact(value)
        return out
    if isinstance(payload, list):
        return [_redact(v) for v in payload]
    return payload


def _require_image_envelope(record: dict[str, Any], label: str) -> dict[str, Any]:
    if record["status"] != 200:
        raise Inconclusive(
            f"{label}: HTTP {record['status']} from {record['url']}: {record['response']}"
        )
    data = record["response"].get("data")
    if not isinstance(data, list) or not data:
        raise ProbeFailure(f"{label}: /images response envelope changed — no `data` array")
    entry = data[0]
    if not isinstance(entry, dict):
        raise ProbeFailure(f"{label}: /images response envelope changed — data[0] is not an object")
    blob = entry.get("b64_json")
    if not isinstance(blob, str) or not blob:
        raise ProbeFailure(
            f"{label}: /images no longer returns inline base64 — data[0] keys are {sorted(entry)}. "
            "The adapter decodes b64_json; a URL-bearing envelope needs a download path."
        )
    recorded = entry.get("b64_json_description")
    described = recorded if isinstance(recorded, dict) else _describe_image(blob)
    if described["width"] is None:
        raise ProbeFailure(
            f"{label}: decoder gap — output container is {described['container']!r} "
            f"(media_type {entry.get('media_type')!r}), which this probe cannot measure. "
            "This is not a knob failure; add a decoder."
        )
    return described


async def _gate_dimensions_are_honoured(session: aiohttp.ClientSession) -> None:
    observations: dict[str, Any] = {}
    try:
        for label, resolution in (("1K", "1K"), ("2K", "2K")):
            record = await _post(
                session,
                "/images",
                {
                    "model": RATIO_MODEL,
                    "prompt": PROMPT,
                    "aspect_ratio": "1:2",
                    "resolution": resolution,
                    "n": 1,
                },
            )
            _write_fixture(
                f"openrouter_image_generate_{label}.json",
                record,
            )
            observations[label] = _require_image_envelope(record, f"dimensions gate ({label})")
    finally:
        _write_fixture("openrouter_image_dimension_gate.json", observations)

    for label, seen in observations.items():
        ratio = seen["width"] / seen["height"]
        if abs(ratio - 0.5) > 0.02:
            raise ProbeFailure(
                f"aspect_ratio is no longer reaching the provider: asked 1:2, "
                f"{label} returned {seen['width']}x{seen['height']} (ratio {ratio:.3f})"
            )

    small = min(observations["1K"]["width"], observations["1K"]["height"])
    large = min(observations["2K"]["width"], observations["2K"]["height"])
    if large < small * 1.5:
        raise ProbeFailure(
            "resolution is no longer reaching the provider: 1K gave "
            f"{observations['1K']['width']}x{observations['1K']['height']} and 2K gave "
            f"{observations['2K']['width']}x{observations['2K']['height']} — not materially larger"
        )


async def _gate_passthrough_placement(session: aiohttp.ClientSession) -> None:
    baseline = await _post(
        session,
        "/images",
        {"model": PASSTHROUGH_MODEL, "prompt": PROMPT, "aspect_ratio": "1:1"},
    )
    _write_fixture(
        "openrouter_image_passthrough_baseline.json",
        baseline,
    )
    _require_image_envelope(baseline, "passthrough baseline")

    flat = await _post(
        session,
        "/images",
        {
            "model": PASSTHROUGH_MODEL,
            "prompt": PROMPT,
            "aspect_ratio": "1:1",
            "provider": {"options": {PASSTHROUGH_SLUG: {PASSTHROUGH_KEY: BAD_STYLE}}},
        },
    )
    _write_fixture(
        "openrouter_image_passthrough_flat.json",
        flat,
    )

    wrapped = await _post(
        session,
        "/images",
        {
            "model": PASSTHROUGH_MODEL,
            "prompt": PROMPT,
            "aspect_ratio": "1:1",
            "provider": {
                "options": {PASSTHROUGH_SLUG: {"parameters": {PASSTHROUGH_KEY: BAD_STYLE}}}
            },
        },
    )
    _write_fixture(
        "openrouter_image_passthrough_wrapped.json",
        wrapped,
    )

    if flat["status"] == wrapped["status"]:
        raise Inconclusive(
            "passthrough gate has lost its discriminating power: the flat and wrapped "
            f"placements both returned HTTP {flat['status']}. Neither conclusion is available."
        )

    if flat["status"] != 400:
        raise Inconclusive(
            f"passthrough gate: the flat placement returned HTTP {flat['status']}, expected a "
            f"provider rejection. Body: {flat['response']}"
        )
    error = flat["response"].get("error") or {}
    rendered = json.dumps(flat["response"])
    if BAD_STYLE not in rendered:
        raise ProbeFailure(
            "passthrough gate: the flat placement was rejected but the response does not echo "
            f"the value we sent, so it cannot be attributed to the provider. Body: {rendered[:400]}"
        )
    if not (error.get("metadata") or {}).get("provider_name"):
        raise ProbeFailure(
            "passthrough gate: the flat placement's rejection carries no provider_name, so it "
            "may have come from OpenRouter's own validator rather than the provider. "
            f"Body: {rendered[:400]}"
        )

    if wrapped["status"] != 200:
        raise Inconclusive(
            f"passthrough gate: the wrapped placement returned HTTP {wrapped['status']}, so we "
            "cannot tell whether the wrapper is ignored or the call simply failed. "
            f"Body: {wrapped['response']}"
        )
    _require_image_envelope(wrapped, "passthrough wrapped")


def _assert_models_shape(payload: dict[str, Any]) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ProbeFailure("/images/models did not return a non-empty data array")
    required = {"id", "name", "architecture", "supported_parameters", "supports_streaming"}
    missing = [
        item.get("id", "<missing>")
        for item in data
        if isinstance(item, dict) and not required <= set(item)
    ]
    if missing:
        raise ProbeFailure(f"/images/models entries missing required fields: {missing[:5]}")

    valid_types = {"enum", "range", "boolean"}
    for item in data:
        params = item.get("supported_parameters")
        if not isinstance(params, dict):
            raise ProbeFailure(
                f"{item.get('id')}: supported_parameters is {type(params).__name__}, not a dict of "
                "typed descriptors — the catalog-driven renderer reads it as a mapping"
            )
        for name, descriptor in params.items():
            kind = descriptor.get("type") if isinstance(descriptor, dict) else None
            if kind not in valid_types:
                raise ProbeFailure(
                    f"{item.get('id')}: capability descriptor {name!r} has unknown type {kind!r}"
                )
            if kind == "enum" and not isinstance(descriptor.get("values"), list):
                raise ProbeFailure(f"{item.get('id')}: enum {name!r} has no values list")
            if kind == "range" and not {"min", "max"} <= set(descriptor):
                raise ProbeFailure(f"{item.get('id')}: range {name!r} has no min/max")


def _assert_endpoints_shape(model_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "data" in payload and "endpoints" not in payload:
        raise ProbeFailure(
            f"{model_id}: endpoints payload is now wrapped in `data` — it was not before; "
            "the adapter unwraps it directly"
        )
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise ProbeFailure(f"{model_id}: no endpoints returned")
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise ProbeFailure(f"{model_id}: endpoint entry is not an object")
        if not isinstance(endpoint.get("provider_slug"), str):
            raise ProbeFailure(
                f"{model_id}: endpoint has no provider_slug — provider.options cannot be keyed"
            )
        if not isinstance(endpoint.get("supported_parameters"), dict):
            raise ProbeFailure(f"{model_id}: endpoint has no supported_parameters")
    return endpoints


def _assert_probe_constants_still_valid(endpoints: list[dict[str, Any]]) -> None:
    slugs = {e.get("provider_slug") for e in endpoints}
    if PASSTHROUGH_SLUG not in slugs:
        raise ProbeFailure(
            f"PASSTHROUGH_SLUG={PASSTHROUGH_SLUG!r} is no longer a provider_slug for "
            f"{PASSTHROUGH_MODEL} (now {sorted(s for s in slugs if s)}). Update the constant."
        )
    for endpoint in endpoints:
        if endpoint.get("provider_slug") != PASSTHROUGH_SLUG:
            continue
        allowed = endpoint.get("allowed_passthrough_parameters") or []
        if PASSTHROUGH_KEY not in allowed:
            raise ProbeFailure(
                f"PASSTHROUGH_KEY={PASSTHROUGH_KEY!r} is no longer in "
                f"{PASSTHROUGH_MODEL}'s allowed_passthrough_parameters ({allowed}). A rejection "
                "would now come from OpenRouter's validator, not the provider."
            )


async def _record_chat_completions_eligibility(session: aiohttp.ClientSession) -> None:
    listing = await _get(session, "/models?output_modalities=image")
    candidates = [
        item
        for item in (listing.get("data") or [])
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and "image" in ((item.get("architecture") or {}).get("output_modalities") or [])
    ]
    if len(candidates) > SWEEP_LIMIT:
        raise ProbeFailure(
            f"eligibility sweep would issue {len(candidates)} billable calls, over the "
            f"{SWEEP_LIMIT} cap. The server-side output_modalities filter may have been "
            "dropped. Raise OPENROUTER_IMAGE_PROBE_SWEEP_LIMIT deliberately if this is real."
        )

    rows: list[dict[str, Any]] = []
    for item in candidates:
        model_id = item["id"]
        modalities = (item.get("architecture") or {}).get("output_modalities") or []
        record = await _post(
            session,
            "/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": PROMPT}],
                "modalities": ["image", "text"] if "text" in modalities else ["image"],
            },
        )
        body = record["response"]
        message = ((body.get("error") or {}).get("message") or "") if isinstance(body, dict) else ""
        produced = any(
            (choice.get("message") or {}).get("images")
            for choice in (body.get("choices") or [])
            if isinstance(choice, dict)
        )
        rows.append(
            {
                "model": model_id,
                "url": record["url"],
                "status": record["status"],
                "image_api_only": EXCLUSION_MARKER in message,
                "produced_image": produced,
                "error": message,
            }
        )
        _write_fixture("openrouter_image_chat_eligibility.json", rows)

    if len(rows) != len(candidates):
        raise ProbeFailure(f"sweep recorded {len(rows)} rows for {len(candidates)} models")


async def main() -> int:
    if not API_KEY:
        print("OPENROUTER_API_KEY is required for the live image probe.", file=sys.stderr)
        return 2
    if not RUN_PROBE:
        print(
            "Set OPENROUTER_RUN_IMAGE_PROBE=1 to acknowledge live image-generation calls.",
            file=sys.stderr,
        )
        return 2

    full_sweep = os.getenv("OPENROUTER_IMAGE_PROBE_FULL_SWEEP", "").strip() == "1"

    async with aiohttp.ClientSession() as session:
        models_payload = await _get(session, "/images/models")
        _write_fixture("openrouter_image_models.json", models_payload)
        _assert_models_shape(models_payload)

        for model_id in (RATIO_MODEL, PASSTHROUGH_MODEL):
            payload = await _get(session, f"/images/models/{model_id}/endpoints")
            _write_fixture(
                f"openrouter_image_endpoints_{model_id.replace('/', '_')}.json", payload
            )
            endpoints = _assert_endpoints_shape(model_id, payload)
            if model_id == PASSTHROUGH_MODEL:
                _assert_probe_constants_still_valid(endpoints)

        await _gate_dimensions_are_honoured(session)
        await _gate_passthrough_placement(session)

        if full_sweep:
            await _record_chat_completions_eligibility(session)
        else:
            print(
                "Skipping the chat-completions eligibility sweep "
                "(set OPENROUTER_IMAGE_PROBE_FULL_SWEEP=1 to run it; it generates one image "
                "per model that still accepts the transport).",
                file=sys.stderr,
            )

    print("image probe gates passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except ProbeFailure as exc:
        print(f"CONTRACT BROKEN: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
