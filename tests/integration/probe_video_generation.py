"""Live OpenRouter video-generation probe.

Run manually:
    PYTHONPATH=. .venv/bin/python tests/integration/probe_video_generation.py

This script intentionally makes live /videos calls when OPENROUTER_API_KEY is set.
It records probe responses under tests/fixtures/ and exits non-zero on any gate
that would invalidate the production adapter assumptions.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
RUN_PROBE = os.getenv("OPENROUTER_RUN_VIDEO_PROBE", "").strip() == "1"

ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class ProbeFailure(AssertionError):
    """A gate proved the production assumption wrong."""


class Inconclusive(AssertionError):
    """A gate could not distinguish the outcomes it exists to distinguish."""


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/",
        "X-OpenRouter-Title": "Open WebUI OpenRouter pipe video probe",
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
    async with session.get(f"{BASE_URL}{path}", headers=_headers()) as resp:
        payload = await _json_or_text(resp)
        if resp.status >= 400:
            raise RuntimeError(f"GET {path} failed HTTP {resp.status}: {payload}")
        return payload


async def _post(session: aiohttp.ClientSession, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with session.post(f"{BASE_URL}{path}", headers=_headers(), json=payload) as resp:
        data = await _json_or_text(resp)
        if resp.status >= 400:
            raise RuntimeError(f"POST {path} failed HTTP {resp.status}: {data}")
        return data


async def _post_record(
    session: aiohttp.ClientSession, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST and record the status instead of raising, so a rejection is evidence."""
    url = f"{BASE_URL}{path}"
    async with session.post(url, headers=_headers(), json=payload) as resp:
        return {"url": url, "status": resp.status, "response": await _json_or_text(resp)}


async def _poll(session: aiohttp.ClientSession, job_id: str, label: str) -> dict[str, Any]:
    deadline = time.monotonic() + int(os.getenv("OPENROUTER_VIDEO_PROBE_TIMEOUT", "600"))
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = await _get(session, f"/videos/{job_id}")
        _write_fixture(f"openrouter_video_poll_{label}.json", last)
        status = str(last.get("status", "")).lower()
        if status in {"completed", "succeeded", "success", "failed", "cancelled", "canceled", "expired"}:
            return last
        await asyncio.sleep(5)
    raise TimeoutError(f"Probe job {job_id} did not finish before timeout. Last payload: {last}")


def _job_id(payload: dict[str, Any]) -> str:
    for key in ("id", "job_id", "jobId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        return _job_id(data)
    return ""


def _write_fixture(name: str, payload: dict[str, Any]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_models_shape(payload: dict[str, Any]) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise AssertionError("/videos/models did not return a non-empty data array")
    required = {"id", "name", "supported_aspect_ratios", "supported_frame_images", "allowed_passthrough_parameters"}
    missing = [item.get("id", "<missing>") for item in data if isinstance(item, dict) and not required <= set(item)]
    if missing:
        raise AssertionError(f"/videos/models entries missing required fields: {missing[:5]}")


async def _submit_and_require_success(
    session: aiohttp.ClientSession,
    *,
    label: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    accepted = await _post(session, "/videos", payload)
    _write_fixture(f"openrouter_video_submit_{label}.json", accepted)
    job_id = _job_id(accepted)
    if not job_id:
        raise AssertionError(f"{label}: submit response had no job id: {accepted}")
    terminal = await _poll(session, job_id, label)
    status = str(terminal.get("status", "")).lower()
    if status not in {"completed", "succeeded", "success"}:
        raise AssertionError(f"{label}: terminal status was {status}: {terminal}")
    return terminal


PASSTHROUGH_MODEL = "alibaba/wan-2.7"
PASSTHROUGH_SLUG = "atlas-cloud"
PASSTHROUGH_KEY = "ratio"
BAD_RATIO = "99:1"


async def _gate_passthrough_placement(session: aiohttp.ClientSession) -> None:
    """Prove the flat provider.options placement reaches the provider, not just OpenRouter.

    Acceptance is not evidence: a discarded parameter still yields a completed video. The
    discriminator is a value the provider itself must reject, attributed by provider_name.
    """
    base = {
        "model": PASSTHROUGH_MODEL,
        "prompt": "A short product shot of a blue cube rotating on a white background.",
    }
    flat = await _post_record(
        session,
        "/videos",
        {**base, "provider": {"options": {PASSTHROUGH_SLUG: {PASSTHROUGH_KEY: BAD_RATIO}}}},
    )
    wrapped = await _post_record(
        session,
        "/videos",
        {
            **base,
            "provider": {
                "options": {PASSTHROUGH_SLUG: {"parameters": {PASSTHROUGH_KEY: BAD_RATIO}}}
            },
        },
    )
    _write_fixture(
        "openrouter_video_passthrough_probe.json", {"flat": flat, "wrapped": wrapped}
    )

    if flat["status"] == wrapped["status"]:
        raise Inconclusive(
            "passthrough gate has lost its discriminating power: the flat and wrapped "
            f"placements both returned HTTP {flat['status']}. The provider may be coercing "
            f"{PASSTHROUGH_KEY!r} rather than rejecting it; pick a key it validates."
        )
    if 200 <= flat["status"] < 300:
        raise ProbeFailure(
            "the flat placement was accepted with a value the provider should reject, so the "
            f"pipe's placement may be inert: {flat}"
        )
    rendered = json.dumps(flat["response"])
    if BAD_RATIO not in rendered:
        raise ProbeFailure(
            "the flat placement was rejected but the response does not echo the value we sent, "
            f"so the rejection cannot be attributed to the provider: {rendered[:400]}"
        )
    if not ((flat["response"].get("error") or {}).get("metadata") or {}).get("provider_name"):
        raise ProbeFailure(
            "the flat placement's rejection carries no provider_name, so it may have come from "
            f"OpenRouter's own validator rather than the provider: {rendered[:400]}"
        )


async def main() -> int:
    if not API_KEY:
        print("OPENROUTER_API_KEY is required for the live video probe.", file=sys.stderr)
        return 2
    if not RUN_PROBE:
        print("Set OPENROUTER_RUN_VIDEO_PROBE=1 to acknowledge live video-generation calls.", file=sys.stderr)
        return 2

    async with aiohttp.ClientSession() as session:
        models_payload = await _get(session, "/videos/models")
        _write_fixture("openrouter_video_models.json", models_payload)
        _assert_models_shape(models_payload)

        await _submit_and_require_success(
            session,
            label="sora_text_only",
            payload={
                "model": "openai/sora-2-pro",
                "prompt": "A three second locked-off shot of a white ceramic mug on a wooden desk.",
            },
        )

        await _submit_and_require_success(
            session,
            label="veo_data_url_frame",
            payload={
                "model": "google/veo-3.1-fast",
                "prompt": "Animate the attached image with a very gentle camera push in.",
                "frame_images": [
                    {
                        "type": "image_url",
                        "frame_type": "first_frame",
                        "image_url": {"url": ONE_PIXEL_PNG_DATA_URL},
                    }
                ],
            },
        )

        await _gate_passthrough_placement(session)

    print("OpenRouter video probe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
