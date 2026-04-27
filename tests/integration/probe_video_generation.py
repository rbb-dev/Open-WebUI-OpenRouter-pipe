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

        shape_a = {
            "model": "google/veo-3.1-fast",
            "prompt": "A short product shot of a red cube rotating on a white background.",
            "provider": {
                "options": {
                    "google-vertex": {
                        "parameters": {
                            "negativePrompt": "blur, flicker",
                        }
                    }
                }
            },
        }
        shape_b = {
            "model": "google/veo-3.1-fast",
            "prompt": "A short product shot of a blue cube rotating on a white background.",
            "provider": {
                "options": {
                    "google-vertex": {
                        "negativePrompt": "blur, flicker",
                    }
                }
            },
        }
        shape_results: dict[str, str] = {}
        for label, payload in (("provider_parameters_wrapper", shape_a), ("provider_bare_options", shape_b)):
            try:
                await _submit_and_require_success(session, label=label, payload=payload)
                shape_results[label] = "accepted"
            except Exception as exc:
                shape_results[label] = f"rejected: {exc}"

        _write_fixture("openrouter_video_provider_passthrough_probe.json", shape_results)
        if shape_results.get("provider_parameters_wrapper") != "accepted":
            raise AssertionError(f"provider.options.<slug>.parameters shape failed: {shape_results}")
        if all(value != "accepted" for value in shape_results.values()):
            raise AssertionError(f"No provider passthrough shape was accepted: {shape_results}")
    print("OpenRouter video probe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
