"""The live video probe's gates must be able to fail.

A gate that only checks "the request was accepted" cannot tell a parameter that reached
the provider from one the provider ignored, and a discarded parameter still produces a
completed video. These tests drive each gate with synthetic records so the failing path
is exercised without touching the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

PROBE_PATH = Path(__file__).resolve().parent / "integration" / "probe_video_generation.py"


def _load_probe(fixture_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_probe_video_generation", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIXTURE_DIR = fixture_dir
    return module


@pytest.fixture
def probe(tmp_path: Path) -> Any:
    return _load_probe(tmp_path)


def _rejection(value: str, provider: str = "AtlasCloud") -> dict[str, Any]:
    return {
        "url": "https://openrouter.ai/api/v1/videos",
        "status": 400,
        "response": {
            "error": {
                "message": f"Invalid ratio '{value}'",
                "metadata": {"provider_name": provider},
            }
        },
    }


def _accepted() -> dict[str, Any]:
    return {
        "url": "https://openrouter.ai/api/v1/videos",
        "status": 200,
        "response": {"id": "job-1", "status": "pending"},
    }


def _drive(probe: Any, flat: dict[str, Any], wrapped: dict[str, Any]) -> None:
    import asyncio

    async def _post_record(_session, _path, payload):
        options = ((payload.get("provider") or {}).get("options") or {}).get(
            probe.PASSTHROUGH_SLUG
        ) or {}
        return wrapped if "parameters" in options else flat

    probe._post_record = _post_record
    asyncio.run(probe._gate_passthrough_placement(object()))


def test_the_gate_passes_when_the_flat_placement_reaches_the_provider(probe):
    _drive(probe, _rejection(probe.BAD_RATIO), _accepted())


def test_the_gate_is_inconclusive_when_both_placements_agree(probe):
    with pytest.raises(probe.Inconclusive, match="discriminating"):
        _drive(probe, _accepted(), _accepted())


def test_the_gate_fails_when_the_flat_placement_is_inert(probe):
    with pytest.raises(probe.ProbeFailure, match="inert"):
        _drive(probe, {**_accepted(), "status": 201}, _accepted())


def test_the_gate_fails_when_the_rejection_does_not_echo_the_value(probe):
    silent = _rejection(probe.BAD_RATIO)
    silent["response"]["error"]["message"] = "Invalid parameter"
    with pytest.raises(probe.ProbeFailure, match="echo"):
        _drive(probe, silent, _accepted())


def test_the_gate_fails_when_the_rejection_is_not_attributed_to_a_provider(probe):
    unattributed = _rejection(probe.BAD_RATIO)
    unattributed["response"]["error"]["metadata"] = {}
    with pytest.raises(probe.ProbeFailure, match="provider_name"):
        _drive(probe, unattributed, _accepted())


def test_the_gate_records_both_placements_for_the_next_reader(probe, tmp_path):
    import json

    _drive(probe, _rejection(probe.BAD_RATIO), _accepted())

    written = json.loads((tmp_path / "openrouter_video_passthrough_probe.json").read_text())
    assert written["flat"]["status"] == 400
    assert written["wrapped"]["status"] == 200


def test_the_probe_makes_no_call_without_both_acknowledgements(probe):
    import asyncio

    calls: list[str] = []

    async def _forbidden(*_a, **_k):
        calls.append("network")
        raise AssertionError("the probe must not touch the network unacknowledged")

    probe.API_KEY = ""
    probe.RUN_PROBE = True
    probe._get = _forbidden
    probe._post = _forbidden
    probe._post_record = _forbidden

    assert asyncio.run(probe.main()) != 0
    assert not calls


def _video_recording() -> dict:
    import json
    from pathlib import Path

    return json.loads(
        (
            Path(__file__).parent / "fixtures" / "openrouter_video_passthrough_probe.json"
        ).read_text()
    )


def test_the_emitted_placement_is_the_one_the_recording_shows_the_provider_honoured():
    """The video schema's own guide shows the `parameters` wrapper; the provider ignores it.

    Every placement is accepted with a job id, so status cannot decide this. The recorded
    discriminator is the produced frame: the honoured placement draws the watermark. Without
    this recording the decision rests on a comment, which is how the image side's placement
    was nearly reversed on a reading of the OpenAPI example block.
    """
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    recorded = _video_recording()
    honoured = [name for name, arm in recorded["arms"].items() if arm["watermark_rendered"]]
    assert honoured == ["flat"], (
        f"the recording must show exactly one placement taking effect; got {honoured}"
    )

    knob = {"watermark": True}
    emitted = VideoGenerationAdapter._normalise_provider_options({"seed": knob})

    assert emitted == recorded["arms"]["flat"]["request_provider"]["options"], (
        f"the pipe must emit the placement the provider honoured. got {emitted!r}"
    )
    assert emitted != recorded["arms"]["wrapped"]["request_provider"]["options"], (
        "re-introducing the parameters wrapper produces the shape the recording shows was "
        "accepted and silently ignored"
    )
