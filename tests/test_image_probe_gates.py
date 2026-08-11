from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

PROBE_PATH = Path(__file__).resolve().parent / "integration" / "probe_image_generation.py"


def _load_probe(fixture_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_probe_image_generation", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIXTURE_DIR = fixture_dir
    return module


@pytest.fixture
def probe(tmp_path: Path) -> Any:
    return _load_probe(tmp_path)


def _png(width: int, height: int) -> str:
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )
    return base64.b64encode(raw).decode()


def _image_body(width: int, height: int, count: int = 1) -> dict[str, Any]:
    return {
        "data": [
            {"b64_json": _png(width, height), "media_type": "image/png"} for _ in range(count)
        ]
    }


def _record(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": "https://openrouter.ai/api/v1/images",
        "method": "POST",
        "status": status,
        "request": {},
        "response": body,
    }


def _drive(probe: Any, gate: Any, responses: list[dict[str, Any]]) -> None:
    queue = list(responses)

    async def fake_post(_session, _path, _payload):
        return queue.pop(0)

    probe._post = fake_post
    asyncio.run(gate(None))


def _recording(name: str) -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"openrouter_image_{name}.json").read_text()
    )


def test_dimensions_gate_passes_on_recorded_reality(probe):
    recorded = _recording("dimension_gate")
    _drive(
        probe,
        probe._gate_dimensions_are_honoured,
        [
            _record(200, _image_body(recorded["1K"]["width"], recorded["1K"]["height"])),
            _record(200, _image_body(recorded["2K"]["width"], recorded["2K"]["height"])),
        ],
    )


def test_the_emitted_placement_is_the_one_the_recording_shows_reaching_the_provider():
    from open_webui_openrouter_pipe.integrations.provider_options import merge_provider_options

    flat = _recording("passthrough_flat")
    wrapped = _recording("passthrough_wrapped")
    knob = flat["request"]["provider"]["options"]["recraft"]

    assert flat["status"] == 400, (
        "the flat placement is believed correct because the provider rejected the bogus "
        "value; a recording that shows 200 would mean the value never reached it"
    )
    assert flat["response"]["error"]["metadata"]["provider_name"], (
        "provider_name is what proves the provider itself answered rather than the router"
    )
    assert wrapped["status"] == 200 and wrapped["response"]["data"], (
        "the wrapped placement is believed inert because the same bogus value generated an "
        "image instead of an error; a recording that shows 400 would reverse the decision"
    )

    emitted = merge_provider_options({}, "recraft", knob)
    assert emitted == flat["request"]["provider"], (
        f"the pipe must emit the placement the recording shows reaching the provider. "
        f"emitted={emitted!r}"
    )
    assert emitted != wrapped["request"]["provider"], (
        "re-introducing the parameters wrapper would produce the shape OpenRouter silently "
        "discarded"
    )


@pytest.mark.parametrize(
    "name",
    ["generate_1K", "generate_2K", "passthrough_baseline", "passthrough_wrapped"],
)
def test_a_recorded_generation_replays_through_the_probes_own_envelope_reader(probe, name):
    record = _recording(name)
    probe._require_image_envelope(record, name)

    asked = record.get("request") or {}
    assert asked.get("model"), f"{name} records no model, so it cannot say what it exercised"
    if name.startswith("generate_"):
        assert asked.get("resolution") == name.removeprefix("generate_"), (
            "a recording named for a resolution tier must have asked for that tier; "
            f"{name} asked for {asked.get('resolution')!r}"
        )


@pytest.mark.parametrize("label", ["1K", "2K"])
def test_the_dimension_gate_agrees_with_the_generation_it_was_derived_from(label):
    gate = _recording("dimension_gate")[label]
    recorded = _recording(f"generate_{label}")["response"]["data"][0]["b64_json_description"]

    assert gate == recorded, (
        "the probe writes both files from one HTTP exchange -- the gate summary is derived "
        "from the generation recording -- so they cannot disagree. If they do, one was "
        f"edited by hand. gate={gate!r} recorded={recorded!r}"
    )


def test_the_recorded_usage_is_translated_out_of_the_chat_spelling():
    from open_webui_openrouter_pipe.core.costs import chat_usage_to_responses_usage

    recorded = _recording("passthrough_baseline")["response"]["usage"]
    assert "prompt_tokens" in recorded and "completion_tokens" in recorded, (
        "/images answers in the chat spelling; if a recording shows the responses spelling "
        "the translation at the transport boundary is no longer needed"
    )

    translated = chat_usage_to_responses_usage(recorded)
    assert translated["input_tokens"] == recorded["prompt_tokens"]
    assert translated["output_tokens"] == recorded["completion_tokens"]


def test_dimensions_gate_catches_resolution_no_longer_applied(probe):
    with pytest.raises(probe.ProbeFailure, match="resolution"):
        _drive(
            probe,
            probe._gate_dimensions_are_honoured,
            [_record(200, _image_body(1024, 2048)), _record(200, _image_body(1024, 2048))],
        )


def test_dimensions_gate_catches_aspect_ratio_no_longer_applied(probe):
    with pytest.raises(probe.ProbeFailure, match="aspect_ratio"):
        _drive(
            probe,
            probe._gate_dimensions_are_honoured,
            [_record(200, _image_body(1024, 1024)), _record(200, _image_body(2048, 2048))],
        )


@pytest.mark.parametrize(
    ("small", "large"),
    [
        ((512, 1024), (1024, 2040)),
        ((720, 1440), (1470, 2940)),
    ],
)
def test_dimensions_gate_tolerates_benign_provider_rounding(probe, small, large):
    _drive(
        probe,
        probe._gate_dimensions_are_honoured,
        [_record(200, _image_body(*small)), _record(200, _image_body(*large))],
    )


def test_dimensions_gate_reports_an_envelope_change_as_such(probe):
    with pytest.raises(probe.ProbeFailure, match="inline base64"):
        _drive(
            probe,
            probe._gate_dimensions_are_honoured,
            [
                _record(200, {"data": [{"url": "https://example.invalid/a.png"}]}),
                _record(200, _image_body(1024, 2048)),
            ],
        )


def test_dimensions_gate_writes_its_fixture_on_the_failing_path(probe, tmp_path):
    with pytest.raises(probe.ProbeFailure):
        _drive(
            probe,
            probe._gate_dimensions_are_honoured,
            [
                _record(200, _image_body(1024, 1024)),
                _record(200, _image_body(2048, 2048)),
            ],
        )
    assert (tmp_path / "openrouter_image_generate_1K.json").exists()
    assert (tmp_path / "openrouter_image_dimension_gate.json").exists()


def _attributed_rejection(probe: Any) -> dict[str, Any]:
    return {
        "error": {
            "message": f"Invalid style '{probe.BAD_STYLE}'",
            "code": 400,
            "metadata": {"provider_name": "Recraft"},
        }
    }


def test_passthrough_gate_passes_on_recorded_reality(probe):
    _drive(
        probe,
        probe._gate_passthrough_placement,
        [
            _record(200, _image_body(1024, 1024)),
            _record(400, _attributed_rejection(probe)),
            _record(200, _image_body(1024, 1024)),
        ],
    )


def test_passthrough_gate_rejects_a_400_that_does_not_echo_the_value(probe):
    with pytest.raises(probe.ProbeFailure, match="echo"):
        _drive(
            probe,
            probe._gate_passthrough_placement,
            [
                _record(200, _image_body(1024, 1024)),
                _record(400, {"error": {"message": "aspect_ratio '1:1' is not supported"}}),
                _record(200, _image_body(1024, 1024)),
            ],
        )


def test_passthrough_gate_rejects_a_400_with_no_provider_attribution(probe):
    with pytest.raises(probe.ProbeFailure, match="provider_name"):
        _drive(
            probe,
            probe._gate_passthrough_placement,
            [
                _record(200, _image_body(1024, 1024)),
                _record(400, {"error": {"message": f"Invalid style '{probe.BAD_STYLE}'"}}),
                _record(200, _image_body(1024, 1024)),
            ],
        )


@pytest.mark.parametrize("status", [429, 502, 401])
def test_passthrough_gate_calls_a_failed_second_leg_inconclusive(probe, status):
    with pytest.raises(probe.Inconclusive, match=str(status)):
        _drive(
            probe,
            probe._gate_passthrough_placement,
            [
                _record(200, _image_body(1024, 1024)),
                _record(400, _attributed_rejection(probe)),
                _record(status, {"error": {"message": "nope"}}),
            ],
        )


def test_passthrough_gate_detects_lost_discriminating_power(probe):
    with pytest.raises(probe.Inconclusive, match="discriminating"):
        _drive(
            probe,
            probe._gate_passthrough_placement,
            [
                _record(200, _image_body(1024, 1024)),
                _record(200, _image_body(1024, 1024)),
                _record(200, _image_body(1024, 1024)),
            ],
        )


def test_passthrough_gate_requires_its_baseline_to_succeed(probe):
    with pytest.raises(probe.Inconclusive, match="500"):
        _drive(
            probe,
            probe._gate_passthrough_placement,
            [
                _record(500, {"error": {"message": "upstream"}}),
                _record(400, _attributed_rejection(probe)),
                _record(200, _image_body(1024, 1024)),
            ],
        )


def test_constants_gate_catches_a_renamed_provider_slug(probe):
    with pytest.raises(probe.ProbeFailure, match="PASSTHROUGH_SLUG"):
        probe._assert_probe_constants_still_valid(
            [
                {
                    "provider_slug": "recraft-ai",
                    "supported_parameters": {},
                    "allowed_passthrough_parameters": ["style"],
                }
            ]
        )


def test_constants_gate_catches_the_key_leaving_the_allow_list(probe):
    with pytest.raises(probe.ProbeFailure, match="PASSTHROUGH_KEY"):
        probe._assert_probe_constants_still_valid(
            [
                {
                    "provider_slug": "recraft",
                    "supported_parameters": {},
                    "allowed_passthrough_parameters": ["controls"],
                }
            ]
        )


def test_constants_gate_accepts_a_multi_endpoint_model(probe):
    probe._assert_probe_constants_still_valid(
        [
            {
                "provider_slug": "someone-else",
                "supported_parameters": {},
                "allowed_passthrough_parameters": [],
            },
            {
                "provider_slug": "recraft",
                "supported_parameters": {},
                "allowed_passthrough_parameters": ["style", "controls"],
            },
        ]
    )


def test_models_shape_rejects_a_list_shaped_supported_parameters(probe):
    with pytest.raises(probe.ProbeFailure, match="not a dict"):
        probe._assert_models_shape(
            {
                "data": [
                    {
                        "id": "x/y",
                        "name": "Y",
                        "architecture": {},
                        "supported_parameters": ["aspect_ratio", "n"],
                        "supports_streaming": False,
                    }
                ]
            }
        )


def test_models_shape_rejects_an_unknown_descriptor_type(probe):
    with pytest.raises(probe.ProbeFailure, match="unknown type"):
        probe._assert_models_shape(
            {
                "data": [
                    {
                        "id": "x/y",
                        "name": "Y",
                        "architecture": {},
                        "supported_parameters": {"n": {"type": "interval", "min": 1, "max": 2}},
                        "supports_streaming": False,
                    }
                ]
            }
        )


def test_endpoints_shape_rejects_a_newly_wrapped_payload(probe):
    with pytest.raises(probe.ProbeFailure, match="wrapped in `data`"):
        probe._assert_endpoints_shape("x/y", {"data": [{"provider_slug": "z"}]})


def test_endpoints_shape_requires_a_provider_slug(probe):
    with pytest.raises(probe.ProbeFailure, match="provider_slug"):
        probe._assert_endpoints_shape(
            "x/y", {"endpoints": [{"supported_parameters": {}}]}
        )


def test_dimensions_decodes_lossless_webp(probe):
    header = b"RIFF" + (0).to_bytes(4, "little") + b"WEBPVP8L" + (0).to_bytes(4, "little")
    bits = (512 - 1) | ((1024 - 1) << 14)
    raw = header + b"\x2f" + bits.to_bytes(4, "little")
    assert probe._dimensions(raw) == (512, 1024)


def test_container_is_sniffed_not_taken_from_the_declared_media_type(probe):
    raw = b"RIFF" + (0).to_bytes(4, "little") + b"WEBPVP8 " + b"\x00" * 16
    assert probe._container(raw).startswith("webp")


@pytest.mark.parametrize(
    ("api_key", "run_probe"), [("", True), ("sk-live", False), ("", False)]
)
@pytest.mark.asyncio
async def test_the_probe_issues_no_call_without_both_acknowledgements(probe, api_key, run_probe):
    calls: list[str] = []

    async def _forbidden(*_a, **_k):
        calls.append("network")
        raise AssertionError("the probe must not touch the network without both acknowledgements")

    probe.API_KEY = api_key
    probe.RUN_PROBE = run_probe
    probe._get = _forbidden
    probe._post = _forbidden

    rc = await probe.main()

    assert rc != 0
    assert not calls, "a live probe fired without an explicit acknowledgement"


@pytest.mark.asyncio
async def test_the_sweep_refuses_to_start_over_the_cap(probe):
    posts: list[str] = []

    async def _post(*_a, **_k):
        posts.append("billable")
        return {}

    async def _get(_session, path, **_k):
        return {"data": [{"id": f"m/{i}", "architecture": {"output_modalities": ["image"]}} for i in range(5)]}

    probe.SWEEP_LIMIT = 2
    probe._post = _post
    probe._get = _get

    with pytest.raises(probe.ProbeFailure, match="cap"):
        await probe._record_chat_completions_eligibility(object())

    assert not posts, "the cap must refuse before any billable call is issued"


@pytest.mark.asyncio
async def test_the_sweep_proceeds_under_the_cap(probe):
    async def _get(_session, path, **_k):
        return {"data": [{"id": "m/0", "architecture": {"output_modalities": ["image"]}}]}

    async def _post(*_a, **_k):
        return {"status": 200, "url": "x", "response": {"choices": []}}

    probe.SWEEP_LIMIT = 60
    probe._get = _get
    probe._post = _post

    await probe._record_chat_completions_eligibility(object())


@pytest.mark.parametrize("blob_len", [40_000, 90_000])
def test_a_written_fixture_never_carries_raw_base64(probe, tmp_path, blob_len):
    blob = "A" * blob_len
    probe._write_fixture("redaction_probe.json", {"response": {"data": [{"b64_json": blob}]}})

    text = (tmp_path / "redaction_probe.json").read_text()
    assert blob not in text, "the probe would commit megabytes of raw base64 into the repo"
    written = json.loads(text)
    entry = written["response"]["data"][0]
    assert entry["b64_json"] == f"<redacted {blob_len} base64 chars>", (
        "redaction must be a property of the writer, not of remembering to wrap each call, "
        "and it must leave b64_json a string so a replayed recording is not misread as an "
        "envelope that stopped carrying inline base64"
    )
    assert isinstance(entry["b64_json_description"], dict), (
        "the digest is what makes the recording evidence; dropping it leaves a placeholder"
    )


@pytest.mark.parametrize(
    ("env", "rc_nonzero"),
    [
        ({}, True),
        ({"OPENROUTER_API_KEY": "k"}, True),
        ({"OPENROUTER_API_KEY": "k", "OPENROUTER_RUN_IMAGE_PROBE": "0"}, True),
    ],
)
def test_the_acknowledgement_is_read_from_the_environment(tmp_path, monkeypatch, env, rc_nonzero):
    import asyncio

    for name in ("OPENROUTER_API_KEY", "OPENROUTER_RUN_IMAGE_PROBE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    probe = _load_probe(tmp_path)
    calls: list[str] = []

    async def _forbidden(*_a, **_k):
        calls.append("network")
        raise AssertionError("the probe must not touch the network unacknowledged")

    probe._get = _forbidden
    probe._post = _forbidden

    assert (asyncio.run(probe.main()) != 0) is rc_nonzero, (
        "the gate must read the environment, not a module attribute a test can set; the '0' "
        "case is what distinguishes == '1' from truthiness"
    )
    assert not calls


def test_the_models_shape_gate_rejects_a_missing_required_field(probe):
    with pytest.raises(probe.ProbeFailure, match="required"):
        probe._assert_models_shape(
            {"data": [{"id": "x/y", "architecture": {}, "supported_parameters": {}}]}
        )


def test_the_models_shape_gate_rejects_an_enum_without_values(probe):
    payload = {
        "data": [
            {
                "id": "x/y",
                "name": "X",
                "architecture": {"output_modalities": ["image"]},
                "supported_parameters": {"aspect_ratio": {"type": "enum"}},
                "supports_streaming": False,
            }
        ]
    }
    with pytest.raises(probe.ProbeFailure):
        probe._assert_models_shape(payload)


def test_the_models_shape_gate_rejects_a_range_without_bounds(probe):
    payload = {
        "data": [
            {
                "id": "x/y",
                "name": "X",
                "architecture": {"output_modalities": ["image"]},
                "supported_parameters": {"n": {"type": "range"}},
                "supports_streaming": False,
            }
        ]
    }
    with pytest.raises(probe.ProbeFailure):
        probe._assert_models_shape(payload)


@pytest.mark.parametrize(
    ("message", "flagged"),
    [("cannot be used with the chat/completions endpoint", True), ("rate limited", False)],
)
@pytest.mark.asyncio
async def test_the_sweep_classifies_the_exclusion_marker(probe, message, flagged):
    async def _get(_session, _path, **_k):
        return {"data": [{"id": "m/0", "architecture": {"output_modalities": ["image"]}}]}

    async def _post(*_a, **_k):
        return {"status": 404, "url": "x", "response": {"error": {"message": message}}}

    probe.SWEEP_LIMIT = 60
    probe._get = _get
    probe._post = _post

    await probe._record_chat_completions_eligibility(object())

    written = json.loads((probe.FIXTURE_DIR / "openrouter_image_chat_eligibility.json").read_text())
    rows = written if isinstance(written, list) else written.get("rows") or written.get("data")
    assert rows and rows[0]["image_api_only"] is flagged, (
        "image_api_only is the field the whole sweep exists to record; it must key on the "
        f"exclusion marker. row was {rows[0] if rows else None!r}"
    )
