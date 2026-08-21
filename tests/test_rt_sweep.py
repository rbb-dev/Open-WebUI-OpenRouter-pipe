"""RED TEAM: full-corpus end-to-end sweep through the real adapter.

Property: for every control every one of the forty recorded contracts renders, a value
that control accepts either arrives on the wire unchanged or is named to the user in a
notification. Silence is the failure -- a knob whose choice is neither sent nor reported
lies to whoever set it.

The disjunction is load-bearing. `IMAGE_SIZE` is free text because OpenRouter documents
two forms for it, a tier and exact pixels, and a fixed list would delete the pixel form;
but a string that is neither is refused, and the user is told. Asserting only the first
half of the disjunction would forbid that check, and asserting only the second would be
satisfied by a pipe that sends nothing and complains about everything.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_image_model_filter_spec,
    render_image_model_filter_source,
)
from tests.test_image_api_path import (  # noqa: F401
    BASE, _adapter, _Emitter, _KeyPipe, _posted, _StubValves, _user_turn_with_images,
    _StubResponsesBody,
)
from tests.test_image_generation import _load_filter_from_source

FIX = Path(__file__).parent / "fixtures"
CONTRACTS = sorted(FIX.glob("openrouter_image_endpoints_*.json"))

_N = [0]


def _mod(spec):
    _N[0] += 1
    return _load_filter_from_source(render_image_model_filter_source(spec), f"rtsw{_N[0]}")


def _choices(uv_cls, name):
    import typing
    f = uv_cls.model_fields[name]
    ann = f.annotation
    out = []
    if str(ann).startswith("typing.Literal") or getattr(ann, "__origin__", None) is typing.Literal:
        out = [v for v in typing.get_args(ann) if v != ""]
    else:
        lo = hi = None
        for m in f.metadata:
            if getattr(m, "ge", None) is not None:
                lo = m.ge
            if getattr(m, "le", None) is not None:
                hi = m.le
        if lo is not None or hi is not None:
            out = [v for v in {lo, hi} if v is not None]
        elif "int" in str(ann):
            out = [1]
        else:
            out = ["SENTINEL"]
    return out


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.stem[27:])
@pytest.mark.asyncio
async def test_every_published_value_reaches_the_wire(path):
    raw = json.loads(path.read_text())
    model_id, records = raw["id"], raw["endpoints"]
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True
    )
    mod = _mod(spec)
    F = mod.Filter()
    uv_cls = mod.Filter.UserValves
    always = {"IMAGE_PROVIDER_OPTIONS_JSON", "IMAGE_REFERENCE_MODE", "IMAGE_REFERENCE_URLS"}
    controls = [n for n in uv_cls.model_fields if n.startswith("IMAGE_") and n not in always]
    problems: list[str] = []
    import time as _t
    for cname in controls:
        for value in _choices(uv_cls, cname):
            uv = uv_cls(**{cname: value})
            body = F.inlet({"model": model_id.replace("/", ".")}, {}, {"valves": uv})
            cfg = dict(body.get("image_config") or {})
            adapter = _adapter(_KeyPipe("sk-x"))
            adapter._endpoint_cache[model_id] = (_t.monotonic(), records)
            emitter = _Emitter()
            result = await _posted(
                adapter,
                body=body,
                responses_body=_StubResponsesBody([{"role": "user", "content": [
                    {"type": "input_text", "text": "draw"}]}]),
                valves=_StubValves("sk-x"),
                event_emitter=emitter,
                normalized_model_id=model_id.replace("/", "."),
                api_model_id=model_id,
            )
            payload = result.payload
            provider_opts = (payload.get("provider") or {}).get("options") or {}
            flat = dict(payload)
            for slug, opts in provider_opts.items():
                for k, v in (opts or {}).items():
                    flat.setdefault(k, v)
            told = " ".join(
                str(event.get("content", ""))
                for event in result.events
                if event.get("type") == "notification"
            )
            for k, v in cfg.items():
                if k not in flat and k not in told:
                    problems.append(
                        f"{cname}={value!r}: {k!r}={v!r} NEITHER ON WIRE NOR REPORTED; "
                        f"payload={payload} notifications={told!r}"
                    )
                elif k in flat and flat[k] != v:
                    problems.append(f"{cname}={value!r}: {k!r} {v!r}->{flat[k]!r}; payload={payload}")
    assert not problems, "\n".join(problems)
