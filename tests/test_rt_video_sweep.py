"""RED TEAM: full video-catalog round trip, every published value to the wire.

Three properties, over every model in the recorded catalog:

1. A value a user can pick reaches the request, under a name the model published.
   The pipe's controls are named once, in the pipe's own spelling, and each model
   publishes its own -- ``google/veo-3.1`` wants ``negativePrompt`` where
   ``kwaivgi/kling-v3.0-pro`` wants ``negative_prompt``. Demanding the pipe's spelling
   on the wire reported a delivered value as lost; accepting anything at all would
   report a value delivered under the wrong name as fine. So the accepted spellings are
   read out of the catalog entry under test and matched as one word, which admits a
   third spelling the day a model publishes one and admits nothing else.

2. The value that arrives is the value that was chosen, not merely a value.

3. A control left at its declared default sends nothing. Every one of these controls
   defaults to an "unset" marker -- ``0`` seconds means "the model picks", ``0.0``
   guidance means "the model keeps its own balance", ``""`` and ``model_default`` mean
   the same -- and forwarding that marker as a setting would report a control as in
   force while the user never touched it. This is why the sweep must not read the
   default back as a lost value: it is the one value whose correct destination is
   nowhere.
"""
from __future__ import annotations

import json
import typing
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.filters.video_filter_renderer import (
    build_video_filter_spec, render_video_filter_source,
)
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from tests.test_video_generation import _load_filter_from_source

FIX = Path(__file__).parent / "fixtures"
CATALOG = json.loads((FIX / "video_models_catalog.json").read_text())["data"]
_MODALITIES = json.loads((FIX / "openrouter_video_input_modalities.json").read_text())["input_modalities"]
for _m in CATALOG:
    _found = _MODALITIES.get(_m.get("id"))
    if _found:
        _m["input_modalities"] = _found
assert len(CATALOG) > 15, f"only {len(CATALOG)} video models"

_N = [0]


def _one_word(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def _published(model: Any) -> list[str]:
    raw = model.get("allowed_passthrough_parameters") if isinstance(model, dict) else None
    return [item for item in (raw or []) if isinstance(item, str) and item]


def _spellings(model: Any, key: str) -> set[str]:
    word = _one_word(key)
    return {key} | {name for name in _published(model) if _one_word(name) == word}


_BY_WORD: dict[str, set[str]] = defaultdict(set)
for _m in CATALOG:
    for _name in _published(_m):
        _BY_WORD[_one_word(_name)].add(_name)
_VARIANTS = {word: names for word, names in _BY_WORD.items() if len(names) > 1}
assert _VARIANTS, (
    "no control in the recorded catalog is published under two spellings, so the arm of "
    "this sweep that accepts a model's own spelling is never driven and could be deleted "
    "with every node still green. Re-record the catalog or drop that arm deliberately."
)


class _Logger:
    def __getattr__(self, _n):
        return lambda *a, **k: None
    def isEnabledFor(self, _l): return False


class _MM:
    async def _is_safe_url(self, url, *, seconds=5.0): return True


class _CatMgr:
    def __init__(self, slugs): self._slugs = slugs
    def get_cached_provider_map(self): return self._slugs


class _Pipe:
    def __init__(self, slugs):
        self._multimodal_handler = _MM()
        self._catalog_manager = _CatMgr(slugs)


def _choices(uv_cls, name):
    """Every value a user can pick that is not the control's own "leave it alone" marker.

    The marker is read off the field rather than guessed at: it is the declared default,
    which across the whole recorded catalog is exactly the value whose inlet writes
    nothing. Spelling it as a literal was what let ``0`` be excluded for the enum and
    integer arms and swept for the float arm, where it arrived as a lost value.
    """
    f = uv_cls.model_fields[name]
    unset = f.default
    ann = f.annotation
    if str(ann).startswith("typing.Literal") or getattr(ann, "__origin__", None) is typing.Literal:
        return [v for v in typing.get_args(ann) if v != unset and v not in ("auto", "model_default")]
    s = str(ann)
    lo = hi = None
    for m in f.metadata:
        if getattr(m, "ge", None) is not None:
            lo = m.ge
        if getattr(m, "le", None) is not None:
            hi = m.le
    if "float" in s:
        return [v for v in {lo, hi} if v is not None and v != unset] or [1.0]
    if "int" in s:
        vals = [v for v in {lo, hi} if v is not None and v != unset] or [7]
        return vals
    if "JSON" in name:
        return ['["https://example.com/a.mp4"]']
    if name.endswith("_URL"):
        return ["https://example.com/a.bin"]
    return ["SENTINEL"]


def _inlet_meta(F, UV, mid, cname, value) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    F.inlet({"model": mid}, meta, {"valves": UV(**{cname: value})}, {})
    return (meta.get("openrouter_pipe") or {}).get("video_generation") or {}


@pytest.mark.parametrize("model", CATALOG, ids=lambda m: m["id"])
@pytest.mark.asyncio
async def test_every_video_control_reaches_the_wire(model):
    mid = model["id"]
    spec = build_video_filter_spec(mid, model)
    _N[0] += 1
    mod = _load_filter_from_source(
        render_video_filter_source(model_id=mid, video_model=model), f"rtv{_N[0]}"
    )
    F = mod.Filter()
    UV = mod.Filter.UserValves
    slugs = {mid: {"providers": ["acme"]}}
    adapter = VideoGenerationAdapter(pipe=cast(Any, _Pipe(slugs)), logger=cast(Any, _Logger()))
    controls = [n for n in UV.model_fields if n.startswith("VIDEO_")
                and not n.startswith("VIDEO_INTENT")
                and n not in ("VIDEO_PROVIDER_OPTIONS_JSON", "VIDEO_FRAME_MODE")]
    problems = []
    for cname in controls:
        unset = UV.model_fields[cname].default
        left_alone = dict(_inlet_meta(F, UV, mid, cname, unset).get("params") or {})
        if left_alone:
            problems.append(
                f"{cname}={unset!r} is the declared default and must send nothing; "
                f"the inlet wrote {left_alone}")
        for value in _choices(UV, cname):
            vm = _inlet_meta(F, UV, mid, cname, value)
            params = dict(vm.get("params") or {})
            if not params:
                problems.append(f"{cname}={value!r}: inlet wrote NO params")
                continue
            withheld: list[tuple[str, str]] = []
            payload = await adapter._build_payload(
                api_model_id=mid, prompt="p", video_meta=vm, video_model=model,
                frame_images=[], provider_options={}, provider_block={},
                input_references=None, withheld=withheld,
            )
            opts = (payload.get("provider") or {}).get("options") or {}
            flat = dict(payload)
            for _slug, o in opts.items():
                for k, v in (o or {}).items():
                    flat.setdefault(k, v)
            for k, v in params.items():
                landed = sorted(name for name in _spellings(model, k) if name in flat)
                if not landed:
                    problems.append(
                        f"{cname}={value!r}: {k!r}={v!r} NOT ON WIRE under any spelling "
                        f"{sorted(_spellings(model, k))}. withheld={withheld} payload={payload}")
                    continue
                wrong = {name: flat[name] for name in landed if flat[name] != v}
                if wrong:
                    problems.append(
                        f"{cname}={value!r}: {k!r} {v!r} -> {wrong}. withheld={withheld}")
    assert not problems, "\n".join(problems)
