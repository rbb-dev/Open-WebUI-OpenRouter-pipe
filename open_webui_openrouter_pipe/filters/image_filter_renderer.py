"""Renders one Open WebUI filter per image model.

Each filter offers exactly the knobs that model's endpoint record publishes -- its
``supported_parameters`` become typed fields, its ``allowed_passthrough_parameters``
become free-text ones. The filter writes every chosen value flat into
``body["image_config"]``; the image adapter is what decides, from the same record,
which of those go at the top level of the request and which belong under
``provider.options.<slug>``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from ..core.config import _OPENROUTER_IMAGE_FILTER_MARKER
from ..core.utils import OWUI_FUNCTION_ID_ILLEGAL_RE as _IMAGE_FILTER_ID_RE
from ..integrations.image_types import (
    PASSTHROUGH_DESCRIPTION,
    RENDERABLE_FIELD_NAME_RE,
    TOP_LEVEL_PARAMS,
    scrub_surrogates,
)
from ..models.registry import sanitize_model_id


def sanitize_image_filter_id(model_id: str) -> str:
    """Derive one model's filter id, on the same terms as the video sibling.

    The thresholds match ``sanitize_video_filter_id`` because both now take a model id:
    the tighter 30/21 pair hashed away the readable part of most ids, and that string is
    what an operator has to recognise in Open WebUI's function list. The empty-input
    fallback is ``model`` rather than ``generic``, which was a retired filter's id.
    """
    raw = (model_id or "").strip().lower()
    cleaned = _IMAGE_FILTER_ID_RE.sub("_", raw)
    if not cleaned:
        cleaned = "model"
    if len(cleaned) > 54:
        suffix = hashlib.sha1(scrub_surrogates(model_id or "").encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:45].rstrip('_')}_{suffix}"
    return f"openrouter_image_filter_{cleaned}"


"""A passthrough name may become a Python identifier, so it must look like one.

OpenRouter picks these names. One containing a hyphen -- ``cfg-scale`` is a real
provider parameter -- renders ``IMAGE_CFG-SCALE:`` and the whole module stops parsing,
which costs the model every other knob it publishes, not just the bad one.
"""

_NOT_A_KNOB = frozenset({"input_references"})
"""Published parameters that are never a user control.

``input_references`` counts the images the request carries; the adapter reads that limit
straight off the endpoint record and fills the list from what the user attached. Excluded
once, here, so it cannot be counted as a knob in one place and skipped in another.
"""


@dataclass(frozen=True, slots=True)
class ImageModelFilterSpec:
    """What one image model actually accepts, read from its published endpoint record.

    Every knob here came from the model's own contract. A knob the model does not
    publish has no field at all, which is the difference from the old fixed variants:
    those offered the same ten aspect ratios to every model, and 33 of 40 rejected at
    least one of them.
    """

    model_id: str
    display_name: str
    function_id: str
    marker: str
    dotted_id: str = ""
    published_anything: bool = False
    """Whether any record published a renderable setting, before agreement was applied.

    A knobless spec has two causes that read very differently: the model offers nothing,
    or its providers publish different things and nothing survives the intersection.
    """
    enums: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    ranges: tuple[tuple[str, int, int], ...] = ()
    supported: tuple[str, ...] = ()
    """Parameters the model declares it supports without publishing a domain.

    OpenRouter writes these as ``{"type": "boolean"}``, which their model schema words as
    "whether the model supports ..." -- a support flag, not a value. The request takes a
    number, so the control is a number with no published bounds.
    """
    passthrough: tuple[str, ...] = ()

    @property
    def knob_count(self) -> int:
        return len(self.enums) + len(self.ranges) + len(self.supported) + len(self.passthrough)

    @property
    def has_knobs(self) -> bool:
        return bool(self.enums or self.ranges or self.supported or self.passthrough)


def _descriptor_enum(descriptor: dict) -> tuple[Any, ...]:
    """The published values, as published.

    Stringifying them rendered a control whose every option the adapter then rejected,
    because it compares the chosen value against the contract's own list -- ``"512"`` is
    not ``512``.
    """
    values = descriptor.get("values")
    if not isinstance(values, list):
        return ()
    return tuple(
        v
        for v in values
        if isinstance(v, (str, int, float))
        and not isinstance(v, bool)
        and v != ""
        # `repr(nan)` is the bare name `nan`, not a literal, so it would render a field
        # annotation referring to an undefined name. JSON admits NaN and Infinity, so a
        # catalog response can carry them.
        and (not isinstance(v, float) or math.isfinite(v))
    )


def _descriptor_bound(descriptor: dict, key: str) -> int | None:
    """The published bound, or None when it is not a whole number.

    A bound the pipe cannot read is not a bound. Substituting a default produced a knob
    advertising "accepts 0 to 0" whose only selectable value was the one the override
    block refuses to send -- a control that looks live and can never do anything.
    """
    raw = descriptor.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if isinstance(raw, float) and raw != int(raw):
        return None
    return int(raw)


def _published_records(endpoint_record: Any) -> list[dict]:
    """Every record the model published, however the caller passed them."""
    if isinstance(endpoint_record, dict):
        return [endpoint_record]
    if isinstance(endpoint_record, list):
        return [item for item in endpoint_record if isinstance(item, dict)]
    return []


def _agreed_parameters(records: list[dict]) -> dict[str, dict]:
    """The descriptors every provider of this model accepts.

    A model served by several providers publishes one record each, and they disagree.
    The routing decision is made per request, long after the controls are rendered, so
    the only set that is right whichever provider serves is the one they all accept:
    enum values intersected, ranges narrowed to the tightest published bounds.
    """
    if not records:
        return {}
    per_record = []
    for record in records:
        supported = record.get("supported_parameters")
        per_record.append(supported if isinstance(supported, dict) else {})
    if not per_record[0]:
        return {}

    agreed: dict[str, dict] = {}
    for name, descriptor in per_record[0].items():
        if not isinstance(descriptor, dict):
            continue
        others = [d.get(name) for d in per_record[1:]]
        if any(not isinstance(other, dict) for other in others):
            continue
        kind = descriptor.get("type")
        if any(other.get("type") != kind for other in others):  # type: ignore[union-attr]
            continue
        if kind == "enum":
            if not isinstance(descriptor.get("values"), list):
                continue
            if any(not isinstance(other.get("values"), list) for other in others):  # type: ignore[union-attr]
                continue
            shared = [
                value
                for value in descriptor["values"]
                if all(value in (other.get("values") or []) for other in others)  # type: ignore[union-attr]
            ]
            if shared:
                agreed[name] = {"type": "enum", "values": shared}
        elif kind == "range":
            lows = [descriptor.get("min"), *(other.get("min") for other in others)]  # type: ignore[union-attr]
            highs = [descriptor.get("max"), *(other.get("max") for other in others)]  # type: ignore[union-attr]
            numeric = all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in (*lows, *highs)
            )
            if numeric:
                agreed[name] = {"type": "range", "min": max(lows), "max": min(highs)}  # type: ignore[type-var]
        else:
            agreed[name] = descriptor
    return agreed


def _agreed_passthrough(records: list[dict]) -> tuple[str, ...]:
    """Passthrough names every provider both names and can be addressed under.

    A record with no provider slug cannot carry a provider option at all -- the adapter
    drops the whole block -- so one such record means no passthrough control is offered.
    """
    if not records:
        return ()
    per_record: list[set[str]] = []
    for record in records:
        slug = record.get("provider_slug")
        if not isinstance(slug, str) or not slug.strip():
            return ()
        names = record.get("allowed_passthrough_parameters")
        if not isinstance(names, list):
            return ()
        per_record.append({name for name in names if isinstance(name, str) and name})
    shared = set.intersection(*per_record) if per_record else set()
    first = records[0].get("allowed_passthrough_parameters") or []
    return tuple(name for name in first if isinstance(name, str) and name in shared)


def build_image_model_filter_spec(
    model_id: str,
    image_model: dict | None = None,
    endpoint_record: dict | list[dict] | None = None,
) -> ImageModelFilterSpec:
    """Turn one model's published contract into the knobs its filter should render.

    ``endpoint_record`` is the model's entry from OpenRouter's per-model endpoint
    listing. When it is missing the spec carries no knobs rather than guessing: an
    invented ratio list is what the fixed variants did, and it is why a third of models
    were offered values they reject.
    """
    model = image_model if isinstance(image_model, dict) else {}
    canonical = (str(model.get("id") or model_id or "")).strip()
    display = str(model.get("name") or canonical).strip() or canonical

    records = _published_records(endpoint_record)
    declared = _agreed_parameters(records)

    enums: list[tuple[str, tuple[str, ...]]] = []
    ranges: list[tuple[str, int, int]] = []
    supported_names: list[str] = []
    for name in TOP_LEVEL_PARAMS:
        descriptor = declared.get(name)
        if not isinstance(descriptor, dict):
            continue
        kind = descriptor.get("type")
        if kind == "enum":
            values = _descriptor_enum(descriptor)
            if values:
                enums.append((name, values))
        elif kind == "range":
            low = _descriptor_bound(descriptor, "min")
            high = _descriptor_bound(descriptor, "max")
            if low is not None and high is not None and high > low:
                ranges.append((name, low, high))
        elif kind == "boolean":
            supported_names.append(name)

    taken = {
        _valve_name(name)
        for name in (*(n for n, _ in enums), *(n for n, _, _ in ranges), *supported_names)
    }
    # `taken` grows as names are accepted, so two published names differing only in case
    # cannot both render: they produce one field, and the second write would put a
    # parameter on the wire that the user was never shown a control for.
    accepted: list[str] = []
    for name in _agreed_passthrough(records):
        if (
            name in TOP_LEVEL_PARAMS
            or name in _NOT_A_KNOB
            or not RENDERABLE_FIELD_NAME_RE.fullmatch(name)
            or _valve_name(name) in taken
        ):
            continue
        taken.add(_valve_name(name))
        accepted.append(name)
    passthrough = tuple(accepted)

    return ImageModelFilterSpec(
        model_id=canonical,
        display_name=display,
        function_id=sanitize_image_filter_id(canonical),
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:{canonical}",
        dotted_id=sanitize_model_id(canonical.lstrip("~")).casefold(),
        published_anything=any(
            isinstance((record.get("supported_parameters") or {}), dict)
            and any(
                name in TOP_LEVEL_PARAMS and name not in _NOT_A_KNOB
                for name in (record.get("supported_parameters") or {})
            )
            or bool(record.get("allowed_passthrough_parameters"))
            for record in records
        ),
        enums=tuple(enums),
        ranges=tuple(ranges),
        supported=tuple(supported_names),
        passthrough=passthrough,
    )


_IMAGE_KNOB_TITLE_OVERRIDES = {
    "aspect_ratio": ("Aspect ratio", "Frame shape."),
    "resolution": ("Resolution", "Output size tier."),
    "n": ("Number of images", "How many images this request asks for."),
    "seed": (
        "Seed",
        (
            "This model supports seeding. OpenRouter publishes no range for it, and does "
            "not promise the same seed repeats an image."
        ),
    ),
    "background": ("Background", "Background treatment."),
    "quality": ("Quality", "Rendering quality tier."),
    "output_format": ("Output format", "Container the image comes back in."),
    "output_compression": ("Output compression", "Compression level, where the format allows one."),
    "size": ("Output size", "Exact pixel dimensions, where the model takes them rather than a tier."),
}

IMAGE_KNOB_TITLES = {
    name: _IMAGE_KNOB_TITLE_OVERRIDES.get(name, (name, "")) for name in TOP_LEVEL_PARAMS
}
"""A title for every parameter that can be rendered, derived from the one list.

Keyed off ``TOP_LEVEL_PARAMS`` rather than written out again, so a name added there
cannot arrive with no title and render a control labelled with its raw parameter name.
"""

assert set(_IMAGE_KNOB_TITLE_OVERRIDES) == set(TOP_LEVEL_PARAMS), (
    "every renderable parameter needs a title, or its control is labelled with its raw "
    "parameter name; and every title must name a parameter that can be rendered"
)


def _valve_name(published: str) -> str:
    """The field name a published parameter is rendered under.

    One producer, because the name decides three things that must agree: the field the
    chat UI draws, the attribute the override block reads back, and whether a later
    parameter would collide with it. Two names differing only in case render one field,
    and pydantic keeps the last -- so the comparison has to be on this value, not on the
    published spelling.
    """
    return f"IMAGE_{published.upper()}"


def _image_field(text: str) -> str:
    return "\n".join(f"        {line}" if line else "" for line in text.splitlines())


def _image_literal_union(values: tuple[Any, ...]) -> str:
    return ", ".join(repr(value) for value in values)


def _render_image_model_user_valves(spec: ImageModelFilterSpec) -> str:
    """Render only the knobs this model publishes, each with its published values.

    An empty contract renders ``pass`` rather than an empty class body, which is not valid
    Python. Callers decide separately whether a knobless filter is worth installing.
    """
    fields: list[str] = []
    for name, values in spec.enums:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        literals = _image_literal_union(("", *values))
        fields.append(
            _image_field(
                f"{_valve_name(name)}: Literal[{literals}] = Field(\n"
                '            default="",\n'
                f'            title="{title}",\n'
                f'            description="{description} Empty uses the model default.",\n'
                "        )"
            )
        )
    for name, low, high in spec.ranges:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        fields.append(
            _image_field(
                f"{_valve_name(name)}: Optional[int] = Field(\n"
                "            default=None,\n"
                f"            ge={low},\n"
                f"            le={high},\n"
                f'            title="{title}",\n'
                f'            description="{description} This model accepts {low} to {high}. '
                'Leave it empty to use the model default.",\n'
                "        )"
            )
        )
    for name in spec.supported:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        fields.append(
            _image_field(
                f"{_valve_name(name)}: Optional[int] = Field(\n"
                "            default=None,\n"
                f'            title="{title}",\n'
                f'            description="{description} Leave it empty to use the model '
                'default.",\n'
                "        )"
            )
        )
    for name in spec.passthrough:
        fields.append(
            _image_field(
                f"{_valve_name(name)}: str = Field(\n"
                '            default="",\n'
                f"            title={name!r},\n"
                f'            description="{PASSTHROUGH_DESCRIPTION}",\n'
                "        )"
            )
        )
    return "\n".join(fields) if fields else "        pass"


def _render_image_overrides(spec: ImageModelFilterSpec) -> str:
    """Read each rendered valve back out into the request, under its published name."""
    lines: list[str] = []
    for name, _values in spec.enums:
        lines.append(f"        value = user_valves.{_valve_name(name)}")
        lines.append('        if value != "":')
        lines.append(f"            overrides[{name!r}] = value")
    for name, _low, _high in spec.ranges:
        lines.append(f"        count = user_valves.{_valve_name(name)}")
        lines.append("        if count is not None:")
        lines.append(f"            overrides[{name!r}] = int(count)")
    for name in spec.supported:
        lines.append(f"        chosen = user_valves.{_valve_name(name)}")
        lines.append("        if chosen is not None:")
        lines.append(f"            overrides[{name!r}] = int(chosen)")
    for name in spec.passthrough:
        lines.append(f'        raw = (user_valves.{_valve_name(name)} or "").strip()')
        lines.append("        if raw:")
        lines.append(f"            overrides[{name!r}] = self._decode(raw, {name!r})")
    return "\n".join(lines) if lines else "        pass"


def render_image_model_filter_source(spec: ImageModelFilterSpec) -> str:
    """Render one model's filter, offering exactly the knobs its contract publishes."""
    return f'''"""OpenRouter image companion filter."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover - OWUI runtime only
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = {spec.marker!r}
IMAGE_FILTER_MODEL_ID = {spec.model_id!r}
IMAGE_FILTER_MODEL_DOTTED = {spec.dotted_id!r}


def _matches_model(raw: str) -> bool:
    # The pipe rewrites "/" to "." before Open WebUI ever sees a model id, and Open
    # WebUI prefixes its own function id, so the runtime body carries
    # "<function_id>.<vendor>.<model>" -- no slash. Normalise both sides to that dotted
    # form and require a "." boundary, so "recraft.recraft-v3" cannot be matched by
    # "notrecraft.recraft-v3". A leading "~" marks a catalog alias and is not part of
    # the identity.
    if not isinstance(raw, str) or not raw:
        return False
    normalised = raw.strip().lstrip("~").replace("/", ".").casefold()
    return normalised == IMAGE_FILTER_MODEL_DOTTED or normalised.endswith(
        "." + IMAGE_FILTER_MODEL_DOTTED
    )


class ImageFilterInputError(ValueError):
    """A value the user typed that this filter will not put on the wire."""


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    class UserValves(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def _keep_what_still_fits(cls, data: Any) -> Any:
            """Drop stored values the model no longer publishes, keep the rest.

            These fields track a live contract, so a provider joining the model can
            narrow a range or remove a ratio while a value the user chose earlier is
            still stored. Open WebUI builds this class from that stored dict and passes
            no valves at all if construction raises -- so one stale entry silently threw
            away every other choice the user had made.
            """
            if not isinstance(data, dict):
                return data
            kept = {{}}
            for name, field in cls.model_fields.items():
                if name not in data:
                    continue
                annotated = (
                    Annotated[tuple([field.annotation, *field.metadata])]
                    if field.metadata
                    else field.annotation
                )
                try:
                    TypeAdapter(annotated).validate_python(data[name])
                except ValidationError:
                    continue
                kept[name] = data[name]
            return kept

{_render_image_model_user_valves(spec)}

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _decode(raw: str, field: str) -> Any:
        """Parse a passthrough value only when the user wrote a JSON container.

        Anything else is passed through as the string it is -- ``style`` really does take
        a bare word like ``realistic_image``, and parsing it would turn ``null`` into
        None and ``123`` into an int. A container that does not parse raises here, where
        the message can name the field, rather than reaching the provider as a string
        that produces an error about something else.
        """
        if raw[:1] not in ("[", "{{"):
            return raw
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise ImageFilterInputError(f"{{field}} is not valid JSON: {{exc}}") from exc

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        if not _matches_model(body.get("model")):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}
{_render_image_overrides(spec)}
        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''
