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
from dataclasses import dataclass, replace
from typing import Any

from ..core.config import (
    _OPENROUTER_IMAGE_FILTER_MARKER,
    _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL,
    _OPENROUTER_IMAGE_GEN_FILTER_MARKER,
    _OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID,
    _PIPE_METADATA_KEY,
)
from ..core.utils import OWUI_FUNCTION_ID_ILLEGAL_RE as _IMAGE_FILTER_ID_RE
from ..integrations.image_types import (
    PASSTHROUGH_DESCRIPTION,
    PASSTHROUGH_ENUMS,
    RENDERABLE_FIELD_NAME_RE,
    SCHEMA_ONLY_PARAMS,
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
    contract_read: bool = False
    published_anything: bool = False
    """Whether any record published a renderable setting, before agreement was applied.

    A knobless spec has two causes that read very differently: the model offers nothing,
    or its providers publish different things and nothing survives the intersection.
    """
    enums: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    narrowed: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    schema_only: tuple[str, ...] = ()
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
            union: list[Any] = list(descriptor["values"])
            for other in others:
                for value in other.get("values") or []:  # type: ignore[union-attr]
                    if value not in union:
                        union.append(value)
            shared = [
                value
                for value in descriptor["values"]
                if all(value in (other.get("values") or []) for other in others)  # type: ignore[union-attr]
            ]
            if shared:
                agreed[name] = {
                    "type": "enum",
                    "values": shared,
                    "narrowed": [value for value in union if value not in shared],
                }
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
    narrowed: list[tuple[str, tuple[Any, ...]]] = []
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
                extra = _descriptor_enum({"values": descriptor.get("narrowed")})
                if extra:
                    narrowed.append((name, extra))
        elif kind == "range":
            low = _descriptor_bound(descriptor, "min")
            high = _descriptor_bound(descriptor, "max")
            if low is not None and high is not None and high > low:
                ranges.append((name, low, high))
        elif kind == "boolean":
            supported_names.append(name)

    typed = {name for name, _ in enums} | {name for name, _, _ in ranges} | set(supported_names)
    schema_only = tuple(name for name in SCHEMA_ONLY_PARAMS if name not in typed)

    taken = set(ALWAYS_ON_VALVE_NAMES) | {
        _valve_name(name)
        for name in (
            *(n for n, _ in enums),
            *(n for n, _, _ in ranges),
            *supported_names,
            *schema_only,
        )
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
        contract_read=endpoint_record is not None,
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
        narrowed=tuple(narrowed),
        schema_only=schema_only,
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


_SCHEMA_ONLY_CAVEAT = (
    "This model publishes no list of what it accepts here, so the value goes out as "
    "typed and the company running it decides. Empty leaves it unset."
)

ALWAYS_ON_CONTROLS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "IMAGE_PROVIDER_OPTIONS_JSON",
        "str",
        '""',
        "Provider options",
        (
            "Extra settings for the company that runs this model, as "
            "a JSON object keyed by its OpenRouter name. Use it for anything this panel does "
            "not already offer. Empty sends nothing."
        ),
    ),
    (
        "IMAGE_REFERENCE_MODE",
        'Literal["auto", "latest-only", "none"]',
        '"auto"',
        "Reference images",
        (
            "Which attached images go to the model as references. "
            "auto sends every one on this turn, oldest first; latest-only sends just the most "
            "recent; none sends none of them."
        ),
    ),
    (
        "IMAGE_REFERENCE_URLS",
        "str",
        '""',
        "Reference image links",
        (
            "Reference images to use as well as, or instead of, the "
            "attached ones: a JSON list of https links or data URLs. These are placed first, "
            "so they survive when the model takes fewer references than are on offer."
        ),
    ),
)

_ALWAYS_ON_VALVES = tuple(
    f"{name}: {annotation} = Field(\n"
    f"            default={default},\n"
    f'            title="{title}",\n'
    f'            description="{description}",\n'
    "        )"
    for name, annotation, default, title, description in ALWAYS_ON_CONTROLS
)

ALWAYS_ON_VALVE_NAMES = frozenset(name for name, *_rest in ALWAYS_ON_CONTROLS)


def _image_shared_by_some(values: tuple[Any, ...]) -> str:
    listed = ", ".join(str(value) for value in values)
    return (
        f"Only some of the companies serving this model accept {listed}; if another one "
        "takes the request you are told it was not sent."
    )


def _render_image_always_on_valves() -> str:
    return "\n".join(_image_field(block) for block in _ALWAYS_ON_VALVES)


def _render_image_model_user_valves(spec: ImageModelFilterSpec) -> str:
    """Render only the knobs this model publishes, each with its published values.

    An empty contract renders ``pass`` rather than an empty class body, which is not valid
    Python. Callers decide separately whether a knobless filter is worth installing.
    """
    fields: list[str] = []
    extra = dict(spec.narrowed)
    for name, values in spec.enums:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        also = extra.get(name, ())
        literals = _image_literal_union(("", *values, *also))
        caveat = (
            f" {_image_shared_by_some(also)}"
            if also
            else ""
        )
        fields.append(
            _image_field(
                f"{_valve_name(name)}: Literal[{literals}] = Field(\n"
                '            default="",\n'
                f'            title="{title}",\n'
                f'            description="{description} Empty uses the model default.{caveat}",\n'
                "        )"
            )
        )
    for name in spec.schema_only:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        fields.append(
            _image_field(
                f"{_valve_name(name)}: str = Field(\n"
                '            default="",\n'
                f'            title="{title}",\n'
                f'            description="{description} {_SCHEMA_ONLY_CAVEAT}",\n'
                "        )"
            )
        )
    for name, low, high in spec.ranges:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        fields.append(
            _image_field(
                f"{_valve_name(name)}: int | None = Field(\n"
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
                f"{_valve_name(name)}: int | None = Field(\n"
                "            default=None,\n"
                f'            title="{title}",\n'
                f'            description="{description} Leave it empty to use the model '
                'default.",\n'
                "        )"
            )
        )
    for name in spec.passthrough:
        published = PASSTHROUGH_ENUMS.get(name)
        if published is not None:
            values, meaning = published
            literals = _image_literal_union(("", *values))
            fields.append(
                _image_field(
                    f"{_valve_name(name)}: Literal[{literals}] = Field(\n"
                    '            default="",\n'
                    f"            title={name!r},\n"
                    f'            description="{meaning} Empty leaves it unset.",\n'
                    "        )"
                )
            )
            continue
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


def _render_image_overrides(
    spec: ImageModelFilterSpec,
    *,
    target: str = "overrides",
    wire_keys: dict[str, str] | None = None,
) -> str:
    """Read each rendered valve back out into the request, under its published name."""
    renamed = wire_keys or {}

    def _key(name: str) -> str:
        return renamed.get(name, name)

    lines: list[str] = []
    for name, _values in spec.enums:
        lines.append(f"        value = user_valves.{_valve_name(name)}")
        lines.append('        if value != "":')
        lines.append(f"            {target}[{_key(name)!r}] = value")
    for name, _low, _high in spec.ranges:
        lines.append(f"        count = user_valves.{_valve_name(name)}")
        lines.append("        if count is not None:")
        lines.append(f"            {target}[{_key(name)!r}] = int(count)")
    for name in spec.schema_only:
        lines.append(f'        wanted = (user_valves.{_valve_name(name)} or "").strip()')
        lines.append("        if wanted:")
        lines.append(f"            {target}[{_key(name)!r}] = wanted")
    for name in spec.supported:
        lines.append(f"        chosen = user_valves.{_valve_name(name)}")
        lines.append("        if chosen is not None:")
        lines.append(f"            {target}[{_key(name)!r}] = int(chosen)")
    for name in spec.passthrough:
        lines.append(f'        raw = (user_valves.{_valve_name(name)} or "").strip()')
        lines.append("        if raw:")
        lines.append(f"            {target}[{_key(name)!r}] = self._decode(raw, {name!r})")
    return "\n".join(lines) if lines else "        pass"


_KEEP_WHAT_STILL_FITS = '''        @model_validator(mode="before")
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
            kept = {}
            for name, field in cls.model_fields.items():
                if name not in data:
                    continue
                annotated = (
                    Annotated[(field.annotation, *field.metadata)]
                    if field.metadata
                    else field.annotation
                )
                try:
                    TypeAdapter(annotated).validate_python(data[name])
                except ValidationError:
                    continue
                kept[name] = data[name]
            return kept'''


def render_image_model_filter_source(spec: ImageModelFilterSpec) -> str:
    """Render one model's filter, offering exactly the knobs its contract publishes."""
    from open_webui_openrouter_pipe import __version__

    return f'''"""OpenRouter image companion filter."""

from __future__ import annotations

import json
import logging
import math
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover - OWUI runtime only
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = {spec.marker!r}
OPENROUTER_PIPE_VERSION = {__version__!r}
IMAGE_FILTER_MODEL_ID = {spec.model_id!r}
IMAGE_FILTER_MODEL_DOTTED = {spec.dotted_id!r}
PIPE_METADATA_KEY = {_PIPE_METADATA_KEY!r}


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


def _json_number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{{text}} is out of range for JSON")
    return value


def _json_constant(literal: str) -> float:
    raise ValueError(f"{{literal}} is not valid JSON")


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    class UserValves(BaseModel):
{_KEEP_WHAT_STILL_FITS}

{_render_image_always_on_valves()}
{_render_image_model_user_valves(spec)}

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _json_object(raw: Any, field: str) -> dict:
        """A JSON object keyed by provider name, or nothing.

        Rejected here rather than on the wire, so the message can name the setting the
        user typed instead of an upstream complaint about a request they never saw.
        """
        if not isinstance(raw, str) or not raw.strip():
            return {{}}
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ImageFilterInputError(f"{{field}} is not valid JSON: {{exc}}") from exc
        if not isinstance(parsed, dict):
            raise ImageFilterInputError(
                f"{{field}} must be a JSON object keyed by provider name."
            )
        return {{
            name.strip(): dict(payload)
            for name, payload in parsed.items()
            if isinstance(name, str) and name.strip() and isinstance(payload, dict)
        }}

    @staticmethod
    def _json_array(raw: Any, field: str) -> list:
        """A JSON list of links, given either as strings or as objects carrying one."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ImageFilterInputError(f"{{field}} is not valid JSON: {{exc}}") from exc
        if not isinstance(parsed, list):
            raise ImageFilterInputError(f"{{field}} must be a JSON list.")
        links = []
        for item in parsed:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url.strip():
                links.append(url.strip())
        return links

    @staticmethod
    def _deep_merge_pipe_provider(existing: Any, options: dict) -> dict:
        """Add these options to whatever another filter already asked for.

        Provider routing writes the same place. Replacing the block outright would drop
        an operator's routing choice whenever a user typed one option.
        """
        merged = dict(existing) if isinstance(existing, dict) else {{}}
        current = merged.get("options")
        merged_options = dict(current) if isinstance(current, dict) else {{}}
        for name, payload in options.items():
            merged_options[name] = payload
        if merged_options:
            merged["options"] = merged_options
        return merged

    @staticmethod
    def _decode(raw: str, field: str) -> Any:
        """Parse a passthrough value without inventing a type the user did not write.

        A JSON container is parsed, and so is a bare number, because 8 of the 17 published
        passthrough names take numbers and a quoted one is a different request. Everything
        else stays the string it is -- ``style`` really does take a bare word like
        ``realistic_image``, and ``null``, ``true`` and ``2K`` are values in their own
        right here rather than spellings of something else.

        ``NaN``, ``Infinity`` and ``1e400`` are refused instead of parsed. No JSON encoder
        can put the float they produce on the wire, so accepting one only moves the failure
        to a place where the message names something else entirely. Alone they stay the
        string that was typed; inside a container they raise here, where the field can be
        named.
        """
        container = raw[:1] in ("[", "{{")
        try:
            parsed = json.loads(raw, parse_float=_json_number, parse_constant=_json_constant)
        except ValueError as exc:
            if container:
                raise ImageFilterInputError(f"{{field}} is not valid JSON: {{exc}}") from exc
            return raw
        if container:
            return parsed
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            return raw
        return parsed

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

        provider_options = self._json_object(
            getattr(user_valves, "IMAGE_PROVIDER_OPTIONS_JSON", ""), "Provider options"
        )
        reference_mode = getattr(user_valves, "IMAGE_REFERENCE_MODE", "auto")
        reference_links = self._json_array(
            getattr(user_valves, "IMAGE_REFERENCE_URLS", ""), "Reference image links"
        )
        if isinstance(__metadata__, dict) and (
            provider_options or reference_links or reference_mode != "auto"
        ):
            previous = __metadata__.get(PIPE_METADATA_KEY)
            pipe_meta = dict(previous) if isinstance(previous, dict) else {{}}
            __metadata__[PIPE_METADATA_KEY] = pipe_meta
            if provider_options:
                pipe_meta["provider"] = self._deep_merge_pipe_provider(
                    pipe_meta.get("provider"), provider_options
                )
            if reference_links or reference_mode != "auto":
                previous_images = pipe_meta.get("image_generation")
                image_meta = (
                    dict(previous_images) if isinstance(previous_images, dict) else {{}}
                )
                image_meta["reference_mode"] = reference_mode
                if reference_links:
                    image_meta["reference_urls"] = reference_links
                pipe_meta["image_generation"] = image_meta
        return body
'''


IMAGE_GEN_TOOL_TIER_KEY_CANDIDATES: tuple[str, ...] = ("resolution", "size", "image_size")

IMAGE_GEN_TOOL_TIER_KEY: str = IMAGE_GEN_TOOL_TIER_KEY_CANDIDATES[0]


def image_gen_tool_wire_keys() -> dict[str, str]:
    return {"resolution": IMAGE_GEN_TOOL_TIER_KEY}


def image_gen_model_note(spec: ImageModelFilterSpec, *, catalog_match: bool) -> str:
    named = spec.model_id or "no model"
    opening = "Which OpenRouter model draws the picture."
    if not catalog_match:
        return (
            f"{opening} No settings are offered for {named}: it is not in the image "
            "model list this pipe has loaded. Check the id if that is unexpected."
        )
    if not spec.contract_read:
        return (
            f"{opening} What {named} accepts could not be read this time, so no settings "
            "are offered; they appear once it can be read again."
        )
    if not spec.has_knobs:
        if spec.published_anything:
            return (
                f"{opening} The companies serving {named} accept different settings, so "
                "none can be offered without knowing which one will take the request. It "
                "draws with its own defaults."
            )
        return (
            f"{opening} {named} publishes no adjustable settings, so it draws with its "
            "own defaults."
        )
    return (
        f"{opening} The settings offered to users are the ones {named} publishes; "
        "choosing another model changes them."
    )


def render_image_gen_filter_source(
    spec: ImageModelFilterSpec,
    *,
    catalog_match: bool,
    selected_model: str = "",
) -> str:
    tool_spec = replace(spec, passthrough=())
    moderation_values, moderation_meaning = PASSTHROUGH_ENUMS["moderation"]
    model_id = scrub_surrogates(
        selected_model.strip() or spec.model_id or _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL
    )
    return f'''"""
title: OR Image Gen
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: {_OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID}
description: Configures OpenRouter image generation for the OpenRouter pipe.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = {_OPENROUTER_IMAGE_GEN_FILTER_MARKER!r}


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        IMAGE_GENERATION_MODEL: str = Field(
            default={model_id!r},
            title="Image generation model",
            description={image_gen_model_note(spec, catalog_match=catalog_match)!r},
        )
        IMAGE_GENERATION_MODERATION: Literal[{_image_literal_union(moderation_values)}] = Field(
            default={moderation_values[0]!r},
            title="Image moderation",
            description={moderation_meaning!r},
        )

    class UserValves(BaseModel):
{_KEEP_WHAT_STILL_FITS}

{_render_image_model_user_valves(tool_spec)}

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.gen")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body

        user_valves = None
        if isinstance(__user__, dict):
            stored = __user__.get("valves")
            if isinstance(stored, self.UserValves):
                user_valves = stored
            elif stored is not None:
                try:
                    user_valves = self.UserValves.model_validate(
                        stored if isinstance(stored, dict) else stored.model_dump()
                    )
                except Exception:  # noqa: BLE001 - a stored valve must not block the turn
                    user_valves = self.UserValves()
        if user_valves is None:
            user_valves = self.UserValves()

        params: dict[str, Any] = {{"model": self.valves.IMAGE_GENERATION_MODEL}}
        if self.valves.IMAGE_GENERATION_MODERATION != {moderation_values[0]!r}:
            params["moderation"] = self.valves.IMAGE_GENERATION_MODERATION
{_render_image_overrides(tool_spec, target="params", wire_keys=image_gen_tool_wire_keys())}

        if isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get({_PIPE_METADATA_KEY!r})
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {{}}
            __metadata__[{_PIPE_METADATA_KEY!r}] = pipe_meta

            prev_tools = pipe_meta.get("server_tools")
            server_tools = dict(prev_tools) if isinstance(prev_tools, dict) else {{}}
            pipe_meta["server_tools"] = server_tools
            server_tools["image_generation"] = params

        return body
'''
