from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..core.config import _OPENROUTER_ROUTER_FILTER_MARKER
from ..core.utils import (
    _ADAPTER_CACHE,
    _KEEP_WHAT_STILL_FITS,
    _PRIORITY_FIELD,
    OWUI_FUNCTION_ID_ILLEGAL_RE,
    scrub_surrogates,
)
from ..integrations.router_catalog import (
    ROUTER_KIND_ENUM,
    ROUTER_KIND_FLOAT,
    ROUTER_KIND_INT,
    ROUTER_KIND_LIST,
    ROUTER_KIND_TOGGLE,
    ROUTER_REWRITE_HAZARD_RE,
    ROUTER_USER_FLAG_PREFIX,
    RouterOption,
)

ROUTER_FILTER_ID_PREFIX = "openrouter_router_"
ROUTER_FILTER_NAME_PREFIX = "Router settings: "

_ROUTER_ADMIN_SENTENCE = "Empty sends nothing for this setting."
_ROUTER_USER_SENTENCE = (
    "Used only where the administrator lets users set it; empty keeps the administrator's value."
)
_ROUTER_LIST_SENTENCE = "Separate entries with commas."

_ROUTER_PLACEHOLDER_RE = re.compile(r"__[A-Z_]+__")

_ROUTER_BOUND_ARGUMENTS = (
    ("minimum", "ge"),
    ("maximum", "le"),
    ("exclusive_minimum", "gt"),
    ("exclusive_maximum", "lt"),
)


@dataclass(frozen=True, slots=True)
class RouterFilterSpec:
    model_id: str
    plugin_id: str
    display_name: str
    function_id: str
    marker: str
    dotted_id: str
    options: tuple[RouterOption, ...]


def sanitize_router_filter_id(model_id: str) -> str:
    raw = model_id.strip()
    cleaned = OWUI_FUNCTION_ID_ILLEGAL_RE.sub("_", raw).lower()
    if not cleaned:
        cleaned = "model"
    if len(cleaned) > 54:
        suffix = hashlib.sha1(scrub_surrogates(model_id).encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:45].rstrip('_')}_{suffix}"
    return f"{ROUTER_FILTER_ID_PREFIX}{cleaned}"


def router_filter_dotted_id(model_id: str) -> str:
    return model_id.strip().replace("/", ".").casefold()


def router_filter_display_name(model_name: str, model_id: str) -> str:
    name = model_name.strip() or model_id.strip()
    if ":" in name:
        name = name.split(":", 1)[1].strip() or name
    return name


def build_router_filter_spec(
    model_id: str,
    plugin_id: str,
    model_name: str,
    options: tuple[RouterOption, ...],
) -> RouterFilterSpec:
    function_id = sanitize_router_filter_id(model_id)
    return RouterFilterSpec(
        model_id=model_id,
        plugin_id=plugin_id,
        display_name=router_filter_display_name(model_name, model_id),
        function_id=function_id,
        marker=f"{_OPENROUTER_ROUTER_FILTER_MARKER}:{function_id}",
        dotted_id=router_filter_dotted_id(model_id),
        options=options,
    )


def _router_text_literal(text: str) -> str:
    clean = scrub_surrogates(text)
    return repr(ROUTER_REWRITE_HAZARD_RE.sub(lambda hit: f"from {hit.group(1)}", clean))


def _router_annotation(option: RouterOption) -> tuple[str, str, str]:
    if option.kind == ROUTER_KIND_ENUM:
        choices = ", ".join(repr(value) for value in ("", *option.choices))
        return f"Literal[{choices}]", "''", ""
    if option.kind == ROUTER_KIND_TOGGLE:
        return "Literal['', 'on', 'off']", "''", ""
    if option.kind in (ROUTER_KIND_INT, ROUTER_KIND_FLOAT):
        bounds = "".join(
            f"            {argument}={getattr(option, attribute)!r},\n"
            for attribute, argument in _ROUTER_BOUND_ARGUMENTS
            if getattr(option, attribute) is not None
        )
        if option.kind == ROUTER_KIND_FLOAT:
            bounds += "            allow_inf_nan=False,\n"
            return "float | None", "None", bounds
        return "int | None", "None", bounds
    return "str", "''", ""


def _router_field_block(name: str, annotation: str, default: str, title: str, description: str, extra: str = "") -> str:
    return (
        f"        {name}: {annotation} = Field(\n"
        f"            default={default},\n"
        f"{extra}"
        f"            title={_router_text_literal(title)},\n"
        f"            description={_router_text_literal(description)},\n"
        "        )"
    )


def _router_sentence_join(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _router_admin_description(option: RouterOption) -> str:
    return _router_sentence_join(_router_vendor_description(option), _ROUTER_ADMIN_SENTENCE)


def _router_user_description(option: RouterOption) -> str:
    return _router_sentence_join(_router_vendor_description(option), _ROUTER_USER_SENTENCE)


def _router_vendor_description(option: RouterOption) -> str:
    text = option.description.strip()
    if option.kind == ROUTER_KIND_LIST and "comma" not in text.lower():
        return _router_sentence_join(text, _ROUTER_LIST_SENTENCE)
    return text


def _router_option_fields(options: tuple[RouterOption, ...], *, admin: bool) -> str:
    blocks: list[str] = []
    for option in options:
        annotation, default, extra = _router_annotation(option)
        blocks.append(
            _router_field_block(
                option.field,
                annotation,
                default,
                option.title,
                _router_admin_description(option) if admin else _router_user_description(option),
                extra,
            )
        )
        if admin:
            lowered = option.title.lower()
            blocks.append(
                _router_field_block(
                    f"{ROUTER_USER_FLAG_PREFIX}{option.field}",
                    "bool",
                    "False",
                    f"Users may set: {lowered}",
                    (
                        f"On lets each user set {lowered} for their own chats; their value "
                        "then replaces the one above. Off keeps the value above for everyone."
                    ),
                )
            )
    return "\n".join(blocks)


def _router_options_table(options: tuple[RouterOption, ...]) -> str:
    return "\n".join(
        f"    ({option.name!r}, {option.field!r}, {option.kind!r}, {option.send_as!r}),"
        for option in options
    )


_ROUTER_FILTER_TEMPLATE = '''"""OpenRouter router settings companion filter."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

__ADAPTER_CACHE__


OWUI_OPENROUTER_PIPE_MARKER = __MARKER__
OPENROUTER_PIPE_VERSION = __VERSION__
ROUTER_MODEL_ID = __MODEL_ID__
ROUTER_MODEL_DOTTED = __DOTTED_ID__
ROUTER_PLUGIN_ID = __PLUGIN_ID__
ROUTER_USER_FLAG_PREFIX = __FLAG_PREFIX__
ROUTER_OPTIONS = (
__OPTIONS_TABLE__
)


def _matches_model(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    base = raw.strip().split(":", 1)[0].split("@", 1)[0]
    normalised = base.replace("/", ".").casefold()
    return normalised == ROUTER_MODEL_DOTTED or normalised.endswith("." + ROUTER_MODEL_DOTTED)


def _chosen(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "list":
        items: list[str] = []
        for part in str(value).split(","):
            entry = part.strip()
            if entry and entry not in items:
                items.append(entry)
        return items or None
    if kind in ("enum", "text"):
        return str(value).strip() or None
    if kind == "toggle":
        return {"on": True, "off": False}.get(value)
    if kind in ("int", "float") and not isinstance(value, bool):
        return value
    return None


class Filter:
    toggle = False

    class Valves(BaseModel):
__KEEP_WHAT_STILL_FITS__

__PRIORITY_FIELD__
__ADMIN_FIELDS__

    class UserValves(BaseModel):
__KEEP_WHAT_STILL_FITS__

__USER_FIELDS__

    def __init__(self) -> None:
        self.toggle = False
        self.valves = self.Valves()
        self.log = logging.getLogger("openrouter.router.settings")

    def _user_settings(self, user: Any) -> Any:
        raw = user.get("valves") if isinstance(user, dict) else None
        if isinstance(raw, self.UserValves):
            return raw
        try:
            if isinstance(raw, BaseModel):
                return self.UserValves(**raw.model_dump(exclude_unset=True))
            if isinstance(raw, dict):
                return self.UserValves(**raw)
        except Exception:
            self.log.debug("Router settings: the user's stored values could not be read", exc_info=True)
        return self.UserValves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        try:
            if not isinstance(body, dict) or not _matches_model(body.get("model")):
                return body
            user_settings = self._user_settings(__user__)
            settings: dict[str, Any] = {}
            outside: dict[str, Any] = {}
            for name, field, kind, send_as in ROUTER_OPTIONS:
                value = None
                if getattr(self.valves, ROUTER_USER_FLAG_PREFIX + field, False):
                    value = _chosen(getattr(user_settings, field, None), kind)
                if value is None:
                    value = _chosen(getattr(self.valves, field, None), kind)
                if value is None:
                    continue
                if send_as:
                    outside[send_as] = ",".join(value) if isinstance(value, list) else value
                else:
                    settings[name] = value
            body.update(outside)
            if settings:
                plugins = body.get("plugins")
                plugins = list(plugins) if isinstance(plugins, list) else []
                at = next(
                    (
                        index
                        for index, plugin in enumerate(plugins)
                        if isinstance(plugin, dict) and plugin.get("id") == ROUTER_PLUGIN_ID
                    ),
                    None,
                )
                entry = dict(plugins[at]) if at is not None else {"id": ROUTER_PLUGIN_ID}
                entry.update(settings)
                if at is None:
                    plugins.append(entry)
                else:
                    plugins[at] = entry
                body["plugins"] = plugins
            return body
        except Exception:
            self.log.exception("Router settings failed; the request goes out unchanged")
            return body
'''


def render_router_filter_source(spec: RouterFilterSpec) -> str:
    from open_webui_openrouter_pipe import __version__

    pieces = {
        "__ADAPTER_CACHE__": _ADAPTER_CACHE,
        "__MARKER__": repr(spec.marker),
        "__VERSION__": repr(__version__),
        "__MODEL_ID__": repr(spec.model_id),
        "__DOTTED_ID__": repr(spec.dotted_id),
        "__PLUGIN_ID__": repr(spec.plugin_id),
        "__FLAG_PREFIX__": repr(ROUTER_USER_FLAG_PREFIX),
        "__OPTIONS_TABLE__": _router_options_table(spec.options),
        "__KEEP_WHAT_STILL_FITS__": _KEEP_WHAT_STILL_FITS,
        "__PRIORITY_FIELD__": _PRIORITY_FIELD,
        "__ADMIN_FIELDS__": _router_option_fields(spec.options, admin=True),
        "__USER_FIELDS__": _router_option_fields(spec.options, admin=False),
    }
    return _ROUTER_PLACEHOLDER_RE.sub(lambda hit: pieces[hit.group(0)], _ROUTER_FILTER_TEMPLATE)
