from __future__ import annotations

from typing import Any

from ..models.registry import OpenRouterModelRegistry, referenced_preset_slug

REASONING_FIELDS = frozenset(
    {"reasoning", "include_reasoning", "reasoning_effort", "thinking_config"}
)
OUTPUT_CAP_FIELDS = frozenset({"max_output_tokens", "max_tokens", "max_completion_tokens"})
VERBOSITY_FIELDS = frozenset({"verbosity", "text"})
PLUGIN_FIELDS = frozenset({"plugins"})
TRUNCATION_FIELDS = frozenset({"truncation"})
MODALITY_FIELDS = frozenset({"modalities"})


def _plugin_id(entry: Any) -> str | None:
    entry_id = entry.get("id") if isinstance(entry, dict) else None
    return entry_id if isinstance(entry_id, str) else None


def _plugins_over(saved: list[Any], requested: Any, disabled: frozenset[str]) -> list[Any]:
    asked = list(requested) if isinstance(requested, list) else []
    asked_by_id: dict[str, Any] = {}
    for entry in asked:
        entry_id = _plugin_id(entry)
        if entry_id is not None:
            asked_by_id[entry_id] = entry
    merged: list[Any] = []
    placed: set[str] = set()
    for entry in saved:
        entry_id = _plugin_id(entry)
        if entry_id is not None and entry_id in asked_by_id:
            merged.append(asked_by_id[entry_id])
            placed.add(entry_id)
        else:
            merged.append(entry)
    for entry in asked:
        if _plugin_id(entry) not in placed:
            merged.append(entry)
    result: list[Any] = []
    for entry in merged:
        entry_id = _plugin_id(entry)
        if entry_id is not None and entry_id in disabled:
            result.append({"id": entry_id, "enabled": False})
        else:
            result.append(entry)
    return result


class PresetSettings:
    __slots__ = ("carried", "saved")

    def __init__(self, saved: dict[str, Any] | None = None) -> None:
        self.saved = saved
        self.carried: frozenset[str] = frozenset()

    @classmethod
    def referenced_by(cls, *references: Any) -> PresetSettings:
        slug = referenced_preset_slug(*references)
        return cls(OpenRouterModelRegistry.preset_settings(slug) if slug else None)

    def note_carried(self, body: Any) -> None:
        if self.saved is None:
            return
        names = set(type(body).model_fields) | set(body.model_extra or ())
        self.carried = frozenset(
            name for name in names if getattr(body, name, None) is not None
        )

    def saves(self, fields: frozenset[str]) -> bool:
        return self.saved is not None and not fields.isdisjoint(self.saved)

    def keeps(self, fields: frozenset[str]) -> bool:
        return self.saves(fields) and self.carried.isdisjoint(fields)

    def merge_under(self, body: Any, *, disabled_plugins: frozenset[str]) -> None:
        saved = self.saved
        if saved is None:
            return
        saved_provider = saved.get("provider")
        if (
            isinstance(saved_provider, dict)
            and saved_provider
            and isinstance(body.provider, dict)
            and body.provider
        ):
            body.provider = {**saved_provider, **body.provider}
        saved_plugins = saved.get("plugins")
        if isinstance(saved_plugins, list) and saved_plugins:
            merged = _plugins_over(saved_plugins, body.plugins, disabled_plugins)
            if body.plugins or merged != saved_plugins:
                body.plugins = merged
