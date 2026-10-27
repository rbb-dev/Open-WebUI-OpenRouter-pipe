from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from ..api.transforms import _get_disable_param
from ..core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_TITLE,
    _PIPE_METADATA_KEY,
    _REMOVE_DROPPED_MODELS_MAX,
    openrouter_attribution_headers,
)
from ..core.utils import _await_if_needed
from ..core.warn_latch import warn_level
from ..integrations.image_catalog import _wants_contracts
from ..storage.persistence import raw_valve_column_decodes
from .catalog_manager import _params_as_mapping, syncs_owui_models
from .registry import OpenRouterModelRegistry, _fingerprint, sanitize_model_id

if TYPE_CHECKING:
    from ..pipe import Pipe

_DROPPED_PASS_SPACING_FLOOR_SECONDS = 600.0
_DROPPED_REMOVALS_PER_PASS = 25
_DROPPED_SIZE_MEMORY = 24
_DROPPED_SHRINK_ALLOWANCE_FLOOR = 3
_DROPPED_SHRINK_ALLOWANCE_DIVISOR = 10
_DROPPED_PLAUSIBILITY_ROWS = 20
_DROPPED_LISTED_SHARE = 0.4
_DROPPED_FAILURES_BEFORE_ABORT = 3
_DROPPED_LIST_TIMEOUT_SECONDS = 15
_DROPPED_ANNOUNCED_IDS = 10
_DROPPED_WARN_COOLDOWN_SECONDS = 6 * 60 * 60
_DROPPED_PRESET_SHAPES = ("@preset", ":preset/")
_DROPPED_FILTER_PREFIX = "openrouter_"
_DROPPED_COUNT_KEY = "REMOVE_DROPPED_MODELS_AFTER_READS"
_DROPPED_SHOWN_VALUE_CHARS = 60
_DROPPED_UNSCANNED_SETTINGS = frozenset({"ui.model_order_list"})
_DROPPED_NAME_SPLIT = re.compile(r"[,\s;\"'\[\]\(\)\{\}]+")
_DROPPED_LISTS = (
    ("chat", "/models"),
    ("image", "/models?output_modalities=image"),
    ("video", "/videos/models"),
)

_warned_dropped_models: dict[str, float] = {}


class _DroppedPassRefused(Exception):
    def __init__(self, cause: str, message: str, *, resets: bool) -> None:
        super().__init__(message)
        self.cause = cause
        self.resets = resets


@dataclass(frozen=True)
class _DroppedStreak:
    count: int
    updated_at: Any
    since: float


@dataclass(frozen=True)
class _DroppedGuards:
    pipe_tails: frozenset[str]
    filter_tails: frozenset[str]
    setting_ids: frozenset[str]
    bases: frozenset[str]


@dataclass(frozen=True)
class _DroppedStoredGate:
    after_reads: int
    syncing: bool
    valves: Any


def _dropped_syncing(valves: Any) -> bool:
    routing = bool(valves.ADMIN_PROVIDER_ROUTING_MODELS or valves.USER_PROVIDER_ROUTING_MODELS)
    return syncs_owui_models(valves, routing)


def _dropped_count_usable(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _REMOVE_DROPPED_MODELS_MAX


def _dropped_complete_ids(payload: Any) -> tuple[list[str], str]:
    if not isinstance(payload, dict):
        return [], "the answer is not an object"
    data = payload.get("data")
    if not isinstance(data, list):
        return [], "the answer carries no list"
    ids: list[str] = []
    for item in data:
        value = item.get("id") if isinstance(item, dict) else None
        if not isinstance(value, str) or not value.strip():
            return [], "an entry carries no id"
        ids.append(value.strip())
    if not ids:
        return [], "the list is empty"
    total = payload.get("total_count")
    if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total != len(data)):
        return [], "its count disagrees with the list"
    links = payload.get("links")
    if isinstance(links, dict) and links.get("next"):
        return [], "the list continues on another page"
    return ids, ""


def _dropped_strings(value: Any, out: set[str]) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _dropped_strings(key, out)
            _dropped_strings(item, out)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _dropped_strings(item, out)
    else:
        out.add(str(value))


def _dropped_tails_named(strings: Iterable[str], prefix: str) -> set[str]:
    names: set[str] = set()
    for fragment in strings:
        for token in _DROPPED_NAME_SPLIT.split(fragment):
            token = token.lstrip("!")
            if not token:
                continue
            if token.startswith(prefix):
                names.add(token[len(prefix):].lower())
            names.add(sanitize_model_id(token).lower())
    return names


def _dropped_exact_ids(strings: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for fragment in strings:
        ids.add(fragment)
        if "," in fragment:
            ids.update(piece.strip() for piece in fragment.split(","))
    return ids


async def _read_pipe_valve_column(pipe_id: str) -> Any:
    from open_webui.internal.db import get_async_db_context
    from open_webui.models.functions import Function
    from sqlalchemy import select

    async with get_async_db_context() as db:
        result = await db.execute(select(Function.valves).filter_by(id=pipe_id))
        return result.scalar_one_or_none()


class DroppedModelRemover:
    def __init__(
        self,
        *,
        pipe: Pipe,
        logger: logging.Logger,
        task_done_callback: Callable[[asyncio.Task], None] | None = None,
    ) -> None:
        self._pipe = pipe
        self.logger = logger
        self._task_done_callback = task_done_callback
        self._task: asyncio.Task | None = None
        self._next_pass_at = 0.0
        self._streaks: dict[str, _DroppedStreak] = {}
        self._sizes: dict[str, deque[int]] = {}
        self._identity: tuple[str, str] | None = None
        self._failed_deletes: set[str] = set()
        self._announced = False
        self._kept: set[str] = set()

    def maybe_schedule(self) -> None:
        try:
            task = self._task
            if task is not None and not task.done():
                return
            now = time.monotonic()
            if now < self._next_pass_at:
                return
            refresh = float(self._pipe.valves.MODEL_CATALOG_REFRESH_SECONDS)
            self._next_pass_at = now + max(refresh, _DROPPED_PASS_SPACING_FLOOR_SECONDS)
            self._task = asyncio.create_task(self.run_pass(), name="openrouter-dropped-models")
            if self._task_done_callback is not None:
                self._task.add_done_callback(self._task_done_callback)
        except Exception as exc:
            self.logger.log(
                self._level("schedule"),
                "Scheduling the check for models OpenRouter no longer lists failed: %s",
                exc,
                exc_info=True,
            )

    def cancel(self) -> asyncio.Task | None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            return task
        return None

    async def run_pass(self) -> None:
        try:
            await self._pass()
        except _DroppedPassRefused as refusal:
            if refusal.resets:
                self._forget_all()
            self.logger.log(
                self._level(refusal.cause),
                "Not counting this check toward removing models OpenRouter no longer lists: %s.",
                refusal,
            )
        except Exception as exc:
            self._forget_all()
            self.logger.log(
                self._level(f"error:{type(exc).__name__}"),
                "The check for models OpenRouter no longer lists stopped on an error and its count "
                "starts again: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _level(cause: str) -> int:
        return warn_level(_warned_dropped_models, cause, cooldown_s=_DROPPED_WARN_COOLDOWN_SECONDS)

    def _prefix(self) -> str:
        return f"{self._pipe.id}."

    def _forget_all(self) -> None:
        self._streaks.clear()
        self._failed_deletes.clear()

    def _candidate(self, row: Any, prefix: str) -> bool:
        row_id = getattr(row, "id", None)
        if not isinstance(row_id, str) or not row_id.startswith(prefix):
            return False
        if getattr(row, "base_model_id", None):
            return False
        tail = row_id[len(prefix):]
        if not tail or any(shape in tail for shape in _DROPPED_PRESET_SHAPES):
            return False
        if not isinstance(_params_as_mapping(getattr(row, "meta", None)).get(_PIPE_METADATA_KEY), dict):
            return False
        return not _get_disable_param(getattr(row, "params", None), "disable_model_metadata_sync")

    @staticmethod
    def _tail(row: Any, prefix: str) -> str:
        return str(row.id)[len(prefix):].lower()

    @staticmethod
    def _needs(row: Any) -> frozenset[str]:
        pipe_meta = _params_as_mapping(getattr(row, "meta", None)).get(_PIPE_METADATA_KEY)
        capabilities = pipe_meta.get("capabilities") if isinstance(pipe_meta, Mapping) else None
        needs = {"chat"}
        if (
            not isinstance(capabilities, Mapping)
            or "image_output" not in capabilities
            or capabilities["image_output"]
        ):
            needs.add("image")
        if (
            not isinstance(capabilities, Mapping)
            or "video_generation" not in capabilities
            or capabilities["video_generation"]
        ):
            needs.add("video")
        return frozenset(needs)

    @staticmethod
    def _lists_to_read(valves: Any) -> frozenset[str]:
        names = {"chat"}
        if getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False) or _wants_contracts(valves):
            names.add("image")
        if getattr(valves, "ENABLE_VIDEO_GENERATION", False):
            names.add("video")
        return frozenset(names)

    @staticmethod
    def _explained(tail: str, known: set[str], pipe_tails: set[str]) -> bool:
        if tail in known:
            return True
        base, separator, _tag = tail.rpartition(":")
        return bool(separator) and tail in pipe_tails and base in known

    async def _models_table(self) -> Any:
        try:
            from open_webui.models.models import Models
        except Exception as exc:
            raise _DroppedPassRefused(
                "model_rows", f"Open WebUI's model table is not available ({exc})", resets=False
            ) from exc
        return Models

    async def _read_rows(self) -> list[Any]:
        table = await self._models_table()
        try:
            rows = await table.get_all_models()
        except Exception as exc:
            raise _DroppedPassRefused(
                "model_rows", f"the model entries could not be read ({exc})", resets=False
            ) from exc
        return list(rows or [])

    async def _stored_valves(self) -> dict[str, Any]:
        pipe_id = str(self._pipe.id or "")
        try:
            from open_webui.models.functions import Functions

            stored = await _await_if_needed(Functions.get_function_valves_by_id(pipe_id))
            if isinstance(stored, dict) and not stored:
                raw = await _read_pipe_valve_column(pipe_id)
                if not raw_valve_column_decodes(raw):
                    raise _DroppedPassRefused(
                        "stored_valves",
                        "the stored settings are encrypted with a different WEBUI_SECRET_KEY",
                        resets=False,
                    )
        except _DroppedPassRefused:
            raise
        except Exception as exc:
            raise _DroppedPassRefused(
                "stored_valves", f"the stored settings could not be read ({exc})", resets=False
            ) from exc
        if not isinstance(stored, dict):
            raise _DroppedPassRefused(
                "stored_valves", "the stored settings could not be read", resets=False
            )
        return stored

    def _count_unusable(self, stored: Mapping[str, Any]) -> bool:
        if _DROPPED_COUNT_KEY not in stored:
            return False
        value = stored[_DROPPED_COUNT_KEY]
        if _dropped_count_usable(value):
            return False
        shown = repr(value)[:_DROPPED_SHOWN_VALUE_CHARS]
        self.logger.log(
            self._level(f"count:{shown}"),
            "Removal of models OpenRouter no longer lists is off: the stored setting %s is %s, which the pipe "
            "cannot use (it takes a whole number from 0 to %d). It stays off until the setting is fixed.",
            _DROPPED_COUNT_KEY,
            shown,
            _REMOVE_DROPPED_MODELS_MAX,
        )
        return True

    async def _stored_gate(self) -> _DroppedStoredGate:
        stored = await self._stored_valves()
        if self._count_unusable(stored):
            return _DroppedStoredGate(0, False, None)
        try:
            valves = type(self._pipe.valves)(
                **{key: value for key, value in stored.items() if value is not None}
            )
            return _DroppedStoredGate(
                int(valves.REMOVE_DROPPED_MODELS_AFTER_READS), _dropped_syncing(valves), valves
            )
        except Exception as exc:
            raise _DroppedPassRefused(
                "stored_valves", f"the stored settings do not load ({exc})", resets=False
            ) from exc

    async def _filter_valve_tails(self, prefix: str) -> set[str]:
        try:
            from open_webui.models.functions import Functions

            filters = await _await_if_needed(
                Functions.get_functions_by_type("filter", active_only=False)
            )
            ids = sorted(
                {
                    str(getattr(row, "id", ""))
                    for row in filters or []
                    if str(getattr(row, "id", "")).startswith(_DROPPED_FILTER_PREFIX)
                }
            )
            if not ids:
                return set()
            by_id = await _await_if_needed(Functions.get_function_valves_by_ids(ids))
        except Exception as exc:
            raise _DroppedPassRefused(
                "filter_valves", f"the settings of the OpenRouter filters could not be read ({exc})", resets=False
            ) from exc
        if not isinstance(by_id, dict) or any(function_id not in by_id for function_id in ids):
            raise _DroppedPassRefused(
                "filter_valves", "the settings of the OpenRouter filters could not be read", resets=False
            )
        strings: set[str] = set()
        _dropped_strings(list(by_id.values()), strings)
        return _dropped_tails_named(strings, prefix)

    async def _setting_ids(self) -> set[str]:
        try:
            from open_webui.models.config import Config
            from open_webui.models.groups import Groups

            keys = sorted(
                key
                for key in (getattr(Config, "DEFAULTS", None) or {})
                if "model" in str(key).lower() and str(key) not in _DROPPED_UNSCANNED_SETTINGS
            )
            if not keys:
                raise _DroppedPassRefused(
                    "settings", "Open WebUI lists no settings that could name a model", resets=False
                )
            values = await _await_if_needed(Config.get_many(*keys))
            groups = await _await_if_needed(Groups.get_all_groups())
        except _DroppedPassRefused:
            raise
        except Exception as exc:
            raise _DroppedPassRefused(
                "settings", f"Open WebUI's model settings could not be read ({exc})", resets=False
            ) from exc
        if not isinstance(values, dict):
            raise _DroppedPassRefused(
                "settings", "Open WebUI's model settings could not be read", resets=False
            )
        strings: set[str] = set()
        _dropped_strings(values, strings)
        for group in groups or []:
            _dropped_strings(getattr(group, "data", None), strings)
        return _dropped_exact_ids(strings)

    async def _read_guards(self, sources: Iterable[Any], rows: list[Any]) -> _DroppedGuards:
        prefix = self._prefix()
        strings: set[str] = set()
        for valves in sources:
            _dropped_strings(valves.model_dump(), strings)
        return _DroppedGuards(
            pipe_tails=frozenset(_dropped_tails_named(strings, prefix)),
            filter_tails=frozenset(await self._filter_valve_tails(prefix)),
            setting_ids=frozenset(await self._setting_ids()),
            bases=frozenset(
                str(row.base_model_id) for row in rows if getattr(row, "base_model_id", None)
            ),
        )

    def _guard_reason(self, row: Any, prefix: str, guards: _DroppedGuards) -> str | None:
        tail = self._tail(row, prefix)
        if tail in guards.pipe_tails:
            return "a setting of this pipe names it"
        if tail in guards.filter_tails:
            return "a setting of one of this pipe's OpenRouter filters names it"
        if row.id in guards.setting_ids:
            return "an Open WebUI setting names it"
        if row.id in guards.bases:
            return "a workspace model is built on it"
        return None

    def _note_kept(self, row: Any, reason: str) -> None:
        if row.id in self._kept:
            return
        self._kept.add(row.id)
        self.logger.info(
            "Kept the Open WebUI entry %s although OpenRouter no longer lists the model: %s.",
            row.id,
            reason,
        )

    async def _read_list(
        self,
        session: aiohttp.ClientSession,
        name: str,
        url: str,
        headers: dict[str, str],
        usual: int,
    ) -> list[str]:
        cause = f"list:{name}"
        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_DROPPED_LIST_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    raise _DroppedPassRefused(
                        cause, f"the {name} list answered HTTP {resp.status}", resets=True
                    )
                payload = await resp.json()
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as exc:
            raise _DroppedPassRefused(
                cause, f"the {name} list could not be read ({type(exc).__name__})", resets=True
            ) from exc
        ids, reason = _dropped_complete_ids(payload)
        if reason:
            raise _DroppedPassRefused(cause, f"the {name} list is not whole: {reason}", resets=True)
        window = self._sizes.setdefault(name, deque(maxlen=_DROPPED_SIZE_MEMORY))
        biggest = max(max(window, default=0), usual)
        window.append(len(ids))
        allowance = max(_DROPPED_SHRINK_ALLOWANCE_FLOOR, biggest // _DROPPED_SHRINK_ALLOWANCE_DIVISOR)
        if biggest and len(ids) < biggest - allowance:
            raise _DroppedPassRefused(
                cause,
                f"the {name} list shrank from {biggest} to {len(ids)} models",
                resets=True,
            )
        return ids

    async def _read_lists(self, valves: Any, api_key: str, base_url: str) -> dict[str, list[str]]:
        wanted = self._lists_to_read(valves)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            **openrouter_attribution_headers(valves),
        }
        read: dict[str, list[str]] = {}
        session = self._pipe._create_http_session()
        try:
            for name, path in _DROPPED_LISTS:
                if name in wanted:
                    usual = OpenRouterModelRegistry.largest_recent_list_size(name, base_url, api_key)
                    read[name] = await self._read_list(session, name, base_url + path, headers, usual)
        finally:
            with contextlib.suppress(Exception):
                await session.close()
        return read

    async def _pass(self) -> None:
        pipe = self._pipe
        valves = pipe.valves
        after_reads = int(valves.REMOVE_DROPPED_MODELS_AFTER_READS)
        if after_reads <= 0 or not _dropped_syncing(valves):
            self._forget_all()
            return
        if self._count_unusable(await self._stored_valves()):
            self._forget_all()
            return
        api_key, key_error = pipe._resolve_openrouter_api_key(valves)
        if not api_key or key_error:
            self._forget_all()
            return
        base_url = str(valves.BASE_URL or "").rstrip("/")
        identity = (base_url, _fingerprint(api_key))
        if identity != self._identity:
            self._forget_all()
            self._sizes.clear()
            self._identity = identity
        rows = await self._read_rows()
        prefix = self._prefix()
        candidates = [row for row in rows if self._candidate(row, prefix)]
        known = {
            sanitize_model_id(model_id).lower() for model_id in OpenRouterModelRegistry.listed_model_ids()
        }
        strings: set[str] = set()
        _dropped_strings(valves.model_dump(), strings)
        pipe_tails = _dropped_tails_named(strings, prefix)
        absent = [
            row
            for row in candidates
            if not self._explained(self._tail(row, prefix), known, pipe_tails)
        ]
        if not absent:
            self._forget_all()
            return
        lists = await self._read_lists(valves, api_key, base_url)
        census = {
            sanitize_model_id(model_id).lower() for ids in lists.values() for model_id in ids
        }
        judged = set(lists)
        pool = [
            self._tail(row, prefix)
            for row in candidates
            if self._tail(row, prefix) not in pipe_tails and self._needs(row) <= judged
        ]
        listed = sum(1 for tail in pool if tail in census)
        if pool and (
            listed == 0
            or (len(pool) >= _DROPPED_PLAUSIBILITY_ROWS and listed < _DROPPED_LISTED_SHARE * len(pool))
        ):
            raise _DroppedPassRefused(
                "implausible",
                f"OpenRouter's lists name only {listed} of the {len(pool)} entries this pipe made, "
                "which is too few to be the catalogue those entries came from",
                resets=True,
            )
        counted: list[Any] = []
        awaiting: list[Any] = []
        for row in absent:
            if self._tail(row, prefix) in census:
                continue
            (counted if self._needs(row) <= judged else awaiting).append(row)
        to_count: list[Any] = []
        if counted:
            guards = await self._read_guards((valves,), rows)
            for row in counted:
                reason = self._guard_reason(row, prefix, guards)
                if reason is None:
                    to_count.append(row)
                else:
                    self._note_kept(row, reason)
        now = time.monotonic()
        streaks: dict[str, _DroppedStreak] = {}
        for row in to_count:
            previous = self._streaks.get(row.id)
            if previous is None or previous.updated_at != row.updated_at:
                previous = _DroppedStreak(0, row.updated_at, now)
            streaks[row.id] = _DroppedStreak(previous.count + 1, row.updated_at, previous.since)
        self._streaks = streaks
        if (to_count or awaiting) and not self._announced:
            self._announce(to_count, awaiting, prefix, after_reads)
        due = sorted(row_id for row_id, streak in streaks.items() if streak.count >= after_reads)
        removed = await self._remove(due) if due else 0
        self.logger.debug(
            "Check for models OpenRouter no longer lists: %d entries not listed, %d counted, "
            "%d waiting for a list this install does not read, %d due, %d removed",
            len(absent),
            len(to_count),
            len(awaiting),
            len(due),
            removed,
        )

    def _announce(
        self, counted: list[Any], awaiting: list[Any], prefix: str, after_reads: int
    ) -> None:
        self._announced = True
        names = ", ".join(
            sorted(str(row.id)[len(prefix):] for row in counted + awaiting)[:_DROPPED_ANNOUNCED_IDS]
        )
        waiting = (
            f" {len(awaiting)} of them wait for the image or video list, which this install does not read."
            if awaiting
            else ""
        )
        self.logger.info(
            "%d Open WebUI model entries made by this pipe are for models OpenRouter no longer lists "
            "(first: %s).%s An entry is removed once %d complete checks in a row miss its model, "
            "at most %d per check on this worker; set REMOVE_DROPPED_MODELS_AFTER_READS to 0 to keep them all.",
            len(counted) + len(awaiting),
            names,
            waiting,
            after_reads,
            _DROPPED_REMOVALS_PER_PASS,
        )

    def _log_removed(self, row: Any, streak: _DroppedStreak) -> None:
        hours = max(0.0, time.monotonic() - streak.since) / 3600
        self.logger.info(
            "Removed the Open WebUI entry %s (%s): OpenRouter did not list the model in %d complete "
            "checks in a row, over about %.1f hours. Owner: %s. Access grants: %d. If OpenRouter "
            "lists the model again it returns as a new entry with the default access for new models.",
            row.id,
            getattr(row, "name", ""),
            streak.count,
            hours,
            "a user" if getattr(row, "user_id", "") else "none",
            len(getattr(row, "access_grants", None) or []),
        )

    async def _remove(self, due: list[str]) -> int:
        gate = await self._stored_gate()
        if gate.after_reads <= 0 or not gate.syncing:
            self._forget_all()
            return 0
        table = await self._models_table()
        rows = await self._read_rows()
        by_id = {str(row.id): row for row in rows}
        guards = await self._read_guards((gate.valves, self._pipe.valves), rows)
        prefix = self._prefix()
        batch: list[Any] = []
        for row_id in sorted(due, key=lambda value: (value in self._failed_deletes, value)):
            streak = self._streaks.get(row_id)
            row = by_id.get(row_id)
            if streak is None:
                continue
            reason = None if row is None else self._guard_reason(row, prefix, guards)
            if row is not None and reason is not None:
                self._note_kept(row, reason)
            if (
                row is None
                or reason is not None
                or not self._candidate(row, prefix)
                or row.updated_at != streak.updated_at
            ):
                self._streaks.pop(row_id, None)
                self._failed_deletes.discard(row_id)
                continue
            batch.append(row)
        removed = 0
        failures = 0
        for row in batch[:_DROPPED_REMOVALS_PER_PASS]:
            live = self._pipe.valves
            if int(live.REMOVE_DROPPED_MODELS_AFTER_READS) <= 0 or not _dropped_syncing(live):
                self._forget_all()
                break
            gate = await self._stored_gate()
            if gate.after_reads <= 0 or not gate.syncing:
                self._forget_all()
                break
            streak = self._streaks.get(str(row.id))
            if streak is None or streak.count < gate.after_reads:
                continue
            existing = await table.get_model_by_id(row.id)
            if (
                existing is None
                or not self._candidate(existing, prefix)
                or existing.updated_at != streak.updated_at
            ):
                self._streaks.pop(str(row.id), None)
                continue
            try:
                await table.delete_model_by_id(row.id)
            except Exception as exc:
                self._failed_deletes.add(str(row.id))
                failures += 1
                self.logger.log(
                    self._level("delete"),
                    "Open WebUI could not remove the entry %s: %s",
                    row.id,
                    exc,
                    exc_info=True,
                )
                if failures >= _DROPPED_FAILURES_BEFORE_ABORT:
                    break
                continue
            failures = 0
            self._failed_deletes.discard(str(row.id))
            self._streaks.pop(str(row.id), None)
            removed += 1
            self._log_removed(existing, streak)
        return removed
