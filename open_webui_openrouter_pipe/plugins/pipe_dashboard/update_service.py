"""Self-update service for the pipe_dashboard Update tab.

Downloads tagged release bundles from the configured GitHub repo, validates
them (size cap, sha256 digest, frontmatter, OWUI-version gate), snapshots the
current function row to OWUI Files, then runs OWUI's own loader sequence
(exec-validate + hot reload) before writing the new content to the row.
All OWUI access goes through lazy imports so tests can patch the real modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
from typing import Any, NoReturn

_pd_update_log = logging.getLogger(__name__)

FUNCTION_ID = "open_webui_openrouter_pipe"
DEFAULT_REPO = "rbb-dev/Open-WebUI-OpenRouter-pipe"
MIN_UPDATE_VERSION = "2.7.0"

_PD_UPDATE_CHECK_MEMO_S = 60.0
_PD_UPDATE_SNAPSHOT_SLOTS = 10
_PD_UPDATE_XLOCK_TIMEOUT_S = 300
_PD_UPDATE_LEADER_TTL_S = 30 * 60
_PD_UPDATE_LEADER_RENEW_S = 10 * 60
_PD_UPDATE_FOLLOWER_POLL_S = 60 * 60.0
_PD_UPDATE_SIZE_CAP = 8 * 1024 * 1024
_PD_UPDATE_NOTES_CAP = 20 * 1024
_PD_UPDATE_AUTO_INTERVAL = 6 * 3600.0
_PD_UPDATE_AUTO_JITTER = (300.0, 900.0)
_PD_UPDATE_BACKOFF = (900.0, 1800.0, 3600.0, 7200.0)

_ASSET_NAMES = {
    False: "open_webui_openrouter_pipe_bundled.py",
    True: "open_webui_openrouter_pipe_bundled_compressed.py",
}

_REPO_RE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.\-]+$"

_TRANSIENT_CODES = frozenset(
    {"offline", "rate_limited", "repo_not_found", "bad_repo_valve", "stale_rev", "update_in_progress"}
)


class UpdateError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def _now() -> float:
    return time.time()


def _files_model() -> Any:
    from open_webui.models.files import Files

    return Files


def _distributed_lock(
    suffix: str = "update_lock", ttl: int = _PD_UPDATE_XLOCK_TIMEOUT_S
) -> Any | None:
    try:
        from open_webui.env import (
            REDIS_KEY_PREFIX,
            WEBSOCKET_MANAGER,
            WEBSOCKET_REDIS_CLUSTER,
            WEBSOCKET_REDIS_URL,
            WEBSOCKET_SENTINEL_HOSTS,
            WEBSOCKET_SENTINEL_PORT,
        )
        from open_webui.socket.utils import RedisLock
        from open_webui.utils.redis import get_sentinels_from_env

        if str(WEBSOCKET_MANAGER or "").lower() != "redis" or not WEBSOCKET_REDIS_URL:
            return None
        return RedisLock(
            redis_url=WEBSOCKET_REDIS_URL,
            lock_name=f"{REDIS_KEY_PREFIX}:{FUNCTION_ID}:{suffix}",
            timeout_secs=ttl,
            redis_sentinels=get_sentinels_from_env(
                WEBSOCKET_SENTINEL_HOSTS, WEBSOCKET_SENTINEL_PORT
            ),
            redis_cluster=bool(WEBSOCKET_REDIS_CLUSTER),
        )
    except Exception:
        _pd_update_log.warning("update: cross-worker lock unavailable", exc_info=True)
        return None


def _snapshot_storage() -> Any:
    from ...storage.owui_files import get_owui_storage

    return get_owui_storage()


async def _materialize_snapshot(record: Any) -> Any:
    from ...storage.owui_files import materialize_owui_file_to_temp

    return await materialize_owui_file_to_temp(
        record,
        user=None,
        logger=_pd_update_log,
        max_bytes=_PD_UPDATE_SIZE_CAP,
        allow_unknown_size=True,
        require_auth=False,
    )


def _client_ssl() -> Any:
    try:
        from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL

        return AIOHTTP_CLIENT_SESSION_SSL
    except Exception:
        return True


async def _http_get_json(session: Any, url: str, *, timeout: float = 15.0) -> tuple[int, Any, dict]:
    import aiohttp

    own = None
    try:
        if session is None or getattr(session, "closed", False):
            own = aiohttp.ClientSession(trust_env=True)
            session = own
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.get(
            url, timeout=client_timeout, allow_redirects=True, ssl=_client_ssl()
        ) as resp:
            headers = {k: v for k, v in resp.headers.items()}
            payload = None
            if resp.status == 200:
                payload = await resp.json(content_type=None)
            return resp.status, payload, headers
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("offline", str(exc)) from exc
    finally:
        if own is not None:
            await own.close()


async def _http_get_bytes(
    session: Any, url: str, *, timeout: float = 60.0, cap: int = _PD_UPDATE_SIZE_CAP
) -> bytes:
    import aiohttp

    own = None
    try:
        if session is None or getattr(session, "closed", False):
            own = aiohttp.ClientSession(trust_env=True)
            session = own
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.get(
            url, timeout=client_timeout, allow_redirects=True, ssl=_client_ssl()
        ) as resp:
            if resp.status != 200:
                raise UpdateError("offline", f"download failed with HTTP {resp.status}")
            declared = resp.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > cap:
                raise UpdateError("validation_failed", "asset exceeds the size cap")
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > cap:
                    raise UpdateError("validation_failed", "asset exceeds the size cap")
                chunks.append(chunk)
            return b"".join(chunks)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("offline", str(exc)) from exc
    finally:
        if own is not None:
            await own.close()


class UpdateService:
    def __init__(self, get_pipe: Any) -> None:
        self._get_pipe = get_pipe
        self._lock = asyncio.Lock()
        self._last_good: dict[str, Any] | None = None
        self._last_error: dict[str, Any] | None = None
        self._auto_skip: dict[str, dict[str, Any]] = {}
        self._auto_last: dict[str, Any] | None = None
        self._backoff_idx = 0
        self._commit_inflight = False
        self._auto_role = "pending"

    # ── plumbing ────────────────────────────────────────────────────────────

    def _pipe(self) -> Any:
        pipe = self._get_pipe() if callable(self._get_pipe) else None
        if pipe is None:
            raise UpdateError("validation_failed", "pipe unavailable")
        return pipe

    def _valves(self) -> Any:
        return self._pipe().valves

    @staticmethod
    def _functions() -> Any:
        from open_webui.models.functions import Functions

        return Functions

    async def _row(self) -> Any:
        row = await self._functions().get_function_by_id(self._pipe().id)
        if row is None:
            raise UpdateError("validation_failed", "function row not found")
        return row

    def _repo(self, value: Any = None) -> str:
        import re

        if value is None:
            value = getattr(self._valves(), "PIPE_DASHBOARD_UPDATE_REPO", "")
        repo = str(value or "")
        if not re.match(_REPO_RE_PATTERN, repo):
            raise UpdateError("bad_repo_valve", f"invalid owner/repo value: {repo!r}")
        return repo

    @staticmethod
    def _frontmatter(content: str) -> dict:
        import open_webui.utils.plugin as owp

        return owp.extract_frontmatter(content) or {}

    def _installed_version(self, row: Any) -> str:
        version = self._frontmatter(getattr(row, "content", "") or "").get("version", "")
        if version:
            return str(version)
        try:
            from open_webui_openrouter_pipe import __version__

            return str(__version__)
        except Exception:
            return "0.0.0"

    def detect_mode(self, content: str | None = None) -> dict[str, Any]:
        import re

        if content:
            if re.search(r"^_BUNDLED_SOURCES_Z\b[^=\n]{0,80}=", content, re.MULTILINE):
                return {"mode": "bundle", "compressed": True}
            if re.search(
                r"^(?:_INTERMEDIATE_PACKAGES\b[^=\n]{0,80}=|def _install_package_alias\b)",
                content,
                re.MULTILINE,
            ):
                return {"mode": "bundle", "compressed": False}
            return {"mode": "package", "compressed": False}
        mod = sys.modules.get(f"function_{self._pipe().id}")
        if mod is not None:
            if hasattr(mod, "_BUNDLED_SOURCES_Z"):
                return {"mode": "bundle", "compressed": True}
            if hasattr(mod, "_INTERMEDIATE_PACKAGES") or hasattr(mod, "_install_package_alias"):
                return {"mode": "bundle", "compressed": False}
        return {"mode": "package", "compressed": False}

    def _session(self) -> Any:
        return getattr(self._pipe(), "_http_session", None)

    # ── check ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_published(published: Any) -> float | None:
        from datetime import datetime

        try:
            return datetime.fromisoformat(str(published).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(a) > Version(b)
        except InvalidVersion:
            return False

    def _latest_block(self, release: dict) -> dict[str, Any]:
        tag = str(release.get("tag_name", "") or "")
        assets_by_name = {a.get("name"): a for a in release.get("assets", []) if isinstance(a, dict)}

        def _slim(name: str) -> dict[str, Any] | None:
            asset = assets_by_name.get(name)
            if not isinstance(asset, dict) or not asset.get("browser_download_url"):
                return None
            return {
                "name": asset.get("name"),
                "size": asset.get("size"),
                "digest": asset.get("digest"),
                "browser_download_url": asset.get("browser_download_url"),
            }

        version = tag[1:] if tag.startswith("v") else tag
        return {
            "version": version,
            "tag": tag,
            "published_at": release.get("published_at"),
            "notes": str(release.get("body", "") or "")[:_PD_UPDATE_NOTES_CAP],
            "supported": not self._version_gt(MIN_UPDATE_VERSION, version),
            "assets": {
                "flat": _slim(_ASSET_NAMES[False]),
                "compressed": _slim(_ASSET_NAMES[True]),
            },
        }

    async def _fetch_latest(self, repo: str) -> dict[str, Any]:
        status, payload, headers = await _http_get_json(
            self._session(), f"https://api.github.com/repos/{repo}/releases/latest"
        )
        if status == 200 and isinstance(payload, dict):
            return payload
        if status == 404:
            raise UpdateError("repo_not_found", f"no releases found for {repo}")
        if status in (403, 429):
            reset = ""
            for wanted in ("x-ratelimit-reset", "retry-after"):
                for key, value in headers.items():
                    if str(key).lower() == wanted and str(value).strip():
                        reset = str(value)
                        break
                if reset:
                    break
            err = UpdateError("rate_limited", "")
            setattr(err, "reset", reset)
            raise err
        raise UpdateError("offline", f"GitHub returned HTTP {status}")

    async def check(self, *, force: bool = False) -> dict[str, Any]:
        valves = await self._row_valves()
        pipe = self._pipe()
        row = await self._row()
        mode = self.detect_mode(getattr(row, "content", "") or "")
        installed_version = self._installed_version(row)
        now = _now()

        repo = ""
        release: dict | None = None
        cached = False
        try:
            repo = self._repo(valves.get("PIPE_DASHBOARD_UPDATE_REPO"))
            memo = self._last_good
            if (
                not force
                and memo is not None
                and memo.get("repo") == repo
                and now - float(memo.get("at", 0.0)) < _PD_UPDATE_CHECK_MEMO_S
            ):
                release = memo.get("release")
                cached = True
            else:
                release = await self._fetch_latest(repo)
                self._last_good = {"repo": repo, "at": now, "release": release}
                self._last_error = None
        except UpdateError as exc:
            self._last_error = {
                "code": exc.code,
                "ts": now,
                "message": exc.message,
                "reset": str(getattr(exc, "reset", "") or ""),
            }
            memo = self._last_good
            if memo is not None and memo.get("repo") == repo:
                release = memo.get("release")

        latest = self._latest_block(release) if isinstance(release, dict) else None
        installed_compressed = bool(mode.get("compressed"))
        update_available = False
        no_matching_asset = False
        eligible_at = None
        if latest is not None:
            update_available = self._version_gt(latest["version"], installed_version)
            installed_asset = latest["assets"]["compressed" if installed_compressed else "flat"]
            no_matching_asset = update_available and installed_asset is None
            published_ts = self._parse_published(latest.get("published_at"))
            if published_ts is not None:
                delay_h = int(valves.get("PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS", 168) or 0)
                eligible_at = published_ts + delay_h * 3600.0

        try:
            records = await self._snapshot_records()
        except Exception:
            _pd_update_log.warning("update: snapshot slot read failed", exc_info=True)
            records = []

        auto_success = None
        for rec in reversed(records):
            if rec["actor"] == "auto":
                auto_success = {
                    "from_version": rec["version"],
                    "to_version": installed_version,
                    "ts": rec["ts"],
                }
                break

        this_worker = None
        if self._auto_skip:
            version, paused = next(reversed(self._auto_skip.items()))
            this_worker = {
                "paused_for_version": version,
                "code": paused.get("code"),
                "ts": paused.get("ts"),
            }
        elif self._auto_last is not None:
            this_worker = dict(self._auto_last)

        memo = self._last_good
        repo_shown = repo or str(valves.get("PIPE_DASHBOARD_UPDATE_REPO") or "")
        return {
            "enabled": bool(valves.get("PIPE_DASHBOARD_UPDATE_ENABLE", True)),
            "repo": repo_shown,
            "repo_is_default": repo_shown == DEFAULT_REPO,
            "installed": {
                "version": installed_version,
                "mode": mode["mode"],
                "compressed": installed_compressed,
            },
            "latest": latest,
            "update_available": update_available,
            "no_matching_asset": no_matching_asset,
            "snapshots": await self._payload_from(records),
            "checked_at": float(memo.get("at", 0.0)) if memo else None,
            "cached": cached,
            "last_check_error": dict(self._last_error) if self._last_error else None,
            "auto": {
                "enabled": bool(valves.get("PIPE_DASHBOARD_UPDATE_AUTO", False)),
                "delay_hours": int(valves.get("PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS", 168) or 0),
                "eligible_at": eligible_at,
                "last_success": auto_success,
                "this_worker": this_worker,
                "role": self._auto_role,
            },
            "rev": getattr(row, "updated_at", None),
            "pipe_id": pipe.id,
        }

    # ── fetch + validate ─────────────────────────────────────────────────────

    async def fetch_and_validate(self, asset: dict[str, Any], *, require_newer: bool) -> str:
        url = str(asset.get("browser_download_url", "") or "")
        declared_size = asset.get("size")
        if isinstance(declared_size, int) and declared_size > _PD_UPDATE_SIZE_CAP:
            raise UpdateError("validation_failed", "asset exceeds the size cap")

        digest = str(asset.get("digest", "") or "")
        if not digest.startswith("sha256:"):
            raise UpdateError("digest_mismatch", "release asset carries no sha256 digest")

        data = await _http_get_bytes(self._session(), url)
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest.removeprefix("sha256:"):
            raise UpdateError("digest_mismatch", "downloaded bytes do not match the release digest")

        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UpdateError("validation_failed", "asset is not valid UTF-8") from exc

        await self._validate_content(content, require_newer=require_newer)
        return content

    async def _validate_content(self, content: str, *, require_newer: bool) -> dict:
        frontmatter = self._frontmatter(content)
        if not frontmatter:
            raise UpdateError("validation_failed", "asset carries no frontmatter header")
        if frontmatter.get("id") != FUNCTION_ID:
            raise UpdateError("validation_failed", f"frontmatter id is {frontmatter.get('id')!r}")
        new_version = str(frontmatter.get("version", "") or "")
        if not new_version:
            raise UpdateError("validation_failed", "frontmatter carries no version")

        if require_newer:
            if self._version_gt(MIN_UPDATE_VERSION, new_version):
                raise UpdateError(
                    "validation_failed",
                    f"releases before {MIN_UPDATE_VERSION} predate the self-updater and would remove it",
                )
            installed = self._installed_version(await self._row())
            if new_version != installed and not self._version_gt(new_version, installed):
                raise UpdateError(
                    "validation_failed",
                    f"version {new_version} is older than installed {installed}; downgrade via a snapshot instead",
                )

        required = str(frontmatter.get("required_open_webui_version", "") or "")
        if required:
            try:
                from open_webui.env import VERSION as owui_version
            except Exception:
                owui_version = ""
            if owui_version and self._version_gt(required, str(owui_version)):
                raise UpdateError(
                    "incompatible_owui",
                    f"release needs Open WebUI >= {required}, running {owui_version}",
                )
        return frontmatter

    # ── loader seam ──────────────────────────────────────────────────────────

    async def _reload_via_loader(self, content: str) -> tuple[Any, dict, str]:
        import open_webui.utils.plugin as owp

        pipe_id = self._pipe().id
        content = owp.replace_imports(content)
        module_name = f"function_{pipe_id}"
        old_module = sys.modules.get(module_name)
        try:
            instance, _ftype, frontmatter = await owp.load_function_module_by_id(
                pipe_id, content=content
            )
        except Exception as exc:
            if old_module is not None:
                sys.modules[module_name] = old_module
            try:
                await self._functions().update_function_by_id(pipe_id, {"is_active": True})
            except Exception:
                _pd_update_log.warning("update: is_active repair failed", exc_info=True)
            raise UpdateError("exec_failed", str(exc)) from exc
        return instance, dict(frontmatter or {}), content

    # ── snapshots (slot-ID files in OWUI Files; nothing durable in function meta) ──

    def _slot_ids(self) -> list[str]:
        pid = self._pipe().id
        return [f"pipe-update-snapshot-{pid}-{i}" for i in range(_PD_UPDATE_SNAPSHOT_SLOTS)]

    async def _slot_rows(self) -> list[Any]:
        rows = await _files_model().get_files_by_ids(self._slot_ids())
        return list(rows or [])

    def _records_from(self, rows: list[Any]) -> list[dict[str, Any]]:
        ids = self._slot_ids()
        records: list[dict[str, Any]] = []
        for row in rows:
            file_id = str(getattr(row, "id", "") or "")
            if file_id not in ids:
                continue
            meta = getattr(row, "meta", None)
            snap = meta.get("update_snapshot") if isinstance(meta, dict) else None
            if not isinstance(snap, dict):
                continue
            ts = snap.get("ts")
            records.append(
                {
                    "file_id": file_id,
                    "slot": ids.index(file_id),
                    "path": getattr(row, "path", None),
                    "version": snap.get("from_version"),
                    "sha256": str(snap.get("sha256") or ""),
                    "size": snap.get("size"),
                    "ts": ts,
                    "actor": snap.get("actor"),
                }
            )
        records.sort(key=lambda r: (r["ts"] if isinstance(r["ts"], (int, float)) else 0.0, r["slot"]))
        return records

    async def _snapshot_records(self) -> list[dict[str, Any]]:
        return self._records_from(await self._slot_rows())

    @staticmethod
    async def _actor_names(records: list[dict[str, Any]]) -> dict[str, str]:
        names: dict[str, str] = {}
        for actor in {str(r.get("actor") or "") for r in records}:
            if not actor or actor == "auto":
                continue
            try:
                from open_webui.models.users import Users

                user = await Users.get_user_by_id(actor)
                name = str(getattr(user, "name", "") or "") if user is not None else ""
                if name:
                    names[actor] = name
            except Exception:
                continue
        return names

    async def _payload_from(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        names = await self._actor_names(records)
        return [
            {
                "file_id": r["file_id"],
                "version": r["version"],
                "ts": r["ts"],
                "size": r["size"],
                "actor": r["actor"],
                "actor_name": names.get(str(r.get("actor") or "")),
                "sha256": r["sha256"],
            }
            for r in reversed(records)
        ]

    async def _delete_blob_path(self, storage: Any, path: Any) -> None:
        if not path or storage is None:
            return
        try:
            await asyncio.to_thread(storage.delete_file, path)
        except Exception:
            _pd_update_log.warning("update: snapshot blob delete failed for %s", path)

    async def _delete_record(self, record: dict[str, Any], storage: Any) -> bool:
        try:
            deleted = await _files_model().delete_file_by_id(record["file_id"])
        except Exception:
            _pd_update_log.warning("update: snapshot record delete failed for %s", record["file_id"])
            return False
        if not deleted:
            _pd_update_log.warning("update: snapshot record delete refused for %s", record["file_id"])
            return False
        await self._delete_blob_path(storage, record.get("path"))
        return True

    async def snapshot_current(self, actor: str, owner_id: str) -> str:
        import io

        row = await self._row()
        content = getattr(row, "content", "") or ""
        data = content.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        try:
            rows = await self._slot_rows()
        except Exception as exc:
            raise UpdateError("validation_failed", f"snapshot storage is unavailable: {exc}") from exc
        records = self._records_from(rows)

        for rec in records:
            if rec["sha256"] == sha:
                return rec["file_id"]

        ids = self._slot_ids()
        taken = {str(getattr(r, "id", "") or "") for r in rows}
        free = [i for i, sid in enumerate(ids) if sid not in taken]
        storage = _snapshot_storage()
        if storage is None:
            raise UpdateError("validation_failed", "Open WebUI storage is unavailable")

        rotation: dict[str, Any] | None = None
        if free:
            slot = free[0]
        else:
            if not records:
                raise UpdateError(
                    "validation_failed", "all snapshot slots are occupied by foreign files"
                )
            rotation = records[0]
            slot = rotation["slot"]

        import uuid

        from_version = self._installed_version(row)
        filename = f"{FUNCTION_ID}-{from_version}-{sha[:8]}-{uuid.uuid4().hex[:8]}.py"
        try:
            _contents, path = await asyncio.to_thread(
                storage.upload_file, io.BytesIO(data), filename, {}
            )
        except Exception as exc:
            raise UpdateError("validation_failed", f"snapshot upload failed: {exc}") from exc

        deferred_blob: Any = None
        if rotation is not None:
            try:
                rotated = await _files_model().delete_file_by_id(rotation["file_id"])
            except Exception as exc:
                await self._delete_blob_path(storage, path)
                raise UpdateError(
                    "validation_failed", f"snapshot slot rotation failed: {exc}"
                ) from exc
            if not rotated:
                await self._delete_blob_path(storage, path)
                raise UpdateError("validation_failed", "snapshot slot rotation was refused")
            deferred_blob = rotation.get("path")
            records = [r for r in records if r["file_id"] != rotation["file_id"]]

        from open_webui.models.files import FileForm

        ts = _now()
        if records:
            last_ts = records[-1]["ts"]
            if isinstance(last_ts, (int, float)) and last_ts >= ts:
                ts = last_ts + 0.001
        entry = {
            "from_version": from_version,
            "sha256": sha,
            "size": len(data),
            "ts": ts,
            "actor": actor,
            "slot": slot,
        }

        async def _abort_insert(reason: str, cause: Exception | None = None) -> NoReturn:
            await self._delete_blob_path(storage, path)
            await self._delete_blob_path(storage, deferred_blob)
            if rotation is not None:
                _pd_update_log.warning(
                    "update: rotation lost snapshot %s after insert failure", rotation["file_id"]
                )
            raise UpdateError("validation_failed", reason) from cause

        try:
            record = await _files_model().insert_new_file(
                owner_id,
                FileForm(
                    **{
                        "id": ids[slot],
                        "filename": filename,
                        "path": path,
                        "data": {},
                        "meta": {
                            "name": filename,
                            "content_type": "text/x-python",
                            "size": len(data),
                            "update_snapshot": entry,
                        },
                    }
                ),
            )
        except Exception as exc:
            await _abort_insert(f"snapshot record insert failed: {exc}", exc)
        if record is None or not getattr(record, "id", None):
            await _abort_insert("snapshot record insert was rejected")

        keep = int(getattr(self._valves(), "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP", 3) or 3)
        survivors = records + [
            {"file_id": ids[slot], "slot": slot, "path": path, "sha256": sha,
             "version": from_version, "size": len(data), "ts": entry["ts"], "actor": actor}
        ]
        while len(survivors) > max(keep, 1):
            old = survivors.pop(0)
            if old["file_id"] == ids[slot]:
                survivors.insert(0, old)
                break
            await self._delete_record(old, storage)

        await self._delete_blob_path(storage, deferred_blob)
        return ids[slot]

    async def _read_blob(self, file_id: str) -> bytes:
        from pathlib import Path

        record = await _files_model().get_file_by_id(file_id)
        if record is None:
            raise UpdateError("not_found", f"snapshot file {file_id} is missing")
        try:
            local_path = await _materialize_snapshot(record)
        except Exception as exc:
            raise UpdateError("validation_failed", f"snapshot could not be read: {exc}") from exc
        try:
            return await asyncio.to_thread(Path(local_path).read_bytes)
        finally:
            try:
                await asyncio.to_thread(Path(local_path).unlink)
            except Exception:
                pass

    # ── rev chain + commit ───────────────────────────────────────────────────

    @staticmethod
    def _coerce_rev(value: Any) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    async def _rev_guard(self, expected: Any) -> int:
        row = await self._row()
        current = self._coerce_rev(getattr(row, "updated_at", None))
        wanted = self._coerce_rev(expected)
        if current is None or wanted is None or current != wanted:
            raise UpdateError(
                "stale_rev", f"function row changed (rev {current} != expected {wanted})"
            )
        return current

    def _require_idle(self) -> None:
        if self._lock.locked() or self._commit_inflight:
            raise UpdateError("update_in_progress", "another update is already running on this worker")

    @staticmethod
    def _dispose_lock(lock: Any | None) -> None:
        client = getattr(lock, "redis", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass
        pool = getattr(client, "connection_pool", None)
        if pool is not None:
            try:
                pool.disconnect()
            except Exception:
                pass

    @staticmethod
    async def _acquire_cross_worker() -> Any | None:
        lock = _distributed_lock()
        if lock is None:
            return None
        acquired = await asyncio.to_thread(lock.aquire_lock)
        if not acquired:
            UpdateService._dispose_lock(lock)
            raise UpdateError("update_in_progress", "another worker is applying an update")
        return lock

    @staticmethod
    async def _release_cross_worker(lock: Any | None) -> None:
        if lock is None:
            return
        try:
            await asyncio.to_thread(lock.release_lock)
        except Exception:
            _pd_update_log.warning("update: cross-worker lock release failed", exc_info=True)
        UpdateService._dispose_lock(lock)

    async def _commit(
        self, content: str, rev: int, request: Any, actor: str, from_version: str
    ) -> dict[str, Any]:
        await self._rev_guard(rev)
        instance, frontmatter, final = await self._reload_via_loader(content)
        await self._rev_guard(rev)
        pipe_id = self._pipe().id
        functions = self._functions()
        written = await functions.update_function_by_id(pipe_id, {"content": final})
        if written is None:
            raise UpdateError(
                "validation_failed",
                "the database rejected the function-row write; the previous version remains active",
            )
        merged = await functions.update_function_metadata_by_id(pipe_id, {"manifest": frontmatter})
        if merged is None:
            _pd_update_log.warning(
                "update: manifest merge was refused (cosmetic); content is persisted"
            )
        if request is not None:
            import open_webui.utils.plugin as owp

            owp.get_functions_cache(request)[pipe_id] = instance
            owp.get_function_contents_cache(request)[pipe_id] = final
        to_version = str(frontmatter.get("version", "") or "")
        _pd_update_log.info(
            "update: applied actor=%s from=%s to=%s sha256=%s",
            actor,
            from_version,
            to_version,
            hashlib.sha256(final.encode("utf-8")).hexdigest()[:12],
        )
        return {"ok": True, "from_version": from_version, "to_version": to_version}

    async def _shielded_commit(
        self,
        content: str,
        rev: int,
        request: Any,
        actor: str,
        from_version: str,
        xlock: Any | None = None,
    ) -> dict[str, Any]:
        self._commit_inflight = True
        loop = asyncio.get_running_loop()
        commit = asyncio.ensure_future(self._commit(content, rev, request, actor, from_version))

        def _settle(fut: Any) -> None:
            self._commit_inflight = False
            if xlock is not None:
                try:
                    loop.create_task(self._release_cross_worker(xlock))
                except Exception:
                    _pd_update_log.warning("update: cross-worker lock release scheduling failed")
            try:
                exc = fut.exception()
            except asyncio.CancelledError:
                return
            if exc is None:
                _pd_update_log.info("update: commit completed (actor=%s)", actor)
            else:
                self._auto_last = {
                    "code": getattr(exc, "code", "internal"),
                    "ts": _now(),
                    "message": str(exc),
                }
                _pd_update_log.warning("update: commit failed: %s", exc)

        commit.add_done_callback(_settle)
        try:
            return await asyncio.shield(commit)
        except asyncio.CancelledError:
            raise

    # ── apply / restore / snapshot_delete ────────────────────────────────────

    async def apply(self, args: dict, *, actor: str, actor_id: str, request: Any) -> dict[str, Any]:
        if request is None and actor != "auto":
            raise UpdateError("internal", "request is required for manual updates")
        self._require_idle()
        async with self._lock:
            xlock = await self._acquire_cross_worker()
            try:
                row = await self._row()
                mode = self.detect_mode(getattr(row, "content", "") or "")
                if mode["mode"] != "bundle":
                    raise UpdateError(
                        "package_mode", "package/stub installs update via the pinned requirement"
                    )
                rev = await self._rev_guard(args.get("rev"))
                snap = await self.check(force=True)
                latest = snap.get("latest")
                if latest is None:
                    err = snap.get("last_check_error") or {}
                    raised = UpdateError(str(err.get("code") or "offline"), str(err.get("message") or ""))
                    if err.get("reset"):
                        setattr(raised, "reset", str(err["reset"]))
                    raise raised
                if actor == "auto":
                    compressed = bool(mode["compressed"])
                else:
                    arg = args.get("compressed")
                    compressed = bool(mode["compressed"]) if arg is None else bool(arg)
                asset = latest["assets"]["compressed" if compressed else "flat"]
                if asset is None:
                    raise UpdateError(
                        "no_matching_asset",
                        f"latest release carries no {'compressed' if compressed else 'flat'} bundle asset",
                    )
                from_version = str(snap["installed"]["version"])
                content = await self.fetch_and_validate(asset, require_newer=True)
                rev = await self._rev_guard(rev)
                owner = actor_id if actor == "auto" else await self._super_admin_id()
                await self.snapshot_current(actor, owner)
                handoff, xlock = xlock, None
                result = await self._shielded_commit(
                    content, rev, request, actor, from_version, xlock=handoff
                )
                self._auto_skip.clear()
                return result
            finally:
                if xlock is not None:
                    await self._release_cross_worker(xlock)

    async def restore(self, args: dict, *, actor: str, actor_id: str, request: Any) -> dict[str, Any]:
        if request is None and actor != "auto":
            raise UpdateError("internal", "request is required for manual restores")
        self._require_idle()
        async with self._lock:
            xlock = await self._acquire_cross_worker()
            try:
                rev = await self._rev_guard(args.get("rev"))
                row = await self._row()
                file_id = str(args.get("file_id") or "")
                try:
                    records = await self._snapshot_records()
                except Exception as exc:
                    raise UpdateError(
                        "validation_failed", f"snapshot storage is unavailable: {exc}"
                    ) from exc
                entry = next((r for r in records if r["file_id"] == file_id), None)
                if entry is None:
                    raise UpdateError("not_found", f"{file_id} is not a known snapshot")
                blob = await self._read_blob(file_id)
                if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                    raise UpdateError(
                        "digest_mismatch", "snapshot content no longer matches its pinned sha256"
                    )
                try:
                    content = blob.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise UpdateError("validation_failed", "snapshot is not valid UTF-8") from exc
                await self._validate_content(content, require_newer=False)
                from_version = self._installed_version(row)
                owner = actor_id if actor == "auto" else await self._super_admin_id()
                await self.snapshot_current(actor, owner)
                handoff, xlock = xlock, None
                result = await self._shielded_commit(
                    content, rev, request, actor, from_version, xlock=handoff
                )
                self._auto_skip.clear()
                return result
            finally:
                if xlock is not None:
                    await self._release_cross_worker(xlock)

    async def snapshot_delete(self, args: dict) -> dict[str, Any]:
        self._require_idle()
        async with self._lock:
            xlock = await self._acquire_cross_worker()
            try:
                file_id = str(args.get("file_id") or "")
                try:
                    records = await self._snapshot_records()
                except Exception as exc:
                    raise UpdateError(
                        "validation_failed", f"snapshot storage is unavailable: {exc}"
                    ) from exc
                entry = next((r for r in records if r["file_id"] == file_id), None)
                if entry is None:
                    raise UpdateError("not_found", f"{file_id} is not a known snapshot")
                if entry["sha256"] != str(args.get("sha256") or ""):
                    raise UpdateError(
                        "stale_snapshot",
                        "snapshot changed since the list was loaded; refresh and retry",
                    )
                if not await self._delete_record(entry, _snapshot_storage()):
                    raise UpdateError(
                        "validation_failed", "snapshot delete failed; refresh and try again"
                    )
                remaining = [r for r in records if r["file_id"] != file_id]
                return {"ok": True, "snapshots": await self._payload_from(remaining)}
            finally:
                await self._release_cross_worker(xlock)

    # ── auto-update loop ─────────────────────────────────────────────────────

    _UPDATE_VALVE_KEYS = (
        "PIPE_DASHBOARD_UPDATE_ENABLE",
        "PIPE_DASHBOARD_UPDATE_AUTO",
        "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS",
        "PIPE_DASHBOARD_UPDATE_REPO",
        "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP",
    )

    async def _row_valves(self) -> dict[str, Any]:
        valves = self._valves()
        merged: dict[str, Any] = {key: getattr(valves, key, None) for key in self._UPDATE_VALVE_KEYS}
        try:
            stored = await self._functions().get_function_valves_by_id(self._pipe().id)
        except Exception:
            stored = None
        if isinstance(stored, dict):
            for key in self._UPDATE_VALVE_KEYS:
                if stored.get(key) is not None:
                    merged[key] = stored[key]
        return merged

    @staticmethod
    async def _super_admin_id() -> str:
        try:
            from open_webui.models.users import Users

            admin = await Users.get_super_admin_user()
            return str(getattr(admin, "id", "") or "") or "system"
        except Exception:
            return "system"

    def _next_backoff(self, exc: UpdateError | None = None) -> float:
        import random

        reset = getattr(exc, "reset", "") if exc is not None else ""
        if reset:
            try:
                value = float(str(reset))
                delta = value if value < 1e6 else value - _now()
                return max(delta, _PD_UPDATE_BACKOFF[0]) + random.uniform(0.0, 60.0)
            except (TypeError, ValueError):
                pass
        idx = self._backoff_idx
        self._backoff_idx = idx + 1
        if idx < len(_PD_UPDATE_BACKOFF):
            return _PD_UPDATE_BACKOFF[idx]
        return _PD_UPDATE_AUTO_INTERVAL

    async def _auto_tick(self) -> float:
        interval = _PD_UPDATE_AUTO_INTERVAL
        try:
            valves = await self._row_valves()
            if not bool(valves.get("PIPE_DASHBOARD_UPDATE_ENABLE", True)):
                return interval
            if not bool(valves.get("PIPE_DASHBOARD_UPDATE_AUTO", False)):
                return interval
            snap = await self.check()
            if (snap.get("installed") or {}).get("mode") != "bundle":
                return interval
            latest = snap.get("latest")
            check_error = snap.get("last_check_error")
            if latest is None:
                if check_error:
                    err = UpdateError(
                        str(check_error.get("code") or "offline"),
                        str(check_error.get("message") or ""),
                    )
                    if check_error.get("reset"):
                        setattr(err, "reset", str(check_error["reset"]))
                    self._auto_last = {"code": err.code, "ts": _now(), "message": err.message}
                    return self._next_backoff(err)
                return interval
            if not snap.get("update_available"):
                self._backoff_idx = 0
                return interval
            if snap.get("no_matching_asset"):
                return interval
            version = str(latest.get("version") or "")
            if version in self._auto_skip:
                return interval
            delay_hours = float(valves.get("PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS") or 0)
            published = self._parse_published(latest.get("published_at"))
            if published is None or _now() < published + delay_hours * 3600.0:
                return interval
            owner = await self._super_admin_id()
            try:
                result = await self.apply(
                    {"rev": snap.get("rev")}, actor="auto", actor_id=owner, request=None
                )
                self._auto_last = {"code": "ok", "ts": _now(), "result": result}
                self._backoff_idx = 0
                return interval
            except asyncio.CancelledError:
                raise
            except UpdateError as exc:
                self._auto_last = {"code": exc.code, "ts": _now(), "message": exc.message}
                if exc.code in _TRANSIENT_CODES:
                    return self._next_backoff(exc)
                self._auto_skip[version] = {"code": exc.code, "ts": _now()}
                _pd_update_log.warning(
                    "update: auto-apply paused for %s on this worker after %s", version, exc.code
                )
                return interval
        except asyncio.CancelledError:
            raise
        except UpdateError as exc:
            self._auto_last = {"code": exc.code, "ts": _now(), "message": exc.message}
            return self._next_backoff(exc)
        except Exception:
            _pd_update_log.warning("update: auto tick failed", exc_info=True)
            return interval

    @staticmethod
    async def _sleep(seconds: float) -> None:
        await asyncio.sleep(seconds)

    @staticmethod
    def _still_leader(lease: Any) -> bool:
        try:
            return lease.redis.get(lease.lock_name) == lease.lock_id
        except Exception:
            return False

    def _renew_leader(self, lease: Any) -> bool:
        try:
            if not self._still_leader(lease):
                return False
            return bool(lease.renew_lock())
        except Exception:
            _pd_update_log.warning("update: leader lease renewal failed", exc_info=True)
            return False

    async def _lead(self, lease: Any | None) -> None:
        while True:
            delay = await self._auto_tick()
            remaining = float(delay)
            while remaining > 0:
                step = min(remaining, float(_PD_UPDATE_LEADER_RENEW_S))
                await self._sleep(step)
                remaining -= step
                if lease is not None and not await asyncio.to_thread(self._renew_leader, lease):
                    self._auto_role = "follower"
                    _pd_update_log.warning("update: leader lease lost — stepping down")
                    return

    async def run_auto_loop(self) -> None:
        import random

        lease: Any | None = None
        try:
            low, high = _PD_UPDATE_AUTO_JITTER
            await self._sleep(low + (high - low) * random.random())
            while True:
                lease = _distributed_lock("update_leader", _PD_UPDATE_LEADER_TTL_S)
                if lease is None:
                    self._auto_role = "solo"
                    await self._lead(None)
                    continue
                if not await asyncio.to_thread(lease.aquire_lock):
                    self._auto_role = "follower"
                    probe, lease = lease, None
                    self._dispose_lock(probe)
                    await self._sleep(_PD_UPDATE_FOLLOWER_POLL_S)
                    continue
                self._auto_role = "leader"
                _pd_update_log.info("update: this worker is the update leader")
                try:
                    await self._lead(lease)
                finally:
                    released, lease = lease, None
                    try:
                        released.release_lock()
                    except Exception:
                        pass
                    self._dispose_lock(released)
        except asyncio.CancelledError:
            if lease is not None:
                try:
                    lease.release_lock()
                except Exception:
                    pass
                self._dispose_lock(lease)
            return
