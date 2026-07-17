"""Tests for the pipe_dashboard self-update service (Update tab)."""

from __future__ import annotations

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import update_service as us

PID = "open_webui_openrouter_pipe"

GOOD_HEADER = (
    '"""\n'
    "title: Open WebUI OpenRouter Responses Pipe\n"
    f"id: {PID}\n"
    "required_open_webui_version: 0.9.1\n"
    "version: 2.7.0\n"
    '"""\n'
    "class Pipe:\n    pass\n"
)

INSTALLED_CONTENT = (
    '"""\n'
    f"id: {PID}\n"
    "version: 2.6.9\n"
    '"""\n'
    "_INTERMEDIATE_PACKAGES = {}\n"
    "class Pipe:\n    pass\n"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valves(**over):
    base = {
        "PIPE_DASHBOARD_UPDATE_ENABLE": True,
        "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP": 3,
        "PIPE_DASHBOARD_UPDATE_REPO": "rbb-dev/Open-WebUI-OpenRouter-pipe",
        "PIPE_DASHBOARD_UPDATE_AUTO": False,
        "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS": 168,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _pipe(**valve_over):
    return SimpleNamespace(id=PID, valves=_valves(**valve_over), _http_session=None)


class _FakeFunctions:
    def __init__(self, content=INSTALLED_CONTENT, meta=None, rev=1000):
        self.row = SimpleNamespace(content=content, meta=dict(meta or {}), updated_at=rev)
        self.updates: list[dict] = []
        self.meta_merges: list[dict] = []
        self.valves_row: dict | None = None
        self.fail_content_write = False
        self.fail_meta_merge = False

    async def get_function_valves_by_id(self, fid, db=None):
        return self.valves_row

    async def get_function_by_id(self, fid, db=None):
        return self.row if fid == PID else None

    async def update_function_by_id(self, fid, updated, db=None):
        if self.fail_content_write and "content" in updated:
            return None
        self.updates.append(dict(updated))
        for k, v in updated.items():
            setattr(self.row, k, v)
        self.row.updated_at += 1
        return self.row

    async def update_function_metadata_by_id(self, fid, metadata, db=None):
        if self.fail_meta_merge:
            return None
        self.meta_merges.append(dict(metadata))
        base = self.row.meta
        if hasattr(base, "model_dump"):
            base = base.model_dump()
        self.row.meta = {**(base or {}), **metadata}
        self.row.updated_at += 1
        return self.row


@pytest.fixture()
def fake_functions(monkeypatch):
    import open_webui.models.functions as owf

    fake = _FakeFunctions()
    monkeypatch.setattr(owf, "Functions", fake)
    return fake


@pytest.fixture()
def flat_module(monkeypatch):
    mod = types.ModuleType(f"function_{PID}")
    setattr(mod, "_INTERMEDIATE_PACKAGES", {})
    monkeypatch.setitem(sys.modules, f"function_{PID}", mod)
    return mod


@pytest.fixture()
def svc(flat_module):
    pipe = _pipe()
    service = us.UpdateService(lambda: pipe)
    return service


def _asset(content: bytes, name="open_webui_openrouter_pipe_bundled.py", digest=True):
    entry = {
        "name": name,
        "size": len(content),
        "browser_download_url": f"https://github.com/x/releases/download/v2.7.0/{name}",
    }
    if digest:
        entry["digest"] = f"sha256:{_sha(content)}"
    return entry


def _release(tag="v2.7.0", assets=None, body="## Changes\n- fix", published="2026-07-01T00:00:00Z"):
    if assets is None:
        assets = [
            _asset(GOOD_HEADER.encode(), "open_webui_openrouter_pipe_bundled.py"),
            _asset(GOOD_HEADER.encode(), "open_webui_openrouter_pipe_bundled_compressed.py"),
        ]
    return {"tag_name": tag, "published_at": published, "body": body, "assets": assets}


@pytest.fixture()
def fake_http(monkeypatch):
    state = SimpleNamespace(json_map={}, bytes_map={}, json_calls=[], bytes_calls=[])

    async def _get_json(session, url, *, timeout=15.0):
        state.json_calls.append(url)
        if url in state.json_map:
            return state.json_map[url]
        return 404, None, {}

    async def _get_bytes(session, url, *, timeout=60.0, cap=us._PD_UPDATE_SIZE_CAP):
        state.bytes_calls.append(url)
        data = state.bytes_map[url]
        if len(data) > cap:
            raise us.UpdateError("validation_failed", "size cap exceeded")
        return data

    monkeypatch.setattr(us, "_http_get_json", _get_json)
    monkeypatch.setattr(us, "_http_get_bytes", _get_bytes)
    return state


# ── detect_mode ──────────────────────────────────────────────────────────────


def test_detect_mode_flat(svc):
    assert svc.detect_mode() == {"mode": "bundle", "compressed": False}


def test_detect_mode_from_row_content(svc):
    assert svc.detect_mode(INSTALLED_CONTENT) == {"mode": "bundle", "compressed": False}
    annotated_flat = INSTALLED_CONTENT.replace(
        "_INTERMEDIATE_PACKAGES = {}",
        "_INTERMEDIATE_PACKAGES: frozenset[str] = frozenset({'api'})",
    )
    assert svc.detect_mode(annotated_flat) == {"mode": "bundle", "compressed": False}
    compressed = INSTALLED_CONTENT.replace(
        "_INTERMEDIATE_PACKAGES = {}", "_BUNDLED_SOURCES_Z: dict[str, str] = {}"
    )
    assert svc.detect_mode(compressed) == {"mode": "bundle", "compressed": True}
    stub = '"""\nid: x\nversion: 1\n"""\nfrom foo import Pipe\n'
    assert svc.detect_mode(stub)["mode"] == "package"


def test_detect_mode_against_real_built_bundles(svc):
    from pathlib import Path

    root = Path(__file__).parent.parent
    flat = root / "open_webui_openrouter_pipe_bundled.py"
    compressed = root / "open_webui_openrouter_pipe_bundled_compressed.py"
    if not flat.exists() or not compressed.exists():
        pytest.skip("built bundles not present")
    assert svc.detect_mode(flat.read_text(encoding="utf-8")) == {
        "mode": "bundle",
        "compressed": False,
    }
    assert svc.detect_mode(compressed.read_text(encoding="utf-8")) == {
        "mode": "bundle",
        "compressed": True,
    }


def test_detect_mode_content_markers_are_anchored(svc):
    embedded = (
        INSTALLED_CONTENT
        + '\n    pattern = "^_BUNDLED_SOURCES_Z\\\\s*="\n    text = "_BUNDLED_SOURCES_Z = fake"\n'
    )
    assert svc.detect_mode(embedded) == {"mode": "bundle", "compressed": False}


def test_detect_mode_compressed(monkeypatch):
    mod = types.ModuleType(f"function_{PID}")
    setattr(mod, "_BUNDLED_SOURCES_Z", {})
    monkeypatch.setitem(sys.modules, f"function_{PID}", mod)
    service = us.UpdateService(lambda: _pipe())
    assert service.detect_mode() == {"mode": "bundle", "compressed": True}


def test_detect_mode_package(monkeypatch):
    mod = types.ModuleType(f"function_{PID}")
    monkeypatch.setitem(sys.modules, f"function_{PID}", mod)
    service = us.UpdateService(lambda: _pipe())
    assert service.detect_mode()["mode"] == "package"


# ── fetch_and_validate ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_rejects_missing_digest(svc, fake_functions, fake_http):
    content = GOOD_HEADER.encode()
    asset = _asset(content, digest=False)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "digest_mismatch"


@pytest.mark.asyncio
async def test_fetch_rejects_digest_mismatch(svc, fake_functions, fake_http):
    content = GOOD_HEADER.encode()
    asset = _asset(b"other-bytes")
    asset["browser_download_url"] = "https://github.com/x/a.py"
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "digest_mismatch"


@pytest.mark.asyncio
async def test_fetch_rejects_wrong_id(svc, fake_functions, fake_http):
    content = GOOD_HEADER.replace(f"id: {PID}", "id: some_other_pipe").encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "validation_failed"


@pytest.mark.asyncio
async def test_fetch_rejects_downgrade(svc, fake_functions, fake_http):
    content = GOOD_HEADER.replace("version: 2.7.0", "version: 2.6.8").encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "validation_failed"


@pytest.mark.asyncio
async def test_fetch_allows_equal_version_for_reinstall(svc, fake_functions, fake_http, monkeypatch):
    monkeypatch.setattr(us, "MIN_UPDATE_VERSION", "2.0.0")
    content = GOOD_HEADER.replace("version: 2.7.0", "version: 2.6.9").encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    out = await svc.fetch_and_validate(asset, require_newer=True)
    assert out == content.decode("utf-8")


@pytest.mark.asyncio
async def test_fetch_rejects_pre_updater_versions(svc, fake_functions, fake_http):
    content = GOOD_HEADER.replace("version: 2.7.0", "version: 2.6.9").encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "validation_failed"
    assert "predate the self-updater" in exc.value.message


@pytest.mark.asyncio
async def test_check_latest_supported_flag(svc, fake_functions, fake_http):
    _wire_latest(fake_http, release=_release(tag="v2.6.9"))
    out = await svc.check()
    assert out["latest"]["supported"] is False
    _wire_latest(fake_http, release=_release(tag="v2.7.0"))
    out = await svc.check(force=True)
    assert out["latest"]["supported"] is True


@pytest.mark.asyncio
async def test_fetch_allows_equal_version_when_not_require_newer(svc, fake_functions, fake_http):
    content = GOOD_HEADER.replace("version: 2.7.0", "version: 2.6.9").encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    out = await svc.fetch_and_validate(asset, require_newer=False)
    assert out == content.decode("utf-8")


@pytest.mark.asyncio
async def test_fetch_rejects_incompatible_owui(svc, fake_functions, fake_http, monkeypatch):
    owe = sys.modules.get("open_webui.env")
    if owe is None:
        owe = types.ModuleType("open_webui.env")
        monkeypatch.setitem(sys.modules, "open_webui.env", owe)
    monkeypatch.setattr(owe, "VERSION", "0.9.0", raising=False)
    content = GOOD_HEADER.encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "incompatible_owui"


@pytest.mark.asyncio
async def test_fetch_rejects_oversize_preflight(svc, fake_functions, fake_http):
    content = GOOD_HEADER.encode()
    asset = _asset(content)
    asset["size"] = us._PD_UPDATE_SIZE_CAP + 1
    with pytest.raises(us.UpdateError) as exc:
        await svc.fetch_and_validate(asset, require_newer=True)
    assert exc.value.code == "validation_failed"
    assert fake_http.bytes_calls == []


@pytest.mark.asyncio
async def test_fetch_uses_owui_extract_frontmatter(svc, fake_functions, fake_http, monkeypatch):
    import open_webui.utils.plugin as owp

    calls = []
    real = owp.extract_frontmatter

    def _spy(content):
        calls.append(True)
        return real(content)

    monkeypatch.setattr(owp, "extract_frontmatter", _spy)
    content = GOOD_HEADER.encode()
    asset = _asset(content)
    fake_http.bytes_map[asset["browser_download_url"]] = content
    out = await svc.fetch_and_validate(asset, require_newer=True)
    assert out == content.decode("utf-8")
    assert calls


# ── check() ──────────────────────────────────────────────────────────────────


def _wire_latest(fake_http, repo="rbb-dev/Open-WebUI-OpenRouter-pipe", release=None):
    rel = release or _release()
    fake_http.json_map[f"https://api.github.com/repos/{repo}/releases/latest"] = (200, rel, {})
    return rel


@pytest.mark.asyncio
async def test_check_memoizes_within_window(svc, fake_functions, fake_http, monkeypatch):
    _wire_latest(fake_http)
    monkeypatch.setattr(us, "_now", lambda: 10_000.0)
    first = await svc.check()
    n_after_first = len(fake_http.json_calls)
    second = await svc.check()
    assert len(fake_http.json_calls) == n_after_first
    assert second["cached"] is True
    assert first["cached"] is False
    assert first["latest"]["version"] == "2.7.0"
    assert first["update_available"] is True


@pytest.mark.asyncio
async def test_check_force_bypasses_memo(svc, fake_functions, fake_http, monkeypatch):
    _wire_latest(fake_http)
    monkeypatch.setattr(us, "_now", lambda: 10_000.0)
    await svc.check()
    before = len([u for u in fake_http.json_calls if "releases/latest" in u])
    await svc.check(force=True)
    after = len([u for u in fake_http.json_calls if "releases/latest" in u])
    assert after == before + 1


@pytest.mark.asyncio
async def test_check_memo_keyed_by_repo(svc, fake_functions, fake_http, monkeypatch):
    _wire_latest(fake_http)
    _wire_latest(fake_http, repo="someone/fork")
    monkeypatch.setattr(us, "_now", lambda: 10_000.0)
    await svc.check()
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_REPO = "someone/fork"
    out = await svc.check()
    assert out["cached"] is False
    assert out["repo"] == "someone/fork"


@pytest.mark.asyncio
async def test_check_bad_repo_valve(svc, fake_functions, fake_http):
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_REPO = "not a repo!!"
    out = await svc.check()
    assert out["last_check_error"]["code"] == "bad_repo_valve"
    assert out["installed"]["version"] == "2.6.9"


@pytest.mark.asyncio
async def test_check_repo_not_found_keeps_installed(svc, fake_functions, fake_http):
    out = await svc.check()
    assert out["last_check_error"]["code"] == "repo_not_found"
    assert out["installed"]["version"] == "2.6.9"
    assert out["latest"] is None


@pytest.mark.asyncio
async def test_check_rate_limited_maps_from_403(svc, fake_functions, fake_http):
    repo = "rbb-dev/Open-WebUI-OpenRouter-pipe"
    fake_http.json_map[f"https://api.github.com/repos/{repo}/releases/latest"] = (
        403,
        None,
        {"X-RateLimit-Reset": "999999"},
    )
    out = await svc.check()
    assert out["last_check_error"]["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_check_failure_keeps_last_known_latest(svc, fake_functions, fake_http, monkeypatch):
    _wire_latest(fake_http)
    clock = SimpleNamespace(t=10_000.0)
    monkeypatch.setattr(us, "_now", lambda: clock.t)
    ok = await svc.check()
    assert ok["latest"]["version"] == "2.7.0"
    repo = "rbb-dev/Open-WebUI-OpenRouter-pipe"
    del fake_http.json_map[f"https://api.github.com/repos/{repo}/releases/latest"]
    clock.t += 120.0
    out = await svc.check(force=True)
    assert out["latest"]["version"] == "2.7.0"
    assert out["last_check_error"]["code"] == "repo_not_found"
    assert out["checked_at"] == 10_000.0


@pytest.mark.asyncio
async def test_check_notes_capped_and_assets_split(svc, fake_functions, fake_http):
    rel = _release(body="x" * (us._PD_UPDATE_NOTES_CAP + 100))
    _wire_latest(fake_http, release=rel)
    out = await svc.check()
    assert len(out["latest"]["notes"]) == us._PD_UPDATE_NOTES_CAP
    assert out["latest"]["assets"]["flat"]["name"] == "open_webui_openrouter_pipe_bundled.py"
    assert out["latest"]["assets"]["compressed"]["name"] == (
        "open_webui_openrouter_pipe_bundled_compressed.py"
    )


@pytest.mark.asyncio
async def test_check_no_matching_asset_for_installed_variant(svc, fake_functions, fake_http):
    rel = _release(assets=[_asset(GOOD_HEADER.encode(), "open_webui_openrouter_pipe_bundled_compressed.py")])
    _wire_latest(fake_http, release=rel)
    out = await svc.check()
    assert out["no_matching_asset"] is True
    assert out["latest"]["assets"]["flat"] is None


@pytest.mark.asyncio
async def test_check_installed_carries_no_derived_dates(svc, fake_functions, fake_http):
    _wire_latest(fake_http)
    out = await svc.check()
    assert "released" not in out["installed"]
    assert "build" not in out["installed"]
    assert not any("releases/tags" in url for url in fake_http.json_calls)


@pytest.mark.asyncio
async def test_check_disabled_flag(svc, fake_functions, fake_http):
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_ENABLE = False
    out = await svc.check()
    assert out["enabled"] is False


# ── snapshot_current ─────────────────────────────────────────────────────────


def _slot_id(i: int) -> str:
    return f"pipe-update-snapshot-{PID}-{i}"


def _seed_slot(store, i, *, version="2.5.0", sha="s", ts=1.0, actor="admin", size=10):
    fid = _slot_id(i)
    path = f"/store/{PID}-{version}-slot{i}.py"
    store.rows[fid] = SimpleNamespace(
        id=fid,
        user_id="owner",
        filename=f"{PID}-{version}.py",
        path=path,
        meta={
            "update_snapshot": {
                "from_version": version,
                "sha256": sha,
                "ts": ts,
                "actor": actor,
                "size": size,
                "slot": i,
            }
        },
    )
    return fid


@pytest.fixture()
def fake_storage(monkeypatch):
    from open_webui_openrouter_pipe.storage import owui_files as of

    store = SimpleNamespace(
        uploads=[],
        deleted=[],
        blobs={},
        rows={},
        order=[],
        fail_delete=False,
        fail_row_delete=False,
        insert_none=False,
    )

    class _Storage:
        @staticmethod
        def upload_file(fobj, filename, tags):
            data = fobj.read()
            path = f"/store/{filename}"
            store.blobs[path] = data
            store.uploads.append(filename)
            store.order.append(("upload", path))
            return data, path

        @staticmethod
        def delete_file(path):
            if store.fail_delete:
                raise RuntimeError("boom")
            store.deleted.append(path)
            store.order.append(("del_blob", path))
            store.blobs.pop(path, None)

    class _Files:
        @staticmethod
        async def insert_new_file(user_id, form, db=None):
            if store.insert_none or form.id in store.rows:
                return None
            rec = SimpleNamespace(
                id=form.id,
                user_id=user_id,
                filename=form.filename,
                path=form.path,
                meta=dict(form.meta or {}),
            )
            store.rows[form.id] = rec
            store.order.append(("insert", form.id))
            return rec

        @staticmethod
        async def get_file_by_id(fid, db=None):
            return store.rows.get(fid)

        @staticmethod
        async def get_files_by_ids(ids, db=None):
            return [store.rows[i] for i in ids if i in store.rows]

        @staticmethod
        async def delete_file_by_id(fid, db=None):
            if store.fail_row_delete:
                return False
            store.rows.pop(fid, None)
            store.deleted.append(fid)
            store.order.append(("del_row", fid))
            return True

    monkeypatch.setattr(of, "get_owui_storage", lambda: _Storage)
    monkeypatch.setattr(us, "_files_model", lambda: _Files)
    store.Files = _Files
    return store


@pytest.mark.asyncio
async def test_snapshot_dedupe_skips_upload(svc, fake_functions, fake_storage):
    sha = _sha(INSTALLED_CONTENT.encode())
    _seed_slot(fake_storage, 0, version="2.6.9", sha=sha, ts=1.0)
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(0)
    assert fake_storage.uploads == []
    assert fake_storage.order == []
    assert fake_functions.meta_merges == []


@pytest.mark.asyncio
async def test_snapshot_dedupe_matches_any_slot_not_just_newest(svc, fake_functions, fake_storage):
    sha = _sha(INSTALLED_CONTENT.encode())
    _seed_slot(fake_storage, 0, version="2.6.9", sha=sha, ts=1.0)
    _seed_slot(fake_storage, 1, version="2.6.8", sha="different", ts=99.0)
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(0)
    assert fake_storage.uploads == []
    assert fake_storage.order == []


@pytest.mark.asyncio
async def test_snapshot_writes_slot_record_not_function_meta(svc, fake_functions, fake_storage):
    rev_before = fake_functions.row.updated_at
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(0)
    record = fake_storage.rows[file_id]
    snap = record.meta["update_snapshot"]
    assert snap["actor"] == "admin"
    assert snap["from_version"] == "2.6.9"
    assert snap["sha256"] == _sha(INSTALLED_CONTENT.encode())
    assert snap["slot"] == 0
    assert record.user_id == "u1"
    assert fake_functions.meta_merges == []
    assert fake_functions.row.updated_at == rev_before


@pytest.mark.asyncio
async def test_snapshot_prunes_oldest_row_then_blob(svc, fake_functions, fake_storage):
    for i in range(3):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    oldest_path = fake_storage.rows[_slot_id(0)].path
    await svc.snapshot_current("admin", "u1")
    assert _slot_id(0) not in fake_storage.rows
    assert set(fake_storage.rows) == {_slot_id(1), _slot_id(2), _slot_id(3)}
    row_at = fake_storage.order.index(("del_row", _slot_id(0)))
    blob_at = fake_storage.order.index(("del_blob", oldest_path))
    assert row_at < blob_at


@pytest.mark.asyncio
async def test_snapshot_full_rotation_deletes_old_blob_last(svc, fake_functions, fake_storage):
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP = 10
    for i in range(10):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    oldest_path = fake_storage.rows[_slot_id(0)].path
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(0)
    ops = fake_storage.order
    upload_at = next(i for i, op in enumerate(ops) if op[0] == "upload")
    row_at = ops.index(("del_row", _slot_id(0)))
    insert_at = ops.index(("insert", _slot_id(0)))
    blob_at = ops.index(("del_blob", oldest_path))
    assert upload_at < row_at < insert_at < blob_at


@pytest.mark.asyncio
async def test_snapshot_keep_shrink_stays_visible_until_next_write(
    svc, fake_functions, fake_http, fake_storage
):
    for i in range(5):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    _wire_latest(fake_http)
    out = await svc.check()
    assert len(out["snapshots"]) == 5
    await svc.snapshot_current("admin", "u1")
    assert len(fake_storage.rows) == 3


@pytest.mark.asyncio
async def test_snapshot_foreign_occupant_untouched_and_unlisted(
    svc, fake_functions, fake_http, fake_storage
):
    fake_storage.rows[_slot_id(0)] = SimpleNamespace(
        id=_slot_id(0), user_id="someone", filename="theirs.bin", path="/store/theirs.bin", meta={}
    )
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP = 1
    _seed_slot(fake_storage, 1, sha="old", ts=1.0)
    _wire_latest(fake_http)
    out = await svc.check()
    assert [s["file_id"] for s in out["snapshots"]] == [_slot_id(1)]
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(2)
    assert _slot_id(0) in fake_storage.rows
    assert fake_storage.rows[_slot_id(0)].meta == {}
    assert _slot_id(1) not in fake_storage.rows


@pytest.mark.asyncio
async def test_snapshot_sort_uses_slot_tiebreak(svc, fake_functions, fake_http, fake_storage):
    _seed_slot(fake_storage, 3, sha="a", ts=5.0)
    _seed_slot(fake_storage, 1, sha="b", ts=5.0)
    _seed_slot(fake_storage, 2, sha="c", ts=4.0)
    _wire_latest(fake_http)
    out = await svc.check()
    assert [s["file_id"] for s in out["snapshots"]] == [_slot_id(3), _slot_id(1), _slot_id(2)]


@pytest.mark.asyncio
async def test_snapshot_insert_none_cleans_blob_and_aborts(svc, fake_functions, fake_storage):
    fake_storage.insert_none = True
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_current("admin", "u1")
    assert exc.value.code == "validation_failed"
    assert any(op[0] == "del_blob" for op in fake_storage.order)
    assert fake_storage.rows == {}


@pytest.mark.asyncio
async def test_snapshot_tolerates_blob_delete_failure(svc, fake_functions, fake_storage):
    for i in range(3):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    fake_storage.fail_delete = True
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(3)
    assert _slot_id(0) not in fake_storage.rows
    assert len(fake_storage.rows) == 3


@pytest.mark.asyncio
async def test_prune_row_delete_refusal_keeps_blob(svc, fake_functions, fake_storage):
    for i in range(3):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    fake_storage.fail_row_delete = True
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(3)
    assert _slot_id(0) in fake_storage.rows
    assert not any(op[0] == "del_blob" for op in fake_storage.order)
    assert len(fake_storage.rows) == 4


@pytest.mark.asyncio
async def test_rotation_insert_failure_frees_both_blobs(svc, fake_functions, fake_storage):
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP = 10
    for i in range(10):
        _seed_slot(fake_storage, i, sha=f"s{i}", ts=float(i + 1))
    oldest_path = fake_storage.rows[_slot_id(0)].path
    fake_storage.insert_none = True
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_current("admin", "u1")
    assert exc.value.code == "validation_failed"
    assert any(op == ("del_blob", oldest_path) for op in fake_storage.order)
    new_blob = next(p for (op, p) in fake_storage.order if op == "upload")
    assert any(op == ("del_blob", new_blob) for op in fake_storage.order)
    assert _slot_id(0) not in fake_storage.rows
    assert len(fake_storage.rows) == 9


@pytest.mark.asyncio
async def test_future_ts_snapshot_does_not_outrank_new_writes(
    svc, fake_functions, fake_http, fake_storage, monkeypatch
):
    monkeypatch.setattr(us, "_now", lambda: 1000.0)
    _seed_slot(fake_storage, 0, sha="future", ts=999_999.0)
    file_id = await svc.snapshot_current("admin", "u1")
    rec = fake_storage.rows[file_id].meta["update_snapshot"]
    assert rec["ts"] > 999_999.0
    _wire_latest(fake_http)
    out = await svc.check()
    assert out["snapshots"][0]["file_id"] == file_id


@pytest.mark.asyncio
async def test_snapshot_actor_names_resolved(svc, fake_functions, fake_http, fake_storage, monkeypatch):
    import open_webui.models.users as owu

    class _Users:
        @staticmethod
        async def get_user_by_id(uid, db=None):
            return SimpleNamespace(id=uid, name="Dev Admin") if uid == "u-123" else None

    monkeypatch.setattr(owu, "Users", _Users, raising=False)
    _seed_slot(fake_storage, 0, version="2.6.0", sha="a", ts=1.0, actor="u-123")
    _seed_slot(fake_storage, 1, version="2.6.5", sha="b", ts=2.0, actor="auto")
    _wire_latest(fake_http)
    out = await svc.check()
    by_id = {s["file_id"]: s for s in out["snapshots"]}
    assert by_id[_slot_id(0)]["actor_name"] == "Dev Admin"
    assert by_id[_slot_id(1)]["actor_name"] is None
    assert by_id[_slot_id(1)]["actor"] == "auto"


@pytest.mark.asyncio
async def test_editor_meta_wipe_leaves_snapshots_intact(svc, fake_functions, fake_http, fake_storage):
    _seed_slot(fake_storage, 0, version="2.6.0", sha="a", ts=1.0)
    _seed_slot(fake_storage, 1, version="2.6.5", sha="b", ts=2.0)
    fake_functions.row.meta = {"manifest": {"version": "2.6.9"}}
    _wire_latest(fake_http)
    out = await svc.check()
    assert [s["file_id"] for s in out["snapshots"]] == [_slot_id(1), _slot_id(0)]


# ── _reload_via_loader ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reload_failure_repairs_sys_modules_and_is_active(svc, fake_functions, flat_module, monkeypatch):
    import open_webui.utils.plugin as owp

    async def _boom(fid, content=None):
        sys.modules[f"function_{PID}"] = types.ModuleType(f"function_{PID}")
        del sys.modules[f"function_{PID}"]
        await owp_functions_set_inactive()
        raise RuntimeError("ImportError: cannot import name X")

    import open_webui.models.functions as owf

    async def owp_functions_set_inactive():
        await owf.Functions.update_function_by_id(PID, {"is_active": False})

    monkeypatch.setattr(owp, "load_function_module_by_id", _boom)
    with pytest.raises(us.UpdateError) as exc:
        await svc._reload_via_loader(GOOD_HEADER)
    assert exc.value.code == "exec_failed"
    assert "ImportError" in str(exc.value)
    assert sys.modules[f"function_{PID}"] is flat_module
    assert {"is_active": True} in fake_functions.updates
    assert not any("content" in u for u in fake_functions.updates)


@pytest.mark.asyncio
async def test_reload_success_passes_replace_imports_content(svc, fake_functions, monkeypatch):
    import open_webui.utils.plugin as owp

    received = {}

    async def _load(fid, content=None):
        received["content"] = content
        return SimpleNamespace(name="instance"), "pipe", {"version": "2.7.0"}

    monkeypatch.setattr(owp, "load_function_module_by_id", _load)
    monkeypatch.setattr(owp, "replace_imports", lambda c: c + "\n# normalized")
    instance, frontmatter, content = await svc._reload_via_loader(GOOD_HEADER)
    assert received["content"] == GOOD_HEADER + "\n# normalized"
    assert content == GOOD_HEADER + "\n# normalized"
    assert frontmatter == {"version": "2.7.0"}
    assert instance.name == "instance"


# ── apply / restore / snapshot_delete ────────────────────────────────────────


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


@pytest.fixture()
def wired(svc, fake_functions, fake_http, fake_storage, monkeypatch):
    import open_webui.utils.plugin as owp

    events: list[str] = []
    rel = _wire_latest(fake_http)
    for asset in rel["assets"]:
        fake_http.bytes_map[asset["browser_download_url"]] = GOOD_HEADER.encode()

    orig_bytes = us._http_get_bytes

    async def _bytes_spy(session, url, **kw):
        events.append("download")
        return await orig_bytes(session, url, **kw)

    monkeypatch.setattr(us, "_http_get_bytes", _bytes_spy)

    async def _load(fid, content=None):
        events.append("loader")
        return SimpleNamespace(name="new-instance"), "pipe", owp.extract_frontmatter(content or "")

    monkeypatch.setattr(owp, "load_function_module_by_id", _load)
    return SimpleNamespace(events=events, functions=fake_functions, http=fake_http, storage=fake_storage)


@pytest.mark.asyncio
async def test_apply_happy_path_order_and_caches(svc, wired):
    req = _request()
    rev = wired.functions.row.updated_at
    out = await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=req)
    assert out["ok"] is True
    assert out["from_version"] == "2.6.9"
    assert out["to_version"] == "2.7.0"
    content_updates = [u for u in wired.functions.updates if "content" in u]
    assert len(content_updates) == 1
    written = content_updates[0]["content"]
    assert written.startswith('"""')
    assert wired.events == ["download", "loader"]
    merges = [m for m in wired.functions.meta_merges if "manifest" in m]
    assert merges and merges[0]["manifest"]["version"] == "2.7.0"
    assert req.app.state.FUNCTIONS[PID].name == "new-instance"
    assert req.app.state.FUNCTION_CONTENTS[PID] == written
    snaps = [r.meta["update_snapshot"] for r in wired.storage.rows.values()]
    assert len(snaps) == 1
    assert snaps[0]["actor"] == "admin"
    assert snaps[0]["from_version"] == "2.6.9"


@pytest.mark.asyncio
async def test_apply_package_mode_rejected(fake_functions, fake_http, monkeypatch):
    fake_functions.row.content = f'"""\nid: {PID}\nversion: 2.6.9\n"""\nfrom pkg import Pipe\n'
    mod = types.ModuleType(f"function_{PID}")
    monkeypatch.setitem(sys.modules, f"function_{PID}", mod)
    service = us.UpdateService(lambda: _pipe())
    with pytest.raises(us.UpdateError) as exc:
        await service.apply({"rev": 1000}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "package_mode"


@pytest.mark.asyncio
async def test_apply_stale_rev_at_entry(svc, wired):
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": 42}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "stale_rev"
    assert wired.events == []


@pytest.mark.asyncio
async def test_apply_rev_recheck_trips_after_download(svc, wired, monkeypatch):
    orig = us._http_get_bytes

    async def _bump_during_download(session, url, **kw):
        wired.functions.row.updated_at += 7
        return await orig(session, url, **kw)

    monkeypatch.setattr(us, "_http_get_bytes", _bump_during_download)
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "stale_rev"
    assert "loader" not in wired.events
    assert not any("content" in u for u in wired.functions.updates)


@pytest.mark.asyncio
async def test_apply_rev_recheck_trips_after_loader(svc, wired, monkeypatch):
    import open_webui.utils.plugin as owp

    async def _load_and_bump(fid, content=None):
        wired.events.append("loader")
        wired.functions.row.updated_at += 7
        return SimpleNamespace(name="new-instance"), "pipe", {"version": "2.7.0"}

    monkeypatch.setattr(owp, "load_function_module_by_id", _load_and_bump)
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "stale_rev"
    assert not any("content" in u for u in wired.functions.updates)


@pytest.mark.asyncio
async def test_apply_request_none_requires_auto_actor(svc, wired):
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=None)
    assert exc.value.code == "internal"
    assert wired.events == []


@pytest.mark.asyncio
async def test_apply_auto_skips_caches_and_forces_variant(svc, wired):
    rev = wired.functions.row.updated_at
    out = await svc.apply(
        {"rev": rev, "compressed": True}, actor="auto", actor_id="sys", request=None
    )
    assert out["ok"] is True
    flat_url = [u for u in wired.http.bytes_calls if u.endswith("open_webui_openrouter_pipe_bundled.py")]
    assert flat_url
    assert not any(
        u.endswith("_compressed.py") for u in wired.http.bytes_calls
    )
    snaps = [r.meta["update_snapshot"] for r in wired.storage.rows.values()]
    assert snaps and snaps[-1]["actor"] == "auto"


@pytest.mark.asyncio
async def test_apply_manual_compressed_override(svc, wired):
    rev = wired.functions.row.updated_at
    await svc.apply({"rev": rev, "compressed": True}, actor="admin", actor_id="u1", request=_request())
    assert any(u.endswith("_compressed.py") for u in wired.http.bytes_calls)


@pytest.mark.asyncio
async def test_apply_no_matching_asset_for_selected_variant(svc, fake_functions, fake_http, fake_storage):
    rel = _release(assets=[_asset(GOOD_HEADER.encode(), "open_webui_openrouter_pipe_bundled_compressed.py")])
    _wire_latest(fake_http, release=rel)
    rev = fake_functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "no_matching_asset"


@pytest.mark.asyncio
async def test_apply_lock_busy_immediate(svc, wired):
    async with svc._lock:
        with pytest.raises(us.UpdateError) as exc:
            await svc.apply(
                {"rev": wired.functions.row.updated_at},
                actor="admin",
                actor_id="u1",
                request=_request(),
            )
    assert exc.value.code == "update_in_progress"


@pytest.mark.asyncio
async def test_restore_unknown_file_id(svc, wired):
    with pytest.raises(us.UpdateError) as exc:
        await svc.restore(
            {"rev": wired.functions.row.updated_at, "file_id": "nope"},
            actor="admin",
            actor_id="u1",
            request=_request(),
        )
    assert exc.value.code == "not_found"


@pytest.mark.asyncio
async def test_restore_blob_hash_mismatch_never_loads(svc, wired, monkeypatch):
    blob = GOOD_HEADER.encode()
    _seed_slot(wired.storage, 0, version="2.5.0", sha="deadbeef", ts=1.0)

    async def _read(self, file_id):
        return blob

    monkeypatch.setattr(us.UpdateService, "_read_blob", _read)
    with pytest.raises(us.UpdateError) as exc:
        await svc.restore(
            {"rev": wired.functions.row.updated_at, "file_id": _slot_id(0)},
            actor="admin",
            actor_id="u1",
            request=_request(),
        )
    assert exc.value.code == "digest_mismatch"
    assert "loader" not in wired.events


@pytest.mark.asyncio
async def test_restore_happy_allows_older_version(svc, wired, monkeypatch):
    old_content = INSTALLED_CONTENT.replace("version: 2.6.9", "version: 2.5.0")
    blob = old_content.encode()
    _seed_slot(wired.storage, 0, version="2.5.0", sha=_sha(blob), ts=1.0)

    async def _read(self, file_id):
        return blob

    monkeypatch.setattr(us.UpdateService, "_read_blob", _read)
    req = _request()
    out = await svc.restore(
        {"rev": wired.functions.row.updated_at, "file_id": _slot_id(0)},
        actor="admin",
        actor_id="u1",
        request=req,
    )
    assert out["ok"] is True
    assert out["to_version"] == "2.5.0"
    assert "loader" in wired.events
    content_updates = [u for u in wired.functions.updates if "content" in u]
    assert len(content_updates) == 1
    snaps = [r.meta["update_snapshot"] for r in wired.storage.rows.values()]
    assert any(s["from_version"] == "2.6.9" for s in snaps)


@pytest.mark.asyncio
async def test_snapshot_delete_unknown(svc, wired):
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_delete({"file_id": "nope", "sha256": "x"})
    assert exc.value.code == "not_found"


@pytest.mark.asyncio
async def test_snapshot_delete_removes_row_then_blob(svc, wired):
    _seed_slot(wired.storage, 0, version="2.5.0", sha="s", ts=1.0)
    _seed_slot(wired.storage, 1, version="2.6.0", sha="t", ts=2.0)
    path_a = wired.storage.rows[_slot_id(0)].path
    out = await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "s"})
    assert set(wired.storage.rows) == {_slot_id(1)}
    row_at = wired.storage.order.index(("del_row", _slot_id(0)))
    blob_at = wired.storage.order.index(("del_blob", path_a))
    assert row_at < blob_at
    assert [s["file_id"] for s in out["snapshots"]] == [_slot_id(1)]


@pytest.mark.asyncio
async def test_snapshot_delete_sha_mismatch_is_stale(svc, wired):
    _seed_slot(wired.storage, 0, version="2.5.0", sha="real", ts=1.0)
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "other"})
    assert exc.value.code == "stale_snapshot"
    assert _slot_id(0) in wired.storage.rows


@pytest.mark.asyncio
async def test_snapshot_delete_row_refusal_is_typed_and_keeps_blob(svc, wired):
    _seed_slot(wired.storage, 0, sha="s", ts=1.0)
    wired.storage.fail_row_delete = True
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "s"})
    assert exc.value.code == "validation_failed"
    assert _slot_id(0) in wired.storage.rows
    assert not any(op[0] == "del_blob" for op in wired.storage.order)


@pytest.mark.asyncio
async def test_snapshot_delete_requires_idle_lock(svc, wired):
    _seed_slot(wired.storage, 0, sha="s", ts=1.0)
    async with svc._lock:
        with pytest.raises(us.UpdateError) as exc:
            await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "s"})
    assert exc.value.code == "update_in_progress"
    assert _slot_id(0) in wired.storage.rows


@pytest.mark.asyncio
async def test_snapshot_delete_holds_cross_worker_lease(svc, wired, monkeypatch):
    _seed_slot(wired.storage, 0, sha="s", ts=1.0)
    contended = _FakeXLock(available=False)
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: contended)
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "s"})
    assert exc.value.code == "update_in_progress"
    assert _slot_id(0) in wired.storage.rows
    granted = _FakeXLock()
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: granted)
    out = await svc.snapshot_delete({"file_id": _slot_id(0), "sha256": "s"})
    assert out["ok"] is True
    assert granted.acquired == 1
    assert granted.released == 1


# ── auto-update loop ─────────────────────────────────────────────────────────


def _auto_valves_row(**over):
    row = {
        "PIPE_DASHBOARD_UPDATE_ENABLE": True,
        "PIPE_DASHBOARD_UPDATE_AUTO": True,
        "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS": 0,
        "PIPE_DASHBOARD_UPDATE_REPO": "rbb-dev/Open-WebUI-OpenRouter-pipe",
        "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP": 3,
    }
    row.update(over)
    return row


@pytest.fixture()
def auto(svc, wired, monkeypatch):
    wired.functions.valves_row = _auto_valves_row()
    applied: list[dict] = []

    async def _apply_spy(args, *, actor, actor_id, request):
        applied.append({"args": dict(args), "actor": actor, "actor_id": actor_id, "request": request})
        return {"ok": True, "from_version": "2.6.9", "to_version": "2.7.0"}

    monkeypatch.setattr(svc, "apply", _apply_spy)
    import open_webui.models.users as owu

    class _Users:
        @staticmethod
        async def get_super_admin_user():
            return SimpleNamespace(id="sys-admin")

    monkeypatch.setattr(owu, "Users", _Users, raising=False)
    return SimpleNamespace(applied=applied, svc=svc, wired=wired)


@pytest.mark.asyncio
async def test_auto_tick_applies_with_super_admin_owner(auto):
    delay = await auto.svc._auto_tick()
    assert len(auto.applied) == 1
    call = auto.applied[0]
    assert call["actor"] == "auto"
    assert call["actor_id"] == "sys-admin"
    assert call["request"] is None
    assert delay == us._PD_UPDATE_AUTO_INTERVAL


@pytest.mark.asyncio
async def test_auto_tick_honors_row_valves_over_ctx(auto):
    auto.svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_AUTO = False
    await auto.svc._auto_tick()
    assert len(auto.applied) == 1


@pytest.mark.asyncio
async def test_auto_enable_gate_overrides_auto(auto):
    auto.wired.functions.valves_row = _auto_valves_row(PIPE_DASHBOARD_UPDATE_ENABLE=False)
    before = len(auto.wired.http.json_calls)
    delay = await auto.svc._auto_tick()
    assert auto.applied == []
    assert len(auto.wired.http.json_calls) == before
    assert delay == us._PD_UPDATE_AUTO_INTERVAL


@pytest.mark.asyncio
async def test_auto_delay_gate_blocks_until_eligible(auto, monkeypatch):
    auto.wired.functions.valves_row = _auto_valves_row(PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS=168)
    published = us.UpdateService._parse_published("2026-07-01T00:00:00Z")
    assert published is not None
    monkeypatch.setattr(us, "_now", lambda: published + 3600.0)
    await auto.svc._auto_tick()
    assert auto.applied == []
    monkeypatch.setattr(us, "_now", lambda: published + 169 * 3600.0)
    await auto.svc._auto_tick()
    assert len(auto.applied) == 1


@pytest.mark.asyncio
async def test_auto_deterministic_failure_pauses_version(auto, monkeypatch):
    async def _fail(args, **kw):
        raise us.UpdateError("exec_failed", "boom")

    monkeypatch.setattr(auto.svc, "apply", _fail)
    delay = await auto.svc._auto_tick()
    assert auto.svc._auto_skip["2.7.0"]["code"] == "exec_failed"
    assert delay == us._PD_UPDATE_AUTO_INTERVAL
    monkeypatch.setattr(auto.svc, "apply", lambda *a, **k: pytest.fail("must not retry paused version"))
    await auto.svc._auto_tick()


@pytest.mark.asyncio
async def test_auto_transient_failure_backs_off_never_skips(auto, monkeypatch):
    async def _fail(args, **kw):
        raise us.UpdateError("rate_limited", "slow down")

    monkeypatch.setattr(auto.svc, "apply", _fail)
    delays = [await auto.svc._auto_tick() for _ in range(5)]
    assert delays[:4] == list(us._PD_UPDATE_BACKOFF)
    assert delays[4] == us._PD_UPDATE_AUTO_INTERVAL
    assert auto.svc._auto_skip == {}


@pytest.mark.asyncio
async def test_auto_rate_limit_reset_schedules_from_header(auto, monkeypatch):
    base = us.UpdateService._parse_published("2026-07-01T00:00:00Z")
    assert base is not None
    monkeypatch.setattr(us, "_now", lambda: base + 1000.0)

    async def _fail(args, **kw):
        err = us.UpdateError("rate_limited", "slow down")
        setattr(err, "reset", str(base + 6000.0))
        raise err

    monkeypatch.setattr(auto.svc, "apply", _fail)
    delay = await auto.svc._auto_tick()
    assert 5000.0 <= delay <= 5060.0


@pytest.mark.asyncio
async def test_auto_check_level_rate_limit_reset_reaches_backoff(auto, monkeypatch):
    base = us.UpdateService._parse_published("2026-07-01T00:00:00Z")
    assert base is not None
    monkeypatch.setattr(us, "_now", lambda: base + 1000.0)
    repo = "rbb-dev/Open-WebUI-OpenRouter-pipe"
    auto.svc._last_good = None
    auto.wired.http.json_map[f"https://api.github.com/repos/{repo}/releases/latest"] = (
        403,
        None,
        {"X-RateLimit-Reset": str(base + 6000.0)},
    )
    delay = await auto.svc._auto_tick()
    assert 5000.0 <= delay <= 5060.0
    assert auto.svc._auto_skip == {}


@pytest.mark.asyncio
async def test_auto_success_resets_backoff(auto, monkeypatch):
    async def _fail(args, **kw):
        raise us.UpdateError("offline", "net down")

    ok = auto.svc.apply
    monkeypatch.setattr(auto.svc, "apply", _fail)
    await auto.svc._auto_tick()
    monkeypatch.setattr(auto.svc, "apply", ok)
    delay = await auto.svc._auto_tick()
    assert delay == us._PD_UPDATE_AUTO_INTERVAL
    monkeypatch.setattr(auto.svc, "apply", _fail)
    assert await auto.svc._auto_tick() == us._PD_UPDATE_BACKOFF[0]


@pytest.mark.asyncio
async def test_auto_no_matching_asset_is_benign(auto, fake_http):
    rel = _release(assets=[_asset(GOOD_HEADER.encode(), "open_webui_openrouter_pipe_bundled_compressed.py")])
    _wire_latest(fake_http, release=rel)
    await auto.svc.check(force=True)
    delay = await auto.svc._auto_tick()
    assert auto.applied == []
    assert auto.svc._auto_skip == {}
    assert delay == us._PD_UPDATE_AUTO_INTERVAL


@pytest.mark.asyncio
async def test_auto_manual_apply_clears_skip_map(svc, wired):
    svc._auto_skip["2.7.0"] = {"code": "exec_failed", "ts": 1.0}
    rev = wired.functions.row.updated_at
    await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert svc._auto_skip == {}


@pytest.mark.asyncio
async def test_auto_cancellation_after_loader_swap_still_commits(svc, wired, monkeypatch):
    import asyncio

    import open_webui.utils.plugin as owp

    wired.functions.valves_row = _auto_valves_row()
    monkeypatch.setattr(us, "_PD_UPDATE_AUTO_JITTER", (0.0, 0.0))
    import open_webui.models.users as owu

    class _Users:
        @staticmethod
        async def get_super_admin_user():
            return SimpleNamespace(id="sys-admin")

    monkeypatch.setattr(owu, "Users", _Users, raising=False)
    holder: dict = {}

    async def _load_and_cancel(fid, content=None):
        wired.events.append("loader")
        holder["task"].cancel()
        await asyncio.sleep(0)
        return SimpleNamespace(name="ni"), "pipe", owp.extract_frontmatter(content or "")

    monkeypatch.setattr(owp, "load_function_module_by_id", _load_and_cancel)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    holder["task"] = task
    await asyncio.wait([task], timeout=5.0)
    assert task.done()
    for _ in range(50):
        if any("content" in u for u in wired.functions.updates):
            break
        await asyncio.sleep(0.01)
    assert any("content" in u for u in wired.functions.updates)


@pytest.mark.asyncio
async def test_check_repo_is_default_flag(svc, fake_functions, fake_http):
    _wire_latest(fake_http)
    out = await svc.check()
    assert out["repo_is_default"] is True
    _wire_latest(fake_http, repo="someone/fork")
    svc._get_pipe().valves.PIPE_DASHBOARD_UPDATE_REPO = "someone/fork"
    out = await svc.check(force=True)
    assert out["repo_is_default"] is False


@pytest.mark.asyncio
async def test_check_empty_body_gives_empty_notes(svc, fake_functions, fake_http):
    _wire_latest(fake_http, release=_release(body=""))
    out = await svc.check()
    assert out["latest"]["notes"] == ""


@pytest.mark.asyncio
async def test_check_auto_last_success_from_slots(svc, fake_functions, fake_http, fake_storage):
    _wire_latest(fake_http)
    _seed_slot(fake_storage, 0, version="2.6.0", sha="a", ts=5.0, actor="admin")
    _seed_slot(fake_storage, 1, version="2.6.5", sha="b", ts=9.0, actor="auto")
    out = await svc.check()
    assert out["auto"]["last_success"] == {
        "from_version": "2.6.5",
        "to_version": "2.6.9",
        "ts": 9.0,
    }


@pytest.mark.asyncio
async def test_auto_yanked_release_never_applies_never_pauses(auto, monkeypatch):
    auto.wired.functions.valves_row = _auto_valves_row(PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS=168)
    published = us.UpdateService._parse_published("2026-07-01T00:00:00Z")
    assert published is not None
    clock = SimpleNamespace(t=published + 3600.0)
    monkeypatch.setattr(us, "_now", lambda: clock.t)
    await auto.svc._auto_tick()
    assert auto.applied == []
    _wire_latest(auto.wired.http, release=_release(tag="v2.6.9"))
    clock.t += 200.0
    await auto.svc._auto_tick()
    assert auto.applied == []
    assert auto.svc._auto_skip == {}


@pytest.mark.asyncio
async def test_http_get_bytes_streaming_cap_aborts():
    class _Content:
        async def iter_chunked(self, size):
            for _ in range(3):
                yield b"x" * (us._PD_UPDATE_SIZE_CAP // 2)

    class _Resp:
        status = 200
        headers = {}
        content = _Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        closed = False

        def get(self, url, **kw):
            return _Resp()

    with pytest.raises(us.UpdateError) as exc:
        await us._http_get_bytes(_Session(), "https://x/y.py")
    assert exc.value.code == "validation_failed"


@pytest.mark.asyncio
async def test_pydantic_meta_object_row_works(svc, fake_functions, fake_http, fake_storage):
    from pydantic import BaseModel, ConfigDict

    class _Meta(BaseModel):
        model_config = ConfigDict(extra="allow")
        manifest: dict | None = {}

    sha = _sha(INSTALLED_CONTENT.encode())
    fake_functions.row.meta = _Meta.model_validate({"manifest": {"version": "2.6.9"}})
    _seed_slot(fake_storage, 0, version="2.6.9", sha=sha, ts=1.0)
    _wire_latest(fake_http)
    out = await svc.check()
    assert out["snapshots"][0]["file_id"] == _slot_id(0)
    file_id = await svc.snapshot_current("admin", "u1")
    assert file_id == _slot_id(0)
    assert fake_storage.uploads == []


@pytest.mark.asyncio
async def test_garbage_slot_meta_skipped(svc, fake_functions, fake_http, fake_storage):
    fake_storage.rows[_slot_id(0)] = SimpleNamespace(
        id=_slot_id(0), user_id="x", filename="f", path="/store/f",
        meta={"update_snapshot": "junk-not-a-dict"},
    )
    fake_storage.rows[_slot_id(1)] = SimpleNamespace(
        id=_slot_id(1), user_id="x", filename="g", path="/store/g", meta=None
    )
    _seed_slot(fake_storage, 2, version="2.6.0", sha="s", ts=1.0)
    _wire_latest(fake_http)
    out = await svc.check()
    assert [s["file_id"] for s in out["snapshots"]] == [_slot_id(2)]


@pytest.mark.asyncio
async def test_null_download_url_asset_treated_missing(svc, fake_functions, fake_http):
    asset = _asset(GOOD_HEADER.encode())
    asset["browser_download_url"] = None
    _wire_latest(fake_http, release=_release(assets=[asset]))
    out = await svc.check()
    assert out["latest"]["assets"]["flat"] is None
    assert out["no_matching_asset"] is True


@pytest.mark.asyncio
async def test_commit_inflight_blocks_second_apply_after_cancel(svc, wired, monkeypatch):
    import asyncio

    release = asyncio.Event()

    async def _slow_commit(content, rev, request, actor, from_version):
        await release.wait()
        return {"ok": True, "from_version": from_version, "to_version": "2.7.0"}

    monkeypatch.setattr(svc, "_commit", _slow_commit)
    rev = wired.functions.row.updated_at

    task = asyncio.get_event_loop().create_task(
        svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if svc._commit_inflight:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert svc._commit_inflight is True
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "update_in_progress"
    release.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if not svc._commit_inflight:
            break
    assert svc._commit_inflight is False


@pytest.mark.asyncio
async def test_commit_failure_after_cancel_recorded(svc, wired, monkeypatch):
    import asyncio

    release = asyncio.Event()

    async def _failing_commit(content, rev, request, actor, from_version):
        await release.wait()
        raise us.UpdateError("stale_rev", "row changed mid-commit")

    monkeypatch.setattr(svc, "_commit", _failing_commit)
    rev = wired.functions.row.updated_at
    task = asyncio.get_event_loop().create_task(
        svc.apply({"rev": rev}, actor="auto", actor_id="sys", request=None)
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if svc._commit_inflight:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    release.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if not svc._commit_inflight:
            break
    assert svc._auto_last is not None
    assert svc._auto_last["code"] == "stale_rev"


@pytest.mark.asyncio
async def test_snapshot_orphan_blob_cleaned_on_insert_failure(svc, fake_functions, fake_storage, monkeypatch):
    async def _boom(user_id, form, db=None):
        raise RuntimeError("insert failed")

    fake_storage.Files.insert_new_file = _boom
    with pytest.raises(us.UpdateError) as exc:
        await svc.snapshot_current("admin", "u1")
    assert exc.value.code == "validation_failed"
    assert "insert failed" in exc.value.message
    assert any(p.startswith("/store/") for p in fake_storage.deleted)
    assert fake_storage.rows == {}
    assert fake_functions.meta_merges == []


class _FakeXLock:
    def __init__(self, available=True):
        self.available = available
        self.acquired = 0
        self.released = 0
        pool = SimpleNamespace(disconnected=0)
        pool.disconnect = lambda: setattr(pool, "disconnected", pool.disconnected + 1)
        client = SimpleNamespace(closed=0, connection_pool=pool)
        client.close = lambda: setattr(client, "closed", client.closed + 1)
        self.redis = client

    def aquire_lock(self):
        self.acquired += 1
        return self.available

    def release_lock(self):
        self.released += 1


@pytest.mark.asyncio
async def test_cross_worker_lock_contended_blocks_before_download(svc, wired, monkeypatch):
    xlock = _FakeXLock(available=False)
    monkeypatch.setattr(us, "_distributed_lock", lambda: xlock)
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "update_in_progress"
    assert wired.events == []
    assert xlock.released == 0


@pytest.mark.asyncio
async def test_cross_worker_lock_released_after_commit(svc, wired, monkeypatch):
    import asyncio

    xlock = _FakeXLock()
    monkeypatch.setattr(us, "_distributed_lock", lambda: xlock)
    rev = wired.functions.row.updated_at
    out = await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert out["ok"] is True
    for _ in range(50):
        if xlock.released:
            break
        await asyncio.sleep(0.01)
    assert xlock.acquired == 1
    assert xlock.released == 1


@pytest.mark.asyncio
async def test_cross_worker_lock_released_on_precommit_failure(svc, wired, monkeypatch):
    xlock = _FakeXLock()
    monkeypatch.setattr(us, "_distributed_lock", lambda: xlock)
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": 42}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "stale_rev"
    assert xlock.released == 1


@pytest.mark.asyncio
async def test_apply_reinstall_same_version_switches_variant(svc, fake_functions, fake_http, fake_storage, monkeypatch):
    import open_webui.utils.plugin as owp

    monkeypatch.setattr(us, "MIN_UPDATE_VERSION", "2.0.0")
    same = GOOD_HEADER.replace("version: 2.7.0", "version: 2.6.9")
    rel = _release(
        tag="v2.6.9",
        assets=[
            _asset(same.encode(), "open_webui_openrouter_pipe_bundled.py"),
            _asset(same.encode(), "open_webui_openrouter_pipe_bundled_compressed.py"),
        ],
    )
    _wire_latest(fake_http, release=rel)
    for asset in rel["assets"]:
        fake_http.bytes_map[asset["browser_download_url"]] = same.encode()

    async def _load(fid, content=None):
        return SimpleNamespace(name="ni"), "pipe", owp.extract_frontmatter(content or "")

    monkeypatch.setattr(owp, "load_function_module_by_id", _load)
    rev = fake_functions.row.updated_at
    out = await svc.apply(
        {"rev": rev, "compressed": True}, actor="admin", actor_id="u1", request=_request()
    )
    assert out["ok"] is True
    assert out["to_version"] == "2.6.9"
    assert any(u.endswith("_compressed.py") for u in fake_http.bytes_calls)


@pytest.mark.asyncio
async def test_auto_never_applies_same_version(auto, fake_http):
    rel = _release(tag="v2.6.9")
    _wire_latest(fake_http, release=rel)
    await auto.svc.check(force=True)
    delay = await auto.svc._auto_tick()
    assert auto.applied == []
    assert delay == us._PD_UPDATE_AUTO_INTERVAL


# ── external-review hardening (M1/M2/M3/L2/L8) ──────────────────────────────


@pytest.mark.asyncio
async def test_commit_content_write_refusal_fails_update(svc, wired):
    wired.functions.fail_content_write = True
    req = _request()
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=req)
    assert exc.value.code == "validation_failed"
    assert "database rejected" in exc.value.message
    assert not hasattr(req.app.state, "FUNCTIONS")
    for _ in range(50):
        if not svc._commit_inflight:
            break
        await asyncio.sleep(0.01)
    assert svc._commit_inflight is False


@pytest.mark.asyncio
async def test_commit_meta_merge_refusal_is_tolerated(svc, wired):
    wired.functions.fail_meta_merge = True
    rev = wired.functions.row.updated_at
    out = await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_commit_rev_guard_trips_before_loader(svc, wired, monkeypatch):
    orig = svc.snapshot_current

    async def _snap_and_bump(actor, owner):
        wired.functions.row.updated_at += 7
        return await orig(actor, owner)

    monkeypatch.setattr(svc, "snapshot_current", _snap_and_bump)
    rev = wired.functions.row.updated_at
    with pytest.raises(us.UpdateError) as exc:
        await svc.apply({"rev": rev}, actor="admin", actor_id="u1", request=_request())
    assert exc.value.code == "stale_rev"
    assert "loader" not in wired.events
    assert not any("content" in u for u in wired.functions.updates)


@pytest.mark.asyncio
async def test_check_reads_row_valves_over_ctx(svc, fake_functions, fake_http):
    _wire_latest(fake_http, repo="row/repo")
    fake_functions.valves_row = {
        "PIPE_DASHBOARD_UPDATE_ENABLE": False,
        "PIPE_DASHBOARD_UPDATE_AUTO": True,
        "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS": 5,
        "PIPE_DASHBOARD_UPDATE_REPO": "row/repo",
    }
    out = await svc.check()
    assert out["enabled"] is False
    assert out["auto"]["enabled"] is True
    assert out["auto"]["delay_hours"] == 5
    assert out["repo"] == "row/repo"
    assert out["latest"] is not None


@pytest.mark.asyncio
async def test_rate_limit_reset_header_case_insensitive(svc, fake_functions, fake_http):
    repo = "rbb-dev/Open-WebUI-OpenRouter-pipe"
    fake_http.json_map[f"https://api.github.com/repos/{repo}/releases/latest"] = (
        429,
        None,
        {"x-ratelimit-reset": "424242"},
    )
    out = await svc.check()
    assert out["last_check_error"]["code"] == "rate_limited"
    assert out["last_check_error"]["reset"] == "424242"


@pytest.mark.asyncio
async def test_contended_lock_client_disposed(svc, wired, monkeypatch):
    xlock = _FakeXLock(available=False)
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: xlock)
    with pytest.raises(us.UpdateError):
        await svc.apply(
            {"rev": wired.functions.row.updated_at}, actor="admin", actor_id="u1", request=_request()
        )
    assert xlock.redis.closed == 1
    assert xlock.redis.connection_pool.disconnected == 1


@pytest.mark.asyncio
async def test_released_lock_client_disposed(svc, wired, monkeypatch):
    xlock = _FakeXLock()
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: xlock)
    out = await svc.apply(
        {"rev": wired.functions.row.updated_at}, actor="admin", actor_id="u1", request=_request()
    )
    assert out["ok"] is True
    for _ in range(50):
        if xlock.redis.closed:
            break
        await asyncio.sleep(0.01)
    assert xlock.redis.closed == 1
    assert xlock.redis.connection_pool.disconnected == 1


# ── leader election ──────────────────────────────────────────────────────────


class _FakeLease:
    def __init__(self, store: dict):
        import uuid

        self.store = store
        self.lock_id = uuid.uuid4().hex
        self.lock_name = "lease"
        self.redis = self
        self.closed = 0
        self.connection_pool = None

    def close(self):
        self.closed += 1

    def get(self, name):
        return self.store.get(name)

    def aquire_lock(self):
        if self.store.get(self.lock_name) is None:
            self.store[self.lock_name] = self.lock_id
            return True
        return False

    def renew_lock(self):
        if self.store.get(self.lock_name) == self.lock_id:
            self.store[self.lock_name] = self.lock_id
            return True
        return False

    def release_lock(self):
        if self.store.get(self.lock_name) == self.lock_id:
            del self.store[self.lock_name]


def _election_env(monkeypatch, svc, store, ticks, sleeps, tick_delay=100.0, stop_after=None):
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: _FakeLease(store))
    monkeypatch.setattr(us, "_PD_UPDATE_AUTO_JITTER", (0.0, 0.0))

    async def _tick():
        ticks.append(us._now())
        if stop_after is not None and len(ticks) >= stop_after:
            raise asyncio.CancelledError()
        return tick_delay

    monkeypatch.setattr(svc, "_auto_tick", _tick)

    async def _fast_sleep(seconds):
        sleeps.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(svc, "_sleep", _fast_sleep)


import asyncio  # noqa: E402


@pytest.mark.asyncio
async def test_leader_election_single_winner(svc, wired, monkeypatch):
    store: dict = {}
    ticks_a: list = []
    sleeps_a: list = []
    _election_env(monkeypatch, svc, store, ticks_a, sleeps_a, tick_delay=10_000.0)
    other = us.UpdateService(lambda: _pipe())
    ticks_b: list = []
    sleeps_b: list = []

    async def _tick_b():
        ticks_b.append(1)
        return 100.0

    monkeypatch.setattr(other, "_auto_tick", _tick_b)

    async def _sleep_b(seconds):
        sleeps_b.append(seconds)
        if len(sleeps_b) >= 3:
            raise asyncio.CancelledError()
        await asyncio.sleep(0)

    monkeypatch.setattr(other, "_sleep", _sleep_b)

    task_a = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    for _ in range(10):
        await asyncio.sleep(0)
        if store.get("lease"):
            break
    task_b = asyncio.get_event_loop().create_task(other.run_auto_loop())
    await asyncio.wait([task_b], timeout=5.0)
    assert ticks_b == []
    assert other._auto_role == "follower"
    assert us._PD_UPDATE_FOLLOWER_POLL_S in sleeps_b
    assert len(ticks_a) >= 1
    assert svc._auto_role == "leader"
    task_a.cancel()
    await asyncio.gather(task_a, return_exceptions=True)


@pytest.mark.asyncio
async def test_follower_takes_over_after_lease_expiry(svc, wired, monkeypatch):
    store: dict = {"lease": "someone-else"}
    ticks: list = []
    sleeps: list = []
    _election_env(monkeypatch, svc, store, ticks, sleeps, stop_after=1)

    original_sleep = svc._sleep

    async def _sleep_and_expire(seconds):
        if seconds == us._PD_UPDATE_FOLLOWER_POLL_S:
            store.pop("lease", None)
        await original_sleep(seconds)

    monkeypatch.setattr(svc, "_sleep", _sleep_and_expire)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    await asyncio.wait([task], timeout=5.0)
    assert len(ticks) == 1
    assert svc._auto_role == "leader"


@pytest.mark.asyncio
async def test_leader_steps_down_when_lease_stolen(svc, wired, monkeypatch):
    store: dict = {}
    ticks: list = []
    sleeps: list = []
    _election_env(monkeypatch, svc, store, ticks, sleeps, tick_delay=3 * us._PD_UPDATE_LEADER_RENEW_S)

    async def _tick_once():
        ticks.append(1)
        if len(ticks) == 1:
            return 3.0 * us._PD_UPDATE_LEADER_RENEW_S
        raise asyncio.CancelledError()

    monkeypatch.setattr(svc, "_auto_tick", _tick_once)

    stolen = {"done": False}
    original_sleep = svc._sleep

    async def _sleep_steal(seconds):
        if not stolen["done"] and seconds == float(us._PD_UPDATE_LEADER_RENEW_S):
            store["lease"] = "thief"
            stolen["done"] = True
        await original_sleep(seconds)

    monkeypatch.setattr(svc, "_sleep", _sleep_steal)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    for _ in range(200):
        await asyncio.sleep(0)
        if svc._auto_role == "follower":
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert svc._auto_role == "follower"
    assert len(ticks) == 1
    assert store["lease"] == "thief"


@pytest.mark.asyncio
async def test_leader_renews_between_ticks(svc, wired, monkeypatch):
    store: dict = {}
    ticks: list = []
    sleeps: list = []
    _election_env(monkeypatch, svc, store, ticks, sleeps, tick_delay=3 * us._PD_UPDATE_LEADER_RENEW_S, stop_after=2)
    renews: list = []
    original = us.UpdateService._renew_leader

    def _spy(self, lease):
        renews.append(1)
        return original(self, lease)

    monkeypatch.setattr(us.UpdateService, "_renew_leader", _spy)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    await asyncio.wait([task], timeout=5.0)
    assert len(renews) >= 3


@pytest.mark.asyncio
async def test_cancelled_leader_releases_lease(svc, wired, monkeypatch):
    store: dict = {}
    ticks: list = []
    sleeps: list = []
    _election_env(monkeypatch, svc, store, ticks, sleeps, tick_delay=10_000.0)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    for _ in range(50):
        await asyncio.sleep(0)
        if ticks:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert store.get("lease") is None


@pytest.mark.asyncio
async def test_no_redis_acts_as_solo_leader(svc, wired, monkeypatch):
    monkeypatch.setattr(us, "_distributed_lock", lambda *a, **k: None)
    monkeypatch.setattr(us, "_PD_UPDATE_AUTO_JITTER", (0.0, 0.0))
    ticks: list = []

    async def _tick():
        ticks.append(1)
        raise asyncio.CancelledError()

    monkeypatch.setattr(svc, "_auto_tick", _tick)

    async def _fast(seconds):
        await asyncio.sleep(0)

    monkeypatch.setattr(svc, "_sleep", _fast)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    await asyncio.wait([task], timeout=5.0)
    assert ticks == [1]
    assert svc._auto_role == "solo"


@pytest.mark.asyncio
async def test_follower_probe_lease_disposed(svc, wired, monkeypatch):
    store: dict = {"lease": "someone-else"}
    leases: list = []

    def _mk(*a, **k):
        lease = _FakeLease(store)
        leases.append(lease)
        return lease

    monkeypatch.setattr(us, "_distributed_lock", _mk)
    monkeypatch.setattr(us, "_PD_UPDATE_AUTO_JITTER", (0.0, 0.0))
    sleeps: list = []

    async def _sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()
        await asyncio.sleep(0)

    monkeypatch.setattr(svc, "_sleep", _sleep)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    await asyncio.wait([task], timeout=5.0)
    assert leases and leases[0].closed >= 1


@pytest.mark.asyncio
async def test_cancelled_leader_lease_disposed(svc, wired, monkeypatch):
    store: dict = {}
    leases: list = []

    def _mk(*a, **k):
        lease = _FakeLease(store)
        leases.append(lease)
        return lease

    monkeypatch.setattr(us, "_distributed_lock", _mk)
    monkeypatch.setattr(us, "_PD_UPDATE_AUTO_JITTER", (0.0, 0.0))
    ticks: list = []

    async def _tick():
        ticks.append(1)
        return 10_000.0

    monkeypatch.setattr(svc, "_auto_tick", _tick)

    async def _fast(seconds):
        await asyncio.sleep(0)

    monkeypatch.setattr(svc, "_sleep", _fast)
    task = asyncio.get_event_loop().create_task(svc.run_auto_loop())
    for _ in range(50):
        await asyncio.sleep(0)
        if ticks:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert leases and any(lease.closed for lease in leases)


def test_pipe_init_attaches_registry_last():
    import ast
    from pathlib import Path

    src = (Path(__file__).parent.parent / "open_webui_openrouter_pipe" / "pipe.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Pipe":
            init = next(
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            )
            last = init.body[-1]
            assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call), (
                "Pipe.__init__ must END with the lifecycle-registry attach call — the update "
                "service's exec-failure repair depends on a failed init never draining the "
                "serving instance"
            )
            func = last.value.func
            assert getattr(func, "attr", "") == "_attach_to_lifecycle_registry"
            return
    raise AssertionError("class Pipe not found in pipe.py")
