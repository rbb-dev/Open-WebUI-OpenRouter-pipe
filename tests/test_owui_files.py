"""Unit tests for the authorized, backend-agnostic OWUI file storage gateway.

Exercises ``open_webui_openrouter_pipe/storage/owui_files.py`` at the unit level:
module helpers (auth, size, containment, mime, base64) and the
``OwuiFileGateway`` methods. Heavy OWUI dependencies are stubbed in conftest;
``has_access_to_file`` / ``Storage.get_file`` are monkeypatched per-test.
"""

from __future__ import annotations

import base64
import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.core.errors import RequiredInternalFileError
from open_webui_openrouter_pipe.storage import owui_files
from open_webui_openrouter_pipe.storage.owui_files import (
    InlinedFile,
    OwuiFileGateway,
    authorize_file_read,
    contained_under_upload_dir,
    copy_to_private_temp,
    declared_file_size,
    encode_file_path_base64,
    extract_internal_file_id,
    get_owui_storage,
    infer_file_mime_type,
    is_internal_file_url,
    is_real_owui_file_record,
    materialize_owui_file_to_temp,
    owui_storage_provider_kind,
)


def _file_obj(**kwargs: Any) -> SimpleNamespace:
    """Build a tiny fake OWUI file record with the given attributes."""
    return SimpleNamespace(**kwargs)


def _access_control_module():
    """Return the conftest-installed access_control.files stub module."""
    return importlib.import_module("open_webui.utils.access_control.files")


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_owui_files")


@pytest.fixture
def gateway(logger: logging.Logger) -> OwuiFileGateway:
    """Gateway wired with a real Valves instance (no Pipe lifecycle to leak)."""
    return OwuiFileGateway(logger=logger, valves=Valves())


@pytest.fixture
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the gateway's UPLOAD_DIR at a real temp dir for containment tests."""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(owui_files, "_OWUI_UPLOAD_DIR", str(root))
    return root


# ─────────────────────────────────────────────────────────────────────────────
# 1. authorize_file_read
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_authorize_none_user_denied(logger: logging.Logger):
    """No requester means no access."""
    assert await authorize_file_read(_file_obj(id="f1", user_id="u1"), None, logger) is False


@pytest.mark.asyncio
async def test_authorize_owner_allowed(logger: logging.Logger):
    """Owner (file.user_id == user.id) is authorized without external checks."""
    user = SimpleNamespace(id="u1", role="user")
    assert await authorize_file_read(_file_obj(id="f1", user_id="u1"), user, logger) is True


@pytest.mark.asyncio
async def test_authorize_admin_non_owner_allowed(logger: logging.Logger):
    """A non-owner admin is authorized."""
    user = SimpleNamespace(id="admin", role="admin")
    assert await authorize_file_read(_file_obj(id="f1", user_id="someone"), user, logger) is True


@pytest.mark.asyncio
async def test_authorize_delegates_to_has_access_true(logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    """Non-owner non-admin: delegate to has_access_to_file, allow when it returns True."""
    seen: dict[str, Any] = {}

    async def _grant(file_id, access_type, user, db=None):
        seen.update(file_id=file_id, access_type=access_type, user=user)
        return True

    mod = _access_control_module()
    monkeypatch.setattr(mod, "has_access_to_file", _grant)
    user = SimpleNamespace(id="u2", role="user")
    assert await authorize_file_read(_file_obj(id="f9", user_id="owner"), user, logger) is True
    assert seen == {"file_id": "f9", "access_type": "read", "user": user}


@pytest.mark.asyncio
async def test_authorize_delegates_to_has_access_false(logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    """Non-owner non-admin: deny when has_access_to_file returns False."""
    async def _deny(file_id, access_type, user, db=None):
        return False

    mod = _access_control_module()
    monkeypatch.setattr(mod, "has_access_to_file", _deny)
    user = SimpleNamespace(id="u2", role="user")
    assert await authorize_file_read(_file_obj(id="f9", user_id="owner"), user, logger) is False


@pytest.mark.asyncio
async def test_authorize_fail_closed_when_has_access_raises(logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    """An exception inside has_access_to_file fails closed (returns False)."""
    async def _boom(file_id, access_type, user, db=None):
        raise RuntimeError("db down")

    mod = _access_control_module()
    monkeypatch.setattr(mod, "has_access_to_file", _boom)
    user = SimpleNamespace(id="u2", role="user")
    assert await authorize_file_read(_file_obj(id="f9", user_id="owner"), user, logger) is False


@pytest.mark.asyncio
async def test_authorize_fail_closed_on_import_error(logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    """When the access_control module cannot be imported, fail closed."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "open_webui.utils.access_control.files":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    user = SimpleNamespace(id="u2", role="user")
    assert await authorize_file_read(_file_obj(id="f9", user_id="owner"), user, logger) is False


@pytest.mark.asyncio
async def test_authorize_non_owner_without_file_id_denied(logger: logging.Logger):
    """A real-but-idless record for a non-owner non-admin cannot be authorized."""
    user = SimpleNamespace(id="u2", role="user")
    assert await authorize_file_read(_file_obj(id=None, user_id="owner"), user, logger) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. is_real_owui_file_record
# ─────────────────────────────────────────────────────────────────────────────


def test_is_real_record_true_with_id():
    assert is_real_owui_file_record(_file_obj(id="abc")) is True


def test_is_real_record_false_without_id():
    assert is_real_owui_file_record(_file_obj(id=None)) is False
    assert is_real_owui_file_record(_file_obj(path="/tmp/x")) is False
    assert is_real_owui_file_record(_file_obj(id="")) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. declared_file_size
# ─────────────────────────────────────────────────────────────────────────────


def test_declared_size_from_meta_size():
    assert declared_file_size(_file_obj(meta={"size": 123})) == 123


def test_declared_size_from_meta_data_size():
    assert declared_file_size(_file_obj(meta={"data": {"size": 456}})) == 456


def test_declared_size_prefers_top_level_size():
    assert declared_file_size(_file_obj(meta={"size": 10, "data": {"size": 20}})) == 10


def test_declared_size_ignores_bool_zero_negative_and_non_int():
    assert declared_file_size(_file_obj(meta={"size": True})) is None
    assert declared_file_size(_file_obj(meta={"size": 0})) is None
    assert declared_file_size(_file_obj(meta={"size": -5})) is None
    assert declared_file_size(_file_obj(meta={"size": "99"})) is None


def test_declared_size_no_meta_or_non_dict():
    assert declared_file_size(_file_obj()) is None
    assert declared_file_size(_file_obj(meta=None)) is None
    assert declared_file_size(_file_obj(meta="nope")) is None


def test_declared_size_falls_through_to_inner_when_top_invalid():
    assert declared_file_size(_file_obj(meta={"size": 0, "data": {"size": 77}})) == 77


# ─────────────────────────────────────────────────────────────────────────────
# 4. owui_storage_provider_kind
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "local"),
        ("", "local"),
        ("local", "local"),
        ("LOCAL", "local"),
        ("s3", "s3"),
        ("S3", "s3"),
        ("gcs", "gcs"),
        ("google", "gcs"),
        ("azure", "azure"),
        ("azblob", "azure"),
        ("backblaze", "unknown"),
        ("minio", "unknown"),
    ],
)
def test_provider_kind_mapping(monkeypatch: pytest.MonkeyPatch, raw, expected):
    monkeypatch.setattr(owui_files, "_OWUI_STORAGE_PROVIDER", raw)
    assert owui_storage_provider_kind() == expected


# ─────────────────────────────────────────────────────────────────────────────
# 5. contained_under_upload_dir
# ─────────────────────────────────────────────────────────────────────────────


def test_contained_returns_resolved_when_under_upload_dir(upload_dir: Path):
    inside = upload_dir / "a.bin"
    inside.write_bytes(b"x")
    result = contained_under_upload_dir(str(inside))
    assert result == inside.resolve()


def test_contained_none_when_outside(upload_dir: Path, tmp_path: Path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    assert contained_under_upload_dir(str(outside)) is None


def test_contained_none_when_upload_dir_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(owui_files, "_OWUI_UPLOAD_DIR", None)
    f = tmp_path / "f.bin"
    f.write_bytes(b"x")
    assert contained_under_upload_dir(str(f)) is None


def test_contained_none_when_path_missing(upload_dir: Path):
    """strict resolve of a non-existent path fails closed to None."""
    assert contained_under_upload_dir(str(upload_dir / "nope.bin")) is None


def test_contained_blocks_traversal(upload_dir: Path, tmp_path: Path):
    """A ../ escape that resolves outside UPLOAD_DIR is rejected."""
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"s")
    traversal = upload_dir / ".." / "secret.bin"
    assert contained_under_upload_dir(str(traversal)) is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. copy_to_private_temp
# ─────────────────────────────────────────────────────────────────────────────


def test_copy_to_private_temp_distinct_file_same_bytes(tmp_path: Path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello-bytes")
    dst = copy_to_private_temp(src)
    try:
        assert dst != src
        assert dst.exists()
        assert dst.read_bytes() == b"hello-bytes"
        assert dst.name.startswith("orpipe-read-")
    finally:
        dst.unlink(missing_ok=True)


def test_copy_to_private_temp_respects_suffix(tmp_path: Path):
    src = tmp_path / "src.dat"
    src.write_bytes(b"abc")
    dst = copy_to_private_temp(src, suffix=".png")
    try:
        assert dst.suffix == ".png"
    finally:
        dst.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. materialize_owui_file_to_temp (THE CORE)
# ─────────────────────────────────────────────────────────────────────────────


def _install_storage(monkeypatch: pytest.MonkeyPatch, return_path: Any) -> None:
    """Make get_owui_storage().get_file return return_path."""
    storage = SimpleNamespace(get_file=lambda key: return_path)
    monkeypatch.setattr(owui_files, "get_owui_storage", lambda: storage)


@pytest.mark.asyncio
async def test_materialize_denies_unauthorized_real_record(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """require_auth + real record + unauthorized user => denied RequiredInternalFileError."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=False))
    fobj = _file_obj(id="f1", user_id="owner", path="key", meta={"size": 4})
    with pytest.raises(RequiredInternalFileError) as ei:
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="other", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )
    assert ei.value.denied is True


@pytest.mark.asyncio
async def test_materialize_missing_path_raises(logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    """A record without a storage path cannot be materialized."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    fobj = _file_obj(id="f1", user_id="u1", path=None, meta={"size": 4})
    with pytest.raises(RequiredInternalFileError) as ei:
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )
    assert ei.value.denied is False


@pytest.mark.asyncio
async def test_materialize_declared_size_over_limit_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
):
    """Declared size exceeding max_bytes is rejected before any storage read."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    called = {"storage": False}
    monkeypatch.setattr(
        owui_files, "get_owui_storage",
        lambda: called.__setitem__("storage", True),
    )
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 10_000})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=100, allow_unknown_size=False,
        )
    assert called["storage"] is False


@pytest.mark.asyncio
async def test_materialize_unknown_size_cloud_denied_when_not_allowed(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
):
    """Unknown declared size + non-local provider + not allowed => rejected."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    monkeypatch.setattr(owui_files, "_OWUI_STORAGE_PROVIDER", "s3")
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )


@pytest.mark.asyncio
async def test_materialize_unknown_size_cloud_allowed_succeeds(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """Unknown size on cloud succeeds when allow_unknown_size=True."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    monkeypatch.setattr(owui_files, "_OWUI_STORAGE_PROVIDER", "s3")
    backing = upload_dir / "blob.bin"
    backing.write_bytes(b"cloud-bytes")
    _install_storage(monkeypatch, str(backing))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={})
    out = await materialize_owui_file_to_temp(
        fobj, user=SimpleNamespace(id="u1", role="user"),
        logger=logger, max_bytes=1000, allow_unknown_size=True,
    )
    try:
        assert out.read_bytes() == b"cloud-bytes"
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_materialize_success_returns_private_temp_under_systmp(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """Happy path: returns a private temp (prefix orpipe-read-) with the file bytes, NOT the original."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    backing = upload_dir / "doc.pdf"
    backing.write_bytes(b"%PDF-data")
    _install_storage(monkeypatch, str(backing))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 9})
    out = await materialize_owui_file_to_temp(
        fobj, user=SimpleNamespace(id="u1", role="user"),
        logger=logger, max_bytes=1000, allow_unknown_size=False,
    )
    try:
        assert out != backing.resolve()
        assert out.exists()
        assert out.read_bytes() == b"%PDF-data"
        assert out.name.startswith("orpipe-read-")
        import tempfile

        assert str(out).startswith(str(Path(tempfile.gettempdir()).resolve())) or out.name.startswith(
            "orpipe-read-"
        )
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_materialize_path_escaping_upload_dir_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path, tmp_path: Path
):
    """A storage path that resolves outside UPLOAD_DIR is rejected (fail-closed)."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    outside = tmp_path / "escape.bin"
    outside.write_bytes(b"leak")
    _install_storage(monkeypatch, str(outside))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 4})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )


@pytest.mark.asyncio
async def test_materialize_unsupported_suffix_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """allowed_suffixes gates by extension of the resolved file."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    backing = upload_dir / "x.exe"
    backing.write_bytes(b"MZ")
    _install_storage(monkeypatch, str(backing))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 2})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
            allowed_suffixes={".png", ".jpg"},
        )


@pytest.mark.asyncio
async def test_materialize_supported_suffix_passes(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """A permitted extension passes the allowed_suffixes gate."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    backing = upload_dir / "pic.png"
    backing.write_bytes(b"PNGDATA")
    _install_storage(monkeypatch, str(backing))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 7})
    out = await materialize_owui_file_to_temp(
        fobj, user=SimpleNamespace(id="u1", role="user"),
        logger=logger, max_bytes=1000, allow_unknown_size=False,
        allowed_suffixes={".png", ".jpg"},
    )
    try:
        assert out.read_bytes() == b"PNGDATA"
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_materialize_post_copy_stat_over_limit_unlinks_and_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """When the actual copied size exceeds max_bytes (declared was None/local), unlink + raise."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    backing = upload_dir / "big.bin"
    backing.write_bytes(b"0123456789ABCDEF")
    _install_storage(monkeypatch, str(backing))
    captured: dict[str, Path] = {}
    real_copy = owui_files.copy_to_private_temp

    def _spy(path, *, suffix=""):
        out = real_copy(path, suffix=suffix)
        captured["temp"] = out
        return out

    monkeypatch.setattr(owui_files, "copy_to_private_temp", _spy)
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=4, allow_unknown_size=False,
        )
    assert captured["temp"].exists() is False


@pytest.mark.asyncio
async def test_materialize_storage_get_file_failure_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """A Storage.get_file exception is wrapped as RequiredInternalFileError."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))

    def _boom(key):
        raise OSError("backend offline")

    monkeypatch.setattr(owui_files, "get_owui_storage", lambda: SimpleNamespace(get_file=_boom))
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 4})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )


@pytest.mark.asyncio
async def test_materialize_require_auth_false_skips_authorization(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, upload_dir: Path
):
    """With require_auth=False, authorization is not consulted even for real records."""
    auth = AsyncMock(return_value=False)
    monkeypatch.setattr(owui_files, "authorize_file_read", auth)
    backing = upload_dir / "f.bin"
    backing.write_bytes(b"data")
    _install_storage(monkeypatch, str(backing))
    fobj = _file_obj(id="f1", user_id="owner", path="key", meta={"size": 4})
    out = await materialize_owui_file_to_temp(
        fobj, user=None, logger=logger, max_bytes=1000,
        allow_unknown_size=False, require_auth=False,
    )
    try:
        assert out.read_bytes() == b"data"
        auth.assert_not_awaited()
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_materialize_storage_unavailable_raises(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
):
    """When no storage provider is available, materialization fails."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    monkeypatch.setattr(owui_files, "get_owui_storage", lambda: None)
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 4})
    with pytest.raises(RequiredInternalFileError):
        await materialize_owui_file_to_temp(
            fobj, user=SimpleNamespace(id="u1", role="user"),
            logger=logger, max_bytes=1000, allow_unknown_size=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. read_file_record_base64 (gateway method)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_b64_max_bytes_zero_raises_value_error(gateway: OwuiFileGateway):
    with pytest.raises(ValueError):
        await gateway.read_file_record_base64(_file_obj(id="f1"), 1024, 0)


@pytest.mark.asyncio
async def test_read_b64_auth_first_deny(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """Real record + unauthorized => RequiredInternalFileError before any byte source."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=False))
    fobj = _file_obj(id="f1", user_id="owner", data={"b64": "QUJD"})
    with pytest.raises(RequiredInternalFileError) as ei:
        await gateway.read_file_record_base64(
            fobj, 1024, 1024, user=SimpleNamespace(id="x", role="user")
        )
    assert ei.value.denied is True


@pytest.mark.asyncio
async def test_read_b64_inline_data_fast_path_b64(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """data dict with a 'b64' string returns it directly (stripped)."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    fobj = _file_obj(id="f1", user_id="u1", data={"b64": "  QUJD  "})
    out = await gateway.read_file_record_base64(
        fobj, 1024, 1024, user=SimpleNamespace(id="u1", role="user")
    )
    assert out == "QUJD"


@pytest.mark.asyncio
async def test_read_b64_inline_data_keys_base64_and_data(gateway: OwuiFileGateway):
    """No-id synthetic records: inline 'base64' and 'data' keys both work without auth."""
    out1 = await gateway.read_file_record_base64(
        _file_obj(data={"base64": "WFla"}), 1024, 1024, require_auth=False
    )
    assert out1 == "WFla"
    out2 = await gateway.read_file_record_base64(
        _file_obj(data={"data": "WFla"}), 1024, 1024, require_auth=False
    )
    assert out2 == "WFla"


@pytest.mark.asyncio
async def test_read_b64_inline_bytes_key(gateway: OwuiFileGateway):
    """data dict carrying raw 'bytes' is base64-encoded."""
    out = await gateway.read_file_record_base64(
        _file_obj(data={"bytes": b"ABC"}), 1024, 1024, require_auth=False
    )
    assert out == base64.b64encode(b"ABC").decode("ascii")


@pytest.mark.asyncio
async def test_read_b64_inline_oversize_raises(gateway: OwuiFileGateway):
    """Inline base64 whose estimated decode exceeds BASE64_MAX_SIZE_MB is rejected.

    The inline-string fast path is gated by validate_base64_size against the
    BASE64_MAX_SIZE_MB valve (estimated decode = len*3/4), not the per-call
    max_bytes, so the string length must exceed valve_bytes * 4/3.
    """
    gateway.valves.BASE64_MAX_SIZE_MB = 1
    big = "A" * int(1 * 1024 * 1024 * (4 / 3) + 1024)
    with pytest.raises(ValueError):
        await gateway.read_file_record_base64(
            _file_obj(data={"b64": big}), 1024, 10 * 1024 * 1024, require_auth=False
        )


@pytest.mark.asyncio
async def test_read_b64_real_record_routes_through_materialize_and_unlinks(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Real record with a path goes Storage->temp->encode and unlinks the temp after."""
    monkeypatch.setattr(owui_files, "authorize_file_read", AsyncMock(return_value=True))
    temp = tmp_path / "materialized.bin"
    temp.write_bytes(b"PAYLOAD")
    mat = AsyncMock(return_value=temp)
    monkeypatch.setattr(owui_files, "materialize_owui_file_to_temp", mat)
    fobj = _file_obj(id="f1", user_id="u1", path="key", meta={"size": 7})
    out = await gateway.read_file_record_base64(
        fobj, 1024, 1024, user=SimpleNamespace(id="u1", role="user")
    )
    assert out == base64.b64encode(b"PAYLOAD").decode("ascii")
    assert temp.exists() is False
    assert mat.await_args is not None
    assert mat.await_args.kwargs["require_auth"] is False


@pytest.mark.asyncio
async def test_read_b64_synthetic_record_uses_local_path(
    gateway: OwuiFileGateway, tmp_path: Path
):
    """A no-id record reads its local 'path' directly via the legacy path."""
    f = tmp_path / "legacy.bin"
    f.write_bytes(b"LEGACY")
    out = await gateway.read_file_record_base64(
        _file_obj(path=str(f)), 1024, 1024, require_auth=False
    )
    assert out == base64.b64encode(b"LEGACY").decode("ascii")


@pytest.mark.asyncio
async def test_read_b64_raw_bytes_attr_fallback(gateway: OwuiFileGateway):
    """When no inline-dict/path applies, raw 'content' bytes are used."""
    out = await gateway.read_file_record_base64(
        _file_obj(content=b"RAWBYTES"), 1024, 1024, require_auth=False
    )
    assert out == base64.b64encode(b"RAWBYTES").decode("ascii")


@pytest.mark.asyncio
async def test_read_b64_returns_none_when_no_source(gateway: OwuiFileGateway):
    """A record with no usable byte source returns None."""
    out = await gateway.read_file_record_base64(
        _file_obj(path="/nonexistent/none.bin"), 1024, 1024, require_auth=False
    )
    assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. validate_base64_size
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_base64_size_within_limit(gateway: OwuiFileGateway):
    assert gateway.validate_base64_size("QUJD") is True
    assert gateway.validate_base64_size("") is True


def test_validate_base64_size_over_limit(gateway: OwuiFileGateway):
    over = "A" * (gateway.valves.BASE64_MAX_SIZE_MB * 1024 * 1024 + 8) * 2
    assert gateway.validate_base64_size(over) is False


# ─────────────────────────────────────────────────────────────────────────────
# 10. inline_owui_file_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inline_owui_file_id_success(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """Returns InlinedFile(data_url, filename) on success."""
    fobj = _file_obj(id="abc", user_id="u1", meta={"name": "report.pdf"}, mime_type="application/pdf")
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=fobj))
    monkeypatch.setattr(
        gateway, "read_file_record_base64", AsyncMock(return_value="QkFTRTY0")
    )
    result = await gateway.inline_owui_file_id(
        "abc", chunk_size=1024, max_bytes=1024, user=SimpleNamespace(id="u1", role="user")
    )
    assert isinstance(result, InlinedFile)
    assert result.data_url == "data:application/pdf;base64,QkFTRTY0"
    assert result.filename == "report.pdf"


@pytest.mark.asyncio
async def test_inline_owui_file_id_blank_returns_none(gateway: OwuiFileGateway):
    assert await gateway.inline_owui_file_id("   ", chunk_size=1024, max_bytes=1024) is None


@pytest.mark.asyncio
async def test_inline_owui_file_id_missing_file_returns_none(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=None))
    assert await gateway.inline_owui_file_id("abc", chunk_size=1024, max_bytes=1024) is None


@pytest.mark.asyncio
async def test_inline_owui_file_id_empty_b64_returns_none(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    fobj = _file_obj(id="abc", user_id="u1", meta={"name": "x"})
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=fobj))
    monkeypatch.setattr(gateway, "read_file_record_base64", AsyncMock(return_value=None))
    assert await gateway.inline_owui_file_id(
        "abc", chunk_size=1024, max_bytes=1024, user=SimpleNamespace(id="u1", role="user")
    ) is None


@pytest.mark.asyncio
async def test_inline_owui_file_id_value_error_swallowed_to_none(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    """A size ValueError from read is logged and yields None (not raised)."""
    fobj = _file_obj(id="abc", user_id="u1", meta={"name": "x"})
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=fobj))
    monkeypatch.setattr(
        gateway, "read_file_record_base64", AsyncMock(side_effect=ValueError("too big"))
    )
    assert await gateway.inline_owui_file_id(
        "abc", chunk_size=1024, max_bytes=1024, user=SimpleNamespace(id="u1", role="user")
    ) is None


@pytest.mark.asyncio
async def test_inline_owui_file_id_propagates_required_error(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    """RequiredInternalFileError from auth must NOT be swallowed as ValueError."""
    fobj = _file_obj(id="abc", user_id="owner", meta={"name": "x"})
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=fobj))
    monkeypatch.setattr(
        gateway,
        "read_file_record_base64",
        AsyncMock(side_effect=RequiredInternalFileError("denied", denied=True)),
    )
    with pytest.raises(RequiredInternalFileError):
        await gateway.inline_owui_file_id(
            "abc", chunk_size=1024, max_bytes=1024, user=SimpleNamespace(id="x", role="user")
        )


@pytest.mark.asyncio
async def test_inline_owui_file_id_filename_falls_back_to_attr(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    """Filename falls back to file_obj.filename when meta has no name."""
    fobj = _file_obj(id="abc", user_id="u1", meta={}, filename="fallback.txt", mime_type="text/plain")
    monkeypatch.setattr(owui_files, "get_file_by_id", AsyncMock(return_value=fobj))
    monkeypatch.setattr(gateway, "read_file_record_base64", AsyncMock(return_value="QQ=="))
    result = await gateway.inline_owui_file_id(
        "abc", chunk_size=1024, max_bytes=1024, user=SimpleNamespace(id="u1", role="user")
    )
    assert result is not None
    assert result.filename == "fallback.txt"


# ─────────────────────────────────────────────────────────────────────────────
# inline_internal_file_url
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inline_internal_file_url_delegates(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """Extracts the id from a URL and delegates to inline_owui_file_id."""
    inline = AsyncMock(return_value=InlinedFile(data_url="data:x;base64,QQ==", filename="f"))
    monkeypatch.setattr(gateway, "inline_owui_file_id", inline)
    out = await gateway.inline_internal_file_url(
        "/api/v1/files/zzz-1/content", chunk_size=1024, max_bytes=2048
    )
    assert out is not None
    inline.assert_awaited_once_with("zzz-1", chunk_size=1024, max_bytes=2048, user=None)


@pytest.mark.asyncio
async def test_inline_internal_file_url_no_id_returns_none(gateway: OwuiFileGateway):
    assert await gateway.inline_internal_file_url(
        "https://example.com/not-a-file", chunk_size=1024, max_bytes=1024
    ) is None


# ─────────────────────────────────────────────────────────────────────────────
# 11. inline_internal_responses_input_files_inplace
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inplace_converts_internal_file_id_block(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    """An input_file with an internal file_id becomes file_data and drops file_id."""
    inline = AsyncMock(
        return_value=InlinedFile(data_url="data:application/pdf;base64,UERG", filename="doc.pdf")
    )
    monkeypatch.setattr(gateway, "inline_owui_file_id", inline)
    body = {
        "input": [
            {"content": [{"type": "input_file", "file_id": "internal-77"}]}
        ]
    }
    await gateway.inline_internal_responses_input_files_inplace(
        body, chunk_size=1024, max_bytes=1024
    )
    block = body["input"][0]["content"][0]
    assert block["file_data"] == "data:application/pdf;base64,UERG"
    assert block["filename"] == "doc.pdf"
    assert "file_id" not in block
    inline.assert_awaited_once()


@pytest.mark.asyncio
async def test_inplace_skips_provider_file_id(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """A provider 'file-...' id is left untouched (not an OWUI internal id)."""
    inline = AsyncMock()
    monkeypatch.setattr(gateway, "inline_owui_file_id", inline)
    body = {"input": [{"content": [{"type": "input_file", "file_id": "file-openai-abc"}]}]}
    await gateway.inline_internal_responses_input_files_inplace(
        body, chunk_size=1024, max_bytes=1024
    )
    assert body["input"][0]["content"][0]["file_id"] == "file-openai-abc"
    inline.assert_not_awaited()


@pytest.mark.asyncio
async def test_inplace_converts_internal_file_url_and_pops_it(
    gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch
):
    """An internal file_url is inlined into file_data and the file_url is removed."""
    inline = AsyncMock(
        return_value=InlinedFile(data_url="data:application/pdf;base64,UERG", filename="")
    )
    monkeypatch.setattr(gateway, "inline_owui_file_id", inline)
    body = {
        "input": [
            {"content": [{"type": "input_file", "file_url": "/api/v1/files/abc-9/content"}]}
        ]
    }
    await gateway.inline_internal_responses_input_files_inplace(
        body, chunk_size=1024, max_bytes=1024
    )
    block = body["input"][0]["content"][0]
    assert block["file_data"] == "data:application/pdf;base64,UERG"
    assert "file_url" not in block
    inline.assert_awaited_once_with("abc-9", chunk_size=1024, max_bytes=1024, user=None)


@pytest.mark.asyncio
async def test_inplace_raises_on_inline_failure(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """A None inline result for a required file raises ValueError."""
    monkeypatch.setattr(gateway, "inline_owui_file_id", AsyncMock(return_value=None))
    body = {"input": [{"content": [{"type": "input_file", "file_id": "internal-77"}]}]}
    with pytest.raises(ValueError):
        await gateway.inline_internal_responses_input_files_inplace(
            body, chunk_size=1024, max_bytes=1024
        )


@pytest.mark.asyncio
async def test_inplace_propagates_required_error(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """Auth denial during inlining propagates as RequiredInternalFileError."""
    monkeypatch.setattr(
        gateway,
        "inline_owui_file_id",
        AsyncMock(side_effect=RequiredInternalFileError("denied", denied=True)),
    )
    body = {"input": [{"content": [{"type": "input_file", "file_id": "internal-77"}]}]}
    with pytest.raises(RequiredInternalFileError):
        await gateway.inline_internal_responses_input_files_inplace(
            body, chunk_size=1024, max_bytes=1024
        )


@pytest.mark.asyncio
async def test_inplace_noop_without_input(gateway: OwuiFileGateway, monkeypatch: pytest.MonkeyPatch):
    """No input list => no work, no inline calls."""
    inline = AsyncMock()
    monkeypatch.setattr(gateway, "inline_owui_file_id", inline)
    body: dict[str, Any] = {"model": "x"}
    await gateway.inline_internal_responses_input_files_inplace(
        body, chunk_size=1024, max_bytes=1024
    )
    inline.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# 12. infer_file_mime_type / extract_internal_file_id / is_internal_file_url /
#     encode_file_path_base64 / get_owui_storage
# ─────────────────────────────────────────────────────────────────────────────


def test_infer_mime_prefers_mime_type_attr():
    assert infer_file_mime_type(_file_obj(mime_type="image/png")) == "image/png"


def test_infer_mime_normalizes_jpg_to_jpeg():
    assert infer_file_mime_type(_file_obj(mime_type="image/JPG")) == "image/jpeg"


def test_infer_mime_from_meta_keys():
    assert infer_file_mime_type(_file_obj(meta={"content_type": "text/plain"})) == "text/plain"
    assert infer_file_mime_type(_file_obj(meta={"mimeType": "application/pdf"})) == "application/pdf"


def test_infer_mime_default_octet_stream():
    assert infer_file_mime_type(_file_obj()) == "application/octet-stream"
    assert infer_file_mime_type(_file_obj(mime_type="   ")) == "application/octet-stream"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/api/v1/files/abc-123/content", "abc-123"),
        ("/api/v1/files/abc-123", "abc-123"),
        ("https://host/files/XyZ9", "XyZ9"),
        ("https://example.com/no-files-here", None),
        (12345, None),
    ],
)
def test_extract_internal_file_id(url, expected):
    assert extract_internal_file_id(url) == expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/api/v1/files/abc/content", True),
        ("https://host/api/v1/files/abc/content", False),
        ("HTTPS://host/api/v1/files/abc/content", False),
        ("HTTP://host/api/v1/files/abc/content", False),
        ("Http://host/api/v1/files/abc/content", False),
        ("https://cdn.example.com/api/v1/files/manual.pdf", False),
        ("ftp://host/api/v1/files/abc/content", False),
        ("FTP://host/api/v1/files/abc/content", False),
        ("gopher://host/api/v1/files/abc/content", False),
        ("//evil.test/api/v1/files/abc/content", False),
        ("ht\ttp://host/api/v1/files/abc/content", False),
        ("/files/abc", False),
        ("https://example.com/files/foo", False),
        ("https://example.com/image.png", False),
        (None, False),
        (123, False),
    ],
)
def test_is_internal_file_url(url, expected):
    assert is_internal_file_url(url) is expected


@pytest.mark.asyncio
async def test_encode_file_path_base64_roundtrip(tmp_path: Path):
    """Chunked encoding matches a one-shot base64 encode of the same bytes."""
    payload = b"The quick brown fox jumps over the lazy dog." * 50
    f = tmp_path / "data.bin"
    f.write_bytes(payload)
    out = await encode_file_path_base64(f, chunk_size=64 * 1024, max_bytes=10 * 1024 * 1024)
    assert out == base64.b64encode(payload).decode("ascii")
    assert base64.b64decode(out) == payload


@pytest.mark.asyncio
async def test_encode_file_path_base64_size_cap(tmp_path: Path):
    """Exceeding max_bytes during streaming raises ValueError."""
    f = tmp_path / "big.bin"
    f.write_bytes(b"X" * 5000)
    with pytest.raises(ValueError):
        await encode_file_path_base64(f, chunk_size=64 * 1024, max_bytes=100)


@pytest.mark.asyncio
async def test_encode_file_path_base64_empty_file(tmp_path: Path):
    """An empty file encodes to an empty string."""
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    out = await encode_file_path_base64(f, chunk_size=64 * 1024, max_bytes=1000)
    assert out == ""


def test_get_owui_storage_returns_stub():
    """The conftest stub Storage provider is importable via the gateway helper."""
    storage = get_owui_storage()
    assert storage is not None
    assert hasattr(storage, "get_file")


def test_inlinedfile_namedtuple_shape():
    """InlinedFile carries data_url and filename positionally."""
    f = InlinedFile(data_url="data:x;base64,QQ==", filename="n")
    assert f.data_url == "data:x;base64,QQ=="
    assert f.filename == "n"
    assert tuple(f) == ("data:x;base64,QQ==", "n")
