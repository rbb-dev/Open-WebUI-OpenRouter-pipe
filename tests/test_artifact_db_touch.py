from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest


class _FakeColumn:
    def __init__(self, name: str):
        self.name = name

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeColumn) and self.name == other.name

    def in_(self, items: list[str]) -> tuple[str, str, tuple[str, ...]]:
        return ("in", self.name, tuple(items))


class _FakeModel:
    id = _FakeColumn("id")
    chat_id = _FakeColumn("chat_id")
    message_id = _FakeColumn("message_id")
    created_at = _FakeColumn("created_at")


@dataclass
class _FakeRow:
    id: str
    payload: dict[str, Any]
    is_encrypted: bool = False


class _FakeQuery:
    def __init__(self, rows: list[_FakeRow]):
        self._rows = rows
        self.filters: list[Any] = []
        self.updated: Optional[dict[Any, Any]] = None

    def filter(self, expr: Any) -> "_FakeQuery":
        self.filters.append(expr)
        return self

    def all(self) -> list[_FakeRow]:
        return list(self._rows)

    def update(self, mapping: dict[Any, Any], synchronize_session: bool = False) -> int:
        self.updated = {"mapping": mapping, "synchronize_session": synchronize_session}
        return len(self._rows)


class _FakeSession:
    def __init__(self, rows: list[_FakeRow], *, update_raises: bool = False):
        self._rows = rows
        self._update_raises = update_raises
        self.query_obj: Optional[_FakeQuery] = None
        self.did_commit = False
        self.did_rollback = False

    def query(self, _model: Any) -> _FakeQuery:
        query = _FakeQuery(self._rows)
        if self._update_raises:
            original_update = query.update

            def _raising_update(mapping: dict[Any, Any], synchronize_session: bool = False) -> int:
                _ = original_update(mapping, synchronize_session=synchronize_session)
                raise RuntimeError("update failed")

            query.update = _raising_update  # type: ignore[method-assign]
        self.query_obj = query
        return query

    def commit(self) -> None:
        self.did_commit = True

    def rollback(self) -> None:
        self.did_rollback = True

    def close(self) -> None:
        return None


def test_db_fetch_sync_touches_created_at_best_effort(pipe_instance) -> None:
    rows = [
        _FakeRow(id="a1", payload={"type": "tool_output", "output": "ok"}),
        _FakeRow(id="b2", payload={"type": "tool_output", "output": "ok"}),
    ]
    read_session = _FakeSession(rows)
    touch_session = _FakeSession(rows)
    sessions = [read_session, touch_session]

    pipe_instance._item_model = _FakeModel
    pipe_instance._session_factory = lambda: sessions.pop(0)

    results = pipe_instance._db_fetch_sync("chat123", None, ["a1", "b2"])
    assert results == {"a1": rows[0].payload, "b2": rows[1].payload}
    assert touch_session.did_commit is True
    assert touch_session.query_obj is not None
    assert ("in", "id", ("a1", "b2")) in touch_session.query_obj.filters
    assert touch_session.query_obj.updated is not None


def test_db_fetch_sync_touch_failure_never_breaks_read(pipe_instance) -> None:
    rows = [_FakeRow(id="a1", payload={"type": "tool_output", "output": "ok"})]
    read_session = _FakeSession(rows)
    touch_session = _FakeSession(rows, update_raises=True)
    sessions = [read_session, touch_session]

    pipe_instance._item_model = _FakeModel
    pipe_instance._session_factory = lambda: sessions.pop(0)

    results = pipe_instance._db_fetch_sync("chat123", None, ["a1"])
    assert results == {"a1": rows[0].payload}
    assert touch_session.did_rollback is True
