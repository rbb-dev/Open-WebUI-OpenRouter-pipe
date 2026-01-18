import logging

from open_webui_openrouter_pipe import Pipe


def test_shutdown_db_executor_non_blocking_by_default() -> None:
    pipe = Pipe()
    pipe.logger = logging.getLogger("tests.shutdown")

    calls: list[tuple[bool, bool | None]] = []

    class FakeExecutor:
        def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
            calls.append((wait, cancel_futures))

    pipe._db_executor = FakeExecutor()  # type: ignore[assignment]
    pipe.shutdown()

    assert pipe._db_executor is None
    assert calls == [(False, True)]


def test_shutdown_falls_back_when_cancel_futures_unsupported() -> None:
    pipe = Pipe()
    pipe.logger = logging.getLogger("tests.shutdown")

    calls: list[bool] = []

    class FakeExecutor:
        def shutdown(self, *, wait: bool = True) -> None:
            calls.append(wait)

    pipe._db_executor = FakeExecutor()  # type: ignore[assignment]
    pipe.shutdown()

    assert pipe._db_executor is None
    assert calls == [False]


def test_shutdown_tolerates_executor_shutdown_exceptions() -> None:
    pipe = Pipe()
    pipe.logger = logging.getLogger("tests.shutdown")

    class FakeExecutor:
        def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
            raise RuntimeError("boom")

    pipe._db_executor = FakeExecutor()  # type: ignore[assignment]
    pipe.shutdown()
    assert pipe._db_executor is None
