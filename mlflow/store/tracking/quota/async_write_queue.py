"""Bounded queue that offloads fire-and-forget tracking writes to a worker.

Unlike :class:`mlflow.utils.async_logging.async_logging_queue.AsyncLoggingQueue`,
which is specialized for ``log_batch`` payloads, this queue dispatches arbitrary
zero-return store write methods. It applies backpressure when full (a bounded
``put`` blocks instead of growing memory without limit) and re-raises the first
background failure on the next ``flush``/``close`` call so that errors are never
silently swallowed.
"""

import atexit
import logging
import threading
from queue import Empty, Queue
from typing import Any, Callable

from mlflow.environment_variables import MLFLOW_QUOTA_ASYNC_QUEUE_MAXSIZE

_logger = logging.getLogger(__name__)


class AsyncWriteQueue:
    def __init__(self, maxsize: int | None = None):
        self._queue: Queue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = Queue(
            maxsize=MLFLOW_QUOTA_ASYNC_QUEUE_MAXSIZE.get() if maxsize is None else maxsize
        )
        self._worker: threading.Thread | None = None
        self._started = threading.Event()
        self._stopping = threading.Event()
        self._pending = threading.Condition()
        self._inflight = 0
        self._first_error: Exception | None = None
        self._closed = False
        atexit.register(self.close)

    def start(self) -> None:
        if self._started.is_set() or self._closed:
            return
        self._worker = threading.Thread(
            target=self._consume,
            name="MLflowQuotaAsyncWriteWorker",
            daemon=True,
        )
        self._started.set()
        self._worker.start()

    def submit(self, func: Callable[..., Any], *args, **kwargs) -> None:
        if self._closed:
            raise RuntimeError("Async write queue is closed; create a new store to resume writes")
        self.start()
        with self._pending:
            self._inflight += 1
        self._queue.put((func, args, kwargs))

    def _consume(self) -> None:
        while True:
            if self._stopping.is_set() and self._queue.empty():
                return
            try:
                func, args, kwargs = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                func(*args, **kwargs)
            except Exception as exc:
                _logger.error("Background tracking write failed: %s", exc, exc_info=True)
                with self._pending:
                    if self._first_error is None:
                        self._first_error = exc
            finally:
                with self._pending:
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._pending.notify_all()

    def flush(self) -> None:
        """Block until every submitted write has been processed."""
        with self._pending:
            self._pending.wait_for(lambda: self._inflight == 0)
        error = self._first_error
        self._first_error = None
        if error is not None:
            raise error

    def close(self) -> None:
        if not self._started.is_set() or self._closed:
            self._closed = True
            return
        self._stopping.set()
        if self._worker is not None:
            self._worker.join()
        self._closed = True
        error = self._first_error
        self._first_error = None
        if error is not None:
            raise error
