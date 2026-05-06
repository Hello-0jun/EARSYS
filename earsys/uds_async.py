"""
Asynchronous UDS transmission bridge.

Holds an internal UdsBridge instance and sends data from a separate worker thread
to keep the main detection loop from blocking.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .uds_bridge import UdsBridge

logger = logging.getLogger(__name__)


class UdsAsyncBridge:
    """UDS bridge wrapper that provides asynchronous sending.

    Supports the context manager protocol and queues `send()` calls quickly on the main thread.
    The internal worker consumes the queue and invokes `UdsBridge.send()`.
    """

    def __init__(self, max_queue: int = 512, worker_join_timeout: float = 2.0) -> None:
        self._bridge = UdsBridge()
        self._queue: "queue.Queue[tuple[int, float]]" = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = threading.Thread(
            target=self._run, name="UdsAsyncBridge-worker", daemon=True
        )
        self._worker.start()
        self._worker_join_timeout = worker_join_timeout
        self._drop_count = 0

    def send(self, status: int, ear: float) -> None:
        """Queue a status update asynchronously. Drop internally if the queue is full."""
        try:
            logger.info("[uds-async] fused_score received: status=%s ear=%.3f", status, ear)
            self._queue.put_nowait((status, ear))
        except queue.Full:
            self._drop_count += 1
            if (self._drop_count & 0xF) == 0:
                logger.warning("[uds-async] queue full, dropped=%d", self._drop_count)

    def close(self) -> None:
        """Request worker shutdown and close the internal bridge."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(self._worker_join_timeout)
        try:
            self._bridge.close()
        except Exception:
            pass

    def __enter__(self) -> "UdsAsyncBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        """Worker loop: pull messages from the queue and send them."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                status, ear = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                logger.info("[uds-async] fused_score sent: status=%s ear=%.3f", status, ear)
                self._bridge.send(status=status, ear=ear)
            except Exception as exc:  # Log and ignore exceptions raised during send.
                logger.exception("[uds-async] send failed: %s", exc)
            finally:
                self._queue.task_done()

        # Worker exit
        logger.debug("[uds-async] worker exiting")
