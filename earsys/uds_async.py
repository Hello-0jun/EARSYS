"""
비동기 UDS 전송 브리지.

기존의 `UdsBridge` 인스턴스를 내부에 보유하고, 별도 워커 스레드에서
전송을 수행하여 메인 감지 루프의 블로킹을 방지합니다.
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
    """비동기 전송을 제공하는 UDS 브리지 래퍼.

    컨텍스트 매니저를 지원하며 `send()` 는 메인 스레드에서 빠르게 큐에 넣습니다.
    내부 워커가 큐를 소비하여 실제 `UdsBridge.send()` 를 호출합니다.
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
        """비동기으로 상태를 큐에 넣습니다. 큐가 포화면 내부적으로 드롭합니다."""
        try:
            self._queue.put_nowait((status, ear))
        except queue.Full:
            self._drop_count += 1
            if (self._drop_count & 0xF) == 0:
                logger.warning("[uds-async] queue full, dropped=%d", self._drop_count)

    def close(self) -> None:
        """워커 종료를 요청하고 내부 브리지를 닫습니다."""
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
        """워커 루프: 큐에서 메시지를 꺼내 실제 전송을 수행합니다."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                status, ear = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._bridge.send(status=status, ear=ear)
            except Exception as exc:  # 보내는 도중 발생하는 예외는 로깅 후 무시
                logger.exception("[uds-async] send failed: %s", exc)
            finally:
                self._queue.task_done()

        # 워커 종료
        logger.debug("[uds-async] worker exiting")
