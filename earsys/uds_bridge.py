"""
UDS(Unix Domain Socket) 브리지.

sleepcare-ws 가 바인딩한 @sleepcare/eye abstract-namespace 소켓으로
EyeFrame 을 SOCK_DGRAM 으로 전송한다.

수신자(sleepcare-ws)가 실행 중이 아니면 sendto 가 ECONNREFUSED 를 반환하며,
이 경우 예외 없이 조용히 드롭한다.
"""

from __future__ import annotations

import logging
import socket
import struct
import time

from earsys.config import (
    EAR_CLOSED_THR,
    EAR_OPEN_THR,
    EYE_FRAME_FORMAT,
    EYE_FRAME_MAGIC,
    EYE_FRAME_VERSION,
    STATUS_AWAKE,
    UDS_EYE_ADDR,
)

logger = logging.getLogger(__name__)


def _ear_to_score(ear: float) -> float:
    """EAR 값을 eye_score (0.0~1.0)로 변환한다."""
    span = EAR_OPEN_THR - EAR_CLOSED_THR
    return max(0.0, min(1.0, (EAR_OPEN_THR - ear) / span))


class UdsBridge:
    """
    EyeFrame 을 @sleepcare/eye UDS 소켓으로 전송하는 클래스.

    컨텍스트 매니저를 지원한다:
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
    """

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._seq: int = 0
        self._open()

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def send(self, status: int, ear: float) -> None:
        """
        EyeFrame 을 생성하여 전송한다.

        매개변수:
            status: STATUS_* 상수 (0=awake, 1=drowsy, 2=no-face)
            ear:    EAR 값 (float). 얼굴 미검출 시 0.0 전달.
        """
        if self._sock is None:
            return

        eye_score = _ear_to_score(ear)
        self._seq += 1
        ts_ms = int(time.time() * 1000)

        frame = struct.pack(
            EYE_FRAME_FORMAT,
            EYE_FRAME_MAGIC,    # 4s  magic
            EYE_FRAME_VERSION,  # B   version
            status,             # B   status
            0,                  # H   reserved
            eye_score,          # f   eye_score
            self._seq,          # I   seq
            ts_ms,              # Q   ts_ms
        )

        try:
            self._sock.sendto(frame, UDS_EYE_ADDR)
        except OSError as exc:
            # ECONNREFUSED: sleepcare-ws 미실행, 조용히 드롭
            if exc.errno not in (111,):  # 111 = ECONNREFUSED
                logger.warning("[uds] sendto 오류: %s", exc)

    def close(self) -> None:
        """소켓을 닫는다. abstract namespace는 자동으로 해제된다."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            finally:
                self._sock = None

    # ------------------------------------------------------------------
    # 컨텍스트 매니저 지원
    # ------------------------------------------------------------------

    def __enter__(self) -> "UdsBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _open(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            # 발신 전용이므로 bind 불필요
            logger.info("[uds] UdsBridge 소켓 생성 완료 → @sleepcare/eye")
        except OSError as exc:
            logger.error("[uds] 소켓 생성 실패: %s", exc)
            self._sock = None
