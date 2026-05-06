"""
UDS (Unix Domain Socket) bridge.

Sends EyeFrame packets as SOCK_DGRAM datagrams to the @sleepcare/eye abstract-namespace socket
bound by sleepcare-ws.

If the receiver (sleepcare-ws) is not running, sendto returns ECONNREFUSED and the packet is dropped quietly.
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
    STATUS_DROWSY,
    STATUS_NO_FACE,
    UDS_EYE_ADDR,
)

logger = logging.getLogger(__name__)


def _status_name(status: int) -> str:
    """Convert a status code to a name string."""
    status_map = {
        STATUS_AWAKE: "AWAKE",
        STATUS_DROWSY: "DROWSY",
        STATUS_NO_FACE: "NO_FACE",
    }
    return status_map.get(status, f"UNKNOWN({status})")


def _ear_to_score(ear: float) -> float:
    """Convert an EAR value to eye_score (0.0 to 1.0)."""
    span = EAR_OPEN_THR - EAR_CLOSED_THR
    return max(0.0, min(1.0, (EAR_OPEN_THR - ear) / span))


class UdsBridge:
    """
    Class that sends EyeFrame packets to the @sleepcare/eye UDS socket.

    Supports the context manager protocol:
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
    """

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._seq: int = 0
        self._drop_count: int = 0
        self._last_drop_log_ts: float = 0.0
        self._was_unavailable: bool = False
        self._open()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(self, status: int, ear: float) -> None:
        """
        Build and send an EyeFrame.

        Parameters:
            status: STATUS_* constant (0=awake, 1=drowsy, 2=no-face)
            ear: EAR value (float). Pass 0.0 when no face is detected.
        """
        if self._sock is None:
            return

        eye_score = _ear_to_score(ear)
        self._seq += 1
        ts_ms = int(time.time() * 1000)
        status_name = _status_name(status)

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
            logger.info(
                "[uds] fused_score sent: status=%s(code=%d) ear=%.3f eye_score=%.3f seq=%d",
                status_name,
                status,
                ear,
                eye_score,
                self._seq,
            )
            if self._was_unavailable:
                logger.info("[uds] @sleepcare/eye delivery recovered")
                self._was_unavailable = False
                self._drop_count = 0
        except OSError as exc:
            # ECONNREFUSED: sleepcare-ws is not running, drop quietly.
            if exc.errno in (111, 2):  # 111 = ECONNREFUSED, 2 = ENOENT
                self._was_unavailable = True
                self._drop_count += 1
                now = time.time()
                if (now - self._last_drop_log_ts) >= 5.0:
                    logger.warning("[uds] @sleepcare/eye unavailable(errno=%s), dropped=%d", exc.errno, self._drop_count)
                    self._last_drop_log_ts = now
                    self._drop_count = 0
                return

            if exc.errno not in (111,):  # 111 = ECONNREFUSED
                logger.warning("[uds] sendto error: %s", exc)

    def close(self) -> None:
        """Close the socket. The abstract namespace is released automatically."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            finally:
                self._sock = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "UdsBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            # No bind is needed because this is send-only.
            logger.info("[uds] UdsBridge socket created -> @sleepcare/eye")
        except OSError as exc:
            logger.error("[uds] socket creation failed: %s", exc)
            self._sock = None
