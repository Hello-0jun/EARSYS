"""UDS (Unix Domain Socket) bridge for EyeFrame datagrams."""

from __future__ import annotations

import errno
import logging
import socket
import struct
import time

from earsys.config import (
    EYE_FRAME_FORMAT,
    EYE_FRAME_MAGIC,
    EYE_FRAME_VERSION,
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
    settings,
)

logger = logging.getLogger(__name__)
RECEIVER_UNAVAILABLE_ERRNOS = {errno.ECONNREFUSED, errno.ENOENT}


def _format_socket_addr(addr: str | bytes) -> str:
    """Return a readable address for logs."""
    if isinstance(addr, bytes) and addr.startswith(b"\x00"):
        return "abstract:" + addr[1:].decode(errors="replace")
    return f"path:{addr}"


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
    span = settings.ear_open_thr - settings.ear_closed_thr
    return max(0.0, min(1.0, (settings.ear_open_thr - ear) / span))


def _pack_eye_frame(status: int, eye_score: float, seq: int, ts_ms: int) -> bytes:
    """Serialize an EyeFrame datagram."""
    return struct.pack(
        EYE_FRAME_FORMAT,
        EYE_FRAME_MAGIC,
        EYE_FRAME_VERSION,
        status,
        0,
        eye_score,
        seq,
        ts_ms,
    )


class UdsBridge:
    """
    Class that sends EyeFrame packets to the configured UDS socket.

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
        uds_addr = settings.uds_socket_addr

        frame = _pack_eye_frame(status=status, eye_score=eye_score, seq=self._seq, ts_ms=ts_ms)

        try:
            self._sock.sendto(frame, uds_addr)
            logger.debug(
                "[dim cyan][uds][/dim cyan] fused_score sent: status=[bold]%s[/bold](code=[cyan]%d[/cyan]) "
                "ear=[yellow]%.3f[/yellow] eye_score=[yellow]%.3f[/yellow] seq=[magenta]%d[/magenta]",
                status_name,
                status,
                ear,
                eye_score,
                self._seq,
            )
            if self._was_unavailable:
                logger.info("[bold green][uds] delivery recovered:[/bold green] %s", _format_socket_addr(uds_addr))
                self._was_unavailable = False
                self._drop_count = 0
        except OSError as exc:
            if exc.errno in RECEIVER_UNAVAILABLE_ERRNOS:
                self._handle_unavailable_receiver(exc)
                return

            logger.warning("[bold red][uds] sendto error:[/bold red] %s", exc)

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

    def __enter__(self) -> UdsBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            logger.info(
                "[bold blue][uds] UdsBridge socket created[/bold blue] -> %s",
                _format_socket_addr(settings.uds_socket_addr),
            )
        except OSError as exc:
            logger.error("[bold red][uds] socket creation failed:[/bold red] %s", exc)
            self._sock = None

    def _handle_unavailable_receiver(self, exc: OSError) -> None:
        self._was_unavailable = True
        self._drop_count += 1
        now = time.time()
        if (now - self._last_drop_log_ts) < 5.0:
            return

        logger.warning(
            "[bold yellow][uds] receiver unavailable[/bold yellow] addr=[cyan]%s[/cyan] "
            "errno=[red]%s[/red] dropped=[bold red]%d[/bold red]",
            _format_socket_addr(settings.uds_socket_addr),
            exc.errno,
            self._drop_count,
        )
        self._last_drop_log_ts = now
        self._drop_count = 0
