"""
Unit tests for the earsys.uds_bridge module.

Run with pytest:
        pytest tests/test_uds_bridge.py -v

Test strategy:
    - _ear_to_score conversion: verify boundary values and linear interpolation
    - UdsBridge.send: verify actual delivery to a locally bound server socket
    - No exception when the receiver is absent (graceful drop)
"""

from __future__ import annotations

import socket
import struct
import time

import pytest

from earsys.config import (
    EAR_CLOSED_THR,
    EAR_OPEN_THR,
    EYE_FRAME_FORMAT,
    EYE_FRAME_MAGIC,
    EYE_FRAME_SIZE,
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
    UDS_EYE_ADDR,
)
from earsys.uds_bridge import UdsBridge, _ear_to_score


# ---------------------------------------------------------------------------
# _ear_to_score
# ---------------------------------------------------------------------------

class TestEarToScore:
    def test_fully_open(self):
        """EAR >= EAR_OPEN_THR -> score = 0.0"""
        assert _ear_to_score(EAR_OPEN_THR) == pytest.approx(0.0)
        assert _ear_to_score(EAR_OPEN_THR + 0.1) == pytest.approx(0.0)

    def test_fully_closed(self):
        """EAR <= EAR_CLOSED_THR -> score = 1.0"""
        assert _ear_to_score(EAR_CLOSED_THR) == pytest.approx(1.0)
        assert _ear_to_score(EAR_CLOSED_THR - 0.1) == pytest.approx(1.0)

    def test_midpoint(self):
        """A midpoint EAR value should be close to 0.5."""
        mid = (EAR_OPEN_THR + EAR_CLOSED_THR) / 2
        assert _ear_to_score(mid) == pytest.approx(0.5, abs=1e-6)

    def test_clamped_to_range(self):
        """The return value should always be within [0.0, 1.0]."""
        for ear in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            score = _ear_to_score(ear)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# UdsBridge.send - verify actual socket transmission
# ---------------------------------------------------------------------------

@pytest.fixture()
def receiver():
    """
    Receiver socket bound to @sleepcare/eye.
    It is closed automatically after the test finishes.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(UDS_EYE_ADDR)
    sock.settimeout(2.0)
    yield sock
    sock.close()


class TestUdsBridgeSend:
    def test_frame_size(self, receiver):
        """The transmitted frame must be exactly EYE_FRAME_SIZE (20) bytes."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        assert len(data) == EYE_FRAME_SIZE

    def test_magic(self, receiver):
        """The magic field must be b'SEYE'."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        magic, ver, status, reserved, eye_score, seq, ts_ms = struct.unpack(
            EYE_FRAME_FORMAT, data
        )
        assert magic == EYE_FRAME_MAGIC

    def test_version(self, receiver):
        """The version field must be 1."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        _, ver, *_ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert ver == 1

    def test_status_drowsy(self, receiver):
        """status=STATUS_DROWSY must be transmitted correctly."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_DROWSY, ear=0.10)
        data, _ = receiver.recvfrom(64)
        _, _, status, _, eye_score, seq, ts_ms = struct.unpack(EYE_FRAME_FORMAT, data)
        assert status == STATUS_DROWSY

    def test_eye_score_range(self, receiver):
        """eye_score must be within the [0.0, 1.0] range."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.25)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert 0.0 <= eye_score <= 1.0

    def test_eye_score_open_eye(self, receiver):
        """Open eyes (EAR >= OPEN_THR) -> eye_score ≈ 0.0"""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=EAR_OPEN_THR)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert eye_score == pytest.approx(0.0, abs=1e-5)

    def test_eye_score_closed_eye(self, receiver):
        """Closed eyes (EAR <= CLOSED_THR) -> eye_score ≈ 1.0"""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_DROWSY, ear=EAR_CLOSED_THR)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert eye_score == pytest.approx(1.0, abs=1e-5)

    def test_seq_monotonic(self, receiver):
        """The seq field must increase monotonically with each send."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
            bridge.send(status=STATUS_AWAKE, ear=0.31)
        seqs = []
        for _ in range(2):
            data, _ = receiver.recvfrom(64)
            _, _, _, _, _, seq, _ = struct.unpack(EYE_FRAME_FORMAT, data)
            seqs.append(seq)
        assert seqs[1] > seqs[0]

    def test_ts_ms_reasonable(self, receiver):
        """ts_ms must be within 5 seconds of the current time."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.30)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, _, _, ts_ms = struct.unpack(EYE_FRAME_FORMAT, data)
        now_ms = int(time.time() * 1000)
        assert abs(ts_ms - now_ms) < 5000

    def test_no_face_status(self, receiver):
        """Verify sending no-face status (STATUS_NO_FACE, ear=0.0)."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_NO_FACE, ear=0.0)
        data, _ = receiver.recvfrom(64)
        _, _, status, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert status == STATUS_NO_FACE
        assert eye_score == pytest.approx(1.0, abs=1e-5)  # ear=0.0 -> score=1.0


# ---------------------------------------------------------------------------
# Graceful drop (no exception when the receiver is absent)
# ---------------------------------------------------------------------------

class TestGracefulDrop:
    def test_no_receiver_no_exception(self):
        """No exception should be raised even when @sleepcare/eye is absent."""
        with UdsBridge() as bridge:
            # No receiver -> ECONNREFUSED is handled internally.
            for _ in range(5):
                bridge.send(status=STATUS_AWAKE, ear=0.32)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_close_on_exit(self):
        """_sock should be None after the with block exits."""
        with UdsBridge() as bridge:
            assert bridge._sock is not None
        assert bridge._sock is None

    def test_close_idempotent(self):
        """Calling close() multiple times should not raise an exception."""
        bridge = UdsBridge()
        bridge.close()
        bridge.close()
