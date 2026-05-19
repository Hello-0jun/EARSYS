"""Unit tests for EyeFrame UDS serialization and delivery behavior."""

from __future__ import annotations

import errno
import struct
import time

import pytest

import earsys.ipc.uds_bridge as uds_bridge_module
from earsys.config import (
    EYE_FRAME_FORMAT,
    EYE_FRAME_MAGIC,
    EYE_FRAME_SIZE,
    EYE_FRAME_VERSION,
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
    settings,
)
from earsys.ipc.uds_bridge import UdsBridge, _ear_to_score

EAR_OPEN_THR = settings.ear_open_thr
EAR_CLOSED_THR = settings.ear_closed_thr


@pytest.mark.parametrize(
    ("ear", "expected"),
    [
        (EAR_OPEN_THR, 0.0),
        (EAR_OPEN_THR + 0.1, 0.0),
        ((EAR_OPEN_THR + EAR_CLOSED_THR) / 2, 0.5),
        (EAR_CLOSED_THR, 1.0),
        (EAR_CLOSED_THR - 0.1, 1.0),
    ],
)
def test_ear_to_score_is_clamped_linear_mapping(ear, expected):
    assert _ear_to_score(ear) == pytest.approx(expected)


@pytest.mark.parametrize("ear", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_ear_to_score_stays_in_unit_interval(ear):
    assert 0.0 <= _ear_to_score(ear) <= 1.0


@pytest.fixture()
def sent_datagrams(monkeypatch):
    sent: list[tuple[bytes, str]] = []

    class FakeSocket:
        def sendto(self, data, addr):
            sent.append((data, addr))
            return len(data)

        def close(self):
            pass

    # Patch the settings singleton so uds_socket_addr returns a fixed test address.
    monkeypatch.setattr(settings, "uds_addr", "path:/tmp/earsys-test-eye.sock")
    monkeypatch.setattr(uds_bridge_module.socket, "socket", lambda *_: FakeSocket())
    return sent


def unpack_frame(data: bytes):
    return struct.unpack(EYE_FRAME_FORMAT, data)


def test_send_serializes_expected_eyeframe(sent_datagrams):
    with UdsBridge() as bridge:
        bridge.send(status=STATUS_DROWSY, ear=EAR_CLOSED_THR)

    data, addr = sent_datagrams[0]
    magic, version, status, reserved, eye_score, seq, ts_ms = unpack_frame(data)

    assert addr == "/tmp/earsys-test-eye.sock"
    assert len(data) == EYE_FRAME_SIZE
    assert magic == EYE_FRAME_MAGIC
    assert version == EYE_FRAME_VERSION
    assert status == STATUS_DROWSY
    assert reserved == 0
    assert eye_score == pytest.approx(1.0, abs=1e-5)
    assert seq == 1
    assert abs(ts_ms - int(time.time() * 1000)) < 5000


def test_send_increments_sequence_numbers(sent_datagrams):
    with UdsBridge() as bridge:
        bridge.send(status=STATUS_AWAKE, ear=0.32)
        bridge.send(status=STATUS_AWAKE, ear=0.31)

    seqs = [unpack_frame(data)[5] for data, _ in sent_datagrams]

    assert seqs == [1, 2]


@pytest.mark.parametrize(
    ("status", "ear", "expected_score"),
    [
        (STATUS_AWAKE, EAR_OPEN_THR, 0.0),
        (STATUS_DROWSY, EAR_CLOSED_THR, 1.0),
        (STATUS_NO_FACE, 0.0, 1.0),
    ],
)
def test_send_preserves_status_and_score(status, ear, expected_score, sent_datagrams):
    with UdsBridge() as bridge:
        bridge.send(status=status, ear=ear)

    _, _, packed_status, _, eye_score, _, _ = unpack_frame(sent_datagrams[0][0])

    assert packed_status == status
    assert eye_score == pytest.approx(expected_score, abs=1e-5)


@pytest.mark.parametrize("socket_errno", [errno.ECONNREFUSED, errno.ENOENT])
def test_send_drops_unavailable_receiver_errors(monkeypatch, socket_errno):
    class RefusingSocket:
        def sendto(self, *_):
            raise OSError(socket_errno, "receiver unavailable")

        def close(self):
            pass

    monkeypatch.setattr(uds_bridge_module.socket, "socket", lambda *_: RefusingSocket())

    with UdsBridge() as bridge:
        bridge.send(status=STATUS_AWAKE, ear=0.32)


def test_send_logs_unexpected_socket_errors(monkeypatch, caplog):
    class FailingSocket:
        def sendto(self, *_):
            raise OSError(errno.EPERM, "operation not permitted")

        def close(self):
            pass

    monkeypatch.setattr(uds_bridge_module.socket, "socket", lambda *_: FailingSocket())

    with UdsBridge() as bridge:
        bridge.send(status=STATUS_AWAKE, ear=0.32)

    assert "sendto error" in caplog.text


def test_close_is_idempotent(sent_datagrams):
    bridge = UdsBridge()

    bridge.close()
    bridge.close()

    assert bridge._sock is None
