"""
earsys.uds_bridge 모듈 단위 테스트.

pytest로 실행:
    pytest tests/test_uds_bridge.py -v

테스트 전략:
  - _ear_to_score 변환 함수: 경계값 + 선형 보간 검증
  - UdsBridge.send: 로컬 바인딩 서버 소켓으로 실제 전송 검증
  - 수신자 없을 때 예외 없음 (graceful drop) 검증
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
        """EAR >= EAR_OPEN_THR → score = 0.0"""
        assert _ear_to_score(EAR_OPEN_THR) == pytest.approx(0.0)
        assert _ear_to_score(EAR_OPEN_THR + 0.1) == pytest.approx(0.0)

    def test_fully_closed(self):
        """EAR <= EAR_CLOSED_THR → score = 1.0"""
        assert _ear_to_score(EAR_CLOSED_THR) == pytest.approx(1.0)
        assert _ear_to_score(EAR_CLOSED_THR - 0.1) == pytest.approx(1.0)

    def test_midpoint(self):
        """중간 EAR 값은 0.5에 가까워야 한다."""
        mid = (EAR_OPEN_THR + EAR_CLOSED_THR) / 2
        assert _ear_to_score(mid) == pytest.approx(0.5, abs=1e-6)

    def test_clamped_to_range(self):
        """반환값은 항상 [0.0, 1.0] 범위이어야 한다."""
        for ear in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            score = _ear_to_score(ear)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# UdsBridge.send — 실제 소켓 전송 검증
# ---------------------------------------------------------------------------

@pytest.fixture()
def receiver():
    """
    @sleepcare/eye 에 바인딩한 수신 소켓.
    테스트 종료 후 자동으로 닫힌다.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(UDS_EYE_ADDR)
    sock.settimeout(2.0)
    yield sock
    sock.close()


class TestUdsBridgeSend:
    def test_frame_size(self, receiver):
        """전송된 프레임은 정확히 EYE_FRAME_SIZE(20) 바이트이어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        assert len(data) == EYE_FRAME_SIZE

    def test_magic(self, receiver):
        """magic 필드는 b'SEYE' 이어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        magic, ver, status, reserved, eye_score, seq, ts_ms = struct.unpack(
            EYE_FRAME_FORMAT, data
        )
        assert magic == EYE_FRAME_MAGIC

    def test_version(self, receiver):
        """version 필드는 1이어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.32)
        data, _ = receiver.recvfrom(64)
        _, ver, *_ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert ver == 1

    def test_status_drowsy(self, receiver):
        """status=STATUS_DROWSY 가 올바르게 전송되어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_DROWSY, ear=0.10)
        data, _ = receiver.recvfrom(64)
        _, _, status, _, eye_score, seq, ts_ms = struct.unpack(EYE_FRAME_FORMAT, data)
        assert status == STATUS_DROWSY

    def test_eye_score_range(self, receiver):
        """eye_score 는 [0.0, 1.0] 범위이어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.25)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert 0.0 <= eye_score <= 1.0

    def test_eye_score_open_eye(self, receiver):
        """열린 눈(EAR >= OPEN_THR) → eye_score ≈ 0.0"""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=EAR_OPEN_THR)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert eye_score == pytest.approx(0.0, abs=1e-5)

    def test_eye_score_closed_eye(self, receiver):
        """닫힌 눈(EAR <= CLOSED_THR) → eye_score ≈ 1.0"""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_DROWSY, ear=EAR_CLOSED_THR)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert eye_score == pytest.approx(1.0, abs=1e-5)

    def test_seq_monotonic(self, receiver):
        """seq 필드는 전송마다 단조 증가해야 한다."""
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
        """ts_ms 는 현재 시각 ±5초 이내이어야 한다."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_AWAKE, ear=0.30)
        data, _ = receiver.recvfrom(64)
        _, _, _, _, _, _, ts_ms = struct.unpack(EYE_FRAME_FORMAT, data)
        now_ms = int(time.time() * 1000)
        assert abs(ts_ms - now_ms) < 5000

    def test_no_face_status(self, receiver):
        """얼굴 미검출(STATUS_NO_FACE, ear=0.0) 전송 검증."""
        with UdsBridge() as bridge:
            bridge.send(status=STATUS_NO_FACE, ear=0.0)
        data, _ = receiver.recvfrom(64)
        _, _, status, _, eye_score, _, _ = struct.unpack(EYE_FRAME_FORMAT, data)
        assert status == STATUS_NO_FACE
        assert eye_score == pytest.approx(1.0, abs=1e-5)  # ear=0.0 → score=1.0


# ---------------------------------------------------------------------------
# Graceful drop (수신자 없을 때 예외 없음)
# ---------------------------------------------------------------------------

class TestGracefulDrop:
    def test_no_receiver_no_exception(self):
        """@sleepcare/eye 수신자가 없어도 예외가 발생하지 않아야 한다."""
        with UdsBridge() as bridge:
            # 수신자 없음 → ECONNREFUSED 를 내부에서 흡수
            for _ in range(5):
                bridge.send(status=STATUS_AWAKE, ear=0.32)


# ---------------------------------------------------------------------------
# 컨텍스트 매니저
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_close_on_exit(self):
        """with 블록 종료 후 _sock 이 None 이어야 한다."""
        with UdsBridge() as bridge:
            assert bridge._sock is not None
        assert bridge._sock is None

    def test_close_idempotent(self):
        """close() 를 여러 번 호출해도 예외가 없어야 한다."""
        bridge = UdsBridge()
        bridge.close()
        bridge.close()
