"""
earsys.shm_bridge 모듈 단위 테스트.

pytest로 실행:
    pytest tests/test_shm_bridge.py -v

테스트 종료 후 /dev/shm/earsys_test_* 파일이 남지 않도록
ShmBridge.close()를 항상 호출합니다.
"""

from __future__ import annotations

import struct

import pytest

from earsys.config import OFF_MAGIC, OFF_SEQ, OFF_STATUS, OFF_VERSION, STATUS_AWAKE, STATUS_DROWSY, STATUS_NO_FACE
from earsys.shm_bridge import ShmBridge

# 테스트 전용 SHM 이름 (실제 운영 SHM과 충돌 방지)
_TEST_SHM = "/earsys_test_shm"


@pytest.fixture()
def shm():
    """테스트마다 새 ShmBridge를 생성하고, 종료 후 SHM을 정리합니다."""
    bridge = ShmBridge(shm_name=_TEST_SHM)
    yield bridge
    bridge.close()


# ---------------------------------------------------------------------------
# 생성 및 헤더 초기화
# ---------------------------------------------------------------------------

class TestShmInit:
    def test_magic_header(self, shm: ShmBridge):
        """SHM magic이 'EARS'로 초기화되어야 합니다."""
        buf = shm._shm.buf
        assert bytes(buf[OFF_MAGIC: OFF_MAGIC + 4]) == b"EARS"

    def test_version_header(self, shm: ShmBridge):
        """SHM 버전이 1로 초기화되어야 합니다."""
        buf = shm._shm.buf
        version = struct.unpack_from("<I", buf, OFF_VERSION)[0]
        assert version == 1

    def test_initial_status_is_awake(self, shm: ShmBridge):
        """초기 상태는 STATUS_AWAKE(0)이어야 합니다."""
        assert shm.read_status() == STATUS_AWAKE

    def test_initial_seq_is_even(self, shm: ShmBridge):
        """초기 시퀀스는 짝수(0)이어야 합니다 (seqlock 규칙)."""
        buf = shm._shm.buf
        seq = struct.unpack_from("<I", buf, OFF_SEQ)[0]
        assert seq % 2 == 0


# ---------------------------------------------------------------------------
# write_status / read_status 라운드트립
# ---------------------------------------------------------------------------

class TestShmReadWrite:
    def test_write_awake(self, shm: ShmBridge):
        shm.write_status(STATUS_AWAKE)
        assert shm.read_status() == STATUS_AWAKE

    def test_write_drowsy(self, shm: ShmBridge):
        shm.write_status(STATUS_DROWSY)
        assert shm.read_status() == STATUS_DROWSY

    def test_write_no_face(self, shm: ShmBridge):
        shm.write_status(STATUS_NO_FACE)
        assert shm.read_status() == STATUS_NO_FACE

    def test_multiple_writes(self, shm: ShmBridge):
        """여러 번 쓰기 후 마지막 값만 읽혀야 합니다."""
        shm.write_status(STATUS_DROWSY)
        shm.write_status(STATUS_NO_FACE)
        shm.write_status(STATUS_AWAKE)
        assert shm.read_status() == STATUS_AWAKE

    def test_seq_increments_on_write(self, shm: ShmBridge):
        """쓰기 후 SEQ가 증가해야 합니다 (짝수 → 더 큰 짝수)."""
        buf = shm._shm.buf
        seq_before = struct.unpack_from("<I", buf, OFF_SEQ)[0]
        shm.write_status(STATUS_DROWSY)
        seq_after = struct.unpack_from("<I", buf, OFF_SEQ)[0]
        assert seq_after > seq_before
        assert seq_after % 2 == 0  # 완료 상태는 항상 짝수


# ---------------------------------------------------------------------------
# 생성자 / 연결자 구분
# ---------------------------------------------------------------------------

class TestShmCreatorFlag:
    def test_creator_flag_is_true(self, shm: ShmBridge):
        """첫 번째 생성자는 _is_creator가 True여야 합니다."""
        assert shm._is_creator is True

    def test_second_open_is_not_creator(self, shm: ShmBridge):
        """같은 이름으로 두 번째 연결은 _is_creator가 False여야 합니다."""
        second = ShmBridge(shm_name=_TEST_SHM)
        try:
            assert second._is_creator is False
        finally:
            second.close()

    def test_second_reads_written_value(self, shm: ShmBridge):
        """생성자가 쓴 값을 연결자가 올바르게 읽어야 합니다."""
        shm.write_status(STATUS_DROWSY)
        reader = ShmBridge(shm_name=_TEST_SHM)
        try:
            assert reader.read_status() == STATUS_DROWSY
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# 컨텍스트 매니저
# ---------------------------------------------------------------------------

class TestShmContextManager:
    def test_context_manager_closes(self):
        """with 블록 종료 후 _shm이 None이어야 합니다."""
        with ShmBridge(shm_name=_TEST_SHM) as bridge:
            assert bridge._shm is not None
        assert bridge._shm is None

    def test_close_idempotent(self, shm: ShmBridge):
        """close()를 여러 번 호출해도 예외가 없어야 합니다."""
        shm.close()
        shm.close()  # 두 번째 호출: 예외 없어야 함
