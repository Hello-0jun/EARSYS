"""
POSIX 공유 메모리 브리지 (seqlock 기반).

SHM 레이아웃:
    [0 :4 ]  magic    = b"EARS"   (4 bytes)
    [4 :8 ]  version  = 1         (uint32 LE)
    [8 :12]  seq      = seqlock   (uint32 LE)  홀수=쓰기 중, 짝수=완료
    [12:16]  status   = 상태 코드 (uint32 LE)

Consumer 측 읽기 패턴 (C/C++ 의사코드):
    do {
        seq1 = read_u32(OFF_SEQ);
        if (seq1 & 1) continue;   // 쓰기 중
        status = read_u32(OFF_STATUS);
        seq2 = read_u32(OFF_SEQ);
    } while (seq1 != seq2);       // 재시도
"""

from __future__ import annotations

import logging
import struct
from multiprocessing import shared_memory

from earsys.config import (
    OFF_MAGIC,
    OFF_SEQ,
    OFF_STATUS,
    OFF_VERSION,
    SHM_NAME,
    SHM_PROTOCOL_VERSION,
    SHM_SIZE,
    STATUS_AWAKE,
)

logger = logging.getLogger(__name__)


class ShmBridge:
    """
    EARSYS POSIX 공유 메모리를 관리하는 클래스.

    생성자가 SHM을 생성(또는 기존 SHM에 연결)합니다.
    이 인스턴스가 SHM을 생성한 경우 close() 시 unlink()도 수행합니다.
    """

    def __init__(
        self,
        shm_name: str = SHM_NAME,
        size: int = SHM_SIZE,
    ) -> None:
        self._name = shm_name.lstrip("/")
        self._size = size
        self._shm: shared_memory.SharedMemory | None = None
        self._is_creator = False
        self._seq: int = 0

        self._open()

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def write_status(self, status_code: int) -> None:
        """
        seqlock 방식으로 상태 코드를 SHM에 원자적으로 기록합니다.

        홀수 seq → 쓰기 중, 짝수 seq → 쓰기 완료.
        """
        if self._shm is None:
            return

        self._seq += 1
        buf = self._shm.buf
        seq_writing = (self._seq << 1) | 1
        seq_done = seq_writing + 1

        struct.pack_into("<I", buf, OFF_SEQ, seq_writing)
        struct.pack_into("<I", buf, OFF_STATUS, status_code)
        struct.pack_into("<I", buf, OFF_SEQ, seq_done)

    def read_status(self) -> int:
        """
        SHM에서 현재 상태 코드를 읽어 반환합니다.

        seqlock 일관성 검사를 수행합니다.
        """
        if self._shm is None:
            return STATUS_AWAKE

        buf = self._shm.buf
        for _ in range(10):  # 최대 10회 재시도
            seq1: int = struct.unpack_from("<I", buf, OFF_SEQ)[0]
            if seq1 & 1:
                continue  # 쓰기 중 — 재시도
            status: int = struct.unpack_from("<I", buf, OFF_STATUS)[0]
            seq2: int = struct.unpack_from("<I", buf, OFF_SEQ)[0]
            if seq1 == seq2:
                return status
        logger.warning("SHM 읽기: seqlock 재시도 초과, 마지막 값 반환")
        return status  # type: ignore[return-value]

    def close(self) -> None:
        """
        SHM 연결을 해제합니다.

        이 인스턴스가 SHM 생성자(creator)인 경우 unlink()도 수행하여
        /dev/shm 항목을 삭제합니다.
        """
        if self._shm is None:
            return
        try:
            self._shm.close()
            if self._is_creator:
                self._shm.unlink()
                logger.debug("SHM unlink 완료: /%s", self._name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHM close/unlink 중 오류: %s", exc)
        finally:
            self._shm = None

    # ------------------------------------------------------------------
    # 컨텍스트 매니저 지원
    # ------------------------------------------------------------------

    def __enter__(self) -> "ShmBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """SHM을 생성하거나 기존 SHM에 연결하고 헤더를 초기화합니다."""
        try:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=True, size=self._size
            )
            self._is_creator = True
            self._init_header()
            logger.info("SHM 생성 완료: /%s (%d bytes)", self._name, self._size)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=False, size=self._size
            )
            self._is_creator = False
            self._validate_header()
            logger.info("기존 SHM 연결: /%s", self._name)

    def _init_header(self) -> None:
        """SHM 헤더를 초기화합니다 (생성자만 호출)."""
        buf = self._shm.buf  # type: ignore[union-attr]
        buf[OFF_MAGIC : OFF_MAGIC + 4] = b"EARS"
        struct.pack_into("<I", buf, OFF_VERSION, SHM_PROTOCOL_VERSION)
        struct.pack_into("<I", buf, OFF_SEQ, 0)
        struct.pack_into("<I", buf, OFF_STATUS, STATUS_AWAKE)

    def _validate_header(self) -> None:
        """기존 SHM의 magic 및 버전을 검증합니다."""
        buf = self._shm.buf  # type: ignore[union-attr]
        magic = bytes(buf[OFF_MAGIC : OFF_MAGIC + 4])
        version: int = struct.unpack_from("<I", buf, OFF_VERSION)[0]

        if magic != b"EARS":
            logger.warning(
                "SHM magic 불일치: expected b'EARS', got %r. "
                "헤더를 재초기화합니다.",
                magic,
            )
            self._init_header()
            return

        if version != SHM_PROTOCOL_VERSION:
            logger.warning(
                "SHM 버전 불일치: expected %d, got %d. "
                "헤더를 재초기화합니다.",
                SHM_PROTOCOL_VERSION,
                version,
            )
            self._init_header()
