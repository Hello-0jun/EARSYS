"""
카메라 추상화 레이어.

GStreamer(libcamerasrc) 백엔드를 기본으로 사용하며,
환경변수 EARSYS_GST_PIPELINE 으로 파이프라인을 재정의할 수 있습니다.
"""

from __future__ import annotations

import logging
from typing import Iterator

import cv2
import numpy as np

from earsys.config import GST_PIPELINE

logger = logging.getLogger(__name__)


class GstreamerCamera:
    """
    OpenCV + GStreamer 백엔드 카메라 래퍼.

    컨텍스트 매니저 또는 이터레이터로 사용합니다:

        with GstreamerCamera() as cam:
            for bgr_frame in cam.frames():
                process(bgr_frame)
    """

    def __init__(self, pipeline: str = GST_PIPELINE) -> None:
        self._pipeline = pipeline
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def open(self) -> None:
        """카메라 스트림을 엽니다. 실패하면 RuntimeError를 발생시킵니다."""
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                "카메라를 열 수 없습니다.\n"
                f"GStreamer 파이프라인을 확인하세요: {self._pipeline}\n"
                "환경변수 EARSYS_GST_PIPELINE 로 파이프라인을 재정의할 수 있습니다."
            )
        logger.info("카메라 오픈 완료: %s", self._pipeline[:60])

    def read(self) -> np.ndarray | None:
        """
        한 프레임을 읽어 BGR NumPy 배열로 반환합니다.

        프레임 읽기에 실패하면 None을 반환합니다.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("프레임 읽기 실패")
            return None
        return frame

    def frames(self, flip: bool = True) -> Iterator[np.ndarray]:
        """
        연속 프레임을 생성하는 제너레이터.

        매개변수:
            flip: True이면 좌우 반전(거울 효과)을 적용합니다.

        읽기 실패 시 StopIteration 하여 루프를 종료합니다.
        """
        while True:
            frame = self.read()
            if frame is None:
                return
            if flip:
                frame = cv2.flip(frame, 1)
            yield frame

    def release(self) -> None:
        """카메라 자원을 해제합니다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("카메라 자원 해제 완료")

    # ------------------------------------------------------------------
    # 컨텍스트 매니저 지원
    # ------------------------------------------------------------------

    def __enter__(self) -> "GstreamerCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # 이터레이터 지원 (직접 for 루프 사용 시)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()
        return self.frames()
