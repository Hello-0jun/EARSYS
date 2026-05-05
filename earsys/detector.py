"""
MediaPipe Face Landmarker 래퍼 클래스.

VIDEO 모드로 초기화하며, 단조 증가하는 타임스탬프를 내부에서 관리합니다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mediapipe as mp

from earsys.config import MODEL_PATH

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    MediaPipe FaceLandmarker를 래핑하는 클래스.

    특징:
    - 모델 파일 존재 여부를 사전에 검증합니다.
    - VIDEO 모드에 필요한 단조 증가 타임스탬프를 `time.monotonic_ns()`로 보장합니다.
    - 컨텍스트 매니저(`with` 문)를 지원합니다.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        num_faces: int = 1,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker 모델 파일을 찾을 수 없습니다: {model_path}\n"
                "환경변수 EARSYS_MODEL_PATH 또는 프로젝트 루트에 face_landmarker.task를 확인하세요."
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=num_faces,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_ns: int = time.monotonic_ns()
        self._last_ms: int = -1
        logger.info("FaceDetector 초기화 완료: 모델=%s", model_path)

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def detect(self, rgb_frame) -> list:
        """
        RGB NumPy 배열 프레임에서 얼굴 랜드마크를 검출합니다.

        매개변수:
            rgb_frame: H×W×3 uint8 NumPy 배열 (RGB 포맷).

        반환값:
            `result.face_landmarks` 리스트.
            얼굴이 없으면 빈 리스트를 반환합니다.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = self._monotonic_ms()
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.face_landmarks

    def close(self) -> None:
        """MediaPipe landmarker를 해제합니다."""
        self._landmarker.close()
        logger.debug("FaceDetector 해제 완료")

    # ------------------------------------------------------------------
    # 컨텍스트 매니저 지원
    # ------------------------------------------------------------------

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _monotonic_ms(self) -> int:
        """
        프로세스 시작 이후 경과 시간(ms)을 단조 증가 정수로 반환합니다.

        `time.time()` 대신 `time.monotonic_ns()`를 사용하여
        시스템 시계 조정에 의한 역행을 방지합니다.
        MediaPipe VIDEO 모드는 타임스탬프 단조 증가를 요구합니다.
        """
        current_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        if current_ms <= self._last_ms:
            current_ms = self._last_ms + 1
        self._last_ms = current_ms
        return current_ms
