"""
MediaPipe Face Landmarker wrapper class.

Initializes in VIDEO mode and manages a monotonically increasing timestamp internally.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mediapipe as mp
import numpy as np

from earsys.config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Wrapper around MediaPipe FaceLandmarker.

    Features:
    - Verifies the model file exists before initialization.
    - Guarantees the monotonically increasing timestamp required by VIDEO mode via time.monotonic_ns().
    - Supports the context manager (`with`) protocol.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        num_faces: int = 1,
    ) -> None:
        model_path = model_path or settings.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model file not found: {model_path}\n"
                "Check EARSYS_MODEL_PATH or place face_landmarker.task in the project root."
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
        logger.info("FaceDetector initialized: model=%s", model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, rgb_frame: np.ndarray) -> list:
        """
        Detect face landmarks from an RGB NumPy array frame.

        Parameters:
            rgb_frame: HxWx3 uint8 NumPy array in RGB format.

        Returns:
            The result.face_landmarks list.
            Returns an empty list when no face is detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = self._monotonic_ms()
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.face_landmarks

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        self._landmarker.close()
        logger.debug("FaceDetector released")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _monotonic_ms(self) -> int:
        """
        Return elapsed time in milliseconds since process start as a monotonically increasing integer.

        Use time.monotonic_ns() instead of time.time() to avoid moving backward when the system clock changes.
        MediaPipe VIDEO mode requires monotonically increasing timestamps.
        """
        current_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        if current_ms <= self._last_ms:
            current_ms = self._last_ms + 1
        self._last_ms = current_ms
        return current_ms
