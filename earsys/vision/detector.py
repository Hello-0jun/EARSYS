"""
MediaPipe Face Landmarker wrapper class.

Initializes in LIVE_STREAM mode and manages a monotonically increasing timestamp internally.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import mediapipe as mp
import numpy as np

from earsys.config import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, settings
from earsys.vision.ear import average_ear, calculate_ear, get_eye_points

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Wrapper around MediaPipe FaceLandmarker.

    Features:
    - Verifies the model file exists before initialization.
    - Guarantees the monotonically increasing timestamp required by LIVE_STREAM mode via time.monotonic_ns().
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

        self._result_queue: queue.Queue = queue.Queue()

        def result_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
            face_landmarks_list = result.face_landmarks
            ear = 0.0
            if face_landmarks_list:
                landmarks = face_landmarks_list[0]
                width = output_image.width
                height = output_image.height
                left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, width, height)
                right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, width, height)
                ear = average_ear(calculate_ear(left_eye), calculate_ear(right_eye))
            self._result_queue.put((face_landmarks_list, ear))

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=num_faces,
            result_callback=result_callback,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_ns: int = time.monotonic_ns()
        self._last_ms: int = -1
        logger.info("[bold blue]FaceDetector initialized[/bold blue]: model=[cyan]%s[/cyan]", model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, rgb_frame: np.ndarray) -> tuple[list, float]:
        """
        Detect face landmarks and calculate EAR from an RGB NumPy array frame.

        Parameters:
            rgb_frame: HxWx3 uint8 NumPy array in RGB format.

        Returns:
            A tuple (face_landmarks_list, ear).
            Returns ([], 0.0) when no face is detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = self._monotonic_ms()

        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

        self._landmarker.detect_async(mp_image, timestamp_ms)
        
        try:
            face_landmarks_list, ear = self._result_queue.get(timeout=1.0)
            return face_landmarks_list, ear
        except queue.Empty:
            logger.warning("FaceLandmarker async detection timed out")
            return [], 0.0

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        self._landmarker.close()
        logger.debug("[dim]FaceDetector released[/dim]")

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
        MediaPipe LIVE_STREAM mode requires monotonically increasing timestamps.
        """
        current_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        if current_ms <= self._last_ms:
            current_ms = self._last_ms + 1
        self._last_ms = current_ms
        return current_ms
