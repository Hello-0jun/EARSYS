"""
Camera abstraction layer.

Uses the GStreamer (libcamerasrc) backend by default,
and the pipeline can be overridden with the EARSYS_GST_PIPELINE environment variable.
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
    OpenCV + GStreamer camera wrapper.

    Use it as a context manager or iterator:

        with GstreamerCamera() as cam:
            for bgr_frame in cam.frames():
                process(bgr_frame)
    """

    def __init__(self, pipeline: str = GST_PIPELINE) -> None:
        self._pipeline = pipeline
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera stream. Raise RuntimeError on failure."""
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                "Unable to open camera.\n"
                f"Check the GStreamer pipeline: {self._pipeline}\n"
                "You can override the pipeline with the EARSYS_GST_PIPELINE environment variable."
            )
        logger.info("Camera opened: %s", self._pipeline[:60])

    def read(self) -> np.ndarray | None:
        """
        Read one frame and return it as a BGR NumPy array.

        Return None if frame capture fails.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return None
        return frame

    def frames(self, flip: bool = True) -> Iterator[np.ndarray]:
        """
        Generator that yields continuous frames.

        Parameters:
            flip: If True, apply a horizontal flip (mirror effect).

        Stops the loop by returning when frame capture fails.
        """
        while True:
            frame = self.read()
            if frame is None:
                return
            if flip:
                frame = cv2.flip(frame, 1)
            yield frame

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Camera resources released")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "GstreamerCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Iterator support (for direct for-loop usage)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()
        return self.frames()
