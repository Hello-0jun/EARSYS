"""OpenCV camera abstraction with system-aware profile fallback."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Iterator

import cv2
import numpy as np

from earsys.camera_profile import CameraProfile, resolve_camera_profiles

logger = logging.getLogger(__name__)

OPENCV_BACKENDS = {
    "gstreamer": cv2.CAP_GSTREAMER,
    "v4l2": cv2.CAP_V4L2,
    "directshow": cv2.CAP_DSHOW,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


def _opencv_backend(backend: str, source: int | str) -> int:
    """Resolve the requested OpenCV backend constant."""
    if backend in OPENCV_BACKENDS:
        return OPENCV_BACKENDS[backend]
    if isinstance(source, str) and "!" in source:
        return cv2.CAP_GSTREAMER
    return cv2.CAP_ANY


class OpenCvCamera:
    """
    OpenCV camera wrapper.

    Use it as a context manager or iterator:

        with OpenCvCamera() as cam:
            for bgr_frame in cam.frames():
                process(bgr_frame)
    """

    def __init__(
        self,
        profile: CameraProfile | None = None,
        profiles: Sequence[CameraProfile] | None = None,
    ) -> None:
        if profiles is not None:
            self._profiles = list(profiles)
        elif profile is not None:
            self._profiles = [profile]
        else:
            self._profiles = resolve_camera_profiles()
        self._profile: CameraProfile | None = None
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera stream. Raise RuntimeError on failure."""
        errors: list[str] = []

        for profile in self._profiles:
            cap = self._open_capture(profile)

            if cap.isOpened():
                self._cap = cap
                self._profile = profile
                logger.info(
                    "Camera opened: label=%s source=%r backend=%s color=%s",
                    profile.label,
                    profile.source,
                    profile.backend,
                    profile.color_format,
                )
                return

            cap.release()
            errors.append(f"{profile.label}: source={profile.source!r}, backend={profile.backend!r}")

        self._cap = None
        raise RuntimeError(_camera_open_error(errors))

    @property
    def color_format(self) -> str:
        """Return the color format expected from the selected camera profile."""
        if self._profile is None:
            return "bgr"
        return self._profile.color_format

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

    def __enter__(self) -> "OpenCvCamera":
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

    def _open_capture(self, profile: CameraProfile) -> cv2.VideoCapture:
        api_preference = _opencv_backend(profile.backend, profile.source)
        cap = cv2.VideoCapture(profile.source, api_preference)
        _apply_capture_properties(cap, profile)
        return cap


GstreamerCamera = OpenCvCamera


def _apply_capture_properties(cap: cv2.VideoCapture, profile: CameraProfile) -> None:
    properties = (
        (cv2.CAP_PROP_FRAME_WIDTH, profile.width),
        (cv2.CAP_PROP_FRAME_HEIGHT, profile.height),
        (cv2.CAP_PROP_FPS, profile.fps),
    )
    for prop, value in properties:
        if value is not None:
            cap.set(prop, value)


def _camera_open_error(errors: Sequence[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return (
        "Unable to open camera with any auto-selected profile.\n"
        f"{details}\n"
        "Set EARSYS_CAMERA_SOURCE, EARSYS_CAMERA_BACKEND, EARSYS_GST_PIPELINE, "
        "or EARSYS_CAMERA_COLOR_FORMAT for this system."
    )
