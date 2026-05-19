"""
Detection loop and dev-visualization helpers.

This module is the single source of truth for the main frame-processing loop.
``earsys.cli`` imports ``run_detection`` from here; ``main.py`` is a thin shim.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from earsys.camera.capture import OpenCvCamera
from earsys.config import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
    settings,
)
from earsys.ipc.uds_async import UdsAsyncBridge as UdsBridge
from earsys.vision.detector import FaceDetector
from earsys.vision.ear import average_ear, calculate_ear, get_eye_points

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dev visualization helpers
# ---------------------------------------------------------------------------

_LANDMARK_COLOR = (180, 180, 180)  # gray for all face points
_EYE_COLOR = (0, 230, 100)  # green for eye points
_DROWSY_COLOR = (0, 60, 255)  # red for drowsy alert
_AWAKE_COLOR = (0, 220, 60)  # green for awake status
_NO_FACE_COLOR = (200, 200, 0)  # yellow for no face
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_landmarks(frame: np.ndarray, landmarks: list, width: int, height: int) -> None:
    """Draw all 468 face landmarks as small dots."""
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, _LANDMARK_COLOR, -1)


def _draw_eye_overlay(
    frame: np.ndarray,
    left_eye: list,
    right_eye: list,
    ear: float,
    status: int,
    closed_frames: int,
) -> None:
    """Draw eye points, EAR value, status, and closed-frame counter on the frame."""
    for pt in left_eye + right_eye:
        cv2.circle(frame, pt, 3, _EYE_COLOR, -1)

    if status == STATUS_DROWSY:
        status_color = _DROWSY_COLOR
        status_text = "DROWSY"
    elif status == STATUS_AWAKE:
        status_color = _AWAKE_COLOR
        status_text = "AWAKE"
    else:
        status_color = _NO_FACE_COLOR
        status_text = "NO FACE"

    # Semi-transparent info bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"EAR: {ear:.3f}", (15, 25), _FONT, 0.7, (255, 230, 100), 2)
    cv2.putText(frame, f"CLOSED FRAMES: {closed_frames}", (15, 50), _FONT, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, status_text, (frame.shape[1] - 160, 35), _FONT, 0.9, status_color, 2)


def _draw_no_face(frame: np.ndarray) -> None:
    """Draw a minimal NO FACE indicator when no landmarks are detected."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "NO FACE", (frame.shape[1] - 160, 35), _FONT, 0.9, _NO_FACE_COLOR, 2)


def _show_debug_frame(
    frame: np.ndarray,
    face_landmarks_list: list,
    ear: float,
    status: int,
    closed_frames: int,
) -> bool:
    """
    Render the debug visualization window.

    Returns:
        True if the user pressed 'q' to quit, False otherwise.
    """
    h, w = frame.shape[:2]
    display = frame.copy()

    if face_landmarks_list:
        landmarks = face_landmarks_list[0]
        _draw_landmarks(display, landmarks, w, h)

        left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, w, h)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, w, h)
        _draw_eye_overlay(display, left_eye, right_eye, ear, status, closed_frames)
    else:
        _draw_no_face(display)

    cv2.imshow("EARSYS  [dev]  — press q to quit", display)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


# ---------------------------------------------------------------------------
# Detection state
# ---------------------------------------------------------------------------


@dataclass
class DetectionStats:
    frames_total: int = 0
    sent_awake: int = 0
    sent_drowsy: int = 0
    sent_no_face: int = 0
    frame_errors: int = 0
    sent_fused_scores: int = 0

    def log_health(self, uptime_sec: int) -> None:
        logger.info(
            "Health check uptime=%ss frames=%d awake=%d drowsy=%d no_face=%d fused_scores=%d frame_errors=%d",
            uptime_sec,
            self.frames_total,
            self.sent_awake,
            self.sent_drowsy,
            self.sent_no_face,
            self.sent_fused_scores,
            self.frame_errors,
        )


@dataclass
class DrowsinessState:
    closed_frames: int = 0
    drowsy_logged: bool = False
    previous_status: int | None = None

    def reset_eye_closure(self) -> None:
        self.closed_frames = 0
        self.drowsy_logged = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_rgb_frame(frame: np.ndarray, color_format: str) -> np.ndarray:
    """Convert an OpenCV frame to RGB according to the configured input format."""
    if color_format == "rgb":
        return frame
    if color_format == "nv12":
        return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _status_from_ear(ear: float, state: DrowsinessState) -> int:
    if ear < settings.ear_threshold:
        state.closed_frames += 1
    else:
        state.reset_eye_closure()

    if state.closed_frames < settings.closed_frames_threshold:
        return STATUS_AWAKE

    if not state.drowsy_logged:
        state.drowsy_logged = True
        logger.warning("Drowsiness detected! EAR=%.3f, consecutive frames=%d", ear, state.closed_frames)
    return STATUS_DROWSY


def _record_sent_status(stats: DetectionStats, status: int) -> None:
    stats.sent_fused_scores += 1
    if status == STATUS_DROWSY:
        stats.sent_drowsy += 1
    elif status == STATUS_AWAKE:
        stats.sent_awake += 1
    elif status == STATUS_NO_FACE:
        stats.sent_no_face += 1


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------


def run_detection(camera: OpenCvCamera, detector: FaceDetector, bridge: UdsBridge) -> bool:
    """
    Main detection loop.

    Returns:
        True  = user initiated shutdown (KeyboardInterrupt or 'q' in dev window)
        False = loop stopped because the camera stream ended or failed
    """
    visualize = settings.feature_visualize_landmarks
    verbose = settings.feature_debug_logging

    state = DrowsinessState()
    stats = DetectionStats()
    health_interval_sec: float = 10.0
    start_monotonic: float = time.monotonic()
    next_health_log: float = start_monotonic + health_interval_sec

    logger.info(
        "Detection loop started — env=%s EAR threshold=%.2f, consecutive frames=%d, visualize=%s",
        settings.env,
        settings.ear_threshold,
        settings.closed_frames_threshold,
        visualize,
    )

    try:
        for bgr_frame in camera.frames(flip=True):
            stats.frames_total += 1
            ear: float = 0.0
            status: int = STATUS_NO_FACE
            face_landmarks_list: list = []

            try:
                height, width = bgr_frame.shape[:2]
                rgb_frame = _to_rgb_frame(bgr_frame, camera.color_format)

                face_landmarks_list = detector.detect(rgb_frame)

                if face_landmarks_list:
                    landmarks = face_landmarks_list[0]

                    left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, width, height)
                    right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, width, height)

                    ear = average_ear(calculate_ear(left_eye), calculate_ear(right_eye))
                    status = _status_from_ear(ear, state)

                    bridge.send(status=status, ear=ear)
                    _record_sent_status(stats, status)
                    state.previous_status = status

                    if verbose:
                        if status == STATUS_DROWSY:
                            logger.debug("DROWSY EAR=%.3f frames=%d", ear, state.closed_frames)
                        else:
                            logger.debug("AWAKE  EAR=%.3f", ear)

                else:
                    state.reset_eye_closure()
                    status = STATUS_NO_FACE

                    # Send over UDS only when the state changes.
                    if status != state.previous_status:
                        bridge.send(status=status, ear=0.0)
                        _record_sent_status(stats, status)
                        state.previous_status = status
                        if verbose:
                            logger.debug("NO_FACE")

            except Exception as e:  # noqa: BLE001 — keep the detection loop alive
                stats.frame_errors += 1
                logger.error("Error while processing frame: %s", e)

            # Dev visualization — runs only when feature flag is enabled
            if visualize:
                quit_requested = _show_debug_frame(bgr_frame, face_landmarks_list, ear, status, state.closed_frames)
                if quit_requested:
                    logger.info("Dev window closed by user ('q' pressed).")
                    cv2.destroyAllWindows()
                    return True

            now_mono = time.monotonic()
            if now_mono >= next_health_log:
                uptime_sec = int(now_mono - start_monotonic)
                stats.log_health(uptime_sec)
                next_health_log = now_mono + health_interval_sec

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping detection loop.")
        if visualize:
            cv2.destroyAllWindows()
        return True

    logger.error("Camera frame stream ended, stopping detection loop.")
    return False
