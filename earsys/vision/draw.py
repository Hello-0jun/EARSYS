"""
Dev visualization helpers for OpenCV frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from earsys.config import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    STATUS_DROWSY,
    STATUS_NO_FACE,
)
from earsys.vision.ear import Point2D, get_eye_points

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EYE_COLOR = (0, 255, 0)
_EAR_COLOR = (0, 255, 255)
_ALERT_COLOR = (0, 0, 255)
_INFO_COLOR = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WINDOW_NAME = "Drowsiness Detection System"


# ---------------------------------------------------------------------------
# Drawing Utilities
# ---------------------------------------------------------------------------

def _draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 1.0,
    thickness: int = 2,
) -> None:
    """Helper to consistently draw text on an OpenCV frame."""
    cv2.putText(frame, text, position, _FONT, font_scale, color, thickness)


def draw_eye_points(frame: np.ndarray, points: list[Point2D]) -> None:
    """Draw eye landmark coordinates as green dots."""
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 2, _EYE_COLOR, -1)


# ---------------------------------------------------------------------------
# Main Visualization
# ---------------------------------------------------------------------------

def show_debug_frame(
    frame: np.ndarray,
    face_landmarks_list: list,
    ear: float,
    status: int,
    closed_frames: int,
) -> bool:
    """
    Render the debug visualization window.

    Returns:
        True if the user pressed ESC (27) or Ctrl+C (3) to quit, False otherwise.
    """
    display = frame.copy()
    height, width = display.shape[:2]

    status_text = "AWAKE"
    if status == STATUS_DROWSY:
        status_text = "DROWSINESS ALERT"
    elif status == STATUS_NO_FACE:
        status_text = "NO FACE"

    if face_landmarks_list:
        landmarks = face_landmarks_list[0]

        left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, width, height)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, width, height)

        draw_eye_points(display, left_eye)
        draw_eye_points(display, right_eye)

        # Draw EAR value
        _draw_text(display, f"EAR: {ear:.3f}", (30, 50), _EAR_COLOR)

        # Draw drowsiness alert
        if status == STATUS_DROWSY:
            _draw_text(display, status_text, (30, 100), _ALERT_COLOR, thickness=3)

    # Draw global status
    _draw_text(display, f"STATUS: {status_text}", (30, 150), _INFO_COLOR, font_scale=0.9)

    # Draw closed frames count
    _draw_text(display, f"CLOSED FRAMES: {closed_frames}", (30, 200), _INFO_COLOR, font_scale=0.8)

    cv2.imshow(_WINDOW_NAME, display)

    # Quit if ESC (27) or Ctrl+C (3) is pressed
    key = cv2.waitKey(1) & 0xFF
    return key == 27 or key == 3
