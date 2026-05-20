"""
Dev visualization helpers for OpenCV frames.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from earsys.config import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    STATUS_AWAKE,
    STATUS_DROWSY,
)
from earsys.vision.ear import get_eye_points

_LANDMARK_COLOR = (180, 180, 180)  # gray for all face points
_EYE_COLOR = (0, 230, 100)  # green for eye points
_DROWSY_COLOR = (0, 60, 255)  # red for drowsy alert
_AWAKE_COLOR = (0, 220, 60)  # green for awake status
_NO_FACE_COLOR = (200, 200, 0)  # yellow for no face
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_landmarks(frame: np.ndarray, landmarks: list, width: int, height: int) -> None:
    """Draw face landmarks using MediaPipe's drawing utilities."""
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])

    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
    )


def draw_eye_overlay(
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


def draw_no_face(frame: np.ndarray) -> None:
    """Draw a minimal NO FACE indicator when no landmarks are detected."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "NO FACE", (frame.shape[1] - 160, 35), _FONT, 0.9, _NO_FACE_COLOR, 2)


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
        True if the user pressed 'q' to quit, False otherwise.
    """
    h, w = frame.shape[:2]
    display = frame.copy()

    if face_landmarks_list:
        landmarks = face_landmarks_list[0]
        draw_landmarks(display, landmarks, w, h)

        left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, w, h)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, w, h)
        draw_eye_overlay(display, left_eye, right_eye, ear, status, closed_frames)
    else:
        draw_no_face(display)

    cv2.imshow("EARSYS  [dev]  — press q to quit", display)
    return (cv2.waitKey(1) & 0xFF) == ord("q")
