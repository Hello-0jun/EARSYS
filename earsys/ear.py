"""
Pure functions for calculating EAR (Eye Aspect Ratio).

Only stateless pure functions are included, which makes unit testing easier.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Point2D = tuple[int | float, int | float]


def euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """Return the Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def calculate_ear(eye_points: Sequence[Point2D]) -> float:
    """
    Calculate EAR (Eye Aspect Ratio).

    Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters:
        eye_points: Sequence of 6 points that form the eye.
            Ordered as [p1, p2, p3, p4, p5, p6].
            p1, p4: horizontal eye corners
            p2, p6: first vertical pair
            p3, p5: second vertical pair

    Returns:
        EAR value (float). Returns 0.0 when the horizontal distance is 0.
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_points must contain exactly 6 points. Got: {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def average_ear(left_ear: float, right_ear: float) -> float:
    """Return the average of the left and right EAR values."""
    return (left_ear + right_ear) / 2.0


def get_eye_points(
    landmarks: Sequence,
    indices: Sequence[int],
    width: int,
    height: int,
) -> list[Point2D]:
    """
    Convert MediaPipe normalized coordinates (0 to 1) to pixel coordinates.

    Parameters:
        landmarks: Landmark list from a MediaPipe FaceLandmarker result.
        indices: List of landmark indices to extract.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A list of pixel coordinate (int, int) tuples.
    """
    return [
        (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        for idx in indices
    ]
