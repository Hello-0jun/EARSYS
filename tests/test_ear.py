"""Unit tests for EAR geometry helpers."""

from __future__ import annotations

import pytest

from earsys.vision.ear import average_ear, calculate_ear, euclidean_distance, get_eye_points


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        ((0, 0), (0, 0), 0.0),
        ((0, 0), (3, 0), 3.0),
        ((0, 0), (0, 4), 4.0),
        ((0, 0), (3, 4), 5.0),
        ((1.5, 2.5), (4.5, 6.5), 5.0),
    ],
)
def test_euclidean_distance(p1, p2, expected):
    assert euclidean_distance(p1, p2) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)], 1.0),
        ([(0, 0), (1, 0), (3, 0), (4, 0), (3, 0), (1, 0)], 0.0),
        ([(5, 5), (5, 7), (5, 7), (5, 5), (5, 3), (5, 3)], 0.0),
    ],
)
def test_calculate_ear(points, expected):
    assert calculate_ear(points) == pytest.approx(expected, abs=1e-6)


def test_calculate_ear_rejects_wrong_point_count():
    with pytest.raises(ValueError, match="exactly 6 points"):
        calculate_ear([(0, 0)] * 5)


def test_calculate_ear_distinguishes_open_and_closed_eye():
    open_eye = [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)]
    closed_eye = [(0, 0), (1, 0.1), (3, 0.1), (4, 0), (3, -0.1), (1, -0.1)]

    assert calculate_ear(open_eye) > 0.23
    assert calculate_ear(closed_eye) < 0.23


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0.3, 0.3, 0.3),
        (0.2, 0.4, 0.3),
        (0.0, 0.0, 0.0),
    ],
)
def test_average_ear(left, right, expected):
    assert average_ear(left, right) == pytest.approx(expected)


class Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def test_get_eye_points_converts_normalized_landmarks_to_pixels():
    landmarks = [Landmark(0.0, 0.0), Landmark(0.5, 0.25), Landmark(1.0, 1.0)]

    assert get_eye_points(landmarks, [0, 1, 2], width=640, height=480) == [
        (0, 0),
        (320, 120),
        (640, 480),
    ]


def test_get_eye_points_selects_requested_indices_only():
    landmarks = [Landmark(0.1 * i, 0.2 * i) for i in range(10)]

    assert get_eye_points(landmarks, [0, 5, 9], width=100, height=100) == [
        (0, 0),
        (50, 100),
        (90, 180),
    ]
