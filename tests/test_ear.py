"""
earsys.ear 모듈 단위 테스트.

pytest로 실행:
    pytest tests/test_ear.py -v
"""

from __future__ import annotations

import math

import pytest

from earsys.ear import average_ear, calculate_ear, euclidean_distance, get_eye_points


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_same_point(self):
        assert euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_horizontal(self):
        assert euclidean_distance((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_vertical(self):
        assert euclidean_distance((0, 0), (0, 4)) == pytest.approx(4.0)

    def test_diagonal(self):
        # 3-4-5 직각삼각형
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_float_coords(self):
        assert euclidean_distance((1.5, 2.5), (4.5, 6.5)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# calculate_ear
# ---------------------------------------------------------------------------

class TestCalculateEar:
    def _make_open_eye(self):
        """완전히 열린 정사각형 눈 (EAR ≈ 1.0)."""
        # p1=(0,0), p4=(4,0) → horizontal=4
        # p2=(1,2), p6=(1,-2) → vertical1=4
        # p3=(3,2), p5=(3,-2) → vertical2=4
        # EAR = (4+4)/(2*4) = 1.0
        return [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)]

    def _make_closed_eye(self):
        """거의 닫힌 눈 (EAR ≈ 0.0)."""
        # 수직 거리가 0에 가까운 경우
        return [(0, 0), (1, 0), (3, 0), (4, 0), (3, 0), (1, 0)]

    def test_open_eye_ear(self):
        points = self._make_open_eye()
        ear = calculate_ear(points)
        assert ear == pytest.approx(1.0, abs=1e-6)

    def test_closed_eye_ear_is_zero(self):
        points = self._make_closed_eye()
        ear = calculate_ear(points)
        assert ear == pytest.approx(0.0, abs=1e-6)

    def test_zero_horizontal_returns_zero(self):
        """p1 == p4 이면 0.0 반환 (ZeroDivision 없음)."""
        points = [(5, 5), (5, 7), (5, 7), (5, 5), (5, 3), (5, 3)]
        assert calculate_ear(points) == 0.0

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="6개"):
            calculate_ear([(0, 0)] * 5)

    def test_typical_awake_ear(self):
        """전형적인 깨어있는 상태의 EAR은 임계값(0.23) 이상이어야 한다."""
        points = self._make_open_eye()
        ear = calculate_ear(points)
        assert ear > 0.23

    def test_typical_drowsy_ear(self):
        """졸린 상태의 EAR은 임계값(0.23) 미만이어야 한다."""
        # 수직 거리를 매우 작게 설정
        points = [(0, 0), (1, 0.1), (3, 0.1), (4, 0), (3, -0.1), (1, -0.1)]
        ear = calculate_ear(points)
        assert ear < 0.23


# ---------------------------------------------------------------------------
# average_ear
# ---------------------------------------------------------------------------

class TestAverageEar:
    def test_same_values(self):
        assert average_ear(0.3, 0.3) == pytest.approx(0.3)

    def test_different_values(self):
        assert average_ear(0.2, 0.4) == pytest.approx(0.3)

    def test_zero(self):
        assert average_ear(0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_eye_points
# ---------------------------------------------------------------------------

class _FakeLandmark:
    """MediaPipe 랜드마크 객체를 흉내 냅니다."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class TestGetEyePoints:
    def _make_landmarks(self, coords: list[tuple[float, float]]):
        return [_FakeLandmark(x, y) for x, y in coords]

    def test_basic_conversion(self):
        landmarks = self._make_landmarks([(0.5, 0.5)] * 6)
        indices = [0, 1, 2, 3, 4, 5]
        points = get_eye_points(landmarks, indices, width=100, height=200)
        assert points == [(50, 100)] * 6

    def test_boundary_values(self):
        landmarks = self._make_landmarks([(0.0, 0.0), (1.0, 1.0)])
        points = get_eye_points(landmarks, [0, 1], width=640, height=480)
        assert points[0] == (0, 0)
        assert points[1] == (640, 480)

    def test_index_selection(self):
        """특정 인덱스만 선택되는지 확인."""
        coords = [(0.1 * i, 0.2 * i) for i in range(10)]
        landmarks = self._make_landmarks(coords)
        points = get_eye_points(landmarks, [0, 5, 9], width=100, height=100)
        assert len(points) == 3
        assert points[0] == (0, 0)
        assert points[1] == (int(0.5 * 100), int(1.0 * 100))
