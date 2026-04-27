"""
EAR(Eye Aspect Ratio) 계산 순수 함수 모음.

외부 상태에 의존하지 않는 순수 함수만 포함하여
단위 테스트가 용이합니다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 타입 별칭
# ---------------------------------------------------------------------------
Point2D = tuple[int | float, int | float]


def euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """두 2D 점 사이의 유클리드 거리를 반환합니다."""
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def calculate_ear(eye_points: Sequence[Point2D]) -> float:
    """
    EAR(Eye Aspect Ratio)를 계산합니다.

    공식:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    매개변수:
        eye_points: 눈을 구성하는 6개 점의 시퀀스.
            [p1, p2, p3, p4, p5, p6] 순서.
            p1, p4 : 눈의 양 끝 (수평)
            p2, p6 : 상/하 수직 쌍 1
            p3, p5 : 상/하 수직 쌍 2

    반환값:
        EAR 값 (float). 수평 거리가 0이면 0.0 반환.
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_points는 정확히 6개여야 합니다. 받은 개수: {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0.0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def average_ear(left_ear: float, right_ear: float) -> float:
    """왼쪽·오른쪽 EAR의 평균을 반환합니다."""
    return (left_ear + right_ear) / 2.0


def get_eye_points(
    landmarks: Sequence,
    indices: Sequence[int],
    width: int,
    height: int,
) -> list[Point2D]:
    """
    MediaPipe 정규화 좌표(0~1)를 픽셀 좌표로 변환합니다.

    매개변수:
        landmarks: MediaPipe FaceLandmarker 결과의 landmark 리스트.
        indices  : 추출할 랜드마크 인덱스 목록.
        width    : 프레임 너비 (픽셀).
        height   : 프레임 높이 (픽셀).

    반환값:
        픽셀 좌표 (int, int) 튜플의 리스트.
    """
    return [
        (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        for idx in indices
    ]
