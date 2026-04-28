"""
설정 상수 및 환경변수 처리.

모든 튜닝 가능한 값은 이 모듈에서 관리하며,
환경변수를 통해 런타임에 재정의할 수 있습니다.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_model_path() -> Path:
    """
    face_landmarker.task 모델 파일의 절대 경로를 반환합니다.

    우선순위:
      1. 환경변수 EARSYS_MODEL_PATH
      2. 실행 파일 기준 자동 탐색

    Nuitka --onefile 동작 원리:
      - ``--include-data-files=face_landmarker.task=face_landmarker.task`` 로 빌드 시
        onefile 런타임은 데이터 파일을 /tmp/onefile_XXX/ 루트에 추출합니다.
      - 이 때 ``__file__`` 은 /tmp/onefile_XXX/earsys/config.pyc 를 가리킵니다.
      - ``Path(__file__).parent.parent`` = /tmp/onefile_XXX/ → 모델과 같은 위치. ✅

    소스 직접 실행 시:
      - ``__file__`` = <프로젝트>/earsys/config.py
      - ``Path(__file__).parent.parent`` = <프로젝트>/ → face_landmarker.task 위치. ✅
    """
    env = os.getenv("EARSYS_MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "face_landmarker.task"


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
MODEL_PATH: Path = _resolve_model_path()

# ---------------------------------------------------------------------------
# UDS 소켓
# ---------------------------------------------------------------------------
# abstract namespace 주소 (첫 바이트 = '\x00')
UDS_EYE_ADDR: bytes = b"\x00sleepcare/eye"

# EyeFrame 프로토콜
EYE_FRAME_MAGIC   = b"SEYE"
EYE_FRAME_VERSION = 1
EYE_FRAME_FORMAT  = "<4sBBHfIQ"   # 24 bytes
EYE_FRAME_SIZE    = 24

# EAR → eye_score 변환 임계값
EAR_OPEN_THR   = 0.30   # EAR >= 이 값 → score = 0.0
EAR_CLOSED_THR = 0.15   # EAR <= 이 값 → score = 1.0

# ---------------------------------------------------------------------------
# 카메라 (GStreamer)
# ---------------------------------------------------------------------------
GST_PIPELINE: str = os.getenv(
    "EARSYS_GST_PIPELINE",
    (
        "libcamerasrc ! "
        "video/x-raw, width=1920, height=1080, framerate=47/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    ),
)

# ---------------------------------------------------------------------------
# EAR / 졸음 감지
# ---------------------------------------------------------------------------
# 눈 랜드마크 인덱스 (MediaPipe Face Landmarker 468-point 기준)
LEFT_EYE_INDICES: list[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES: list[int] = [362, 385, 387, 263, 373, 380]

# 눈 감김 판정 EAR 임계값 (값이 작을수록 더 엄격)
EAR_THRESHOLD: float = float(os.getenv("EARSYS_EAR_THRESHOLD", "0.23"))

# 연속으로 몇 프레임 동안 눈이 감겨야 졸음으로 판정할지
CLOSED_FRAMES_THRESHOLD: int = int(os.getenv("EARSYS_CLOSED_FRAMES", "20"))

# ---------------------------------------------------------------------------
# 상태 코드
# ---------------------------------------------------------------------------
STATUS_AWAKE = 0
STATUS_DROWSY = 1
STATUS_NO_FACE = 2
