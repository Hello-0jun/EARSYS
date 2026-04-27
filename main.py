"""
EARSYS — EAR 기반 졸음 감지 시스템 진입점.

이 파일은 최소한의 진입점 역할만 합니다.
비즈니스 로직은 earsys/ 패키지의 각 모듈에 위치합니다.

실행:
    python main.py

환경변수:
    EARSYS_MODEL_PATH       face_landmarker.task 경로 (기본: 프로젝트 루트)
    EARSYS_SHM_NAME         POSIX SHM 이름 (기본: /earsys_drowsy_shm)
    EARSYS_GST_PIPELINE     GStreamer 파이프라인 문자열
    EARSYS_EAR_THRESHOLD    눈 감김 EAR 임계값 (기본: 0.23)
    EARSYS_CLOSED_FRAMES    졸음 판정 연속 프레임 수 (기본: 20)
    EARSYS_LOG_LEVEL        로그 레벨 (기본: INFO)
"""

from __future__ import annotations

import logging
import os
import sys

import cv2

from earsys.alarm import play_alarm
from earsys.camera import GstreamerCamera
from earsys.config import (
    CLOSED_FRAMES_THRESHOLD,
    EAR_THRESHOLD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
)
from earsys.detector import FaceDetector
from earsys.ear import average_ear, calculate_ear, get_eye_points
from earsys.shm_bridge import ShmBridge

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("EARSYS_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("earsys.main")


# ---------------------------------------------------------------------------
# 감지 루프
# ---------------------------------------------------------------------------

def run_detection(camera: GstreamerCamera, detector: FaceDetector, shm: ShmBridge) -> None:
    """메인 감지 루프. KeyboardInterrupt 시 정상 종료합니다."""
    closed_frames: int = 0
    alarm_triggered: bool = False

    logger.info(
        "감지 루프 시작 — EAR 임계값=%.2f, 연속 프레임=%d",
        EAR_THRESHOLD,
        CLOSED_FRAMES_THRESHOLD,
    )

    try:
        for bgr_frame in camera.frames(flip=True):
            height, width = bgr_frame.shape[:2]
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            face_landmarks_list = detector.detect(rgb_frame)

            if face_landmarks_list:
                landmarks = face_landmarks_list[0]

                left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, width, height)
                right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, width, height)

                ear = average_ear(calculate_ear(left_eye), calculate_ear(right_eye))

                if ear < EAR_THRESHOLD:
                    closed_frames += 1
                else:
                    closed_frames = 0
                    alarm_triggered = False

                if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                    status = STATUS_DROWSY
                    if not alarm_triggered:
                        play_alarm()
                        alarm_triggered = True
                        logger.warning("졸음 감지! EAR=%.3f, 연속 프레임=%d", ear, closed_frames)
                else:
                    status = STATUS_AWAKE

            else:
                closed_frames = 0
                alarm_triggered = False
                status = STATUS_NO_FACE

            shm.write_status(status)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: 감지 루프를 종료합니다.")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> int:
    """
    EARSYS 메인 함수.

    반환값:
        0 = 정상 종료, 1 = 오류 종료
    """
    try:
        with ShmBridge() as shm, FaceDetector() as detector, GstreamerCamera() as camera:
            run_detection(camera, detector, shm)
    except FileNotFoundError as exc:
        logger.error("모델 파일 없음: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("카메라 오류: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("예기치 않은 오류: %s", exc)
        return 1

    logger.info("EARSYS 정상 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
