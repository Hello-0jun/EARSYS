"""
EARSYS — EAR 기반 졸음 감지 시스템 진입점.

이 파일은 최소한의 진입점 역할만 합니다.
비즈니스 로직은 earsys/ 패키지의 각 모듈에 위치합니다.

실행:
    python main.py

환경변수:
    EARSYS_MODEL_PATH       face_landmarker.task 경로 (기본: 프로젝트 루트)
    EARSYS_GST_PIPELINE     GStreamer 파이프라인 문자열
    EARSYS_EAR_THRESHOLD    눈 감김 EAR 임계값 (기본: 0.23)
    EARSYS_CLOSED_FRAMES    졸음 판정 연속 프레임 수 (기본: 20)
    EARSYS_LOG_LEVEL        로그 레벨 (기본: INFO)
    EARSYS_CAMERA_REOPEN_SEC 카메라 재오픈 대기초 (기본: 2.0)
    EARSYS_CAMERA_MAX_RETRIES 카메라 재오픈 최대 횟수 (기본: 0=무제한)
"""

from __future__ import annotations

import logging
import os
import sys
import time

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
from earsys.uds_bridge import UdsBridge

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

def run_detection(camera: GstreamerCamera, detector: FaceDetector, bridge: UdsBridge) -> bool:
    """
    메인 감지 루프.

    반환값:
        True  = 사용자 종료(KeyboardInterrupt)
        False = 카메라 스트림 종료/오류로 루프 중단
    """
    closed_frames: int = 0
    alarm_triggered: bool = False
    health_interval_sec: float = 10.0
    start_monotonic: float = time.monotonic()
    next_health_log: float = start_monotonic + health_interval_sec

    frames_total: int = 0
    sent_awake: int = 0
    sent_drowsy: int = 0
    sent_no_face: int = 0
    frame_errors: int = 0

    logger.info(
        "감지 루프 시작 — EAR 임계값=%.2f, 연속 프레임=%d",
        EAR_THRESHOLD,
        CLOSED_FRAMES_THRESHOLD,
    )

    try:
        for bgr_frame in camera.frames(flip=True):
            frames_total += 1
            try:
                height, width = bgr_frame.shape[:2]
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                face_landmarks_list = detector.detect(rgb_frame)

                if face_landmarks_list:
                    landmarks = face_landmarks_list[0]

                    left_eye  = get_eye_points(landmarks, LEFT_EYE_INDICES,  width, height)
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

                    # eye_score는 ear 값으로 항상 계산
                    bridge.send(status=status, ear=ear)
                    if status == STATUS_DROWSY:
                        sent_drowsy += 1
                    else:
                        sent_awake += 1

                else:
                    closed_frames = 0
                    alarm_triggered = False
                    # 얼굴 미검출: eye_score = 0.0 (ear=0.0 전달)
                    bridge.send(status=STATUS_NO_FACE, ear=0.0)
                    sent_no_face += 1
            
            except Exception as e:
                frame_errors += 1
                logger.error("프레임 처리 중 오류가 발생했습니다: %s", e)

            now_mono = time.monotonic()
            if now_mono >= next_health_log:
                uptime_sec = int(now_mono - start_monotonic)
                logger.info(
                    "헬스체크 uptime=%ss frames=%d awake=%d drowsy=%d no_face=%d frame_errors=%d",
                    uptime_sec,
                    frames_total,
                    sent_awake,
                    sent_drowsy,
                    sent_no_face,
                    frame_errors,
                )
                next_health_log = now_mono + health_interval_sec

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: 감지 루프를 종료합니다.")
        return True

    logger.error("카메라 프레임 스트림이 종료되어 감지 루프를 중단합니다.")
    return False


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> int:
    """
    EARSYS 메인 함수.

    반환값:
        0 = 정상 종료, 1 = 오류 종료
    """
    camera_reopen_sec = float(os.getenv("EARSYS_CAMERA_REOPEN_SEC", "2.0"))
    camera_max_retries = int(os.getenv("EARSYS_CAMERA_MAX_RETRIES", "0"))

    try:
        with UdsBridge() as bridge, FaceDetector() as detector:
            reopen_attempt = 0
            while True:
                try:
                    with GstreamerCamera() as camera:
                        user_stopped = run_detection(camera, detector, bridge)
                        if user_stopped:
                            logger.info("사용자 요청으로 EARSYS를 종료합니다.")
                            return 0
                except RuntimeError as exc:
                    logger.error("카메라 오픈/동작 오류: %s", exc)

                reopen_attempt += 1
                if camera_max_retries > 0 and reopen_attempt > camera_max_retries:
                    logger.error(
                        "카메라 재오픈 재시도 한도 초과(%d회)로 종료합니다.",
                        camera_max_retries,
                    )
                    return 1

                logger.warning(
                    "카메라 재오픈 재시도 %d회 후 %.1f초 대기합니다.",
                    reopen_attempt,
                    camera_reopen_sec,
                )
                time.sleep(camera_reopen_sec)
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
