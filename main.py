"""
EARSYS - entry point for the EAR-based drowsiness detection system.

This file only provides a minimal entry point.
Business logic lives in the modules under the earsys/ package.

Run:
    python main.py

Environment variables:
    EARSYS_MODEL_PATH        Path to face_landmarker.task (default: project root)
    EARSYS_GST_PIPELINE      GStreamer pipeline string
    EARSYS_EAR_THRESHOLD     EAR threshold for closed eyes (default: 0.23)
    EARSYS_CLOSED_FRAMES     Number of consecutive frames for drowsiness (default: 20)
    EARSYS_LOG_LEVEL         Log level (default: INFO)
    EARSYS_CAMERA_REOPEN_SEC Wait time before reopening the camera (default: 2.0)
    EARSYS_CAMERA_MAX_RETRIES Maximum camera reopen attempts (default: 0 = unlimited)
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
from earsys.uds_async import UdsAsyncBridge as UdsBridge

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("EARSYS_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("earsys.main")


# ---------------------------------------------------------------------------
# Detection loop
# ---------------------------------------------------------------------------

def run_detection(camera: GstreamerCamera, detector: FaceDetector, bridge: UdsBridge) -> bool:
    """
    Main detection loop.

    Returns:
        True  = user initiated shutdown (KeyboardInterrupt)
        False = loop stopped because the camera stream ended or failed
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
    sent_fused_scores: int = 0

    # Track state changes: only suppress duplicate NO_FACE sends.
    previous_status: int | None = None

    logger.info(
        "Detection loop started - EAR threshold=%.2f, consecutive frames=%d",
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
                            logger.warning("Drowsiness detected! EAR=%.3f, consecutive frames=%d", ear, closed_frames)
                    else:
                        status = STATUS_AWAKE

                    bridge.send(status=status, ear=ear)
                    sent_fused_scores += 1
                    previous_status = status
                    if status == STATUS_DROWSY:
                        sent_drowsy += 1
                        logger.info("Sent DROWSY status: EAR=%.3f, consecutive frames=%d", ear, closed_frames)
                    else:
                        sent_awake += 1
                        logger.info("Sent AWAKE status: EAR=%.3f", ear)

                else:
                    closed_frames = 0
                    alarm_triggered = False
                    status = STATUS_NO_FACE
                    
                    # Send over UDS only when the state changes.
                    if status != previous_status:
                        bridge.send(status=status, ear=0.0)
                        sent_fused_scores += 1
                        previous_status = status
                        sent_no_face += 1
                        logger.info("Sent NO_FACE status")
            
            except Exception as e:
                frame_errors += 1
                logger.error("Error while processing frame: %s", e)

            now_mono = time.monotonic()
            if now_mono >= next_health_log:
                uptime_sec = int(now_mono - start_monotonic)
                logger.info(
                    "Health check uptime=%ss frames=%d awake=%d drowsy=%d no_face=%d fused_scores=%d frame_errors=%d",
                    uptime_sec,
                    frames_total,
                    sent_awake,
                    sent_drowsy,
                    sent_no_face,
                    sent_fused_scores,
                    frame_errors,
                )
                next_health_log = now_mono + health_interval_sec

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping detection loop.")
        return True

    logger.error("Camera frame stream ended, stopping detection loop.")
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Main EARSYS function.

    Returns:
        0 = normal exit, 1 = error exit
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
                            logger.info("Shutting down EARSYS on user request.")
                            return 0
                except RuntimeError as exc:
                    logger.error("Camera open/runtime error: %s", exc)

                reopen_attempt += 1
                if camera_max_retries > 0 and reopen_attempt > camera_max_retries:
                    logger.error("Exiting after exceeding camera reopen retry limit (%d).", camera_max_retries)
                    return 1

                logger.warning(
                    "Retrying camera open %d time(s); waiting %.1f seconds.",
                    reopen_attempt,
                    camera_reopen_sec,
                )
                time.sleep(camera_reopen_sec)
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Camera error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        return 1

    logger.info("EARSYS exited normally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
