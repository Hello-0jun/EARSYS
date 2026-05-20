"""
Visualization script for manual testing on Windows/Linux.
"""

import os

import pytest

if os.getenv("EARSYS_RUN_VISUAL_TESTS") != "1":
    pytest.skip("manual OpenCV visualization test", allow_module_level=True)

import queue
import time

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "../face_landmarker.task"
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
CLOSED_FRAMES_THRESHOLD = 20

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


result_queue: queue.Queue = queue.Queue(maxsize=1)

def result_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    face_landmarks_list = result.face_landmarks
    ear = 0.0
    if face_landmarks_list:
        face = face_landmarks_list[0]
        width = output_image.width
        height = output_image.height
        left_eye = get_eye_points(face, LEFT_EYE, width, height)
        right_eye = get_eye_points(face, RIGHT_EYE, width, height)
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        
    try:
        result_queue.put_nowait((face_landmarks_list, ear))
    except queue.Full:
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            result_queue.put_nowait((face_landmarks_list, ear))
        except queue.Full:
            pass

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=result_callback,
)
landmarker = FaceLandmarker.create_from_options(options)


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_eye_points(landmarks, indices, width, height):
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append((int(lm.x * width), int(lm.y * height)))
    return points


def draw_eye_points(frame, points):
    for pt in points:
        cv2.circle(frame, pt, 2, (0, 255, 0), -1)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Unable to open the camera.")

closed_frames = 0
last_timestamp_ms = -1

face_landmarks_list = []
ear = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    current_timestamp_ms = time.monotonic_ns() // 1_000_000
    if current_timestamp_ms <= last_timestamp_ms:
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms
    
    landmarker.detect_async(mp_image, current_timestamp_ms)

    try:
        face_landmarks_list, ear = result_queue.get_nowait()
    except queue.Empty:
        pass

    status_text = "AWAKE"

    if face_landmarks_list:
        face = face_landmarks_list[0]
        left_eye = get_eye_points(face, LEFT_EYE, w, h)
        right_eye = get_eye_points(face, RIGHT_EYE, w, h)

        draw_eye_points(frame, left_eye)
        draw_eye_points(frame, right_eye)

        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= CLOSED_FRAMES_THRESHOLD:
            status_text = "DROWSINESS ALERT"
            cv2.putText(frame, status_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        closed_frames = 0
        status_text = "NO FACE"

    cv2.putText(frame, f"STATUS: {status_text}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"CLOSED FRAMES: {closed_frames}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("EARSYS Visual Test (No UDS)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
