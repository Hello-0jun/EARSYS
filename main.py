# ==============================
# 필요한 라이브러리 불러오기
# ==============================
import cv2                      # OpenCV: 영상 처리 및 웹캠 제어
import mediapipe as mp          # MediaPipe: 얼굴 랜드마크 검출
import numpy as np              # NumPy: 수학 계산
import time                     # 타임스탬프 생성
import winsound                 # Windows 경고음 출력

# ==============================
# Face Landmarker 모델 경로
# ==============================
# main.py와 같은 폴더에 모델을 둘 경우 파일명만 입력
MODEL_PATH = "face_landmarker.task"

# ==============================
# MediaPipe Face Landmarker 설정
# ==============================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face Landmarker 옵션 설정
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,  # 비디오 스트림 모드
    num_faces=1                            # 한 명만 인식
)

# Face Landmarker 객체 생성
landmarker = FaceLandmarker.create_from_options(options)

# ==============================
# 눈 랜드마크 인덱스 정의
# ==============================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==============================
# 졸음 감지 기준 설정
# ==============================
EAR_THRESHOLD = 0.23            # 눈 감김 판정 기준
CLOSED_FRAMES_THRESHOLD = 20    # 연속 프레임 기준

closed_frames = 0               # 눈 감김 프레임 카운트
alarm_on = False                # 알람 중복 방지

# ==============================
# 두 점 사이 거리 계산 함수
# ==============================
def euclidean_distance(p1, p2):
    """두 점 사이의 유클리드 거리 계산"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ==============================
# EAR 계산 함수
# ==============================
def calculate_ear(eye_points):
    """
    EAR(Eye Aspect Ratio) 계산 공식
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)

# ==============================
# 눈 좌표 추출 함수
# ==============================
def get_eye_points(landmarks, indices, width, height):
    """정규화된 좌표를 픽셀 좌표로 변환"""
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
    return points

# ==============================
# 눈 좌표 화면 표시 함수
# ==============================
def draw_eye_points(frame, points):
    """눈 좌표를 초록색 점으로 표시"""
    for pt in points:
        cv2.circle(frame, pt, 2, (0, 255, 0), -1)

# ==============================
# 경고음 함수
# ==============================
def play_alarm():
    """Windows 경고음 출력"""
    winsound.Beep(2000, 500)

# ==============================
# 웹캠 실행
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# ==============================
# 메인 루프
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 좌우 반전 (거울 효과)
    frame = cv2.flip(frame, 1)

    # 프레임 크기 저장
    height, width = frame.shape[:2]

    # OpenCV(BGR) → MediaPipe(RGB) 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 이미지 객체 생성
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # 타임스탬프 생성 (밀리초 단위)
    timestamp_ms = int(time.time() * 1000)

    # 얼굴 랜드마크 검출
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # 기본 상태
    status_text = "AWAKE"

    # 얼굴이 검출된 경우
    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        # 눈 좌표 추출
        left_eye = get_eye_points(face_landmarks, LEFT_EYE, width, height)
        right_eye = get_eye_points(face_landmarks, RIGHT_EYE, width, height)

        # 눈 좌표 표시
        draw_eye_points(frame, left_eye)
        draw_eye_points(frame, right_eye)

        # EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # EAR 값 출력
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 눈 감김 여부 판단
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0
            alarm_on = False

        # 졸음 판정
        if closed_frames >= CLOSED_FRAMES_THRESHOLD:
            status_text = "DROWSINESS ALERT"

            cv2.putText(
                frame,
                status_text,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            if not alarm_on:
                play_alarm()
                alarm_on = True

    else:
        # 얼굴 미검출 시 초기화
        closed_frames = 0
        alarm_on = False
        status_text = "NO FACE"

    # 상태 표시
    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # 눈 감김 프레임 표시
    cv2.putText(
        frame,
        f"CLOSED FRAMES: {closed_frames}",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # 화면 출력
    cv2.imshow("Drowsiness Detection System", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# 자원 해제
# ==============================
cap.release()
cv2.destroyAllWindows()