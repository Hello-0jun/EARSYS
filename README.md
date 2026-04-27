# EARSYS

> **EAR-based Drowsiness Detection System**
> EAR(Eye Aspect Ratio) 알고리즘 기반 실시간 졸음 감지 시스템

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)](https://mediapipe.dev)

---

## 개요

EARSYS는 MediaPipe Face Landmarker를 사용하여 카메라 영상에서 눈의 개폐 상태를 실시간으로 분석합니다.
눈 감김 지속 시간이 임계값을 초과하면 졸음으로 판정하고, POSIX 공유 메모리(SHM)를 통해 외부 프로세스(예: ClockApp LVGL 앱)에 상태를 전달합니다.

```
카메라(GStreamer) ─→ MediaPipe FaceLandmarker ─→ EAR 계산 ─→ SHM 상태 기록
                                                              └─→ 알람
```

---

## 프로젝트 구조

```
EARSYS/
├── main.py                      # 진입점 (earsys/ 패키지 조합)
├── face_landmarker.task         # MediaPipe 모델 파일 (별도 다운로드)
├── requirements.txt
│
├── earsys/                      # 핵심 패키지
│   ├── __init__.py
│   ├── config.py                # 설정 상수 / 환경변수 처리
│   ├── ear.py                   # EAR 계산 순수 함수
│   ├── detector.py              # MediaPipe FaceLandmarker 래퍼
│   ├── shm_bridge.py            # POSIX SHM 읽기/쓰기 (seqlock)
│   ├── camera.py                # GStreamer 카메라 추상화
│   └── alarm.py                 # 알람 인터페이스
│
├── tests/
│   ├── test_ear.py              # EAR 함수 단위 테스트
│   ├── test_shm_bridge.py       # SHM 브리지 단위 테스트
│   └── test_visual_opencv.py    # 수동 시각화 테스트 (웹캠)
│
└── scripts/
    ├── earsys.service           # systemd 서비스 파일
    └── rpi_picamera2_example.py # Picamera2 백엔드 참고 코드
```

---

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 모델 파일 다운로드

```bash
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
     -O face_landmarker.task
```

### 3. 실행

```bash
python main.py
```

---

## 환경변수 설정

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `EARSYS_MODEL_PATH` | `<프로젝트 루트>/face_landmarker.task` | 모델 파일 경로 |
| `EARSYS_SHM_NAME` | `/earsys_drowsy_shm` | POSIX SHM 이름 |
| `EARSYS_GST_PIPELINE` | `libcamerasrc ! ...` | GStreamer 파이프라인 |
| `EARSYS_EAR_THRESHOLD` | `0.23` | 눈 감김 판정 EAR 임계값 |
| `EARSYS_CLOSED_FRAMES` | `20` | 졸음 판정 연속 프레임 수 |
| `EARSYS_LOG_LEVEL` | `INFO` | 로그 레벨 (DEBUG/INFO/WARNING) |

---

## 공유 메모리 프로토콜

SHM 이름: `/earsys_drowsy_shm` (기본값)

| 오프셋 | 크기 | 설명 |
|--------|------|------|
| 0 | 4 bytes | Magic: `EARS` |
| 4 | 4 bytes | 버전: `1` (uint32 LE) |
| 8 | 4 bytes | Seqlock 시퀀스 번호 (홀수=쓰기 중) |
| 12 | 4 bytes | 상태 코드 (0=AWAKE, 1=DROWSY, 2=NO_FACE) |

**Consumer(C/C++) 읽기 패턴:**

```c
uint32_t seq1, seq2, status;
do {
    seq1 = read_u32(shm + 8);
    if (seq1 & 1) continue;          // 쓰기 중 → 재시도
    status = read_u32(shm + 12);
    seq2 = read_u32(shm + 8);
} while (seq1 != seq2);             // 재시도 완료
```

---

## 단위 테스트

```bash
pytest tests/test_ear.py tests/test_shm_bridge.py -v
```

---

## Raspberry Pi 배포 (systemd)

```bash
# 서비스 파일 배포
sudo cp scripts/earsys.service /etc/systemd/system/

# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable earsys
sudo systemctl start earsys

# 로그 확인
journalctl -u earsys -f
```

> **서비스 파일 수정 필요:** `User`, `WorkingDirectory`, `ExecStart` 경로를 실제 환경에 맞게 수정하세요.

---

## EAR 알고리즘

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

        p2    p3
    p1              p4
        p6    p5

EAR > 0.23 → 눈 뜸 (AWAKE)
EAR < 0.23 → 눈 감김 카운트 증가
연속 20 프레임 이상 → DROWSY 판정
```
