# EARSYS

EAR(Eye Aspect Ratio) 기반 실시간 졸음 감지 시스템입니다. MediaPipe Face Landmarker로 얼굴 랜드마크를 추적하고, 눈 감김 상태를 계산한 뒤 EyeFrame 데이터그램을 Unix domain socket(UDS)으로 전송합니다.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)](https://mediapipe.dev)

```text
Camera(OpenCV) -> FaceLandmarker(MediaPipe) -> EAR -> EyeFrame(UDS)
```

## Features

- EAR 기반 AWAKE/DROWSY/NO_FACE 상태 판정
- Linux, Raspberry Pi, Windows, macOS 카메라 자동 프로필 선택
- GStreamer, V4L2, DirectShow, AVFoundation, OpenCV 기본 백엔드 지원
- UDS datagram 기반 EyeFrame 프로토콜
- 카메라 재연결 루프와 비동기 UDS 전송
- 하드웨어 의존 없는 pytest 단위 테스트

## Requirements

- Python `3.12`
- `face_landmarker.task` MediaPipe 모델 파일
- OpenCV에서 접근 가능한 카메라
- Linux에서 UDS receiver를 함께 사용할 경우 UDS 권한/주소 설정

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

모델 파일을 프로젝트 루트에 내려받습니다.

```bash
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -O face_landmarker.task
```

실행합니다.

```bash
python main.py
```

패키지 entry point로 실행하려면 editable install을 추가로 수행합니다.

```bash
python -m pip install -e .
earsys
```

## Configuration

모든 실행 설정은 환경변수로 override할 수 있습니다. 기본값은 자동 카메라 선택을 사용합니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EARSYS_MODEL_PATH` | `<project>/face_landmarker.task` | MediaPipe 모델 경로 |
| `EARSYS_CAMERA_SOURCE` | `auto` | 카메라 인덱스, 장치 경로, 파일 경로, URL |
| `EARSYS_CAMERA_BACKEND` | `auto` | `auto`, `gstreamer`, `v4l2`, `directshow`, `avfoundation` |
| `EARSYS_GST_PIPELINE` | 없음 | 지정 시 최우선으로 사용할 GStreamer pipeline |
| `EARSYS_CAMERA_COLOR_FORMAT` | `auto` | `auto`, `bgr`, `rgb`, `nv12` |
| `EARSYS_CAMERA_WIDTH` | `640` | 일반 카메라 프로필 요청 너비 |
| `EARSYS_CAMERA_HEIGHT` | `480` | 일반 카메라 프로필 요청 높이 |
| `EARSYS_CAMERA_FPS` | `30` | 일반 카메라 프로필 요청 FPS |
| `EARSYS_UDS_ADDR` | `abstract:earsys/eye` | `abstract:<name>` 또는 `path:<path>` |
| `EARSYS_EAR_THRESHOLD` | `0.23` | 눈 감김 판정 EAR threshold |
| `EARSYS_CLOSED_FRAMES` | `20` | DROWSY 판정에 필요한 연속 감김 프레임 수 |
| `EARSYS_LOG_LEVEL` | `INFO` | Python logging level |
| `EARSYS_CAMERA_REOPEN_SEC` | `2.0` | 카메라 재연결 대기 시간 |
| `EARSYS_CAMERA_MAX_RETRIES` | `0` | 카메라 재연결 최대 횟수. `0`은 무제한 |

## Camera Selection

`EARSYS_CAMERA_SOURCE=auto`일 때 EARSYS는 실행 시스템에 맞는 후보를 만들고, 실제로 열리는 첫 후보를 사용합니다.

| 시스템 | 자동 후보 순서 |
|--------|----------------|
| Raspberry Pi Linux | OpenCV GStreamer 지원 + `libcamera-vid` 존재 시 `libcamerasrc` NV12 pipeline |
| Linux | `/dev/video0`부터 `/dev/video3`까지 V4L2 장치, 이후 OpenCV 기본 카메라 |
| Windows | DirectShow 기본 카메라, 이후 OpenCV 기본 카메라 |
| macOS | AVFoundation 기본 카메라, 이후 OpenCV 기본 카메라 |
| 기타 | OpenCV 기본 카메라 |

자동 선택이 맞지 않으면 명시 설정을 사용합니다.

```bash
# Linux V4L2
EARSYS_CAMERA_SOURCE=/dev/video0 \
EARSYS_CAMERA_BACKEND=v4l2 \
python main.py

# Windows DirectShow
EARSYS_CAMERA_SOURCE=0 \
EARSYS_CAMERA_BACKEND=directshow \
python main.py

# macOS AVFoundation
EARSYS_CAMERA_SOURCE=0 \
EARSYS_CAMERA_BACKEND=avfoundation \
python main.py

# Raspberry Pi libcamera / GStreamer
EARSYS_GST_PIPELINE='libcamerasrc ! video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! queue leaky=downstream max-size-buffers=1 ! appsink drop=true max-buffers=1 sync=false' \
EARSYS_CAMERA_COLOR_FORMAT=nv12 \
python main.py
```

## EyeFrame UDS Protocol

EARSYS는 `SOCK_DGRAM` Unix domain socket으로 24바이트 EyeFrame을 전송합니다.

기본 주소:

```text
abstract:earsys/eye
```

파일시스템 socket을 사용할 경우:

```bash
EARSYS_UDS_ADDR=path:/run/earsys/eye.sock python main.py
```

패킷 형식은 little-endian `<4sBBHfIQ>`입니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `magic` | `4s` | `SEYE` |
| `version` | `uint8` | `1` |
| `status` | `uint8` | `0=AWAKE`, `1=DROWSY`, `2=NO_FACE` |
| `reserved` | `uint16` | 예약 필드 |
| `eye_score` | `float32` | `0.0`은 눈 뜸, `1.0`은 눈 감김 |
| `seq` | `uint32` | 송신 시퀀스 번호 |
| `ts_ms` | `uint64` | Unix epoch milliseconds |

수신 예시:

```python
import socket
import struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(b"\x00earsys/eye")

data, _ = sock.recvfrom(64)
magic, version, status, reserved, eye_score, seq, ts_ms = struct.unpack("<4sBBHfIQ", data)
```

## Project Layout

```text
EARSYS/
├── main.py
├── face_landmarker.task
├── pyproject.toml
├── requirements.txt
├── earsys/
│   ├── camera.py           # OpenCV camera wrapper and fallback opening
│   ├── camera_profile.py   # system-aware camera profile selection
│   ├── config.py           # environment-backed constants
│   ├── detector.py         # MediaPipe FaceLandmarker wrapper
│   ├── ear.py              # pure EAR geometry helpers
│   ├── uds_async.py        # async send queue
│   └── uds_bridge.py       # EyeFrame UDS serialization/sending
├── tests/
│   ├── test_camera.py
│   ├── test_camera_profile.py
│   ├── test_ear.py
│   ├── test_uds_bridge.py
│   └── test_visual_opencv.py
└── scripts/
    └── earsys.service
```

## Tests

기본 테스트는 실제 카메라, UDS receiver, MediaPipe 모델 파일에 의존하지 않습니다.

```bash
pytest -q
python -m compileall main.py earsys tests
```

수동 OpenCV 시각화 테스트는 명시적으로 켭니다.

```bash
EARSYS_RUN_VISUAL_TESTS=1 pytest tests/test_visual_opencv.py -q
```

## systemd Example

`scripts/earsys.service`는 Linux systemd 배포용 예시입니다. 실행 사용자, 카메라 권한, 모델 경로, UDS 주소는 대상 시스템에 맞게 설정해야 합니다.

```bash
sudo cp scripts/earsys.service /etc/systemd/system/
sudo install -d /etc/earsys

sudo tee /etc/earsys/earsys.env >/dev/null <<'EOF'
EARSYS_MODEL_PATH=/opt/earsys/face_landmarker.task
EARSYS_CAMERA_SOURCE=auto
EARSYS_CAMERA_BACKEND=auto
EARSYS_UDS_ADDR=abstract:earsys/eye
EARSYS_LOG_LEVEL=INFO
EOF

sudo systemctl daemon-reload
sudo systemctl enable earsys
sudo systemctl start earsys
journalctl -u earsys -f
```

## EAR Logic

EAR 공식:

```text
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        p2    p3
    p1              p4
        p6    p5
```

판정 흐름:

- `EAR < EARSYS_EAR_THRESHOLD`이면 눈 감김 프레임을 누적합니다.
- 누적 프레임이 `EARSYS_CLOSED_FRAMES` 이상이면 `DROWSY`를 전송합니다.
- 얼굴이 감지되지 않으면 `NO_FACE`를 전송합니다.
- 같은 `NO_FACE` 상태는 반복 전송하지 않습니다.
