# EARSYS

EAR(Eye Aspect Ratio) 기반 실시간 졸음 감지 시스템입니다. MediaPipe Face Landmarker로 얼굴 랜드마크를 추적하고, 눈 감김 상태를 계산한 뒤 EyeFrame 데이터그램을 Unix domain socket(UDS)으로 전송합니다.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)](https://mediapipe.dev)

```text
Camera(OpenCV) -> FaceLandmarker(MediaPipe) -> EAR -> EyeFrame(UDS)
```

## Features

- EAR 기반 AWAKE/DROWSY/NO_FACE 상태 판정
- Linux, Raspberry Pi, Windows, macOS 카메라 자동 프로필 선택 (GStreamer, V4L2, DirectShow, AVFoundation 등)
- UDS datagram 기반 EyeFrame 프로토콜 전송
- 카메라 재연결 루프와 비동기 UDS 전송
- 모듈화된 프로젝트 구조 및 하드웨어 독립적인 단위 테스트 지원
- Typer & Rich 기반의 미려한 CLI 제공

## Requirements

- Python `3.12+`
- `uv` (현대적인 파이썬 패키지 매니저)
- `face_landmarker.task` MediaPipe 모델 파일
- OpenCV에서 접근 가능한 카메라
- Linux에서 UDS receiver를 함께 사용할 경우 UDS 권한/주소 설정

## Quick Start

프로젝트 복제 후 `uv`를 이용하여 환경을 구축합니다.

```bash
# 의존성 설치 및 가상 환경 생성
uv sync

# MediaPipe 모델 다운로드
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -O face_landmarker.task

# 실행 (dev 모드로 시각화 창 활성화)
uv run earsys run --env dev --visualize
```

## Configuration

설정은 환경변수 혹은 `.env.dev`, `.env.prod` 파일을 통해 제어할 수 있습니다. `earsys config` 명령어로 현재 적용된 설정과 시스템 호환성을 확인할 수 있습니다.

| 환경 변수 | 기본값 | 설명 |
|------|--------|------|
| `EARSYS_ENV` | `prod` | `dev` 또는 `prod` 실행 환경 |
| `EARSYS_MODEL_PATH` | `<project>/face_landmarker.task` | MediaPipe 모델 경로 |
| `EARSYS_CAMERA_SOURCE` | `auto` | 카메라 인덱스, 장치 경로, 파일 경로, URL |
| `EARSYS_CAMERA_BACKEND` | `auto` | `auto`, `gstreamer`, `v4l2`, `directshow`, `avfoundation` |
| `EARSYS_GST_PIPELINE` | 없음 | 지정 시 최우선으로 사용할 GStreamer pipeline |
| `EARSYS_CAMERA_COLOR_FORMAT` | `auto` | `auto`, `bgr`, `rgb`, `nv12` |
| `EARSYS_UDS_ADDR` | `abstract:earsys/eye` | `abstract:<name>` 또는 `path:<path>` |
| `EARSYS_EAR_THRESHOLD` | `0.23` | 눈 감김 판정 EAR threshold |
| `EARSYS_CLOSED_FRAMES` | `20` | DROWSY 판정에 필요한 연속 감김 프레임 수 |
| `EARSYS_CAMERA_MAX_RETRIES` | `0` | 카메라 재연결 최대 횟수. `0`은 무제한 |

## Project Layout

최근 리팩터링을 통해 시스템 아키텍처가 도메인별로 분리되었습니다.

```text
EARSYS/
├── AGENTS.md               # AI 코딩 어시스턴트를 위한 가이드라인
├── README.md               # 프로젝트 개요 및 설명서
├── pyproject.toml          # 의존성 및 프로젝트 메타데이터 (uv, ruff, pytest 설정)
├── main.py                 # 구버전 호환성 및 CLI 진입점
├── earsys/                 # 핵심 패키지
│   ├── cli.py              # Typer CLI 인터페이스
│   ├── config.py           # 환경변수(Pydantic-settings) 기반 설정 관리
│   ├── loop.py             # 주 감지 루프 및 시각화 코드
│   ├── camera/             # 카메라 관리 모듈
│   │   ├── capture.py      # OpenCV 캡처 추상화
│   │   └── profile.py      # 하드웨어 탐색 및 카메라 프로필 선택
│   ├── vision/             # 비전 및 랜드마크 계산 모듈
│   │   ├── detector.py     # MediaPipe FaceLandmarker 래퍼
│   │   └── ear.py          # EAR 수치 계산
│   └── ipc/                # 통신 모듈
│       ├── uds_async.py    # 비동기 전송 래퍼
│       └── uds_bridge.py   # UDS 소켓 및 바이너리 프로토콜 통신
└── tests/                  # 단위 테스트 (Pytest)
```

## EyeFrame UDS Protocol

EARSYS는 `SOCK_DGRAM` Unix domain socket으로 24바이트 EyeFrame을 전송합니다.

기본 주소: `abstract:earsys/eye`

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

## Tests

기본 테스트는 실제 하드웨어 의존성 없이 실행됩니다.

```bash
uv run pytest -q
```
