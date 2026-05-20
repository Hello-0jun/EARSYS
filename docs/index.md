# EARSYS Documentation

**EARSYS** is an EAR (Eye Aspect Ratio) based drowsiness detection system. It uses MediaPipe Face Landmarker to continuously track facial landmarks, derives an eye-openness metric from them, and streams the results as compact binary datagrams over a Unix Domain Socket so any other process on the same machine can consume them without polling.

```
Camera (OpenCV) → FaceLandmarker (MediaPipe) → EAR → EyeFrame (UDS)
```

## Key Features

- **Cross-platform camera acquisition** — automatic profile selection for GStreamer + libcamera (Raspberry Pi), V4L2 (Linux), DirectShow (Windows), and AVFoundation (macOS).
- **Non-blocking IPC** — the detection loop never stalls on socket I/O; a daemon worker thread drains the send queue asynchronously.
- **Environment-driven configuration** — every tunable parameter is controlled by an `EARSYS_*` environment variable, with no values buried in code.
- **Hardware-independent test suite** — all camera and socket interactions are injected through interfaces, so tests run without any physical devices.
- **Rich CLI** — a Typer + Rich interface with a live dashboard, start/stop banners, and a camera reconnect progress bar.

## Quick Start

```bash
# Install dependencies
uv sync

# Download the MediaPipe model
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -O face_landmarker.task

# Run (dev mode — shows visualisation window)
uv run earsys run --env dev --visualize

# Check what the system detected and what settings are active
uv run earsys config
```

## Where to Go Next

| Topic | Page |
|-------|------|
| How the pieces fit together | [Architecture](architecture.md) |
| Setting up a local dev environment | [Development Setup](setup.md) |
| All CLI commands and flags | [Usage](usage.md) |
| Deploying to Raspberry Pi / Linux | [Deployment](deployment.md) |
| Coding style and test patterns | [Conventions](conventions.md) |
| Writing and running tests | [Testing](testing.md) |
| Full configuration reference | [Configuration](api/config.md) |
| EyeFrame UDS protocol | [UDS Bridge](api/ipc/uds_bridge.md) |
