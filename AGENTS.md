# EARSYS Agent Guidelines

Welcome, AI Coding Assistant! This document provides an overview of the EARSYS architecture and established coding guidelines to help you effectively navigate and modify this project.

## Architecture & Layout

EARSYS is an EAR-based drowsiness detection system using MediaPipe and OpenCV, sending its results over Unix Domain Sockets (UDS). The project is organized into domain-specific subpackages:

```text
earsys/
├── cli.py             # Typer-based CLI interface
├── config.py          # Centralized configuration (pydantic-settings)
├── loop.py            # Main application loop (`run_detection`)
├── camera/
│   ├── capture.py     # OpenCV camera abstraction
│   └── profile.py     # Hardware capability probing and camera selection
├── vision/
│   ├── detector.py    # MediaPipe FaceLandmarker wrapper
│   └── ear.py         # Pure EAR geometry and distance calculations
└── ipc/
    ├── uds_async.py   # Asynchronous queue-based UDS bridge wrapper
    └── uds_bridge.py  # Synchronous socket communication
```

### Key Data Flows
1. **Camera**: `OpenCvCamera` yields raw BGR frames.
2. **Vision**: `FaceDetector` (MediaPipe) extracts facial landmarks.
3. **Logic**: `average_ear` calculates eye aspect ratio.
4. **IPC**: Results are serialized into a 24-byte `EyeFrame` binary struct and transmitted via UDS using `UdsBridge`.

## Coding Standards

1. **Modern Python**: Use Python 3.12+ features (e.g., `|` union types, walrus operator `:=`).
2. **Strict Type Hinting**: All functions must have type hints. `ruff` will complain if `ANN` rules are violated (except in tests).
3. **Configuration**: Do NOT hardcode settings. Use `earsys.config.settings` for all environment-variable-backed configurations. Feature flags are boolean properties derived dynamically or read from `pydantic-settings`.
4. **Linting and Formatting**: We use `ruff`.
   - Run `uv run ruff check --fix` and `uv run ruff format` before concluding any task.
   - Max line length is 120.
5. **Testing**: `pytest` is used. Tests must not depend on actual hardware (mock `cv2.VideoCapture` and `socket.socket`).
6. **Dependency Management**: We use `uv` and `pyproject.toml` (PEP 735 dependency groups).

## AI Workflow Instructions

- **Understand Context**: Before making structural changes, refer back to this document to see if a domain subdirectory already handles that concern.
- **Maintain Testability**: If you add new hardware interactions or system probes, make sure they are mockable. See `probe_system_capabilities()` in `earsys/camera/profile.py` for inspiration.
- **Documentation**: Keep `README.md` and docstrings aligned with your code changes.
