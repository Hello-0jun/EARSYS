# Testing

EARSYS has a hardware-independent test suite. No camera, socket, or MediaPipe model is required to run any test. All external interactions are replaced with lightweight fakes injected through `monkeypatch` or constructor arguments.

## Running Tests

```bash
# Fast pass — quiet output
uv run pytest -q

# Verbose — shows every test name and status
uv run pytest -v

# Single file
uv run pytest tests/test_ear.py -v

# Single test
uv run pytest tests/test_uds_bridge.py::test_send_serializes_expected_eyeframe -v
```

The CI pipeline also compiles all sources after the tests as a quick import sanity check:

```bash
uv run python -m compileall earsys tests
```

## Test Files

| File | What it covers |
|------|----------------|
| `test_ear.py` | `calculate_ear`, `average_ear`, `get_eye_points` — pure geometry functions |
| `test_camera.py` | `OpenCvCamera` open / profile fallback / context manager / frame iteration |
| `test_camera_profile.py` | `resolve_camera_profiles` — Linux, Windows, macOS, explicit env, GStreamer + libcamera |
| `test_uds_bridge.py` | EyeFrame serialization, sequence numbers, error silencing, `UdsBridge` lifecycle |
| `test_visual_opencv.py` | `show_debug_frame` — gated behind `EARSYS_RUN_VISUAL_TESTS=1` |

## Mocking Patterns

### Socket — `UdsBridge`

Replace `socket.socket` with a fake that records calls:

```python
import earsys.ipc.uds_bridge as uds_bridge_module
from earsys.ipc.uds_bridge import UdsBridge

def test_something(monkeypatch):
    sent: list[tuple[bytes, str]] = []

    class FakeSocket:
        def sendto(self, data, addr):
            sent.append((data, addr))
            return len(data)

        def close(self):
            pass

    monkeypatch.setattr(
        uds_bridge_module.socket, "socket", lambda *_: FakeSocket()
    )

    with UdsBridge() as bridge:
        bridge.send(status=0, ear=0.32)

    assert len(sent) == 1
```

### Camera — `OpenCvCamera`

Replace `cv2.VideoCapture` with a fake that returns pre-built frames:

```python
import cv2
import numpy as np

class FakeCapture:
    def __init__(self, frames):
        self._frames = iter(frames)

    def isOpened(self):
        return True

    def read(self):
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self):
        pass

    def set(self, *_):
        pass

def test_camera_reads_frames(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCapture([frame]))

    with OpenCvCamera() as cam:
        result = cam.read()

    assert result is not None
```

### Camera Profiles — `SystemCapabilities` injection

Never touch real `/dev/videoN` paths. Pass a `SystemCapabilities` snapshot directly:

```python
from earsys.camera.profile import resolve_camera_profiles, SystemCapabilities
from pathlib import Path

def test_linux_libcamera_profile_is_first():
    profiles = resolve_camera_profiles(
        env={},
        caps=SystemCapabilities(
            gstreamer=True,
            libcamera=True,
            v4l2_devices=(Path("/dev/video0"),),
        ),
    )
    assert profiles[0].backend == "gstreamer"
    assert "libcamera" in profiles[0].label
```

## Visual Tests

`test_visual_opencv.py` opens a real OpenCV window and is skipped in CI. To run it locally:

```bash
EARSYS_RUN_VISUAL_TESTS=1 uv run pytest tests/test_visual_opencv.py -v
```

## EyeFrame Unpacking Helper

When asserting on the binary content of a sent datagram, unpack it with:

```python
import struct
from earsys.config import EYE_FRAME_FORMAT

def unpack_frame(data: bytes):
    return struct.unpack(EYE_FRAME_FORMAT, data)

magic, version, status, reserved, eye_score, seq, ts_ms = unpack_frame(data)
```

## pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
```

The `ANN`, `SIM`, and `PTH` Ruff rules are relaxed for the `tests/` directory (see `per-file-ignores` in `pyproject.toml`).
