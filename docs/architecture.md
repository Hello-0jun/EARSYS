# Architecture

EARSYS is structured around four domain-specific concerns that each map to a dedicated sub-package. The boundary between them is intentionally thin — data flows in a single direction and no domain imports from one that is logically "downstream" of it.

## Data Flow

``` mermaid
flowchart TD
    CAM["Camera\nOpenCV"] -->|"BGR / NV12 frames"| DET
    DET["FaceDetector\nMediaPipe LIVE_STREAM"] -->|"face_landmarks_list"| EAR
    EAR["EAR Calculation\nearsys.vision.ear"] -->|"EAR float 0.0 – ~0.4"| STATE
    STATE["DrowsinessState\nearsys.loop"] -->|"AWAKE / DROWSY / NO_FACE"| ASYNC
    ASYNC["UdsAsyncBridge"] -->|"queue"| BRIDGE
    BRIDGE["UdsBridge"] -->|"24-byte EyeFrame datagram"| UDS
    UDS(["Unix Domain Socket\nSOCK_DGRAM\nabstract:earsys/eye"])
```

## Sub-packages

### `earsys.camera`

Owns everything related to physical frame acquisition.

- **`profile.py`** probes the running system for available hardware (GStreamer, libcamera, V4L2 devices) and returns an ordered list of `CameraProfile` candidates. All probes are isolated functions that accept no side-effectful globals, making them straightforward to mock in tests.
- **`capture.py`** wraps `cv2.VideoCapture` behind `OpenCvCamera`, a context-manager that iterates through the profile list and opens the first one that succeeds.

### `earsys.vision`

Owns landmark extraction and the EAR metric.

- **`detector.py`** wraps MediaPipe's `FaceLandmarker` in `LIVE_STREAM` mode. It maintains a monotonically-increasing timestamp internally (required by the API) and surfaces results through a single-slot queue so the main loop always receives the freshest inference rather than a stale one.
- **`ear.py`** contains only pure, stateless functions. `calculate_ear` implements the standard six-point formula; `average_ear` merges both eyes into a single scalar.
- **`draw.py`** renders landmark overlays and status text onto an OpenCV window. It is only invoked when `feature_visualize_landmarks` is enabled and is never imported in production paths.

### `earsys.ipc`

Owns serialization and socket delivery.

- **`uds_bridge.py`** packs each detection result into a 24-byte `EyeFrame` struct and sends it as a UDP-style datagram over a Unix Domain Socket. Receiver-unavailable errors (`ECONNREFUSED`, `ENOENT`) are silently dropped with a rate-limited log warning so a missing consumer never crashes the detector.
- **`uds_async.py`** wraps `UdsBridge` in a daemon worker thread with a bounded queue. The main detection loop calls `send()` on `UdsAsyncBridge` and returns immediately — socket I/O is fully off the hot path.

### `earsys.config`

Single source of truth for all tunable values and protocol constants.

The module-level `settings` singleton is a `pydantic-settings` `BaseSettings` instance. It reads from environment variables (prefix `EARSYS_`) and the appropriate `.env.<env>` file on import. Hardware capability flags (`feature_gstreamer`, `feature_libcamera`, `feature_v4l2`) are `@property` methods that delegate to the probing functions in `earsys.camera.profile` on first access.

## EyeFrame Binary Protocol

Every detection result is serialized into a fixed 24-byte datagram and delivered over `SOCK_DGRAM` to the configured UDS address.

Format: little-endian `<4sBBHfIQ>`

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 B | `magic` | `b"SEYE"` |
| 4 | 1 B | `version` | `1` |
| 5 | 1 B | `status` | `0` AWAKE · `1` DROWSY · `2` NO_FACE |
| 6 | 2 B | `reserved` | always `0` |
| 8 | 4 B | `eye_score` | `0.0` open → `1.0` closed, float32 |
| 12 | 4 B | `seq` | monotonically-increasing send counter |
| 16 | 8 B | `ts_ms` | Unix epoch milliseconds |

`eye_score` is derived from EAR via a clamped linear interpolation between `ear_open_thr` (default `0.30`) and `ear_closed_thr` (default `0.15`).

### Receiving a Frame

```python
import socket, struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(b"\x00earsys/eye")   # abstract namespace

data, _ = sock.recvfrom(64)
magic, version, status, reserved, eye_score, seq, ts_ms = struct.unpack("<4sBBHfIQ", data)
```

## Camera Profile Resolution

When no explicit camera settings are provided, `resolve_camera_profiles()` returns candidates in the following priority order (Linux):

``` mermaid
flowchart LR
    ENV{"Explicit env vars?\nGST_PIPELINE\nCAMERA_SOURCE"}
    ENV -->|Yes| EXPLICIT["Use explicit profile\n\(skip auto-detection\)"]
    ENV -->|No| PLAT{Platform}

    PLAT -->|Linux| L1["GStreamer + libcamera\nlibcamerasrc NV12"]
    L1 --> L2["V4L2 devices\n/dev/video0 … /dev/video3"]
    L2 --> L3["OpenCV default\nCAP_ANY index 0"]

    PLAT -->|Windows| W1["DirectShow\nCAP_DSHOW"]
    W1 --> W2["OpenCV default"]

    PLAT -->|macOS| M1["AVFoundation\nCAP_AVFOUNDATION"]
    M1 --> M2["OpenCV default"]
```

Each candidate is tried in order by `OpenCvCamera`. The first one that opens successfully is used; the rest are discarded.
