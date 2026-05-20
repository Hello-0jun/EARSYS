# Configuration (`earsys.config`)

The `earsys.config` module provides centralized configuration management using `pydantic-settings`.
Settings are resolved hierarchically, prioritizing environment variables over `.env` files.

## Resolution Order

Settings are applied in the following priority (highest wins):

1. **Process environment variables** — `EARSYS_*` set before the process starts or injected by the CLI.
2. **CLI flags** — translated into `os.environ` entries before `AppSettings` is imported.
3. **`.env.<env>` file** — `.env.dev` when `EARSYS_ENV=dev`, `.env.prod` otherwise.
4. **Code defaults** — the `Field(default=...)` values in `AppSettings`.

## Environment Variables

All variables use the prefix `EARSYS_`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `EARSYS_ENV` | `prod` | `dev` or `prod`. Selects which `.env` file is loaded. |
| `EARSYS_LOG_LEVEL` | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `EARSYS_MODEL_PATH` | `<project root>/face_landmarker.task` | Absolute path to the MediaPipe `face_landmarker.task` model file. |

### Unix Domain Socket

| Variable | Default | Description |
|----------|---------|-------------|
| `EARSYS_UDS_ADDR` | `abstract:earsys/eye` | UDS address. Use `abstract:<name>` for Linux abstract namespace or `path:<path>` / bare path for filesystem sockets. |

### Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `EARSYS_CAMERA_SOURCE` | `auto` | Camera index (`0`), device path (`/dev/video0`), or stream URL. `auto` triggers system detection. |
| `EARSYS_CAMERA_BACKEND` | `auto` | OpenCV backend: `auto`, `gstreamer`, `v4l2`, `directshow`, `avfoundation`. |
| `EARSYS_CAMERA_COLOR_FORMAT` | `auto` | Input frame color format: `auto`, `bgr`, `rgb`, `nv12`. |
| `EARSYS_GST_PIPELINE` | *(unset)* | Full GStreamer pipeline string. When set, all other camera settings are ignored. |
| `EARSYS_CAMERA_WIDTH` | `640` | Capture width in pixels. |
| `EARSYS_CAMERA_HEIGHT` | `480` | Capture height in pixels. |
| `EARSYS_CAMERA_FPS` | `30` | Capture frame rate. |
| `EARSYS_CAMERA_REOPEN_SEC` | `2.0` | Seconds to wait before retrying a failed camera open. |
| `EARSYS_CAMERA_MAX_RETRIES` | `0` | Maximum camera reconnect attempts. `0` means unlimited. |

### EAR / Drowsiness Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `EARSYS_EAR_THRESHOLD` | `0.23` | EAR value below which a frame is counted as a closed eye. |
| `EARSYS_CLOSED_FRAMES_THRESHOLD` | `20` | Consecutive closed-eye frames required to trigger `DROWSY`. |
| `EARSYS_EAR_OPEN_THR` | `0.30` | EAR ≥ this value maps to `eye_score = 0.0`. |
| `EARSYS_EAR_CLOSED_THR` | `0.15` | EAR ≤ this value maps to `eye_score = 1.0`. |

### Feature Flags

| Variable | Default (dev) | Default (prod) | Description |
|----------|--------------|----------------|-------------|
| `EARSYS_FEATURE_VISUALIZE_LANDMARKS` | `true` | `false` | Show OpenCV debug window with landmark and EAR overlays. |
| `EARSYS_FEATURE_DEBUG_LOGGING` | `true` | `false` | Emit per-frame `DEBUG` log entries. |
| `EARSYS_FEATURE_UDS_ENABLED` | `true` | `true` | Enable UDS datagram output. |

## Hardware Capability Properties

These are `@property` methods on `AppSettings` — they are **not** backed by environment variables and cannot be set via `.env` files. Each delegates to a probe function in `earsys.camera.profile`.

| Property | `True` when |
|----------|-------------|
| `settings.feature_gstreamer` | OpenCV was compiled with GStreamer support (`cv2.getBuildInformation()` contains `"gstreamer"`). |
| `settings.feature_libcamera` | `libcamera-vid` is found on `PATH`. |
| `settings.feature_v4l2` | At least one of `/dev/video0` … `/dev/video3` exists and is readable. |

## Protocol Constants

The following constants are defined at module level and are not tunable:

```python
EYE_FRAME_MAGIC   = b"SEYE"
EYE_FRAME_VERSION = 1
EYE_FRAME_FORMAT  = "<4sBBHfIQ"   # little-endian, 24 bytes
EYE_FRAME_SIZE    = 24

STATUS_AWAKE   = 0
STATUS_DROWSY  = 1
STATUS_NO_FACE = 2

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
```

## `AppSettings` API Reference

::: earsys.config.AppSettings
