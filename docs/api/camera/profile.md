# Camera Profile (`earsys.camera.profile`)

This module is responsible for translating the current runtime environment into an ordered list of `CameraProfile` candidates. It separates *capability detection* from *camera opening* — `OpenCvCamera` only needs to iterate the list and try each one.

## `CameraProfile`

An immutable dataclass describing a single camera configuration attempt.

```python
@dataclass(frozen=True)
class CameraProfile:
    source: int | str       # OpenCV VideoCapture source argument
    backend: str            # e.g. "gstreamer", "v4l2", "directshow", "auto"
    color_format: str       # "bgr", "rgb", or "nv12"
    width: int | None       # None = let the driver decide
    height: int | None
    fps: int | None
    label: str              # human-readable name used in log messages
```

## `SystemCapabilities`

A frozen snapshot of what the current machine supports. Obtained once at startup via `probe_system_capabilities()` and passed around rather than re-probing on every call.

| Field | Type | Probe method |
|-------|------|-------------|
| `gstreamer` | `bool` | `cv2.getBuildInformation()` contains `"gstreamer"` |
| `libcamera` | `bool` | `shutil.which("libcamera-vid") is not None` |
| `v4l2_devices` | `tuple[Path, ...]` | `/dev/video0` … `/dev/video3` exist and are readable |

```python
from earsys.camera.profile import probe_system_capabilities

caps = probe_system_capabilities()
print(caps.gstreamer)     # True / False
print(caps.has_v4l2)      # True when at least one device exists
print(caps.v4l2_devices)  # (PosixPath('/dev/video0'),)
```

## `resolve_camera_profiles()`

```python
def resolve_camera_profiles(
    env: Mapping[str, str] | None = None,
    caps: SystemCapabilities | None = None,
) -> list[CameraProfile]:
```

Returns an ordered list of profiles. `OpenCvCamera` tries them in order and stops at the first one that opens successfully.

### Resolution Logic

**Step 1 — Explicit operator settings win**

If any of the following variables are set (and not `"auto"`), only that profile is returned and auto-detection is skipped entirely:

- `EARSYS_GST_PIPELINE` — constructs a GStreamer profile from the raw pipeline string.
- `EARSYS_CAMERA_SOURCE` and/or `EARSYS_CAMERA_BACKEND` — constructs a profile from the explicit source and backend.

**Step 2 — Platform auto-detection**

When no explicit settings are given, the platform is detected via `platform.system()`:

=== "Linux"

    Profiles are returned in this priority order:

    1. **libcamera GStreamer** — when both `caps.gstreamer` and `caps.libcamera` are `True`.
       Uses a `libcamerasrc` NV12 pipeline optimised for Raspberry Pi.
    2. **V4L2 device nodes** — one profile per readable `/dev/videoN`, using `CAP_V4L2`.
    3. **OpenCV default** — `CAP_ANY` on source `0` as a final fallback.

=== "Windows"

    1. DirectShow (`CAP_DSHOW`) on source `0`.
    2. OpenCV default (`CAP_ANY`) on source `0`.

=== "macOS"

    1. AVFoundation (`CAP_AVFOUNDATION`) on source `0`.
    2. OpenCV default (`CAP_ANY`) on source `0`.

### Using in Tests

Pass explicit `env` and `caps` arguments to avoid touching the real environment or hardware:

```python
from earsys.camera.profile import resolve_camera_profiles, SystemCapabilities
from pathlib import Path

# Simulate a Raspberry Pi with GStreamer + libcamera
profiles = resolve_camera_profiles(
    env={},
    caps=SystemCapabilities(
        gstreamer=True,
        libcamera=True,
        v4l2_devices=(Path("/dev/video0"),),
    ),
)
assert profiles[0].label == "libcamera GStreamer"
```

## Color Format Inference

When a GStreamer pipeline string is used, the color format is inferred from the pipeline string automatically:

| Pipeline substring | Inferred `color_format` |
|--------------------|------------------------|
| `format=nv12` | `"nv12"` |
| `format=rgb` | `"rgb"` |
| anything else | `"bgr"` |

The explicit `EARSYS_CAMERA_COLOR_FORMAT` setting always overrides inference.
