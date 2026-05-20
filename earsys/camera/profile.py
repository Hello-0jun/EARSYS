"""System-aware camera profile selection."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
LINUX_VIDEO_DEVICE_COUNT = 4


@dataclass(frozen=True)
class CameraProfile:
    """OpenCV camera configuration selected for the current system."""

    source: int | str
    backend: str
    color_format: str = "bgr"
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    label: str = "camera"


@dataclass(frozen=True)
class CameraSettings:
    """Camera-related environment settings parsed from the process environment."""

    source: str | None
    backend: str | None
    color_format: str
    gstreamer_pipeline: str | None
    width: int
    height: int
    fps: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> CameraSettings:
        return cls(
            source=_normalized_env(env, "EARSYS_CAMERA_SOURCE"),
            backend=_normalized_env(env, "EARSYS_CAMERA_BACKEND"),
            color_format=env.get("EARSYS_CAMERA_COLOR_FORMAT", "auto").lower(),
            gstreamer_pipeline=_normalized_env(env, "EARSYS_GST_PIPELINE"),
            width=_int_env(env, "EARSYS_CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH),
            height=_int_env(env, "EARSYS_CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT),
            fps=_int_env(env, "EARSYS_CAMERA_FPS", DEFAULT_CAMERA_FPS),
        )


# ---------------------------------------------------------------------------
# Environment parsing helpers
# ---------------------------------------------------------------------------


def _normalized_env(env: Mapping[str, str], name: str) -> str | None:
    """Return the env var value, or None when absent or set to 'auto'."""
    value = env.get(name)
    return None if not value or value.lower() == "auto" else value


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    return int(value) if (value := env.get(name)) else default


def _parse_source(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


# ---------------------------------------------------------------------------
# Color-format inference
# ---------------------------------------------------------------------------


def _infer_color_format(source: str, explicit_format: str = "auto") -> str:
    """Infer the frame color format from a pipeline/source string."""
    if explicit_format != "auto":
        return explicit_format
    lowered = source.lower()
    if "format=nv12" in lowered:
        return "nv12"
    if "format=rgb" in lowered:
        return "rgb"
    return "bgr"


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def _opencv_gstreamer_available() -> bool:
    """Return True when the installed OpenCV was built with GStreamer support."""
    try:
        import cv2
    except ImportError:
        return False
    return "gstreamer" in cv2.getBuildInformation().lower()


def _libcamera_available() -> bool:
    """Return True when libcamera-vid is present on PATH (proxy for libcamera stack)."""
    return shutil.which("libcamera-vid") is not None


def _v4l2_devices() -> list[Path]:
    """Return a list of /dev/videoN paths that exist and are readable."""
    return [
        device
        for index in range(LINUX_VIDEO_DEVICE_COUNT)
        if (device := Path(f"/dev/video{index}")).exists() and os.access(device, os.R_OK)
    ]


# ---------------------------------------------------------------------------
# System capability snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemCapabilities:
    """Snapshot of runtime-probed hardware/software capabilities.

    Use :func:`probe_system_capabilities` to obtain an instance.
    Each field corresponds to a single, independently-mockable probe so that
    unit tests can exercise any combination without touching real hardware.
    """

    gstreamer: bool = False
    """OpenCV was compiled with GStreamer support."""

    libcamera: bool = False
    """libcamera-vid is available on PATH (libcamera stack present)."""

    v4l2_devices: tuple[Path, ...] = field(default_factory=tuple)
    """V4L2 video device nodes that exist and are readable."""

    @property
    def has_v4l2(self) -> bool:
        """True when at least one V4L2 device is available."""
        return bool(self.v4l2_devices)


def probe_system_capabilities() -> SystemCapabilities:
    """Probe the current system and return a :class:`SystemCapabilities` snapshot.

    This is the single entry-point for all capability detection.  Call it once
    at startup and pass the result around rather than calling individual probes
    scattered across the codebase.
    """
    return SystemCapabilities(
        gstreamer=_opencv_gstreamer_available(),
        libcamera=_libcamera_available(),
        v4l2_devices=tuple(_v4l2_devices()),
    )


def _default_libcamera_pipeline(width: int, height: int, fps: int) -> str:
    return (
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},format=NV12,framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------


def _explicit_profiles(settings: CameraSettings) -> list[CameraProfile] | None:
    """Return profiles derived from explicit operator settings, or None for auto-detection."""
    if settings.gstreamer_pipeline:
        return [
            CameraProfile(
                source=settings.gstreamer_pipeline,
                backend="gstreamer",
                color_format=_infer_color_format(settings.gstreamer_pipeline, settings.color_format),
                label="environment GStreamer pipeline",
            )
        ]

    if settings.source is None and settings.backend is None:
        return None

    source = _parse_source(settings.source) if settings.source else 0
    backend = (settings.backend or "auto").lower()
    return [
        CameraProfile(
            source=source,
            backend=backend,
            color_format=_infer_color_format(str(source), settings.color_format),
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            label="environment camera source",
        )
    ]


def _linux_profiles(
    settings: CameraSettings,
    caps: SystemCapabilities,
) -> list[CameraProfile]:
    """Return camera profiles for Linux ordered by expected suitability.

    Priority (highest first):
      1. libcamera via GStreamer  — when GStreamer + libcamera stack both detected
      2. V4L2 device nodes        — for each readable /dev/videoN
      3. OpenCV auto              — generic last-resort fallback
    """
    profiles: list[CameraProfile] = []

    if caps.gstreamer and caps.libcamera:
        profiles.append(
            CameraProfile(
                source=_default_libcamera_pipeline(settings.width, settings.height, settings.fps),
                backend="gstreamer",
                color_format="nv12",
                label="libcamera GStreamer",
            )
        )

    profiles += [
        CameraProfile(
            source=str(device),
            backend="v4l2",
            color_format="bgr",
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            label=f"Linux V4L2 {device}",
        )
        for device in caps.v4l2_devices
    ]

    profiles.append(
        CameraProfile(
            source=0,
            backend="auto",
            color_format="bgr",
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            label="Linux OpenCV default",
        )
    )
    return profiles


def _single_default_profile(settings: CameraSettings, backend: str, label: str) -> CameraProfile:
    return CameraProfile(
        source=0,
        backend=backend,
        color_format="bgr",
        width=settings.width,
        height=settings.height,
        fps=settings.fps,
        label=label,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_camera_profiles(
    env: Mapping[str, str] | None = None,
    caps: SystemCapabilities | None = None,
) -> list[CameraProfile]:
    """
    Return camera profiles ordered by expected suitability for this system.

    Explicit operator settings (EARSYS_CAMERA_SOURCE, EARSYS_GST_PIPELINE, etc.)
    are always preferred over auto-detected fallbacks.  When no explicit settings
    are given, the system capabilities are probed (GStreamer, libcamera, V4L2
    devices, etc.) and a prioritised list is returned.

    Args:
        env:  Environment variable mapping (defaults to ``os.environ``).
        caps: Pre-computed :class:`SystemCapabilities` snapshot.  When *None*,
              :func:`probe_system_capabilities` is called automatically.
              Pass an explicit value in tests to avoid touching real hardware.
    """
    settings = CameraSettings.from_env(os.environ if env is None else env)
    explicit = _explicit_profiles(settings)
    if explicit is not None:
        return explicit

    system = platform.system().lower()
    if system == "linux":
        resolved_caps = caps if caps is not None else probe_system_capabilities()
        return _linux_profiles(settings, resolved_caps)
    if system == "windows":
        return [
            _single_default_profile(settings, "directshow", "Windows DirectShow default"),
            _single_default_profile(settings, "auto", "Windows OpenCV default"),
        ]
    if system == "darwin":
        return [
            _single_default_profile(settings, "avfoundation", "macOS AVFoundation default"),
            _single_default_profile(settings, "auto", "macOS OpenCV default"),
        ]
    return [_single_default_profile(settings, "auto", f"{system or 'unknown'} OpenCV default")]
