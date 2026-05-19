"""System-aware camera profile selection."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
    """Camera-related environment settings."""

    source: str | None
    backend: str | None
    color_format: str
    gstreamer_pipeline: str | None
    width: int
    height: int
    fps: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "CameraSettings":
        return cls(
            source=_normalized_env(env, "EARSYS_CAMERA_SOURCE"),
            backend=_normalized_env(env, "EARSYS_CAMERA_BACKEND"),
            color_format=env.get("EARSYS_CAMERA_COLOR_FORMAT", "auto").lower(),
            gstreamer_pipeline=_normalized_env(env, "EARSYS_GST_PIPELINE"),
            width=_int_env(env, "EARSYS_CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH),
            height=_int_env(env, "EARSYS_CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT),
            fps=_int_env(env, "EARSYS_CAMERA_FPS", DEFAULT_CAMERA_FPS),
        )

    @property
    def has_explicit_source(self) -> bool:
        return self.source is not None

    @property
    def has_explicit_backend(self) -> bool:
        return self.backend is not None


def _normalized_env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if not value or value.lower() == "auto":
        return None
    return value


def _parse_source(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if not value:
        return default
    return int(value)


def _infer_color_format(source: str, explicit_format: str = "auto") -> str:
    if explicit_format != "auto":
        return explicit_format
    lowered = source.lower()
    if "format=nv12" in lowered:
        return "nv12"
    if "format=rgb" in lowered:
        return "rgb"
    return "bgr"


def _is_raspberry_pi() -> bool:
    model_path = Path("/proc/device-tree/model")
    try:
        return "raspberry pi" in model_path.read_text(errors="ignore").lower()
    except OSError:
        return False


def _opencv_gstreamer_available() -> bool:
    try:
        import cv2
    except ImportError:
        return False
    return "gstreamer" in cv2.getBuildInformation().lower()


def _default_libcamera_pipeline(width: int, height: int, fps: int) -> str:
    return (
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},format=NV12,framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _camera_profile(
    source: int | str,
    backend: str,
    color_format: str,
    width: int | None,
    height: int | None,
    fps: int | None,
    label: str,
) -> CameraProfile:
    return CameraProfile(
        source=source,
        backend=backend,
        color_format=color_format,
        width=width,
        height=height,
        fps=fps,
        label=label,
    )


def _explicit_profiles(settings: CameraSettings) -> list[CameraProfile] | None:
    if settings.gstreamer_pipeline:
        return [
            _camera_profile(
                settings.gstreamer_pipeline,
                "gstreamer",
                _infer_color_format(settings.gstreamer_pipeline, settings.color_format),
                None,
                None,
                None,
                "environment GStreamer pipeline",
            )
        ]

    if not settings.has_explicit_source and not settings.has_explicit_backend:
        return None

    source = _parse_source(settings.source) if settings.source else 0
    backend = (settings.backend or "auto").lower()
    color_format = _infer_color_format(str(source), settings.color_format)
    return [
        _camera_profile(
            source,
            backend,
            color_format,
            settings.width,
            settings.height,
            settings.fps,
            "environment camera source",
        )
    ]


def _linux_profiles(settings: CameraSettings) -> list[CameraProfile]:
    profiles: list[CameraProfile] = []

    if _is_raspberry_pi() and _opencv_gstreamer_available() and shutil.which("libcamera-vid"):
        profiles.append(
            _camera_profile(
                _default_libcamera_pipeline(settings.width, settings.height, settings.fps),
                "gstreamer",
                "nv12",
                None,
                None,
                None,
                "Raspberry Pi libcamera",
            )
        )

    for index in range(LINUX_VIDEO_DEVICE_COUNT):
        device = Path(f"/dev/video{index}")
        if device.exists():
            profiles.append(
                _camera_profile(
                    str(device),
                    "v4l2",
                    "bgr",
                    settings.width,
                    settings.height,
                    settings.fps,
                    f"Linux V4L2 {device}",
                )
            )

    profiles.append(
        _camera_profile(
            0,
            "auto",
            "bgr",
            settings.width,
            settings.height,
            settings.fps,
            "Linux OpenCV default",
        )
    )
    return profiles


def _single_default_profile(settings: CameraSettings, backend: str, label: str) -> CameraProfile:
    return _camera_profile(0, backend, "bgr", settings.width, settings.height, settings.fps, label)


def resolve_camera_profiles(env: Mapping[str, str] | None = None) -> list[CameraProfile]:
    """
    Return camera profiles ordered by expected suitability for this system.

    Explicit environment settings are treated as an operator decision and are
    tried before any auto-detected fallback.
    """
    settings = CameraSettings.from_env(os.environ if env is None else env)
    explicit_profiles = _explicit_profiles(settings)
    if explicit_profiles is not None:
        return explicit_profiles

    system = platform.system().lower()
    if system == "linux":
        return _linux_profiles(settings)
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
