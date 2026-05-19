"""Unit tests for system-aware camera profile selection."""

from __future__ import annotations

from pathlib import Path

import pytest

import earsys.camera_profile as camera_profile
from earsys.camera_profile import CameraProfile, resolve_camera_profiles


def assert_profile(
    profile: CameraProfile,
    *,
    source,
    backend: str,
    color_format: str = "bgr",
) -> None:
    assert profile.source == source
    assert profile.backend == backend
    assert profile.color_format == color_format


def test_environment_gstreamer_pipeline_overrides_everything(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    profiles = resolve_camera_profiles(
        {
            "EARSYS_GST_PIPELINE": "videotestsrc ! video/x-raw,format=NV12 ! appsink",
            "EARSYS_CAMERA_SOURCE": "/dev/video0",
            "EARSYS_CAMERA_BACKEND": "v4l2",
        }
    )

    assert len(profiles) == 1
    assert_profile(profiles[0], source="videotestsrc ! video/x-raw,format=NV12 ! appsink", backend="gstreamer", color_format="nv12")


def test_environment_camera_source_overrides_auto_detection(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Windows")

    profiles = resolve_camera_profiles(
        {
            "EARSYS_CAMERA_SOURCE": "2",
            "EARSYS_CAMERA_BACKEND": "directshow",
            "EARSYS_CAMERA_COLOR_FORMAT": "rgb",
            "EARSYS_CAMERA_WIDTH": "1280",
            "EARSYS_CAMERA_HEIGHT": "720",
            "EARSYS_CAMERA_FPS": "60",
        }
    )

    assert len(profiles) == 1
    assert_profile(profiles[0], source=2, backend="directshow", color_format="rgb")
    assert profiles[0].width == 1280
    assert profiles[0].height == 720
    assert profiles[0].fps == 60


def test_environment_backend_can_override_default_source(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    profiles = resolve_camera_profiles({"EARSYS_CAMERA_BACKEND": "v4l2"})

    assert len(profiles) == 1
    assert_profile(profiles[0], source=0, backend="v4l2")


@pytest.mark.parametrize(
    ("system", "backend"),
    [
        ("Windows", "directshow"),
        ("Darwin", "avfoundation"),
    ],
)
def test_desktop_operating_systems_prefer_native_backend(monkeypatch, system, backend):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: system)

    profiles = resolve_camera_profiles({})

    assert_profile(profiles[0], source=0, backend=backend)
    assert_profile(profiles[1], source=0, backend="auto")


def test_linux_auto_uses_existing_v4l2_devices_before_default(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")
    monkeypatch.setattr(camera_profile, "_is_raspberry_pi", lambda: False)
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in {"/dev/video1", "/dev/video3"})

    profiles = resolve_camera_profiles({})

    assert [profile.source for profile in profiles] == ["/dev/video1", "/dev/video3", 0]
    assert [profile.backend for profile in profiles] == ["v4l2", "v4l2", "auto"]


def test_raspberry_pi_auto_prefers_libcamera_when_available(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")
    monkeypatch.setattr(camera_profile, "_is_raspberry_pi", lambda: True)
    monkeypatch.setattr(camera_profile, "_opencv_gstreamer_available", lambda: True)
    monkeypatch.setattr(camera_profile.shutil, "which", lambda command: "/usr/bin/libcamera-vid")
    monkeypatch.setattr(Path, "exists", lambda self: False)

    profiles = resolve_camera_profiles({})

    assert profiles[0].backend == "gstreamer"
    assert profiles[0].color_format == "nv12"
    assert "libcamerasrc" in profiles[0].source


def test_raspberry_pi_falls_back_when_libcamera_is_unavailable(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")
    monkeypatch.setattr(camera_profile, "_is_raspberry_pi", lambda: True)
    monkeypatch.setattr(camera_profile, "_opencv_gstreamer_available", lambda: True)
    monkeypatch.setattr(camera_profile.shutil, "which", lambda command: None)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    profiles = resolve_camera_profiles({})

    assert_profile(profiles[0], source=0, backend="auto")


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        ("src ! video/x-raw,format=NV12 ! appsink", "nv12"),
        ("src ! video/x-raw,format=RGB ! appsink", "rgb"),
        ("src ! appsink", "bgr"),
    ],
)
def test_gstreamer_pipeline_color_format_is_inferred(pipeline, expected):
    profiles = resolve_camera_profiles({"EARSYS_GST_PIPELINE": pipeline})

    assert profiles[0].color_format == expected
