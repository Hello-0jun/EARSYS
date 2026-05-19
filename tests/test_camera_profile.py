"""Unit tests for system-aware camera profile selection."""

from __future__ import annotations

from pathlib import Path

import pytest

import earsys.camera.profile as camera_profile
from earsys.camera.profile import (
    CameraProfile,
    SystemCapabilities,
    probe_system_capabilities,
    resolve_camera_profiles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# Convenience: fully-disabled capabilities (no hardware at all)
_NO_CAPS = SystemCapabilities(gstreamer=False, libcamera=False, v4l2_devices=())

# Convenience: only GStreamer+libcamera, no V4L2
_LIBCAM_CAPS = SystemCapabilities(gstreamer=True, libcamera=True, v4l2_devices=())


# ---------------------------------------------------------------------------
# Explicit environment override tests  (caps param is ignored for these)
# ---------------------------------------------------------------------------


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
    assert_profile(
        profiles[0],
        source="videotestsrc ! video/x-raw,format=NV12 ! appsink",
        backend="gstreamer",
        color_format="nv12",
    )


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


# ---------------------------------------------------------------------------
# Desktop OS default backend tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Linux capability-injection tests  (no monkeypatching of probes needed)
# ---------------------------------------------------------------------------


def test_linux_auto_uses_existing_v4l2_devices_before_default(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    caps = SystemCapabilities(
        gstreamer=False,
        libcamera=False,
        v4l2_devices=(Path("/dev/video1"), Path("/dev/video3")),
    )
    profiles = resolve_camera_profiles({}, caps=caps)

    assert [p.source for p in profiles] == ["/dev/video1", "/dev/video3", 0]
    assert [p.backend for p in profiles] == ["v4l2", "v4l2", "auto"]


def test_linux_auto_prefers_libcamera_when_gstreamer_and_libcamera_available(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    profiles = resolve_camera_profiles({}, caps=_LIBCAM_CAPS)

    assert profiles[0].backend == "gstreamer"
    assert profiles[0].color_format == "nv12"
    assert "libcamerasrc" in profiles[0].source


def test_linux_auto_falls_back_when_libcamera_unavailable(monkeypatch):
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    caps = SystemCapabilities(gstreamer=True, libcamera=False, v4l2_devices=())
    profiles = resolve_camera_profiles({}, caps=caps)

    assert_profile(profiles[0], source=0, backend="auto")


def test_linux_no_caps_yields_only_default(monkeypatch):
    """With no detected capabilities the only profile is the OpenCV generic fallback."""
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    profiles = resolve_camera_profiles({}, caps=_NO_CAPS)

    assert len(profiles) == 1
    assert_profile(profiles[0], source=0, backend="auto")


def test_linux_libcamera_and_v4l2_combined(monkeypatch):
    """libcamera profile should come before V4L2 profiles."""
    monkeypatch.setattr(camera_profile.platform, "system", lambda: "Linux")

    caps = SystemCapabilities(
        gstreamer=True,
        libcamera=True,
        v4l2_devices=(Path("/dev/video0"),),
    )
    profiles = resolve_camera_profiles({}, caps=caps)

    assert profiles[0].backend == "gstreamer"
    assert "libcamerasrc" in profiles[0].source
    assert profiles[1].backend == "v4l2"
    assert profiles[1].source == "/dev/video0"
    assert profiles[2].backend == "auto"


# ---------------------------------------------------------------------------
# GStreamer pipeline color-format inference
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Capability probe unit tests
# ---------------------------------------------------------------------------


class TestLibcameraAvailable:
    def test_returns_true_when_binary_on_path(self, monkeypatch):
        monkeypatch.setattr(camera_profile.shutil, "which", lambda _: "/usr/bin/libcamera-vid")
        assert camera_profile._libcamera_available() is True

    def test_returns_false_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(camera_profile.shutil, "which", lambda _: None)
        assert camera_profile._libcamera_available() is False


class TestV4l2Devices:
    def test_returns_existing_readable_devices(self, tmp_path, monkeypatch):
        # Simulate /dev/video0 and /dev/video2 existing and readable
        dev0 = tmp_path / "video0"
        dev2 = tmp_path / "video2"
        dev0.touch()
        dev2.touch()

        def fake_path(s: str) -> Path:
            name = s.split("/")[-1]  # e.g. "video0"
            candidate = tmp_path / name
            return candidate if candidate.exists() else Path(s)

        original_path = camera_profile.Path

        def patched_path(s):  # type: ignore[no-untyped-def]
            if isinstance(s, str) and s.startswith("/dev/video"):
                return fake_path(s)
            return original_path(s)

        monkeypatch.setattr(camera_profile, "Path", patched_path)
        monkeypatch.setattr(camera_profile.os, "access", lambda p, _: p in (dev0, dev2))

        devices = camera_profile._v4l2_devices()
        assert set(devices) == {dev0, dev2}

    def test_excludes_unreadable_devices(self, tmp_path, monkeypatch):
        dev0 = tmp_path / "video0"
        dev0.touch()
        original_path = camera_profile.Path

        def patched_path(s):  # type: ignore[no-untyped-def]
            if isinstance(s, str) and s.startswith("/dev/video"):
                return dev0 if s.endswith("0") else original_path(s)
            return original_path(s)

        monkeypatch.setattr(camera_profile, "Path", patched_path)
        monkeypatch.setattr(camera_profile.os, "access", lambda _p, _m: False)

        devices = camera_profile._v4l2_devices()
        assert devices == []


class TestSystemCapabilities:
    def test_has_v4l2_true_when_devices_present(self):
        caps = SystemCapabilities(v4l2_devices=(Path("/dev/video0"),))
        assert caps.has_v4l2 is True

    def test_has_v4l2_false_when_no_devices(self):
        caps = SystemCapabilities()
        assert caps.has_v4l2 is False

    def test_probe_system_capabilities_returns_instance(self, monkeypatch):
        monkeypatch.setattr(camera_profile, "_opencv_gstreamer_available", lambda: True)
        monkeypatch.setattr(camera_profile, "_libcamera_available", lambda: False)
        monkeypatch.setattr(camera_profile, "_v4l2_devices", lambda: [Path("/dev/video0")])

        caps = probe_system_capabilities()

        assert isinstance(caps, SystemCapabilities)
        assert caps.gstreamer is True
        assert caps.libcamera is False
        assert caps.v4l2_devices == (Path("/dev/video0"),)
