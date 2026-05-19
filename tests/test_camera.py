"""Unit tests for the OpenCV camera wrapper."""

from __future__ import annotations

from typing import ClassVar

import pytest

import earsys.camera.capture as camera_module
from earsys.camera.capture import OpenCvCamera, _opencv_backend
from earsys.camera.profile import CameraProfile


class FakeCapture:
    opened_sources: ClassVar[set[int | str]] = set()
    calls: ClassVar[list[tuple[int | str, int]]] = []
    released: ClassVar[list[int | str]] = []
    properties: ClassVar[list[tuple[int | str, int, int]]] = []

    def __init__(self, source, backend):
        self.source = source
        self.backend = backend
        self.calls.append((source, backend))

    def isOpened(self):
        return self.source in self.opened_sources

    def set(self, prop, value):
        self.properties.append((self.source, prop, value))
        return True

    def read(self):
        return False, None

    def release(self):
        self.released.append(self.source)


@pytest.fixture(autouse=True)
def reset_fake_capture():
    FakeCapture.opened_sources = set()
    FakeCapture.calls = []
    FakeCapture.released = []
    FakeCapture.properties = []


@pytest.fixture()
def fake_video_capture(monkeypatch):
    monkeypatch.setattr("earsys.camera.capture.cv2.VideoCapture", FakeCapture)
    return FakeCapture


@pytest.mark.parametrize(
    ("backend", "source", "expected"),
    [
        ("auto", 0, camera_module.cv2.CAP_ANY),
        ("gstreamer", "pipeline ! appsink", camera_module.cv2.CAP_GSTREAMER),
        ("v4l2", "/dev/video0", camera_module.cv2.CAP_V4L2),
        ("directshow", 0, camera_module.cv2.CAP_DSHOW),
        ("avfoundation", 0, camera_module.cv2.CAP_AVFOUNDATION),
        ("auto", "videotestsrc ! appsink", camera_module.cv2.CAP_GSTREAMER),
    ],
)
def test_opencv_backend_resolution(backend, source, expected):
    assert _opencv_backend(backend, source) == expected


def test_open_falls_back_to_next_profile(fake_video_capture):
    fake_video_capture.opened_sources = {1}
    first = CameraProfile(source=0, backend="auto", label="first")
    second = CameraProfile(source=1, backend="auto", color_format="rgb", label="second")

    camera = OpenCvCamera(profiles=[first, second])
    camera.open()

    assert fake_video_capture.calls == [(0, 0), (1, 0)]
    assert fake_video_capture.released == [0]
    assert camera.color_format == "rgb"


def test_open_applies_requested_capture_properties(fake_video_capture):
    fake_video_capture.opened_sources = {0}
    profile = CameraProfile(source=0, backend="auto", width=1280, height=720, fps=60)

    OpenCvCamera(profile=profile).open()

    assert [value for _, _, value in fake_video_capture.properties] == [1280, 720, 60]


def test_open_reports_all_failed_profiles(fake_video_capture):
    profile = CameraProfile(source=0, backend="auto", label="unavailable")
    camera = OpenCvCamera(profile=profile)

    with pytest.raises(RuntimeError, match="unavailable"):
        camera.open()


def test_context_manager_releases_selected_capture(fake_video_capture):
    fake_video_capture.opened_sources = {0}
    profile = CameraProfile(source=0, backend="auto")

    with OpenCvCamera(profile=profile):
        pass

    assert fake_video_capture.released == [0]
