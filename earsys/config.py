"""
Configuration management via pydantic-settings.

Settings are loaded from (in priority order, highest first):
  1. Environment variables  (e.g.  EARSYS_ENV=prod python main.py)
  2. .env.<env> file        (e.g.  .env.dev  or  .env.prod)

Usage:
    from earsys.config import settings

    path = settings.model_path
    if settings.feature_visualize_landmarks:
        ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
UnixSocketAddress = str | bytes

# ---------------------------------------------------------------------------
# EyeFrame protocol constants (not tunable — kept as module-level constants)
# ---------------------------------------------------------------------------
EYE_FRAME_MAGIC = b"SEYE"
EYE_FRAME_VERSION = 1
EYE_FRAME_FORMAT = "<4sBBHfIQ"  # 24 bytes
EYE_FRAME_SIZE = 24

# Eye landmark indices (MediaPipe Face Landmarker 468-point model)
LEFT_EYE_INDICES: list[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES: list[int] = [362, 385, 387, 263, 373, 380]

# Status codes
STATUS_AWAKE = 0
STATUS_DROWSY = 1
STATUS_NO_FACE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_env_file() -> list[Path]:
    """Return the .env.<env> path based on EARSYS_ENV (read eagerly before model init)."""
    env = os.getenv("EARSYS_ENV", "prod")
    project_root = Path(__file__).parent.parent
    env_file = project_root / f".env.{env}"
    return [env_file] if env_file.exists() else []


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """
    All tunable EARSYS application settings.

    Environment variable prefix: ``EARSYS_``

    Typical usage::

        EARSYS_ENV=dev python main.py
        EARSYS_ENV=prod python main.py
    """

    model_config = SettingsConfigDict(
        env_prefix="EARSYS_",
        env_file=_default_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Environment ---------------------------------------------------------
    env: Literal["dev", "prod"] = Field(default="prod", description="Runtime environment.")

    # ---- Logging -------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Python logging level.")

    # ---- Model ---------------------------------------------------------------
    model_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "face_landmarker.task",
        description="Absolute path to face_landmarker.task.",
    )

    # ---- UDS socket ----------------------------------------------------------
    uds_addr: str = Field(
        default="abstract:earsys/eye",
        description=(
            "Unix domain socket address. "
            "Prefix 'abstract:' for Linux abstract namespace, "
            "'path:' or bare path for filesystem socket."
        ),
    )

    # ---- Camera --------------------------------------------------------------
    camera_source: str = Field(default="auto", description="OpenCV camera index/path/URL.")
    camera_backend: str = Field(
        default="auto",
        description="OpenCV backend: auto, gstreamer, v4l2, directshow, avfoundation.",
    )
    camera_color_format: str = Field(
        default="auto",
        description="Input frame color format: auto, bgr, rgb, nv12.",
    )
    gst_pipeline: str | None = Field(default=None, description="Optional GStreamer pipeline string.")

    # ---- EAR / drowsiness detection ------------------------------------------
    ear_threshold: float = Field(default=0.23, description="EAR threshold for closed-eye detection.")
    closed_frames_threshold: int = Field(
        default=20,
        description="Consecutive closed-eye frames to trigger drowsiness.",
    )

    # ---- Camera reconnect ----------------------------------------------------
    camera_reopen_sec: float = Field(default=2.0, description="Seconds to wait before reopening camera.")
    camera_max_retries: int = Field(default=0, description="Max camera reopen attempts (0 = unlimited).")

    # ---- EAR → eye_score thresholds ------------------------------------------
    ear_open_thr: float = Field(default=0.30, description="EAR >= this → eye_score = 0.0 (open).")
    ear_closed_thr: float = Field(default=0.15, description="EAR <= this → eye_score = 1.0 (closed).")

    # ---- Feature flags -------------------------------------------------------
    feature_visualize_landmarks: bool = Field(
        default=False,
        description="Show OpenCV window with face landmarks and EAR overlay (dev only).",
    )
    feature_debug_logging: bool = Field(
        default=False,
        description="Enable verbose per-frame debug logging.",
    )

    # ---- Capability feature flags (runtime-probed, read-only) ----------------
    #
    # These are NOT environment-variable-tunable. They are derived by probing the
    # current system at import time via :mod:`earsys.camera.profile`.
    # Downstream code can gate behaviour on these without knowing which probe
    # detected the capability.

    @property
    def feature_gstreamer(self) -> bool:
        """True when OpenCV was compiled with GStreamer support (runtime probe)."""
        from earsys.camera.profile import _opencv_gstreamer_available

        return _opencv_gstreamer_available()

    @property
    def feature_libcamera(self) -> bool:
        """True when libcamera-vid is present on PATH (runtime probe)."""
        from earsys.camera.profile import _libcamera_available

        return _libcamera_available()

    @property
    def feature_v4l2(self) -> bool:
        """True when at least one readable /dev/videoN device exists (runtime probe)."""
        from earsys.camera.profile import _v4l2_devices

        return bool(_v4l2_devices())

    # ---- Validators ----------------------------------------------------------

    @field_validator("camera_backend", "camera_color_format", mode="before")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()

    # ---- Derived helpers -----------------------------------------------------

    @property
    def uds_socket_addr(self) -> UnixSocketAddress:
        """
        Resolve the UDS address into a value usable by the socket layer.

        Returns bytes for abstract namespace sockets (Linux),
        or str for filesystem path sockets.
        """
        raw = self.uds_addr
        if raw.startswith("abstract:"):
            return b"\x00" + raw.removeprefix("abstract:").encode()
        if raw.startswith("path:"):
            return raw.removeprefix("path:")
        return raw

    @property
    def is_dev(self) -> bool:
        """Return True when running in the development environment."""
        return self.env == "dev"


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = AppSettings()
