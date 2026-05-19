"""
Configuration constants and environment variable handling.

All tunable values are managed in this module and can be overridden at runtime
through environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

UnixSocketAddress = Union[str, bytes]


def _resolve_model_path() -> Path:
    """
    Return the absolute path to the face_landmarker.task model file.

    Priority:
      1. EARSYS_MODEL_PATH environment variable
      2. Auto-discovery relative to the running file

    Nuitka --onefile behavior:
      - When built with ``--include-data-files=face_landmarker.task=face_landmarker.task``,
        the onefile runtime extracts data files to the /tmp/onefile_XXX/ root.
      - In that case, ``__file__`` points to /tmp/onefile_XXX/earsys/config.pyc.
      - ``Path(__file__).parent.parent`` = /tmp/onefile_XXX/ -> the same location as the model.

    When running from source:
      - ``__file__`` = <project>/earsys/config.py
      - ``Path(__file__).parent.parent`` = <project>/ -> the face_landmarker.task location.
    """
    env = os.getenv("EARSYS_MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "face_landmarker.task"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH: Path = _resolve_model_path()

# ---------------------------------------------------------------------------
# UDS socket
# ---------------------------------------------------------------------------
def _resolve_uds_addr() -> UnixSocketAddress:
    """
    Resolve the destination Unix domain socket address.

    Supported values for EARSYS_UDS_ADDR:
      - abstract:<name>  Linux abstract namespace socket
      - path:<path>      Filesystem socket path
      - <path>           Filesystem socket path
    """
    raw = os.getenv("EARSYS_UDS_ADDR", "abstract:earsys/eye")
    if raw.startswith("abstract:"):
        return b"\x00" + raw.removeprefix("abstract:").encode()
    if raw.startswith("path:"):
        return raw.removeprefix("path:")
    return raw


UDS_EYE_ADDR: UnixSocketAddress = _resolve_uds_addr()

# EyeFrame protocol
EYE_FRAME_MAGIC   = b"SEYE"
EYE_FRAME_VERSION = 1
EYE_FRAME_FORMAT  = "<4sBBHfIQ"   # 24 bytes
EYE_FRAME_SIZE    = 24

# EAR -> eye_score conversion thresholds
EAR_OPEN_THR   = 0.30   # EAR >= this value -> score = 0.0
EAR_CLOSED_THR = 0.15   # EAR <= this value -> score = 1.0

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_SOURCE: str = os.getenv("EARSYS_CAMERA_SOURCE", "auto")
CAMERA_BACKEND: str = os.getenv("EARSYS_CAMERA_BACKEND", "auto").lower()
CAMERA_COLOR_FORMAT: str = os.getenv("EARSYS_CAMERA_COLOR_FORMAT", "auto").lower()
GST_PIPELINE: str | None = os.getenv("EARSYS_GST_PIPELINE")

# ---------------------------------------------------------------------------
# EAR / drowsiness detection
# ---------------------------------------------------------------------------
# Eye landmark indices (MediaPipe Face Landmarker 468-point model)
LEFT_EYE_INDICES: list[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES: list[int] = [362, 385, 387, 263, 373, 380]

# EAR threshold for closed-eye detection (smaller value means stricter detection)
EAR_THRESHOLD: float = float(os.getenv("EARSYS_EAR_THRESHOLD", "0.23"))

# Number of consecutive frames with closed eyes required to trigger drowsiness
CLOSED_FRAMES_THRESHOLD: int = int(os.getenv("EARSYS_CLOSED_FRAMES", "20"))

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
STATUS_AWAKE = 0
STATUS_DROWSY = 1
STATUS_NO_FACE = 2
