"""
Detection loop and dev-visualization helpers.

This module is the single source of truth for the main frame-processing loop.
``earsys.cli`` imports ``run_detection`` from here; ``main.py`` is a thin shim.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from earsys.camera.capture import OpenCvCamera
from earsys.config import (
    STATUS_AWAKE,
    STATUS_DROWSY,
    STATUS_NO_FACE,
    settings,
)
from earsys.ipc.uds_async import UdsAsyncBridge as UdsBridge
from earsys.vision.detector import FaceDetector
from earsys.vision.draw import show_debug_frame

logger = logging.getLogger(__name__)





# ---------------------------------------------------------------------------
# Detection state
# ---------------------------------------------------------------------------


@dataclass
class DetectionStats:
    frames_total: int = 0
    sent_awake: int = 0
    sent_drowsy: int = 0
    sent_no_face: int = 0
    frame_errors: int = 0
    sent_fused_scores: int = 0


@dataclass
class DrowsinessState:
    closed_frames: int = 0
    drowsy_logged: bool = False
    previous_status: int | None = None

    def reset_eye_closure(self) -> None:
        self.closed_frames = 0
        self.drowsy_logged = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_rgb_frame(frame: np.ndarray, color_format: str) -> np.ndarray:
    """Convert an OpenCV frame to RGB according to the configured input format."""
    if color_format == "rgb":
        return frame
    if color_format == "nv12":
        return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _status_from_ear(ear: float, state: DrowsinessState) -> int:
    if ear < settings.ear_threshold:
        state.closed_frames += 1
    else:
        state.reset_eye_closure()

    if state.closed_frames < settings.closed_frames_threshold:
        return STATUS_AWAKE

    if not state.drowsy_logged:
        state.drowsy_logged = True
        logger.warning(
            "[bold red blink]Drowsiness detected![/bold red blink] "
            "EAR=[yellow]%.3f[/yellow], consecutive frames=[red]%d[/red]",
            ear,
            state.closed_frames,
        )
    return STATUS_DROWSY


def _record_sent_status(stats: DetectionStats, status: int) -> None:
    stats.sent_fused_scores += 1
    if status == STATUS_DROWSY:
        stats.sent_drowsy += 1
    elif status == STATUS_AWAKE:
        stats.sent_awake += 1
    elif status == STATUS_NO_FACE:
        stats.sent_no_face += 1


def _build_dashboard(
    ear: float,
    status: int,
    state: DrowsinessState,
    stats: DetectionStats,
    uptime_sec: int,
) -> Panel:
    status_str = "AWAKE"
    status_color = "bold green"
    if status == STATUS_DROWSY:
        status_str = "DROWSY"
        status_color = "bold red blink"
    elif status == STATUS_NO_FACE:
        status_str = "NO FACE"
        status_color = "dim"

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_column("Key2", style="bold cyan")
    table.add_column("Value2")

    table.add_row("Status", f"[{status_color}]{status_str}[/]", "Uptime", f"{uptime_sec}s")
    table.add_row("EAR", f"[yellow]{ear:.3f}[/]", "Frames", str(stats.frames_total))
    table.add_row("Closed Frames", f"[red]{state.closed_frames}[/]", "Fused Sent", str(stats.sent_fused_scores))
    table.add_row(
        "Errors",
        f"[red]{stats.frame_errors}[/]",
        "UDS Packets",
        f"Awake: {stats.sent_awake} | Drowsy: {stats.sent_drowsy}",
    )

    return Panel(table, title="[bold]EARSYS Live Dashboard[/bold]", border_style="bright_blue")


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------


def run_detection(camera: OpenCvCamera, detector: FaceDetector, bridge: UdsBridge | None, console=None) -> bool:
    """
    Main detection loop.

    Returns:
        True  = user initiated shutdown (KeyboardInterrupt or 'q' in dev window)
        False = loop stopped because the camera stream ended or failed
    """
    visualize = settings.feature_visualize_landmarks

    state = DrowsinessState()
    stats = DetectionStats()
    start_monotonic: float = time.monotonic()

    logger.info(
        "[bold cyan]Detection loop started[/bold cyan] — env=[bold green]%s[/bold green] "
        "EAR threshold=[yellow]%.2f[/yellow], consecutive frames=[red]%d[/red], "
        "visualize=[magenta]%s[/magenta]",
        settings.env,
        settings.ear_threshold,
        settings.closed_frames_threshold,
        visualize,
    )

    try:
        last_dashboard_update = 0.0
        dashboard_interval = 0.25  # 4 FPS

        ear: float = 0.0
        status: int = STATUS_NO_FACE
        face_landmarks_list: list = []

        with Live(
            _build_dashboard(ear, status, state, stats, 0),
            console=console,
            refresh_per_second=4,
        ) as live:
            for bgr_frame in camera.frames(flip=True):
                stats.frames_total += 1

                try:
                    height, width = bgr_frame.shape[:2]
                    rgb_frame = _to_rgb_frame(bgr_frame, camera.color_format)

                    result = detector.detect(rgb_frame)

                    if result is not None:
                        face_landmarks_list, ear = result

                        if face_landmarks_list:
                            status = _status_from_ear(ear, state)

                            if bridge is not None:
                                bridge.send(status=status, ear=ear)
                            _record_sent_status(stats, status)
                            state.previous_status = status

                        else:
                            state.reset_eye_closure()
                            status = STATUS_NO_FACE

                            # Send over UDS only when the state changes.
                            if status != state.previous_status:
                                if bridge is not None:
                                    bridge.send(status=status, ear=0.0)
                                _record_sent_status(stats, status)
                                state.previous_status = status

                except Exception as e:  # noqa: BLE001 — keep the detection loop alive
                    stats.frame_errors += 1
                    logger.error("[bold red]Error while processing frame:[/bold red] %s", e)

                # Dev visualization — runs only when feature flag is enabled
                if visualize:
                    quit_requested = show_debug_frame(bgr_frame, face_landmarks_list, ear, status, state.closed_frames)
                    if quit_requested:
                        logger.info("[bold blue]Dev window closed by user ('q' pressed).[/bold blue]")
                        cv2.destroyAllWindows()
                        return True

                now_mono = time.monotonic()
                if now_mono - last_dashboard_update >= dashboard_interval:
                    uptime_sec = int(now_mono - start_monotonic)
                    live.update(_build_dashboard(ear, status, state, stats, uptime_sec))
                    last_dashboard_update = now_mono

    except KeyboardInterrupt:
        logger.info("[bold yellow]KeyboardInterrupt:[/bold yellow] stopping detection loop.")
        if visualize:
            cv2.destroyAllWindows()
        return True

    logger.error("[bold red]Camera frame stream ended,[/bold red] stopping detection loop.")
    return False
