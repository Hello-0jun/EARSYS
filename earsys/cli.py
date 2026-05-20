"""
EARSYS CLI — Typer + Rich based command-line interface.

Commands:
    earsys run      Start the main detection loop
    earsys config   Print the current resolved configuration
    earsys version  Print version information
"""

from __future__ import annotations

import logging
import sys
import time
from importlib.metadata import version as pkg_version
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="earsys",
    help="EAR-based drowsiness detection system.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _setup_logging(log_level: str) -> None:
    """Configure root logger with RichHandler."""
    logging.basicConfig(
        level=log_level.upper(),
        format="[dim]%(name)s[/dim] \u2502 %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=True,
                markup=True,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

_BANNER_WIDTH = 52


def _print_start_banner(env: str, ear_thr: float, closed_frames: int, visualize: bool, uds: bool) -> None:
    lines = Text()
    lines.append("  env            ", style="dim")
    lines.append(f"{env}\n", style="bold cyan" if env == "dev" else "bold green")
    lines.append("  ear-threshold  ", style="dim")
    lines.append(f"{ear_thr:.2f}\n", style="yellow")
    lines.append("  closed-frames  ", style="dim")
    lines.append(f"{closed_frames}\n", style="yellow")
    lines.append("  visualize      ", style="dim")
    lines.append("on\n" if visualize else "off\n", style="green" if visualize else "red")
    lines.append("  uds            ", style="dim")
    lines.append("on\n" if uds else "off\n", style="green" if uds else "red")
    console.print(
        Panel(
            lines,
            title="[bold]EARSYS[/bold] starting",
            border_style="bright_blue",
            width=_BANNER_WIDTH,
        )
    )


def _print_stop_banner(reason: str) -> None:
    console.print(
        Panel(
            f"[dim]{reason}[/dim]",
            title="[bold]EARSYS[/bold] stopped",
            border_style="dim",
            width=_BANNER_WIDTH,
        )
    )


# ---------------------------------------------------------------------------
# Camera reconnect wait
# ---------------------------------------------------------------------------


def _camera_wait(seconds: float) -> None:
    """Sleep for `seconds` with a Rich progress bar."""
    with Progress(
        TextColumn("[dim]Camera reconnect in"),
        BarColumn(bar_width=24),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("wait", total=int(seconds * 10))
        for _ in range(int(seconds * 10)):
            time.sleep(0.1)
            progress.advance(task)


# ---------------------------------------------------------------------------
# run sub-command
# ---------------------------------------------------------------------------


def _run_with_retries(detector, bridge) -> tuple[int, str]:
    from earsys.camera.capture import OpenCvCamera
    from earsys.config import settings
    from earsys.loop import run_detection

    reopen_attempt = 0
    while True:
        try:
            with OpenCvCamera() as camera:
                user_stopped = run_detection(camera, detector, bridge, console=console)
                if user_stopped:
                    return 0, "Stopped by user request."
        except RuntimeError as exc:
            logging.getLogger("earsys.cli").error("Camera open/runtime error: %s", exc)

        reopen_attempt += 1
        if settings.camera_max_retries > 0 and reopen_attempt > settings.camera_max_retries:
            return 1, f"Exceeded camera retry limit ({settings.camera_max_retries})."

        wait = settings.camera_reopen_sec
        logging.getLogger("earsys.cli").warning("Retrying camera (%d) in %.1f s…", reopen_attempt, wait)
        _camera_wait(wait)


@app.command()
def run(
    env: Annotated[
        str | None,
        typer.Option("--env", help="Runtime environment: [cyan]dev[/cyan] | [green]prod[/green]", show_default=True),
    ] = None,
    ear_threshold: Annotated[
        float | None,
        typer.Option("--ear-threshold", help="EAR threshold for closed-eye detection.", show_default=True),
    ] = None,
    closed_frames: Annotated[
        int | None,
        typer.Option("--closed-frames", help="Consecutive closed-eye frames to trigger drowsiness.", show_default=True),
    ] = None,
    visualize: Annotated[
        bool | None,
        typer.Option("--visualize/--no-visualize", help="Show landmark visualisation window (dev)."),
    ] = None,
    uds: Annotated[
        bool | None,
        typer.Option("--uds/--no-uds", help="Enable UDS socket output."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG | INFO | WARNING | ERROR.", show_default=True),
    ] = None,
    camera_source: Annotated[
        str | None,
        typer.Option("--camera-source", help="OpenCV camera index, path, or URL.", show_default=True),
    ] = None,
) -> None:
    """[bold]Start[/bold] the EAR drowsiness detection loop."""
    import os

    # Apply CLI overrides as env vars *before* importing settings
    if env is not None:
        os.environ["EARSYS_ENV"] = env
    if ear_threshold is not None:
        os.environ["EARSYS_EAR_THRESHOLD"] = str(ear_threshold)
    if closed_frames is not None:
        os.environ["EARSYS_CLOSED_FRAMES_THRESHOLD"] = str(closed_frames)
    if visualize is not None:
        os.environ["EARSYS_FEATURE_VISUALIZE_LANDMARKS"] = "true" if visualize else "false"
    if uds is not None:
        os.environ["EARSYS_FEATURE_UDS_ENABLED"] = "true" if uds else "false"
    if log_level is not None:
        os.environ["EARSYS_LOG_LEVEL"] = log_level.upper()
    if camera_source is not None:
        os.environ["EARSYS_CAMERA_SOURCE"] = camera_source

    # Late imports so env overrides are applied before pydantic-settings reads them
    from contextlib import nullcontext

    from earsys.config import settings
    from earsys.ipc.uds_async import UdsAsyncBridge as UdsBridge
    from earsys.vision.detector import FaceDetector

    _setup_logging(settings.log_level)
    _print_start_banner(
        env=settings.env,
        ear_thr=settings.ear_threshold,
        closed_frames=settings.closed_frames_threshold,
        visualize=settings.feature_visualize_landmarks,
        uds=settings.feature_uds_enabled,
    )

    exit_code = 0
    stop_reason = "Stopped normally."

    try:
        bridge_ctx = UdsBridge() if settings.feature_uds_enabled else nullcontext()
        with bridge_ctx as bridge, FaceDetector() as detector:
            exit_code, stop_reason = _run_with_retries(detector, bridge)

    except FileNotFoundError as exc:
        err_console.print(f"[bold red]Model file not found:[/bold red] {exc}")
        exit_code = 1
        stop_reason = "Model file not found."
    except RuntimeError as exc:
        err_console.print(f"[bold red]Camera error:[/bold red] {exc}")
        exit_code = 1
        stop_reason = "Camera error."
    except Exception as exc:  # noqa: BLE001
        err_console.print_exception()
        exit_code = 1
        stop_reason = f"Unexpected error: {exc}"

    _print_stop_banner(stop_reason)
    raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# config sub-command
# ---------------------------------------------------------------------------


@app.command()
def config() -> None:
    """Show the [bold]current[/bold] resolved configuration."""
    from earsys.config import settings

    table = Table(
        title="EARSYS Configuration",
        box=box.ROUNDED,
        border_style="bright_blue",
        show_lines=True,
        expand=False,
    )
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Description", style="dim")

    rows: list[tuple[str, str, str]] = [
        ("env", settings.env, "Runtime environment"),
        ("log_level", settings.log_level, "Python logging level"),
        ("model_path", str(settings.model_path), "Face landmarker model"),
        ("uds_addr", settings.uds_addr, "Unix domain socket"),
        ("camera_source", settings.camera_source, "Camera source"),
        ("camera_backend", settings.camera_backend, "OpenCV backend"),
        ("camera_color_format", settings.camera_color_format, "Frame color format"),
        ("gst_pipeline", str(settings.gst_pipeline or "—"), "GStreamer pipeline"),
        ("ear_threshold", f"{settings.ear_threshold:.3f}", "EAR closed-eye threshold"),
        ("closed_frames_threshold", str(settings.closed_frames_threshold), "Consecutive frames"),
        ("camera_reopen_sec", f"{settings.camera_reopen_sec:.1f} s", "Camera reconnect wait"),
        ("camera_max_retries", str(settings.camera_max_retries), "Max retries (0=∞)"),
        ("ear_open_thr", f"{settings.ear_open_thr:.2f}", "EAR → eye_score=0"),
        ("ear_closed_thr", f"{settings.ear_closed_thr:.2f}", "EAR → eye_score=1"),
        ("feature.visualize_landmarks", str(settings.feature_visualize_landmarks), "Landmark window"),
        ("feature.uds_enabled", str(settings.feature_uds_enabled), "UDS socket output"),
        ("feature.debug_logging", str(settings.feature_debug_logging), "Verbose frame log"),
    ]

    for key, val, desc in rows:
        value_style = "green" if val in {"True", "dev"} else ("red" if val == "False" else "white")
        table.add_row(key, Text(val, style=value_style), desc)

    console.print()
    console.print(table)

    # ---- Capability feature flags ----------------------------------------
    from earsys.camera.profile import probe_system_capabilities

    caps = probe_system_capabilities()
    cap_table = Table(
        title="System Capabilities",
        box=box.ROUNDED,
        border_style="bright_magenta",
        show_lines=True,
        expand=False,
    )
    cap_table.add_column("Capability", style="bold magenta", no_wrap=True)
    cap_table.add_column("Detected", style="white")
    cap_table.add_column("Notes", style="dim")

    def _cap_row(name: str, detected: bool, notes: str) -> tuple[str, Text, str]:
        icon = Text("✔  yes" if detected else "✘  no", style="green" if detected else "red")
        return (name, icon, notes)

    cap_rows = [
        _cap_row("gstreamer", caps.gstreamer, "OpenCV built with GStreamer support"),
        _cap_row("libcamera", caps.libcamera, "libcamera-vid found on PATH"),
        _cap_row("v4l2", caps.has_v4l2, f"devices: {', '.join(str(d) for d in caps.v4l2_devices) or '—'}"),
    ]
    for name, icon, notes in cap_rows:
        cap_table.add_row(name, icon, notes)

    console.print(cap_table)
    console.print()


# ---------------------------------------------------------------------------
# version sub-command
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the [bold]version[/bold] and exit."""
    try:
        ver = pkg_version("earsys")
    except Exception:  # noqa: BLE001
        ver = "unknown"

    console.print(
        Panel(
            f"[bold white]earsys[/bold white]  [bold cyan]{ver}[/bold cyan]",
            border_style="bright_blue",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Typer app entrypoint (called from pyproject.toml script)."""
    app()


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
