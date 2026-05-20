# Conventions

This page documents the project-wide conventions that all contributors and AI coding assistants should follow.

## Python Version

EARSYS targets **Python 3.12** exclusively (`requires-python = ">=3.12,<3.13"`). Use 3.12+ features where they improve clarity:

- `X | Y` union types instead of `Optional[X]` or `Union[X, Y]`
- Walrus operator `:=` for combined assignment and condition checks
- `from __future__ import annotations` at the top of every module (deferred evaluation)

## Type Hints

All public functions and methods must carry full type annotations. The `ANN` Ruff rule set is active and will fail CI if annotations are missing.

```python
# Correct
def calculate_ear(eye_points: Sequence[Point2D]) -> float: ...

# Wrong — ANN001 / ANN201 violation
def calculate_ear(eye_points): ...
```

Exceptions: `tests/` and `earsys/cli.py` have `ANN` relaxed via `per-file-ignores`.

## Configuration — No Hardcoding

All tunable values must come from the `settings` singleton in `earsys.config`. Hardcoding thresholds or addresses is a code smell here because the same binary runs in both development and production with different parameters.

```python
# Correct
from earsys.config import settings

if ear < settings.ear_threshold:
    ...

# Wrong
if ear < 0.23:   # hardcoded — do not do this
    ...
```

## Linting and Formatting

Run the following before every commit:

```bash
uv run ruff check --fix
uv run ruff format
```

The active rule sets are:

| Code | Description |
|------|-------------|
| `E`, `W` | pycodestyle errors and warnings |
| `F` | Pyflakes |
| `I` | isort — import ordering |
| `B` | flake8-bugbear |
| `UP` | pyupgrade — enforces modern syntax |
| `C4` | flake8-comprehensions |
| `SIM` | flake8-simplify |
| `ANN` | type annotation enforcement |
| `BLE` | disallows bare `except Exception` without re-raise or logging |
| `PTH` | enforces `pathlib` over `os.path` string manipulation |
| `RUF` | Ruff-specific rules |

`ANN401` (allowing `Any`) is globally ignored. Max line length is **120 characters**.

## Import Order

```python
from __future__ import annotations  # always first

# stdlib
import logging
import time

# third-party
import cv2
import numpy as np

# internal
from earsys.config import settings
```

isort enforces this automatically; `earsys` is declared as `known-first-party`.

## Resource Lifecycle — Context Managers

Hardware resources (`OpenCvCamera`, `FaceDetector`, `UdsBridge`, `UdsAsyncBridge`) must always be used through `with` blocks. Never call `.open()` or `.close()` manually in production code.

```python
with FaceDetector() as detector, UdsAsyncBridge() as bridge:
    with OpenCvCamera() as camera:
        run_detection(camera, detector, bridge)
```

## Error Handling

- **Detection loop**: Use `except Exception` with `# noqa: BLE001` to keep the loop alive through transient frame errors. Always log the exception.
- **UDS delivery**: `ECONNREFUSED` and `ENOENT` are expected when the consumer is not yet running. Drop silently and emit a rate-limited warning (no more than once every 5 seconds).
- **Camera failure**: Raise `RuntimeError` and let the CLI reconnect loop in `cli.py` handle retries.

## Logging

`RichHandler` is active, so Rich markup is valid inside log format strings.

```python
logger = logging.getLogger(__name__)  # one logger per module

logger.warning(
    "[bold red]Drowsiness detected![/bold red] "
    "EAR=[yellow]%.3f[/yellow], frames=[red]%d[/red]",
    ear,
    state.closed_frames,
)
```

Use the log levels consistently:

| Level | When to use |
|-------|-------------|
| `DEBUG` | Per-frame detail, only visible with `feature_debug_logging` |
| `INFO` | Successful initialisation, state transitions |
| `WARNING` | Recoverable anomalies (UDS drop, camera retry) |
| `ERROR` | Conditions that will or may stop the loop |

## Testing

### Hardware Independence

Real hardware must never be touched in unit tests. Mock `cv2.VideoCapture` and `socket.socket` using `monkeypatch` and inject `SystemCapabilities` snapshots directly into `resolve_camera_profiles`.

```python
# Inject capabilities without touching /dev/videoN
profiles = resolve_camera_profiles(
    env={},
    caps=SystemCapabilities(gstreamer=True, libcamera=True),
)

# Intercept socket calls
monkeypatch.setattr(uds_bridge_module.socket, "socket", lambda *_: FakeSocket())
```

### Test File Responsibilities

| File | Covers |
|------|--------|
| `test_ear.py` | `calculate_ear`, `average_ear`, `get_eye_points` — pure functions |
| `test_camera.py` | `OpenCvCamera` open/fallback/context manager |
| `test_camera_profile.py` | `resolve_camera_profiles` across platforms and env overrides |
| `test_uds_bridge.py` | EyeFrame serialisation, sequence numbers, error handling |
| `test_visual_opencv.py` | `show_debug_frame` — skipped unless `EARSYS_RUN_VISUAL_TESTS=1` |

### CLI Lazy Import Pattern

`cli.py` applies env-var overrides into `os.environ` before importing `earsys.config`. This is necessary because `pydantic-settings` reads environment variables at class definition time. When adding new CLI options, follow the same pattern:

```python
@app.command()
def run(...) -> None:
    import os
    if some_option is not None:
        os.environ["EARSYS_SOME_SETTING"] = str(some_option)

    # Late import — settings now sees the override above
    from earsys.config import settings
```

## Adding a New Hardware Probe

If you need to gate behaviour on a new system capability:

1. Add a private probe function in `earsys/camera/profile.py` that accepts no parameters and touches only the system (e.g. `shutil.which`, `Path.exists`).
2. Add a field to `SystemCapabilities` with a clear docstring.
3. Wire it into `probe_system_capabilities()`.
4. Expose it as a `@property` on `AppSettings` in `earsys/config.py` following the pattern of `feature_gstreamer`.
5. Write a unit test that exercises the new probe through `SystemCapabilities(new_flag=True/False)` injection.
