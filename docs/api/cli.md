# CLI (`earsys.cli`)

EARSYS exposes three sub-commands through a Typer + Rich interface. The entry point is registered as `earsys` in `pyproject.toml`.

```bash
uv run earsys [COMMAND] [OPTIONS]
```

Running `earsys` with no arguments prints the help screen.

---

## `earsys run`

Starts the drowsiness detection loop.

```bash
uv run earsys run [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | `dev` \| `prod` | from `EARSYS_ENV` | Runtime environment. `dev` loads `.env.dev`; `prod` loads `.env.prod`. |
| `--ear-threshold` | float | `0.23` | EAR value below which a frame counts as a closed eye. |
| `--closed-frames` | int | `20` | Number of consecutive closed-eye frames required to trigger `DROWSY`. |
| `--visualize` / `--no-visualize` | bool | `false` | Show an OpenCV window with landmark overlay and EAR value. |
| `--uds` / `--no-uds` | bool | `true` | Enable or disable UDS datagram output. |
| `--log-level` | str | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `--camera-source` | str | `auto` | Camera index (`0`), device path (`/dev/video0`), or stream URL. |

All flags are **optional**. When omitted the value is read from the active `.env` file or the environment. CLI flags always win over `.env` values because they are injected into `os.environ` before `pydantic-settings` initialises.

### Startup sequence

1. CLI overrides are written into `os.environ`.
2. `earsys.config.settings` is imported (picks up the overrides).
3. A start banner is printed via Rich.
4. `FaceDetector` and `UdsAsyncBridge` context managers are entered.
5. `_run_with_retries` opens `OpenCvCamera` and calls `run_detection`.
6. On camera failure, the loop waits `camera_reopen_sec` seconds (shown as a progress bar) and retries.
7. After `camera_max_retries` failures (0 = unlimited) the command exits with code `1`.

### Examples

```bash
# Development — visualisation on, verbose logging
uv run earsys run --env dev --visualize --log-level DEBUG

# Production — explicit thresholds
uv run earsys run --ear-threshold 0.20 --closed-frames 25

# Standalone inference without broadcasting to any consumer
uv run earsys run --no-uds

# Target a specific V4L2 device
uv run earsys run --camera-source /dev/video2
```

---

## `earsys config`

Prints the currently resolved configuration and probed system capabilities.

```bash
uv run earsys config
```

Output is split into two Rich tables:

- **EARSYS Configuration** — every `AppSettings` field with its resolved value and a short description.
- **System Capabilities** — runtime probes for GStreamer, libcamera, and V4L2 devices.

This command is useful for verifying that environment variable overrides are being picked up correctly and for diagnosing camera compatibility issues before the first run.

---

## `earsys version`

Prints the installed package version and exits.

```bash
uv run earsys version
```

The version is read from the package metadata via `importlib.metadata`. If the package is not installed in the active environment the output will show `unknown`.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Clean shutdown (user pressed `q`, ESC, or Ctrl+C) |
| `1` | Error — model file not found, camera retry limit exceeded, or unexpected exception |
