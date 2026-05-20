# Usage

EARSYS provides a Typer-based CLI interface to easily run the drowsiness detection system.

## Command Line Interface

You can run the application directly through `uv`, which isolates the environment and guarantees dependencies are correct.

```bash
# General usage
uv run earsys [COMMAND] [OPTIONS]
```

### Starting the Vision Loop

To start the drowsiness detection loop capturing from the camera:

```bash
uv run earsys run
```

#### Feature Flags & Run Options

The `run` command supports several useful feature flags and overrides that modify the application behavior completely at runtime without touching `.env` files:

*   `--visualize` / `--no-visualize` **(Feature Flag)**
    *   Enable or disable the OpenCV window showing real-time MediaPipe face landmarks and EAR progression. Ideal for visual debugging and threshold calibration.
*   `--uds` / `--no-uds` **(Feature Flag)**
    *   Enable or disable broadcasting eye state across the Unix Domain Socket. Can be explicitly turned off (`--no-uds`) if you just want to run local inference.
*   `--env [dev|prod]`
    *   Start in development or production mode. (Defaults generally read from `EARSYS_ENV`)
*   `--ear-threshold FLOAT` 
    *   Override the default numeric threshold (e.g. `0.23`) for determining an eye as "closed".
*   `--closed-frames INTEGER`
    *   Consecutive frames the EAR must remain below the threshold before drowsiness is triggered. 
*   `--log-level [DEBUG|INFO|WARNING|ERROR]`
    *   Adjust console log output verbosity.
*   `--camera-source TEXT`
    *   Target a specific hardware camera. Valid variables include an index (`0`, `1`), a raw device (`/dev/video0`), or a stream URL.

**Advanced Startup Example:**
```bash
uv run earsys run --env dev --visualize --no-uds --log-level DEBUG --camera-source 0
```

### Checking the Configuration

To print out the currently active configuration including dynamically applied environment overrides and probed system capabilities:

```bash
uv run earsys config
```
