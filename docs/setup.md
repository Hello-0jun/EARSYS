# Development Setup

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | `3.12.x` — 3.13 is not yet supported |
| [uv](https://docs.astral.sh/uv/) | latest |
| `face_landmarker.task` | MediaPipe float16 model |
| Camera | any device reachable by OpenCV |

## First-time Setup

```bash
# 1. Clone the repository
git clone https://github.com/WhiPaper/EARSYS.git
cd EARSYS

# 2. Create the virtual environment and install runtime dependencies
uv sync

# 3. Install development tools (ruff, pytest, build, scalene)
uv sync --group dev

# 4. Download the MediaPipe face landmarker model
wget -q \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -O face_landmarker.task
```

The model file must live at the project root or at the path pointed to by `EARSYS_MODEL_PATH`.

## Running in Development Mode

The `dev` environment enables the OpenCV visualisation window and verbose debug logging by default. These are controlled by `.env.dev` which is loaded automatically when `EARSYS_ENV=dev`.

```bash
# Start with landmark visualisation
uv run earsys run --env dev --visualize

# Override individual thresholds at the command line
uv run earsys run --env dev --ear-threshold 0.20 --closed-frames 15

# Print the currently resolved configuration and probed capabilities
uv run earsys config
```

## Dependency Groups

`pyproject.toml` uses PEP 735 dependency groups.

| Group | Contents | Install command |
|-------|----------|-----------------|
| *(default)* | mediapipe, opencv, numpy, pydantic-settings, typer, rich | `uv sync` |
| `dev` | ruff, pytest, build, scalene | `uv sync --group dev` |
| `docs` | mkdocstrings, zensical | `uv sync --group docs` |

## Running the Documentation Site Locally

```bash
uv sync --group docs
uv run zensical serve
```

## Linting and Formatting

Before every commit, run:

```bash
uv run ruff check --fix
uv run ruff format
```

The CI pipeline will reject any commit where either check fails.

## Running Tests

```bash
# Quick pass
uv run pytest -q

# With full output
uv run pytest -v

# Single module
uv run pytest tests/test_ear.py -v
```

Tests never touch real hardware. See the [Conventions](conventions.md) page for the mocking patterns required when writing new tests.
