# Deployment

## systemd Service (Linux / Raspberry Pi)

The repository ships a ready-made service unit at `scripts/earsys.service`.

### Installation

```bash
# 1. Copy the service unit
sudo cp scripts/earsys.service /etc/systemd/system/earsys.service

# 2. Create an optional environment override file
sudo mkdir -p /etc/earsys
sudo tee /etc/earsys/earsys.env <<'EOF'
EARSYS_ENV=prod
EARSYS_EAR_THRESHOLD=0.23
EARSYS_CLOSED_FRAMES_THRESHOLD=20
EOF

# 3. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable earsys
sudo systemctl start earsys
```

### Monitoring

```bash
sudo systemctl status earsys
sudo journalctl -u earsys -f
```

### Service Behaviour

The unit is configured for resilient production use:

- **`Restart=on-failure`** — restarts automatically on abnormal exit.
- **`RestartSec=5s`** — waits 5 seconds between restart attempts.
- **`StartLimitBurst=5`** within a **`StartLimitInterval=600s`** window — halts after five rapid failures in ten minutes to prevent thrashing.
- **`PrivateTmp=true`** and **`NoNewPrivileges=true`** — basic systemd hardening.

The `EnvironmentFile` directive uses the `-` prefix, so the service starts normally even when `/etc/earsys/earsys.env` does not exist.

## Wheel Distribution

EARSYS is packaged as a pure-Python wheel. The `face_landmarker.task` model file is **not bundled** — it must be deployed separately.

### Building Locally

```bash
uv sync --group dev
python -m build
# Produces:
#   dist/earsys-0.3.0-py3-none-any.whl
#   dist/earsys-0.3.0.tar.gz
```

### Installing on a Target Device

```bash
pip install earsys-0.3.0-py3-none-any.whl

# Download the model separately
wget -q \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Point EARSYS at it if not in the working directory
export EARSYS_MODEL_PATH=/opt/earsys/face_landmarker.task
```

## GitHub Releases (Automated)

Pushing a `v*` tag triggers the **EARSYS Package & Release** workflow (`.github/workflows/package.yml`).

``` mermaid
flowchart TD
    TAG["git push tag v*\ne.g. v0.3.0"] --> BUILD

    subgraph BUILD["build job  (ubuntu-latest)"]
        B1["checkout"] --> B2["python -m build"]
        B2 --> B3["upload-artifact: dist/\n*.whl  +  *.tar.gz"]
    end

    BUILD --> RELEASE

    subgraph RELEASE["release job  (needs: build)"]
        R1["download-artifact: dist/"] --> R2["extract version tag"]
        R2 --> R3["softprops/action-gh-release\nattach *.whl + *.tar.gz"]
    end
```

```bash
git tag v0.3.0
git push origin v0.3.0
```

The workflow can also be triggered manually from the GitHub Actions UI with an explicit tag input.

## Environment Configuration

Production deployments should configure EARSYS entirely through environment variables or the `/etc/earsys/earsys.env` file. The key production settings are:

| Variable | Recommended value | Reason |
|----------|-------------------|--------|
| `EARSYS_ENV` | `prod` | Disables visualisation and verbose logging |
| `EARSYS_LOG_LEVEL` | `INFO` | Sufficient detail without per-frame noise |
| `EARSYS_FEATURE_VISUALIZE_LANDMARKS` | `false` | No display available on headless devices |
| `EARSYS_FEATURE_DEBUG_LOGGING` | `false` | Reduces I/O overhead |
| `EARSYS_CAMERA_MAX_RETRIES` | `0` | Unlimited reconnection — let systemd handle the restart budget |
| `EARSYS_UDS_ADDR` | `abstract:earsys/eye` | Default; change only if the consumer binds a different address |

See the [Configuration](api/config.md) reference for the full variable list.
