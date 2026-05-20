# Profiling with `py-spy`

Because EARSYS connects multiple high-frequency domains (Camera I/O, MediaPipe Inference, Socket IPC), identifying bottlenecks is crucial during development. We recommend using [`py-spy`](https://github.com/benfred/py-spy), a sampling profiler for Python that lets you visualize where the CPU is spending time without modifying the code.

## 1. Install `py-spy`
If not already installed globally:
```bash
uv pip install py-spy
```

## 2. Generate a Flame Graph
To record a profiling session and generate an interactive SVG flame graph:
```bash
sudo py-spy record -o profile.svg -- uv run earsys run
```
> **Note**: `sudo` might be explicitly required depending on your OS configuration to allow `py-spy` to read the process memory.

## 3. Live Process Top
To see a live view of functions taking the most CPU time:
```bash
sudo py-spy top -- uv run earsys run
```

Open `profile.svg` in any web browser to interactively explore and zoom into the call stack to identify whether MediaPipe models or OpenCV captures are blocking your detection thread.
