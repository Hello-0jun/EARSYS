# Camera Capture (`earsys.camera.capture`)

Handles OpenCV frame abstractions, backend definitions, and capture generators.

## `OpenCvCamera` class
Responsible for reading streams from local USB/V4L2 cameras, virtual loops, or remote streams via GStreamer.

### Features
1. Abstract `grab` looping mechanism over standard OpenCV behavior.
2. Built-in reconnection logic based on `settings.camera_max_retries` avoiding fatal runtime panics.
3. Formats BGR frame iterations properly.

### Iteration
Can be looped simply through `camera.frames()` which acts as a Python Generator yielding standard image buffers.
