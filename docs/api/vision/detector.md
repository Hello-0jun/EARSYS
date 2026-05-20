# Face Detector (`earsys.vision.detector`)

Provides the MediaPipe Face Landmarker wrapper class `FaceDetector`.

## `FaceDetector` class

Initializes in `LIVE_STREAM` mode and manages a monotonically increasing timestamp internally.

### Context Manager Support
Supports the context manager protocol (`with FaceDetector(...) as detector:`) to ensure safe cleanup of the MediaPipe API upon exit.

### Methods

#### `__init__(self, model_path: Path | None = None, num_faces: int = 1)`
Instantiates the detector. It requires `face_landmarker.task` file. Ensures system constraints and bootstraps an asynchronous MediaPipe callback mapping detected elements to the local `_result_queue`.

#### `detect(self, rgb_frame: np.ndarray) -> tuple[list, float] | None`
Detects face landmarks and calculates EAR from an RGB NumPy array frame.
- **`rgb_frame`**: `HxWx3 uint8` NumPy array in RGB format.
- Returns a tuple `(face_landmarks_list, ear)` if a new result is ready.
- Returns `None` if the queue is empty (inference still running).

#### `close(self)`
Manual execution function to release the MediaPipe landmarker and clear buffers.
