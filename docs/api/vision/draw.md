# Drawing Utilities (`earsys.vision.draw`)

Dev-only visualization helpers that render MediaPipe landmarks, EAR values, and drowsiness status onto an OpenCV window. This module is only active when `feature_visualize_landmarks` is `true` and is never imported in production code paths.

## `show_debug_frame()`

```python
def show_debug_frame(
    frame: np.ndarray,
    face_landmarks_list: list,
    ear: float,
    status: int,
    closed_frames: int,
) -> bool:
```

Renders a single annotated frame into an OpenCV named window and polls for a quit keypress.

### Arguments

| Parameter | Description |
|-----------|-------------|
| `frame` | Raw BGR frame from `OpenCvCamera`. A copy is made internally; the original is not mutated. |
| `face_landmarks_list` | The `face_landmarks` list from a MediaPipe `FaceLandmarkerResult`. May be empty. |
| `ear` | Current EAR value to display. |
| `status` | One of `STATUS_AWAKE`, `STATUS_DROWSY`, or `STATUS_NO_FACE`. |
| `closed_frames` | Current consecutive closed-eye frame count for the counter overlay. |

### Return Value

Returns `True` if the user pressed **ESC** (key code `27`) or **Ctrl+C** (key code `3`) in the window, signalling the detection loop to stop. Returns `False` otherwise.

### Rendered Overlays

When a face is detected (`face_landmarks_list` is non-empty):

- **Eye landmark dots** — six green circles per eye drawn at the pixel coordinates of the MediaPipe indices defined in `LEFT_EYE_INDICES` and `RIGHT_EYE_INDICES`.
- **EAR value** — cyan text at position `(30, 50)`.
- **DROWSINESS ALERT** — red bold text at `(30, 100)` when status is `DROWSY`.

Regardless of detection:

- **STATUS** — white text at `(30, 150)` showing `AWAKE`, `DROWSY`, or `NO FACE`.
- **CLOSED FRAMES** — white text at `(30, 200)`.

The window is named `"Drowsiness Detection System"` and is created automatically by `cv2.imshow` on first call.

## `draw_eye_points()`

```python
def draw_eye_points(frame: np.ndarray, points: list[Point2D]) -> None:
```

Draws a 2-pixel filled green circle for each point in `points` directly onto `frame`. Used internally by `show_debug_frame`.
