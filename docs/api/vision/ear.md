# EAR Calculation (`earsys.vision.ear`)

Pure functions for calculating EAR (Eye Aspect Ratio). These functions are stateless, isolated operations making unit testing robust and flexible across CPU/hardware iterations.

## Functions

### `euclidean_distance(p1: tuple, p2: tuple) -> float`
Calculates the spatial distance between two generic 2D points.

### `calculate_ear(eye_points: Sequence) -> float`
Core algorithm. Computes the specific Eye Aspect Ratio formula.
*Formula:*
`EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`

Where `eye_points` ordered as `[p1, p2, p3, p4, p5, p6]`.
Returns 0.0 if numerical calculation cannot determine a horizontal line.

### `average_ear(left_ear: float, right_ear: float) -> float`
Helper function returning the combined average across both eyes. Used by detector stream values directly.

### `get_eye_points(landmarks, indices, width, height) -> list[Point2D]`
Transforms MediaPipe’s internal normalized layout floating values `(x: 0...1, y: 0...1)` into literal integer resolution coordinates matched to the currently mapped recording bounds.
