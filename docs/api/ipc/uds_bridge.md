# UDS Bridge (`earsys.ipc.uds_bridge`)

This module owns the binary serialization of detection results and their delivery over a Unix Domain Socket.

## EyeFrame Protocol

Every detection event is packed into a **24-byte** fixed-size datagram and sent as a `SOCK_DGRAM` (connectionless) message. The receiver never needs to maintain a connection — it just binds to the address and reads.

### Packet Layout

Format string: little-endian `<4sBBHfIQ`

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 B | `char[4]` | `magic` | Always `b"SEYE"`. Lets the receiver sanity-check the frame. |
| 4 | 1 B | `uint8` | `version` | Protocol version. Currently `1`. |
| 5 | 1 B | `uint8` | `status` | `0` = AWAKE · `1` = DROWSY · `2` = NO_FACE |
| 6 | 2 B | `uint16` | `reserved` | Always `0`. Reserved for future use. |
| 8 | 4 B | `float32` | `eye_score` | `0.0` (fully open) to `1.0` (fully closed) |
| 12 | 4 B | `uint32` | `seq` | Monotonically increasing sequence number per bridge instance. |
| 16 | 8 B | `uint64` | `ts_ms` | Unix epoch in milliseconds at time of send. |

### `eye_score` Derivation

`eye_score` is a clamped linear interpolation of the raw EAR value between the configured open and closed thresholds:

```
eye_score = clamp(
    (ear_open_thr - EAR) / (ear_open_thr - ear_closed_thr),
    0.0,
    1.0
)
```

Defaults: `ear_open_thr = 0.30`, `ear_closed_thr = 0.15`.

An EAR of `0.30` or above maps to `0.0`; an EAR of `0.15` or below maps to `1.0`. Values between interpolate linearly.

### Receiving Frames

```python
import socket
import struct

FRAME_FORMAT = "<4sBBHfIQ"

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(b"\x00earsys/eye")   # \x00 prefix = Linux abstract namespace

while True:
    data, _ = sock.recvfrom(64)
    magic, version, status, reserved, eye_score, seq, ts_ms = struct.unpack(FRAME_FORMAT, data)
    print(f"status={status} eye_score={eye_score:.3f} seq={seq}")
```

For a filesystem socket instead of the abstract namespace:

```python
sock.bind("/tmp/my-consumer.sock")
# and set EARSYS_UDS_ADDR=path:/tmp/my-consumer.sock on the sender side
```

---

## `UdsBridge`

Synchronous bridge. Constructs a `SOCK_DGRAM` socket on init and sends each frame with `sendto`.

```python
with UdsBridge() as bridge:
    bridge.send(status=STATUS_AWAKE, ear=0.32)
```

### Error Handling

| Condition | Behaviour |
|-----------|-----------|
| Receiver not yet running (`ECONNREFUSED`, `ENOENT`) | Drop silently; emit a rate-limited warning at most once every 5 seconds. Recovers automatically when the receiver appears. |
| Any other `OSError` | Log a `WARNING` with the errno and continue. |
| Socket creation fails | `self._sock` is set to `None`; subsequent `send()` calls are no-ops. |

The bridge intentionally never raises from `send()` — a missing consumer must not crash the detection loop.

---

## `UdsAsyncBridge`

Wraps `UdsBridge` in a daemon worker thread with a bounded `queue.Queue`.

```python
with UdsAsyncBridge() as bridge:
    bridge.send(status=STATUS_AWAKE, ear=0.32)  # returns immediately
```

`send()` enqueues a `(status, ear)` tuple and returns without blocking. The worker thread dequeues items and calls `UdsBridge.send()`. If the queue fills up (default max: 512 items) incoming items are dropped and a warning is logged every 16 drops.

On context manager exit, `close()` signals the stop event, joins the worker thread with a 2-second timeout, and closes the underlying socket.

`UdsAsyncBridge` is what the detection loop uses so that socket I/O never appears on the hot path.
