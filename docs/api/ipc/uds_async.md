# UDS Async (`earsys.ipc.uds_async`)

Provides an asynchronous abstraction layer utilizing Unix Domain Sockets to dispatch frames without blocking the MediaPipe frame loop.

## `UdsAsyncBridge`
An interface designed to launch a dedicated background threading loop handling outgoing serialization via Python's native `socket` interface.

### Responsibilities
- Starts and orchestrates connection mapping towards `abstract:earsys/eye` or generic `path:` configurations.
- Serializes Python datatypes using exactly 24-bytes `EyeFrame` binary struct formats.
- Ensures no main loop pausing occurs by relying on a Queue-driven approach. 
