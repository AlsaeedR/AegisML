"""
AegisML Testing Agent — Telemetry Event Bus.

A minimal, in-memory, thread-safe pub/sub used to stream real-time
container telemetry (CPU, memory, log lines) from sandbox_runner.py
(running the Docker container) to api.py (serving a Server-Sent
Events endpoint to the frontend).

Each audit gets its own queue, keyed by audit_id. sandbox_runner.py
pushes events into the queue as the container runs; api.py's SSE
endpoint pulls events out and forwards them to the connected client.

This is intentionally simple (module-level dict + threading.Lock) to
match the project's current in-memory AUDIT_SESSIONS pattern. For a
production/multi-worker deployment this should be replaced with a
shared backend such as Redis Pub/Sub.
"""

import queue
import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_streams: Dict[str, "queue.Queue"] = {}

# Sentinel event pushed when a container run has finished (successfully,
# with an error, or skipped) so the SSE endpoint knows to close the stream.
DONE_EVENT = {"event": "done"}


def create_stream(audit_id: str) -> "queue.Queue":
    """Creates (or resets) the telemetry queue for a given audit_id."""
    with _lock:
        q: "queue.Queue" = queue.Queue()
        _streams[audit_id] = q
        return q


def get_stream(audit_id: str) -> Optional["queue.Queue"]:
    with _lock:
        return _streams.get(audit_id)


def publish(audit_id: Optional[str], event: Dict[str, Any]) -> None:
    """
    Pushes a telemetry event onto the audit's queue. Silently does nothing
    if audit_id is None or no stream was registered - this keeps
    sandbox_runner.py safe to call even when triggered outside the HITL
    API flow (e.g. from main.py's direct CLI execution).
    """
    if not audit_id:
        return
    with _lock:
        q = _streams.get(audit_id)
    if q is not None:
        q.put(event)


def close_stream(audit_id: Optional[str]) -> None:
    """Publishes the DONE sentinel and removes the queue from the registry."""
    if not audit_id:
        return
    publish(audit_id, dict(DONE_EVENT))
    with _lock:
        _streams.pop(audit_id, None)
