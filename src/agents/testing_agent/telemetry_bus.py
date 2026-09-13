import queue
import threading
from typing import Any, Dict, Optional
_lock = threading.Lock()
_streams: Dict[str, 'queue.Queue'] = {}
DONE_EVENT = {'event': 'done'}

def create_stream(audit_id: str) -> 'queue.Queue':
    with _lock:
        q: 'queue.Queue' = queue.Queue()
        _streams[audit_id] = q
        return q

def get_stream(audit_id: str) -> Optional['queue.Queue']:
    with _lock:
        return _streams.get(audit_id)

def publish(audit_id: Optional[str], event: Dict[str, Any]) -> None:
    if not audit_id:
        return
    with _lock:
        q = _streams.get(audit_id)
    if q is not None:
        q.put(event)

def close_stream(audit_id: Optional[str]) -> None:
    if not audit_id:
        return
    publish(audit_id, dict(DONE_EVENT))
    with _lock:
        _streams.pop(audit_id, None)
