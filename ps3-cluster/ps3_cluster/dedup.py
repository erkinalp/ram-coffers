"""Bounded request-id dedup for retried batches.

A caller that loses its answer cannot tell whether the coordinator died before
or after driving its downstream work. Retrying is therefore *at-least-once
execution* in general — unless the retry reaches the same coordinator process,
which is exactly what an ordered endpoint list makes likely: a head server
listening on two addresses, or a reconnect after the socket dropped, is the same
process with the same memory.

This cache turns that case back into exactly-once execution. A batch carries a
64-bit request id (``REQ_FLAG_REQUEST_ID``); the coordinator runs the batch under
that id and remembers the encoded response frame, so a retry of the same logical
batch gets the first attempt's bytes instead of a second fan-out. A retry that
arrives while the first attempt is still running waits for it rather than racing
it. ALF's work queues behave the same way about identity: a work block is
enqueued once and its result is fetched by handle, not recomputed per poll (ALF
Programmer's Guide, SDK 3.0).

Two hard bounds keep this from becoming unbounded per-token state on a console
farm's head server:

* ``max_entries`` — at most this many response frames are retained, evicted
  oldest-completed-first. Worst-case memory is ``max_entries`` times the largest
  response frame, which ``MAX_BATCH_ENTRIES``/``MAX_FRAME_BYTES`` already bound.
* ``ttl`` — an entry older than this is dropped on the next touch; a retry that
  arrives later re-executes (and is then at-least-once again, honestly).

A retry that lands on a *different* replica process cannot be deduplicated
without shared state, which this deliberately does not introduce: such a retry
is at-least-once execution, and exactly-once *reduction* is guaranteed by the
caller instead (it accepts one complete answer and drops the rest).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

#: Retained response frames per coordinator.
DEFAULT_DEDUP_ENTRIES = 128

#: How long a completed response stays replayable, in seconds.
DEFAULT_DEDUP_TTL = 60.0


class _Slot:
    """One logical batch: in flight, then its response frame."""

    __slots__ = ("done", "frame", "error", "started", "finished")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.frame: Optional[bytes] = None
        self.error: Optional[BaseException] = None
        self.started = time.monotonic()
        self.finished: Optional[float] = None


class DedupCache:
    """Runs a batch at most once per request id, within explicit bounds."""

    def __init__(self, max_entries: int = DEFAULT_DEDUP_ENTRIES,
                 ttl: float = DEFAULT_DEDUP_TTL) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        self.max_entries = max_entries
        self.ttl = ttl
        self._lock = threading.Lock()
        self._slots: Dict[int, _Slot] = {}
        #: Observability, asserted by the tests.
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expiries = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._slots)

    def run(self, request_id: Optional[int], compute: Callable[[], bytes],
            timeout: Optional[float] = None) -> bytes:
        """Return the response frame for ``request_id``, computing it once.

        ``request_id`` of ``None`` (a caller that did not name its batch) always
        computes: there is nothing to deduplicate on.
        """
        if request_id is None:
            return compute()
        with self._lock:
            self._expire_locked()
            slot = self._slots.get(request_id)
            if slot is not None:
                self.hits += 1
                mine = False
            else:
                self.misses += 1
                self._evict_locked()
                slot = self._slots[request_id] = _Slot()
                mine = True
        if not mine:
            return self._await(request_id, slot, timeout)
        try:
            frame = compute()
        except BaseException as exc:  # noqa: BLE001 - handed to every waiter
            slot.error = exc
            slot.finished = time.monotonic()
            slot.done.set()
            with self._lock:
                self._slots.pop(request_id, None)
            raise
        slot.frame = frame
        slot.finished = time.monotonic()
        slot.done.set()
        return frame

    def replay(self, request_id: int) -> Optional[bytes]:
        """The remembered frame for ``request_id``, if it is still cached."""
        with self._lock:
            self._expire_locked()
            slot = self._slots.get(request_id)
        return None if slot is None or not slot.done.is_set() else slot.frame

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()

    # -- internals ---------------------------------------------------------
    def _await(self, request_id: int, slot: _Slot,
               timeout: Optional[float]) -> bytes:
        """Wait for the attempt already running under this id."""
        if not slot.done.wait(timeout):
            raise TimeoutError(f"request {request_id} is still running "
                               f"upstream of this retry")
        if slot.error is not None:
            raise slot.error
        assert slot.frame is not None
        return slot.frame

    def _expire_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl
        stale = [rid for rid, slot in self._slots.items()
                 if slot.finished is not None and slot.finished < cutoff]
        for rid in stale:
            del self._slots[rid]
            self.expiries += 1

    def _evict_locked(self) -> None:
        while len(self._slots) >= self.max_entries:
            completed = [(slot.finished, rid)
                         for rid, slot in self._slots.items()
                         if slot.finished is not None]
            if completed:
                victim = min(completed)[1]
            else:
                # Every slot is still running: drop the oldest, whose caller
                # will simply not be able to replay it.
                victim = min((slot.started, rid)
                             for rid, slot in self._slots.items())[1]
            del self._slots[victim]
            self.evictions += 1
