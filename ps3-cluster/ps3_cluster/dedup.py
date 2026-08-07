"""Bounded request-id dedup for retried batches.

A caller that loses its answer cannot tell whether the coordinator died before
or after driving its downstream work. Retrying is therefore *at-least-once
execution* in general. A reconnect after a dropped socket reaches the same
coordinator process and can use the same in-memory cache; the same is true if a
single process programmatically listens on multiple addresses. Normal
``run_subcluster.py --standby N`` / ``run_region.py --standby N`` invocations are
separate processes with separate caches, so a retry that reaches them cannot be
deduplicated without shared state and is at-least-once execution.

This cache turns the same-process case back into exactly-once execution.

A batch carries a 64-bit request id (``REQ_FLAG_REQUEST_ID``); the coordinator runs the batch under
that id and remembers the encoded response frame, so a retry of the same logical
batch gets the first attempt's bytes instead of a second fan-out. A retry that
arrives while the first attempt is still running waits for it rather than racing
it. ALF's work queues behave the same way about identity: a work block is
enqueued once and its result is fetched by handle, not recomputed per poll (ALF
Programmer's Guide, SDK 3.0).

The cache also binds each id to a deterministic fingerprint of the logical
request. A reused id with a different activation, token, layer, deadline,
entries, or fast flag is a programming error and is rejected as a bad request
rather than replaying a stale response.

Three hard bounds keep this from becoming unbounded per-token state on a console
farm's head server:

* ``max_entries`` — at most this many logical batches are tracked (in flight or
  completed). When the limit is reached and every tracked batch is still
  running, new batches are rejected with ``ERR_DEDUP_CAPACITY`` so an in-flight
  batch can never be evicted before it completes; completed batches are evicted
  oldest-first when room is needed.
* ``max_bytes`` — completed response frames are counted precisely; the cache
  evicts completed least-recently-finished frames until a new response fits.
  A response larger than ``max_bytes`` is returned but not cached, so a later
  retry re-executes it. Worst-case retained memory is bounded by
  ``min(max_entries * max_response, max_bytes)``.
* ``ttl`` — a completed response older than this is dropped on the next touch; a
  retry that arrives later re-executes (and is then at-least-once again, honestly).

A retry that lands on a *different* replica process cannot be deduplicated
without shared state, which this deliberately does not introduce: such a retry
is at-least-once execution, and exactly-once *reduction* is guaranteed by the
caller instead (it accepts one complete answer and drops the rest).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional, Tuple

#: Retained response frames per coordinator.
DEFAULT_DEDUP_ENTRIES = 128

#: How long a completed response stays replayable, in seconds.
DEFAULT_DEDUP_TTL = 60.0

#: Total completed-response byte budget. 128 MiB is deliberately conservative:
#: at K3 width (7168, fp32) one row is 28 KB, so this is ~4700 cached rows at
#: full width, or several max-sized batch responses, before eviction.
DEFAULT_DEDUP_BYTES = 128 * 1024 * 1024


class _Slot:
    """One logical batch: in flight, then its response frame."""

    __slots__ = ("done", "frame", "error", "started", "finished",
                 "fingerprint", "cached", "frame_size")

    def __init__(self, fingerprint: bytes) -> None:
        self.done = threading.Event()
        self.frame: Optional[bytes] = None
        self.error: Optional[BaseException] = None
        self.started = time.monotonic()
        self.finished: Optional[float] = None
        self.fingerprint = fingerprint
        self.cached = True  # False for errors or oversized responses
        self.frame_size = 0


class MismatchedRequestError(ValueError):
    """Raised when the same request id is reused with different content."""


class DedupCapacityError(RuntimeError):
    """Raised when the dedup cache has no room for another in-flight batch."""


class DedupCache:
    """Runs a batch at most once per (request_id, fingerprint), within bounds."""

    def __init__(self, max_entries: int = DEFAULT_DEDUP_ENTRIES,
                 ttl: float = DEFAULT_DEDUP_TTL,
                 max_bytes: int = DEFAULT_DEDUP_BYTES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_entries = max_entries
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._slots: Dict[int, _Slot] = {}
        self._bytes = 0
        #: Observability, asserted by the tests.
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expiries = 0
        self.rejected = 0
        self.oversized = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._slots)

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def run(self, request_id: Optional[int], fingerprint: bytes,
            compute: Callable[[], bytes],
            timeout: Optional[float] = None) -> bytes:
        """Return the response frame for ``request_id``, computing it once.

        ``request_id`` of ``None`` (a caller that did not name its batch) always
        computes: there is nothing to deduplicate on. ``fingerprint`` is a
        stable identifier of the logical request content (e.g. a SHA-256
        digest). The same id with a different fingerprint is rejected as a bad
        request; the same request with the same fingerprint is a retry.
        """
        if request_id is None:
            return compute()
        with self._lock:
            self._expire_locked()
            existing = self._slots.get(request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise MismatchedRequestError(
                        f"request id {request_id} reused with different content")
                self.hits += 1
                mine = False
                slot = existing
            else:
                if not self._reserve_slot_locked():
                    self.rejected += 1
                    raise DedupCapacityError(
                        f"dedup capacity full ({self.max_entries} in flight)")
                self.misses += 1
                slot = self._slots[request_id] = _Slot(fingerprint)
                mine = True
        if not mine:
            return self._await(request_id, slot, timeout)
        try:
            frame = compute()
        except BaseException as exc:  # noqa: BLE001 - handed to every waiter
            slot.error = exc
            slot.finished = time.monotonic()
            slot.cached = False
            slot.done.set()
            with self._lock:
                self._slots.pop(request_id, None)
            raise
        self._finish(request_id, slot, frame)
        return frame

    def replay(self, request_id: int) -> Optional[bytes]:
        """The remembered frame for ``request_id``, if it is still cached."""
        with self._lock:
            self._expire_locked()
            slot = self._slots.get(request_id)
        if slot is None or not slot.done.is_set() or not slot.cached:
            return None
        return slot.frame

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()
            self._bytes = 0

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

    def _reserve_slot_locked(self) -> bool:
        """Make room for one new in-flight batch without evicting the running.

        Completed entries are evicted oldest-first. If every entry is in flight,
        return False so the caller can reject with backpressure.
        """
        while len(self._slots) >= self.max_entries:
            completed = [(slot.finished, rid)
                         for rid, slot in self._slots.items()
                         if slot.done.is_set() and slot.cached]
            if not completed:
                return False
            victim = min(completed)[1]
            self._drop_locked(victim)
            self.evictions += 1
        return True

    def _finish(self, request_id: int, slot: _Slot, frame: bytes) -> None:
        """Store ``frame`` and account its bytes, evicting to fit if needed."""
        size = len(frame)
        slot.frame = frame
        slot.finished = time.monotonic()
        slot.frame_size = size
        slot.done.set()
        with self._lock:
            if size > self.max_bytes:
                # Honest: too large to cache; still answer, but don't retain.
                # Pop the slot so later retries re-execute and memory/capacity
                # are released; waiters already woken hold a reference and get
                # this first result.
                slot.cached = False
                self.oversized += 1
                self._slots.pop(request_id, None)
                return
            self._make_bytes_locked(size)
            self._bytes += size

    def _make_bytes_locked(self, needed: int) -> None:
        """Evict completed cached frames oldest-first until ``needed`` bytes fit."""
        while self._bytes + needed > self.max_bytes and self._slots:
            candidates: list[Tuple[float, int]] = []
            for rid, slot in self._slots.items():
                if slot.done.is_set() and slot.cached and slot.finished:
                    candidates.append((slot.finished, rid))
            if not candidates:
                # Nothing evictable; the new frame is too large and will be
                # marked oversized before adding to _bytes, so this path is only
                # reached when the whole budget is in use by in-flight work.
                break
            victim = min(candidates)[1]
            self._drop_locked(victim)
            self.evictions += 1

    def _expire_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl
        stale = [rid for rid, slot in self._slots.items()
                 if slot.done.is_set() and slot.cached
                 and slot.finished is not None and slot.finished < cutoff]
        for rid in stale:
            self._drop_locked(rid)
            self.expiries += 1

    def _drop_locked(self, request_id: int) -> None:
        slot = self._slots.pop(request_id, None)
        if slot is not None and slot.done.is_set() and slot.cached:
            self._bytes = max(0, self._bytes - slot.frame_size)
