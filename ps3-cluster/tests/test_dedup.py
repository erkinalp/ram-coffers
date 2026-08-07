"""Bounds and semantics of the coordinator's request-id dedup cache.

These are unit tests of ``DedupCache`` itself: a head server must be able to
replay a retried batch without re-running its consoles, must bind each id to
the request content it names, and must not accumulate per-token state without
limit while doing so. Concurrency is driven with events, not sleeps.
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.dedup import (  # noqa: E402
    DEFAULT_DEDUP_BYTES,
    DEFAULT_DEDUP_ENTRIES,
    DEFAULT_DEDUP_TTL,
    DedupCache,
    DedupCapacityError,
    MismatchedRequestError,
)


def fp(val: int) -> bytes:
    """A fingerprint stand-in."""
    return val.to_bytes(32, "big")


def fp2(seed: str) -> bytes:
    """A different fingerprint stand-in."""
    return (seed * 32)[:32].encode()


class TestRunOnce(unittest.TestCase):
    def test_defaults_are_bounded(self):
        cache = DedupCache()
        self.assertEqual(cache.max_entries, DEFAULT_DEDUP_ENTRIES)
        self.assertEqual(cache.ttl, DEFAULT_DEDUP_TTL)
        self.assertEqual(cache.max_bytes, DEFAULT_DEDUP_BYTES)
        for bad in ({"max_entries": 0}, {"ttl": 0.0}, {"ttl": -1.0},
                    {"max_bytes": -1}):
            with self.assertRaises(ValueError):
                DedupCache(**bad)

    def test_a_repeated_id_with_same_fingerprint_replays_the_first_frame(self):
        cache = DedupCache()
        calls = []

        def compute():
            calls.append(1)
            return b"answer"

        self.assertEqual(cache.run(7, fp(1), compute), b"answer")
        self.assertEqual(cache.run(7, fp(1), compute), b"answer")
        self.assertEqual(len(calls), 1)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertEqual(cache.replay(7), b"answer")

    def test_a_reused_id_with_different_fingerprint_is_rejected(self):
        cache = DedupCache()

        def compute():
            return b"first"

        self.assertEqual(cache.run(7, fp(1), compute), b"first")
        with self.assertRaises(MismatchedRequestError):
            cache.run(7, fp(2), lambda: b"second")
        # It does not replay the remembered frame, and keeps the slot.
        self.assertEqual(cache.replay(7), b"first")
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 1)

    def test_distinct_ids_each_run(self):
        cache = DedupCache()
        self.assertEqual(cache.run(1, fp(1), lambda: b"a"), b"a")
        self.assertEqual(cache.run(2, fp(1), lambda: b"b"), b"b")
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.hits, 0)

    def test_an_unnamed_batch_always_runs(self):
        """No id, nothing to deduplicate on, and no state kept."""
        cache = DedupCache()
        calls = []
        for _ in range(3):
            cache.run(None, b"ignored", lambda: calls.append(1) or b"x")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(cache), 0)

    def test_a_failed_attempt_is_not_remembered(self):
        cache = DedupCache()

        def boom():
            raise RuntimeError("console fell over")

        with self.assertRaises(RuntimeError):
            cache.run(9, fp(1), boom)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.replay(9))
        # Retry with the same logical request content may run again because the
        # failure did not produce a cached response.
        self.assertEqual(cache.run(9, fp(1), lambda: b"second try"),
                         b"second try")


class TestConcurrentDuplicates(unittest.TestCase):
    def test_a_duplicate_waits_for_the_attempt_in_flight(self):
        cache = DedupCache()
        running = threading.Event()
        release = threading.Event()
        calls = []

        def slow():
            calls.append(1)
            running.set()
            release.wait(10)
            return b"one answer"

        first = []
        worker = threading.Thread(
            target=lambda: first.append(cache.run(3, fp(1), slow)))
        worker.start()
        self.assertTrue(running.wait(10))
        second = []
        waiter = threading.Thread(
            target=lambda: second.append(
                cache.run(3, fp(1), slow, timeout=10)))
        waiter.start()
        release.set()
        worker.join(10)
        waiter.join(10)
        self.assertEqual(first, [b"one answer"])
        self.assertEqual(second, [b"one answer"])
        self.assertEqual(len(calls), 1)

    def test_a_duplicate_with_mismatched_fingerprint_while_in_flight(self):
        cache = DedupCache()
        running = threading.Event()
        release = threading.Event()

        def slow():
            running.set()
            release.wait(10)
            return b"one answer"

        worker = threading.Thread(target=lambda: cache.run(3, fp(1), slow))
        worker.start()
        self.assertTrue(running.wait(10))
        try:
            with self.assertRaises(MismatchedRequestError):
                cache.run(3, fp(2), slow)
        finally:
            release.set()
            worker.join(10)

    def test_two_concurrent_same_id_callers_run_once(self):
        """Two concurrent same-id callers produce exactly one execution."""
        cache = DedupCache()
        start = threading.Event()
        release = threading.Event()
        calls = []
        results = []

        def slow():
            start.set()
            calls.append(1)
            release.wait(10)
            return b"the only answer"

        def wait_for_it():
            results.append(cache.run(1, fp(1), slow, timeout=10))

        t1 = threading.Thread(target=wait_for_it)
        t2 = threading.Thread(target=wait_for_it)
        t1.start()
        t2.start()
        self.assertTrue(start.wait(10))
        release.set()
        t1.join(10)
        t2.join(10)
        self.assertEqual(calls, [1])
        self.assertEqual(results,
                         [b"the only answer", b"the only answer"])

    def test_waiting_on_an_attempt_that_outlives_the_deadline_times_out(self):
        cache = DedupCache()
        running = threading.Event()
        release = threading.Event()

        def slow():
            running.set()
            release.wait(10)
            return b"late"

        worker = threading.Thread(target=lambda: cache.run(4, fp(1), slow))
        worker.start()
        try:
            self.assertTrue(running.wait(10))
            with self.assertRaises(TimeoutError):
                cache.run(4, fp(1), slow, timeout=0.05)
        finally:
            release.set()
            worker.join(10)

    def test_a_duplicate_sees_the_first_attempts_exception(self):
        cache = DedupCache()
        running = threading.Event()
        release = threading.Event()

        def failing():
            running.set()
            release.wait(10)
            raise RuntimeError("head server lost its console")

        worker = threading.Thread(
            target=lambda: self.assertRaises(RuntimeError, cache.run, 5,
                                             fp(1), failing))
        worker.start()
        self.assertTrue(running.wait(10))
        waiter_error = []

        def wait_for_it():
            try:
                cache.run(5, fp(1), failing, timeout=10)
            except RuntimeError as exc:
                waiter_error.append(str(exc))

        waiter = threading.Thread(target=wait_for_it)
        waiter.start()
        release.set()
        worker.join(10)
        waiter.join(10)
        self.assertEqual(waiter_error, ["head server lost its console"])


class TestBounds(unittest.TestCase):
    def test_completed_entries_are_evicted_at_the_limit(self):
        cache = DedupCache(max_entries=3)
        for rid in range(6):
            cache.run(rid, fp(rid), lambda: b"x")
        self.assertLessEqual(len(cache), 3)
        self.assertGreaterEqual(cache.evictions, 3)
        self.assertIsNone(cache.replay(0))          # oldest is gone
        self.assertEqual(cache.replay(5), b"x")     # newest is replayable

    def test_in_flight_slots_are_never_evicted(self):
        cache = DedupCache(max_entries=2)
        running1 = threading.Event()
        running2 = threading.Event()
        release = threading.Event()
        order = []

        def make_slow(n):
            def slow():
                order.append(n)
                (running1 if n == 1 else running2).set()
                release.wait(10)
                return b"mine"
            return slow

        w1 = threading.Thread(target=lambda: cache.run(1, fp(1), make_slow(1)))
        w2 = threading.Thread(target=lambda: cache.run(2, fp(2), make_slow(2)))
        w1.start()
        w2.start()
        self.assertTrue(running1.wait(10))
        self.assertTrue(running2.wait(10))
        try:
            # Both slots are in flight; there is no completed entry to evict.
            with self.assertRaises(DedupCapacityError):
                cache.run(3, fp(3), lambda: b"backpressure")
            self.assertEqual(cache.rejected, 1)
        finally:
            release.set()
            w1.join(10)
            w2.join(10)

    def test_byte_budget_evicts_completed_frames(self):
        cache = DedupCache(max_entries=10, max_bytes=8)
        cache.run(1, fp(1), lambda: b"12345")  # 5 bytes
        cache.run(2, fp(2), lambda: b"AB")    # 2 bytes, total 7
        # A 6-byte response needs to evict the oldest 5-byte frame.
        cache.run(3, fp(3), lambda: b"second")
        self.assertEqual(cache.bytes_used, 8)
        self.assertIsNone(cache.replay(1))
        self.assertEqual(cache.replay(2), b"AB")
        self.assertEqual(cache.replay(3), b"second")
        self.assertGreaterEqual(cache.evictions, 1)

    def test_oversized_response_is_answered_but_not_cached(self):
        cache = DedupCache(max_entries=10, max_bytes=5)
        frame = b"0123456789" * 2  # 20 bytes, > 5
        calls = []

        def compute():
            calls.append(1)
            return frame

        self.assertEqual(cache.run(1, fp(1), compute), frame)
        self.assertEqual(cache.bytes_used, 0)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.replay(1))
        self.assertEqual(cache.oversized, 1)
        # A later retry re-executes because the oversized frame was not retained.
        self.assertEqual(cache.run(1, fp(1), compute), frame)
        self.assertEqual(len(calls), 2)

    def test_oversized_responses_max_bytes_zero(self):
        cache = DedupCache(max_entries=5, max_bytes=0)
        frame = b"not empty"
        self.assertEqual(cache.run(1, fp(1), lambda: frame), frame)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.bytes_used, 0)
        self.assertIsNone(cache.replay(1))
        self.assertEqual(cache.oversized, 1)

    def test_oversized_responses_do_not_fill_the_slot_map(self):
        cache = DedupCache(max_entries=2, max_bytes=5)
        frame = b"!" * 10
        for rid in range(5):
            self.assertEqual(cache.run(rid, fp(rid), lambda: frame), frame)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.oversized, 5)

    def test_oversized_concurrent_duplicates_share_the_first_result(self):
        cache = DedupCache(max_entries=5, max_bytes=5)
        running = threading.Event()
        release = threading.Event()
        calls = []
        frame = b"!" * 10

        def compute():
            calls.append(1)
            running.set()
            release.wait(10)
            return frame

        first = []
        worker = threading.Thread(
            target=lambda: first.append(cache.run(9, fp(9), compute)))
        worker.start()
        self.assertTrue(running.wait(10))
        second = []
        waiter = threading.Thread(
            target=lambda: second.append(
                cache.run(9, fp(9), compute, timeout=10)))
        waiter.start()
        release.set()
        worker.join(10)
        waiter.join(10)
        self.assertEqual(first, [frame])
        self.assertEqual(second, [frame])
        self.assertEqual(len(calls), 1)
        # After completion the oversized slot is gone; a retry re-executes.
        self.assertEqual(cache.run(9, fp(9), compute), frame)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cache), 0)

    def test_an_expired_entry_is_dropped_and_re_executed(self):
        cache = DedupCache(max_entries=8, ttl=0.01)
        calls = []
        cache.run(1, fp(1), lambda: calls.append(1) or b"x")
        for _ in range(200):
            if cache.replay(1) is None:
                break
            threading.Event().wait(0.01)
        self.assertIsNone(cache.replay(1))
        self.assertGreaterEqual(cache.expiries, 1)
        cache.run(1, fp(1), lambda: calls.append(1) or b"x")
        # A retry after expiry honestly re-executes.
        self.assertEqual(len(calls), 2)

    def test_clear_drops_everything(self):
        cache = DedupCache()
        cache.run(1, fp(1), lambda: b"x")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.bytes_used, 0)
        self.assertIsNone(cache.replay(1))


if __name__ == "__main__":
    unittest.main()
