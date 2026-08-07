"""Bounds and semantics of the coordinator's request-id dedup cache.

These are unit tests of ``DedupCache`` itself: a head server must be able to
replay a retried batch without re-running its consoles, and must not accumulate
per-token state without limit while doing so. Concurrency is driven with events,
not sleeps.
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.dedup import (DEFAULT_DEDUP_ENTRIES,  # noqa: E402
                               DEFAULT_DEDUP_TTL, DedupCache)


class TestRunOnce(unittest.TestCase):
    def test_defaults_are_bounded(self):
        cache = DedupCache()
        self.assertEqual(cache.max_entries, DEFAULT_DEDUP_ENTRIES)
        self.assertEqual(cache.ttl, DEFAULT_DEDUP_TTL)
        for bad in ({"max_entries": 0}, {"ttl": 0.0}, {"ttl": -1.0}):
            with self.assertRaises(ValueError):
                DedupCache(**bad)

    def test_a_repeated_id_replays_the_first_frame(self):
        cache = DedupCache()
        calls = []

        def compute():
            calls.append(1)
            return b"answer"

        self.assertEqual(cache.run(7, compute), b"answer")
        self.assertEqual(cache.run(7, compute), b"answer")
        self.assertEqual(len(calls), 1)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertEqual(cache.replay(7), b"answer")

    def test_distinct_ids_each_run(self):
        cache = DedupCache()
        self.assertEqual(cache.run(1, lambda: b"a"), b"a")
        self.assertEqual(cache.run(2, lambda: b"b"), b"b")
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.hits, 0)

    def test_an_unnamed_batch_always_runs(self):
        """No id, nothing to deduplicate on, and no state kept."""
        cache = DedupCache()
        calls = []
        for _ in range(3):
            cache.run(None, lambda: calls.append(1) or b"x")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(cache), 0)

    def test_a_failed_attempt_is_not_remembered(self):
        cache = DedupCache()

        def boom():
            raise RuntimeError("console fell over")

        with self.assertRaises(RuntimeError):
            cache.run(9, boom)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.replay(9))
        self.assertEqual(cache.run(9, lambda: b"second try"), b"second try")


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
            target=lambda: first.append(cache.run(3, slow)))
        worker.start()
        self.assertTrue(running.wait(10))
        second = []
        waiter = threading.Thread(
            target=lambda: second.append(cache.run(3, slow, timeout=10)))
        waiter.start()
        release.set()
        worker.join(10)
        waiter.join(10)
        self.assertEqual(first, [b"one answer"])
        self.assertEqual(second, [b"one answer"])
        self.assertEqual(len(calls), 1)

    def test_waiting_on_an_attempt_that_outlives_the_deadline_times_out(self):
        cache = DedupCache()
        running = threading.Event()
        release = threading.Event()

        def slow():
            running.set()
            release.wait(10)
            return b"late"

        worker = threading.Thread(target=lambda: cache.run(4, slow))
        worker.start()
        try:
            self.assertTrue(running.wait(10))
            with self.assertRaises(TimeoutError):
                cache.run(4, slow, timeout=0.05)
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
                                             failing))
        worker.start()
        self.assertTrue(running.wait(10))
        waiter_error = []

        def wait_for_it():
            try:
                cache.run(5, failing, timeout=10)
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
            cache.run(rid, lambda: b"x")
        self.assertLessEqual(len(cache), 3)
        self.assertGreaterEqual(cache.evictions, 3)
        self.assertIsNone(cache.replay(0))          # oldest is gone
        self.assertEqual(cache.replay(5), b"x")     # newest is replayable

    def test_an_expired_entry_is_dropped_and_re_executed(self):
        cache = DedupCache(max_entries=8, ttl=0.01)
        calls = []
        cache.run(1, lambda: calls.append(1) or b"x")
        for _ in range(200):
            if cache.replay(1) is None:
                break
            threading.Event().wait(0.01)
        self.assertIsNone(cache.replay(1))
        self.assertGreaterEqual(cache.expiries, 1)
        cache.run(1, lambda: calls.append(1) or b"x")
        self.assertEqual(len(calls), 2)             # honestly at-least-once

    def test_an_in_flight_entry_can_be_evicted_without_breaking_its_caller(self):
        """Under pressure the oldest slot goes; its own caller still answers."""
        cache = DedupCache(max_entries=1)
        running = threading.Event()
        release = threading.Event()
        answer = []

        def slow():
            running.set()
            release.wait(10)
            return b"mine"

        worker = threading.Thread(
            target=lambda: answer.append(cache.run(1, slow)))
        worker.start()
        try:
            self.assertTrue(running.wait(10))
            cache.run(2, lambda: b"other")   # forces the eviction
            self.assertGreaterEqual(cache.evictions, 1)
        finally:
            release.set()
            worker.join(10)
        self.assertEqual(answer, [b"mine"])

    def test_clear_drops_everything(self):
        cache = DedupCache()
        cache.run(1, lambda: b"x")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.replay(1))


if __name__ == "__main__":
    unittest.main()
