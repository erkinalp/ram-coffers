"""Real-socket tests for the persistent/pooled P3XC transport.

Every test here boots an actual worker on loopback and speaks the wire protocol
to it; nothing is mocked. Concurrency is proven with barriers and events, so no
assertion depends on a latency threshold.
"""

import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _netutil import (HangUpServer, ReorderingServer, RunningNode,  # noqa: E402
                      dead_endpoint, linear_expert, reader_threads)
from ps3_cluster.errors import (NodeConnectError, NodeDisconnected,  # noqa: E402
                                NodeError, NodeTimeout, TransportClosed)
from ps3_cluster.transport import (PersistentSocketTransport,  # noqa: E402
                                   PooledSocketTransport)


class TestPersistentConnectionReuse(unittest.TestCase):
    def test_one_connection_serves_many_requests(self):
        fn, W = linear_expert(11)
        with RunningNode(2, 3, fn) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=5.0)
            try:
                for token in range(8):
                    x = np.full(4, float(token) + 1.0, dtype=np.float32)
                    y = transport.dispatch("n", 2, 3, token, x)
                    np.testing.assert_allclose(y, W @ x, rtol=1e-5, atol=1e-5)
                self.assertEqual(transport.connects_opened["n"], 1)
                self.assertEqual(transport.requests_sent["n"], 8)
                self.assertEqual(transport.connection_count("n"), 1)
                # The worker saw one TCP accept for all eight requests.
                self.assertEqual(node.accepted, 1)
                self.assertEqual(node.calls, 8)
            finally:
                transport.close()

    def test_pooled_alias_is_the_persistent_transport(self):
        self.assertIs(PooledSocketTransport, PersistentSocketTransport)


class TestConcurrentInFlight(unittest.TestCase):
    def test_requests_to_one_node_overlap(self):
        """Four requests to one node must be in flight simultaneously.

        The expert blocks on a barrier that only trips when all four have
        arrived, so the test can only pass if the coordinator does not serialise
        request/response pairs. The worker answers one request at a time per
        connection, so the transport must spread them over its pool.
        """
        parties = 4
        barrier = threading.Barrier(parties, timeout=10)
        fn, W = linear_expert(5)

        def blocking(x):
            barrier.wait()
            return fn(x)

        with RunningNode(1, 1, blocking) as node:
            transport = PersistentSocketTransport(
                {"n": node.endpoint}, timeout=10.0,
                max_connections_per_node=parties)
            try:
                pending = [transport.submit("n", 1, 1, token,
                                            np.full(4, token + 1.0, np.float32))
                           for token in range(parties)]
                for token, request in enumerate(pending):
                    x = np.full(4, token + 1.0, np.float32)
                    np.testing.assert_allclose(request.result(), W @ x,
                                               rtol=1e-5, atol=1e-5)
                self.assertEqual(node.calls, parties)
            finally:
                transport.close()

    def test_many_threads_share_the_transport(self):
        fn, W = linear_expert(17)
        n_threads = 8
        results = {}
        errors = []
        start = threading.Barrier(n_threads, timeout=10)

        with RunningNode(0, 0, fn) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=10.0)

            def worker(token):
                try:
                    start.wait()
                    x = np.full(4, token + 1.0, np.float32)
                    results[token] = transport.dispatch("n", 0, 0, token, x)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,))
                       for t in range(n_threads)]
            try:
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=15)
                self.assertEqual(errors, [])
                self.assertEqual(len(results), n_threads)
                for token, y in results.items():
                    x = np.full(4, token + 1.0, np.float32)
                    np.testing.assert_allclose(y, W @ x, rtol=1e-5, atol=1e-5)
            finally:
                transport.close()


class TestOutOfOrderCorrelation(unittest.TestCase):
    def test_responses_arriving_reversed_are_matched_by_identity(self):
        with ReorderingServer(4, 6, batch=3) as server:
            transport = PersistentSocketTransport({"n": server.endpoint},
                                                  timeout=10.0,
                                                  max_connections_per_node=1)
            try:
                tokens = [101, 202, 303]
                # max_connections_per_node=1 pins all three to one socket, and
                # the server only answers once all three have arrived, so this
                # also proves multiplexing on a single connection.
                pending = [transport.submit("n", 4, 6, token,
                                            np.zeros(3, np.float32))
                           for token in tokens]
                # The server replies newest-first.
                for token, request in zip(tokens, pending):
                    y = request.result()
                    np.testing.assert_allclose(y, np.full(3, float(token)),
                                               rtol=0, atol=0)
                self.assertEqual(server.accepted, 1)
            finally:
                transport.close()


class TestHeartbeat(unittest.TestCase):
    def test_ping_pong_round_trip(self):
        fn, _ = linear_expert(3)
        with RunningNode(7, 8, fn) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=5.0)
            try:
                rtt = transport.ping("n")
                self.assertGreaterEqual(rtt, 0.0)
                self.assertTrue(transport.alive("n"))
                # A heartbeat shares the connection with expert traffic.
                x = np.arange(4, dtype=np.float32)
                transport.dispatch("n", 7, 8, 1, x)
                self.assertTrue(transport.alive("n"))
                self.assertEqual(node.accepted, 1)
            finally:
                transport.close()

    def test_dead_node_is_named_in_the_failure(self):
        transport = PersistentSocketTransport({"gone": dead_endpoint()},
                                              timeout=2.0)
        try:
            with self.assertRaises(NodeConnectError) as ctx:
                transport.ping("gone")
            self.assertEqual(ctx.exception.node_id, "gone")
            self.assertIn("gone", str(ctx.exception))
            self.assertFalse(transport.alive("gone"))
        finally:
            transport.close()

    def test_node_that_drops_the_connection_is_reported(self):
        with HangUpServer() as server:
            transport = PersistentSocketTransport(
                {"n": server.endpoint}, timeout=5.0)
            try:
                with self.assertRaises(NodeDisconnected) as ctx:
                    transport.ping("n")
                self.assertEqual(ctx.exception.node_id, "n")
                self.assertFalse(transport.alive("n"))
            finally:
                transport.close()


class TestFailureCleanup(unittest.TestCase):
    def test_timeout_raises_and_does_not_mis_correlate_the_late_reply(self):
        release = threading.Event()
        fn, W = linear_expert(21)

        def slow(x):
            release.wait(10)
            return fn(x)

        with RunningNode(3, 4, slow) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=0.2)
            try:
                with self.assertRaises(NodeTimeout) as ctx:
                    transport.dispatch("n", 3, 4, 55, np.ones(4, np.float32))
                self.assertEqual(ctx.exception.node_id, "n")
                # Let the abandoned request finish; its late response must be
                # dropped, and a fresh request must still get its own answer.
                release.set()
                x = np.arange(1, 5, dtype=np.float32)
                y = transport.dispatch("n", 3, 4, 56, x)
                np.testing.assert_allclose(y, W @ x, rtol=1e-5, atol=1e-5)
            finally:
                transport.close()

    def test_error_frame_becomes_a_node_error_and_keeps_the_connection(self):
        fn, W = linear_expert(9)
        with RunningNode(5, 5, fn) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=5.0)
            try:
                # Wrong expert for this node -> the worker answers ERR.
                with self.assertRaises(NodeError) as ctx:
                    transport.dispatch("n", 5, 6, 1, np.ones(4, np.float32))
                self.assertEqual(ctx.exception.node_id, "n")
                x = np.ones(4, np.float32)
                np.testing.assert_allclose(
                    transport.dispatch("n", 5, 5, 2, x), W @ x,
                    rtol=1e-5, atol=1e-5)
                self.assertEqual(node.accepted, 1)
            finally:
                transport.close()

    def test_expert_exception_is_reported_per_node(self):
        def broken(x):
            raise ValueError("expert blew up")

        with RunningNode(0, 1, broken) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=5.0)
            try:
                with self.assertRaises(NodeError):
                    transport.dispatch("n", 0, 1, 1, np.ones(4, np.float32))
            finally:
                transport.close()

    def test_close_releases_sockets_and_reader_threads(self):
        fn, _ = linear_expert(2)
        before = set(reader_threads())
        with RunningNode(1, 1, fn) as node:
            transport = PersistentSocketTransport({"n": node.endpoint},
                                                  timeout=5.0)
            transport.dispatch("n", 1, 1, 0, np.ones(4, np.float32))
            self.assertEqual(transport.connection_count(), 1)
            transport.close()
            self.assertEqual(transport.connection_count(), 0)
            for _ in range(100):
                if set(reader_threads()) <= before:
                    break
                threading.Event().wait(0.02)
            self.assertLessEqual(set(reader_threads()), before)
            with self.assertRaises(TransportClosed):
                transport.dispatch("n", 1, 1, 1, np.ones(4, np.float32))
            transport.close()  # idempotent

    def test_context_manager_closes(self):
        fn, _ = linear_expert(4)
        with RunningNode(1, 1, fn) as node:
            with PersistentSocketTransport({"n": node.endpoint},
                                           timeout=5.0) as transport:
                transport.dispatch("n", 1, 1, 0, np.ones(4, np.float32))
            self.assertEqual(transport.connection_count(), 0)

    def test_unknown_node_is_a_connect_error(self):
        transport = PersistentSocketTransport({}, timeout=1.0)
        try:
            with self.assertRaises(NodeConnectError):
                transport.dispatch("nope", 0, 0, 0, np.ones(4, np.float32))
        finally:
            transport.close()


if __name__ == "__main__":
    unittest.main()
