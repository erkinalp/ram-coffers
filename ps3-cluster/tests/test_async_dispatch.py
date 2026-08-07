"""Top-k fan-out tests: concurrency, determinism, cleanup, liveness.

The concurrency proof is a barrier shared by the selected experts: it only trips
when every expert of the top-k stage is executing at the same time, so a serial
dispatcher deadlocks the barrier instead of merely being slower. No assertion
depends on a latency threshold.
"""

import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _netutil import RunningNode, dead_endpoint, linear_expert  # noqa: E402
from ps3_cluster.dispatch import (DistributedExpertDispatcher,  # noqa: E402
                                  ExpertPlacement, LoopbackTransport,
                                  RetryPolicy)
from ps3_cluster.errors import NodeError  # noqa: E402
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402

TOPK = 4


class _Cluster:
    """``TOPK`` real expert nodes in layer 0, plus a wired-up dispatcher."""

    def __init__(self, gate_fn=None, **dispatcher_kwargs):
        self.nodes = []
        self.weights = []
        self.placement = ExpertPlacement()
        for expert in range(TOPK):
            fn, W = linear_expert(100 + expert)
            self.weights.append(W)
            wrapped = gate_fn(fn) if gate_fn is not None else fn
            node = RunningNode(0, expert, wrapped)
            self.nodes.append(node)
            self.placement.assign(0, expert, f"n{expert}")
        self.transport = PersistentSocketTransport(
            {f"n{i}": node.endpoint for i, node in enumerate(self.nodes)},
            timeout=15.0)
        self.dispatcher = DistributedExpertDispatcher(
            self.placement, self.transport, **dispatcher_kwargs)

    def expected(self, x, gates):
        out = (gates[0] * (self.weights[0] @ x)).astype(np.float32)
        for i in range(1, TOPK):
            out = out + (gates[i] * (self.weights[i] @ x)).astype(np.float32)
        return out

    def close(self):
        self.dispatcher.close()
        self.transport.close()
        for node in self.nodes:
            node.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestConcurrentTopK(unittest.TestCase):
    def test_topk_experts_run_simultaneously(self):
        barrier = threading.Barrier(TOPK, timeout=15)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        with _Cluster(gate_fn=gated) as cluster:
            x = np.arange(1, 5, dtype=np.float32)
            gates = [0.4, 0.3, 0.2, 0.1]
            out = cluster.dispatcher.run_expert_stage(0, x, list(range(TOPK)),
                                                      gates, token_id=1)
            np.testing.assert_allclose(out, cluster.expected(x, gates),
                                       rtol=1e-5, atol=1e-5)
            self.assertEqual([node.calls for node in cluster.nodes],
                             [1] * TOPK)

    def test_serial_dispatch_would_not_satisfy_the_barrier(self):
        """max_workers=1 forces serial fan-out, so the barrier must time out.

        This is the control for the previous test: it shows the barrier really
        is a concurrency proof and not something a serial loop can pass.
        """
        barrier = threading.Barrier(TOPK, timeout=1.0)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        with _Cluster(gate_fn=gated, max_workers=1) as cluster:
            with self.assertRaises(NodeError):
                cluster.dispatcher.run_expert_stage(
                    0, np.ones(4, np.float32), list(range(TOPK)),
                    [0.25] * TOPK, token_id=2)

    def test_stages_can_be_in_flight_together(self):
        barrier = threading.Barrier(2 * TOPK, timeout=15)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        with _Cluster(gate_fn=gated) as cluster:
            xs = [np.full(4, 1.0, np.float32), np.full(4, 2.0, np.float32)]
            gates = [0.25] * TOPK
            stages = [cluster.dispatcher.submit_expert_stage(
                0, x, list(range(TOPK)), gates, token_id=token)
                for token, x in enumerate(xs, start=10)]
            for x, stage in zip(xs, stages):
                np.testing.assert_allclose(stage.result(),
                                           cluster.expected(x, gates),
                                           rtol=1e-5, atol=1e-5)
            self.assertEqual([node.calls for node in cluster.nodes],
                             [2] * TOPK)


class TestDeterminism(unittest.TestCase):
    def test_concurrent_reduction_is_bit_identical_to_serial(self):
        experts, weights = {}, []
        placement = ExpertPlacement()
        for i in range(8):
            fn, W = linear_expert(200 + i, dim=8)
            experts[f"n{i}"] = fn
            weights.append(W)
            placement.assign(0, i, f"n{i}")
        transport = LoopbackTransport(experts)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(8).astype(np.float32)
        gates = [0.31, 0.07, 0.19, 0.11, 0.03, 0.13, 0.09, 0.07]

        serial = None
        for i, gate in enumerate(gates):
            contrib = (gate * (weights[i] @ x)).astype(np.float32)
            serial = contrib if serial is None else serial + contrib

        with DistributedExpertDispatcher(placement, transport) as dispatcher:
            for _ in range(5):
                out = dispatcher.run_expert_stage(0, x, list(range(8)), gates)
                # Bit-for-bit: contributions are scaled as they arrive but
                # summed in top-k order.
                self.assertTrue(np.array_equal(out, serial))

    def test_empty_topk_returns_zeros(self):
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        with DistributedExpertDispatcher(
                placement, LoopbackTransport({"n0": linear_expert(1)[0]})
        ) as dispatcher:
            out = dispatcher.run_expert_stage(0, np.ones(4, np.float32), [], [])
            np.testing.assert_array_equal(out, np.zeros(4, np.float32))

    def test_length_mismatch_rejected(self):
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        with DistributedExpertDispatcher(
                placement, LoopbackTransport({"n0": linear_expert(1)[0]})
        ) as dispatcher:
            with self.assertRaises(ValueError):
                dispatcher.run_expert_stage(0, np.ones(4, np.float32), [0], [])


class TestFailureCleanup(unittest.TestCase):
    def test_one_failing_expert_does_not_leak_the_others(self):
        """A broken expert surfaces as a node-specific error.

        The sibling calls are still awaited, so no coordinator thread is left
        running and no socket is left half-read: the transport is immediately
        reusable afterwards.
        """
        with _Cluster() as cluster:
            x = np.ones(4, np.float32)
            gates = [0.25] * TOPK
            # Expert 9 lives on no node in this layer -> route to a node that
            # will answer ERR because it owns a different expert.
            cluster.placement.assign(0, 9, "n1")
            with self.assertRaises(NodeError) as ctx:
                cluster.dispatcher.run_expert_stage(0, x, [0, 9, 2, 3], gates)
            self.assertEqual(ctx.exception.node_id, "n1")
            # Every sibling still ran exactly once and the transport is intact.
            out = cluster.dispatcher.run_expert_stage(0, x, list(range(TOPK)),
                                                      gates)
            np.testing.assert_allclose(out, cluster.expected(x, gates),
                                       rtol=1e-5, atol=1e-5)
            self.assertEqual(cluster.transport.connection_count("n1"), 1)

    def test_dispatcher_close_shuts_the_fanout_pool(self):
        with _Cluster() as cluster:
            cluster.dispatcher.run_expert_stage(0, np.ones(4, np.float32),
                                                [0, 1], [0.5, 0.5])
            cluster.dispatcher.close()
            self.assertEqual(
                [t for t in threading.enumerate()
                 if t.name.startswith("p3xc-dispatch") and t.is_alive()
                 and not t.daemon], [])
            # A closed dispatcher lazily rebuilds its pool.
            cluster.dispatcher.run_expert_stage(0, np.ones(4, np.float32),
                                                [0, 1], [0.5, 0.5])


class TestLiveness(unittest.TestCase):
    def test_check_liveness_names_the_dead_node(self):
        with _Cluster() as cluster:
            cluster.transport.add_endpoint("ghost", dead_endpoint())
            alive = cluster.dispatcher.check_liveness(
                ["n0", "n1", "ghost"], timeout=2.0)
            self.assertEqual(alive, {"n0": True, "n1": True, "ghost": False})

    def test_check_liveness_requires_a_ping_capable_transport(self):
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        with DistributedExpertDispatcher(
                placement, LoopbackTransport({"n0": linear_expert(1)[0]})
        ) as dispatcher:
            with self.assertRaises(NotImplementedError):
                dispatcher.check_liveness(["n0"])


class TestRetryPolicyValidation(unittest.TestCase):
    def test_attempts_must_be_positive(self):
        with self.assertRaises(ValueError):
            RetryPolicy(attempts=0)

    def test_none_policy_is_single_attempt(self):
        self.assertEqual(RetryPolicy.none().attempts, 1)


if __name__ == "__main__":
    unittest.main()
