"""Retry/failover semantics: no expert contribution is ever counted twice.

Each test counts the requests that actually reached each node, so a silent
double-dispatch would show up as an extra ``calls`` count even when the numeric
result happens to look right.
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
                                  ExpertPlacement, RetryPolicy)
from ps3_cluster.errors import (NodeConnectError, NodeError,  # noqa: E402
                                NodeTimeout)
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402


class TestReplicaPlacement(unittest.TestCase):
    def test_replica_registration_does_not_change_canonical_placement(self):
        placement = ExpertPlacement.one_per_node(1, 2)
        primary = placement.node_for(0, 1)
        placement.assign_replica(0, 1, "standby")
        self.assertEqual(placement.node_for(0, 1), primary)
        self.assertEqual(placement.replicas_for(0, 1), ["standby"])
        self.assertEqual(placement.nodes_for(0, 1), [primary, "standby"])
        self.assertEqual(placement.expert_on("standby"), (0, 1))
        self.assertEqual(len(placement), 2)  # replicas are not placements

    def test_replica_must_differ_from_primary(self):
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        with self.assertRaises(ValueError):
            placement.assign_replica(0, 0, "n0")

    def test_replica_registration_is_idempotent(self):
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        placement.assign_replica(0, 0, "r0")
        placement.assign_replica(0, 0, "r0")
        self.assertEqual(placement.replicas_for(0, 0), ["r0"])


class _FailoverFixture:
    """Expert 0 with a dead (or slow) primary and a live replica."""

    def __init__(self, primary_endpoint, policy, replica_fn=None):
        fn, self.W = linear_expert(400)
        self.replica = RunningNode(0, 0, replica_fn or fn)
        self.placement = ExpertPlacement()
        self.placement.assign(0, 0, "primary")
        self.placement.assign_replica(0, 0, "replica")
        self.transport = PersistentSocketTransport(
            {"primary": primary_endpoint, "replica": self.replica.endpoint},
            timeout=1.0)
        self.dispatcher = DistributedExpertDispatcher(
            self.placement, self.transport, retry_policy=policy)

    def close(self):
        self.dispatcher.close()
        self.transport.close()
        self.replica.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestSafeFailover(unittest.TestCase):
    def test_unreachable_primary_fails_over_to_the_replica_once(self):
        with _FailoverFixture(dead_endpoint(), RetryPolicy(attempts=2)) as fix:
            x = np.arange(1, 5, dtype=np.float32)
            stage = fix.dispatcher.submit_expert_stage(
                0, x, [0], [1.0], token_id=5)
            np.testing.assert_allclose(stage.result(), fix.W @ x,
                                       rtol=1e-5, atol=1e-5)
            # The connect never reached a node, so retrying is at-most-once:
            # exactly one expert evaluation happened, on the replica.
            self.assertEqual(fix.replica.calls, 1)
            self.assertEqual(fix.dispatcher.transport.requests_sent,
                             {"replica": 1})
            self.assertEqual(stage.attempts[0], ["primary", "replica"])

    def test_no_retry_policy_reports_the_dead_primary(self):
        with _FailoverFixture(dead_endpoint(), RetryPolicy.none()) as fix:
            with self.assertRaises(NodeConnectError) as ctx:
                fix.dispatcher.run_expert_stage(0, np.ones(4, np.float32), [0],
                                                [1.0])
            self.assertEqual(ctx.exception.node_id, "primary")
            self.assertEqual(fix.replica.calls, 0)

    def test_retry_without_replicas_stays_on_the_primary(self):
        with _FailoverFixture(dead_endpoint(),
                              RetryPolicy(attempts=3,
                                          use_replicas=False)) as fix:
            with self.assertRaises(NodeConnectError):
                fix.dispatcher.run_expert_stage(0, np.ones(4, np.float32), [0],
                                                [1.0])
            self.assertEqual(fix.replica.calls, 0)


class TestErrorFrameFailover(unittest.TestCase):
    def _fixture(self, policy):
        """A primary that owns the *wrong* expert, so it answers ERR."""
        wrong = RunningNode(0, 7, linear_expert(999)[0])
        fix = _FailoverFixture(wrong.endpoint, policy)
        fix.wrong = wrong
        return fix

    def test_err_is_not_retried_by_default(self):
        fix = self._fixture(RetryPolicy(attempts=2))
        try:
            with self.assertRaises(NodeError) as ctx:
                fix.dispatcher.run_expert_stage(0, np.ones(4, np.float32), [0],
                                                [1.0])
            self.assertEqual(ctx.exception.node_id, "primary")
            self.assertEqual(fix.replica.calls, 0)
        finally:
            fix.close()
            fix.wrong.close()

    def test_err_failover_counts_the_expert_once(self):
        fix = self._fixture(RetryPolicy(attempts=2, retry_on_node_error=True))
        try:
            x = np.arange(4, dtype=np.float32) + 1.0
            out = fix.dispatcher.run_expert_stage(0, x, [0], [1.0])
            np.testing.assert_allclose(out, fix.W @ x, rtol=1e-5, atol=1e-5)
            # The ERR carried no activation, so the single replica result is the
            # only contribution reduced.
            self.assertEqual(fix.replica.calls, 1)
            self.assertEqual(fix.wrong.calls, 0)
        finally:
            fix.close()
            fix.wrong.close()


class TestTimeoutSemantics(unittest.TestCase):
    def test_timeout_is_not_retried_by_default(self):
        """At-most-once: a request that may still be running is not re-sent."""
        release = threading.Event()

        def slow(x):
            release.wait(10)
            return x

        slow_node = RunningNode(0, 0, slow)
        fix = _FailoverFixture(slow_node.endpoint, RetryPolicy(attempts=3))
        try:
            with self.assertRaises(NodeTimeout) as ctx:
                fix.dispatcher.run_expert_stage(0, np.ones(4, np.float32), [0],
                                                [1.0])
            self.assertEqual(ctx.exception.node_id, "primary")
            self.assertEqual(fix.replica.calls, 0)
        finally:
            release.set()
            fix.close()
            slow_node.close()

    def test_opt_in_timeout_retry_is_at_least_once_but_reduces_once(self):
        """With ``retry_on_timeout`` the expert may run twice on the cluster.

        The stale first evaluation is dropped by the transport (its correlation
        key was retired), so exactly one contribution is reduced -- the point
        being that at-least-once *execution* still yields exactly-once
        *reduction*.
        """
        release = threading.Event()
        fn, W = linear_expert(400)

        def slow(x):
            release.wait(10)
            return fn(x)

        slow_node = RunningNode(0, 0, slow)
        fix = _FailoverFixture(slow_node.endpoint,
                               RetryPolicy(attempts=2, retry_on_timeout=True))
        try:
            x = np.arange(1, 5, dtype=np.float32)
            out = fix.dispatcher.run_expert_stage(0, x, [0], [1.0])
            np.testing.assert_allclose(out, W @ x, rtol=1e-5, atol=1e-5)
            self.assertEqual(fix.replica.calls, 1)
            release.set()
            # The primary did start the work: that is the at-least-once cost.
            self.assertEqual(fix.transport.requests_sent,
                             {"primary": 1, "replica": 1})
        finally:
            release.set()
            fix.close()
            slow_node.close()


class TestFailoverInsideTopK(unittest.TestCase):
    def test_only_the_failed_position_fails_over(self):
        nodes, weights = [], []
        placement = ExpertPlacement()
        endpoints = {}
        for expert in range(3):
            fn, W = linear_expert(500 + expert)
            weights.append(W)
            node = RunningNode(0, expert, fn)
            nodes.append(node)
            placement.assign(0, expert, f"n{expert}")
            endpoints[f"n{expert}"] = node.endpoint
        # Expert 1's primary is dead; its replica holds the same expert.
        replica = RunningNode(0, 1, linear_expert(501)[0])
        nodes.append(replica)
        endpoints["n1"] = dead_endpoint()
        endpoints["n1-replica"] = replica.endpoint
        placement.assign_replica(0, 1, "n1-replica")

        transport = PersistentSocketTransport(endpoints, timeout=2.0)
        dispatcher = DistributedExpertDispatcher(
            placement, transport, retry_policy=RetryPolicy())
        try:
            x = np.arange(1, 5, dtype=np.float32)
            gates = [0.5, 0.3, 0.2]
            stage = dispatcher.submit_expert_stage(0, x, [0, 1, 2], gates)
            expected = sum(g * (W @ x) for g, W in zip(gates, weights))
            np.testing.assert_allclose(stage.result(), expected,
                                       rtol=1e-5, atol=1e-5)
            self.assertEqual(stage.attempts[0], ["n0"])
            self.assertEqual(stage.attempts[1], ["n1", "n1-replica"])
            self.assertEqual(stage.attempts[2], ["n2"])
            self.assertEqual([n.calls for n in nodes], [1, 0, 1, 1])
        finally:
            dispatcher.close()
            transport.close()
            for node in nodes:
                node.close()


if __name__ == "__main__":
    unittest.main()
