"""Subcluster grouping and hierarchical partial reduction.

Grouping defaults to 22 nodes per subcluster, the size AFRL's Condor cluster
used behind each coordinating server (Barnell et al., IEEE HPEC 2012). The
end-to-end case runs real expert nodes split across two subclusters and checks
the partial sums the head servers would forward.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _netutil import RunningNode, linear_expert  # noqa: E402
from ps3_cluster.dispatch import (DistributedExpertDispatcher,  # noqa: E402
                                  ExpertPlacement)
from ps3_cluster.subcluster import (DEFAULT_SUBCLUSTER_SIZE,  # noqa: E402
                                    SubclusterPlan, hierarchical_reduce,
                                    partial_reduce)
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402


class TestSubclusterPlan(unittest.TestCase):
    def test_default_size_is_condors_22(self):
        self.assertEqual(DEFAULT_SUBCLUSTER_SIZE, 22)

    def test_groups_of_22_with_a_short_tail(self):
        nodes = [f"ps3-{i:04d}" for i in range(50)]
        plan = SubclusterPlan(nodes)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan.n_nodes, 50)
        sizes = [len(plan.members(g)) for g in plan.group_ids()]
        self.assertEqual(sizes, [22, 22, 6])
        self.assertEqual(plan.subcluster_of("ps3-0000"), "sc-0000")
        self.assertEqual(plan.subcluster_of("ps3-0021"), "sc-0000")
        self.assertEqual(plan.subcluster_of("ps3-0022"), "sc-0001")
        self.assertTrue(plan.same_subcluster("ps3-0000", "ps3-0021"))
        self.assertFalse(plan.same_subcluster("ps3-0021", "ps3-0022"))

    def test_grouping_is_stable_for_the_same_node_order(self):
        nodes = [f"ps3-{i:04d}" for i in range(45)]
        self.assertEqual(SubclusterPlan(nodes).groups(),
                         SubclusterPlan(nodes).groups())

    def test_duplicate_and_bad_size_rejected(self):
        with self.assertRaises(ValueError):
            SubclusterPlan(["a", "a"])
        with self.assertRaises(ValueError):
            SubclusterPlan(["a"], size=0)

    def test_for_placement_covers_one_layer(self):
        placement = ExpertPlacement.one_per_node(3, 30)
        plan = SubclusterPlan.for_placement(placement, layer=1, size=22)
        self.assertEqual(plan.n_nodes, 30)
        self.assertEqual([len(plan.members(g)) for g in plan.group_ids()],
                         [22, 8])
        for node in plan.members("sc-0000"):
            self.assertTrue(node.startswith("ps3-L001-"))

    def test_placement_is_unchanged_by_grouping(self):
        placement = ExpertPlacement.one_per_node(2, 5)
        before = placement.node_for(1, 3)
        SubclusterPlan.for_placement(placement, size=2)
        self.assertEqual(placement.node_for(1, 3), before)

    def test_group_by_subcluster_keeps_positions_and_order(self):
        plan = SubclusterPlan([f"n{i}" for i in range(6)], size=2)
        buckets = plan.group_by_subcluster(["n4", "n1", "n0", "n5"])
        self.assertEqual(buckets, [("sc-0000", [1, 2]), ("sc-0002", [0, 3])])


class TestPartialReduction(unittest.TestCase):
    def test_partial_reduce_sums_in_position_order(self):
        contributions = {2: np.full(3, 2.0, np.float32),
                         0: np.full(3, 1.0, np.float32)}
        np.testing.assert_array_equal(partial_reduce(contributions, [2, 0]),
                                      np.full(3, 3.0, np.float32))

    def test_hierarchical_matches_flat_and_reports_partials(self):
        rng = np.random.default_rng(3)
        node_ids = [f"n{i}" for i in range(6)]
        plan = SubclusterPlan(node_ids, size=3)
        contributions = {i: rng.standard_normal(4).astype(np.float32)
                         for i in range(6)}
        seen = {}
        out = hierarchical_reduce(
            contributions, node_ids, plan,
            on_partial=lambda g, p: seen.__setitem__(g, p))
        flat = sum(contributions[i] for i in range(6))
        np.testing.assert_allclose(out, flat, rtol=1e-6, atol=1e-6)
        self.assertEqual(sorted(seen), ["sc-0000", "sc-0001"])
        np.testing.assert_allclose(
            seen["sc-0000"], contributions[0] + contributions[1] +
            contributions[2], rtol=0, atol=0)
        # Deterministic for a fixed plan.
        self.assertTrue(np.array_equal(
            out, hierarchical_reduce(contributions, node_ids, plan)))

    def test_hierarchical_reduce_needs_contributions(self):
        plan = SubclusterPlan(["n0"], size=1)
        with self.assertRaises(ValueError):
            hierarchical_reduce({}, ["n0"], plan)


class TestHierarchicalDispatch(unittest.TestCase):
    def test_topk_reduced_through_two_subclusters(self):
        nodes, weights = [], []
        placement = ExpertPlacement()
        for expert in range(4):
            fn, W = linear_expert(300 + expert)
            weights.append(W)
            nodes.append(RunningNode(0, expert, fn))
            placement.assign(0, expert, f"n{expert}")
        plan = SubclusterPlan(placement.node_ids(0), size=2)
        transport = PersistentSocketTransport(
            {f"n{i}": n.endpoint for i, n in enumerate(nodes)}, timeout=15.0)
        dispatcher = DistributedExpertDispatcher(placement, transport,
                                                 subclusters=plan)
        try:
            self.assertEqual(len(plan), 2)
            x = np.arange(1, 5, dtype=np.float32)
            gates = [0.4, 0.3, 0.2, 0.1]
            out = dispatcher.run_expert_stage(0, x, [0, 1, 2, 3], gates)
            expected = sum(g * (W @ x) for g, W in zip(gates, weights))
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
            self.assertEqual([n.calls for n in nodes], [1, 1, 1, 1])
        finally:
            dispatcher.close()
            transport.close()
            for node in nodes:
                node.close()


if __name__ == "__main__":
    unittest.main()
