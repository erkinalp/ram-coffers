import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster import topology as T


class TestTopology(unittest.TestCase):
    def test_k3_total_experts(self):
        self.assertEqual(T.KIMI_K3.total_experts, 92 * 896)
        self.assertEqual(T.KIMI_K3.total_experts, 82432)

    def test_k3_expert_fits_one_node(self):
        plan = T.plan_cluster(T.KIMI_K3)
        self.assertEqual(plan.experts_split_across, 1)
        self.assertLessEqual(plan.per_expert_mb, T.PS3_USABLE_RAM_MB)

    def test_default_is_one_expert_per_node(self):
        # The canonical design: exactly one expert/node even though more fit.
        plan = T.plan_cluster(T.KIMI_K3)
        self.assertEqual(plan.experts_per_node, 1)
        self.assertEqual(plan.expert_nodes, 82432)
        self.assertGreater(plan.capacity_experts_per_node, 1)  # 200MB/19MB

    def test_rsx_widens_capacity_but_not_default_placement(self):
        # RSX (GameOS exploit) adds hot capacity; placement still 1/node unless asked.
        base = T.plan_cluster(T.KIMI_K3)
        rsx = T.plan_cluster(T.KIMI_K3, rsx=True)
        self.assertGreater(rsx.node_capacity_mb, base.node_capacity_mb)
        self.assertGreater(rsx.capacity_experts_per_node, base.capacity_experts_per_node)
        self.assertEqual(rsx.expert_nodes, 82432)  # still 1/node by default
        self.assertTrue(any("GameOS" in w for w in rsx.warnings))

    def test_packing_reduces_node_count(self):
        # Opt into 2 experts/node -> ~half the expert nodes.
        plan = T.plan_cluster(T.KIMI_K3, experts_per_node=2)
        self.assertEqual(plan.experts_per_node, 2)
        self.assertEqual(plan.expert_nodes, (82432 + 1) // 2)

    def test_packing_clamped_to_capacity(self):
        plan = T.plan_cluster(T.KIMI_K3, experts_per_node=9999)
        self.assertEqual(plan.experts_per_node, plan.capacity_experts_per_node)
        self.assertTrue(any("clamping" in w for w in plan.warnings))

    def test_k3_node_count(self):
        plan = T.plan_cluster(T.KIMI_K3)
        # 82,432 expert nodes + 92 layer coordinators + embed/lm_head shards.
        self.assertEqual(plan.expert_nodes, 82432)
        self.assertEqual(plan.layer_nodes, 92)
        self.assertGreaterEqual(plan.io_nodes, 2)
        self.assertEqual(plan.total_nodes,
                         plan.expert_nodes + plan.layer_nodes + plan.io_nodes)

    def test_k3_is_mostly_idle_per_token(self):
        plan = T.plan_cluster(T.KIMI_K3)
        # top-16 x 92 layers = 1472 expert activations; vast majority idle.
        self.assertGreater(plan.idle_fraction, 0.95)
        self.assertLess(plan.active_nodes_per_token, plan.total_nodes)

    def test_big_expert_splits(self):
        big = T.ModelProfile(
            name="big", n_layers=2, experts_per_layer=4, top_k=2,
            hidden_size=4096, expert_bytes=500 * 1024 * 1024,
            resident_bytes_per_layer=10 * 1024 * 1024,
            embed_bytes=100 * 1024 * 1024, lm_head_bytes=100 * 1024 * 1024)
        plan = T.plan_cluster(big)
        self.assertGreater(plan.experts_split_across, 1)
        self.assertTrue(any("exceeds a node" in w for w in plan.warnings))

    def test_placement_table_small(self):
        small = T.ModelProfile(
            name="small", n_layers=2, experts_per_layer=3, top_k=1,
            hidden_size=64, expert_bytes=8 * 1024 * 1024,
            resident_bytes_per_layer=4 * 1024 * 1024,
            embed_bytes=8 * 1024 * 1024, lm_head_bytes=8 * 1024 * 1024)
        table = T.placement_table(small)
        experts = [n for n in table if n.role == "expert"]
        layers = [n for n in table if n.role == "layer"]
        self.assertEqual(len(experts), 6)
        self.assertEqual(len(layers), 2)
        self.assertTrue(all(n.fits for n in table))

    def test_k3_040b_total_experts(self):
        # Layer 0 is dense; layers 1-7 each hold 8 routed experts.
        self.assertEqual(T.KIMI_K3_040B.moe_layer_count, 7)
        self.assertEqual(T.KIMI_K3_040B.total_experts, 7 * 8)
        self.assertEqual(T.KIMI_K3_040B.total_experts, 56)

    def test_k3_040b_expert_fits_one_node(self):
        plan = T.plan_cluster(T.KIMI_K3_040B)
        self.assertEqual(plan.experts_split_across, 1)
        self.assertLessEqual(plan.per_expert_mb, T.PS3_USABLE_RAM_MB)
        self.assertGreater(plan.capacity_experts_per_node, 1)

    def test_k3_040b_default_one_expert_per_node(self):
        plan = T.plan_cluster(T.KIMI_K3_040B)
        self.assertEqual(plan.experts_per_node, 1)
        self.assertEqual(plan.expert_nodes, 56)
        self.assertEqual(plan.layer_nodes, 8)
        self.assertEqual(plan.io_nodes, 2)
        self.assertEqual(plan.total_nodes, 56 + 8 + 2)

    def test_k3_040b_is_not_densely_idle(self):
        # 0.40B is small enough that each token lights up a meaningful fraction.
        plan = T.plan_cluster(T.KIMI_K3_040B)
        # top-2 * 7 MoE layers = 14 routed experts; + 8 layer + 2 I/O = 24 active.
        self.assertEqual(plan.active_nodes_per_token, 2 * 7 + 8 + 2)
        self.assertEqual(plan.active_nodes_per_token, 24)
        self.assertEqual(plan.idle_fraction, 1.0 - 24 / plan.total_nodes)

    def test_k3_040b_placement_table_skips_dense_layer(self):
        table = T.placement_table(T.KIMI_K3_040B)
        experts = [n for n in table if n.role == "expert"]
        layers = [n for n in table if n.role == "layer"]
        self.assertEqual(len(experts), 56)
        self.assertEqual(len(layers), 8)
        self.assertTrue(all(n.fits for n in table))
        # No experts assigned to dense layer 0.
        self.assertFalse(any(n.layer == 0 and n.role == "expert" for n in table))


if __name__ == "__main__":
    unittest.main()
