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


if __name__ == "__main__":
    unittest.main()
