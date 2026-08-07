"""Cluster config: declared subcluster membership drives both ends.

The config is the single source of truth for which head server fronts which
consoles, so these tests check that the derived placement stays canonical (one
expert x one layer / node, replicas additive) and that the derived plan reflects
the *declared* grouping rather than an ordering.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.deployment import ClusterConfig  # noqa: E402
from ps3_cluster.subcluster import DEFAULT_SUBCLUSTER_SIZE  # noqa: E402

DOC = {
    "subcluster_size": 2,
    "subclusters": [
        {"id": "sc-0000", "host": "0.0.0.0", "port": 8100, "members": [
            {"layer": 3, "expert": 0, "node": "ps3-L003-E0000",
             "host": "10.0.0.10", "port": 9000,
             "replicas": [{"node": "ps3-L003-E0000-b", "host": "10.0.0.210",
                           "port": 9000}]},
            {"layer": 3, "expert": 1, "node": "ps3-L003-E0001",
             "host": "10.0.0.11", "port": 9000},
        ]},
        {"id": "sc-0001", "host": "0.0.0.0", "port": 8101, "members": [
            {"layer": 3, "expert": 2, "node": "ps3-L003-E0002",
             "host": "10.0.0.12", "port": 9000},
        ]},
    ],
}


class TestClusterConfig(unittest.TestCase):
    def setUp(self):
        self.config = ClusterConfig.from_dict(DOC)

    def test_groups_and_members(self):
        self.assertEqual(self.config.group_ids(), ["sc-0000", "sc-0001"])
        self.assertEqual(len(self.config.members()), 3)
        self.assertEqual([m.expert for m in self.config.members("sc-0000")],
                         [0, 1])
        self.assertEqual(self.config.layers(), [3])

    def test_placement_is_canonical_with_additive_replicas(self):
        placement = self.config.placement()
        self.assertEqual(len(placement), 3)  # replicas are not extra experts
        self.assertEqual(placement.node_for(3, 0), "ps3-L003-E0000")
        self.assertEqual(placement.replicas_for(3, 0), ["ps3-L003-E0000-b"])
        self.assertEqual(placement.replicas_for(3, 1), [])

    def test_endpoints(self):
        self.assertEqual(self.config.group_endpoints(),
                         {"sc-0000": [("0.0.0.0", 8100)],
                          "sc-0001": [("0.0.0.0", 8101)]})
        endpoints = self.config.expert_endpoints("sc-0000")
        self.assertEqual(endpoints["ps3-L003-E0001"], ("10.0.0.11", 9000))
        self.assertIn("ps3-L003-E0000-b", endpoints)  # replica reachable
        self.assertNotIn("ps3-L003-E0002", endpoints)  # other subcluster's

    def test_plan_follows_declared_grouping(self):
        plan = self.config.plan()
        self.assertEqual(plan.groups(),
                         {"sc-0000": ["ps3-L003-E0000", "ps3-L003-E0001"],
                          "sc-0001": ["ps3-L003-E0002"]})
        self.assertEqual(plan.subcluster_of("ps3-L003-E0002"), "sc-0001")

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cluster.json")
            self.config.save(path)
            reloaded = ClusterConfig.load(path)
            self.assertEqual(reloaded.to_dict(), self.config.to_dict())
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["subcluster_size"], 2)

    def test_oversized_subcluster_rejected(self):
        doc = json.loads(json.dumps(DOC))
        doc["subcluster_size"] = 1
        with self.assertRaises(ValueError):
            ClusterConfig.from_dict(doc)

    def test_duplicate_node_rejected(self):
        doc = json.loads(json.dumps(DOC))
        doc["subclusters"][1]["members"][0]["node"] = "ps3-L003-E0000"
        with self.assertRaises(ValueError):
            ClusterConfig.from_dict(doc)

    def test_empty_config_rejected(self):
        with self.assertRaises(ValueError):
            ClusterConfig.from_dict({"subclusters": []})


REGION_DOC = {
    "subcluster_size": 1,
    "regions": [
        {"id": "rg-0000", "host": "0.0.0.0", "port": 8200,
         "standby": [{"host": "0.0.0.0", "port": 8300}],
         "subclusters": ["sc-0000", "sc-0001"]},
        {"id": "rg-0001", "host": "0.0.0.0", "port": 8201,
         "subclusters": ["sc-0002"]},
    ],
    "subclusters": [
        {"id": f"sc-{i:04d}", "host": "0.0.0.0", "port": 8100 + i,
         **({"standby": [{"host": "0.0.0.0", "port": 8150 + i}]}
            if i == 0 else {}),
         "members": [{"layer": 3, "expert": i, "node": f"ps3-L003-E{i:04d}",
                      "host": "10.0.0.10", "port": 9000 + i}]}
        for i in range(3)
    ],
}


class TestThreeTierConfig(unittest.TestCase):
    """Regions, ordered endpoint lists, and the plan each tier runs on."""

    def setUp(self):
        self.config = ClusterConfig.from_dict(REGION_DOC)

    def test_regions_and_their_endpoint_lists(self):
        self.assertEqual(self.config.region_ids(), ["rg-0000", "rg-0001"])
        self.assertEqual(self.config.region_endpoints(),
                         {"rg-0000": [("0.0.0.0", 8200), ("0.0.0.0", 8300)],
                          "rg-0001": [("0.0.0.0", 8201)]})
        self.assertEqual(self.config.region("rg-0000").subclusters,
                         ("sc-0000", "sc-0001"))
        with self.assertRaises(KeyError):
            self.config.region("rg-0002")

    def test_a_region_sees_only_its_own_heads(self):
        self.assertEqual(self.config.group_endpoints("rg-0000"),
                         {"sc-0000": [("0.0.0.0", 8100), ("0.0.0.0", 8150)],
                          "sc-0001": [("0.0.0.0", 8101)]})
        self.assertEqual(list(self.config.group_endpoints("rg-0001")),
                         ["sc-0002"])
        self.assertEqual(self.config.plan("rg-0001").group_ids(), ["sc-0002"])
        self.assertEqual([m.expert for m in
                          self.config.region_members("rg-0000")], [0, 1])

    def test_tiered_plan_maps_consoles_to_regions(self):
        tiered = self.config.tiered_plan()
        self.assertEqual(tiered.group_ids(), ["rg-0000", "rg-0001"])
        self.assertEqual(tiered.subcluster_of("ps3-L003-E0001"), "rg-0000")
        self.assertEqual(tiered.subcluster_of("ps3-L003-E0002"), "rg-0001")

    def test_a_config_without_regions_has_no_tiered_plan(self):
        two_tier = ClusterConfig.from_dict(DOC)
        self.assertEqual(two_tier.region_ids(), [])
        self.assertEqual(two_tier.region_endpoints(), {})
        with self.assertRaises(ValueError):
            two_tier.tiered_plan()

    def test_json_roundtrip_keeps_regions_and_standby_addresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cluster.json")
            self.config.save(path)
            reloaded = ClusterConfig.load(path)
        self.assertEqual(reloaded.to_dict(), self.config.to_dict())
        self.assertEqual(reloaded.region_endpoints(),
                         self.config.region_endpoints())
        self.assertEqual(reloaded.to_dict()["regions"][1].get("standby"), None)

    def test_with_regions_deals_heads_out_contiguously(self):
        config = ClusterConfig.from_dict(DOC).with_regions(2, port_base=9200)
        self.assertEqual(config.region_endpoints(),
                         {"rg-0000": [("0.0.0.0", 9200)],
                          "rg-0001": [("0.0.0.0", 9201)]})
        self.assertEqual(config.region("rg-0000").subclusters, ("sc-0000",))
        one = ClusterConfig.from_dict(DOC).with_regions(1)
        self.assertEqual(one.region("rg-0000").subclusters,
                         ("sc-0000", "sc-0001"))
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                ClusterConfig.from_dict(DOC).with_regions(bad)


class TestForLayer(unittest.TestCase):
    def test_default_size_is_condors_22(self):
        consoles = [(e, f"ps3-L000-E{e:04d}", ("127.0.0.1", 9000 + e))
                    for e in range(46)]
        config = ClusterConfig.for_layer(0, consoles)
        self.assertEqual(config.subcluster_size, DEFAULT_SUBCLUSTER_SIZE)
        self.assertEqual([len(s.members) for s in config.subclusters],
                         [22, 22, 2])
        self.assertEqual(config.group_ids(),
                         ["sc-0000", "sc-0001", "sc-0002"])

    def test_head_ports_are_numbered(self):
        consoles = [(e, f"ps3-L001-E{e:04d}", ("127.0.0.1", 9000 + e))
                    for e in range(4)]
        config = ClusterConfig.for_layer(1, consoles, size=2,
                                         head_host="10.1.1.1",
                                         head_port_base=8200)
        self.assertEqual(config.group_endpoints(),
                         {"sc-0000": [("10.1.1.1", 8200)],
                          "sc-0001": [("10.1.1.1", 8201)]})
        self.assertEqual(config.plan().subcluster_of("ps3-L001-E0003"),
                         "sc-0001")


if __name__ == "__main__":
    unittest.main()
