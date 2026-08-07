"""The documented three-tier bring-up, run as real processes.

`test_three_tier.py` exercises the coordinators in-process; this file checks that
the CLIs the README tells an operator to run actually compose into a farm:
generated config -> console workers -> head servers -> regional coordinators ->
a layer client whose output is bit-identical to the flat dispatcher's. Every
process is a subprocess, and each one's stdout pipe is closed on the way out so a
leak fails the suite as a ResourceWarning.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

from ps3_cluster.deployment import ClusterConfig  # noqa: E402
from ps3_cluster.dispatch import DistributedExpertDispatcher  # noqa: E402
from ps3_cluster.errors import TransportError  # noqa: E402
from ps3_cluster.hierarchy import (HierarchicalExpertDispatcher,  # noqa: E402
                                   SubclusterTransport)
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402
from _netutil import free_port  # noqa: E402
import pack_expert  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
LAYER = 3
HIDDEN = 64
INTER = 32
N_EXPERTS = 4


class _Processes:
    """Subprocesses that must all be reaped, pipes included."""

    def __init__(self):
        self.procs = []

    def start(self, args):
        proc = subprocess.Popen([sys.executable] + args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        self.procs.append(proc)
        return proc

    def close(self):
        for proc in reversed(self.procs):
            proc.terminate()
            try:
                proc.wait(20)
            finally:
                if proc.stdout is not None:
                    proc.stdout.close()
        self.procs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _wait_for(transport, group_id, deadline=25.0):
    """PING a coordinator until it answers, so no test races a bind()."""
    end = time.time() + deadline
    while True:
        try:
            transport.ping(group_id)
            return
        except TransportError:
            if time.time() > end:
                raise
            time.sleep(0.1)


class TestThreeTierDeploymentCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rng = np.random.default_rng(11)
        self.weights = {}
        self.expert_ports = [free_port() for _ in range(N_EXPERTS)]
        for expert in range(N_EXPERTS):
            path = os.path.join(self.tmp.name, f"e{expert}.exp")
            g = (self.rng.standard_normal((INTER, HIDDEN))
                 * 0.1).astype(np.float32)
            u = (self.rng.standard_normal((INTER, HIDDEN))
                 * 0.1).astype(np.float32)
            d = (self.rng.standard_normal((HIDDEN, INTER))
                 * 0.1).astype(np.float32)
            self.weights[expert] = (
                path, pack_expert.pack_expert(path, g, u, d, LAYER, expert))

    def _config(self, standby=True):
        """Generate a 2-region / 4-head / 4-console config through the CLI."""
        path = os.path.join(self.tmp.name, "cluster.json")
        head_base = free_port()
        region_base = free_port()
        out = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "gen_cluster_config.py"),
             "--layer", str(LAYER), "--experts", str(N_EXPERTS),
             "--size", "1", "--expert-host", "127.0.0.1",
             "--head-host", "127.0.0.1", "--head-port-base", str(head_base),
             "--regions", "2", "--region-host", "127.0.0.1",
             "--region-port-base", str(region_base)]
            + (["--region-standby", "1"] if standby else [])
            + ["-o", path],
            capture_output=True, text=True, check=True)
        self.assertIn("2 regions", out.stdout)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for spec, port in zip(doc["subclusters"], self.expert_ports):
            spec["members"][0]["port"] = port
        if standby:
            self.assertEqual([[{"host": "127.0.0.1",
                                "port": region_base + 2 + index}]
                              for index in range(2)],
                             [r["standby"] for r in doc["regions"]])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        return path, ClusterConfig.from_dict(doc)

    def _bring_up(self, procs, path, config, region_args=()):
        for expert, port in enumerate(self.expert_ports):
            procs.start([os.path.join(TOOLS, "run_expert.py"),
                         self.weights[expert][0], "--host", "127.0.0.1",
                         "--port", str(port)])
        for spec in config.subclusters:
            procs.start([os.path.join(TOOLS, "run_subcluster.py"),
                         "--config", path, "--subcluster", spec.group_id])
        for spec in config.regions:
            procs.start([os.path.join(TOOLS, "run_region.py"),
                         "--config", path, "--region", spec.region_id]
                        + list(region_args))

    def _flat_reference(self, config, x, experts, gates):
        """The one-tier path over the same consoles, for comparison."""
        transport = PersistentSocketTransport(config.expert_endpoints(),
                                              timeout=20.0)
        dispatcher = DistributedExpertDispatcher(config.placement(), transport)
        try:
            for node_id in config.placement().node_ids(LAYER):
                _wait_for(transport, node_id)
            return dispatcher.run_expert_stage(LAYER, x, experts, gates,
                                               token_id=99)
        finally:
            dispatcher.close()
            transport.close()

    def test_generated_config_deploys_and_matches_the_flat_dispatcher(self):
        path, config = self._config()
        x = (self.rng.standard_normal(HIDDEN) * 0.5).astype(np.float32)
        experts = [3, 0, 2]                      # interleaved across regions
        gates = [np.float32(0.5), np.float32(0.25), np.float32(0.125)]
        with _Processes() as procs:
            self._bring_up(procs, path, config)
            expected = self._flat_reference(config, x, experts, gates)
            transport = SubclusterTransport(config.region_endpoints(),
                                            timeout=20.0)
            dispatcher = HierarchicalExpertDispatcher(
                config.placement(), config.tiered_plan(), transport)
            try:
                for region_id in config.region_ids():
                    _wait_for(transport, region_id)
                got = dispatcher.run_expert_stage(LAYER, x, experts, gates,
                                                  token_id=7)
                self.assertTrue(np.array_equal(got, expected))
                self.assertEqual(sorted(dispatcher.check_liveness()),
                                 config.region_ids())
            finally:
                dispatcher.close()
                transport.close()

    def test_a_region_standby_process_serves_the_same_config(self):
        path, config = self._config()
        x = np.ones(HIDDEN, dtype=np.float32)
        experts, gates = [0, 1], [np.float32(0.5), np.float32(0.5)]
        with _Processes() as procs:
            self._bring_up(procs, path, config)
            region = config.regions[0]
            procs.start([os.path.join(TOOLS, "run_region.py"),
                         "--config", path, "--region", region.region_id,
                         "--standby", "0"])
            expected = self._flat_reference(config, x, experts, gates)
            # Point the layer at the standby address only.
            transport = SubclusterTransport(
                {region.region_id: [region.standby[0]]}, timeout=20.0)
            dispatcher = HierarchicalExpertDispatcher(
                config.placement(), config.tiered_plan(), transport)
            try:
                _wait_for(transport, region.region_id)
                got = dispatcher.run_expert_stage(LAYER, x, experts, gates,
                                                  token_id=8)
                self.assertTrue(np.array_equal(got, expected))
            finally:
                dispatcher.close()
                transport.close()

    def test_list_and_check_members_report_the_tree(self):
        path, config = self._config(standby=False)
        listed = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "run_region.py"),
             "--config", path, "--list"],
            capture_output=True, text=True, check=True)
        for region_id in config.region_ids():
            self.assertIn(region_id, listed.stdout)
        with _Processes() as procs:
            self._bring_up(procs, path, config)
            transport = SubclusterTransport(config.region_endpoints(),
                                            timeout=20.0)
            try:
                for region_id in config.region_ids():
                    _wait_for(transport, region_id)
            finally:
                transport.close()
            checked = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "run_region.py"),
                 "--config", path, "--region", config.region_ids()[0],
                 "--check-members"],
                capture_output=True, text=True, check=True)
            self.assertEqual(
                json.loads(checked.stdout),
                {g: True for g in config.region("rg-0000").subclusters})

    def test_run_layer_cli_matches_flat_dispatcher_three_tier(self):
        path, config = self._config()
        x = (self.rng.standard_normal(HIDDEN) * 0.5).astype(np.float32)
        experts = [3, 0, 2]
        gates = [np.float32(0.5), np.float32(0.25), np.float32(0.125)]
        x_path = os.path.join(self.tmp.name, "x.npy")
        out_path = os.path.join(self.tmp.name, "out.npy")
        np.save(x_path, x)
        with _Processes() as procs:
            self._bring_up(procs, path, config)
            expected = self._flat_reference(config, x, experts, gates)
            subprocess.run(
                [sys.executable, os.path.join(TOOLS, "run_layer.py"),
                 "--config", path, "--layer", str(LAYER), "--token", "7"]
                + ["--experts"] + [str(e) for e in experts]
                + ["--gates"] + [str(g) for g in gates]
                + ["--activation", x_path, "--output", out_path,
                   "--timeout", "20"],
                capture_output=True, text=True, check=True)
            got = np.load(out_path)
            self.assertTrue(np.array_equal(got, expected))

    def test_run_layer_cli_matches_flat_dispatcher_two_tier(self):
        path = os.path.join(self.tmp.name, "two-tier.json")
        subprocess.run(
            [sys.executable, os.path.join(TOOLS, "gen_cluster_config.py"),
             "--layer", str(LAYER), "--experts", str(N_EXPERTS),
             "--size", "1", "--expert-host", "127.0.0.1",
             "--head-host", "127.0.0.1", "--head-port-base", str(free_port()),
             "-o", path], capture_output=True, text=True, check=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for spec, port in zip(doc["subclusters"], self.expert_ports):
            spec["members"][0]["port"] = port
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        config = ClusterConfig.from_dict(doc)
        x = np.ones(HIDDEN, dtype=np.float32)
        experts, gates = [0, 1, 3], [0.5, 0.25, 0.125]
        x_path = os.path.join(self.tmp.name, "x2.npy")
        out_path = os.path.join(self.tmp.name, "out2.npy")
        np.save(x_path, x)
        with _Processes() as procs:
            for expert, port in enumerate(self.expert_ports):
                procs.start([os.path.join(TOOLS, "run_expert.py"),
                             self.weights[expert][0], "--host", "127.0.0.1",
                             "--port", str(port)])
            for spec in config.subclusters:
                procs.start([os.path.join(TOOLS, "run_subcluster.py"),
                             "--config", path, "--subcluster", spec.group_id])
            expected = self._flat_reference(config, x, experts, gates)
            subprocess.run(
                [sys.executable, os.path.join(TOOLS, "run_layer.py"),
                 "--config", path, "--layer", str(LAYER), "--token", "9"]
                + ["--experts"] + [str(e) for e in experts]
                + ["--gates"] + [str(g) for g in gates]
                + ["--activation", x_path, "--output", out_path,
                   "--timeout", "20"],
                capture_output=True, text=True, check=True)
            got = np.load(out_path)
            self.assertTrue(np.array_equal(got, expected))

    def test_run_layer_cli_ping_reports_coordinators(self):
        path, config = self._config()
        x_path = os.path.join(self.tmp.name, "x3.npy")
        out_path = os.path.join(self.tmp.name, "out3.npy")
        np.save(x_path, np.ones(HIDDEN, dtype=np.float32))
        with _Processes() as procs:
            self._bring_up(procs, path, config)
            result = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "run_layer.py"),
                 "--config", path, "--layer", str(LAYER), "--token", "1",
                 "--experts", "0", "--gates", "1.0",
                 "--activation", x_path, "--output", out_path, "--ping",
                 "--timeout", "10"],
                capture_output=True, text=True, check=True)
            for region_id in config.region_ids():
                self.assertIn(region_id, result.stdout)

    def test_a_two_tier_config_refuses_to_serve_a_region(self):
        path = os.path.join(self.tmp.name, "two-tier.json")
        subprocess.run(
            [sys.executable, os.path.join(TOOLS, "gen_cluster_config.py"),
             "--layer", str(LAYER), "--experts", "2", "--size", "1",
             "-o", path], capture_output=True, text=True, check=True)
        failed = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "run_region.py"),
             "--config", path, "--region", "rg-0000"],
            capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("declares no regions", failed.stderr)


if __name__ == "__main__":
    unittest.main()
