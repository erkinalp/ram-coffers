"""End-to-end: Python coordinator -> compiled C expert worker -> MXFP4 kernel.

Builds a real .exp file, boots the host-sim build of expert_ppu.c, dispatches an
activation through the same SocketTransport the coordinator uses, and checks the
C MXFP4 SwiGLU output matches the numpy reference derived from the identical
packed bytes. This proves the C node and the Python coordinator interoperate and
that mxfp4.h/p3xc.h agree with protocol.py.

Skipped automatically if the host binary hasn't been built (`make host`).
"""

import os
import sys
import subprocess
import time
import socket
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ps3_cluster.dispatch import ExpertPlacement, SocketTransport, DistributedExpertDispatcher
from ps3_cluster.errors import NodeError
from ps3_cluster.transport import PersistentSocketTransport
import pack_expert

BIN = os.path.join(ROOT, "build", "expert_node_host")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(os.path.exists(BIN), "run `make host` first")
class TestCellKernel(unittest.TestCase):
    def test_c_worker_matches_numpy(self):
        hidden, inter = 128, 256
        layer, expert = 4, 9
        rng = np.random.default_rng(123)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)

        exp_path = os.path.join(ROOT, "build", "test.exp")
        gate_p, up_p, down_p = pack_expert.pack_expert(
            exp_path, g, u, d, layer=layer, expert=expert)

        x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
        ref = pack_expert.reference_forward(gate_p, up_p, down_p, hidden, inter, x)

        port = _free_port()
        proc = subprocess.Popen([BIN, exp_path, str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            # wait for the listener
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)

            placement = ExpertPlacement()
            placement.assign(layer, expert, "cnode")
            transport = SocketTransport({"cnode": ("127.0.0.1", port)})
            disp = DistributedExpertDispatcher(placement, transport)
            out = disp.run_expert_stage(layer, x, [expert], [1.0], token_id=7)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_c_worker_serves_a_persistent_connection(self):
        """The C worker must answer many requests (and PINGs) per connection.

        Same binary as above, driven by the pooled transport: one TCP accept
        carries several REQ frames plus a heartbeat, which is what the
        coordinator does for the lifetime of a run.
        """
        hidden, inter = 64, 128
        layer, expert = 2, 3
        rng = np.random.default_rng(4242)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)

        exp_path = os.path.join(ROOT, "build", "test_persistent.exp")
        gate_p, up_p, down_p = pack_expert.pack_expert(
            exp_path, g, u, d, layer=layer, expert=expert)

        port = _free_port()
        proc = subprocess.Popen([BIN, exp_path, str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transport = PersistentSocketTransport({"cnode": ("127.0.0.1", port)},
                                              timeout=30.0)
        try:
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)

            self.assertGreaterEqual(transport.ping("cnode"), 0.0)
            for token in range(4):
                x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
                ref = pack_expert.reference_forward(gate_p, up_p, down_p,
                                                    hidden, inter, x)
                y = transport.dispatch("cnode", layer, expert, token, x)
                np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
            # One connection carried the heartbeat and all four requests.
            self.assertEqual(transport.connects_opened["cnode"], 1)
            self.assertEqual(transport.requests_sent["cnode"], 4)
        finally:
            transport.close()
            proc.terminate()
            proc.wait(timeout=5)

    def test_c_worker_accepts_a_second_connection_while_one_is_idle(self):
        """An open, idle connection must not block the worker's accept loop.

        The pooled coordinator holds several connections per node open at once,
        so the worker forks a process per connection; with a single-threaded
        accept-and-serve loop this test would hang on the second connection.
        """
        hidden, inter = 32, 64
        layer, expert = 6, 1
        rng = np.random.default_rng(77)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)
        exp_path = os.path.join(ROOT, "build", "test_two_conn.exp")
        gate_p, up_p, down_p = pack_expert.pack_expert(
            exp_path, g, u, d, layer=layer, expert=expert)

        port = _free_port()
        proc = subprocess.Popen([BIN, exp_path, str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transport = PersistentSocketTransport({"cnode": ("127.0.0.1", port)},
                                              timeout=30.0,
                                              max_connections_per_node=2)
        idle = None
        try:
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)

            # A connection that is open but never sends anything.
            idle = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
            ref = pack_expert.reference_forward(gate_p, up_p, down_p, hidden,
                                                inter, x)
            y = transport.dispatch("cnode", layer, expert, 1, x)
            np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
        finally:
            if idle is not None:
                idle.close()
            transport.close()
            proc.terminate()
            proc.wait(timeout=5)

    def test_c_worker_reports_a_wrong_expert_as_err(self):
        hidden, inter = 32, 64
        rng = np.random.default_rng(11)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)
        exp_path = os.path.join(ROOT, "build", "test_err.exp")
        pack_expert.pack_expert(exp_path, g, u, d, layer=1, expert=1)

        port = _free_port()
        proc = subprocess.Popen([BIN, exp_path, str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transport = PersistentSocketTransport({"cnode": ("127.0.0.1", port)},
                                              timeout=30.0)
        try:
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            with self.assertRaises(NodeError) as ctx:
                transport.dispatch("cnode", 1, 2, 9,
                                   np.ones(hidden, np.float32))
            self.assertEqual(ctx.exception.node_id, "cnode")
            # The connection survives the ERR and still serves requests.
            self.assertGreaterEqual(transport.ping("cnode"), 0.0)
            self.assertEqual(transport.connects_opened["cnode"], 1)
        finally:
            transport.close()
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
