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


if __name__ == "__main__":
    unittest.main()
