"""The documented console launcher really serves an .exp file.

`tools/run_expert.py` is the numpy stand-in for `build/expert_node_host`, and it
is what the deployment docs tell an operator to run when there is no Cell
toolchain, so it has to load the same packed weights the C worker reads and
answer real P3XC frames.
"""

import os
import socket
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

from ps3_cluster.errors import TransportError  # noqa: E402
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402
import pack_expert  # noqa: E402
import run_expert  # noqa: E402

CLI = os.path.join(ROOT, "tools", "run_expert.py")


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestWorkerCli(unittest.TestCase):
    def _packed(self, path, hidden=64, inter=64, layer=3, expert=1):
        rng = np.random.default_rng(7)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)
        packed = pack_expert.pack_expert(path, g, u, d, layer, expert)
        return packed, hidden, inter

    def test_load_exp_recovers_the_header_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.exp")
            (gp, up, dp), hidden, inter = self._packed(path)
            layer, expert, h, i, forward = run_expert.load_exp(path)
            self.assertEqual((layer, expert, h, i), (3, 1, hidden, inter))
            x = np.arange(hidden, dtype=np.float32) * 0.01
            np.testing.assert_array_equal(
                forward(x),
                pack_expert.reference_forward(gp, up, dp, hidden, inter, x))

    def test_cli_serves_the_expert_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.exp")
            (gp, up, dp), hidden, inter = self._packed(path)
            port = _free_port()
            proc = subprocess.Popen(
                [sys.executable, CLI, path, "--host", "127.0.0.1",
                 "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            transport = PersistentSocketTransport({"c": ("127.0.0.1", port)},
                                                  timeout=10.0)
            try:
                x = np.arange(hidden, dtype=np.float32) * 0.01
                deadline = time.time() + 15
                while True:
                    try:
                        y = transport.dispatch("c", 3, 1, 5, x)
                        break
                    except TransportError:
                        if time.time() > deadline:
                            raise
                        time.sleep(0.1)
                np.testing.assert_allclose(
                    y, pack_expert.reference_forward(gp, up, dp, hidden,
                                                     inter, x),
                    rtol=1e-5, atol=1e-5)
                # Same connection, second request: the CLI keeps serving.
                transport.dispatch("c", 3, 1, 6, x)
                self.assertGreater(transport.ping("c"), 0.0)
            finally:
                transport.close()
                proc.terminate()
                self.assertEqual(proc.wait(20), 0)


if __name__ == "__main__":
    unittest.main()
