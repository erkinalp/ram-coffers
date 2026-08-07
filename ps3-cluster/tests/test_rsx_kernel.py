"""RSX shader backend: validate the Cg fragment-shader algorithm on x86.

There is no RSX GPU or toolchain here, so we exercise `expert_node_rsxemu`, the
worker built with -DGEMV_RSX_EMU, whose gemv() runs rsx/rsx_gemv_emu.h -- a
faithful CPU model of expert_rsx.cg's per-fragment math (R8 byte normalise/
recover, 16-entry LUT lookup, E8M0 exp2 scale, nibble unpack). If the shader
formulation were wrong, this would diverge from the numpy reference.

Skipped if the binary hasn't been built (`make host`).
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

BIN = os.path.join(ROOT, "build", "expert_node_rsxemu")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(os.path.exists(BIN), "run `make host` first")
class TestRsxKernel(unittest.TestCase):
    def test_rsx_shader_emu_matches_numpy(self):
        hidden, inter = 96, 128
        layer, expert = 2, 3
        rng = np.random.default_rng(2024)
        g = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        u = (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float32)
        d = (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float32)

        exp_path = os.path.join(ROOT, "build", "test_rsx.exp")
        gate_p, up_p, down_p = pack_expert.pack_expert(
            exp_path, g, u, d, layer=layer, expert=expert)

        x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
        ref = pack_expert.reference_forward(gate_p, up_p, down_p, hidden, inter, x)

        port = _free_port()
        proc = subprocess.Popen([BIN, exp_path, str(port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)

            placement = ExpertPlacement()
            placement.assign(layer, expert, "rsxnode")
            transport = SocketTransport({"rsxnode": ("127.0.0.1", port)})
            disp = DistributedExpertDispatcher(placement, transport)
            out = disp.run_expert_stage(layer, x, [expert], [1.0], token_id=1)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

        # Shader math uses fp32 throughout; allow a slightly looser tolerance.
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
