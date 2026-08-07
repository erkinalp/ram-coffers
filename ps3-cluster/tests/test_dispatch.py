import os
import sys
import threading
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.dispatch import (ExpertPlacement, LoopbackTransport,
                                  SocketTransport, DistributedExpertDispatcher)
from ps3_cluster.node import ExpertNode, serve


def make_expert(seed):
    """A deterministic linear 'expert': y = W @ x, W fixed by seed."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((4, 4)).astype(np.float32)

    def fn(x):
        return (W @ x.astype(np.float32))
    return fn, W


class TestDispatch(unittest.TestCase):
    def test_placement_one_per_node(self):
        p = ExpertPlacement.one_per_node(2, 3)
        self.assertEqual(len(p), 6)
        self.assertEqual(p.node_for(1, 2), "ps3-L001-E0002")
        self.assertEqual(p.expert_on("ps3-L000-E0000"), (0, 0))

    def test_loopback_combines_topk(self):
        # Two experts on two nodes in layer 0; combine with gate weights.
        fn0, W0 = make_expert(0)
        fn1, W1 = make_expert(1)
        placement = ExpertPlacement()
        placement.assign(0, 0, "n0")
        placement.assign(0, 1, "n1")
        transport = LoopbackTransport({"n0": fn0, "n1": fn1})
        disp = DistributedExpertDispatcher(placement, transport)

        x = np.arange(4, dtype=np.float32)
        gates = [0.7, 0.3]
        out = disp.run_expert_stage(0, x, [0, 1], gates)
        expected = 0.7 * (W0 @ x) + 0.3 * (W1 @ x)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_only_selected_experts_dispatched(self):
        # Layer with 4 experts, route to only 2: exactly 2 dispatches fire.
        experts = {f"n{i}": make_expert(i)[0] for i in range(4)}
        placement = ExpertPlacement()
        for i in range(4):
            placement.assign(0, i, f"n{i}")
        transport = LoopbackTransport(experts)
        disp = DistributedExpertDispatcher(placement, transport)
        disp.run_expert_stage(0, np.ones(4, np.float32), [1, 3], [0.5, 0.5])
        dispatched = [d[0] for d in transport.dispatch_log]
        self.assertEqual(sorted(dispatched), ["n1", "n3"])
        self.assertNotIn("n0", dispatched)
        self.assertNotIn("n2", dispatched)

    def test_end_to_end_over_tcp(self):
        # Spin up a real expert node on localhost and dispatch through a socket.
        fn, W = make_expert(7)
        node = ExpertNode(layer=3, expert=5, expert_fn=fn)
        server = serve("127.0.0.1", 0, node)
        host, port = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            placement = ExpertPlacement()
            placement.assign(3, 5, "remote")
            transport = SocketTransport({"remote": (host, port)})
            disp = DistributedExpertDispatcher(placement, transport)
            x = np.array([1, 2, 3, 4], dtype=np.float32)
            out = disp.run_expert_stage(3, x, [5], [1.0], token_id=99)
            np.testing.assert_allclose(out, W @ x, rtol=1e-5, atol=1e-5)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
