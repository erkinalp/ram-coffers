"""Deployed subcluster hierarchy over real sockets.

Every test here runs the actual three-tier topology on loopback: expert workers,
one ``SubclusterService`` process-equivalent per subcluster (a real TCP server
with its own pooled connections to its consoles), and a layer-side hierarchical
dispatcher talking batch frames to those services. Concurrency is proven with
barriers, never with latency thresholds.
"""

import os
import socket
import struct
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _netutil import (FrameTap, RunningNode, dead_endpoint,  # noqa: E402
                      linear_expert, reader_threads)
from ps3_cluster import batch as B  # noqa: E402
from ps3_cluster import protocol as P  # noqa: E402
from ps3_cluster.coordinator import (SubclusterCoordinator,  # noqa: E402
                                     SubclusterService)
from ps3_cluster.deployment import (ClusterConfig, MemberSpec,  # noqa: E402
                                    ReplicaSpec, SubclusterSpec)
from ps3_cluster.dispatch import (DistributedExpertDispatcher,  # noqa: E402
                                  RetryPolicy)
from ps3_cluster.errors import SubclusterError  # noqa: E402
from ps3_cluster.hierarchy import (HierarchicalExpertDispatcher,  # noqa: E402
                                   PendingBatch, SubclusterTransport)
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402

LAYER = 3
DIM = 4
GATES = [0.5, 0.25, 0.125, 0.0625]


def node_id(expert):
    return f"ps3-L{LAYER:03d}-E{expert:04d}"


class _Farm:
    """Expert workers + one live coordinator service per subcluster."""

    def __init__(self, n_experts=4, size=2, gate_fn=None, timeout=10.0,
                 max_connections_per_group=4, tap=False,
                 upstream_workers=16, retry_policy=None, dim=DIM,
                 fast=False, allow_fast=True, wrap=None):
        self.dim = dim
        self.nodes = []
        self.weights = []
        for expert in range(n_experts):
            fn, W = linear_expert(seed=100 + expert, dim=dim)
            self.weights.append(W)
            if wrap is not None:      # per-expert wrapper, for ordering tests
                served = wrap(expert, fn)
            else:
                served = gate_fn(fn) if gate_fn else fn
            self.nodes.append(RunningNode(LAYER, expert, served))
        consoles = [(expert, node_id(expert), node.endpoint)
                    for expert, node in enumerate(self.nodes)]
        self.config = ClusterConfig.for_layer(LAYER, consoles, size=size,
                                              head_host="127.0.0.1")
        self.services = []
        self.taps = []
        endpoints = {}
        for spec in self.config.subclusters:
            coordinator = SubclusterCoordinator(
                spec.group_id, self.config.placement(spec.group_id),
                self.config.expert_endpoints(spec.group_id), timeout=timeout,
                retry_policy=retry_policy, allow_fast=allow_fast)
            service = SubclusterService(coordinator, host="127.0.0.1", port=0,
                                        upstream_workers=upstream_workers)
            service.start()
            self.services.append(service)
            if tap:
                proxy = FrameTap(service.address)
                self.taps.append(proxy)
                endpoints[spec.group_id] = proxy.endpoint
            else:
                endpoints[spec.group_id] = service.address
        self.transport = SubclusterTransport(
            endpoints, timeout=timeout,
            max_connections_per_group=max_connections_per_group)
        self.plan = self.config.plan()
        self.placement = self.config.placement()
        self.dispatcher = HierarchicalExpertDispatcher(
            self.placement, self.plan, self.transport, fast=fast)

    # -- helpers -----------------------------------------------------------
    def run(self, x, expert_ids, gates, token_id=1, timeout=None, fast=None):
        return self.dispatcher.run_expert_stage(LAYER, x, expert_ids, gates,
                                                token_id, timeout, fast=fast)

    def flat_reference(self, x, expert_ids, gates, token_id=99, plan=None):
        """The flat dispatcher's own answer over the same consoles."""
        flat, transport = self.flat_dispatcher(plan=plan)
        try:
            return flat.run_expert_stage(LAYER, x, expert_ids, gates,
                                         token_id)
        finally:
            flat.close()
            transport.close()

    def flat_dispatcher(self, plan=None):
        """A flat dispatcher over the same consoles, for reference results."""
        transport = PersistentSocketTransport(self.config.expert_endpoints(),
                                              timeout=10.0)
        return DistributedExpertDispatcher(self.placement, transport,
                                           subclusters=plan), transport

    def service_for(self, group_id):
        for service in self.services:
            if service.group_id == group_id:
                return service
        raise KeyError(group_id)

    def close(self):
        self.dispatcher.close()
        self.transport.close()
        for proxy in self.taps:
            proxy.close()
        for service in self.services:
            service.stop()
        for node in self.nodes:
            node.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestHierarchicalResults(unittest.TestCase):
    def test_two_subclusters_serve_one_stage(self):
        x = np.arange(DIM, dtype=np.float32) + 1.0
        with _Farm(n_experts=4, size=2) as farm:
            out = farm.run(x, [0, 1, 2, 3], GATES)
            expected = sum(g * (W @ x) for g, W in zip(GATES, farm.weights))
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
            self.assertEqual([node.calls for node in farm.nodes], [1] * 4)
            self.assertEqual([s.coordinator.batches_served
                              for s in farm.services], [1, 1])

    def test_bit_identical_to_flat_dispatch_across_subclusters(self):
        """Default mode reproduces the flat reduction exactly, not closely."""
        x = np.linspace(-1.0, 1.0, DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2) as farm:
            hier = farm.run(x, [0, 1, 2, 3], GATES, token_id=7)
            np.testing.assert_array_equal(
                hier, farm.flat_reference(x, [0, 1, 2, 3], GATES))

    def test_fast_mode_matches_the_grouped_in_process_reduction(self):
        """Opt-in fast mode is the old association: partial per subcluster."""
        x = np.linspace(-1.0, 1.0, DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2, fast=True) as farm:
            hier = farm.run(x, [0, 1, 2, 3], GATES, token_id=7)
            np.testing.assert_array_equal(
                hier, farm.flat_reference(x, [0, 1, 2, 3], GATES,
                                          plan=farm.plan))

    def test_single_subcluster_is_bit_identical_to_flat_dispatch(self):
        """One group covers the stage, so the association matches flat sum."""
        x = np.linspace(2.0, -3.0, DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=4) as farm:
            self.assertEqual(len(farm.plan), 1)
            hier = farm.run(x, [0, 1, 2, 3], GATES, token_id=11)
            np.testing.assert_array_equal(
                hier, farm.flat_reference(x, [0, 1, 2, 3], GATES))

    def test_partial_selection_only_touches_involved_subclusters(self):
        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2) as farm:
            out = farm.run(x, [2, 3], [0.5, 0.5])
            expected = 0.5 * (farm.weights[2] @ x) + 0.5 * (farm.weights[3] @ x)
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
            served = {s.group_id: s.coordinator.batches_served
                      for s in farm.services}
            self.assertEqual(served, {"sc-0000": 0, "sc-0001": 1})
            self.assertEqual([node.calls for node in farm.nodes], [0, 0, 1, 1])

    def test_grouping_of_a_selection(self):
        with _Farm(n_experts=4, size=2) as farm:
            self.assertEqual(farm.dispatcher.group_selection(LAYER, [3, 0, 2]),
                             [("sc-0000", [1]), ("sc-0001", [0, 2])])


class TestExactReduction(unittest.TestCase):
    """Default mode must equal the flat dispatcher bit for bit.

    The flat dispatcher is the reference, not a re-derived grouped sum: these
    compare against ``DistributedExpertDispatcher`` output over the same
    consoles, with routings whose top-k positions interleave across subclusters.
    """

    def test_randomized_interleaved_routings_are_bit_identical(self):
        rng = np.random.default_rng(20260807)
        with _Farm(n_experts=8, size=3, dim=8) as farm:
            self.assertEqual(len(farm.plan), 3)
            for trial in range(12):
                k = int(rng.integers(2, 8))
                experts = [int(e) for e in rng.permutation(8)[:k]]
                gates = [float(g) for g in
                         rng.standard_normal(k).astype(np.float32)]
                x = rng.standard_normal(8).astype(np.float32)
                groups = farm.dispatcher.group_selection(LAYER, experts)
                hier = farm.run(x, experts, gates, token_id=100 + trial)
                flat = farm.flat_reference(x, experts, gates,
                                           token_id=200 + trial)
                np.testing.assert_array_equal(
                    hier, flat,
                    err_msg=f"experts={experts} groups={groups}")
            # At least one trial really did interleave positions across groups.
            interleaved = farm.dispatcher.group_selection(LAYER,
                                                          [0, 3, 1, 4, 2])
            self.assertGreater(len(interleaved), 1)
            self.assertNotEqual(
                sorted(p for _g, ps in interleaved for p in ps),
                [p for _g, ps in interleaved for p in ps])

    def test_out_of_order_completion_does_not_change_the_result(self):
        """Experts finish in reverse position order; the sum must not care."""
        done = [threading.Event() for _ in range(4)]

        def wrap(expert, fn):
            def served(x):
                if expert + 1 < len(done):
                    # Wait for the *later* expert to finish first.
                    self.assertTrue(done[expert + 1].wait(15))
                y = fn(x)
                done[expert].set()
                return y
            return served

        x = np.linspace(-2.0, 3.0, DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2, wrap=wrap) as farm:
            hier = farm.run(x, [0, 1, 2, 3], GATES, token_id=31, timeout=20)
            self.assertTrue(all(event.is_set() for event in done))
            np.testing.assert_array_equal(
                hier, farm.flat_reference(x, [0, 1, 2, 3], GATES))

    def test_replica_failover_stays_bit_identical(self):
        """A failed-over contribution lands at its own position, unscaled."""
        fn0, _W0 = linear_expert(seed=31, dim=DIM)
        fn1, _W1 = linear_expert(seed=32, dim=DIM)
        fn2, _W2 = linear_expert(seed=33, dim=DIM)
        with RunningNode(LAYER, 0, fn0) as node0, \
                RunningNode(LAYER, 1, fn1) as replica1, \
                RunningNode(LAYER, 2, fn2) as node2:
            members = [
                MemberSpec(LAYER, 0, node_id(0), node0.endpoint),
                MemberSpec(LAYER, 1, node_id(1), dead_endpoint(),
                           replicas=(ReplicaSpec(node_id(1) + "-b",
                                                 replica1.endpoint),)),
                MemberSpec(LAYER, 2, node_id(2), node2.endpoint),
            ]
            # Expert 1 (whose primary is dead) shares a head server with
            # expert 0, so position 1 comes back from a different subcluster
            # than position 2.
            config = ClusterConfig(
                [SubclusterSpec("sc-0000", ("127.0.0.1", 0),
                                tuple(members[:2])),
                 SubclusterSpec("sc-0001", ("127.0.0.1", 0),
                                tuple(members[2:]))],
                subcluster_size=2)
            policy = RetryPolicy(attempts=2)
            services = [SubclusterService(
                SubclusterCoordinator(spec.group_id,
                                      config.placement(spec.group_id),
                                      config.expert_endpoints(spec.group_id),
                                      timeout=10.0, retry_policy=policy),
                host="127.0.0.1", port=0).start()
                for spec in config.subclusters]
            transport = SubclusterTransport(
                {s.group_id: s.address for s in services}, timeout=10.0)
            dispatcher = HierarchicalExpertDispatcher(config.placement(),
                                                      config.plan(), transport)
            flat_transport = PersistentSocketTransport(
                config.expert_endpoints(), timeout=10.0)
            flat = DistributedExpertDispatcher(config.placement(),
                                               flat_transport,
                                               retry_policy=policy)
            try:
                x = np.arange(DIM, dtype=np.float32) - 1.5
                gates = [0.5, 0.25, 0.125]
                hier = dispatcher.run_expert_stage(LAYER, x, [0, 1, 2], gates,
                                                   token_id=41, timeout=20)
                reference = flat.run_expert_stage(LAYER, x, [0, 1, 2], gates,
                                                  token_id=42, timeout=20)
                np.testing.assert_array_equal(hier, reference)
                self.assertEqual(replica1.calls, 2)  # once per dispatcher
            finally:
                dispatcher.close()
                transport.close()
                flat.close()
                flat_transport.close()
                for service in services:
                    service.stop()


class TestFastMode(unittest.TestCase):
    def test_fast_costs_fewer_upstream_bytes_than_exact(self):
        x = np.arange(256, dtype=np.float32)
        with _Farm(n_experts=4, size=4, tap=True, dim=256) as farm:
            farm.run(x, [0, 1, 2, 3], GATES, token_id=1)
            exact_down = farm.taps[0].bytes_down
            farm.run(x, [0, 1, 2, 3], GATES, token_id=2, fast=True)
            fast_down = farm.taps[0].bytes_down - exact_down
            # Four contributions come back exactly, one partial sum in fast
            # mode: that is the documented bandwidth tradeoff.
            self.assertGreater(exact_down, 3 * x.nbytes)
            self.assertLess(fast_down, 2 * x.nbytes)

    def test_fast_flag_travels_in_the_request(self):
        with _Farm(n_experts=2, size=2, tap=True) as farm:
            x = np.ones(DIM, dtype=np.float32)
            farm.run(x, [0, 1], [0.5, 0.25], fast=True)
            request = B.decode_batch_request(
                farm.taps[0].frames_of_type(P.MSG_BREQ)[0])
            self.assertTrue(request["fast"])
            self.assertEqual(farm.services[0].coordinator.fast_batches, 1)

    def test_exact_is_the_default_on_the_wire(self):
        with _Farm(n_experts=2, size=2, tap=True) as farm:
            x = np.ones(DIM, dtype=np.float32)
            farm.run(x, [0, 1], [0.5, 0.25])
            request = B.decode_batch_request(
                farm.taps[0].frames_of_type(P.MSG_BREQ)[0])
            self.assertFalse(request["fast"])
            self.assertEqual(farm.services[0].coordinator.fast_batches, 0)

    def test_run_batch_agrees_in_both_modes_for_one_whole_group(self):
        """Nothing interleaves here, so the two associations coincide."""
        x = np.arange(DIM, dtype=np.float32) - 2.0
        entries = [B.BatchEntry(0, 0.5), B.BatchEntry(1, 0.25)]
        with _Farm(n_experts=2, size=2) as farm:
            exact = farm.transport.run_batch("sc-0000", LAYER, 51, x, entries,
                                             timeout=10)
            quick = farm.transport.run_batch("sc-0000", LAYER, 52, x, entries,
                                             timeout=10, fast=True)
            np.testing.assert_array_equal(exact, quick)

    def test_a_head_server_can_refuse_fast_batches(self):
        with _Farm(n_experts=2, size=2, allow_fast=False) as farm:
            x = np.ones(DIM, dtype=np.float32)
            with self.assertRaises(SubclusterError) as ctx:
                farm.run(x, [0, 1], [0.5, 0.25], fast=True)
            self.assertEqual(ctx.exception.code, B.ERR_BAD_REQUEST)
            # Exact mode still works on the same head server.
            np.testing.assert_array_equal(
                farm.run(x, [0, 1], [0.5, 0.25], token_id=2),
                farm.flat_reference(x, [0, 1], [0.5, 0.25]))


class TestWireEconomy(unittest.TestCase):
    def test_activation_is_sent_once_per_subcluster_request(self):
        with _Farm(n_experts=4, size=4, tap=True, dim=256) as farm:
            x = np.arange(256, dtype=np.float32)  # 1 KiB activation
            farm.run(x, [0, 1, 2, 3], GATES)
            tap = farm.taps[0]
            requests = tap.frames_of_type(P.MSG_BREQ)
            self.assertEqual(len(requests), 1)
            request = B.decode_batch_request(requests[0])
            self.assertEqual(len(request["entries"]), 4)
            np.testing.assert_array_equal(request["array"], x)
            # The activation appears exactly once in what crossed the link.
            payload = x.astype(">f4").tobytes()
            self.assertEqual(tap.frames_up[0].count(payload), 1)
            self.assertLess(tap.bytes_up, 2 * x.nbytes)

    def test_one_batch_per_subcluster_not_per_expert(self):
        with _Farm(n_experts=4, size=2, tap=True, dim=64) as farm:
            x = np.zeros(64, dtype=np.float32)
            farm.run(x, [0, 1, 2, 3], GATES)
            for tap in farm.taps:
                self.assertEqual(len(tap.frames_of_type(P.MSG_BREQ)), 1)
                self.assertEqual(len(B.decode_batch_request(
                    tap.frames_of_type(P.MSG_BREQ)[0])["entries"]), 2)


class TestConcurrency(unittest.TestCase):
    def test_experts_inside_one_subcluster_run_simultaneously(self):
        barrier = threading.Barrier(4, timeout=15)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=4, gate_fn=gated) as farm:
            out = farm.run(x, [0, 1, 2, 3], GATES)
            expected = sum(g * (W @ x) for g, W in zip(GATES, farm.weights))
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_subclusters_run_simultaneously(self):
        """The barrier only trips if both head servers are working at once."""
        barrier = threading.Barrier(4, timeout=15)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2, gate_fn=gated) as farm:
            out = farm.run(x, [0, 1, 2, 3], GATES)
            expected = sum(g * (W @ x) for g, W in zip(GATES, farm.weights))
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_two_batches_in_flight_on_one_upstream_connection(self):
        """Multiple requests per upstream connection, answered out of order.

        The pool is capped at one connection per head server, so both batches
        share a socket; the barrier needs all four expert calls of both batches
        live at once, which a coordinator that handled frames strictly one at a
        time could not do.
        """
        barrier = threading.Barrier(4, timeout=15)

        def gated(fn):
            def wrapped(x):
                barrier.wait()
                return fn(x)
            return wrapped

        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=2, size=2, gate_fn=gated,
                   max_connections_per_group=1) as farm:
            # Open the one permitted socket first, so both batches are known to
            # share it rather than racing the pool's first connect.
            farm.transport.ping("sc-0000")
            first = farm.dispatcher.submit_expert_stage(
                LAYER, x, [0, 1], [0.5, 0.25], token_id=21)
            second = farm.dispatcher.submit_expert_stage(
                LAYER, x, [0, 1], [0.5, 0.25], token_id=22)
            expected = (0.5 * (farm.weights[0] @ x)
                        + 0.25 * (farm.weights[1] @ x))
            for stage in (second, first):  # reap out of submission order
                np.testing.assert_allclose(stage.result(15), expected,
                                           rtol=1e-5, atol=1e-5)
            self.assertEqual(farm.transport.connection_count("sc-0000"), 1)
            self.assertEqual([node.calls for node in farm.nodes], [2, 2])

    def test_upstream_and_downstream_connections_are_reused(self):
        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=2, size=2, tap=True) as farm:
            for token in range(5):
                farm.run(x, [0, 1], [0.5, 0.25], token_id=token)
            self.assertEqual(farm.taps[0].accepted, 1)
            self.assertEqual(farm.transport.connection_count(), 1)
            # Each console accepted one connection from its head server.
            self.assertEqual([node.accepted for node in farm.nodes], [1, 1])
            self.assertEqual([node.calls for node in farm.nodes], [5, 5])


class TestHeartbeat(unittest.TestCase):
    def test_layer_pings_head_servers(self):
        with _Farm(n_experts=4, size=2) as farm:
            self.assertEqual(farm.dispatcher.check_liveness(),
                             {"sc-0000": True, "sc-0001": True})
            self.assertGreaterEqual(farm.transport.ping("sc-0000"), 0.0)

    def test_head_server_pings_its_consoles(self):
        with _Farm(n_experts=2, size=2) as farm:
            coordinator = farm.services[0].coordinator
            self.assertEqual(coordinator.check_members(),
                             {node_id(0): True, node_id(1): True})

    def test_dead_head_server_is_reported_not_raised(self):
        with _Farm(n_experts=2, size=2) as farm:
            farm.transport.add_group("sc-dead", dead_endpoint())
            self.assertFalse(farm.transport.alive("sc-dead", timeout=2.0))
            self.assertEqual(
                farm.dispatcher.check_liveness(["sc-0000", "sc-dead"]),
                {"sc-0000": True, "sc-dead": False})

    def test_heartbeat_shares_the_batch_connection(self):
        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=2, size=2, tap=True) as farm:
            farm.run(x, [0, 1], [0.5, 0.25])
            farm.transport.ping("sc-0000")
            farm.run(x, [0, 1], [0.5, 0.25], token_id=2)
            self.assertEqual(farm.taps[0].accepted, 1)
            self.assertEqual(len(farm.taps[0].frames_of_type(P.MSG_PING)), 1)


class TestMalformedFrames(unittest.TestCase):
    """A head server must reject junk clearly and stay usable."""

    def _talk(self, address, frame):
        sock = socket.create_connection(address, timeout=5)
        try:
            sock.sendall(frame)
            return sock, P.read_frame(sock.recv)
        except BaseException:
            sock.close()
            raise

    def test_bad_flags_get_a_bad_request_error_and_the_connection_survives(self):
        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=2, size=2) as farm:
            body = B.encode_batch_request(
                LAYER, 5, x, [B.BatchEntry(0, 0.5)])[4:]
            head = len(body) - 8 - 8
            forged = (body[:head] + struct.pack("!HHI", 1, 0x0002, 0)
                      + body[head + 8:])
            frame = struct.pack("!I", len(forged)) + forged
            sock, msg = self._talk(farm.services[0].address, frame)
            try:
                err = B.parse_batch_error(msg)
                self.assertEqual(err["code"], B.ERR_BAD_REQUEST)
                self.assertIn("flags", err["detail"])
                # Same connection, now a valid request.
                sock.sendall(B.encode_batch_request(
                    LAYER, 6, x, [B.BatchEntry(0, 1.0)]))
                good = B.parse_batch_response(P.read_frame(sock.recv))
                self.assertEqual(good["n_reduced"], 1)
                self.assertTrue(good["per_expert"])
                self.assertEqual(good["experts"], [0])
                np.testing.assert_allclose(good["array"][0],
                                           farm.weights[0] @ x,
                                           rtol=1e-5, atol=1e-5)
            finally:
                sock.close()

    def test_expert_frame_type_is_refused(self):
        with _Farm(n_experts=2, size=2) as farm:
            frame = P.encode(P.MSG_RSP, LAYER, 0, 3, np.ones(2, np.float32))
            sock, msg = self._talk(farm.services[0].address, frame)
            try:
                err = B.parse_batch_error(msg)
                self.assertEqual(err["code"], B.ERR_BAD_REQUEST)
                self.assertIn("msg_type", err["detail"])
            finally:
                sock.close()

    def test_absurd_length_prefix_drops_the_connection(self):
        with _Farm(n_experts=2, size=2) as farm:
            sock = socket.create_connection(farm.services[0].address,
                                            timeout=5)
            try:
                sock.sendall(struct.pack("!I", P.MAX_FRAME_BYTES + 1)
                             + b"P3XC")
                self.assertEqual(sock.recv(16), b"")  # closed, not allocated
            finally:
                sock.close()
            # The service is still serving other clients.
            x = np.ones(DIM, dtype=np.float32)
            np.testing.assert_allclose(farm.run(x, [0, 1], [0.5, 0.25]),
                                       0.5 * (farm.weights[0] @ x)
                                       + 0.25 * (farm.weights[1] @ x),
                                       rtol=1e-5, atol=1e-5)

    def test_expert_not_in_this_subcluster(self):
        x = np.ones(DIM, dtype=np.float32)
        with _Farm(n_experts=4, size=2) as farm:
            # sc-0000 holds experts 0 and 1; ask it for expert 3.
            with self.assertRaises(SubclusterError) as ctx:
                farm.transport.run_batch("sc-0000", LAYER, 9, x,
                                         [B.BatchEntry(3, 1.0)], timeout=10)
            err = ctx.exception
            self.assertEqual(err.code, B.ERR_UNKNOWN_EXPERT)
            self.assertEqual([f.expert for f in err.failures], [3])
            self.assertTrue(err.safe_to_retry)


class TestFailureSemantics(unittest.TestCase):
    def _farm_with(self, members, size=2, retry_policy=None, timeout=5.0):
        """Build a one-subcluster farm from explicit member specs."""
        config = ClusterConfig(
            [SubclusterSpec("sc-0000", ("127.0.0.1", 0), tuple(members))],
            subcluster_size=size)
        coordinator = SubclusterCoordinator(
            "sc-0000", config.placement(), config.expert_endpoints(),
            timeout=timeout, retry_policy=retry_policy)
        service = SubclusterService(coordinator, port=0).start()
        transport = SubclusterTransport({"sc-0000": service.address},
                                        timeout=timeout)
        dispatcher = HierarchicalExpertDispatcher(config.placement(),
                                                  config.plan(), transport)
        return config, service, transport, dispatcher

    def test_unreachable_console_yields_a_structured_error_not_a_short_sum(self):
        fn, W = linear_expert(seed=1, dim=DIM)
        with RunningNode(LAYER, 0, fn) as live:
            members = [
                MemberSpec(LAYER, 0, node_id(0), live.endpoint),
                MemberSpec(LAYER, 1, node_id(1), dead_endpoint()),
            ]
            _config, service, transport, dispatcher = self._farm_with(members)
            try:
                with self.assertRaises(SubclusterError) as ctx:
                    dispatcher.run_expert_stage(LAYER, np.ones(DIM, np.float32),
                                                [0, 1], [0.5, 0.5],
                                                token_id=3, timeout=10)
                err = ctx.exception
                self.assertEqual(err.node_id, "sc-0000")
                self.assertEqual([f.expert for f in err.failures], [1])
                self.assertEqual(err.failures[0].reason,
                                 B.ERR_NODE_UNREACHABLE)
                self.assertEqual(err.failures[0].node_id, node_id(1))
                # Nothing about the failure is retryable-unsafe: expert 1 never
                # ran, so a layer-level retry stays at-most-once.
                self.assertTrue(err.safe_to_retry)
            finally:
                dispatcher.close()
                transport.close()
                service.stop()
            self.assertEqual(live.calls, 1)  # the sibling still ran once

    def test_expert_error_frame_is_attributed(self):
        def broken(_x):
            raise ValueError("expert weights corrupt")

        fn, _W = linear_expert(seed=2, dim=DIM)
        with RunningNode(LAYER, 0, fn) as good, \
                RunningNode(LAYER, 1, broken) as bad:
            members = [MemberSpec(LAYER, 0, node_id(0), good.endpoint),
                       MemberSpec(LAYER, 1, node_id(1), bad.endpoint)]
            _config, service, transport, dispatcher = self._farm_with(members)
            try:
                with self.assertRaises(SubclusterError) as ctx:
                    dispatcher.run_expert_stage(LAYER, np.ones(DIM, np.float32),
                                                [0, 1], [0.5, 0.5],
                                                token_id=4, timeout=10)
                err = ctx.exception
                self.assertEqual(err.failures[0].reason, B.ERR_NODE_ERROR)
                self.assertEqual(err.failures[0].node_id, node_id(1))
                self.assertFalse(err.safe_to_retry)
            finally:
                dispatcher.close()
                transport.close()
                service.stop()

    def test_replica_failover_inside_the_subcluster_counts_once(self):
        fn0, W0 = linear_expert(seed=3, dim=DIM)
        fn1, W1 = linear_expert(seed=4, dim=DIM)
        with RunningNode(LAYER, 0, fn0) as primary0, \
                RunningNode(LAYER, 1, fn1) as replica1:
            members = [
                MemberSpec(LAYER, 0, node_id(0), primary0.endpoint),
                # Expert 1's primary console is dead; its replica is live.
                MemberSpec(LAYER, 1, node_id(1), dead_endpoint(),
                           replicas=(ReplicaSpec(node_id(1) + "-b",
                                                 replica1.endpoint),)),
            ]
            _config, service, transport, dispatcher = self._farm_with(
                members, retry_policy=RetryPolicy(attempts=2))
            try:
                x = np.arange(DIM, dtype=np.float32)
                out = dispatcher.run_expert_stage(LAYER, x, [0, 1], [0.5, 0.25],
                                                  token_id=5, timeout=10)
                np.testing.assert_allclose(
                    out, 0.5 * (W0 @ x) + 0.25 * (W1 @ x),
                    rtol=1e-5, atol=1e-5)
                # Failover contributed expert 1 exactly once.
                self.assertEqual(replica1.calls, 1)
                self.assertEqual(primary0.calls, 1)
                self.assertEqual(service.coordinator.batches_served, 1)
            finally:
                dispatcher.close()
                transport.close()
                service.stop()

    def test_replica_hint_pins_the_entry_to_the_standby_console(self):
        """A layer that knows the primary is down can say so in the batch."""
        fn, W = linear_expert(seed=9, dim=DIM)
        with RunningNode(LAYER, 0, fn) as primary, \
                RunningNode(LAYER, 0, fn) as standby:
            members = [MemberSpec(LAYER, 0, node_id(0), primary.endpoint,
                                  replicas=(ReplicaSpec(node_id(0) + "-b",
                                                        standby.endpoint),))]
            _config, service, transport, dispatcher = self._farm_with(
                members, size=1)
            try:
                x = np.arange(DIM, dtype=np.float32)
                out = dispatcher.run_expert_stage(LAYER, x, [0], [0.5],
                                                  token_id=11, timeout=10,
                                                  replicas=[1])
                np.testing.assert_allclose(out, 0.5 * (W @ x),
                                           rtol=1e-5, atol=1e-5)
                self.assertEqual(standby.calls, 1)
                self.assertEqual(primary.calls, 0)

                # A hint naming a replica that was never configured is a
                # request error, not a node failure.
                with self.assertRaises(SubclusterError) as ctx:
                    dispatcher.run_expert_stage(LAYER, x, [0], [0.5],
                                                token_id=12, timeout=10,
                                                replicas=[7])
                self.assertEqual(ctx.exception.code, B.ERR_BAD_REQUEST)
                self.assertEqual(standby.calls, 1)
                self.assertEqual(primary.calls, 0)
            finally:
                dispatcher.close()
                transport.close()
                service.stop()

    def test_downstream_deadline_is_bounded_by_the_layers_budget(self):
        release = threading.Event()

        def slow(x):
            release.wait(30)
            return x.astype(np.float32)

        with RunningNode(LAYER, 0, slow) as node:
            members = [MemberSpec(LAYER, 0, node_id(0), node.endpoint)]
            # The coordinator's own ceiling is 30 s; the request's deadline_ms
            # asks for 1 s, and the layer waits generously for the BERR so the
            # assertion is about the coordinator's bound, not the client's.
            _config, service, transport, dispatcher = self._farm_with(
                members, size=1, timeout=30.0)
            try:
                pending = transport.submit_batch(
                    "sc-0000", LAYER, 6, np.ones(DIM, np.float32),
                    [B.BatchEntry(0, 1.0)], deadline_ms=1000)
                with self.assertRaises(SubclusterError) as ctx:
                    pending.partial(20)
                self.assertEqual(ctx.exception.failures[0].reason,
                                 B.ERR_NODE_TIMEOUT)
                # A timed-out expert may still be running: not safe to retry.
                self.assertFalse(ctx.exception.safe_to_retry)
            finally:
                release.set()
                dispatcher.close()
                transport.close()
                service.stop()

    def test_short_partial_is_refused_by_the_layer(self):
        """A BRSP covering fewer experts than asked must not be reduced."""
        class _Frame:
            def __init__(self, msg):
                self._msg = msg

            def done(self):
                return True

            def message(self, timeout=None):
                return self._msg

        msg = B.decode_batch_response(
            B.encode_batch_response(LAYER, 1, np.ones(DIM, np.float32), 1)[4:])
        pending = PendingBatch("sc-0000",
                               [B.BatchEntry(0, 1.0), B.BatchEntry(1, 1.0)],
                               [0, 1], _Frame(msg), fast=True)
        with self.assertRaises(SubclusterError) as ctx:
            pending.partial()
        self.assertIn("covers 1 of 2", str(ctx.exception))

    def test_a_partial_sum_is_refused_when_the_layer_asked_for_rows(self):
        """Exact mode must not silently accept a re-associated sum."""
        class _Frame:
            def __init__(self, msg):
                self._msg = msg

            def done(self):
                return True

            def message(self, timeout=None):
                return self._msg

        msg = B.decode_batch_response(
            B.encode_batch_response(LAYER, 1, np.ones(DIM, np.float32), 1)[4:])
        pending = PendingBatch("sc-0000", [B.BatchEntry(0, 1.0)], [0],
                               _Frame(msg))
        with self.assertRaises(SubclusterError) as ctx:
            pending.contributions()
        self.assertIn("re-associate", str(ctx.exception))

    def test_an_unrequested_expert_in_a_response_is_refused(self):
        class _Frame:
            def __init__(self, msg):
                self._msg = msg

            def done(self):
                return True

            def message(self, timeout=None):
                return self._msg

        msg = B.decode_batch_response(B.encode_batch_contributions(
            LAYER, 1, [np.ones(DIM, np.float32)], [9])[4:])
        pending = PendingBatch("sc-0000", [B.BatchEntry(0, 1.0)], [4],
                               _Frame(msg))
        with self.assertRaises(SubclusterError) as ctx:
            pending.contributions()
        self.assertIn("expert 9", str(ctx.exception))


class TestLifecycle(unittest.TestCase):
    def test_stopping_a_service_releases_its_sockets_and_threads(self):
        before = len(reader_threads())
        farm = _Farm(n_experts=2, size=2)
        x = np.ones(DIM, dtype=np.float32)
        farm.run(x, [0, 1], [0.5, 0.25])
        self.assertGreater(len(reader_threads()), before)
        address = farm.services[0].address
        farm.close()
        for _ in range(50):
            if len(reader_threads()) <= before:
                break
            threading.Event().wait(0.05)
        self.assertLessEqual(len(reader_threads()), before)
        with self.assertRaises(OSError):
            socket.create_connection(address, timeout=2).close()

    def test_transport_refuses_requests_after_close(self):
        with _Farm(n_experts=2, size=2) as farm:
            x = np.ones(DIM, dtype=np.float32)
            farm.run(x, [0, 1], [0.5, 0.25])
            farm.transport.close()
            with self.assertRaises(Exception):
                farm.run(x, [0, 1], [0.5, 0.25], token_id=2)

    def test_coordinator_refuses_batches_after_close(self):
        with _Farm(n_experts=2, size=2) as farm:
            coordinator = farm.services[0].coordinator
            coordinator.close()
            request = B.decode_batch_request(B.encode_batch_request(
                LAYER, 1, np.ones(DIM, np.float32),
                [B.BatchEntry(0, 1.0)])[4:])
            err = B.decode_batch_error(coordinator.handle_batch(request)[4:])
            self.assertEqual(err["code"], B.ERR_SHUTTING_DOWN)


if __name__ == "__main__":
    unittest.main()
