"""Three-tier deployment over real sockets: layer -> regions -> heads -> consoles.

Every test boots the whole topology on loopback — expert workers, one subcluster
head service each, regional coordinator services above them, and a layer-side
dispatcher — and talks P3XC batch frames between the tiers. Failure injection uses
proxies and events (a proxy that hangs up after forwarding a request, a service
that is stopped, an expert gated on an event), never sleeps or latency thresholds.

The load-bearing assertion is ``np.array_equal`` against
``DistributedExpertDispatcher``: adding a tier must not change a single bit of
the layer's reduction.
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
from ps3_cluster.coordinator import (CoordinatorService,  # noqa: E402
                                     SubclusterCoordinator)
from ps3_cluster.dedup import DedupCache  # noqa: E402
from ps3_cluster.deployment import ClusterConfig  # noqa: E402
from ps3_cluster.dispatch import DistributedExpertDispatcher  # noqa: E402
from ps3_cluster.errors import SubclusterError, TransportError  # noqa: E402
from ps3_cluster.hierarchy import (HierarchicalExpertDispatcher,  # noqa: E402
                                   LinkRetryPolicy, SubclusterTransport,
                                   next_request_id)
from ps3_cluster.regional import RegionalCoordinator  # noqa: E402
from ps3_cluster.transport import PersistentSocketTransport  # noqa: E402

LAYER = 5
DIM = 4


def node_id(expert):
    return f"ps3-L{LAYER:03d}-E{expert:04d}"


class CutAfterRequest:
    """Proxy that forwards a request, then drops the connection.

    Models the ambiguous failure the retry rules are about: the head server
    behind it really does drive its consoles, and the caller never learns the
    answer. ``forwarded`` counts requests that made it through.
    """

    def __init__(self, target, host="127.0.0.1", swallow=False):
        self.target = target
        #: True to never forward at all, which is a timeout, not an ambiguity.
        self.swallow = swallow
        self.forwarded = 0
        self._lock = threading.Lock()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(8)
        self.host, self.port = self._sock.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return (self.host, self.port)

    def _serve(self):
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,),
                             daemon=True).start()

    def _handle(self, client):
        try:
            head = _recv_exact(client, 4)
            if head is None:
                return
            (length,) = struct.unpack("!I", head)
            body = _recv_exact(client, length)
            if body is None or self.swallow:
                return
            upstream = socket.create_connection(self.target, timeout=5)
            try:
                upstream.sendall(head + body)
                # Wait for the downstream tier to answer, then throw it away:
                # the work has happened, the caller cannot know.
                _recv_exact(upstream, 4)
                with self._lock:
                    self.forwarded += 1
            finally:
                upstream.close()
        except OSError:
            return
        finally:
            client.close()

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


class _TieredFarm:
    """The full three-tier farm on loopback.

    ``head_standby``/``region_standby`` give each head server (subcluster or
    region) a second listening address served by the *same* coordinator, which is
    how a real standby is deployed: same membership, same bounded dedup cache.
    ``tap_regions``/``tap_heads`` insert byte-counting proxies on a link so a test
    can see what actually crossed it.
    """

    def __init__(self, n_experts=8, size=2, n_regions=2, timeout=10.0,
                 allow_fast=True, fast=False, head_standby=False,
                 region_standby=False, tap_regions=False, tap_heads=False,
                 retry_policy=None, layer_policy=None, wrap=None,
                 dedup=None, dim=DIM):
        self.dim = dim
        self.nodes = []
        self.weights = []
        for expert in range(n_experts):
            fn, W = linear_expert(seed=300 + expert, dim=dim)
            self.weights.append(W)
            self.nodes.append(RunningNode(
                LAYER, expert, wrap(expert, fn) if wrap else fn))
        consoles = [(expert, node_id(expert), node.endpoint)
                    for expert, node in enumerate(self.nodes)]
        self.config = ClusterConfig.for_layer(
            LAYER, consoles, size=size,
            head_host="127.0.0.1").with_regions(n_regions, host="127.0.0.1")
        self.placement = self.config.placement()
        self.plan = self.config.tiered_plan()
        self.taps = []
        self.head_services = {}
        self.head_endpoints = {}
        for spec in self.config.subclusters:
            coordinator = SubclusterCoordinator(
                spec.group_id, self.config.placement(spec.group_id),
                self.config.expert_endpoints(spec.group_id), timeout=timeout,
                allow_fast=allow_fast)
            self.head_services[spec.group_id] = self._expose(
                coordinator, spec.group_id, head_standby, tap_heads,
                self.head_endpoints)
        self.region_services = {}
        self.region_coordinators = {}
        self.region_endpoints = {}
        for spec in self.config.regions:
            coordinator = RegionalCoordinator(
                spec.region_id, self.placement,
                self.config.plan(spec.region_id),
                {g: self.head_endpoints[g] for g in spec.subclusters},
                timeout=timeout, allow_fast=allow_fast,
                retry_policy=retry_policy, dedup=dedup)
            self.region_coordinators[spec.region_id] = coordinator
            self.region_services[spec.region_id] = self._expose(
                coordinator, spec.region_id, region_standby, tap_regions,
                self.region_endpoints)
        self.transport = SubclusterTransport(
            self.region_endpoints, timeout=timeout,
            retry_policy=layer_policy)
        self.dispatcher = HierarchicalExpertDispatcher(
            self.placement, self.plan, self.transport, fast=fast,
            retry_policy=layer_policy)

    def _expose(self, coordinator, group_id, standby, tap, endpoints):
        """Start one or two services for a coordinator, recording addresses."""
        services = [CoordinatorService(coordinator, host="127.0.0.1",
                                       port=0).start()]
        if standby:
            services.append(CoordinatorService(coordinator, host="127.0.0.1",
                                               port=0).start())
        addresses = []
        for service in services:
            if tap:
                proxy = FrameTap(service.address)
                self.taps.append(proxy)
                addresses.append(proxy.endpoint)
            else:
                addresses.append(service.address)
        endpoints[group_id] = addresses
        return services

    # -- helpers -----------------------------------------------------------
    def run(self, x, expert_ids, gates, token_id=1, timeout=None, fast=None):
        return self.dispatcher.run_expert_stage(LAYER, x, expert_ids, gates,
                                                token_id, timeout, fast=fast)

    def flat_reference(self, x, expert_ids, gates, token_id=99):
        transport = PersistentSocketTransport(
            self.config.expert_endpoints(), timeout=10.0)
        flat = DistributedExpertDispatcher(self.placement, transport)
        try:
            return flat.run_expert_stage(LAYER, x, expert_ids, gates, token_id)
        finally:
            flat.close()
            transport.close()

    def expert_calls(self):
        return [node.calls for node in self.nodes]

    def region_of(self, expert):
        return self.plan.subcluster_of(node_id(expert))

    def stop_primary(self, group_id):
        """Stop a head server's primary listener, leaving its standby up."""
        services = (self.region_services.get(group_id)
                    or self.head_services[group_id])
        services[0].stop_listening()

    def close(self):
        self.dispatcher.close()
        self.transport.close()
        for proxy in self.taps:
            proxy.close()
        for group in (self.region_services, self.head_services):
            for services in group.values():
                for index, service in enumerate(services):
                    # One coordinator behind however many listeners it has.
                    service.stop(close_coordinator=index == 0)
        for node in self.nodes:
            node.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestThreeTierExactness(unittest.TestCase):
    def test_positions_interleave_across_regions_and_subclusters(self):
        with _TieredFarm(n_experts=8, size=2, n_regions=2) as farm:
            self.assertEqual(len(farm.config.regions), 2)
            self.assertEqual(len(farm.config.subclusters), 4)
            # A selection whose top-k order zig-zags between the two regions.
            selection = [0, 5, 1, 6, 2]
            regions = [farm.region_of(e) for e in selection]
            self.assertGreater(len(set(regions)), 1)
            self.assertNotEqual(regions, sorted(regions))
            groups = farm.dispatcher.group_selection(LAYER, selection)
            self.assertEqual(len(groups), 2)  # one batch per region, not per head

    def test_bit_identical_to_flat_dispatch_over_three_tiers(self):
        rng = np.random.default_rng(20240607)
        with _TieredFarm(n_experts=8, size=2, n_regions=2) as farm:
            for trial in range(12):
                x = rng.standard_normal(DIM).astype(np.float32)
                k = int(rng.integers(2, 6))
                experts = list(rng.permutation(8)[:k].astype(int))
                gates = rng.standard_normal(k).astype(np.float32).tolist()
                got = farm.run(x, experts, gates, token_id=trial + 1,
                               timeout=10)
                want = farm.flat_reference(x, experts, gates,
                                           token_id=1000 + trial)
                self.assertTrue(np.array_equal(got, want),
                                f"trial {trial} experts={experts} differs")

    def test_out_of_order_completion_does_not_change_the_result(self):
        """The last position answers first; the sum must not notice."""
        gate = threading.Event()
        order = []

        def wrap(expert, fn):
            def served(x):
                if expert == 0:
                    gate.wait(10)          # released by the last expert
                else:
                    order.append(expert)
                    if expert == 7:
                        gate.set()
                return fn(x)
            return served

        with _TieredFarm(n_experts=8, size=2, n_regions=2, wrap=wrap) as farm:
            x = np.arange(DIM, dtype=np.float32) - 1.5
            experts = [0, 7]
            gates = [0.5, 0.125]
            got = farm.run(x, experts, gates, token_id=3, timeout=10)
            self.assertEqual(order[0], 7)
            self.assertTrue(np.array_equal(
                got, farm.flat_reference(x, experts, gates, token_id=4)))

    def test_a_regions_rows_keep_their_original_positions(self):
        """Two regions, reversed selection: tags must not be re-sorted."""
        with _TieredFarm(n_experts=4, size=1, n_regions=2) as farm:
            x = np.ones(DIM, dtype=np.float32)
            experts = [3, 0, 2, 1]
            gates = [0.5, 0.25, 0.125, 0.0625]
            self.assertTrue(np.array_equal(
                farm.run(x, experts, gates, token_id=7, timeout=10),
                farm.flat_reference(x, experts, gates, token_id=8)))


class TestThreeTierFastMode(unittest.TestCase):
    def test_exact_is_the_default_on_both_links(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         tap_regions=True) as farm:
            x = np.ones(DIM, dtype=np.float32)
            farm.run(x, [0, 3], [0.5, 0.25], token_id=1, timeout=10)
            requests = [B.decode_batch_request(body)
                        for tap in farm.taps
                        for body in tap.frames_of_type(B.MSG_BREQ)]
            self.assertTrue(requests)
            self.assertFalse(any(r["fast"] for r in requests))

    def test_fast_mode_is_a_partial_and_travels_to_both_tiers(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         tap_regions=True) as farm:
            x = np.ones(DIM, dtype=np.float32)
            quick = farm.run(x, [0, 3], [0.5, 0.25], token_id=2, timeout=10,
                             fast=True)
            requests = [B.decode_batch_request(body)
                        for tap in farm.taps
                        for body in tap.frames_of_type(B.MSG_BREQ)]
            self.assertTrue(all(r["fast"] for r in requests))
            # Same value here (one expert per region, so nothing re-associates),
            # but the wire carried one partial per region instead of rows.
            self.assertTrue(np.array_equal(
                quick, farm.flat_reference(x, [0, 3], [0.5, 0.25])))

    def test_a_region_can_refuse_fast_batches(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         allow_fast=False) as farm:
            x = np.ones(DIM, dtype=np.float32)
            with self.assertRaises(SubclusterError) as ctx:
                farm.run(x, [0, 3], [0.5, 0.25], token_id=3, timeout=10,
                         fast=True)
            self.assertEqual(ctx.exception.code, B.ERR_BAD_REQUEST)
            self.assertTrue(np.array_equal(
                farm.run(x, [0, 3], [0.5, 0.25], token_id=4, timeout=10),
                farm.flat_reference(x, [0, 3], [0.5, 0.25])))


class TestWireEconomy(unittest.TestCase):
    def test_activation_travels_once_per_immediate_downstream_group(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2, dim=256,
                         tap_regions=True, tap_heads=True) as farm:
            x = np.ones(256, dtype=np.float32)
            # Four experts: two per region, two per subcluster below it.
            farm.run(x, [0, 1, 2, 3], [0.5, 0.25, 0.125, 0.0625],
                     token_id=1, timeout=10)
            per_tap = [len(tap.frames_of_type(B.MSG_BREQ))
                       for tap in farm.taps]
            # One batch on each of the two layer->region links and each of the
            # two region->head links, never one per expert.
            self.assertEqual(sorted(per_tap), [1, 1, 1, 1])
            for tap in farm.taps:
                request = B.decode_batch_request(
                    tap.frames_of_type(B.MSG_BREQ)[0])
                self.assertEqual(len(request["entries"]), 2)
                self.assertEqual(len(request["array"]), 256)

    def test_persistent_connections_are_reused_across_tokens(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2) as farm:
            x = np.ones(DIM, dtype=np.float32)
            for token in range(4):
                farm.run(x, [0, 3], [0.5, 0.25], token_id=token + 1,
                         timeout=10)
            # One connection per region for four tokens, and the regions in turn
            # keep one per head server.
            self.assertEqual(sum(farm.transport.connects_opened.values()), 2)
            self.assertEqual(farm.transport.connection_count(), 2)
            for coordinator in farm.region_coordinators.values():
                self.assertEqual(
                    sum(coordinator.transport.connects_opened.values()), 1)


class TestLinkFailover(unittest.TestCase):
    def test_dead_region_primary_fails_over_to_its_standby(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         region_standby=True) as farm:
            x = np.arange(DIM, dtype=np.float32)
            region = farm.region_of(0)
            primary, standby = farm.region_endpoints[region]
            farm.stop_primary(region)
            got = farm.run(x, [0, 3], [0.5, 0.25], token_id=1, timeout=10)
            self.assertTrue(np.array_equal(
                got, farm.flat_reference(x, [0, 3], [0.5, 0.25])))
            opened = farm.transport.connects_by_endpoint
            self.assertEqual(opened.get((region, standby)), 1)
            self.assertFalse(farm.transport.endpoint_healthy(region, primary))

    def test_dead_head_primary_fails_over_below_the_region(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=1,
                         head_standby=True) as farm:
            x = np.arange(DIM, dtype=np.float32)
            head = farm.config.subclusters[0].group_id
            farm.stop_primary(head)
            got = farm.run(x, [0, 1, 2, 3], [0.5, 0.25, 0.125, 0.0625],
                           token_id=1, timeout=10)
            self.assertTrue(np.array_equal(
                got, farm.flat_reference(x, [0, 1, 2, 3],
                                         [0.5, 0.25, 0.125, 0.0625])))
            region = list(farm.region_coordinators.values())[0]
            standby = farm.head_endpoints[head][1]
            self.assertEqual(
                region.transport.connects_by_endpoint.get((head, standby)), 1)

    def test_both_region_endpoints_unavailable_is_reported_not_reduced(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         region_standby=True) as farm:
            region = farm.region_of(0)
            farm.region_endpoints[region] = [dead_endpoint(), dead_endpoint()]
            farm.transport.add_group(region, farm.region_endpoints[region])
            with self.assertRaises(TransportError) as ctx:
                farm.run(np.ones(DIM, np.float32), [0, 3], [0.5, 0.25],
                         token_id=2, timeout=10)
            self.assertEqual(ctx.exception.node_id, region)
            self.assertTrue(ctx.exception.safe_to_retry)  # nothing was sent

    def test_a_cold_endpoint_is_still_tried_when_every_endpoint_is_cold(self):
        """Health steers preference; it must never remove the last endpoint."""
        with _TieredFarm(n_experts=4, size=2, n_regions=1) as farm:
            region = farm.plan.group_ids()[0]
            only = farm.region_endpoints[region][0]
            farm.transport.mark_endpoint_dead(region, only)
            self.assertFalse(farm.transport.endpoint_healthy(region, only))
            x = np.ones(DIM, dtype=np.float32)
            self.assertTrue(np.array_equal(
                farm.run(x, [0, 1], [0.5, 0.25], token_id=3, timeout=10),
                farm.flat_reference(x, [0, 1], [0.5, 0.25])))

    def test_heartbeat_steers_the_next_request_to_the_standby(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2,
                         region_standby=True) as farm:
            region = farm.region_of(0)
            primary, standby = farm.region_endpoints[region]
            farm.stop_primary(region)
            alive = farm.dispatcher.check_liveness([region], timeout=5)
            self.assertTrue(alive[region])       # answered by the standby
            self.assertFalse(farm.transport.endpoint_healthy(region, primary))
            opened = farm.transport.connects_by_endpoint
            self.assertIsNone(opened.get((region, primary)))
            self.assertEqual(opened.get((region, standby)), 1)


class TestAmbiguousRetry(unittest.TestCase):
    def test_a_drop_after_execution_is_not_retried_by_default(self):
        """At-most-once by default: the batch may already have run."""
        with _TieredFarm(n_experts=2, size=2, n_regions=1) as farm:
            region = farm.plan.group_ids()[0]
            live = farm.region_services[region][0].address
            with CutAfterRequest(live) as cut:
                farm.transport.add_group(region, [cut.endpoint, live])
                with self.assertRaises(TransportError):
                    farm.run(np.ones(DIM, np.float32), [0, 1], [0.5, 0.25],
                             token_id=1, timeout=5)
                self.assertEqual(cut.forwarded, 1)
                self.assertEqual(farm.expert_calls(), [1, 1])

    def test_opt_in_retry_to_the_same_process_executes_once(self):
        """Same coordinator, same request id: the dedup cache replays it."""
        policy = LinkRetryPolicy(attempts=2, retry_ambiguous=True)
        with _TieredFarm(n_experts=2, size=2, n_regions=1,
                         layer_policy=policy) as farm:
            region = farm.plan.group_ids()[0]
            live = farm.region_services[region][0].address
            with CutAfterRequest(live) as cut:
                farm.transport.add_group(region, [cut.endpoint, live])
                x = np.arange(DIM, dtype=np.float32) + 1.0
                got = farm.run(x, [0, 1], [0.5, 0.25], token_id=2, timeout=5)
                self.assertEqual(cut.forwarded, 1)
                self.assertEqual(farm.expert_calls(), [1, 1])
                self.assertEqual(
                    farm.region_coordinators[region].dedup.hits, 1)
                self.assertTrue(np.array_equal(
                    got, farm.flat_reference(x, [0, 1], [0.5, 0.25])))

    def test_opt_in_retry_to_another_replica_is_at_least_once(self):
        """Two processes share no state, so the work may run twice.

        The layer still reduces exactly one complete answer, and it is still
        bit-identical to the flat sum: that is the guarantee, not single
        execution.
        """
        policy = LinkRetryPolicy(attempts=2, retry_ambiguous=True)
        with _TieredFarm(n_experts=2, size=2, n_regions=1,
                         layer_policy=policy) as farm:
            region = farm.plan.group_ids()[0]
            spec = farm.config.regions[0]
            twin = RegionalCoordinator(
                region, farm.placement, farm.config.plan(region),
                {g: farm.head_endpoints[g] for g in spec.subclusters},
                timeout=10.0)
            with CoordinatorService(twin, host="127.0.0.1", port=0) as second:
                with CutAfterRequest(
                        farm.region_services[region][0].address) as cut:
                    farm.transport.add_group(region, [cut.endpoint,
                                                      second.address])
                    x = np.arange(DIM, dtype=np.float32) + 1.0
                    got = farm.run(x, [0, 1], [0.5, 0.25], token_id=2,
                                   timeout=5)
                    self.assertEqual(cut.forwarded, 1)
                    self.assertEqual(farm.expert_calls(), [2, 2])
                    self.assertEqual(twin.dedup.hits, 0)
                    self.assertTrue(np.array_equal(
                        got, farm.flat_reference(x, [0, 1], [0.5, 0.25])))

    def test_timeout_with_opt_in_retry_reaches_the_standby(self):
        policy = LinkRetryPolicy(attempts=2, retry_ambiguous=True)
        with _TieredFarm(n_experts=2, size=2, n_regions=1,
                         layer_policy=policy) as farm:
            region = farm.plan.group_ids()[0]
            live = farm.region_services[region][0].address
            with CutAfterRequest(live, swallow=True) as blackhole:
                farm.transport.add_group(region, [blackhole.endpoint, live])
                x = np.ones(DIM, dtype=np.float32)
                got = farm.run(x, [0, 1], [0.5, 0.25], token_id=3, timeout=2)
                self.assertEqual(blackhole.forwarded, 0)
                self.assertEqual(farm.expert_calls(), [1, 1])
                self.assertTrue(np.array_equal(
                    got, farm.flat_reference(x, [0, 1], [0.5, 0.25])))

    def test_a_retry_to_the_same_process_replays_instead_of_re_running(self):
        """Same request id, same coordinator: dedup, not a second fan-out."""
        cache = DedupCache(max_entries=8, ttl=30.0)
        with _TieredFarm(n_experts=2, size=2, n_regions=1,
                         dedup=cache) as farm:
            region = farm.plan.group_ids()[0]
            x = np.arange(DIM, dtype=np.float32)
            entries = [B.BatchEntry(0, 0.5), B.BatchEntry(1, 0.25)]
            request_id = next_request_id()
            first = farm.transport.call_batch(region, LAYER, 11, x, entries,
                                              timeout=10,
                                              request_id=request_id)
            second = farm.transport.call_batch(region, LAYER, 11, x, entries,
                                               timeout=10,
                                               request_id=request_id)
            self.assertEqual(sorted(first), sorted(second))
            for position in first:
                self.assertTrue(np.array_equal(first[position],
                                               second[position]))
            self.assertEqual(cache.hits, 1)
            self.assertEqual(farm.expert_calls(), [1, 1])

    def test_a_late_duplicate_answer_is_dropped_not_reduced(self):
        """A cancelled attempt's key is retired, so its reply is discarded."""
        with _TieredFarm(n_experts=2, size=2, n_regions=1) as farm:
            region = farm.plan.group_ids()[0]
            x = np.ones(DIM, dtype=np.float32)
            entries = [B.BatchEntry(0, 0.5), B.BatchEntry(1, 0.25)]
            request_id = next_request_id()
            abandoned = farm.transport.submit_batch(
                region, LAYER, 12, x, entries, request_id=request_id)
            abandoned.cancel()
            # The head server's answer for the cancelled attempt arrives on the
            # same connection; the retry must get its own, once.
            again = farm.transport.call_batch(region, LAYER, 12, x, entries,
                                              timeout=10,
                                              request_id=request_id)
            self.assertEqual(len(again), 2)
            flat = farm.flat_reference(x, [0, 1], [0.5, 0.25])
            summed = again[0] + again[1]
            self.assertTrue(np.array_equal(summed, flat))

    def test_a_foreign_request_id_in_a_reply_is_refused(self):
        with _TieredFarm(n_experts=2, size=2, n_regions=1) as farm:
            region = farm.plan.group_ids()[0]
            x = np.ones(DIM, dtype=np.float32)
            entries = [B.BatchEntry(0, 0.5)]
            pending = farm.transport.submit_batch(region, LAYER, 13, x, entries,
                                                  request_id=next_request_id())
            pending.request_id = next_request_id()   # pretend it was another's
            with self.assertRaises(SubclusterError) as ctx:
                pending.contributions(10)
            self.assertIn("request id", str(ctx.exception))


class TestRegionalFailureSemantics(unittest.TestCase):
    def test_a_dead_console_yields_a_structured_error_through_both_tiers(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2) as farm:
            farm.nodes[0].close()
            with self.assertRaises(SubclusterError) as ctx:
                farm.run(np.ones(DIM, np.float32), [0, 3], [0.5, 0.25],
                         token_id=1, timeout=10)
            failures = ctx.exception.failures
            self.assertTrue(failures)
            # Attribution survives the extra tier: the console, not the region.
            self.assertEqual(failures[0].node_id, node_id(0))
            self.assertEqual(failures[0].expert, 0)

    def test_a_region_refuses_an_expert_it_does_not_front(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=2) as farm:
            region = farm.region_of(0)
            outsider = next(e for e in range(4)
                            if farm.region_of(e) != region)
            with self.assertRaises(SubclusterError) as ctx:
                farm.transport.run_batch(region, LAYER, 2,
                                         np.ones(DIM, np.float32),
                                         [B.BatchEntry(outsider, 0.5)],
                                         timeout=10)
            self.assertEqual(ctx.exception.code, B.ERR_UNKNOWN_EXPERT)

    def test_a_dead_head_below_a_region_is_an_error_not_a_short_sum(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=1) as farm:
            region = list(farm.region_coordinators.values())[0]
            head = farm.config.subclusters[0].group_id
            region.transport.add_group(head, [dead_endpoint()])
            with self.assertRaises(SubclusterError) as ctx:
                farm.run(np.ones(DIM, np.float32), [0, 1, 2, 3],
                         [0.5, 0.25, 0.125, 0.0625], token_id=3, timeout=10)
            self.assertEqual(ctx.exception.code, B.ERR_NODE_UNREACHABLE)
            self.assertTrue(ctx.exception.failures)

    def test_a_closed_region_refuses_new_batches(self):
        with _TieredFarm(n_experts=2, size=2, n_regions=1) as farm:
            region_id = farm.plan.group_ids()[0]
            farm.region_coordinators[region_id].close()
            with self.assertRaises(SubclusterError) as ctx:
                farm.run(np.ones(DIM, np.float32), [0, 1], [0.5, 0.25],
                         token_id=4, timeout=10)
            self.assertEqual(ctx.exception.code, B.ERR_SHUTTING_DOWN)


class TestLifecycle(unittest.TestCase):
    def test_shutdown_releases_sockets_and_reader_threads(self):
        before = len(reader_threads())
        farm = _TieredFarm(n_experts=4, size=2, n_regions=2,
                           region_standby=True, head_standby=True)
        x = np.ones(DIM, dtype=np.float32)
        farm.run(x, [0, 3], [0.5, 0.25], token_id=1, timeout=10)
        self.assertGreater(len(reader_threads()), before)
        farm.close()
        for _ in range(50):
            if len(reader_threads()) <= before:
                break
            threading.Event().wait(0.05)
        self.assertLessEqual(len(reader_threads()), before)
        for service in [s for group in farm.region_services.values()
                        for s in group]:
            self.assertFalse(service.is_alive())

    def test_region_counts_batches_and_experts(self):
        with _TieredFarm(n_experts=4, size=2, n_regions=1) as farm:
            region = list(farm.region_coordinators.values())[0]
            x = np.ones(DIM, dtype=np.float32)
            farm.run(x, [0, 1, 2], [0.5, 0.25, 0.125], token_id=1, timeout=10)
            self.assertEqual(region.batches_served, 1)
            self.assertEqual(region.experts_called, 3)
            self.assertEqual(region.fast_batches, 0)


class TestTwoTierCompatibility(unittest.TestCase):
    def test_a_config_without_regions_still_serves_two_tiers(self):
        consoles = [(e, node_id(e), ("127.0.0.1", 9000 + e)) for e in range(4)]
        config = ClusterConfig.for_layer(LAYER, consoles, size=2)
        self.assertEqual(config.regions, [])
        with self.assertRaises(ValueError):
            config.tiered_plan()
        self.assertEqual(len(config.plan()), 2)

    def test_one_region_over_every_head_is_a_valid_degenerate_tree(self):
        with _TieredFarm(n_experts=4, size=1, n_regions=1) as farm:
            self.assertEqual(len(farm.plan), 1)
            x = np.arange(DIM, dtype=np.float32)
            experts, gates = [0, 1, 2, 3], [0.5, 0.25, 0.125, 0.0625]
            self.assertTrue(np.array_equal(
                farm.run(x, experts, gates, token_id=1, timeout=10),
                farm.flat_reference(x, experts, gates)))


if __name__ == "__main__":
    unittest.main()
