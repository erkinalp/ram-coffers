"""Regional coordinator: a head server whose downstream peers are head servers.

Condor's 1,716 consoles were grouped as 22-console subclusters behind
coordinating servers, and those servers were themselves driven from above rather
than by every client (Barnell et al., *Advanced Multi-INT Sensor Exploitation*,
IEEE HPEC 2012). At Kimi-K3 scale the same argument applies one level up: with
~78 subcluster heads per layer, a layer coordinator that talks to all of them
holds 78 connections and issues 78 batches per token. Insert regions and it holds
one connection per region.

    layer coordinator
        |  BREQ: activation once + [(expert, gate), ...]          (one per region)
        v
    RegionalCoordinator                 (this module)
        |  BREQ: activation once per *subcluster* it involves
        v
    SubclusterCoordinator               (coordinator.py, one per 22 consoles)
        |  P3XC REQ, concurrently
        v
    expert workers                      (22 PS3s, one expert each)

There is no second protocol and no second dispatcher: a region is the layer-side
:class:`~.hierarchy.HierarchicalExpertDispatcher` pointed at subcluster heads,
wrapped in the same :class:`~.coordinator.BaseCoordinator` server contract the
subcluster heads answer. That composition is what makes the depth of the tree a
deployment decision — a region could serve regions in turn, unchanged.

Numerics
--------
Exact mode (the default) is preserved through the extra tier for the same reason
it works with two: every row is tagged with the expert that produced it, and the
region merges its subclusters' rows without adding anything, so the layer still
performs one left-to-right fp32 accumulation over the whole top-k in ascending
position order. Three tiers are therefore ``np.array_equal`` to the flat
dispatcher, no matter how the token's positions interleave across regions and
subclusters.

Fast mode is explicitly approximate at both links: subclusters fold their experts
into partials, the region folds those partials into one, and the layer adds one
partial per region. Each fold re-associates the fp32 additions, so logits — and
so token choices — can differ. It has to be asked for (``REQ_FLAG_FAST``).

Failure semantics
-----------------
A region answers with every requested contribution or a ``BERR``, never a short
sum, exactly as a subcluster does. Failures keep their attribution: a console
that timed out is reported with *its* node id and reason, so an operator reading
the layer's error still sees which console failed under which head. Downstream
link failover (a subcluster head's replica endpoints) and retry policy are the
transport's, shared with the layer link (``hierarchy.py``); dedup of retried
batches is the coordinator base's (``dedup.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .batch import (BatchEntry, BatchFailure, ERR_UNKNOWN,
                    encode_batch_contributions, encode_batch_error,
                    encode_batch_response)
from .coordinator import (BaseCoordinator, CoordinatorService,
                          DEFAULT_UPSTREAM_WORKERS, failure_reason)
from .dedup import DedupCache
from .dispatch import ExpertPlacement
from .errors import SubclusterError, TransportError
from .hierarchy import (HierarchicalExpertDispatcher, LinkRetryPolicy,
                        SubclusterTransport)
from .subcluster import SubclusterPlan
from .transport import DEFAULT_TIMEOUT, EndpointSpec


class RegionalCoordinator(BaseCoordinator):
    """Fans a batch out across this region's subcluster heads.

    Parameters
    ----------
    region_id:
        Logical id of the region, used to attribute failures upstream.
    placement:
        The consoles under this region (canonical placement, sliced), so the
        region can tell whether it holds a requested expert and which head
        fronts it.
    plan:
        ``SubclusterPlan`` over those consoles: which head server owns which.
    head_endpoints:
        ``group_id -> endpoint``, or an ordered primary-first list per head.
    allow_fast:
        Whether to honour ``REQ_FLAG_FAST``. A region in fast mode also asks its
        subclusters for partials, compounding the re-association.
    retry_policy:
        Retry behaviour for the region -> subcluster link.
    """

    def __init__(self, region_id: str, placement: ExpertPlacement,
                 plan: SubclusterPlan,
                 head_endpoints: Dict[str, EndpointSpec],
                 timeout: float = DEFAULT_TIMEOUT,
                 max_workers: Optional[int] = None,
                 retry_policy: Optional[LinkRetryPolicy] = None,
                 transport: Optional[SubclusterTransport] = None,
                 allow_fast: bool = True,
                 dedup: Optional[DedupCache] = None):
        super().__init__(region_id, timeout=timeout, allow_fast=allow_fast,
                         dedup=dedup)
        self.placement = placement
        self.plan = plan
        self._owns_transport = transport is None
        self.transport = (
            SubclusterTransport(head_endpoints, timeout=timeout,
                                retry_policy=retry_policy)
            if transport is None else transport)
        self.dispatcher = HierarchicalExpertDispatcher(
            placement, plan, self.transport, max_workers=max_workers,
            retry_policy=retry_policy)

    # -- membership --------------------------------------------------------
    @property
    def region_id(self) -> str:
        return self.group_id

    def member_groups(self) -> List[str]:
        """The subcluster heads this region fronts (its downstream group)."""
        return self.plan.group_ids()

    def member_nodes(self, layer: Optional[int] = None) -> List[str]:
        return self.placement.node_ids(layer)

    def owns(self, layer: int, expert: int) -> bool:
        try:
            node_id = self.placement.node_for(layer, expert)
        except KeyError:
            return False
        return self.plan.has_node(node_id)

    def check_members(self, timeout: Optional[float] = None
                      ) -> Dict[str, bool]:
        """PING every subcluster head in the region; ``group_id -> alive``."""
        return self.dispatcher.check_liveness(self.member_groups(), timeout)

    # -- batch handling ----------------------------------------------------
    def _serve_batch(self, msg: dict, timeout: float) -> bytes:
        """Split one batch across the region's heads and merge the answers."""
        layer = msg["layer"]
        token_id = msg["token_id"]
        request_id = msg.get("request_id")
        entries = msg["entries"]
        fast = bool(msg.get("fast"))
        try:
            stage = self.dispatcher.submit_expert_stage(
                layer, msg["array"], [e.expert for e in entries],
                [e.gate for e in entries], token_id, timeout=timeout,
                replicas=[e.replica for e in entries], fast=fast)
        except ValueError as exc:
            return encode_batch_error(
                layer, token_id, ERR_UNKNOWN,
                [BatchFailure(e.expert, ERR_UNKNOWN, self.group_id)
                 for e in entries], str(exc)[:400], request_id=request_id)
        try:
            if fast:
                # Approximate on purpose: partials of partials.
                partial = stage.result(timeout)
            else:
                contributions = stage.contributions(timeout)
        except SubclusterError as exc:
            return self._forward_error(layer, token_id, entries, exc,
                                       request_id)
        except TransportError as exc:
            return self._link_error(layer, token_id, entries, exc, request_id)
        self._count(entries, fast)
        if fast:
            return encode_batch_response(layer, token_id, partial,
                                         len(entries), request_id=request_id)
        if len(contributions) != len(entries):
            # Cannot happen through a conforming head, and must not be passed
            # off as a complete answer if it somehow does.
            return encode_batch_error(
                layer, token_id, ERR_UNKNOWN, (),
                f"{self.group_id} gathered {len(contributions)} of "
                f"{len(entries)} contributions", request_id=request_id)
        positions = sorted(contributions)
        return encode_batch_contributions(
            layer, token_id, [contributions[p] for p in positions],
            [entries[p].expert for p in positions], request_id=request_id)

    def _forward_error(self, layer: int, token_id: int,
                       entries: Sequence[BatchEntry],
                       exc: SubclusterError,
                       request_id: Optional[int] = None) -> bytes:
        """Re-emit a subcluster's BERR upward, attribution intact."""
        failures = list(exc.failures)
        if not failures:
            failures = [BatchFailure(e.expert, exc.code, exc.node_id)
                        for e in entries]
        return encode_batch_error(layer, token_id, exc.code, failures,
                                  f"{self.group_id}/{exc.node_id}: "
                                  f"{exc.detail}"[:400],
                                  request_id=request_id)

    def _link_error(self, layer: int, token_id: int,
                    entries: Sequence[BatchEntry],
                    exc: TransportError,
                    request_id: Optional[int] = None) -> bytes:
        """A head server was unreachable, timed out, or dropped the link."""
        reason = failure_reason(exc)
        group_id = exc.node_id or self.group_id
        return encode_batch_error(
            layer, token_id, reason,
            [BatchFailure(e.expert, reason, group_id) for e in entries],
            f"{self.group_id}: {exc}"[:400], request_id=request_id)

    # -- teardown ----------------------------------------------------------
    def _close(self) -> None:
        self.dispatcher.close()
        if self._owns_transport:
            self.transport.close()


def serve_region(region_id: str, placement: ExpertPlacement,
                 plan: SubclusterPlan,
                 head_endpoints: Dict[str, EndpointSpec],
                 host: str = "0.0.0.0", port: int = 0,
                 timeout: float = DEFAULT_TIMEOUT,
                 retry_policy: Optional[LinkRetryPolicy] = None,
                 allow_fast: bool = True,
                 dedup: Optional[DedupCache] = None,
                 upstream_workers: int = DEFAULT_UPSTREAM_WORKERS
                 ) -> CoordinatorService:
    """Build (but do not start) a regional service on ``host:port``."""
    coordinator = RegionalCoordinator(region_id, placement, plan,
                                      head_endpoints, timeout=timeout,
                                      retry_policy=retry_policy,
                                      allow_fast=allow_fast, dedup=dedup)
    return CoordinatorService(coordinator, host=host, port=port,
                              upstream_workers=upstream_workers)


def region_addresses(services: Sequence[CoordinatorService]
                     ) -> Dict[str, List[Tuple[str, int]]]:
    """``region_id -> [endpoint, ...]`` for services sharing a coordinator.

    A region that runs one coordinator behind two sockets (primary and standby
    address) collapses to one entry with both endpoints, which is exactly the
    shape :class:`~.hierarchy.SubclusterTransport` wants.
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for service in services:
        out.setdefault(service.group_id, []).append(service.address)
    return out
