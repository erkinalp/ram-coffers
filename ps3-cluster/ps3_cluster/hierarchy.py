"""Layer-side hierarchical dispatch: one request per subcluster, not per expert.

The flat :class:`~.dispatch.DistributedExpertDispatcher` opens the top-k calls
from the layer coordinator straight to the consoles. That is the right shape for
a rack; it is the wrong shape for a Condor-sized farm, where 1,716 consoles sat
in subclusters of 22 behind coordinating servers precisely so the top of the
tree talked to ~78 heads instead of ~1,700 consoles (Barnell et al., IEEE HPEC
2012).

This module is the top half of that tree. It groups a token's top-k selection by
subcluster, sends **one** batched request per involved subcluster (the shared
activation once, plus the ``(expert, gate)`` list — see ``batch.py``), and sums
the returned partials in subcluster-id order. Both the flat dispatcher and the
canonical one-expert-per-layer-per-node placement are untouched: the same
``ExpertPlacement`` decides which console owns an expert, and a
``SubclusterPlan`` decides only which head server fronts that console.

Numerics
--------
Contributions are summed within a subcluster in ascending expert order and the
partials are then summed in ascending group-id order, which is exactly the
association :func:`~.subcluster.hierarchical_reduce` performs in-process. So
deploying real coordinator processes does not change the result by one bit. It
is *not* bit-identical to the flat left-to-right sum unless a single subcluster
covers the whole stage, because grouping re-associates the float32 additions;
see ``docs/PS3_CLUSTER_PORT.md``.

Failure semantics
-----------------
A subcluster either delivers every expert it was asked for or answers ``BERR``.
The stage therefore raises :class:`~.errors.SubclusterError` — naming the
failed experts and their consoles — rather than reducing a token through fewer
experts than the router chose. Whether to retry is the layer's decision:
``SubclusterError.safe_to_retry`` is true only when no expert can have run.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .batch import (ERR_UNKNOWN, MSG_BERR, MSG_BRSP, BatchEntry,
                    encode_batch_request, parse_batch_error,
                    parse_batch_response, NO_EXPERT)
from .dispatch import DEFAULT_FANOUT_WORKERS, ExpertPlacement
from .errors import NodeDisconnected, SubclusterError, TransportError
from .subcluster import SubclusterPlan
from .transport import (DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT, Endpoint,
                        PersistentSocketTransport)


class PendingBatch:
    """One batched subcluster request in flight."""

    __slots__ = ("group_id", "entries", "_frame")

    def __init__(self, group_id: str, entries: Sequence[BatchEntry], frame):
        self.group_id = group_id
        self.entries = list(entries)
        self._frame = frame

    def done(self) -> bool:
        return self._frame.done()

    def partial(self, timeout: Optional[float] = None) -> np.ndarray:
        """Await the subcluster's partial sum, validating completeness."""
        msg = self._frame.message(timeout)
        if msg["msg_type"] == MSG_BERR:
            err = parse_batch_error(msg)
            raise SubclusterError(self.group_id, err["code"], err["failures"],
                                  err["detail"])
        if msg["msg_type"] != MSG_BRSP:
            raise NodeDisconnected(self.group_id,
                                   f"unexpected msg_type {msg['msg_type']}")
        rsp = parse_batch_response(msg)
        if rsp["n_reduced"] != len(self.entries):
            # Refuse a partial that does not cover the whole request: a missing
            # expert must be an error, never a quietly smaller sum.
            raise SubclusterError(
                self.group_id, ERR_UNKNOWN, (),
                f"partial covers {rsp['n_reduced']} of {len(self.entries)} "
                f"requested experts")
        return rsp["array"]

    def cancel(self) -> None:
        self._frame.cancel()


class SubclusterTransport:
    """Persistent pooled P3XC channel from a layer coordinator to head servers.

    Thin wrapper over :class:`~.transport.PersistentSocketTransport`: the pool,
    the reader threads, the correlation and the timeout handling are the same
    code the flat path uses, only the frames differ. Correlation key is
    ``(layer, 0xFFFF, token_id)``, which every subcluster frame echoes, so
    several batches can be in flight on one connection and may answer out of
    order.
    """

    def __init__(self, group_endpoints: Dict[str, Endpoint],
                 timeout: float = DEFAULT_TIMEOUT,
                 connect_timeout: Optional[float] = None,
                 max_connections_per_group: int = DEFAULT_POOL_SIZE):
        self.timeout = timeout
        self._transport = PersistentSocketTransport(
            group_endpoints, timeout=timeout, connect_timeout=connect_timeout,
            max_connections_per_node=max_connections_per_group)

    # -- membership --------------------------------------------------------
    def add_group(self, group_id: str, endpoint: Endpoint) -> None:
        self._transport.add_endpoint(group_id, endpoint)

    def connection_count(self, group_id: Optional[str] = None) -> int:
        return self._transport.connection_count(group_id)

    @property
    def requests_sent(self) -> Dict[str, int]:
        return self._transport.requests_sent

    @property
    def connects_opened(self) -> Dict[str, int]:
        return self._transport.connects_opened

    # -- requests ----------------------------------------------------------
    def submit_batch(self, group_id: str, layer: int, token_id: int,
                     x: np.ndarray, entries: Sequence[BatchEntry],
                     deadline_ms: int = 0) -> PendingBatch:
        frame = encode_batch_request(layer, token_id, x, entries,
                                     deadline_ms=deadline_ms)
        key = (layer, NO_EXPERT, token_id)
        return PendingBatch(group_id, entries,
                            self._transport.submit_raw(group_id, key, frame))

    def run_batch(self, group_id: str, layer: int, token_id: int,
                  x: np.ndarray, entries: Sequence[BatchEntry],
                  timeout: Optional[float] = None) -> np.ndarray:
        deadline_ms = 0 if timeout is None else max(1, int(timeout * 1000))
        return self.submit_batch(group_id, layer, token_id, x, entries,
                                 deadline_ms).partial(timeout)

    # -- liveness / teardown ----------------------------------------------
    def ping(self, group_id: str, timeout: Optional[float] = None) -> float:
        """Heartbeat a head server. Says the coordinator is up, not its 22."""
        return self._transport.ping(group_id, timeout)

    def alive(self, group_id: str, timeout: Optional[float] = None) -> bool:
        return self._transport.alive(group_id, timeout)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "SubclusterTransport":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HierarchicalStage:
    """Handle for a top-k stage already fanned out across subclusters."""

    __slots__ = ("_futures", "_shape", "groups")

    def __init__(self, futures: Dict[str, object],
                 groups: List[Tuple[str, List[int]]], shape):
        self._futures = futures
        self._shape = shape
        #: ``[(group_id, [top-k positions])]``, in reduction order.
        self.groups = groups

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        """Sum the subclusters' partials in group-id order."""
        deadline = None if timeout is None else time.monotonic() + timeout
        partials: Dict[str, np.ndarray] = {}
        errors: Dict[str, BaseException] = {}
        for group_id, _positions in self.groups:
            remaining = (None if deadline is None
                         else max(0.0, deadline - time.monotonic()))
            try:
                partials[group_id] = self._futures[group_id].result(remaining)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[group_id] = exc
        if errors:
            # Every sibling batch has been awaited, so nothing is orphaned.
            raise errors[sorted(errors)[0]]
        if not partials:
            return np.zeros(self._shape, dtype=np.float32)
        out = None
        for group_id, _positions in self.groups:
            partial = partials[group_id]
            out = partial if out is None else out + partial
        return out


class HierarchicalExpertDispatcher:
    """Runs a layer's expert stage through subcluster coordinators.

    Same public shape as :class:`~.dispatch.DistributedExpertDispatcher` —
    ``submit_expert_stage`` / ``run_expert_stage`` over ``(expert_ids,
    gate_weights)`` — so a caller swaps one for the other. The difference is
    what goes on the wire: one batched request per subcluster involved in the
    token's routing, sent concurrently, instead of one request per expert.
    """

    def __init__(self, placement: ExpertPlacement, plan: SubclusterPlan,
                 transport: SubclusterTransport,
                 max_workers: Optional[int] = None):
        self.placement = placement
        self.plan = plan
        self.transport = transport
        self._max_workers = max_workers
        self._pool_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._token_ids = itertools.count(1)

    # -- lifecycle ---------------------------------------------------------
    def _executor(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=(DEFAULT_FANOUT_WORKERS
                                 if self._max_workers is None
                                 else self._max_workers),
                    thread_name_prefix="p3xc-hier")
            return self._pool

    def close(self) -> None:
        """Shut the fan-out pool down. Does not close the transport."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def __enter__(self) -> "HierarchicalExpertDispatcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- dispatch ----------------------------------------------------------
    def group_selection(self, layer: int, expert_ids: Sequence[int]
                        ) -> List[Tuple[str, List[int]]]:
        """``[(group_id, [positions])]`` for a top-k selection, group order."""
        node_ids = [self.placement.node_for(layer, int(e)) for e in expert_ids]
        return self.plan.group_by_subcluster(node_ids)

    def submit_expert_stage(self, layer: int, x: np.ndarray,
                            expert_ids: Sequence[int],
                            gate_weights: Sequence[float],
                            token_id: int = 0,
                            timeout: Optional[float] = None,
                            replicas: Optional[Sequence[int]] = None
                            ) -> HierarchicalStage:
        """Send one batch per involved subcluster, concurrently.

        ``replicas`` optionally pins a position to a standby console (0 = the
        canonical one); the hint travels in the batch entry and the head server
        starts its failover there.
        """
        if len(expert_ids) != len(gate_weights):
            raise ValueError("expert_ids and gate_weights length mismatch")
        if replicas is not None and len(replicas) != len(expert_ids):
            raise ValueError("expert_ids and replicas length mismatch")
        groups = self.group_selection(layer, expert_ids)
        futures: Dict[str, object] = {}
        if groups:
            executor = self._executor()
            deadline_ms = (0 if timeout is None
                           else max(1, int(timeout * 1000)))
            for group_id, positions in groups:
                entries = [
                    BatchEntry(expert=int(expert_ids[p]),
                               gate=float(gate_weights[p]),
                               replica=(0 if replicas is None
                                        else int(replicas[p])))
                    for p in positions]
                futures[group_id] = executor.submit(
                    self._call_subcluster, group_id, layer, token_id, x,
                    entries, deadline_ms, timeout)
        return HierarchicalStage(futures, groups, x.shape)

    def _call_subcluster(self, group_id: str, layer: int, token_id: int,
                         x: np.ndarray, entries: List[BatchEntry],
                         deadline_ms: int,
                         timeout: Optional[float]) -> np.ndarray:
        pending = self.transport.submit_batch(group_id, layer, token_id, x,
                                              entries, deadline_ms)
        return pending.partial(timeout)

    def run_expert_stage(self, layer: int, x: np.ndarray,
                         expert_ids: Sequence[int],
                         gate_weights: Sequence[float],
                         token_id: int = 0,
                         timeout: Optional[float] = None,
                         replicas: Optional[Sequence[int]] = None
                         ) -> np.ndarray:
        """Combine top-k expert outputs through the subcluster hierarchy."""
        stage = self.submit_expert_stage(layer, x, expert_ids, gate_weights,
                                         token_id, timeout, replicas)
        return stage.result(timeout)

    def next_token_id(self) -> int:
        """A fresh correlation id, for callers batching many tokens."""
        return next(self._token_ids) & 0x7FFFFFFF

    # -- liveness ----------------------------------------------------------
    def check_liveness(self, group_ids: Optional[Sequence[str]] = None,
                       timeout: Optional[float] = None) -> Dict[str, bool]:
        """PING each head server; ``group_id -> alive``."""
        groups = (self.plan.group_ids() if group_ids is None
                  else list(group_ids))
        if not groups:
            return {}
        executor = self._executor()
        futures = {g: executor.submit(self.transport.ping, g, timeout)
                   for g in groups}
        alive: Dict[str, bool] = {}
        for group_id, future in futures.items():
            try:
                future.result()
                alive[group_id] = True
            except TransportError:
                alive[group_id] = False
        return alive
