"""Layer-side hierarchical dispatch: one request per subcluster, not per expert.

The flat :class:`~.dispatch.DistributedExpertDispatcher` opens the top-k calls
from the layer coordinator straight to the consoles. That is the right shape for
a rack; it is the wrong shape for a Condor-sized farm, where 1,716 consoles sat
in subclusters of 22 behind coordinating servers precisely so the top of the
tree talked to ~78 heads instead of ~1,700 consoles (Barnell et al., IEEE HPEC
2012).

This module is the top half of that tree. It groups a token's top-k selection by
subcluster, sends **one** batched request per involved subcluster (the shared
activation once, plus the ``(expert, gate)`` list — see ``batch.py``), and
combines the answers. Both the flat dispatcher and the canonical
one-expert-per-layer-per-node placement are untouched: the same
``ExpertPlacement`` decides which console owns an expert, and a
``SubclusterPlan`` decides only which head server fronts that console.

Numerics
--------
Default (exact) mode is **bit-identical to the flat dispatcher**. Each subcluster
returns its ``gate_j * y_j`` contributions individually, tagged with their
experts, and this module puts them back at their top-k positions and accumulates
ascending-position, left-to-right in float32 — the same operations in the same
order as :meth:`~.dispatch.DispatchStage.reduce`. Neither the grouping nor the
completion order affects the result, so ``np.array_equal`` holds against the flat
path. The price is upstream bandwidth: k activation-sized rows come back per
token instead of one per subcluster.

``fast=True`` opts into the old behaviour — each subcluster folds its experts into
one partial sum and this module adds the partials in group-id order. That saves
upstream bytes but re-associates the float32 additions whenever a token's
positions interleave across subclusters, so results are reproducible yet *not*
bit-identical, and the resulting logit differences can change which token is
sampled. It is never the default; see ``docs/PS3_CLUSTER_PORT.md``.

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
from .protocol import ProtocolError
from .subcluster import SubclusterPlan
from .transport import (DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT, Endpoint,
                        PersistentSocketTransport)


class PendingBatch:
    """One batched subcluster request in flight."""

    __slots__ = ("group_id", "entries", "positions", "fast", "_frame")

    def __init__(self, group_id: str, entries: Sequence[BatchEntry],
                 positions: Sequence[int], frame, fast: bool = False):
        self.group_id = group_id
        self.entries = list(entries)
        #: The layer's top-k position each entry came from, entry order.
        self.positions = list(positions)
        self.fast = fast
        self._frame = frame

    def done(self) -> bool:
        return self._frame.done()

    def _response(self, timeout: Optional[float]) -> dict:
        msg = self._frame.message(timeout)
        if msg["msg_type"] == MSG_BERR:
            err = parse_batch_error(msg)
            raise SubclusterError(self.group_id, err["code"], err["failures"],
                                  err["detail"])
        if msg["msg_type"] != MSG_BRSP:
            raise NodeDisconnected(self.group_id,
                                   f"unexpected msg_type {msg['msg_type']}")
        try:
            rsp = parse_batch_response(msg)
        except ProtocolError as exc:
            raise SubclusterError(self.group_id, ERR_UNKNOWN, (),
                                  str(exc)) from exc
        if rsp["n_reduced"] != len(self.entries):
            # Refuse an answer that does not cover the whole request: a missing
            # expert must be an error, never a quietly smaller sum.
            raise SubclusterError(
                self.group_id, ERR_UNKNOWN, (),
                f"answer covers {rsp['n_reduced']} of {len(self.entries)} "
                f"requested experts")
        return rsp

    def partial(self, timeout: Optional[float] = None) -> np.ndarray:
        """Await a *fast* reply: the subcluster's single partial sum."""
        rsp = self._response(timeout)
        if rsp["per_expert"]:
            raise SubclusterError(
                self.group_id, ERR_UNKNOWN, (),
                "subcluster returned per-expert contributions for a fast "
                "request")
        return rsp["array"]

    def contributions(self, timeout: Optional[float] = None
                      ) -> Dict[int, np.ndarray]:
        """Await an *exact* reply: ``top-k position -> gate_j * y_j``.

        The rows are matched to positions through the expert tags rather than
        their order on the wire, so a head server is free to answer in whatever
        order its consoles finished.
        """
        rsp = self._response(timeout)
        if not rsp["per_expert"]:
            raise SubclusterError(
                self.group_id, ERR_UNKNOWN, (),
                "subcluster returned one partial sum for an exact request; it "
                "would re-associate the layer's reduction")
        position_of = {entry.expert: position
                       for entry, position in zip(self.entries,
                                                  self.positions)}
        rows = rsp["array"]
        out: Dict[int, np.ndarray] = {}
        for row, expert in enumerate(rsp["experts"]):
            position = position_of.get(expert)
            if position is None:
                raise SubclusterError(
                    self.group_id, ERR_UNKNOWN, (),
                    f"subcluster returned expert {expert}, which was not "
                    f"requested")
            out[position] = rows[row]
        return out

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
                     deadline_ms: int = 0,
                     positions: Optional[Sequence[int]] = None,
                     fast: bool = False) -> PendingBatch:
        """Send one BREQ. ``fast`` asks for a partial sum instead of rows."""
        frame = encode_batch_request(layer, token_id, x, entries,
                                     deadline_ms=deadline_ms, fast=fast)
        key = (layer, NO_EXPERT, token_id)
        return PendingBatch(group_id, entries,
                            range(len(entries)) if positions is None
                            else positions,
                            self._transport.submit_raw(group_id, key, frame),
                            fast=fast)

    def run_batch(self, group_id: str, layer: int, token_id: int,
                  x: np.ndarray, entries: Sequence[BatchEntry],
                  timeout: Optional[float] = None,
                  fast: bool = False) -> np.ndarray:
        """One batch in, one vector out.

        Exact by default: the rows come back per expert and are summed here in
        entry order. ``fast=True`` asks the head server for the partial sum
        instead, which re-associates the additions.
        """
        deadline_ms = 0 if timeout is None else max(1, int(timeout * 1000))
        pending = self.submit_batch(group_id, layer, token_id, x, entries,
                                    deadline_ms, fast=fast)
        if fast:
            return pending.partial(timeout)
        contributions = pending.contributions(timeout)
        ordered = sorted(contributions)
        out = contributions[ordered[0]]
        for position in ordered[1:]:
            out = out + contributions[position]
        return out

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

    __slots__ = ("_futures", "_shape", "_fast", "groups")

    def __init__(self, futures: Dict[str, object],
                 groups: List[Tuple[str, List[int]]], shape,
                 fast: bool = False):
        self._futures = futures
        self._shape = shape
        self._fast = fast
        #: ``[(group_id, [top-k positions])]``, in reduction order.
        self.groups = groups

    @property
    def fast(self) -> bool:
        """True if the subclusters were asked for partial sums."""
        return self._fast

    def _await_all(self, timeout: Optional[float]) -> Dict[str, object]:
        """Await every batch, then raise the first failure in group order."""
        deadline = None if timeout is None else time.monotonic() + timeout
        answers: Dict[str, object] = {}
        errors: Dict[str, BaseException] = {}
        for group_id, _positions in self.groups:
            remaining = (None if deadline is None
                         else max(0.0, deadline - time.monotonic()))
            try:
                answers[group_id] = self._futures[group_id].result(remaining)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[group_id] = exc
        if errors:
            # Every sibling batch has been awaited, so nothing is orphaned.
            raise errors[sorted(errors)[0]]
        return answers

    def contributions(self, timeout: Optional[float] = None
                      ) -> Dict[int, np.ndarray]:
        """``top-k position -> gate_j * y_j`` across every subcluster."""
        if self._fast:
            raise ValueError("a fast stage returns partial sums, not "
                             "per-expert contributions")
        merged: Dict[int, np.ndarray] = {}
        for group in self._await_all(timeout).values():
            merged.update(group)
        return merged

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        """Combine the subclusters' answers.

        Exact mode accumulates every contribution in ascending top-k position,
        which is the flat dispatcher's association bit for bit. Fast mode sums
        one partial per subcluster in group-id order instead.
        """
        if not self.groups:
            return np.zeros(self._shape, dtype=np.float32)
        if self._fast:
            partials = self._await_all(timeout)
            out = None
            for group_id, _positions in self.groups:
                partial = partials[group_id]
                out = partial if out is None else out + partial
            return out
        contributions = self.contributions(timeout)
        if not contributions:
            return np.zeros(self._shape, dtype=np.float32)
        ordered = sorted(contributions)
        out = contributions[ordered[0]]
        for position in ordered[1:]:
            out = out + contributions[position]
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
                 max_workers: Optional[int] = None, fast: bool = False):
        self.placement = placement
        self.plan = plan
        self.transport = transport
        #: Opt-in throughput mode: one partial sum per subcluster, which
        #: re-associates the fp32 reduction and can change token choices.
        self.fast = fast
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
                            replicas: Optional[Sequence[int]] = None,
                            fast: Optional[bool] = None
                            ) -> HierarchicalStage:
        """Send one batch per involved subcluster, concurrently.

        ``replicas`` optionally pins a position to a standby console (0 = the
        canonical one); the hint travels in the batch entry and the head server
        starts its failover there. ``fast`` overrides the dispatcher's mode for
        this stage.
        """
        use_fast = self.fast if fast is None else fast
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
                    entries, positions, deadline_ms, timeout, use_fast)
        return HierarchicalStage(futures, groups, x.shape, fast=use_fast)

    def _call_subcluster(self, group_id: str, layer: int, token_id: int,
                         x: np.ndarray, entries: List[BatchEntry],
                         positions: Sequence[int], deadline_ms: int,
                         timeout: Optional[float], fast: bool):
        pending = self.transport.submit_batch(group_id, layer, token_id, x,
                                              entries, deadline_ms,
                                              positions=positions, fast=fast)
        return pending.partial(timeout) if fast else pending.contributions(
            timeout)

    def run_expert_stage(self, layer: int, x: np.ndarray,
                         expert_ids: Sequence[int],
                         gate_weights: Sequence[float],
                         token_id: int = 0,
                         timeout: Optional[float] = None,
                         replicas: Optional[Sequence[int]] = None,
                         fast: Optional[bool] = None) -> np.ndarray:
        """Combine top-k expert outputs through the subcluster hierarchy.

        Bit-identical to ``DistributedExpertDispatcher.run_expert_stage`` unless
        fast mode is selected.
        """
        stage = self.submit_expert_stage(layer, x, expert_ids, gate_weights,
                                         token_id, timeout, replicas, fast)
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
