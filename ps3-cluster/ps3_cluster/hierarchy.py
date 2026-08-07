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

Tiers
-----
The downstream peer is a *coordinator*, not necessarily a subcluster head: point
this at regional coordinators with a :class:`~.subcluster.TieredPlan` and the
same code drives a three-tier tree (layer -> region -> subcluster -> console).
The rows come back tagged by expert either way, so bit identity does not care
how many tiers they crossed.

Failure semantics
-----------------
A coordinator either delivers every expert it was asked for or answers ``BERR``.
The stage therefore raises :class:`~.errors.SubclusterError` — naming the
failed experts and their consoles — rather than reducing a token through fewer
experts than the router chose. Whether to retry is the layer's decision:
``SubclusterError.safe_to_retry`` is true only when no expert can have run.

Link failover and retry
-----------------------
A logical coordinator may be configured with an ordered endpoint list (primary
first). Two things then protect a batch from a dead head:

* *Connection* failover, inside one attempt: opening a connection walks the list,
  so a refused primary costs a connect (``transport.py``).
* *Request* retry, across attempts, driven by :class:`LinkRetryPolicy`. The two
  failure classes are deliberately not treated alike:

  - **Safe before send** — no endpoint accepted, the pool was exhausted, or the
    write itself failed. P3XC frames are length-prefixed, so a partially written
    frame is never executed: nothing downstream ran, and retrying is free of
    consequence. Retried by default.
  - **Ambiguous** — the frame went out and then the answer did not come back
    (timeout, or the peer closed). The head may have driven all 22 consoles and
    died before replying. Retrying is therefore **at-least-once execution**, so
    it happens only with ``retry_ambiguous=True``.

Either way reduction is exactly once. Each attempt carries the same 64-bit
request id, so a retry that reaches the same coordinator process replays the
first attempt's answer from its dedup cache (``dedup.py``); an abandoned attempt
is cancelled, which retires its correlation key so a late reply is dropped rather
than reduced; and a reply whose echoed request id is not the one being awaited is
rejected outright.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .batch import (ERR_UNKNOWN, MAX_REQUEST_ID, MSG_BERR, MSG_BRSP, BatchEntry,
                    encode_batch_request, parse_batch_error,
                    parse_batch_response, NO_EXPERT)
from .dispatch import DEFAULT_FANOUT_WORKERS, ExpertPlacement
from .errors import NodeDisconnected, SubclusterError, TransportError
from .protocol import ProtocolError
from .subcluster import GroupingPlan
from .transport import (DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT, Endpoint,
                        EndpointSpec, PersistentSocketTransport)

#: Process-unique high bits, so two layer coordinators talking to one head server
#: cannot mint the same request id and dedup each other's batches.
_ID_SALT = random.getrandbits(24) << 40
_ID_COUNTER = itertools.count(1)


def next_request_id() -> int:
    """A fresh 64-bit id naming one logical batch, retries included."""
    return (_ID_SALT | (next(_ID_COUNTER) & 0xFFFFFFFFFF)) & MAX_REQUEST_ID


class LinkRetryPolicy(NamedTuple):
    """How hard to retry one coordinator link, and how much risk to accept.

    ``attempts`` counts total attempts per logical batch (``1`` disables retry).
    ``retry_ambiguous`` opts into retrying a batch that may already be running
    downstream, which is at-least-once *execution*; reduction stays exactly once.
    """

    attempts: int = 2
    retry_safe: bool = True
    retry_ambiguous: bool = False


class PendingBatch:
    """One batched subcluster request in flight."""

    __slots__ = ("group_id", "entries", "positions", "fast", "request_id",
                 "_frame")

    def __init__(self, group_id: str, entries: Sequence[BatchEntry],
                 positions: Sequence[int], frame, fast: bool = False,
                 request_id: Optional[int] = None):
        self.group_id = group_id
        self.entries = list(entries)
        #: The layer's top-k position each entry came from, entry order.
        self.positions = list(positions)
        self.fast = fast
        #: The id this attempt carried, reused by a retry of the same batch.
        self.request_id = request_id
        self._frame = frame

    def done(self) -> bool:
        return self._frame.done()

    @property
    def endpoint(self) -> Endpoint:
        """Which endpoint of the coordinator this attempt went to."""
        return self._frame.endpoint

    def _check_id(self, answered: Optional[int]) -> None:
        """Refuse an answer that names a different logical batch.

        A peer that does not echo ids at all (``None``) is accepted: correlation
        then rests on ``(layer, token_id)`` as it did before ids existed.
        """
        if (self.request_id is not None and answered is not None
                and answered != self.request_id):
            raise SubclusterError(
                self.group_id, ERR_UNKNOWN, (),
                f"answer carries request id {answered}, not {self.request_id}")

    def _response(self, timeout: Optional[float]) -> dict:
        msg = self._frame.message(timeout)
        if msg["msg_type"] == MSG_BERR:
            err = parse_batch_error(msg)
            self._check_id(err["request_id"])
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
        self._check_id(rsp["request_id"])
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

    def __init__(self, group_endpoints: Dict[str, EndpointSpec],
                 timeout: float = DEFAULT_TIMEOUT,
                 connect_timeout: Optional[float] = None,
                 max_connections_per_group: int = DEFAULT_POOL_SIZE,
                 retry_policy: Optional[LinkRetryPolicy] = None):
        self.timeout = timeout
        #: Default retry behaviour for :meth:`call_batch`.
        self.retry_policy = (LinkRetryPolicy() if retry_policy is None
                             else retry_policy)
        self._transport = PersistentSocketTransport(
            group_endpoints, timeout=timeout, connect_timeout=connect_timeout,
            max_connections_per_node=max_connections_per_group)

    # -- membership --------------------------------------------------------
    def add_group(self, group_id: str, endpoint: EndpointSpec) -> None:
        """Register a head server: one endpoint, or primary-first replicas."""
        self._transport.add_endpoint(group_id, endpoint)

    def endpoints_for(self, group_id: str) -> List[Endpoint]:
        return self._transport.endpoints_for(group_id)

    def mark_endpoint_dead(self, group_id: str, endpoint: Endpoint,
                           cooldown: Optional[float] = None) -> None:
        self._transport.mark_endpoint_dead(group_id, endpoint, cooldown)

    def endpoint_healthy(self, group_id: str, endpoint: Endpoint) -> bool:
        return self._transport.endpoint_healthy(group_id, endpoint)

    def connection_count(self, group_id: Optional[str] = None) -> int:
        return self._transport.connection_count(group_id)

    @property
    def requests_sent(self) -> Dict[str, int]:
        return self._transport.requests_sent

    @property
    def connects_opened(self) -> Dict[str, int]:
        return self._transport.connects_opened

    @property
    def connects_by_endpoint(self) -> Dict[Tuple[str, Endpoint], int]:
        return self._transport.connects_by_endpoint

    # -- requests ----------------------------------------------------------
    def submit_batch(self, group_id: str, layer: int, token_id: int,
                     x: np.ndarray, entries: Sequence[BatchEntry],
                     deadline_ms: int = 0,
                     positions: Optional[Sequence[int]] = None,
                     fast: bool = False,
                     request_id: Optional[int] = None) -> PendingBatch:
        """Send one BREQ. ``fast`` asks for a partial sum instead of rows.

        ``request_id`` names the logical batch on the wire; pass the *same* id
        again when retrying it so the coordinator can recognise the retry.
        """
        frame = encode_batch_request(layer, token_id, x, entries,
                                     deadline_ms=deadline_ms, fast=fast,
                                     request_id=request_id)
        key = (layer, NO_EXPERT, token_id)
        return PendingBatch(group_id, entries,
                            range(len(entries)) if positions is None
                            else positions,
                            self._transport.submit_raw(group_id, key, frame),
                            fast=fast, request_id=request_id)

    def call_batch(self, group_id: str, layer: int, token_id: int,
                   x: np.ndarray, entries: Sequence[BatchEntry],
                   positions: Optional[Sequence[int]] = None,
                   deadline_ms: int = 0, timeout: Optional[float] = None,
                   fast: bool = False,
                   policy: Optional[LinkRetryPolicy] = None,
                   request_id: Optional[int] = None):
        """One batch through one logical coordinator, with link failover.

        Returns the partial sum (fast) or ``{position: contribution}`` (exact).
        Every attempt reuses one request id; an abandoned attempt is cancelled so
        a late answer cannot be reduced. See the module docstring for which
        failures are retried and why.
        """
        policy = self.retry_policy if policy is None else policy
        if request_id is None:
            request_id = next_request_id()
        attempt = 0
        while True:
            attempt += 1
            last = attempt >= policy.attempts
            try:
                pending = self.submit_batch(group_id, layer, token_id, x,
                                            entries, deadline_ms,
                                            positions=positions, fast=fast,
                                            request_id=request_id)
            except TransportError:
                # Nothing was executed: no frame, or a partial one the peer
                # cannot parse. Retrying costs only the round trip.
                if last or not policy.retry_safe:
                    raise
                continue
            try:
                return (pending.partial(timeout) if fast
                        else pending.contributions(timeout))
            except SubclusterError:
                raise  # the coordinator answered; retrying will not help
            except TransportError:
                # Ambiguous: the batch may be running downstream right now.
                pending.cancel()
                self.mark_endpoint_dead(group_id, pending.endpoint)
                if last or not policy.retry_ambiguous:
                    raise

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
        answer = self.call_batch(group_id, layer, token_id, x, entries,
                                 deadline_ms=deadline_ms, timeout=timeout,
                                 fast=fast)
        if fast:
            return answer
        contributions = answer
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

    def __init__(self, placement: ExpertPlacement, plan: GroupingPlan,
                 transport: SubclusterTransport,
                 max_workers: Optional[int] = None, fast: bool = False,
                 retry_policy: Optional[LinkRetryPolicy] = None):
        self.placement = placement
        #: Any plan that can bucket console ids by the peer that fronts them: a
        #: :class:`~.subcluster.SubclusterPlan` for two tiers, a
        #: :class:`~.subcluster.TieredPlan` for three.
        self.plan = plan
        self.transport = transport
        #: Retry behaviour for the link below this dispatcher.
        self.retry_policy = retry_policy
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
        return self.transport.call_batch(group_id, layer, token_id, x, entries,
                                         positions=positions,
                                         deadline_ms=deadline_ms,
                                         timeout=timeout, fast=fast,
                                         policy=self.retry_policy)

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
