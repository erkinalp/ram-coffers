"""Distributed per-expert dispatch -- the PS3-cluster port of AirLLM #316.

AirLLM #316 (`_setup_expert_streaming` in ``airllm_base.py``) makes a 2.8T MoE
run on one small GPU by hooking every expert module: a forward *pre* hook loads
that expert's weights from the local safetensors shard onto the device, the
expert runs, and a *post* hook evicts them back to ``meta``. Because the model
only *calls* the experts a token routes to, only those hooks fire, so a token
materialises ~1 GB out of a ~55 GB layer.

This module keeps that structure exactly, and changes one thing: the expert's
weights are not streamed from local disk on demand -- they live **permanently in
the RAM of the PS3 that owns that expert**. So the pre-hook's "materialise this
expert" step becomes "send this expert's input activation to its owning node and
await the output". The coordinator never holds expert weights at all.

Correspondence with AirLLM #316:

    #316 (single box)                    this module (PS3 farm)
    ----------------------------------   -----------------------------------
    _expert_pre_hook: load_layer_subset  Transport.dispatch(layer, expert, x)
      -> move_layer_to_device            (remote node already has weights)
    expert.forward(x) on GPU             expert.forward(x) on the owning PS3
    _expert_post_hook: evict to meta     no-op (weights stay resident remotely)
    router calls only top-k experts      dispatcher sends only to top-k nodes

The dispatcher is framework-agnostic (operates on numpy arrays) so it can be
unit-tested with a loopback transport and no torch. See ``README.md`` for how it
slots underneath a transformers ``forward`` via the same hook points #316 uses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import socket
import threading
import time
import numpy as np

from .errors import (NodeConnectError, NodeDisconnected, NodeError,
                     NodeTimeout, TransportError)
from .protocol import (encode, decode, MSG_REQ, MSG_RSP, MSG_ERR, ProtocolError,
                       read_frame)
from .subcluster import SubclusterPlan, hierarchical_reduce


class ExpertPlacement:
    """Maps (layer, expert) -> node id, and back. This *is* the routing table
    the coordinator uses instead of #316's on-disk shard offsets."""

    def __init__(self) -> None:
        self._to_node: Dict[Tuple[int, int], str] = {}
        self._to_expert: Dict[str, Tuple[int, int]] = {}
        self._replicas: Dict[Tuple[int, int], List[str]] = {}

    def assign(self, layer: int, expert: int, node_id: str) -> None:
        self._to_node[(layer, expert)] = node_id
        self._to_expert[node_id] = (layer, expert)

    def assign_replica(self, layer: int, expert: int, node_id: str) -> None:
        """Register a standby console holding the same expert.

        Replicas are *optional* and additive: the canonical 1-expert x 1-layer
        / node placement decided by :meth:`assign` is unchanged, and a replica
        is only ever contacted when the primary fails in a way the retry policy
        deems safe (see :class:`RetryPolicy`).
        """
        if node_id == self._to_node.get((layer, expert)):
            raise ValueError("replica must differ from the primary node")
        replicas = self._replicas.setdefault((layer, expert), [])
        if node_id not in replicas:
            replicas.append(node_id)
        self._to_expert.setdefault(node_id, (layer, expert))

    def node_for(self, layer: int, expert: int) -> str:
        return self._to_node[(layer, expert)]

    def replicas_for(self, layer: int, expert: int) -> List[str]:
        return list(self._replicas.get((layer, expert), ()))

    def nodes_for(self, layer: int, expert: int) -> List[str]:
        """Primary first, then replicas: the failover order for one expert."""
        return [self.node_for(layer, expert)] + self.replicas_for(layer, expert)

    def expert_on(self, node_id: str) -> Tuple[int, int]:
        return self._to_expert[node_id]

    def node_ids(self, layer: Optional[int] = None) -> List[str]:
        """Sorted primary node ids, optionally restricted to one layer."""
        return sorted(node for (l, _e), node in self._to_node.items()
                      if layer is None or l == layer)

    def __len__(self) -> int:
        return len(self._to_node)

    @classmethod
    def one_per_node(cls, n_layers: int, experts_per_layer: int,
                     prefix: str = "ps3") -> "ExpertPlacement":
        """Canonical 1-expert x 1-layer / node assignment."""
        p = cls()
        for layer in range(n_layers):
            for e in range(experts_per_layer):
                p.assign(layer, e, f"{prefix}-L{layer:03d}-E{e:04d}")
        return p


class Transport:
    """Send an expert's input activation to its node and get the output back.

    Implementations must be safe to call from several coordinator threads at
    once: the dispatcher fans the top-k calls out concurrently.
    """

    def dispatch(self, node_id: str, layer: int, expert: int,
                 token_id: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ping(self, node_id: str,
             timeout: Optional[float] = None) -> float:
        """Liveness probe; returns the round-trip time in seconds.

        Optional: only transports that hold a channel to the node can answer
        it (see ``transport.PersistentSocketTransport``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support PING/PONG liveness")

    def close(self) -> None:  # pragma: no cover - optional
        pass


class LoopbackTransport(Transport):
    """In-process transport for tests / a single-box simulation.

    Holds each expert's forward function directly, so a dispatch is a plain call.
    Serialises through the wire protocol so the big-endian round-trip is
    exercised in tests exactly as it would be over a socket."""

    def __init__(self, experts: Dict[str, Callable[[np.ndarray], np.ndarray]]):
        self._experts = experts
        self.dispatch_log: List[Tuple[str, int, int]] = []

    def dispatch(self, node_id, layer, expert, token_id, x):
        self.dispatch_log.append((node_id, layer, expert))
        frame = encode(MSG_REQ, layer, expert, token_id, x)
        # Simulate the node parsing the request off the wire.
        req = decode(frame[4:])
        y = self._experts[node_id](req["array"])
        rsp = encode(MSG_RSP, layer, expert, token_id, np.ascontiguousarray(y))
        return decode(rsp[4:])["array"]


class SocketTransport(Transport):
    """TCP transport to real expert nodes, one short-lived connection per
    dispatch. Kept for compatibility (and because it needs no node-side state);
    ``transport.PersistentSocketTransport`` is the pooled, multiplexing
    replacement and is a drop-in for it.

    Failures are reported with the node-attributed exceptions from
    ``errors.py`` (all of which are ``RuntimeError`` subclasses, as the original
    ``ERR`` handling raised) so a retry policy can classify them."""

    def __init__(self, endpoints: Dict[str, Tuple[str, int]], timeout: float = 30.0):
        self._endpoints = endpoints
        self._timeout = timeout

    def dispatch(self, node_id, layer, expert, token_id, x):
        host, port = self._endpoints[node_id]
        try:
            with socket.create_connection((host, port),
                                          timeout=self._timeout) as s:
                s.sendall(encode(MSG_REQ, layer, expert, token_id, x))
                msg = read_frame(s.recv)
        except (OSError, ProtocolError) as exc:
            raise NodeConnectError(node_id, f"dispatch failed: {exc}") from exc
        if msg["msg_type"] == MSG_ERR:
            raise NodeError(node_id, msg["layer"], msg["expert"],
                            msg["token_id"])
        return msg["array"]


class RetryPolicy:
    """When a failed expert call may be re-sent, and to whom.

    Retries are opt-in per failure class because an expert contribution must
    never be counted twice:

    * ``attempts`` bounds the total tries per expert (1 = no retry).
    * A failure whose exception is ``safe_to_retry`` (see ``errors.py``) never
      reached the node, so retrying keeps **at-most-once** semantics.
    * ``retry_on_node_error`` retries an explicit ``ERR`` frame. An ``ERR``
      carries no output activation, so this also cannot double-count; it is off
      by default because the same node will usually fail the same way.
    * ``retry_on_timeout`` retries when a node's deadline expires before any
      response arrives. The request may still be executing, so this is
      **at-least-once** execution (the *result* is still reduced exactly once).
    * ``retry_on_disconnect`` retries when a connection drops while requests are
      in flight. Like timeout, this can execute the expert twice; the first
      arriving response is the one that is used.
    * ``use_replicas`` sends the retry to the next replica endpoint registered
      with :meth:`ExpertPlacement.assign_replica`, falling back to the primary
      when no replica exists.
    """

    def __init__(self, attempts: int = 2, retry_on_node_error: bool = False,
                 retry_on_timeout: bool = False,
                 retry_on_disconnect: bool = False, use_replicas: bool = True):
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self.attempts = attempts
        self.retry_on_node_error = retry_on_node_error
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_disconnect = retry_on_disconnect
        self.use_replicas = use_replicas

    def should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, NodeError):
            return self.retry_on_node_error
        if isinstance(exc, TransportError):
            if exc.safe_to_retry:
                return True
            if isinstance(exc, NodeDisconnected):
                return self.retry_on_disconnect
            if isinstance(exc, NodeTimeout):
                return self.retry_on_timeout
            return False
        return False

    @classmethod
    def none(cls) -> "RetryPolicy":
        """No retries at all: strict at-most-once, one attempt per expert."""
        return cls(attempts=1)


#: The default: retry only failures that provably never reached a node.
SAFE_RETRY = RetryPolicy()

#: Fan-out threads when ``max_workers`` is unset. Comfortably above K3's
#: top-16 so a stage never partially serialises; threads are idle-blocked on
#: sockets, not CPU-bound.
DEFAULT_FANOUT_WORKERS = 64


class DispatchStage:
    """Handle for a top-k expert stage whose calls are already in flight."""

    __slots__ = ("_futures", "_node_ids", "_plan", "_shape", "attempts")

    def __init__(self, futures, node_ids: List[str],
                 plan: Optional[SubclusterPlan], shape,
                 attempts: Dict[int, List[str]]):
        self._futures = futures
        self._node_ids = node_ids
        self._plan = plan
        self._shape = shape
        #: position -> nodes actually contacted, in order (failover audit).
        self.attempts = attempts

    @property
    def node_ids(self) -> List[str]:
        """The primary node contacted per top-k position."""
        return list(self._node_ids)

    def gather(self, timeout: Optional[float] = None
               ) -> Tuple[Dict[int, np.ndarray], Dict[int, BaseException]]:
        """Await every call, returning contributions and per-position errors.

        Every sibling call is awaited even after one fails, so no thread or
        socket is orphaned. The failure map is keyed by top-k position, which is
        what a subcluster coordinator needs to name the experts it could not
        deliver (see ``coordinator.py``). A call that outlives the stage deadline
        is reported as :class:`~.errors.NodeTimeout` against the node it was
        sent to; the request itself is left to the transport, which retires the
        correlation key on its own deadline and drops any late response rather
        than reducing it.
        """
        contributions: Dict[int, np.ndarray] = {}
        errors: Dict[int, BaseException] = {}
        # One deadline for the whole stage, not per expert.
        deadline = None if timeout is None else time.monotonic() + timeout
        for position, future in sorted(self._futures.items()):
            try:
                contributions[position] = future.result(
                    None if deadline is None
                    else max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                node = (self._node_ids[position]
                        if position < len(self._node_ids) else "?")
                errors[position] = NodeTimeout(
                    node, f"expert call did not finish within the stage "
                          f"deadline of {timeout} s")
            except BaseException as exc:  # noqa: BLE001 - returned to caller
                errors[position] = exc
        return contributions, errors

    def reduce(self, contributions: Dict[int, np.ndarray]) -> np.ndarray:
        """Sum contributions in the stage's fixed order."""
        if not contributions:
            return np.zeros(self._shape, dtype=np.float32)
        if self._plan is not None:
            return hierarchical_reduce(contributions, self._node_ids,
                                       self._plan)
        ordered = sorted(contributions)
        out = contributions[ordered[0]]
        for position in ordered[1:]:
            out = out + contributions[position]
        return out

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        """Reduce the fan-out once every expert has answered.

        Each contribution ``gate_j * y_j`` is scaled on the worker thread as it
        arrives; the *summation* then happens in a fixed order (ascending top-k
        position, or subcluster-by-subcluster for a hierarchical plan) so the
        result is bit-for-bit reproducible and, in the flat case, identical to
        the old serial implementation.
        """
        contributions, errors = self.gather(timeout)
        if errors:
            raise errors[min(errors)]
        return self.reduce(contributions)


class DistributedExpertDispatcher:
    """Runs one MoE layer's expert stage across the cluster.

    Given the router's decision for a token (which experts, with what gate
    weights), dispatch to exactly those experts' nodes and combine the outputs.
    This is the distributed equivalent of the model invoking the selected expert
    submodules -- only the chosen nodes ever do work, which is the whole reason
    a ~82k-expert model is affordable.

    The top-k calls **fan out concurrently**: with top-16 routing the stage
    costs one expert's latency plus reduction, not sixteen serial round trips.
    This mirrors how ALF pushes a batch of work blocks into all accelerators'
    queues and reduces the outputs as they complete, rather than driving one
    accelerator at a time (ALF Programmer's Guide, SDK 3.0). Concurrency is
    off by default only in the degenerate ``max_workers=1`` case; the public
    synchronous API (``run_expert_stage``) is unchanged.
    """

    def __init__(self, placement: ExpertPlacement, transport: Transport,
                 max_workers: Optional[int] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 subclusters: Optional[SubclusterPlan] = None):
        self.placement = placement
        self.transport = transport
        self.retry_policy = SAFE_RETRY if retry_policy is None else retry_policy
        self.subclusters = subclusters
        self._max_workers = max_workers
        self._pool_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None

    # -- lifecycle ---------------------------------------------------------
    def _executor(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=(DEFAULT_FANOUT_WORKERS
                                 if self._max_workers is None
                                 else self._max_workers),
                    thread_name_prefix="p3xc-dispatch")
            return self._pool

    def close(self) -> None:
        """Shut the fan-out pool down. Does not close the transport."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def __enter__(self) -> "DistributedExpertDispatcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- dispatch ----------------------------------------------------------
    def _candidates(self, layer: int, expert: int,
                    replica: int = 0) -> List[str]:
        """Failover order for one expert, optionally pinned to a replica.

        ``replica=0`` is the canonical console; a caller that already knows the
        primary is down (a subcluster batch may carry that hint per entry) can
        start further along the list instead of paying the failed attempt again.
        """
        candidates = (self.placement.nodes_for(layer, expert)
                      if self.retry_policy.use_replicas
                      else [self.placement.node_for(layer, expert)])
        if replica:
            if replica >= len(candidates):
                raise ValueError(
                    f"expert {expert} on layer {layer} has no replica "
                    f"{replica} ({len(candidates) - 1} configured)")
            candidates = candidates[replica:]
        return candidates

    def _call_expert(self, layer: int, expert: int, token_id: int,
                     x: np.ndarray, gate: float,
                     trail: List[str], replica: int = 0) -> np.ndarray:
        """One expert call with the retry/failover policy applied."""
        policy = self.retry_policy
        candidates = self._candidates(layer, expert, replica)
        last: Optional[BaseException] = None
        for attempt in range(policy.attempts):
            node = candidates[min(attempt, len(candidates) - 1)]
            trail.append(node)
            try:
                y = self.transport.dispatch(node, layer, expert, token_id, x)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                last = exc
                if attempt + 1 >= policy.attempts or not policy.should_retry(exc):
                    raise
                continue
            return (gate * y).astype(np.float32)
        raise last  # pragma: no cover - loop always returns or raises

    def submit_expert_stage(self, layer: int, x: np.ndarray,
                            expert_ids: Sequence[int],
                            gate_weights: Sequence[float],
                            token_id: int = 0,
                            replicas: Optional[Sequence[int]] = None
                            ) -> DispatchStage:
        """Fan out the top-k calls and return without waiting for them.

        The optional asynchronous API: callers that overlap layers or batch
        tokens can hold several stages in flight. ``DispatchStage.result()``
        performs the reduction. ``replicas`` optionally pins each position to a
        standby console (0 = the canonical one), which is how a subcluster batch
        forwards the replica hint its entries carry.
        """
        if len(expert_ids) != len(gate_weights):
            raise ValueError("expert_ids and gate_weights length mismatch")
        if replicas is not None and len(replicas) != len(expert_ids):
            raise ValueError("expert_ids and replicas length mismatch")
        hints = ([0] * len(expert_ids) if replicas is None
                 else [int(r) for r in replicas])
        node_ids = [self._candidates(layer, int(e), r)[0]
                    for e, r in zip(expert_ids, hints)]
        attempts: Dict[int, List[str]] = {}
        futures = {}
        if expert_ids:
            executor = self._executor()
            for position, (expert, gate) in enumerate(zip(expert_ids,
                                                          gate_weights)):
                trail: List[str] = []
                attempts[position] = trail
                futures[position] = executor.submit(
                    self._call_expert, layer, int(expert), token_id, x,
                    float(gate), trail, hints[position])
        return DispatchStage(futures, node_ids, self.subclusters, x.shape,
                             attempts)

    def run_expert_stage(self, layer: int, x: np.ndarray,
                         expert_ids: Sequence[int],
                         gate_weights: Sequence[float],
                         token_id: int = 0,
                         timeout: Optional[float] = None) -> np.ndarray:
        """Combine top-k expert outputs: sum_j gate_j * expert_j(x).

        Synchronous, exactly as before; the difference is that the k calls are
        in flight together. If any expert fails, the remaining calls are still
        awaited (so no thread or socket is orphaned) and the first failure is
        raised.
        """
        stage = self.submit_expert_stage(layer, x, expert_ids, gate_weights,
                                         token_id)
        return stage.result(timeout)

    def active_nodes_for(self, layer: int,
                         expert_ids: Sequence[int]) -> List[str]:
        return [self.placement.node_for(layer, e) for e in expert_ids]

    # -- liveness ----------------------------------------------------------
    def check_liveness(self, node_ids: Iterable[str],
                       timeout: Optional[float] = None) -> Dict[str, bool]:
        """PING every named node; ``node_id -> alive``.

        Requires a transport with a ``ping`` method (the persistent transport).
        """
        nodes = list(node_ids)
        if not nodes:
            return {}
        executor = self._executor()
        futures = {node: executor.submit(self.transport.ping, node, timeout)
                   for node in nodes}
        alive: Dict[str, bool] = {}
        for node, future in futures.items():
            try:
                future.result()
                alive[node] = True
            except TransportError:
                alive[node] = False
        return alive
