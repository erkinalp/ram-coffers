"""Coordinator processes: the head server for 22 consoles, and its base class.

The base class is the reusable half. A coordinator is anything that accepts a
batched BREQ, drives *something* below it, and answers with either every
requested contribution or a structured BERR; whether "below" means consoles
(:class:`SubclusterCoordinator`, here) or more coordinators
(:class:`~.regional.RegionalCoordinator`) is the only difference, so the tier
count is a deployment choice rather than a protocol one.

Condor, the 1,716-console AFRL PS3 cluster, was wired as **subclusters of 22
PlayStation 3s behind a coordinating server**, and the heads aggregated results
upward (Barnell et al., IEEE HPEC 2012). URI's Gravity Grid and the PS3
lattice-Boltzmann work likewise interpose a supervising process between the
network and the consoles (Nomura et al., IJCS 2008). ``subcluster.py`` models
that grouping as data; this module is the *process*: a service that a layer
coordinator talks to instead of talking to 22 consoles itself.

    layer coordinator                 (one host)
        |  BREQ: activation once + [(expert, gate), ...]
        v
    SubclusterCoordinator            (one head server per 22 consoles)
        |  P3XC REQ, pooled persistent connections, fanned out concurrently
        v
    expert workers                   (22 PS3s, one expert each)
        |  P3XC RSP
        v
    SubclusterCoordinator  -- BRSP: gate_j * y_j per expert --> layer coordinator

Why it matters on this hardware: the head server absorbs the k-way fan-out, so
the layer coordinator holds one connection per subcluster instead of one per
console and sends the activation once per subcluster — the same reason ALF has
the host enqueue one work-block list per accelerator rather than driving each
SPE individually (ALF Programmer's Guide, SDK 3.0).

Everything below the coordinator is reused, not reimplemented: the pooled
persistent transport (``transport.py``), the concurrent fan-out and the
retry/replica policy (``dispatch.py``).

Execution vs reduction semantics
--------------------------------
A subcluster answers a batch with **either** a result covering every requested
expert **or** a ``BERR`` naming the experts that failed and the consoles they
live on. It never returns a short answer, so the layer coordinator cannot
silently reduce a token through k-1 experts.

By default the answer is **one weighted contribution per expert**
(``gate_j * y_j``, tagged with expert ``j``) and no summation happens here: the
layer accumulates every subcluster's contributions strictly in top-k order, which
is the only way the hierarchy can be bit-identical to the flat dispatcher — a
token's positions interleave across subclusters, so folding a subset into one
fp32 partial would re-associate the additions. A request that sets
``REQ_FLAG_FAST`` instead gets one partial sum per subcluster (*fast* mode): less
upstream bandwidth, but a changed reduction order, so logits and token choices
can differ.

Retries inside the subcluster follow ``RetryPolicy`` (default: only failures that
provably never reached a console), and a late response to an abandoned request is
dropped by the transport rather than reduced — so a contribution lands exactly
once even when the underlying call was executed twice.
"""

from __future__ import annotations

import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .batch import (ERR_NODE_DISCONNECTED, ERR_NODE_ERROR, ERR_NODE_TIMEOUT,
                    ERR_NODE_UNREACHABLE, ERR_SHUTTING_DOWN, ERR_UNKNOWN,
                    ERR_UNKNOWN_EXPERT, ERR_BAD_REQUEST, BatchFailure,
                    encode_batch_contributions, encode_batch_error,
                    encode_batch_response, parse_batch_request)
from .dedup import DedupCache
from .dispatch import (DistributedExpertDispatcher, ExpertPlacement,
                       RetryPolicy)
from .errors import (NodeConnectError, NodeDisconnected, NodeError,
                     NodeTimeout, PoolExhausted, TransportClosed)
from .protocol import (MSG_BREQ, MSG_PING, MSG_PONG, ProtocolError, encode,
                       read_frame)
from .transport import DEFAULT_TIMEOUT, PersistentSocketTransport

#: Upstream frames handled concurrently per coordinator. A head server's work is
#: waiting on 22 consoles, so this is a blocking-IO bound, not a CPU one.
DEFAULT_UPSTREAM_WORKERS = 16


def failure_reason(exc: BaseException) -> int:
    """Map a transport failure onto a BERR reason code."""
    if isinstance(exc, (NodeConnectError, PoolExhausted)):
        return ERR_NODE_UNREACHABLE
    if isinstance(exc, NodeTimeout):
        return ERR_NODE_TIMEOUT
    if isinstance(exc, NodeError):
        return ERR_NODE_ERROR
    if isinstance(exc, NodeDisconnected):
        return ERR_NODE_DISCONNECTED
    if isinstance(exc, TransportClosed):
        return ERR_SHUTTING_DOWN
    if isinstance(exc, KeyError):
        return ERR_UNKNOWN_EXPERT
    return ERR_UNKNOWN


class BaseCoordinator:
    """The part of a coordinator that does not depend on what is below it.

    Owns the checks every tier owes its caller — am I shutting down, do I hold
    these experts, may I answer a fast request, how long have I got — and the
    dedup cache that makes a retried batch execute once per process. Subclasses
    implement :meth:`owns` and :meth:`_serve_batch`.
    """

    def __init__(self, group_id: str, timeout: float = DEFAULT_TIMEOUT,
                 allow_fast: bool = True,
                 dedup: Optional[DedupCache] = None) -> None:
        self.group_id = group_id
        self.timeout = timeout
        self.allow_fast = allow_fast
        #: Bounded request-id cache; ``None`` disables replay entirely.
        self.dedup = DedupCache() if dedup is None else dedup
        self._lock = threading.Lock()
        self._closed = False
        #: Observability, asserted by the tests.
        self.batches_served = 0
        self.experts_called = 0
        self.fast_batches = 0

    # -- to implement ------------------------------------------------------
    def owns(self, layer: int, expert: int) -> bool:
        raise NotImplementedError

    def _serve_batch(self, msg: dict, timeout: float) -> bytes:
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError

    # -- batch handling ----------------------------------------------------
    def handle_batch(self, msg: dict) -> bytes:
        """Run one decoded BREQ and return the BRSP (or BERR) frame to send.

        A batch that names a ``request_id`` runs under the dedup cache: a retry
        of the same logical batch (a reconnect, or the caller reaching this
        process through another of its endpoints) replays the first attempt's
        frame instead of fanning out again. A retry that lands on a *different*
        process cannot be recognised and is at-least-once execution; the caller
        still reduces exactly one answer.
        """
        layer = msg["layer"]
        token_id = msg["token_id"]
        request_id = msg.get("request_id")
        entries = msg["entries"]
        if self._closed:
            return encode_batch_error(layer, token_id, ERR_SHUTTING_DOWN,
                                      detail=f"{self.group_id} is shutting "
                                             f"down",
                                      request_id=request_id)
        missing = [e for e in entries if not self.owns(layer, e.expert)]
        if missing:
            return encode_batch_error(
                layer, token_id, ERR_UNKNOWN_EXPERT,
                [BatchFailure(e.expert, ERR_UNKNOWN_EXPERT, self.group_id)
                 for e in missing],
                f"{self.group_id} does not hold "
                f"{len(missing)} of {len(entries)} requested experts",
                request_id=request_id)
        if msg.get("fast") and not self.allow_fast:
            return encode_batch_error(
                layer, token_id, ERR_BAD_REQUEST,
                detail=f"{self.group_id} does not serve fast "
                       f"(partial-sum) batches",
                request_id=request_id)
        timeout = self.timeout
        if msg.get("deadline_ms"):
            timeout = min(timeout, msg["deadline_ms"] / 1000.0)
        try:
            return self.dedup.run(request_id,
                                  lambda: self._serve_batch(msg, timeout),
                                  timeout=timeout)
        except TimeoutError as exc:
            return encode_batch_error(layer, token_id, ERR_NODE_TIMEOUT,
                                      detail=str(exc)[:400],
                                      request_id=request_id)

    def _count(self, entries: Sequence[object], fast: bool) -> None:
        with self._lock:
            self.batches_served += 1
            self.experts_called += len(entries)
            if fast:
                self.fast_batches += 1

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._close()


class SubclusterCoordinator(BaseCoordinator):
    """Fans a batch out to this subcluster's consoles and answers upstream.

    Parameters
    ----------
    group_id:
        Subcluster id, used to attribute failures upstream.
    placement:
        Placement restricted to this subcluster's experts (primaries, plus any
        replicas). Canonical placement is unchanged: this is the same
        ``ExpertPlacement`` table, sliced.
    endpoints:
        ``node_id -> (host, port)`` for those consoles.
    timeout:
        Ceiling on one downstream expert call; a request's ``deadline_ms``
        lowers it further.
    allow_fast:
        Whether to honour ``REQ_FLAG_FAST`` and answer with a single partial sum.
        True by default (a layer has to ask for it); set False on a head server
        that must never hand back a re-associated sum.
    """

    def __init__(self, group_id: str, placement: ExpertPlacement,
                 endpoints: Dict[str, Tuple[str, int]],
                 timeout: float = DEFAULT_TIMEOUT,
                 max_workers: Optional[int] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 transport: Optional[PersistentSocketTransport] = None,
                 allow_fast: bool = True,
                 dedup: Optional[DedupCache] = None):
        super().__init__(group_id, timeout=timeout, allow_fast=allow_fast,
                         dedup=dedup)
        self.placement = placement
        self._owns_transport = transport is None
        self.transport = (PersistentSocketTransport(endpoints, timeout=timeout)
                          if transport is None else transport)
        self.dispatcher = DistributedExpertDispatcher(
            placement, self.transport, max_workers=max_workers,
            retry_policy=retry_policy)

    # -- membership --------------------------------------------------------
    def member_nodes(self, layer: Optional[int] = None) -> List[str]:
        return self.placement.node_ids(layer)

    def owns(self, layer: int, expert: int) -> bool:
        try:
            self.placement.node_for(layer, expert)
        except KeyError:
            return False
        return True

    def check_members(self, layer: Optional[int] = None,
                      timeout: Optional[float] = None) -> Dict[str, bool]:
        """PING every console in the subcluster; ``node_id -> alive``."""
        return self.dispatcher.check_liveness(self.member_nodes(layer), timeout)

    # -- batch handling ----------------------------------------------------
    def _serve_batch(self, msg: dict, timeout: float) -> bytes:
        """Fan one batch out to the consoles and encode the answer."""
        layer = msg["layer"]
        token_id = msg["token_id"]
        request_id = msg.get("request_id")
        entries = msg["entries"]
        fast = bool(msg.get("fast"))
        try:
            # A non-zero replica hint pins that entry to a standby console; an
            # unconfigured hint is a request error, not a node failure.
            stage = self.dispatcher.submit_expert_stage(
                layer, msg["array"], [e.expert for e in entries],
                [e.gate for e in entries], token_id,
                replicas=[e.replica for e in entries])
        except ValueError as exc:
            return encode_batch_error(
                layer, token_id, ERR_BAD_REQUEST,
                [BatchFailure(e.expert, ERR_BAD_REQUEST, self.group_id)
                 for e in entries if e.replica], str(exc)[:400],
                request_id=request_id)
        contributions, errors = stage.gather(timeout)
        self._count(entries, fast)
        if errors:
            failures = []
            for position in sorted(errors):
                trail = stage.attempts.get(position) or [self.group_id]
                failures.append(BatchFailure(entries[position].expert,
                                             failure_reason(errors[position]),
                                             trail[-1]))
            return encode_batch_error(layer, token_id, failures[0].reason,
                                      failures,
                                      str(errors[min(errors)])[:400],
                                      request_id=request_id)
        if fast:
            return encode_batch_response(layer, token_id,
                                         stage.reduce(contributions),
                                         len(entries), request_id=request_id)
        # Exact mode: hand the weighted contributions up untouched, tagged with
        # their experts, and let the layer sum them in its own top-k order.
        positions = sorted(contributions)
        return encode_batch_contributions(
            layer, token_id, [contributions[p] for p in positions],
            [entries[p].expert for p in positions], request_id=request_id)

    # -- teardown ----------------------------------------------------------
    def _close(self) -> None:
        self.dispatcher.close()
        if self._owns_transport:
            self.transport.close()


class _UpstreamHandler(socketserver.BaseRequestHandler):
    """One persistent connection from a layer coordinator.

    Frames are read until the peer closes, and each batch is executed on the
    server's upstream pool, so a slow subcluster request does not block the next
    one on the same connection. Responses are therefore allowed to come back
    **out of order**; every frame echoes ``(layer, expert=0xFFFF, token_id)``,
    which is exactly the correlation key the pooled transport keys on.
    """

    def handle(self) -> None:
        server: "CoordinatorServer" = self.server  # type: ignore[assignment]
        send_lock = threading.Lock()
        inflight: List[object] = []
        while True:
            try:
                msg = read_frame(self.request.recv)
            except (ProtocolError, OSError):
                break  # peer closed, or framing is unrecoverable
            if msg["msg_type"] == MSG_PING:
                # Answered by the head server itself: a PONG means "this
                # coordinator is up", not "all 22 consoles are up" (use
                # SubclusterCoordinator.check_members for that).
                self._send(send_lock, encode(MSG_PONG, msg["layer"],
                                             msg["expert"], msg["token_id"],
                                             np.zeros(1, np.float32)))
                continue
            if msg["msg_type"] != MSG_BREQ:
                self._send(send_lock, encode_batch_error(
                    msg["layer"], msg["token_id"], ERR_BAD_REQUEST,
                    detail=f"unsupported msg_type {msg['msg_type']}"))
                continue
            try:
                request = parse_batch_request(msg)
            except ProtocolError as exc:
                # The frame boundary was intact, so keep the connection and let
                # the peer see exactly what it got wrong.
                self._send(send_lock, encode_batch_error(
                    msg["layer"], msg["token_id"], ERR_BAD_REQUEST,
                    detail=str(exc)))
                continue
            inflight.append(server.upstream_pool.submit(
                self._run, send_lock, request))
        for future in inflight:
            future.exception()  # never leave a batch running past the socket

    def _run(self, send_lock: threading.Lock, request: dict) -> None:
        server: "CoordinatorServer" = self.server  # type: ignore[assignment]
        try:
            frame = server.coordinator.handle_batch(request)
        except Exception as exc:  # noqa: BLE001 - must answer, never hang
            frame = encode_batch_error(request["layer"], request["token_id"],
                                       ERR_UNKNOWN, detail=repr(exc)[:400],
                                       request_id=request.get("request_id"))
        self._send(send_lock, frame)

    def _send(self, send_lock: threading.Lock, frame: bytes) -> None:
        with send_lock:
            try:
                self.request.sendall(frame)
            except OSError:
                pass  # peer went away; the read loop will notice


class CoordinatorServer(socketserver.ThreadingTCPServer):
    """Serves BREQ frames for any coordinator, subcluster or regional."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int,
                 coordinator: BaseCoordinator,
                 upstream_workers: int = DEFAULT_UPSTREAM_WORKERS):
        super().__init__((host, port), _UpstreamHandler)
        self.coordinator = coordinator
        self.upstream_pool = ThreadPoolExecutor(
            max_workers=upstream_workers,
            thread_name_prefix=f"p3xc-sc-{coordinator.group_id}")

    def server_close(self) -> None:
        super().server_close()
        self.upstream_pool.shutdown(wait=True)


class CoordinatorService:
    """A coordinator running in its own thread on a real socket.

    The deployable unit: ``run_subcluster.py`` and ``run_region.py`` start one
    per head server, and the tests start several in-process on loopback. Two
    services may share one coordinator, which is how a head server offers a
    primary and a standby address without duplicating its dedup state.
    """

    def __init__(self, coordinator: BaseCoordinator,
                 host: str = "127.0.0.1", port: int = 0,
                 upstream_workers: int = DEFAULT_UPSTREAM_WORKERS):
        self.coordinator = coordinator
        self.server = CoordinatorServer(host, port, coordinator,
                                        upstream_workers=upstream_workers)
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            name=f"p3xc-sc-accept-{coordinator.group_id}", daemon=True)

    @property
    def group_id(self) -> str:
        return self.coordinator.group_id

    @property
    def address(self) -> Tuple[str, int]:
        return self.server.server_address[:2]

    def start(self) -> "CoordinatorService":
        self._thread.start()
        return self

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, close_coordinator: bool = True) -> None:
        """Stop accepting, drain in-flight batches, release every socket.

        ``close_coordinator=False`` gives up only this listener, which is what a
        head server's second (standby) address needs: the coordinator behind it
        keeps serving the primary address.
        """
        self.stop_listening()
        if close_coordinator:
            self.coordinator.close()

    def stop_listening(self) -> None:
        """Release this socket, leaving the coordinator itself usable."""
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "CoordinatorService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


#: Phase-2 names, kept so existing deployments and tests still import them.
SubclusterServer = CoordinatorServer
SubclusterService = CoordinatorService


def serve_subcluster(group_id: str, placement: ExpertPlacement,
                     endpoints: Dict[str, Tuple[str, int]],
                     host: str = "0.0.0.0", port: int = 0,
                     timeout: float = DEFAULT_TIMEOUT,
                     retry_policy: Optional[RetryPolicy] = None,
                     allow_fast: bool = True,
                     dedup: Optional[DedupCache] = None) -> CoordinatorService:
    """Build (but do not start) a subcluster service."""
    coordinator = SubclusterCoordinator(group_id, placement, endpoints,
                                        timeout=timeout,
                                        retry_policy=retry_policy,
                                        allow_fast=allow_fast, dedup=dedup)
    return CoordinatorService(coordinator, host=host, port=port)
