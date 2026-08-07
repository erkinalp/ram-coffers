"""Persistent, pooled P3XC transport for asynchronous expert coordination.

``SocketTransport`` (``dispatch.py``) opens a TCP connection per dispatch. At
Kimi-K3 scale that is one connect/teardown per expert per token per layer: with
top-16 routing over 92 layers a single token pays ~1,500 three-way handshakes,
and the coordinator cannot keep more than one expert call in flight per thread.

This module replaces that with the pattern IBM's Cell middleware used on exactly
this hardware: **persistent connections carrying a queue of work items**, with
responses correlated back to their requests rather than assumed to arrive in
order. ALF keeps long-lived work queues per accelerator and pushes work blocks
into them (ALF Programmer's Guide, SDK 3.0); DaCS keeps a persistent host↔
accelerator connection with asynchronous message send/receive and remote error
notification (DaCS Programmer's Guide, SDK 3.0). The PS3 HPC clusters that
inspired this port — the AFRL "Condor" cluster's 22-console subclusters behind
coordinating servers (Barnell et al., IEEE HPEC 2012), URI's Gravity Grid, and
the PS3 lattice-Boltzmann work (Nomura et al., IJCS 2008) — all rely on standing
inter-console channels with PPE-side supervision, never per-message connects.

Design
------
* One :class:`_Connection` owns one TCP socket, a write lock, and a background
  reader thread. Writes are serialised (a P3XC frame is written with a single
  ``sendall`` under the lock, so frames never interleave); reads happen only on
  the reader thread, which parks each decoded frame on the matching pending slot
  and wakes its waiter. Multiple requests may therefore be in flight on one
  socket and responses may come back **out of order**.
* Correlation identity is the frame triple ``(layer, expert, token_id)``, which
  every P3XC response already echoes, so no wire-format change is needed. Under
  the canonical one-expert-per-layer-per-node placement, a node only ever sees
  one ``(layer, expert)`` pair, so ``token_id`` alone distinguishes concurrent
  requests to it; the pool additionally refuses to put two requests with the
  same correlation key on the same connection. ``ERR`` frames from a worker that
  does not own the requested expert legitimately carry the *worker's* own
  ``(layer, expert)``, so an unmatched frame falls back to a unique-``token_id``
  match before being dropped.
* A node gets a bounded pool (``max_connections_per_node``) rather than a single
  socket: it caps the number of correlation-key collisions that must serialise,
  and lets the C worker — which processes requests sequentially on a connection
  (see ``ppu/expert_ppu.c``) — still be driven concurrently.
* A logical peer may have **several endpoints**, given in preference order:
  ``{"sc-0000": [("head-a", 8100), ("head-b", 8100)]}``. Opening a connection
  walks the list and stops at the first endpoint that accepts, so a dead primary
  costs one refused connect rather than the request. A failed connect (or an
  explicit :meth:`PersistentSocketTransport.mark_endpoint_dead` after a heartbeat
  failure) puts that endpoint in a short cooldown, which only reorders
  preference: every endpoint is still tried before the transport gives up, so
  correctness never depends on the health state being accurate. Consoles are
  configured the same way, but their standby copies stay modelled as replica
  *nodes* in ``ExpertPlacement`` — an expert's weights live on a particular
  console, so failing over to another console is a placement decision rather
  than a connection detail.

Failure semantics are documented in ``docs/PS3_CLUSTER_PORT.md``; briefly, a
timeout permanently retires that correlation key on that connection (so a late
response is recognised and dropped rather than mis-correlated) and is reported
as :class:`~.errors.NodeTimeout`, which the dispatcher treats as *not* safely
retryable by default because the expert may still be running.
"""

from __future__ import annotations

import errno
import itertools
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .dispatch import Transport
from .errors import (NodeConnectError, NodeDisconnected, NodeError,
                     NodeTimeout, PoolExhausted, TransportClosed,
                     TransportError)
from .protocol import (MSG_ERR, MSG_PING, MSG_PONG, MSG_REQ, MSG_RSP,
                       ProtocolError, decode, encode)

Endpoint = Tuple[str, int]
#: One endpoint, or an ordered primary-first list of them, for one logical peer.
EndpointSpec = Union[Endpoint, Sequence[Endpoint]]
CorrelationKey = Tuple[int, int, int]

DEFAULT_TIMEOUT = 30.0
DEFAULT_POOL_SIZE = 4

#: How long a failed endpoint is deprioritised. Only affects preference order.
DEFAULT_ENDPOINT_COOLDOWN = 5.0


def _as_endpoints(spec: EndpointSpec) -> List[Endpoint]:
    """Normalise ``(host, port)`` or a list of them into a preference list."""
    if (isinstance(spec, tuple) and len(spec) == 2
            and isinstance(spec[0], str) and isinstance(spec[1], int)):
        return [spec]
    endpoints = [(str(host), int(port)) for host, port in spec]
    if not endpoints:
        raise ValueError("a peer needs at least one endpoint")
    return endpoints


class _Pending:
    """One in-flight request awaiting its response frame."""

    __slots__ = ("key", "event", "message", "error")

    def __init__(self, key: CorrelationKey):
        self.key = key
        self.event = threading.Event()
        self.message: Optional[dict] = None
        self.error: Optional[BaseException] = None

    def complete(self, message: dict) -> None:
        self.message = message
        self.event.set()

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.event.set()


class _Connection:
    """One persistent socket to one node, with a background reader thread."""

    def __init__(self, node_id: str, endpoint: Endpoint,
                 connect_timeout: float):
        self.node_id = node_id
        self.endpoint = endpoint
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending: Dict[CorrelationKey, _Pending] = {}
        self._poisoned: set = set()
        self._closed = False
        self._recv_buf = b""
        try:
            self._sock = socket.create_connection(endpoint,
                                                  timeout=connect_timeout)
        except OSError as exc:
            raise NodeConnectError(node_id,
                                   f"connect to {endpoint[0]}:{endpoint[1]} "
                                   f"failed: {exc}") from exc
        self._sock.settimeout(None)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"p3xc-reader-{node_id}",
            daemon=True)
        self._reader.start()

    # -- state ------------------------------------------------------------
    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def in_flight(self) -> int:
        with self._state_lock:
            return len(self._pending)

    def holds_key(self, key: CorrelationKey) -> bool:
        """True if ``key`` is in flight *or* abandoned on this connection.

        An abandoned (timed-out) key stays reserved for the life of the
        connection: a late response still carrying it must be droppable, which
        is only sound if the key is never handed out again here.
        """
        with self._state_lock:
            return key in self._pending or key in self._poisoned

    # -- request/response -------------------------------------------------
    def send(self, key: CorrelationKey, frame: bytes) -> _Pending:
        """Register ``key`` and write ``frame``. Returns the pending slot."""
        pending = _Pending(key)
        with self._state_lock:
            if self._closed:
                raise TransportClosed(self.node_id, "connection closed")
            if key in self._pending or key in self._poisoned:
                raise ValueError(f"correlation key {key} already in flight")
            self._pending[key] = pending
        try:
            with self._write_lock:
                self._sock.sendall(frame)
        except OSError as exc:
            self._retire(NodeDisconnected(self.node_id, f"send failed: {exc}"))
            raise NodeDisconnected(self.node_id, f"send failed: {exc}") from exc
        return pending

    def wait(self, pending: _Pending, timeout: float) -> dict:
        if not pending.event.wait(timeout):
            # A response may still arrive here for a key we have given up on,
            # so retire the key (not the whole connection: sibling requests on
            # this socket are unaffected by one slow expert).
            with self._state_lock:
                self._pending.pop(pending.key, None)
                self._poisoned.add(pending.key)
            raise NodeTimeout(self.node_id,
                              f"no response for {pending.key} within "
                              f"{timeout:g}s")
        with self._state_lock:
            self._pending.pop(pending.key, None)
        if pending.error is not None:
            raise pending.error
        assert pending.message is not None
        return pending.message

    def cancel(self, pending: _Pending) -> None:
        with self._state_lock:
            self._pending.pop(pending.key, None)
            self._poisoned.add(pending.key)

    # -- reader -----------------------------------------------------------
    def _read_loop(self) -> None:
        try:
            while True:
                body = self._read_frame()
                if body is None:
                    self._retire(NodeDisconnected(self.node_id,
                                                  "connection closed by peer"))
                    return
                try:
                    msg = decode(body)
                except ProtocolError as exc:
                    self._retire(NodeDisconnected(self.node_id,
                                                  f"protocol error: {exc}"))
                    return
                self._deliver(msg)
        except OSError as exc:
            self._retire(NodeDisconnected(self.node_id, f"recv failed: {exc}"))

    def _read_frame(self) -> Optional[bytes]:
        head = self._read_exact(4)
        if head is None:
            return None
        (length,) = struct.unpack("!I", head)
        return self._read_exact(length)

    def _read_exact(self, n: int) -> Optional[bytes]:
        while len(self._recv_buf) < n:
            try:
                chunk = self._sock.recv(65536)
            except OSError as exc:
                if exc.errno == errno.EBADF or self.closed:
                    return None
                raise
            if not chunk:
                return None
            self._recv_buf += chunk
        out, self._recv_buf = self._recv_buf[:n], self._recv_buf[n:]
        return out

    def _deliver(self, msg: dict) -> None:
        key = (msg["layer"], msg["expert"], msg["token_id"])
        with self._state_lock:
            if key in self._poisoned:
                return  # late response to an abandoned request
            pending = self._pending.get(key)
            if pending is None:
                # An ERR from a worker that does not own the requested expert
                # echoes the worker's own (layer, expert); fall back to a
                # unique token_id match.
                candidates = [p for k, p in self._pending.items()
                              if k[2] == msg["token_id"]]
                if len(candidates) == 1:
                    pending = candidates[0]
        if pending is None:
            return  # response to an abandoned request: drop it
        pending.complete(msg)

    # -- teardown ---------------------------------------------------------
    def _retire(self, error: BaseException) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, {}
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        for p in pending.values():
            p.fail(error)

    def close(self) -> None:
        self._retire(TransportClosed(self.node_id, "transport closed"))
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)


class PersistentSocketTransport(Transport):
    """Thread-safe pooled P3XC transport with multiple requests in flight.

    Parameters
    ----------
    endpoints:
        ``node_id -> (host, port)``.
    timeout:
        Per-request response deadline in seconds.
    connect_timeout:
        TCP connect deadline; defaults to ``timeout``.
    max_connections_per_node:
        Bound on the per-node connection pool.
    """

    def __init__(self, endpoints: Dict[str, EndpointSpec],
                 timeout: float = DEFAULT_TIMEOUT,
                 connect_timeout: Optional[float] = None,
                 max_connections_per_node: int = DEFAULT_POOL_SIZE,
                 endpoint_cooldown: float = DEFAULT_ENDPOINT_COOLDOWN):
        if max_connections_per_node < 1:
            raise ValueError("max_connections_per_node must be >= 1")
        self._endpoints = {node: _as_endpoints(spec)
                           for node, spec in endpoints.items()}
        self._cooldown = endpoint_cooldown
        self._dead_until: Dict[Tuple[str, Endpoint], float] = {}
        self._timeout = timeout
        self._connect_timeout = (timeout if connect_timeout is None
                                 else connect_timeout)
        self._pool_size = max_connections_per_node
        self._lock = threading.Lock()
        self._pools: Dict[str, List[_Connection]] = {}
        self._closed = False
        self._ping_ids = itertools.count(1)
        # Observability, used by the tests to prove connection reuse.
        self.connects_opened: Dict[str, int] = {}
        self.connects_by_endpoint: Dict[Tuple[str, Endpoint], int] = {}
        self.requests_sent: Dict[str, int] = {}

    # -- endpoints --------------------------------------------------------
    def add_endpoint(self, node_id: str, endpoint: EndpointSpec) -> None:
        """Register one endpoint, or an ordered primary-first list of them."""
        with self._lock:
            self._endpoints[node_id] = _as_endpoints(endpoint)

    def endpoints_for(self, node_id: str) -> List[Endpoint]:
        """Every endpoint for a peer, in configured (preference) order."""
        try:
            return list(self._endpoints[node_id])
        except KeyError:
            raise NodeConnectError(node_id, "no endpoint configured") from None

    def endpoint_for(self, node_id: str) -> Endpoint:
        """The endpoint a new connection would prefer right now."""
        return self._preferred(node_id)[0]

    def _preferred(self, node_id: str) -> List[Endpoint]:
        """Endpoints with cooled-down ones moved to the back, order kept."""
        endpoints = self.endpoints_for(node_id)
        now = time.monotonic()
        with self._lock:
            healthy = [e for e in endpoints
                       if self._dead_until.get((node_id, e), 0.0) <= now]
        if not healthy or len(healthy) == len(endpoints):
            return endpoints
        return healthy + [e for e in endpoints if e not in healthy]

    def mark_endpoint_dead(self, node_id: str, endpoint: Endpoint,
                           cooldown: Optional[float] = None) -> None:
        """Deprioritise ``endpoint`` for a while (heartbeat/failover hook).

        Only preference is affected: no request fails because an endpoint is
        marked dead, and a marked endpoint is still tried once every other
        endpoint has refused.
        """
        with self._lock:
            self._dead_until[(node_id, endpoint)] = (
                time.monotonic() + (self._cooldown if cooldown is None
                                    else cooldown))

    def mark_endpoint_live(self, node_id: str, endpoint: Endpoint) -> None:
        with self._lock:
            self._dead_until.pop((node_id, endpoint), None)

    def endpoint_healthy(self, node_id: str, endpoint: Endpoint) -> bool:
        with self._lock:
            return self._dead_until.get((node_id, endpoint),
                                        0.0) <= time.monotonic()

    # -- pool -------------------------------------------------------------
    def _acquire(self, node_id: str, key: CorrelationKey) -> _Connection:
        """Pick (or open) a connection with ``key`` not already in flight."""
        endpoints = self._preferred(node_id)
        with self._lock:
            if self._closed:
                raise TransportClosed(node_id, "transport closed")
            pool = self._pools.setdefault(node_id, [])
            pool[:] = [c for c in pool if not c.closed]
            free = [c for c in pool if not c.holds_key(key)]
            # Prefer a socket to an endpoint that is not in cooldown; a socket to
            # a suspect endpoint is still usable, just not first choice.
            now = time.monotonic()
            healthy = [c for c in free
                       if self._dead_until.get((node_id, c.endpoint),
                                               0.0) <= now]
            idle = [c for c in (healthy or free) if c.in_flight == 0]
            if idle:
                return idle[0]
            if len(pool) >= self._pool_size:
                if not free:
                    raise PoolExhausted(
                        node_id,
                        f"correlation key {key} in flight on every pooled "
                        f"connection ({self._pool_size})")
                # At the pool bound: multiplex onto the least-loaded socket.
                return min(healthy or free, key=lambda c: c.in_flight)
            # Otherwise grow the pool rather than queue behind a busy socket: a
            # worker answers one request at a time per connection, threading
            # only across connections (see ppu/expert_ppu.c, node.py).
        conn = self._connect(node_id, endpoints)
        with self._lock:
            if self._closed:
                conn.close()
                raise TransportClosed(node_id, "transport closed")
            self._pools.setdefault(node_id, []).append(conn)
            self.connects_opened[node_id] = (
                self.connects_opened.get(node_id, 0) + 1)
            slot = (node_id, conn.endpoint)
            self.connects_by_endpoint[slot] = (
                self.connects_by_endpoint.get(slot, 0) + 1)
        return conn

    def _connect(self, node_id: str,
                 endpoints: Sequence[Endpoint]) -> _Connection:
        """Open a socket to the first endpoint that accepts one."""
        failures: List[str] = []
        for endpoint in endpoints:
            try:
                conn = _Connection(node_id, endpoint, self._connect_timeout)
            except NodeConnectError as exc:
                self.mark_endpoint_dead(node_id, endpoint)
                failures.append(str(exc))
                continue
            self.mark_endpoint_live(node_id, endpoint)
            return conn
        raise NodeConnectError(node_id, "; ".join(failures)
                               or "no endpoint configured")

    def connection_count(self, node_id: Optional[str] = None) -> int:
        """Live connections, for one node or the whole transport."""
        with self._lock:
            if node_id is not None:
                return len([c for c in self._pools.get(node_id, [])
                            if not c.closed])
            return sum(len([c for c in pool if not c.closed])
                       for pool in self._pools.values())

    # -- request path -----------------------------------------------------
    def submit_raw(self, node_id: str, key: CorrelationKey, frame: bytes,
                   count: bool = True) -> "PendingFrame":
        """Write an arbitrary P3XC frame and return a handle for its reply.

        The connection machinery is frame-agnostic: correlation only needs the
        peer to echo ``(layer, expert, token_id)``, which the subcluster frames
        in ``batch.py`` do as well. This is the hook the hierarchical transport
        uses so it does not have to duplicate the pool or the reader threads.

        ``count=False`` keeps the frame out of ``requests_sent``, which counts
        work sent to a node rather than heartbeats.
        """
        conn = self._acquire(node_id, key)
        pending = conn.send(key, frame)
        if count:  # heartbeats are not work; they must not skew the counter
            with self._lock:
                self.requests_sent[node_id] = (
                    self.requests_sent.get(node_id, 0) + 1)
        return PendingFrame(conn, pending, node_id, self._timeout)

    def submit(self, node_id: str, layer: int, expert: int, token_id: int,
               x: np.ndarray) -> "PendingRequest":
        """Write a REQ frame and return a handle to await the response."""
        key: CorrelationKey = (layer, expert, token_id)
        frame = encode(MSG_REQ, layer, expert, token_id, x)
        return PendingRequest(self.submit_raw(node_id, key, frame))

    def dispatch(self, node_id: str, layer: int, expert: int, token_id: int,
                 x: np.ndarray) -> np.ndarray:
        """Synchronous request/response, API-compatible with ``Transport``."""
        return self.submit(node_id, layer, expert, token_id, x).result()

    def ping(self, node_id: str, timeout: Optional[float] = None) -> float:
        """PING/PONG liveness probe. Returns the round-trip time in seconds.

        Raises :class:`NodeTimeout` / :class:`NodeConnectError` /
        :class:`NodeDisconnected` naming the node, so callers can attribute a
        failure to a console rather than to "the cluster" (the coarse-grained
        remote error notification DaCS provides for Cell accelerators).
        """
        deadline = self._timeout if timeout is None else timeout
        # PING carries no expert identity; the worker echoes its own
        # (layer, expert) in the PONG, so correlate on a private token id.
        token_id = (next(self._ping_ids) & 0x7FFFFFFF) | 0x80000000
        key: CorrelationKey = (0xFFFF, 0xFFFF, token_id)
        frame = encode(MSG_PING, 0xFFFF, 0xFFFF, token_id,
                       np.zeros(1, np.float32))
        started = time.monotonic()
        handle = self.submit_raw(node_id, key, frame, count=False)
        try:
            msg = handle.message(deadline)
        except TransportError:
            # A peer that accepts connections but does not answer is worse than
            # one that refuses: steer new requests at the next endpoint.
            self.mark_endpoint_dead(node_id, handle.endpoint)
            raise
        if msg["msg_type"] != MSG_PONG:
            raise NodeError(node_id, msg["layer"], msg["expert"],
                            msg["token_id"])
        self.mark_endpoint_live(node_id, handle.endpoint)
        return time.monotonic() - started

    def alive(self, node_id: str, timeout: Optional[float] = None) -> bool:
        """``ping`` reduced to a bool, for liveness sweeps."""
        try:
            self.ping(node_id, timeout)
            return True
        except TransportError:
            return False

    # -- teardown ---------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._closed = True
            pools, self._pools = self._pools, {}
        for pool in pools.values():
            for conn in pool:
                conn.close()

    def __enter__(self) -> "PersistentSocketTransport":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PendingFrame:
    """Handle for one in-flight frame on a persistent connection."""

    __slots__ = ("_conn", "_pending", "node_id", "_timeout", "_message")

    def __init__(self, conn: _Connection, pending: _Pending, node_id: str,
                 timeout: float):
        self._conn = conn
        self._pending = pending
        self.node_id = node_id
        self._timeout = timeout
        self._message: Optional[dict] = None

    @property
    def key(self) -> CorrelationKey:
        return self._pending.key

    @property
    def endpoint(self) -> Endpoint:
        """Which endpoint of the peer this frame went to."""
        return self._conn.endpoint

    def done(self) -> bool:
        return self._pending.event.is_set()

    def message(self, timeout: Optional[float] = None) -> dict:
        """Block for the decoded reply frame, raising a node-specific error."""
        if self._message is None:
            self._message = self._conn.wait(
                self._pending,
                self._timeout if timeout is None else timeout)
        return self._message

    def cancel(self) -> None:
        """Abandon the request; a late response is dropped by the reader."""
        self._conn.cancel(self._pending)


class PendingRequest:
    """Handle for one in-flight expert call on a persistent connection."""

    __slots__ = ("_frame", "_result")

    def __init__(self, frame: PendingFrame):
        self._frame = frame
        self._result: Optional[np.ndarray] = None

    @property
    def node_id(self) -> str:
        return self._frame.node_id

    @property
    def key(self) -> CorrelationKey:
        return self._frame.key

    def done(self) -> bool:
        return self._frame.done()

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        """Block for the response array, raising a node-specific error."""
        if self._result is not None:
            return self._result
        msg = self._frame.message(timeout)
        if msg["msg_type"] == MSG_ERR:
            raise NodeError(self.node_id, msg["layer"], msg["expert"],
                            msg["token_id"])
        if msg["msg_type"] != MSG_RSP:
            raise NodeDisconnected(
                self.node_id, f"unexpected msg_type {msg['msg_type']}")
        self._result = msg["array"]
        return self._result

    def cancel(self) -> None:
        """Abandon the request; a late response is dropped by the reader."""
        self._frame.cancel()


#: ``PooledSocketTransport`` is the migration alias for code that wants the
#: pooled behaviour by name. ``dispatch.SocketTransport`` keeps its original
#: connect-per-dispatch semantics untouched for compatibility.
PooledSocketTransport = PersistentSocketTransport
