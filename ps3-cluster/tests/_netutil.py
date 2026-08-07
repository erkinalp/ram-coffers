"""Shared helpers for the network-level tests.

Everything here drives *real* sockets: the tests boot real expert workers on
loopback and talk P3XC to them. Synchronisation uses events and barriers rather
than sleeps so the assertions are not timing-sensitive.
"""

import os
import socket
import struct
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.node import ExpertNode, ExpertServer  # noqa: E402
from ps3_cluster.protocol import (MSG_REQ, MSG_RSP, decode,  # noqa: E402
                                  encode)


class CountingExpertServer(ExpertServer):
    """Expert server that counts accepted connections and served requests."""

    def __init__(self, host, port, expert_node):
        self.accepted = 0
        self._accept_lock = threading.Lock()
        super().__init__(host, port, expert_node)

    def get_request(self):
        request = super().get_request()
        with self._accept_lock:
            self.accepted += 1
        return request


class CountingExpert:
    """Expert callable that records how many requests it ran."""

    def __init__(self, fn):
        self._fn = fn
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, x):
        with self._lock:
            self.calls += 1
        return self._fn(x)


def linear_expert(seed, dim=4):
    """Deterministic linear expert ``y = W @ x`` plus its weight matrix."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dim, dim)).astype(np.float32)

    def fn(x):
        return W @ x.astype(np.float32)
    return fn, W


class RunningNode:
    """A live expert worker on loopback, with its server thread."""

    def __init__(self, layer, expert, fn, host="127.0.0.1"):
        self.expert_fn = CountingExpert(fn)
        self.server = CountingExpertServer(
            host, 0, ExpertNode(layer=layer, expert=expert,
                                expert_fn=self.expert_fn))
        self.host, self.port = self.server.server_address
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return (self.host, self.port)

    @property
    def accepted(self):
        return self.server.accepted

    @property
    def calls(self):
        return self.expert_fn.calls

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ReorderingServer:
    """Raw P3XC server that answers a batch of requests in reverse order.

    Used to prove the coordinator correlates responses by token/request identity
    rather than by arrival order: the reply for the *last* request is written
    first. The response payload encodes the request's ``token_id`` so a
    mis-correlation is visible in the data, not just in the header.
    """

    def __init__(self, layer, expert, batch=2, host="127.0.0.1"):
        self.layer = layer
        self.expert = expert
        self.batch = batch
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(4)
        self.host, self.port = self._sock.getsockname()
        self.accepted = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return (self.host, self.port)

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn):
        pending = []
        try:
            while not self._stop.is_set():
                msg = self._read(conn)
                if msg is None:
                    return
                if msg["msg_type"] != MSG_REQ:
                    continue
                pending.append(msg)
                if len(pending) < self.batch:
                    continue
                for msg in reversed(pending):
                    payload = (msg["array"].astype(np.float32)
                               + float(msg["token_id"]))
                    conn.sendall(encode(MSG_RSP, self.layer, self.expert,
                                        msg["token_id"], payload))
                pending = []
        except OSError:
            return
        finally:
            conn.close()

    @staticmethod
    def _read(conn):
        head = _recv_exact(conn, 4)
        if head is None:
            return None
        (length,) = struct.unpack("!I", head)
        body = _recv_exact(conn, length)
        if body is None:
            return None
        return decode(body)

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


class HangUpServer:
    """P3XC endpoint that accepts a connection and then drops it.

    Models a console that dies (or an OtherOS kernel that reboots) with the
    coordinator's connection established: the coordinator must notice the FIN
    and report the failure against that node id.
    """

    def __init__(self, host="127.0.0.1", read_first=True):
        self._read_first = read_first
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(4)
        self.host, self.port = self._sock.getsockname()
        self.accepted = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return (self.host, self.port)

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.accepted += 1
            if self._read_first:
                _recv_exact(conn, 4)
            conn.close()

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


class FrameTap:
    """Byte-counting P3XC proxy in front of a real service.

    Sits between a layer coordinator and a subcluster coordinator on loopback
    and records every frame body travelling upstream (client -> service) plus
    the byte totals each way, so a test can assert what actually went over the
    wire — e.g. that a subcluster request carries the activation once rather
    than once per expert.
    """

    def __init__(self, target, host="127.0.0.1"):
        self.target = target
        self.frames_up = []
        self.bytes_up = 0
        self.bytes_down = 0
        self.accepted = 0
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
            with self._lock:
                self.accepted += 1
            try:
                upstream = socket.create_connection(self.target, timeout=5)
            except OSError:
                client.close()
                continue
            threading.Thread(target=self._pump_up, args=(client, upstream),
                             daemon=True).start()
            threading.Thread(target=self._pump_down, args=(upstream, client),
                             daemon=True).start()

    def _pump_up(self, client, upstream):
        """Forward client -> service, parsing frames as they pass."""
        try:
            while True:
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!I", head)
                body = _recv_exact(client, length)
                if body is None:
                    return
                with self._lock:
                    self.frames_up.append(body)
                    self.bytes_up += 4 + len(body)
                upstream.sendall(head + body)
        except OSError:
            return
        finally:
            for sock in (client, upstream):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _pump_down(self, upstream, client):
        try:
            while True:
                chunk = upstream.recv(65536)
                if not chunk:
                    return
                with self._lock:
                    self.bytes_down += len(chunk)
                client.sendall(chunk)
        except OSError:
            return

    def frames_of_type(self, msg_type):
        with self._lock:
            return [body for body in self.frames_up if body[5] == msg_type]

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


def dead_endpoint():
    """An address with nothing listening (bound then closed)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    addr = s.getsockname()
    s.close()
    return addr


def reader_threads():
    """Names of live transport reader threads, for leak assertions."""
    return [t.name for t in threading.enumerate()
            if t.name.startswith("p3xc-reader")]
