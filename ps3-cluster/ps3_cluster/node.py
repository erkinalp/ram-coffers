"""Expert worker that runs on a single PS3 (Cell/OtherOS).

Each console holds exactly one expert of one layer, resident in its 256 MB XDR
RAM for the lifetime of the process -- the opposite of AirLLM #316, which streams
experts off disk on demand. The worker listens on TCP, and for every request
runs its resident expert's forward pass and returns the output activation.

The heavy matmul is meant to be handed to the Cell SPEs via libspe2 (see
``cell-compat.h`` and ``docs/PS3_CLUSTER_PORT.md``); this reference worker uses
numpy so the protocol and lifecycle are testable off-hardware. The expert
callable is injected, so a real deployment plugs in the SPE-backed kernel while
tests plug in a numpy one.
"""

from __future__ import annotations

import socketserver
from typing import Callable, Optional
import numpy as np

from .protocol import (encode, read_frame, MSG_REQ, MSG_RSP, MSG_ERR,
                       MSG_PING, MSG_PONG, ProtocolError)

ExpertFn = Callable[[np.ndarray], np.ndarray]


class ExpertNode:
    """Holds one resident expert and applies it to inputs.

    ``load_mxfp4`` mirrors #316's decompress-on-use: the packed 4-bit payload is
    kept in RAM and expanded per call so the resident footprint stays ~4x
    smaller. Off-hardware we accept a plain float expert function instead."""

    def __init__(self, layer: int, expert: int, expert_fn: ExpertFn,
                 packed_bytes: Optional[bytes] = None):
        self.layer = layer
        self.expert = expert
        self._fn = expert_fn
        self._packed = packed_bytes  # resident MXFP4 payload, if any

    @property
    def resident_bytes(self) -> int:
        return len(self._packed) if self._packed is not None else 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self._fn(x)).astype(np.float32)


class _Handler(socketserver.BaseRequestHandler):
    """Serves a *persistent* connection: frames are read until the peer closes.

    The coordinator's pooled transport keeps one connection per node open for
    the life of the run and may have several requests in flight on it, so the
    worker must not treat one frame as one connection. Requests on a single
    connection are handled sequentially (a PS3's SPEs are already saturated by
    one expert GEMV), while the server itself is threaded, so a node with
    several pooled connections works on them in parallel.
    """

    def handle(self) -> None:
        node: ExpertNode = self.server.expert_node  # type: ignore[attr-defined]
        while True:
            try:
                msg = read_frame(self.request.recv)
            except (ProtocolError, OSError):
                return  # peer closed, or a malformed frame: drop the socket
            if not self._handle_one(node, msg):
                return

    def _handle_one(self, node: ExpertNode, msg: dict) -> bool:
        """Answer one frame. Returns False to close the connection."""
        try:
            if msg["msg_type"] == MSG_PING:
                # Heartbeat: answered on the same connection, interleaved with
                # expert work, so liveness does not need a second channel.
                self._send(encode(MSG_PONG, node.layer, node.expert,
                                  msg["token_id"], np.zeros(1, np.float32)))
                return True
            if msg["msg_type"] != MSG_REQ:
                return False
            if (msg["layer"], msg["expert"]) != (node.layer, node.expert):
                self._send(encode(MSG_ERR, node.layer, node.expert,
                                  msg["token_id"], np.zeros(1, np.float32)))
                return True
            try:
                y = node.forward(msg["array"])
            except Exception:
                self._send(encode(MSG_ERR, node.layer, node.expert,
                                  msg["token_id"], np.zeros(1, np.float32)))
                return True
            self._send(encode(MSG_RSP, node.layer, node.expert,
                              msg["token_id"], y))
            return True
        except OSError:
            return False

    def _send(self, frame: bytes) -> None:
        self.request.sendall(frame)


class ExpertServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, expert_node: ExpertNode):
        super().__init__((host, port), _Handler)
        self.expert_node = expert_node


def serve(host: str, port: int, expert_node: ExpertNode) -> ExpertServer:
    """Create (but do not block on) an expert server. Caller runs
    ``serve_forever`` in a thread or the main loop."""
    return ExpertServer(host, port, expert_node)
