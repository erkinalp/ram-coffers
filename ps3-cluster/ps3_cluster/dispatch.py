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

from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np

from .protocol import (encode, decode, MSG_REQ, MSG_RSP, MSG_ERR, read_frame)


class ExpertPlacement:
    """Maps (layer, expert) -> node id, and back. This *is* the routing table
    the coordinator uses instead of #316's on-disk shard offsets."""

    def __init__(self) -> None:
        self._to_node: Dict[Tuple[int, int], str] = {}
        self._to_expert: Dict[str, Tuple[int, int]] = {}

    def assign(self, layer: int, expert: int, node_id: str) -> None:
        self._to_node[(layer, expert)] = node_id
        self._to_expert[node_id] = (layer, expert)

    def node_for(self, layer: int, expert: int) -> str:
        return self._to_node[(layer, expert)]

    def expert_on(self, node_id: str) -> Tuple[int, int]:
        return self._to_expert[node_id]

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
    """Send an expert's input activation to its node and get the output back."""

    def dispatch(self, node_id: str, layer: int, expert: int,
                 token_id: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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
    """TCP transport to real expert nodes. One short-lived connection per
    dispatch keeps node bookkeeping trivial; a pooled variant is a drop-in."""

    def __init__(self, endpoints: Dict[str, Tuple[str, int]], timeout: float = 30.0):
        self._endpoints = endpoints
        self._timeout = timeout

    def dispatch(self, node_id, layer, expert, token_id, x):
        import socket
        host, port = self._endpoints[node_id]
        with socket.create_connection((host, port), timeout=self._timeout) as s:
            s.sendall(encode(MSG_REQ, layer, expert, token_id, x))
            msg = read_frame(s.recv)
        if msg["msg_type"] == MSG_ERR:
            raise RuntimeError(f"expert node {node_id} error")
        return msg["array"]


class DistributedExpertDispatcher:
    """Runs one MoE layer's expert stage across the cluster.

    Given the router's decision for a token (which experts, with what gate
    weights), dispatch to exactly those experts' nodes and combine the outputs.
    This is the distributed equivalent of the model invoking the selected expert
    submodules -- only the chosen nodes ever do work, which is the whole reason
    a ~82k-expert model is affordable."""

    def __init__(self, placement: ExpertPlacement, transport: Transport):
        self.placement = placement
        self.transport = transport

    def run_expert_stage(self, layer: int, x: np.ndarray,
                         expert_ids: Sequence[int],
                         gate_weights: Sequence[float],
                         token_id: int = 0) -> np.ndarray:
        """Combine top-k expert outputs: sum_j gate_j * expert_j(x)."""
        if len(expert_ids) != len(gate_weights):
            raise ValueError("expert_ids and gate_weights length mismatch")
        out = None
        for e, g in zip(expert_ids, gate_weights):
            node = self.placement.node_for(layer, e)
            y = self.transport.dispatch(node, layer, e, token_id, x)
            contrib = (g * y).astype(np.float32)
            out = contrib if out is None else out + contrib
        if out is None:
            return np.zeros_like(x, dtype=np.float32)
        return out

    def active_nodes_for(self, layer: int,
                         expert_ids: Sequence[int]) -> List[str]:
        return [self.placement.node_for(layer, e) for e in expert_ids]
