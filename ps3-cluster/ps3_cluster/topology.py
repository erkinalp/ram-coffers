"""1-expert x 1-layer per node topology planner for a PS3 cluster.

RAM Coffers (upstream) partitions a model across NUMA banks *inside one POWER8
box*. This module takes the same "put each piece of knowledge in a known bank"
idea and stretches it across a farm of PlayStation 3 consoles: a coffer becomes
a whole console, and the unit of placement is a single MoE expert of a single
decoder layer.

That granularity is what makes an otherwise impossible model tractable on 256 MB
nodes: a sparse-MoE expert is tiny (Kimi K3: ~17-19 MB packed MXFP4), and top-k
routing means only a handful of the ~82k experts run per token.

The planner is pure arithmetic -- no hardware needed -- so it can size a cluster
and emit a placement table on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

# PS3 (fat, OtherOS) usable RAM. 256 MB XDR total; the hypervisor + kernel eat
# a slice, so budget conservatively for what an expert process can actually use.
PS3_TOTAL_RAM_MB = 256
PS3_USABLE_RAM_MB = 200  # after hypervisor/kernel/runtime overhead

# RSX GDDR3: 256 MB total, ~240 MB usable after the framebuffer. Two very
# different regimes:
#   - OtherOS (hypervisor): mappable but READ bandwidth is ~16 MB/s, so it is
#     unusable for weights read every token. Treat as cold capacity only ->
#     do NOT add it to the hot per-node budget by default.
#   - GameOS exploit (AsbestOS): full ~22.4 GB/s access + programmable shaders,
#     so it is a genuine second tier that can hold hot experts.
PS3_RSX_RAM_MB = 256
PS3_RSX_USABLE_MB = 240


@dataclass
class ModelProfile:
    """Everything the planner needs about an MoE checkpoint."""
    name: str
    n_layers: int
    experts_per_layer: int
    top_k: int                 # experts routed per token per layer
    hidden_size: int
    expert_bytes: int          # packed size of ONE expert (MXFP4 etc.)
    resident_bytes_per_layer: int  # attention+router+norms+shared expert
    embed_bytes: int
    lm_head_bytes: int
    dtype: str = "mxfp4"

    @property
    def total_experts(self) -> int:
        return self.n_layers * self.experts_per_layer


# Kimi K3 as characterised by AirLLM PR #316: 2.8T MXFP4 multimodal MoE,
# 82,432 experts across 92 layers, top-16 routing, ~17 GB packed per layer.
KIMI_K3 = ModelProfile(
    name="kimi-k3",
    n_layers=92,
    experts_per_layer=896,           # 82,432 / 92
    top_k=16,                        # K3 routes a token to 16 experts/layer
    hidden_size=7168,
    expert_bytes=19 * 1024 * 1024,   # ~17 GB / 896 experts per layer
    resident_bytes_per_layer=48 * 1024 * 1024,
    embed_bytes=800 * 1024 * 1024,   # sharded across several embed nodes
    lm_head_bytes=800 * 1024 * 1024,
    dtype="mxfp4",
)


@dataclass
class NodeSpec:
    role: str                 # "expert" | "layer" | "embed" | "lm_head"
    layer: int
    expert: Optional[int]
    resident_mb: float
    fits: bool


@dataclass
class ClusterPlan:
    model: str
    usable_ram_mb: int
    expert_nodes: int
    layer_nodes: int
    io_nodes: int
    total_nodes: int
    experts_split_across: int      # >1 when a single expert exceeds a node
    active_nodes_per_token: int    # how many consoles light up per token
    idle_fraction: float
    per_expert_mb: float
    node_capacity_mb: float = 0.0            # hot RAM per node (XDR [+ RSX])
    experts_per_node: int = 1                # placement choice (design = 1)
    capacity_experts_per_node: int = 1       # how many could fit at capacity
    rsx: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _split_factor(item_bytes: int, usable_bytes: int) -> int:
    """How many nodes one item must be sharded across to fit."""
    if item_bytes <= usable_bytes:
        return 1
    return (item_bytes + usable_bytes - 1) // usable_bytes


def plan_cluster(model: ModelProfile,
                 usable_ram_mb: int = PS3_USABLE_RAM_MB,
                 rsx: bool = False,
                 rsx_usable_mb: int = PS3_RSX_USABLE_MB,
                 experts_per_node: int = 1) -> ClusterPlan:
    """Compute a placement plan for ``model``.

    Defaults to the canonical **1 expert x 1 layer / node** design: one expert
    resident in a node's XDR RAM, everything else headroom (activations, KV,
    runtime). ``rsx=True`` models a GameOS-exploit boot where the RSX's ~240 MB
    GDDR3 is a usable hot tier (full ~22.4 GB/s), widening per-node capacity;
    ``experts_per_node`` opts into packing multiple experts on one console to
    trade node count for per-node load (only valid up to the capacity the plan
    reports as ``capacity_experts_per_node``).
    """
    usable_bytes = usable_ram_mb * 1024 * 1024
    warnings: List[str] = []

    # Hot capacity per node: XDR only under OtherOS; XDR + RSX under GameOS.
    node_capacity_bytes = usable_bytes
    if rsx:
        node_capacity_bytes += rsx_usable_mb * 1024 * 1024
        warnings.append(
            "rsx=True assumes a GameOS exploit (AsbestOS) for full-speed RSX "
            "access; under OtherOS the RSX reads at ~16 MB/s and cannot hold hot "
            "experts (cold storage only)")

    expert_split = _split_factor(model.expert_bytes, node_capacity_bytes)
    if expert_split > 1:
        warnings.append(
            f"one expert ({model.expert_bytes/2**20:.1f} MB) exceeds a node's "
            f"{node_capacity_bytes/2**20:.0f} MB; splitting each expert across "
            f"{expert_split} nodes")

    # How many whole experts a node *could* hold at this capacity (informational).
    capacity_experts_per_node = max(1, node_capacity_bytes // model.expert_bytes)
    if experts_per_node < 1:
        experts_per_node = 1
    if experts_per_node > capacity_experts_per_node:
        warnings.append(
            f"experts_per_node={experts_per_node} exceeds capacity "
            f"({capacity_experts_per_node} fit in {node_capacity_bytes/2**20:.0f} MB); "
            f"clamping")
        experts_per_node = capacity_experts_per_node

    if expert_split > 1:
        expert_nodes = model.total_experts * expert_split
    else:
        expert_nodes = (model.total_experts + experts_per_node - 1) // experts_per_node
        if experts_per_node > 1:
            warnings.append(
                f"packing {experts_per_node} experts/node -> "
                f"{expert_nodes:,} expert nodes (vs {model.total_experts:,} at 1/node)")
    layer_nodes = model.n_layers  # one coordinator per layer holds resident modules

    if model.resident_bytes_per_layer > usable_bytes:
        warnings.append(
            f"per-layer resident modules ({model.resident_bytes_per_layer/2**20:.1f} MB) "
            f"exceed a node; layer coordinators must shard too")

    embed_nodes = _split_factor(model.embed_bytes, usable_bytes)
    lm_head_nodes = _split_factor(model.lm_head_bytes, usable_bytes)
    io_nodes = embed_nodes + lm_head_nodes

    total = expert_nodes + layer_nodes + io_nodes

    # Per token: top_k experts per layer light up, plus each layer coordinator,
    # plus the embed + lm_head shards. With experts packed together, the number
    # of distinct nodes touched can be lower (co-resident experts), but bound it
    # simply by the activations dispatched.
    active = (model.top_k * expert_split * model.n_layers
              + layer_nodes + io_nodes)
    idle_fraction = 1.0 - (active / total) if total else 0.0

    return ClusterPlan(
        model=model.name,
        usable_ram_mb=usable_ram_mb,
        expert_nodes=expert_nodes,
        layer_nodes=layer_nodes,
        io_nodes=io_nodes,
        total_nodes=total,
        experts_split_across=expert_split,
        active_nodes_per_token=active,
        idle_fraction=idle_fraction,
        per_expert_mb=model.expert_bytes / 2**20,
        node_capacity_mb=node_capacity_bytes / 2**20,
        experts_per_node=experts_per_node,
        capacity_experts_per_node=capacity_experts_per_node,
        rsx=rsx,
        warnings=warnings,
    )


def placement_table(model: ModelProfile,
                     usable_ram_mb: int = PS3_USABLE_RAM_MB) -> List[NodeSpec]:
    """Enumerate node assignments. Large for K3 (~82k rows) -- use for small
    profiles / slices; ``plan_cluster`` gives the summary for the full farm."""
    usable_mb = usable_ram_mb
    nodes: List[NodeSpec] = []
    for layer in range(model.n_layers):
        nodes.append(NodeSpec("layer", layer, None,
                              model.resident_bytes_per_layer / 2**20,
                              model.resident_bytes_per_layer / 2**20 <= usable_mb))
        for e in range(model.experts_per_layer):
            mb = model.expert_bytes / 2**20
            nodes.append(NodeSpec("expert", layer, e, mb, mb <= usable_mb))
    return nodes
