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
    dense_layer_idxs: Optional[List[int]] = None
    vision_bytes: int = 0      # packed vision tower (added to embed nodes)
    projector_bytes: int = 0   # packed multimodal projector (added to embed nodes)
    shared_experts_per_layer: int = 0  # always-active shared experts in each MoE layer

    @property
    def moe_layer_count(self) -> int:
        dense = set(self.dense_layer_idxs or [])
        return self.n_layers - len(dense.intersection(range(self.n_layers)))

    @property
    def total_experts(self) -> int:
        return self.moe_layer_count * self.experts_per_layer


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

# Kimi K3 0.40B: https://huggingface.co/inference-optimization/Kimi-K3-0.40B
# 395.6M params, 8 hidden layers (layer 0 is dense, layers 1-7 are MoE),
# 8 routed experts per MoE layer, top-2 routing, latent-space experts (512-dim).
# Byte counts are MXFP4 packed estimates from the safetensors checkpoint.
_K3_040B_EXPERT_PARAMS = 393_216
_K3_040B_BLOCK = 32
KIMI_K3_040B = ModelProfile(
    name="kimi-k3-0.40b",
    n_layers=8,
    experts_per_layer=8,
    top_k=2,
    hidden_size=1024,
    expert_bytes=(_K3_040B_EXPERT_PARAMS // 2)          # packed 4-bit weights
                 + (_K3_040B_EXPERT_PARAMS // _K3_040B_BLOCK),  # E8M0 block scales
    resident_bytes_per_layer=4_069_821,                  # max non-expert layer (dense layer 0)
    embed_bytes=89_128_960,                              # embed_tokens (packed)
    lm_head_bytes=89_128_960,                            # lm_head (packed)
    dtype="mxfp4",
    dense_layer_idxs=[0],
    vision_bytes=2_587_400,                              # vision_tower (packed)
    projector_bytes=1_114_656,                           # mm_projector (packed)
    shared_experts_per_layer=1,
)

# DeepSeek V4 Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
# 284B total / 13B activated, 43 decoder layers, 256 routed experts/layer,
# top-6 routing, FP4-packed routed experts (same MXFP4/E2M1 + E8M0 block scale
# layout as the PS3 kernel), one shared expert per layer.
# Byte counts are from the safetensors index (layer-2 is the largest non-expert
# layer; embed/lm-head are BF16 vocab matrices).
_V4_FLASH_EXPERT_BYTES = 13_369_344
DEEPSEEK_V4_FLASH = ModelProfile(
    name="deepseek-v4-flash",
    n_layers=43,
    experts_per_layer=256,
    top_k=6,
    hidden_size=4096,
    expert_bytes=_V4_FLASH_EXPERT_BYTES,
    resident_bytes_per_layer=173_503_576,
    embed_bytes=1_059_061_760,
    lm_head_bytes=1_059_061_760,
    dtype="fp4",
    shared_experts_per_layer=1,
)

# DeepSeek V4 Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
# 1.6T total / 49B activated, 61 decoder layers, 384 routed experts/layer,
# top-6 routing, FP4-packed routed experts.  Layer-2 is again the largest
# non-expert layer (~413 MiB), so Pro layer nodes need XDR+RSX or sharding.
_V4_PRO_EXPERT_BYTES = 35_094_528
DEEPSEEK_V4_PRO = ModelProfile(
    name="deepseek-v4-pro",
    n_layers=61,
    experts_per_layer=384,
    top_k=6,
    hidden_size=7168,
    expert_bytes=_V4_PRO_EXPERT_BYTES,
    resident_bytes_per_layer=433_447_448,
    embed_bytes=1_852_958_720,
    lm_head_bytes=1_852_958_720,
    dtype="fp4",
    shared_experts_per_layer=1,
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
    node_capacity_mb: float = 0.0            # combined hot memory per node
    xdr_capacity_mb: float = 0.0             # XDR hot pool
    rsx_capacity_mb: float = 0.0             # RSX hot pool (0 if not rsx)
    experts_per_node: int = 1                # placement choice (design = 1)
    capacity_experts_per_node: int = 1       # total that could fit (XDR+RSX)
    xdr_capacity_experts_per_node: int = 0
    rsx_capacity_experts_per_node: int = 0
    xdr_experts_per_node: int = 0
    rsx_experts_per_node: int = 0
    rsx: bool = False
    moe_layers: int = 0
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
    GDDR3 is a usable hot tier (full ~22.4 GB/s); in that regime the planner
    treats XDR and RSX as two distinct hot pools on the same console, so many
    experts can fit by filling XDR first and spilling the rest to RSX.
    ``experts_per_node`` opts into packing multiple experts on one console to
    trade node count for per-node load.
    """
    usable_bytes = usable_ram_mb * 1024 * 1024
    rsx_bytes = rsx_usable_mb * 1024 * 1024 if rsx else 0
    node_capacity_bytes = usable_bytes + rsx_bytes
    max_pool_bytes = max(usable_bytes, rsx_bytes)
    warnings: List[str] = []
    moe_count = model.moe_layer_count
    total_experts = moe_count * model.experts_per_layer

    if rsx:
        warnings.append(
            "rsx=True assumes a GameOS exploit (AsbestOS) for full-speed RSX "
            "access; under OtherOS the RSX reads at ~16 MB/s and cannot hold hot "
            "experts (cold storage only)")

    # A single expert must fit in *one* pool (XDR or RSX), not be split across
    # them.  Splitting only happens when an expert exceeds the larger pool.
    expert_split = _split_factor(model.expert_bytes, max_pool_bytes)
    if expert_split > 1:
        warnings.append(
            f"one expert ({model.expert_bytes/2**20:.1f} MB) exceeds the "
            f"{max_pool_bytes/2**20:.0f} MB largest pool; splitting each expert "
            f"across {expert_split} nodes")

    # Per-pool packing limits.
    xdr_fit = usable_bytes // model.expert_bytes
    rsx_fit = rsx_bytes // model.expert_bytes
    capacity_experts_per_node = max(1, xdr_fit + rsx_fit)

    if experts_per_node < 1:
        experts_per_node = 1

    if expert_split > 1:
        expert_nodes = total_experts * expert_split
        if experts_per_node > 1:
            warnings.append(
                f"experts_per_node={experts_per_node} ignored: each expert is "
                f"already split across {expert_split} nodes")
        experts_per_node = 1
        xdr_experts_per_node = 0
        rsx_experts_per_node = 0
    else:
        if experts_per_node > capacity_experts_per_node:
            warnings.append(
                f"experts_per_node={experts_per_node} exceeds capacity "
                f"({capacity_experts_per_node} fit across XDR+RSX); clamping")
            experts_per_node = capacity_experts_per_node

        xdr_experts_per_node = min(experts_per_node, xdr_fit)
        rsx_experts_per_node = max(0, experts_per_node - xdr_experts_per_node)
        if rsx_experts_per_node > rsx_fit:
            # Should not happen because of capacity clamp above, but keep safe.
            xdr_experts_per_node += rsx_experts_per_node - rsx_fit
            rsx_experts_per_node = rsx_fit

        expert_nodes = (total_experts + experts_per_node - 1) // experts_per_node
        if experts_per_node > 1:
            warnings.append(
                f"packing {experts_per_node} experts/node "
                f"({xdr_experts_per_node} XDR, {rsx_experts_per_node} RSX) -> "
                f"{expert_nodes:,} expert nodes (vs {total_experts:,} at 1/node)")
    layer_nodes = model.n_layers  # one coordinator per layer holds resident modules

    if model.resident_bytes_per_layer > node_capacity_bytes:
        warnings.append(
            f"per-layer resident modules ({model.resident_bytes_per_layer/2**20:.1f} MB) "
            f"exceed combined node memory ({node_capacity_bytes/2**20:.0f} MB); "
            f"layer coordinators must shard too")

    input_bytes = model.embed_bytes + model.vision_bytes + model.projector_bytes
    embed_nodes = _split_factor(input_bytes, node_capacity_bytes)
    lm_head_nodes = _split_factor(model.lm_head_bytes, node_capacity_bytes)
    io_nodes = embed_nodes + lm_head_nodes

    total = expert_nodes + layer_nodes + io_nodes

    # Per token: top_k routed experts per MoE layer light up, plus every layer
    # coordinator, plus the embed + lm_head shards. Dense layers add no routed
    # expert nodes. With experts packed together, the number of distinct nodes
    # touched can be lower (co-resident experts), but bound it simply by the
    # activations dispatched.
    active = (model.top_k * expert_split * moe_count
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
        xdr_capacity_mb=usable_ram_mb,
        rsx_capacity_mb=rsx_usable_mb if rsx else 0,
        experts_per_node=experts_per_node,
        capacity_experts_per_node=capacity_experts_per_node,
        xdr_capacity_experts_per_node=xdr_fit,
        rsx_capacity_experts_per_node=rsx_fit,
        xdr_experts_per_node=xdr_experts_per_node,
        rsx_experts_per_node=rsx_experts_per_node,
        rsx=rsx,
        moe_layers=moe_count,
        warnings=warnings,
    )


def placement_table(model: ModelProfile,
                     usable_ram_mb: int = PS3_USABLE_RAM_MB) -> List[NodeSpec]:
    """Enumerate node assignments. Large for K3 (~82k rows) -- use for small
    profiles / slices; ``plan_cluster`` gives the summary for the full farm."""
    usable_mb = usable_ram_mb
    dense = set(model.dense_layer_idxs or [])
    nodes: List[NodeSpec] = []
    for layer in range(model.n_layers):
        nodes.append(NodeSpec("layer", layer, None,
                              model.resident_bytes_per_layer / 2**20,
                              model.resident_bytes_per_layer / 2**20 <= usable_mb))
        if layer in dense:
            continue
        for e in range(model.experts_per_layer):
            mb = model.expert_bytes / 2**20
            nodes.append(NodeSpec("expert", layer, e, mb, mb <= usable_mb))
    return nodes
