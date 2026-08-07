"""PS3-cluster port of AirLLM #316 per-expert streaming for RAM Coffers.

Runs a sparse-MoE model (e.g. Kimi K3, 2.8T) across a farm of PlayStation 3
consoles under a "1 expert x 1 layer / node" placement: every console holds one
MoE expert resident in its 256 MB RAM, and top-k routing means only a handful of
nodes work per token.
"""

from .topology import (ModelProfile, ClusterPlan, KIMI_K3, plan_cluster,
                       placement_table, PS3_TOTAL_RAM_MB, PS3_USABLE_RAM_MB)
from .dispatch import (ExpertPlacement, Transport, LoopbackTransport,
                       SocketTransport, DistributedExpertDispatcher)
from .node import ExpertNode, ExpertServer, serve

__all__ = [
    "ModelProfile", "ClusterPlan", "KIMI_K3", "plan_cluster",
    "placement_table", "PS3_TOTAL_RAM_MB", "PS3_USABLE_RAM_MB",
    "ExpertPlacement", "Transport", "LoopbackTransport", "SocketTransport",
    "DistributedExpertDispatcher", "ExpertNode", "ExpertServer", "serve",
]
