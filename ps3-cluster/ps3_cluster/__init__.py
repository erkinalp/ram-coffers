"""PS3-cluster port of AirLLM #316 per-expert streaming for RAM Coffers.

Runs a sparse-MoE model (e.g. Kimi K3, 2.8T) across a farm of PlayStation 3
consoles under a "1 expert x 1 layer / node" placement: every console holds one
MoE expert resident in its 256 MB RAM, and top-k routing means only a handful of
nodes work per token.

Expert calls are coordinated asynchronously: ``transport.py`` keeps persistent
pooled P3XC connections with several requests in flight per node, and
``dispatch.py`` fans a layer's top-k calls out together and reduces them in a
fixed order. ``subcluster.py`` groups nodes into Condor-style subclusters of 22
for hierarchical fan-out.
"""

from .topology import (ModelProfile, ClusterPlan, KIMI_K3, plan_cluster,
                       placement_table, PS3_TOTAL_RAM_MB, PS3_USABLE_RAM_MB)
from .errors import (TransportError, NodeConnectError, NodeError, NodeTimeout,
                     NodeDisconnected, PoolExhausted, TransportClosed)
from .dispatch import (ExpertPlacement, Transport, LoopbackTransport,
                       SocketTransport, DistributedExpertDispatcher,
                       DispatchStage, RetryPolicy, SAFE_RETRY)
from .subcluster import (SubclusterPlan, DEFAULT_SUBCLUSTER_SIZE,
                         hierarchical_reduce, partial_reduce)
from .transport import (PersistentSocketTransport, PooledSocketTransport,
                        PendingRequest)
from .node import ExpertNode, ExpertServer, serve

__all__ = [
    "ModelProfile", "ClusterPlan", "KIMI_K3", "plan_cluster",
    "placement_table", "PS3_TOTAL_RAM_MB", "PS3_USABLE_RAM_MB",
    "ExpertPlacement", "Transport", "LoopbackTransport", "SocketTransport",
    "DistributedExpertDispatcher", "DispatchStage", "RetryPolicy",
    "SAFE_RETRY", "PersistentSocketTransport", "PooledSocketTransport",
    "PendingRequest", "SubclusterPlan", "DEFAULT_SUBCLUSTER_SIZE",
    "hierarchical_reduce", "partial_reduce", "TransportError",
    "NodeConnectError", "NodeError", "NodeTimeout", "NodeDisconnected",
    "PoolExhausted", "TransportClosed",
    "ExpertNode", "ExpertServer", "serve",
]
