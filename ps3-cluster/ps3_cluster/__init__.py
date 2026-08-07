"""PS3-cluster port of AirLLM #316 per-expert streaming for RAM Coffers.

Runs a sparse-MoE model (e.g. Kimi K3, 2.8T) across a farm of PlayStation 3
consoles under a "1 expert x 1 layer / node" placement: every console holds one
MoE expert resident in its 256 MB RAM, and top-k routing means only a handful of
nodes work per token.

Expert calls are coordinated asynchronously: ``transport.py`` keeps persistent
pooled P3XC connections with several requests in flight per node, and
``dispatch.py`` fans a layer's top-k calls out together and reduces them in a
fixed order. ``subcluster.py`` groups nodes into Condor-style subclusters of 22
for hierarchical fan-out, ``coordinator.py`` runs a real head-server process per
subcluster, and ``hierarchy.py`` is the layer-side half that sends one batched
request per subcluster.
"""

from .topology import (ModelProfile, ClusterPlan, KIMI_K3, plan_cluster,
                       placement_table, PS3_TOTAL_RAM_MB, PS3_USABLE_RAM_MB)
from .errors import (TransportError, NodeConnectError, NodeError, NodeTimeout,
                     NodeDisconnected, PoolExhausted, TransportClosed,
                     SubclusterError)
from .dispatch import (ExpertPlacement, Transport, LoopbackTransport,
                       SocketTransport, DistributedExpertDispatcher,
                       DispatchStage, RetryPolicy, SAFE_RETRY)
from .subcluster import (SubclusterPlan, DEFAULT_SUBCLUSTER_SIZE,
                         hierarchical_reduce, partial_reduce)
from .transport import (PersistentSocketTransport, PooledSocketTransport,
                        PendingRequest, PendingFrame)
from .batch import (BatchEntry, BatchFailure, MAX_BATCH_ENTRIES,
                    encode_batch_request, decode_batch_request,
                    encode_batch_response, decode_batch_response,
                    encode_batch_error, decode_batch_error, decode_batch)
from .coordinator import (SubclusterCoordinator, SubclusterServer,
                          SubclusterService, serve_subcluster)
from .hierarchy import (SubclusterTransport, HierarchicalExpertDispatcher,
                        HierarchicalStage, PendingBatch)
from .deployment import (ClusterConfig, SubclusterSpec, MemberSpec,
                         ReplicaSpec)
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
    "PoolExhausted", "TransportClosed", "SubclusterError",
    "PendingFrame", "BatchEntry", "BatchFailure", "MAX_BATCH_ENTRIES",
    "encode_batch_request", "decode_batch_request", "encode_batch_response",
    "decode_batch_response", "encode_batch_error", "decode_batch_error",
    "decode_batch", "SubclusterCoordinator", "SubclusterServer",
    "SubclusterService", "serve_subcluster", "SubclusterTransport",
    "HierarchicalExpertDispatcher", "HierarchicalStage", "PendingBatch",
    "ClusterConfig", "SubclusterSpec", "MemberSpec", "ReplicaSpec",
    "ExpertNode", "ExpertServer", "serve",
]
