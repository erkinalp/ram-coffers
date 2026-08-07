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
subcluster, ``regional.py`` runs an optional third tier of head-of-heads
processes, and ``hierarchy.py`` is the layer-side half that sends one batched
request per immediate downstream group. Hierarchical dispatch is bit-identical to
the flat path by default, at any depth; ``fast=True`` trades that for a smaller
upstream reply. ``dedup.py`` bounds the replay state a head keeps so a retried
batch is answered once rather than run twice.
"""

from .topology import (ModelProfile, ClusterPlan, KIMI_K3, plan_cluster,
                       placement_table, PS3_TOTAL_RAM_MB, PS3_USABLE_RAM_MB)
from .errors import (TransportError, NodeConnectError, NodeError, NodeTimeout,
                     NodeDisconnected, PoolExhausted, TransportClosed,
                     SubclusterError)
from .dispatch import (ExpertPlacement, Transport, LoopbackTransport,
                       SocketTransport, DistributedExpertDispatcher,
                       DispatchStage, RetryPolicy, SAFE_RETRY)
from .subcluster import (SubclusterPlan, TieredPlan, DEFAULT_SUBCLUSTER_SIZE,
                         hierarchical_reduce, partial_reduce)
from .transport import (PersistentSocketTransport, PooledSocketTransport,
                        PendingRequest, PendingFrame)
from .batch import (BatchEntry, BatchFailure, MAX_BATCH_ENTRIES,
                    MAX_REQUEST_ID, REQ_FLAG_FAST, REQ_FLAG_REQUEST_ID,
                    RSP_FLAG_PER_EXPERT, RSP_FLAG_REQUEST_ID,
                    encode_batch_request, decode_batch_request,
                    encode_batch_response, encode_batch_contributions,
                    decode_batch_response,
                    encode_batch_error, decode_batch_error, decode_batch)
from .coordinator import (BaseCoordinator, SubclusterCoordinator,
                          CoordinatorServer, CoordinatorService,
                          SubclusterServer, SubclusterService,
                          serve_subcluster)
from .regional import RegionalCoordinator, serve_region, region_addresses
from .dedup import (DedupCache, DEFAULT_DEDUP_BYTES, DEFAULT_DEDUP_ENTRIES,
                    DEFAULT_DEDUP_TTL, DedupCapacityError,
                    MismatchedRequestError)
from .hierarchy import (SubclusterTransport, HierarchicalExpertDispatcher,
                        HierarchicalStage, PendingBatch, LinkRetryPolicy,
                        next_request_id)
from .deployment import (ClusterConfig, SubclusterSpec, RegionSpec, MemberSpec,
                         ReplicaSpec)
from .node import ExpertNode, ExpertServer, serve

__all__ = [
    "ModelProfile", "ClusterPlan", "KIMI_K3", "plan_cluster",
    "placement_table", "PS3_TOTAL_RAM_MB", "PS3_USABLE_RAM_MB",
    "ExpertPlacement", "Transport", "LoopbackTransport", "SocketTransport",
    "DistributedExpertDispatcher", "DispatchStage", "RetryPolicy",
    "SAFE_RETRY", "PersistentSocketTransport", "PooledSocketTransport",
    "PendingRequest", "SubclusterPlan", "TieredPlan",
    "DEFAULT_SUBCLUSTER_SIZE",
    "hierarchical_reduce", "partial_reduce", "TransportError",
    "NodeConnectError", "NodeError", "NodeTimeout", "NodeDisconnected",
    "PoolExhausted", "TransportClosed", "SubclusterError",
    "PendingFrame", "BatchEntry", "BatchFailure", "MAX_BATCH_ENTRIES",
    "MAX_REQUEST_ID", "REQ_FLAG_FAST", "REQ_FLAG_REQUEST_ID",
    "RSP_FLAG_PER_EXPERT", "RSP_FLAG_REQUEST_ID",
    "encode_batch_request", "decode_batch_request", "encode_batch_response",
    "encode_batch_contributions", "decode_batch_response",
    "encode_batch_error", "decode_batch_error",
    "decode_batch", "BaseCoordinator", "SubclusterCoordinator",
    "CoordinatorServer", "CoordinatorService", "SubclusterServer",
    "SubclusterService", "serve_subcluster", "RegionalCoordinator",
    "serve_region", "region_addresses", "DedupCache",
    "DEFAULT_DEDUP_ENTRIES", "DEFAULT_DEDUP_TTL", "DEFAULT_DEDUP_BYTES",
    "DedupCapacityError", "MismatchedRequestError", "SubclusterTransport",
    "HierarchicalExpertDispatcher", "HierarchicalStage", "PendingBatch",
    "LinkRetryPolicy", "next_request_id",
    "ClusterConfig", "SubclusterSpec", "RegionSpec", "MemberSpec",
    "ReplicaSpec",
    "ExpertNode", "ExpertServer", "serve",
]
