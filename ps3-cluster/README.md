# ps3-cluster — Kimi K3 on a PlayStation 3 farm

A port of [AirLLM PR #316](https://github.com/lyogavin/airllm/pull/316)
(per-expert streaming for Kimi K3, 2.8T MXFP4 MoE) to a cluster of PlayStation 3
consoles under OtherOS, using a **1 expert × 1 layer / node** placement: every
console permanently holds one MoE expert in its 256 MB RAM, and top-k routing
means only a handful of consoles work per token.

Full design writeup: [`../docs/PS3_CLUSTER_PORT.md`](../docs/PS3_CLUSTER_PORT.md).

## Quick start

```bash
make host                                   # build the portable host-sim worker
python3 -m unittest discover -s tests -v    # run all tests
python3 tools/plan_k3.py                     # size the cluster for Kimi K3
```

## What's here

- **Python coordinator side** (`ps3_cluster/`): the #316 port. `dispatch.py`
  turns the router's top-k decision into activations sent to the owning consoles
  instead of local disk loads, fanning the top-k calls out concurrently;
  `transport.py` holds persistent pooled P3XC connections with several requests
  in flight per node; `subcluster.py` groups nodes into Condor-style
  subclusters of 22 for hierarchical fan-out; `coordinator.py` runs a real head
  server per subcluster and `hierarchy.py` is the layer-side half that sends one
  batched request per subcluster (`batch.py` is that wire format,
  `deployment.py` the JSON membership config); `errors.py` is the
  node-attributed failure hierarchy; `topology.py` plans the 1-expert/node
  layout; `protocol.py` is the big-endian wire codec; `node.py` is a reference
  (numpy) expert worker.
- **Cell kernels** (`common/`, `spu/`, `ppu/`): the MVP compute path. `mxfp4.h`
  dequantises microscaling-FP4; `expert_spu.c` is the SPE GEMV kernel with
  DMA-streamed weight tiles; `expert_ppu.c` is the PPE driver that keeps one
  expert resident, fans each matmul across the SPEs via libspe2, and serves the
  P3XC protocol over TCP.
- **RSX GPU backend** (`rsx/`): an alternative to the SPE path for the
  GameOS-exploit boot. `expert_rsx.cg` is a Cg fragment shader doing the MXFP4
  GEMV on the RSX; `expert_rsx.c` is the PSGL/Cg host driver; `rsx_gemv_emu.h`
  is a CPU model of the shader used to validate it on x86. Select with `USE_RSX`
  (real) or `GEMV_RSX_EMU` (emulated); see `../docs/PS3_CLUSTER_PORT.md`.
- **`cell-compat.h`**: the PS3 analogue of `power8-compat.h` — big-endian ppc64,
  classic AltiVec only (no VSX/MMA), SPE local-store budget.

## Asynchronous coordination

A token at K3 scale touches ~16 consoles per layer across 92 layers. Two things
make that affordable, both taken from how PS3 clusters and IBM's Cell
middleware actually worked:

**Persistent connections.** `SocketTransport` opens a TCP connection per
dispatch; `PersistentSocketTransport` (alias `PooledSocketTransport`) keeps a
bounded pool of long-lived connections per node, serialises writes, allows
several requests in flight, and correlates responses by the frame's
`(layer, expert, token_id)` identity so they may come back out of order. This
is ALF's persistent per-accelerator work queue and DaCS's standing host↔
accelerator channel, at cluster scale.

**Concurrent top-k.** `DistributedExpertDispatcher.run_expert_stage` submits all
selected experts together and reduces the results as they arrive; the summation
itself happens in top-k order, so the output is bit-identical to the old serial
reduction. `submit_expert_stage` returns a `DispatchStage` for callers that want
several stages in flight.

```python
from ps3_cluster import (DistributedExpertDispatcher, ExpertPlacement,
                         PersistentSocketTransport, RetryPolicy,
                         SubclusterPlan)

transport = PersistentSocketTransport(endpoints, timeout=30.0,
                                      max_connections_per_node=4)
dispatcher = DistributedExpertDispatcher(
    placement, transport,
    retry_policy=RetryPolicy(attempts=2),                # safe retries only
    subclusters=SubclusterPlan(placement.node_ids(), 22))

y = dispatcher.run_expert_stage(layer, x, expert_ids, gate_weights, token_id)
alive = dispatcher.check_liveness(dispatcher.active_nodes_for(layer, expert_ids))
dispatcher.close(); transport.close()
```

## Deployed hierarchy (subcluster coordinators)

Condor, the 1,716-console AFRL cluster, was wired as **subclusters of 22 PS3s
behind a coordinating server**; the heads absorbed the fan-out and aggregated
upward. That is a process topology, not just a grouping, and `coordinator.py`
implements it:

```
layer coordinator ── BREQ (activation once + [(expert, gate)…]) ──▶ head server
                 ◀────────────── BRSP (one partial sum) ──────────┤  (per 22)
                                                                  ├─▶ 22 consoles
                                        P3XC REQ/RSP, pooled, concurrent
```

Bring a farm up from one shared config file. Each console runs a worker —
`build/expert_node_host expert.exp <port>` on real hardware, or its numpy
counterpart `tools/run_expert.py` when there is no Cell toolchain — and each
subcluster runs one head server:

```bash
python3 tools/gen_cluster_config.py --layer 3 --experts 44 \
    --expert-host 10.0.0.10 --head-host 10.0.1.1 -o cluster.json   # 2 × 22

# on each console (port from cluster.json; --identity skips the .exp file)
python3 tools/pack_expert.py L003-E0000.exp --layer 3 --expert 0
python3 tools/run_expert.py L003-E0000.exp --port 9000

# on each head server; --host/--port override the config, --timeout sets the
# downstream budget in seconds and --attempts the retry count per expert
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000
python3 tools/run_subcluster.py --config cluster.json --list
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000 \
    --check-members                      # PING every console behind this head
```

The layer coordinator reads the same file and talks only to the heads:

```python
from ps3_cluster import (ClusterConfig, HierarchicalExpertDispatcher,
                         SubclusterTransport)

config = ClusterConfig.load("cluster.json")
transport = SubclusterTransport(config.group_endpoints(), timeout=30.0)
dispatcher = HierarchicalExpertDispatcher(config.placement(), config.plan(),
                                          transport)
y = dispatcher.run_expert_stage(layer, x, expert_ids, gate_weights, token_id)
alive = dispatcher.check_liveness()
dispatcher.close(); transport.close()
```

Properties, all covered by `tests/test_hierarchy.py` over real sockets:

- **One activation per subcluster**, not per expert: a `BREQ` carries the
  activation once plus 8 bytes per selected expert, so a top-4-in-one-subcluster
  pick costs ~28 KB instead of ~112 KB at K3 width (fixed big-endian, versioned,
  every count bounded — see `ps3_cluster/batch.py`).
- **Concurrent inside and across subclusters**: the head server fans its
  experts out through the same pooled transport as the flat path, and the layer
  sends the involved heads' batches together.
- **Multiple batches per upstream connection**, correlated by
  `(layer, 0xFFFF, token_id)`, so replies may come back out of order.
- **All-or-error partials**: a subcluster returns a sum covering *every*
  requested expert (after any configured replica failover) or a `BERR` naming
  the failed experts and their consoles, raised as `SubclusterError` with
  `safe_to_retry` set only when no expert can have run. A short sum is never
  reduced. Passing `replicas=[...]` pins entries to standby consoles when the
  layer already knows a primary is down.
- **Numerics**: contributions are summed in ascending expert order inside a
  subcluster and partials in ascending group order — bit-identical to the
  in-process `hierarchical_reduce`, and bit-identical to the flat sum when one
  subcluster covers the stage. Grouping otherwise re-associates float32
  additions, so a multi-subcluster result can differ from the flat sum in the
  last bits.
- The flat `DistributedExpertDispatcher` and the canonical
  one-expert-per-layer-per-node placement are unchanged; a head server is an
  overlay, not a placement authority.

**Heartbeats.** `transport.ping(node)` / `.alive(node)` use the protocol's
PING/PONG frames on the same connection as expert traffic; both the Python and C
workers answer them. Failures raise node-attributed exceptions
(`NodeConnectError`, `NodeError`, `NodeTimeout`, `NodeDisconnected`) so a log
names the console, not "the cluster".

**Failover.** `ExpertPlacement.assign_replica` registers optional standby
consoles without changing canonical placement. Retries are opt-in per failure
class: unreachable-node failures are retried (at-most-once, the request never
arrived), `ERR` frames only with `retry_on_node_error=True`, and timeouts only
with `retry_on_timeout=True` — which is **at-least-once execution**, though a
stale response is dropped rather than summed, so a contribution is never
double-counted. Full semantics: [`../docs/PS3_CLUSTER_PORT.md`](../docs/PS3_CLUSTER_PORT.md).

## Wiring into a real model (coordinator shim)

`DistributedExpertDispatcher` is framework-agnostic. To drive a live HF model,
mirror #316's hook points: where AirLLM's `_expert_pre_hook` calls
`load_layer_subset(...)`, instead call `dispatcher.run_expert_stage(layer, x,
expert_ids, gate_weights)` with the router's decision for that layer, and skip
the local weight load entirely (the experts stay on `meta` forever on the
coordinator). Everything else — `generate()`, KV cache, sampling — is unchanged.

## Building for a real PS3

Install the [ps3dev toolchain](https://github.com/ps3dev/ps3toolchain), then:

```bash
make ps3        # ppu-gcc + spu-gcc, SPU kernel embedded, libspe2 fan-out
./build/expert_node_ps3 expert_L07_E0042.exp 3830
```
