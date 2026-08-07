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
  server per subcluster, `regional.py` runs the same coordinator one tier up
  over a set of head servers, and `hierarchy.py` is the client half that sends
  one batched request per immediate downstream group (`batch.py` is that wire
  format, `deployment.py` the JSON membership config, `dedup.py` the bounded
  retry-replay cache); `errors.py` is the
  node-attributed failure hierarchy; `topology.py` plans the 1-expert/node
  layout; `protocol.py` is the big-endian wire codec; `node.py` is a reference
  (numpy) expert worker.
- **Cell kernels** (`common/`, `spu/`, `ppu/`): the MVP compute path. `mxfp4.h`
  dequantises microscaling-FP4; `expert_spu.c` is the SPE GEMV kernel with
  DMA-streamed weight tiles; `expert_ppu.c` is the PPE driver that keeps one
  expert resident, selects the GEMV backend (scalar PPE, Cell SPEs via libspe2,
  or RSX fragment shader) at compile time, and serves the
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
selected experts together and accumulates the results; the summation itself
happens in top-k order so the output is bit-identical to the old serial
reduction. (Each contribution is weighted as its data arrives, but the additions
are always performed in the fixed top-k order.) `submit_expert_stage` returns a
`DispatchStage` for callers that want several stages in flight.

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
                 ◀── BRSP (one gate·y row per expert, tagged) ─────┤  (per 22)
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

# on each head server
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000
python3 tools/run_subcluster.py --config cluster.json --list
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000 \
    --check-members                      # PING every console behind this head
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000 \
    --refuse-fast                        # serve exact batches only
```

`run_subcluster.py` options, all optional except `--config`:

| option | meaning |
| --- | --- |
| `--subcluster ID` | which head to serve; required unless `--list` |
| `--host H`, `--port P` | listen elsewhere than the config says (a NAT'd or test bring-up) |
| `--standby N` | listen on the head's Nth extra config address instead (see three tiers below) |
| `--timeout S` | downstream budget in seconds for one batch |
| `--attempts N` | tries per expert across its primary and replicas |
| `--retry-on-timeout` | also retry a timed-out expert call (at-least-once execution) |
| `--retry-on-disconnect` | also retry a console that dropped mid-flight (at-least-once) |
| `--retry-on-node-error` | also retry on `ERR`/`NodeError` responses |
| `--no-replicas` | ignore any configured expert replicas |
| `--dedup-entries N`, `--dedup-ttl S`, `--dedup-bytes B` | bound of the reconnect replay cache |
| `--refuse-fast` | reject `fast` (partial-sum) batches with `ERR_BAD_REQUEST` |
| `--list` | print every head in the config and exit |
| `--check-members` | PING every console behind this head and exit |

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
- **All-or-error responses**: a subcluster answers for *every*
  requested expert (after any configured replica failover) or a `BERR` naming
  the failed experts and their consoles, raised as `SubclusterError` with
  `safe_to_retry` set only when no expert can have run. A short response is
  never reduced. Passing `replicas=[...]` pins entries to standby consoles when the
  layer already knows a primary is down.
- **Numerics (exact by default)**: a head server does *not* reduce. Its `BRSP`
  carries one `gate_j * y_j` row per expert, tagged with the expert it belongs
  to, and the layer accumulates them in ascending original top-k position with
  the same float32 additions as `DistributedExpertDispatcher`. The hierarchical
  result is therefore bit-identical to the flat one — `np.array_equal`, not
  `allclose` — no matter how the top-k positions interleave across subclusters
  or in what order the consoles finish. The cost is upstream bandwidth: `k`
  vectors instead of one per subcluster (the activation still travels once).
- **`fast=True` (opt-in, approximate)**: `HierarchicalExpertDispatcher(...,
  fast=True)`, or per call `run_expert_stage(..., fast=True)`, sets a request
  flag that asks each head server for a single partial sum instead. That
  re-associates the float32 additions when a group holds a non-contiguous slice
  of the top-k, so **logits and therefore token choices can change**. Never the
  default; a head server started with `--refuse-fast` rejects such batches with
  `ERR_BAD_REQUEST`.
- The flat `DistributedExpertDispatcher` and the canonical
  one-expert-per-layer-per-node placement are unchanged; a head server is an
  overlay, not a placement authority.

## Three tiers, and failover between coordinators

Condor's heads reported to servers above them; a single layer coordinator
speaking to 78 heads is the same bottleneck one tier up. `regional.py` adds the
middle tier, and it is the *same* coordinator: `RegionalCoordinator` subclasses
the `BaseCoordinator` that head servers use, speaks the same BREQ/BRSP/BERR, and
its downstream links are `SubclusterTransport` links — so a region is a client of
heads exactly as the layer is a client of regions, and a fourth tier would be
another instance of the same pair.

```
layer coordinator ─BREQ─▶ region rg-0000 (primary, standby…) ─BREQ─▶ head sc-0000 ─▶ 22 consoles
                  ─BREQ─▶ region rg-0001 …                   ─BREQ─▶ head sc-0001 …
                  ◀─BRSP: gate_j·y_j rows, tagged with the *global* top-k position
```

```bash
python3 tools/gen_cluster_config.py --layer 3 --experts 88 \
    --expert-host 10.0.0.10 --head-host 10.0.1.1 \
    --regions 2 --region-host 10.0.2.1 \
    --head-standby 1 --region-standby 1 -o cluster.json

python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000
python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000 \
    --standby 0                          # independent standby for same subcluster
python3 tools/run_region.py --config cluster.json --region rg-0000
python3 tools/run_region.py --config cluster.json --region rg-0000 --standby 0
python3 tools/run_region.py --config cluster.json --list
python3 tools/run_region.py --config cluster.json --region rg-0000 \
    --check-members                      # PING every head under this region
python3 tools/run_region.py --config cluster.json --region rg-0000 \
    --attempts 3 --retry-ambiguous       # see "retry semantics" below

python3 tools/run_layer.py --config cluster.json --layer 3 --token 0 \
    --experts 0 1 2 3 --gates 0.4 0.3 0.2 0.1 \
    --activation activation.npy --output out.npy
python3 tools/run_layer.py --config cluster.json --layer 3 --token 0 \
    --experts 0 1 2 3 --gates 0.4 0.3 0.2 0.1 \
    --activation activation.json --output out.json --fast --retry-attempts 3 \
    --retry-ambiguous --timeout 30.0
```

`--standby N` starts an *independent* coordinator process that fronts the same
subcluster/region as the primary, listening on the Nth extra address in the
config. It has its own memory and its own bounded dedup cache; it is not the same
process as the primary. Same-process deduplication only happens when a single
process listens on multiple addresses programmatically, or when a dropped socket
reconnects to the same process. Because normal primary/standby deployments are
separate processes, a retry that reaches the standby is at-least-once execution
(the caller still reduces exactly one complete answer). The config generator
creates the standby block as the next port block on the same host for laptop
bring-ups; for a production farm pass `--head-standby-host` / `--region-standby-host`,
or edit the `standby` blocks, so a standby does not share a machine with the
primary it covers. Without standby addresses `--standby N` exits with
`rg-0000 has 0 standby addresses, no index N`.

`run_region.py` takes most of the same options as `run_subcluster.py` except
that the layer→region link uses `--retry-ambiguous` instead of the per-expert
`--retry-on-timeout`/`--retry-on-disconnect`/`--retry-on-node-error` flags. `--region
ID` names the region and `--check-members` pings the heads under it rather than
consoles.

The layer coordinator points at the regions instead of the heads; nothing else
about its code changes:

```python
config = ClusterConfig.load("cluster.json")
transport = SubclusterTransport(config.region_endpoints(), timeout=30.0)
dispatcher = HierarchicalExpertDispatcher(config.placement(),
                                          config.tiered_plan(), transport)
y = dispatcher.run_expert_stage(layer, x, expert_ids, gate_weights, token_id)
```

`config.tiered_plan()` is a `TieredPlan`: consoles → heads → regions, so the
layer groups a token's top-k by *region* while each region regroups its slice by
head. Exactness is unaffected at any depth, because no tier reduces in exact
mode — the rows travel up tagged with the expert they belong to and only the
layer adds them, in its own top-k order. A config with no `regions` block still
drives the two-tier path, and `DistributedExpertDispatcher` still drives the
one-tier path.

**Coordinator replicas.** Every coordinator address in the config is an ordered
list: `{"id": "rg-0000", "host": …, "port": …, "standby": [{"host": …, "port":
…}]}`, at layer→region and region→head alike (expert replicas are unchanged).
The transport prefers the first endpoint it believes is alive and walks the list
on failure; PING/PONG *steers* that preference but never removes an endpoint, so
a cold or wrongly-marked-dead address is still tried when it is the only one
left. Head servers and regions keep no durable/shared state, so a standby is
just another listener on the same config; the bounded dedup/health caches held
by each process are ephemeral and do not span primary and standby. Health
preference changes only after an explicit `ping`/`check_liveness` or an observed
request failure — it is not a background monitor.

**Retry semantics, by what the caller can prove.** Each batch carries a 64-bit
request id (`REQ_FLAG_REQUEST_ID`) that a coordinator echoes in its BRSP *and*
its BERR, and that a retry reuses:

| situation | what is known | default |
| --- | --- | --- |
| no endpoint accepted the connection or the frame | downstream cannot have run | retried (`retry_safe`, at-most-once) |
| timeout, or the link died after the frame went out | downstream **may** be running | not retried; `LinkRetryPolicy(retry_ambiguous=True)` opts in |
| answer arrives naming another request id | not this batch | refused, never reduced |

The two exception shapes differ in what a tool can read off them: a coordinator
that answered with a BERR raises `SubclusterError` with the wire `code` set and
per-expert `failures`, while a coordinator whose *every* address was unreachable
raises `NodeConnectError` with `code=None` and no failures, because no frame ever
came back to carry them. Both name the endpoint, and neither is ever reduced.

An ambiguous retry is at-least-once *execution* and stays exactly-once
*reduction*: the abandoned attempt's correlation key is retired, so its late
answer is dropped rather than added, and the caller reduces exactly one complete
response. A retry that reaches the *same* coordinator process replays the first
attempt's frame from a bounded `DedupCache` (`--dedup-entries`, `--dedup-ttl`;
128 entries / 60 s by default, LRU-evicted and TTL-expired, so no unbounded
per-token state) and does not re-run the experts. A retry that reaches a
*different* process cannot be recognised — there is no shared store — and the
work does run twice.

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

## Limitations and validation

The coordination layer is tested on CPU-only loopback sockets; no physical PS3
has run the cluster code. For smaller-model validation without the full 82 k
node table, a tiny Kimi-K3 checkpoint such as
`inference-optimization/Kimi-K3-0.40B` on Hugging Face is a good target for a
single-node/few-node end-to-end run.
