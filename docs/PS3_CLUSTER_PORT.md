# PS3 Cluster Port: AirLLM per-expert streaming → 1 expert × 1 layer / node

This document explains the `ps3-cluster/` subtree: a port of
[AirLLM PR #316](https://github.com/lyogavin/airllm/pull/316) ("Support Kimi K3
(2.8T) — runs on a single card in 3.72GB") to a farm of PlayStation 3 consoles
running Linux under OtherOS (or a GameOS exploit such as AsbestOS), within the
RAM Coffers framing.

## The core idea

AirLLM #316 makes a 2.8T-parameter sparse-MoE model run on one small GPU by
**streaming individual experts**. A forward *pre*-hook on each expert module
loads that expert's weights from the local safetensors shard onto the device;
the expert runs; a *post*-hook evicts it back to `meta`. Because a sparse-MoE
layer only *calls* the experts a token routes to (top-k), only those hooks fire,
so a token materialises ~1 GB out of a ~55 GB layer instead of the whole thing.

RAM Coffers (upstream) applies a related idea *inside one POWER8 box*: partition
model knowledge across NUMA memory banks ("coffers") and only touch the relevant
bank. This port stretches a coffer to be an **entire PS3 console**, and shrinks
the unit of placement to a **single MoE expert of a single decoder layer**:

> **1 expert × 1 layer / node** — each console holds exactly one expert,
> resident in its 256 MB XDR RAM for the process lifetime.

The transformation from #316 is a single conceptual swap:

| AirLLM #316 (one box)                        | PS3 farm (this port)                                   |
|----------------------------------------------|--------------------------------------------------------|
| `_expert_pre_hook`: `load_layer_subset` from disk → device | `Transport.dispatch(layer, expert, x)` → owning console |
| expert weights stream in per token           | expert weights are **permanently resident** on the node |
| `_expert_post_hook`: evict to `meta`         | no-op (weights never leave the node)                    |
| model calls only the top-k expert submodules | dispatcher sends activations only to the top-k nodes    |
| MXFP4 kept packed, expanded on GPU           | MXFP4 kept packed in RAM, expanded on the Cell SPEs     |

So #316 trades **time** (stream experts from disk per token, I/O-bound) while
this port trades **space** (one expert per console, network-bound). The router's
top-k decision, which in #316 decides which hooks fire, here becomes the map of
which consoles light up.

## Why K3 fits

Kimi K3, as characterised by #316: 2.8T params, **82,432 experts across 92
layers** (~896/layer), MXFP4 4-bit, top-16 routing. A whole layer is ~17 GB
packed. Per expert that is **~17–19 MB packed** — trivially resident in a PS3's
256 MB. Top-16 routing means only ~1,500 of the ~82.5k nodes work per token
(see `tools/plan_k3.py`):

```
expert nodes         : 82,432
layer-coordinator    : 92        (attention + router + norms + shared expert)
embed/lm_head nodes  : 8
TOTAL consoles       : 82,532
active nodes / token : ~1,572    (98.1% idle per token)
```

Absurd physically; clean architecturally. This is a **throughput** design
(batched creative-writing / bulk generation), not a low-latency agentic runtime:
a token traverses ~92 layer hops sequentially, so per-token latency is high and
useful throughput comes from micro-batching many sequences through the pipeline.

## Components

```
ps3-cluster/
  cell-compat.h            PS3 Cell/OtherOS ISA compat (BE ppc64, VMX-only, SPE budget)
  common/
    p3xc.h                 wire protocol in C (byte-identical to protocol.py)
    mxfp4.h                MXFP4 (microscaling FP4) dequant + dot, shared PPE/SPU/host
  spu/expert_spu.c         SPE kernel: DMA-streamed MXFP4 GEMV slice
  ppu/expert_ppu.c         PPE driver: resident expert + backend select + TCP worker
  rsx/expert_rsx.cg        RSX Cg fragment shader: MXFP4 GEMV, one row/fragment
  rsx/expert_rsx.c         RSX host driver (PSGL/Cg), USE_RSX-guarded
  rsx/rsx_gemv_emu.h       CPU model of the shader math, for host testing
  ps3_cluster/
    protocol.py            length-prefixed big-endian frame codec
    topology.py            1-expert-1-layer/node planner + Kimi K3 profile
    dispatch.py            the #316 port: ExpertPlacement + Transport + dispatcher
    transport.py           persistent pooled P3XC transport (multiple in flight)
    subcluster.py          Condor-style subclusters of 22 + hierarchical reduce
    errors.py              node-attributed transport failure hierarchy
    node.py                reference expert worker (numpy) + TCP server
  tools/
    pack_expert.py         pack a SwiGLU expert to MXFP4 .exp + numpy reference
    plan_k3.py             print the cluster plan
  tests/                   protocol / topology / dispatch / cross-language kernel
  Makefile                 `make host` (portable) | `make ps3` (ppu-gcc + spu-gcc)
```

## Data path for one expert call

1. Coordinator runs the transformer graph with expert weights on `meta`. The
   router picks top-k `(layer, expert)` pairs and their gate weights.
2. For each pick, `DistributedExpertDispatcher` looks up the owning console in
   `ExpertPlacement` and sends the input activation as a P3XC `REQ` frame
   (fp32, big-endian on the wire; a no-op byteswap on the BE PPE).
3. The console's `expert_ppu` worker runs the resident expert's SwiGLU FFN —
   `down(silu(gate·x) * up·x)` — with each GEMV split across the Cell SPEs
   (`expert_spu.c`), MXFP4 weight tiles DMA-streamed through the 256 KB local
   store. It replies with a `RSP` frame.
4. The dispatcher combines the top-k outputs with the gate weights:
   `y = Σ gate_j · expert_j(x)`.

Steps 2–4 happen for all k picks *concurrently* — see below.

## Asynchronous coordination

The first cut of the dispatcher opened a TCP connection per expert call and
walked the top-k list one expert at a time, so a layer cost
`k × (connect + compute + round-trip)` even though the k consoles are
independent machines. Every PS3 cluster that did real work solved this the same
way, and so does this port.

### Persistent connections and connection lifecycle

`PersistentSocketTransport` (`transport.py`; `PooledSocketTransport` is an alias
for migration) keeps a bounded pool of long-lived TCP connections per node. This
is the cluster-scale form of ALF's persistent per-accelerator work queue and
DaCS's standing host↔accelerator connection: the expensive channel is
established once and reused for the whole run, and work is handed to it
asynchronously.

```
coordinator                                     console (expert node)
  connect()  ------------------------------->   accept(), fork per connection
  REQ  (layer, expert, token=7)  ----------->   compute SwiGLU on SPE/RSX/PPE
  REQ  (layer, expert, token=8)  ----------->     (queued behind token=7)
  PING (token=8)                 ----------->   PONG
  <----------------------------------- RSP (token=7)
  <----------------------------------- RSP (token=8)
  ... connection stays open for the life of the run ...
  close()   -------------------------------->   reader thread joined, fd closed
```

Wire-level rules:

- **One writer at a time.** A per-connection lock makes each frame's bytes
  contiguous on the wire; encoding happens outside the lock.
- **One reader per connection.** A daemon reader thread owns all `recv`, decodes
  frames, and hands each to the waiter it belongs to. Callers never read the
  socket, so several requests can be outstanding.
- **Correlation by request identity.** A waiter is keyed by the frame's
  `(layer, expert, token_id)`, so responses may arrive in any order. Worker
  `ERR` frames, whose `layer`/`expert` describe the *node* rather than the
  request, fall back to matching a unique in-flight token.
- **Bounded pool, not queueing.** A worker computes one request at a time per
  connection, so the transport prefers an idle connection, then grows the pool up
  to `max_connections_per_node`, and only multiplexes onto a busy connection at
  the bound.
- **Deterministic teardown.** `close()` shuts each socket down, joins its reader,
  fails every outstanding waiter, and makes further use raise `TransportClosed`.
  It is idempotent, and the transport is a context manager.

The protocol did not change: framing is the same fixed big-endian P3XC, and the
old `SocketTransport` (connection per dispatch) still works and is still used by
the existing tests.

### Concurrent top-k with deterministic reduction

`DistributedExpertDispatcher.run_expert_stage` submits all k expert calls to a
thread pool at once and waits for the whole stage; `submit_expert_stage` returns
a `DispatchStage` if the caller wants several stages in flight (the double
buffering ALF used between host and accelerator, applied to layers).

Concurrency does not affect the numbers. Each contribution is scaled by its gate
weight as it arrives, but the **summation is always performed in top-k order**,
never in completion order, so results are bit-identical to the old serial loop
(`tests/test_async_dispatch.py` asserts `np.array_equal`, not a tolerance).

### Heartbeats

`transport.ping(node)` returns a round-trip time using the protocol's PING/PONG
frames, `alive(node)` reduces that to a bool, and
`dispatcher.check_liveness(nodes)` probes many nodes concurrently. Heartbeats
share the connection with expert traffic, and both workers answer them
(`node.py`, `ppu/expert_ppu.c`). This is the coordinator-side supervision the
Gravity Grid and pLBM PS3 clusters ran from their head nodes, minus MPI.

### Subclusters

AFRL's Condor cluster grouped its 1,716 PS3s into subclusters of **22** behind
coordinating servers rather than one flat fabric. `subcluster.py` provides that
shape as data: `SubclusterPlan(node_ids, size=22)` groups nodes stably by input
order, answers membership queries, and buckets a token's active nodes by group.
`hierarchical_reduce` then sums each group's contributions into a partial — what
a head server would forward upstream — and sums the partials in group order,
reporting each partial through an `on_partial` callback.

`subcluster.py` itself is only the grouping and reduction plan: it does not
change placement (canonical 1 expert × 1 layer / node is untouched) and adds no
dependency. The processes that use it are below.

### Deployed subcluster coordinators

`coordinator.py` turns that grouping into Condor's actual process topology: one
head-server process per subcluster, with the layer coordinator talking to heads
instead of to consoles.

```
            layer coordinator  (hierarchy.py: HierarchicalExpertDispatcher)
              |                       |
     BREQ (activation once   +  [(expert, gate), ...]),  one per subcluster,
     sent to the involved heads concurrently, several batches per connection
              v                       v
     head server sc-0000        head server sc-0001      (coordinator.py:
     SubclusterCoordinator      SubclusterCoordinator     SubclusterService)
       | | | ... (<=22)           | | | ... (<=22)
     P3XC REQ/RSP over the *same* pooled persistent transport as the flat path,
     fanned out concurrently and gate-scaled, but *not* summed
              |                       |
     BRSP: gate_j*y_j per expert, each row tagged with its expert
              \_______________________/
                          v
            layer sums the rows in ascending original top-k position,
            with the same fp32 adds as DistributedExpertDispatcher
```

**Wire format (`batch.py`).** Three new P3XC message types reuse the existing
fixed big-endian header and array payload plus a type-specific trailer, so an
expert worker's parser is untouched (it simply never sees these types) and
length prefixing is unchanged:

| Type | Value | Body |
|---|---|---|
| `BREQ` | 6 | header (`expert=0xFFFF`) + activation + `n_entries:u16`, `flags:u16` (`0x0001` = `REQ_FLAG_FAST`, `0x0002` = `REQ_FLAG_REQUEST_ID`, others rejected), `deadline_ms:u32`, then `n_entries × {expert:u16, replica:u8, reserved:u8, gate:f32}`, then `request_id:u64` if flagged |
| `BRSP` (exact, default) | 7 | header + `n_reduced` stacked `gate_j*y_j` rows + `n_reduced:u16`, `flags:u16` = `0x0001` (`RSP_FLAG_PER_EXPERT`) \| `0x0002` (`RSP_FLAG_REQUEST_ID`), then `n_reduced × expert:u16` tags in row order, then the echoed `request_id:u64` if flagged |
| `BRSP` (fast, opt-in) | 7 | header + one partial sum + `n_reduced:u16`, `flags:u16` (`RSP_FLAG_PER_EXPERT` clear), then the echoed `request_id:u64` if flagged |
| `BERR` | 8 | header + `[0.0]` + `code:u16`, `n_failures:u16`, then `n_failures × {expert:u16, reason:u16, node_len:u16, node bytes}`, then `detail_len:u16` + UTF-8 detail, then optionally the echoed `request_id:u64` (a trailer of any other length is refused) |

The activation travels **once per subcluster request** — 8 bytes per additional
expert instead of another 28 KB at K3 width (hidden 7168, fp32) — which is the
same reason ALF has the host enqueue one work-block descriptor list per
accelerator rather than one message per SPE. Every count and string is bounded
(`MAX_BATCH_ENTRIES`, `MAX_STRING_BYTES`, `MAX_FRAME_BYTES`, `MAX_DEADLINE_MS`),
reserved fields must be zero, duplicate experts in one batch are rejected, and
`read_frame` refuses a zero-length or oversized length prefix *before*
allocating, so a corrupt or hostile frame cannot make a head server allocate
without limit. A malformed `BREQ` is answered with `BERR ERR_BAD_REQUEST` on the
same connection; an unrecoverable framing error drops the connection.

**Lifecycle.** An upstream connection is persistent and carries many batches:
each `BREQ` is executed on the head server's own pool, so replies may come back
out of order, correlated by `(layer, 0xFFFF, token_id)` — the identity the
pooled transport already keys on, which is why the layer side is a thin wrapper
over `PersistentSocketTransport` rather than a second transport. `PING` is
answered by the head server itself (a `PONG` means *this coordinator* is up;
`SubclusterCoordinator.check_members()` probes the 22 consoles). A request's
`deadline_ms` bounds the head server's downstream calls to
`min(deadline, its own timeout)`, so a stuck console cannot hold an upstream
request past the layer's budget. `SubclusterService.stop()` stops accepting,
drains in-flight batches, joins the accept thread, and closes the downstream
transport and its reader threads.

**Execution vs reduction semantics.** A subcluster returns either a response
covering *every* requested expert (after any configured replica failover) or a
`BERR` naming each failed expert, its reason and its console — surfaced to the
layer as `SubclusterError`, whose `safe_to_retry` is true only when every named
failure provably never ran an expert. A short sum is never produced and never
reduced: the layer also verifies `n_reduced` against the number of experts it
asked for. Retries inside a subcluster follow the same `RetryPolicy` as the flat
path, and a late response to an abandoned request is dropped by the transport
rather than summed, so each contribution is reduced exactly once even if the
underlying call executed twice. An entry's `replica` byte pins that expert to a
standby console (0 = the canonical one), so a layer that already knows the
primary is down does not pay the failed attempt again; a hint naming a replica
that is not configured is answered `BERR ERR_BAD_REQUEST` rather than silently
falling back to the primary.

**Numerics: exact by default.** A head server must not collapse its slice of the
top-k to one fp32 partial, because that slice is generally *not contiguous* in
the layer's order: for positions `a0, b1, a2` split over groups `a` and `b`,
grouping computes `(a0 + a2) + b1` where the flat path computes
`(a0 + b1) + a2`, and those differ in the last bits, which is enough to move a
logit and therefore a token choice. So in the default mode the head server
scales but does not reduce: its `BRSP` carries one `gate_j * y_j` row per
expert, each tagged with its expert id, and the layer places each row back at
the top-k position it asked for and accumulates strictly in ascending position
with the same float32 adds as `DistributedExpertDispatcher.reduce`. The
deployed hierarchical result is therefore **bit-identical to flat dispatch**
(`np.array_equal`), independently of how positions interleave across
subclusters, of completion order, and of whether an expert was served by a
replica. A response tagged with an expert the layer did not request, or carrying
a single partial when rows were asked for, is refused rather than reduced.

The price is upstream bandwidth: `k` vectors instead of one per subcluster (the
*activation* still travels once, so a batch is still far cheaper than one
request per expert). That tradeoff is deliberate — bit identity is mandatory.

**Fast mode (opt-in, approximate).** `REQ_FLAG_FAST` — exposed as
`HierarchicalExpertDispatcher(..., fast=True)`, `run_expert_stage(...,
fast=True)`, `submit_expert_stage(..., fast=True)` and
`SubclusterTransport.submit_batch(..., fast=True)` — asks each head server for a
single partial sum, reducing in ascending expert order within the group and
ascending group id at the layer. It saves upstream bytes and matches the
in-process `hierarchical_reduce`, and it is reproducible, but it re-associates
the float32 additions: **logits, and therefore sampled tokens, can differ from
the flat path**. It is never the default, and a head server run with
`--refuse-fast` (`SubclusterCoordinator(..., allow_fast=False)`) answers such a
batch with `BERR ERR_BAD_REQUEST` instead of a partial.

**Configuration (`deployment.py`).** One JSON document declares which head
server fronts which consoles, and both ends read it, so the two sides cannot
disagree about who owns an expert:

```json
{"subcluster_size": 22,
 "subclusters": [{"id": "sc-0000", "host": "10.0.1.1", "port": 8100,
                  "members": [{"layer": 3, "expert": 0,
                              "node": "ps3-L003-E0000",
                              "host": "10.0.0.10", "port": 9000,
                              "replicas": [{"node": "ps3-L003-E0000-b",
                                            "host": "10.0.0.210",
                                            "port": 9000}]}]}]}
```

`ClusterConfig` derives the canonical placement (replicas additive, never extra
experts), the console endpoints, the head endpoints, and a `SubclusterPlan` built
from the *declared* membership. No MPI, no external runtime; CPU-only hosts
included. The three process tiers are started like this:

| Tier | Command |
|---|---|
| console | `build/expert_node_host expert.exp <port>` (real hardware), or `python3 tools/run_expert.py expert.exp --port <port>` — the numpy reference worker, `--identity` for a trivial expert |
| head server | `python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000`; `--host`/`--port` override the config, `--standby N` serves the head's Nth extra address, `--timeout` sets the downstream budget in seconds, `--attempts` the retries per expert and `--retry-ambiguous` widens which failures are retried, `--dedup-entries`/`--dedup-ttl` bound the replay cache, `--refuse-fast` rejects approximate batches with `ERR_BAD_REQUEST`, `--list` and `--check-members` inspect without serving |
| layer | any process holding a `SubclusterTransport` + `HierarchicalExpertDispatcher` over the same `cluster.json` (see `ps3-cluster/README.md`) |

`tools/gen_cluster_config.py` writes the config for one layer (`--experts`,
`--size`, `--expert-host`/`--expert-port-base`, `--head-host`/`--head-port-base`,
and `--head-standby N`/`--head-standby-host` for the extra head addresses that
`--standby` serves; `--regions`/`--region-host`/`--region-port-base` and
`--region-standby N`/`--region-standby-host` do the same one tier up). Standby
ports are the next block after the primaries on the same host — usable as
generated on one machine, and a starting point for a farm, where the hosts should
be edited so a standby does not share a machine with the primary it covers.
A head server exits cleanly on SIGINT/SIGTERM, reporting how many batches it
served.

### Regional coordinators: the same tier, one level up

Condor's heads answered to servers above them, and one layer coordinator holding
78 head connections per layer is the bottleneck the heads were meant to remove.
`regional.py` adds that tier without adding a protocol: `RegionalCoordinator`
subclasses the same `BaseCoordinator` as `SubclusterCoordinator` (shutdown
checks, ownership checks, fast-mode policy, deadline arithmetic, dedup) and its
downstream side is a `SubclusterTransport` — the *layer's* client code. A region
is therefore a head server whose "consoles" are head servers, and a fourth tier
is the same pair again.

```
                 layer coordinator          (HierarchicalExpertDispatcher over a TieredPlan)
                  |                    |
        BREQ per *region*, activation once, entries tagged with global top-k positions
                  v                    v
        region rg-0000            region rg-0001         (RegionalCoordinator +
        primary + standby…        primary + standby…      CoordinatorService)
          |          |
        BREQ per *head*, activation once again per immediate downstream group
          v          v
        head sc-0000  head sc-0001  …                    (SubclusterCoordinator)
          |             |
        22 consoles    22 consoles                       (P3XC REQ/RSP)
```

`ClusterConfig.tiered_plan()` returns a `TieredPlan` (consoles → heads →
regions): the layer groups a token's top-k by region, each region regroups its
slice by head. **No tier reduces in exact mode**, so bit identity is a property
of the depth-independent rule "only the layer adds, in its own top-k order":
`tests/test_three_tier.py` asserts `np.array_equal` against
`DistributedExpertDispatcher` over randomised routings whose positions interleave
across both regions and subclusters, with reversed completion order. Fast mode
nests too — partials of partials — and is correspondingly *more* re-associated.
A config with no `regions` block still drives the two-tier path, and the flat
dispatcher still drives the one-tier path; canonical placement is untouched at
every depth.

**Coordinator replicas and health steering.** Every coordinator address in the
config is an ordered list (`"standby": [{"host": …, "port": …}]`) at layer→region
and region→head alike, in addition to the existing expert replicas. The
transport sends to the first endpoint it believes is alive and walks the list on
failure. `PING`/`PONG` only *steers* that preference: an endpoint marked dead is
sorted last, never removed, so a wrongly-marked or never-probed address is still
tried when it is the only one left — correctness never depends on health state.
Head servers and regions are stateless, so a standby is just another listener on
the same `cluster.json` (`--standby N`).

**Request identity.** Each batch carries a 64-bit id (`next_request_id()`:
process-random high bits + a counter, so two layer coordinators cannot mint the
same id) which every coordinator echoes in its `BRSP` *and* its `BERR`, and which
a retry of the same logical batch reuses. An answer naming a different id is
refused rather than reduced. Correlation on the wire still keys on
`(layer, 0xFFFF, token_id)`; the id is the *logical* identity on top of it.

**Retry classes on a coordinator link (`LinkRetryPolicy`).**

| Class | What the caller can prove | Default |
|---|---|---|
| safe-before-send | no endpoint accepted the connection or the frame; downstream cannot have started | retried on the next endpoint (`retry_safe=True`) — at-most-once |
| ambiguous | timeout, or the link died after the frame went out; downstream may be running or finished | **not** retried; `retry_ambiguous=True` opts in — at-least-once execution |
| wrong identity | the answer echoes another request id | refused, never reduced |

On an ambiguous failure the attempt's correlation key is retired and the endpoint
is marked dead before the retry, so the abandoned answer is dropped rather than
summed. **Exactly-once reduction is preserved in every class**; only *execution*
weakens, and only when explicitly opted into.

**Bounded idempotency (`dedup.py`).** A coordinator remembers the response frame
per request id in a `DedupCache`: a retry that lands on the *same* process (a
reconnect, or the caller reaching it through another of its own addresses)
replays that frame instead of fanning out again, and a duplicate arriving while
the first attempt is still running waits for it rather than starting a second.
The cache is bounded and expiring — `--dedup-entries` (128) and `--dedup-ttl`
(60 s), LRU eviction, failed attempts not remembered — so no unbounded per-token
state accumulates; an evicted or expired id honestly re-executes. A retry that
lands on a *different* replica process cannot be recognised: there is no shared
store, and introducing one would mean a consensus dependency this design
refuses. That case is at-least-once execution with exactly-once reduction, and
it is documented rather than papered over.

**Launching the third tier.**

| Tier | Command |
|---|---|
| region | `python3 tools/run_region.py --config cluster.json --region rg-0000`; `--standby N` serves the region's Nth extra address, `--attempts`/`--retry-ambiguous` set the link policy, `--dedup-entries`/`--dedup-ttl` bound the replay cache, `--refuse-fast` rejects approximate batches, `--list`/`--check-members` inspect without serving |
| layer | `SubclusterTransport(config.region_endpoints())` + `HierarchicalExpertDispatcher(config.placement(), config.tiered_plan(), transport)` |

`tools/gen_cluster_config.py --regions N --region-host … --region-port-base …`
writes the three-tier config, dealing heads out contiguously so a token's top-k
lands in few regions; add `--region-standby N` (and `--head-standby N`) for the
extra addresses `--standby` serves, or `--standby N` exits with `rg-0000 has 0
standby addresses, no index 0`. `tests/test_deployment_cli.py` brings the whole
thing up as real processes through these CLIs and compares against the flat
dispatcher.

### Failure semantics

Failures are node-attributed (`errors.py`), so a log names the console:

| Exception | Cause | Safe to retry |
|---|---|---|
| `NodeConnectError` | connect failed / unknown node | yes — request never left the coordinator |
| `NodeDisconnected` | peer closed mid-request | no — the expert may have run |
| `NodeTimeout` | no response within the deadline | no — the expert may still be running |
| `NodeError` | worker answered `ERR` | no by default — usually a placement/shape bug, not transient |
| `PoolExhausted` | identical request already in flight on every pooled connection | yes |
| `TransportClosed` | use after `close()` | no |

A coordinator link carries the same exceptions, and tooling that inspects `.code`
should expect two shapes: a coordinator that *answered* raises `SubclusterError`
with the BERR `code` and per-expert `failures`, while a coordinator whose every
configured address was unreachable raises `NodeConnectError` with `code=None` and
no failures — nothing came back to carry them. Neither is ever reduced.

`RetryPolicy` decides which of those are retried, and
`ExpertPlacement.assign_replica(layer, expert, node)` registers optional standby
consoles holding the same expert (replicas are *not* placements: `node_for` and
`len(placement)` are unchanged). On a retryable failure the dispatcher tries the
next replica, and `DispatchStage.attempts` records the trail per top-k position.

Execution semantics, stated precisely:

- **Default (`RetryPolicy()`): at-most-once execution.** Only failures that
  provably never reached a node are retried, so an expert is evaluated at most
  once per top-k position.
- **`retry_on_timeout=True`: at-least-once execution.** A timed-out request may
  still be running on the original console while the replica computes the same
  activation. Use it only for idempotent experts — which a stateless FFN is.
- **Exactly-once reduction, always.** A timed-out request's correlation key is
  retired, so if the slow response does turn up it is dropped rather than summed;
  each top-k position contributes exactly one value to `y`. There is no retry
  path that can double-count an expert.

The default is at-most-once precisely because silently double-dispatching would
be indistinguishable from a routing bug at the numeric level.

### Worker side

Both workers keep a connection open and serve frames until the peer hangs up.
The C worker forks a process per connection: `fork()` shares the resident MXFP4
weights copy-on-write and they are only ever read, so a pooled coordinator costs
no extra XDR RAM, while a single idle connection can no longer stall the accept
loop. Requests on one connection are still served sequentially — the SPEs are
already fanned out across one GEMV, so a second concurrent request per console
would contend for them rather than add throughput. Concurrency comes from having
many consoles and, where useful, several connections per console.

## Compute backends

`expert_ppu.c` selects the GEMV backend at compile time (all share one ABI, so
the coordinator can't tell them apart):

| Backend | Macro | Where it runs | Build |
|---------|-------|---------------|-------|
| Scalar PPE | *(default)* | any CPU (host-sim, or PPE fallback) | `make host` → `expert_node_host` |
| Cell SPEs | `USE_SPE` (auto on `__PPU__` + `HAVE_LIBSPE2`) | 6 SPEs via libspe2, DMA-streamed MXFP4 tiles | `make ps3` |
| RSX GPU | `USE_RSX` | RSX fragment shader via PSGL/Cg (GameOS exploit) | `make ps3-rsx` |
| RSX shader model | `GEMV_RSX_EMU` | CPU model of the shader, for testing | `make host` → `expert_node_rsxemu` |

**RSX shader path.** `rsx/expert_rsx.cg` computes one output row per fragment:
the MXFP4-packed weights are an R8 texture (`row_bytes × rows`), the activation
an R32F texture, and the 16-entry E2M1 table a 16×1 LUT texture; the shader
unpacks nibbles, applies the E8M0 `exp2` block scale, multiplies by the input
texels, accumulates, and writes to a `1 × rows` fp32 render target that the host
reads back. This targets the GameOS-exploit boot where the RSX (NV47/G70-class,
SM3.0, fp32 fragments) is fully programmable; it is an alternative to the SPE
backend, not a replacement — a node can use whichever engine is available.

Because there is no RSX toolchain or GPU in CI, `rsx/rsx_gemv_emu.h` reproduces
the shader's exact per-fragment math (byte normalise/recover, LUT lookup,
`exp2` scale, nibble unpack) on the CPU, and `tests/test_rsx_kernel.py` checks
that the emulated shader output matches the numpy reference end to end. A full K3
row (n=7168 → 224 blocks) is a long SM3.0 loop and may need column-tiling into
multiple passes on real hardware; the host driver is structured to allow that.

## Cell / OtherOS constraints (see `cell-compat.h`)

- **Big-endian ppc64 PPE.** GGUF/safetensors are little-endian; multi-byte
  values are byte-swapped on load. The wire protocol is fixed big-endian so BE
  nodes pay nothing and only x86 hosts swap.
- **No VSX / no MMA / no POWER8-9 builtins.** RAM Coffers' VSX collapse kernels
  must select scalar/classic-AltiVec paths when `GGML_PS3_CELL` is set.
- **6 usable SPEs** under OtherOS (7 with a GameOS exploit), **256 KB** local
  store each → weights must be DMA-streamed in tiles, never resident whole.
- **256 MB XDR RAM** per console is the binding constraint and the reason the
  placement unit is a single expert.
- **RSX GDDR3 (256 MB, ~240 MB usable)** is a *conditional* second tier:
  - Under **OtherOS** it is mappable via the hypervisor but reads at only
    ~16 MB/s, so it is useless for weights read every token — cold storage at
    best. XDR stays the binding constraint and placement stays 1 expert/node.
  - Under a **GameOS exploit (AsbestOS)** you get full ~22.4 GB/s access *and*
    the programmable NV47/G70 shader pipeline, making RSX a genuine hot tier: a
    node can hold ~2 experts (one in XDR, one in RSX) and could even run the
    expert GEMV as RSX fragment shaders instead of on the SPEs. The planner
    models this with `plan_cluster(..., rsx=True)` /
    `tools/plan_k3.py --rsx`, which widens per-node capacity (200 → 440 MB) and
    reports how many experts fit; packing is opt-in via `experts_per_node` so
    the default stays the canonical 1-expert/node design.

## Building & testing

```bash
cd ps3-cluster
make host                                  # portable host-sim worker
python3 -m unittest discover -s tests -v   # protocol/topology/dispatch/kernel
python3 tools/plan_k3.py                    # cluster sizing for Kimi K3
```

The cross-language test (`tests/test_cell_kernel.py`) packs a real MXFP4 expert,
boots the compiled C worker, dispatches through the Python coordinator's
`SocketTransport`, and asserts the C SwiGLU/MXFP4 output matches the numpy
reference — proving `mxfp4.h` + `p3xc.h` agree with the Python side. It also
drives the same binary over a pooled connection (many `REQ`s plus a `PING` on one
accept) and checks that an idle connection does not block the accept loop.

The coordination tests drive **real sockets** — real workers on loopback, no
mocks — and use barriers/events rather than latency thresholds so they are not
timing-sensitive:

| File | Covers |
|---|---|
| `test_persistent_transport.py` | connection reuse, overlapping requests, out-of-order correlation, PING/PONG, timeout and late-response handling, `ERR` recovery, socket/thread cleanup |
| `test_async_dispatch.py` | top-k experts running simultaneously (a barrier that a serial dispatcher cannot satisfy — with `max_workers=1` as the control), bit-identical reduction, error cleanup, liveness |
| `test_subcluster.py` | grouping into 22s, stability, partial/hierarchical reduction, dispatch reduced through two subclusters |
| `test_failover.py` | replica registration, safe-only retries, `ERR`/timeout policies, and per-node request counts proving no expert is counted twice |
| `test_batch_protocol.py` | batch frame round-trips for both response shapes, the `REQ_FLAG_FAST`/`RSP_FLAG_PER_EXPERT` flags and unassigned-flag rejection, per-expert rows surviving the wire bit for bit with their tags, forged row counts and duplicate tags refused, activation carried once regardless of k, bounded counts/strings, reserved-field and duplicate-expert rejection, truncation, version mismatch, expert-frame compatibility |
| `test_deployment.py` | JSON config round-trip, canonical placement with additive replicas, plan built from declared membership, size/duplicate validation, 22-console grouping, region declarations with ordered standby addresses, per-region head endpoints and `TieredPlan`, region generation, and two-tier configs still loading |
| `test_dedup.py` | bounded replay: a repeated id answered from cache without re-running, a duplicate arriving mid-flight waiting for the first, failures not remembered, TTL expiry, LRU eviction, counters, unflagged requests never cached |
| `test_three_tier.py` | two regions over four head servers over eight consoles, in-process: three-tier output `np.array_equal` to the **flat dispatcher's own** output under randomised interleaving across regions *and* subclusters, positions preserved through both tiers, reverse-order completion, fast mode as an explicit partial (and a region refusing it), a dead region primary and a dead head primary failing over before send, both region endpoints down reported rather than reduced, a cold endpoint still tried when every endpoint is cold, heartbeat steering the next batch, timeout with opt-in ambiguous retry, a retry to the same process replaying instead of re-running, a retry to another replica executing twice but reduced once, a late duplicate answer dropped, a foreign request id refused, structured errors through both tiers, activation once per immediate downstream group, connection reuse across tokens, and socket/thread cleanup |
| `test_deployment_cli.py` | the documented bring-up as real processes: `gen_cluster_config.py --regions` → four `run_expert.py` consoles → four `run_subcluster.py` heads → two `run_region.py` regions → a layer client whose output is `np.array_equal` to the flat dispatcher's over the same consoles; a `--standby` region process serving the same config, `--list`/`--check-members`, and a two-tier config refusing `--region` |
| `test_hierarchy.py` | two live head servers over four consoles: default-mode output `np.array_equal` to the **flat dispatcher's own** output over the same consoles — across 12 randomised routings whose top-k positions interleave over three subclusters, when consoles complete in reverse position order, and when one contribution comes from a replica — plus fast mode matching the grouped reduction, the fast flag appearing only when asked for, its smaller reply, and a head server refusing fast batches; one `BREQ` per subcluster with the activation appearing exactly once on the wire (asserted through a byte-counting proxy), barriers proving concurrency within *and* across subclusters, two batches multiplexed on one upstream connection, upstream/downstream connection reuse, head-server and console heartbeats, malformed/oversized/unsupported frames, unreachable and erroring consoles as structured `BERR`, replica failover counting an expert once, deadline propagation, and socket/thread cleanup after `stop()` |

On real hardware, `make ps3` (with the
[ps3dev toolchain](https://github.com/ps3dev/ps3toolchain)) builds the identical
worker with the SPU kernel embedded and libspe2 fan-out enabled; the coordinator
cannot tell a simulated node from a real console.

## Status

MVP. The dispatch layer, persistent pooled transport, concurrent top-k fan-out,
heartbeats, subcluster planner, deployable per-subcluster and regional
coordinator processes with their batched wire format and JSON membership config,
coordinator endpoint replicas with request ids and bounded replay,
retry/failover hooks, wire protocol, MXFP4
kernel, SwiGLU FFN, topology planner, host-sim worker, SPE kernel, and RSX shader
backend (with a CPU-validated shader model) are implemented; the numeric path is tested end to
end against a numpy reference. Not yet done: a coordinator shim that hangs
`DistributedExpertDispatcher` off a real transformers `forward` via #316's hook
points (documented in `ps3-cluster/README.md`), dependency-scheduled multi-stage
work queues à la ALF, load-aware
replica selection, AltiVec-vectorised SPU dequant,
column-tiling the RSX shader for full-width K3 rows, and validation on physical
PS3 hardware (both the SPE and RSX paths compile only against their toolchains,
which aren't present in CI, so they are checked by syntax/stub compilation plus
the CPU shader model).

The coordination layer is validated on loopback sockets only. Nothing here has
run on a physical PlayStation 3, and the latency/bandwidth characteristics of a
real console farm (100 Mbit NICs, OtherOS hypervisor overhead) are not modelled.

Further limitations of the deployed hierarchy specifically: the tree is three
levels (layer → region → head → console) and the coordinator abstraction is
recursive, but only three levels are configurable through `ClusterConfig` — a
fourth would need a config change, not a protocol one. A retry that reaches a
*different* coordinator process re-executes the batch downstream (at-least-once
execution, exactly-once reduction), because coordinators share no dedup state and
nothing here introduces a consensus service to give them one. Health state is
advisory: steering is per-transport and per-process, so a fresh layer coordinator
still pays one failed attempt to discover a dead primary. Batching is per
immediate downstream group per token, not across tokens, and there is no
load-aware choice among replicas or regions. Default-mode
replies cost `k` vectors upstream rather than one per subcluster, which is the
price of bit identity; the opt-in fast mode trades that back for a partial sum
and can change token choices (see numerics above). Requests on one console
connection remain sequential, as before. For non-PS3 model validation, a small
Kimi-K3 checkpoint (e.g. `inference-optimization/Kimi-K3-0.40B` on Hugging Face)
is a useful single-node/few-node target.

## Prior art

The coordination design follows what PS3 clusters and Cell middleware actually
did, rather than inventing a scheme:

- M. Barnell et al., *[High Performance Computing (HPC) and Data Analytics on the
  Condor Cluster](https://ieee-hpec.org/2012/index_htm_files/Barnell.pdf)*, IEEE
  HPEC 2012 — AFRL's 1,716-console cluster, organised as subclusters of 22 PS3s
  behind coordinating servers. Source of the default subcluster size.
- University of Rhode Island, *[PS3 Gravity Grid](https://web.uri.edu/gravity/ps3/)*
  — a PS3 cluster running black-hole simulations with a head node supervising the
  consoles' PPEs while the SPEs do the numerics.
- K. Nomura et al., *[A Metascalable Computing Framework for Large
  Spatiotemporal-Scale Atomistic
  Simulations](https://aiichironakano.github.io/cs653/Nomura-pLBM-IJCS08.pdf)*
  — PS3 lattice-Boltzmann with PPE supervision and SPE offload.
- IBM, *[ALF Programmer's Guide and API Reference
  v3.0](https://arcb.csc.ncsu.edu/~mueller/cluster/ps3/SDK3.0/docs/lib/ALF_Prog_Guide_API_v3.0.pdf)*
  — persistent per-accelerator work queues, asynchronous task submission,
  dependency scheduling and double buffering. The model behind the persistent
  transport and stage pipelining.
- IBM, *[DaCS Programmer's Guide and API Reference
  v3.0](https://arcb.csc.ncsu.edu/~mueller/cluster/ps3/SDK3.0/docs/lib/DaCS_Prog_Guide_API_v3.0.pdf)*
  — standing host↔accelerator connections, remote error notification and
  process/group topology. The model behind connection reuse and node-attributed
  errors.

Borrowed here are the topology and the communication patterns, not the software:
this port uses plain TCP and threads, with no MPI, ALF or DaCS dependency, and
the dependency-scheduled work queues and dynamic load balancing of ALF are *not*
implemented.
