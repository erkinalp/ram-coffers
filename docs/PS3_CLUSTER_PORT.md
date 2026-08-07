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

This is deliberately *only* a grouping and reduction plan: it does not change
placement (canonical 1 expert × 1 layer / node is untouched), does not introduce
MPI or any other dependency, and the coordinator still speaks P3XC directly to
each console. Actually running a coordinator process per subcluster is future
work that this abstraction is meant to make mechanical.

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

On real hardware, `make ps3` (with the
[ps3dev toolchain](https://github.com/ps3dev/ps3toolchain)) builds the identical
worker with the SPU kernel embedded and libspe2 fan-out enabled; the coordinator
cannot tell a simulated node from a real console.

## Status

MVP. The dispatch layer, persistent pooled transport, concurrent top-k fan-out,
heartbeats, subcluster planner, retry/failover hooks, wire protocol, MXFP4
kernel, SwiGLU FFN, topology planner, host-sim worker, SPE kernel, and RSX shader
backend (with a CPU-validated shader model) are implemented; the numeric path is tested end to
end against a numpy reference. Not yet done: a coordinator shim that hangs
`DistributedExpertDispatcher` off a real transformers `forward` via #316's hook
points (documented in `ps3-cluster/README.md`), a coordinator process per
subcluster (the planner describes the grouping but fan-out is still flat from one
coordinator), dependency-scheduled multi-stage work queues à la ALF, load-aware
replica selection, AltiVec-vectorised SPU dequant,
column-tiling the RSX shader for full-width K3 rows, and validation on physical
PS3 hardware (both the SPE and RSX paths compile only against their toolchains,
which aren't present in CI, so they are checked by syntax/stub compilation plus
the CPU shader model).

The coordination layer is validated on loopback sockets only. Nothing here has
run on a physical PlayStation 3, and the latency/bandwidth characteristics of a
real console farm (100 Mbit NICs, OtherOS hypervisor overhead) are not modelled.

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
