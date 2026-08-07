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
  ppu/expert_ppu.c         PPE driver: resident expert + libspe2 fan-out + TCP worker
  ps3_cluster/
    protocol.py            length-prefixed big-endian frame codec
    topology.py            1-expert-1-layer/node planner + Kimi K3 profile
    dispatch.py            the #316 port: ExpertPlacement + Transport + dispatcher
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
reference — proving `mxfp4.h` + `p3xc.h` agree with the Python side.

On real hardware, `make ps3` (with the
[ps3dev toolchain](https://github.com/ps3dev/ps3toolchain)) builds the identical
worker with the SPU kernel embedded and libspe2 fan-out enabled; the coordinator
cannot tell a simulated node from a real console.

## Status

MVP. The dispatch layer, wire protocol, MXFP4 kernel, SwiGLU FFN, topology
planner, and host-sim worker are implemented and tested end to end. Not yet
done: a coordinator shim that hangs `DistributedExpertDispatcher` off a real
transformers `forward` via #316's hook points (documented in
`ps3-cluster/README.md`), AltiVec-vectorised SPU dequant, and validation on
physical PS3 hardware.
