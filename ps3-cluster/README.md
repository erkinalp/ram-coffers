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
  instead of local disk loads; `topology.py` plans the 1-expert/node layout;
  `protocol.py` is the big-endian wire codec; `node.py` is a reference
  (numpy) expert worker.
- **Cell kernels** (`common/`, `spu/`, `ppu/`): the MVP compute path. `mxfp4.h`
  dequantises microscaling-FP4; `expert_spu.c` is the SPE GEMV kernel with
  DMA-streamed weight tiles; `expert_ppu.c` is the PPE driver that keeps one
  expert resident, fans each matmul across the SPEs via libspe2, and serves the
  P3XC protocol over TCP.
- **`cell-compat.h`**: the PS3 analogue of `power8-compat.h` — big-endian ppc64,
  classic AltiVec only (no VSX/MMA), SPE local-store budget.

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
