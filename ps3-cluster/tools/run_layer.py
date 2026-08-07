#!/usr/bin/env python3
"""Run one token through a P3XC subcluster hierarchy.

This is a small operational client, not a full model shim. It loads the shared
cluster config, builds the corresponding two- or three-tier dispatch plan, asks
coordinators for weighted expert contributions, and reduces them exactly (or,
with ``--fast``, approximately).

Two-tier: layer -> heads -> consoles

    python3 tools/run_layer.py --config cluster.json --layer 3 --token 1 \
        --experts 0 1 2 3 --gates 0.5 0.25 0.125 0.0625 \
        --activation input.npy --output out.npy

Three-tier: layer -> regions -> heads -> consoles

    python3 tools/run_layer.py --config cluster.json --layer 3 --token 2 \
        --experts 4 5 --gates 0.75 0.25 \
        --activation input.json --output out.json \
        --fast --retry-ambiguous

``--activation`` is either a ``.npy`` file or a JSON array. ``--output`` is
written in the same format it was read, defaulting to JSON.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ps3_cluster.deployment import ClusterConfig
from ps3_cluster.hierarchy import (HierarchicalExpertDispatcher,
                                   LinkRetryPolicy, SubclusterTransport)


def _read_activation(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    with open(path, "r", encoding="utf-8") as fh:
        return np.array(json.load(fh), dtype=np.float32)


def _write_result(path: str, arr: np.ndarray) -> None:
    if path.endswith(".npy"):
        np.save(path, arr)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(arr.tolist(), fh)
            fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="cluster config JSON")
    ap.add_argument("--layer", type=int, default=0,
                    help="layer whose experts are selected")
    ap.add_argument("--token", type=int, default=1,
                    help="token id used for correlation")
    ap.add_argument("--experts", type=int, nargs="+", required=True,
                    help="expert ids in top-k order")
    ap.add_argument("--gates", type=float, nargs="+", required=True,
                    help="gate weight for each --experts entry")
    ap.add_argument("--activation", required=True,
                    help="input vector (.npy or JSON array)")
    ap.add_argument("--output", required=True,
                    help="where to write the resulting vector")
    ap.add_argument("--fast", action="store_true",
                    help="ask each coordinator for one partial sum (not exact)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-batch ceiling, seconds")
    ap.add_argument("--retry-attempts", type=int, default=2,
                    help="tries per logical coordinator call")
    ap.add_argument("--retry-ambiguous", action="store_true",
                    help="retry a batch that may already be in flight (can "
                         "execute experts twice; reduction uses one result)")
    ap.add_argument("--no-retry-safe", action="store_true",
                    help="do not retry frames that were never sent")
    ap.add_argument("--ping", action="store_true",
                    help="PING all coordinators and exit without running")
    args = ap.parse_args()

    if len(args.experts) != len(args.gates):
        ap.error("--experts and --gates must have the same length")

    config = ClusterConfig.load(args.config)
    x = _read_activation(args.activation)
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    if config.regions:
        plan = config.tiered_plan()
        group_endpoints = config.region_endpoints()
    else:
        plan = config.plan()
        group_endpoints = config.group_endpoints()

    policy = LinkRetryPolicy(
        attempts=args.retry_attempts,
        retry_safe=not args.no_retry_safe,
        retry_ambiguous=args.retry_ambiguous)

    transport = SubclusterTransport(group_endpoints, retry_policy=policy)
    dispatcher = HierarchicalExpertDispatcher(
        config.placement(), plan, transport, fast=args.fast)

    with transport, dispatcher:
        if args.ping:
            for group_id, alive in sorted(
                    dispatcher.check_liveness(timeout=args.timeout).items()):
                print(f"{group_id}\t{'up' if alive else 'down'}")
            return 0
        out = dispatcher.run_expert_stage(
            args.layer, x, args.experts, args.gates,
            token_id=args.token, timeout=args.timeout)

    _write_result(args.output, out)
    print(f"wrote {args.output}: shape {out.shape} dtype {out.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
