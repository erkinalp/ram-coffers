#!/usr/bin/env python3
"""Print the PS3-cluster plan for a model profile (default: Kimi K3)."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster import topology as T


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--usable-mb", type=int, default=T.PS3_USABLE_RAM_MB)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    plan = T.plan_cluster(T.KIMI_K3, a.usable_mb)
    if a.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return
    print("PS3 cluster plan for Kimi K3 (1 expert x 1 layer / node)")
    print("=" * 56)
    print(f"  usable RAM/node      : {plan.usable_ram_mb} MB")
    print(f"  per-expert (MXFP4)   : {plan.per_expert_mb:.1f} MB")
    print(f"  expert nodes         : {plan.expert_nodes:,}")
    print(f"  layer-coordinator    : {plan.layer_nodes:,}")
    print(f"  embed/lm_head nodes  : {plan.io_nodes:,}")
    print(f"  TOTAL consoles       : {plan.total_nodes:,}")
    print(f"  expert split factor  : {plan.experts_split_across}")
    print(f"  active nodes / token : {plan.active_nodes_per_token:,}")
    print(f"  idle fraction / token: {plan.idle_fraction*100:.2f}%")
    for w in plan.warnings:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
