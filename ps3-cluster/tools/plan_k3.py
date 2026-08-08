#!/usr/bin/env python3
"""Print the PS3-cluster plan for a model profile (default: Kimi K3)."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster import topology as T


PROFILES = {
    "kimi-k3": T.KIMI_K3,
    "kimi-k3-0.40b": T.KIMI_K3_040B,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="kimi-k3",
                    help="model profile to plan")
    ap.add_argument("--usable-mb", type=int, default=T.PS3_USABLE_RAM_MB)
    ap.add_argument("--rsx", action="store_true",
                    help="model a GameOS-exploit boot with full-speed RSX GDDR3 as a hot tier")
    ap.add_argument("--experts-per-node", type=int, default=1,
                    help="pack N experts per console (default 1 = the canonical design)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    model = PROFILES[a.profile]
    plan = T.plan_cluster(model, a.usable_mb, rsx=a.rsx,
                          experts_per_node=a.experts_per_node)
    if a.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return
    boot = "GameOS-exploit + RSX" if plan.rsx else "OtherOS (XDR only)"
    print(f"PS3 cluster plan for {model.name} ({plan.experts_per_node} expert x 1 layer / node)")
    print(f"  boot path            : {boot}")
    print("=" * 56)
    print(f"  hot RAM/node         : {plan.node_capacity_mb:.0f} MB")
    print(f"  moe layers           : {plan.moe_layers}")
    print(f"  per-expert (MXFP4)   : {plan.per_expert_mb:.1f} MB")
    print(f"  experts fit / node   : {plan.capacity_experts_per_node}")
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
