#!/usr/bin/env python3
"""Generate a subcluster cluster config for one layer.

    # 44 experts of layer 3 as two Condor-sized subclusters of 22
    python3 tools/gen_cluster_config.py --layer 3 --experts 44 \
        --expert-host 10.0.0.10 --head-host 10.0.1.1 -o cluster.json

Console ids follow the canonical ``ps3-L<layer>-E<expert>`` placement, and
consoles are numbered off ``--expert-port-base`` on a single host by default so
the file is usable on a laptop; a real farm edits the hosts or generates it from
its own inventory.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.deployment import ClusterConfig
from ps3_cluster.subcluster import DEFAULT_SUBCLUSTER_SIZE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--size", type=int, default=DEFAULT_SUBCLUSTER_SIZE,
                    help="consoles per subcluster (Condor used 22)")
    ap.add_argument("--expert-host", default="127.0.0.1")
    ap.add_argument("--expert-port-base", type=int, default=9000)
    ap.add_argument("--head-host", default="0.0.0.0")
    ap.add_argument("--head-port-base", type=int, default=8100)
    ap.add_argument("-o", "--output", help="write JSON here instead of stdout")
    args = ap.parse_args()

    consoles = [(expert, f"ps3-L{args.layer:03d}-E{expert:04d}",
                 (args.expert_host, args.expert_port_base + expert))
                for expert in range(args.experts)]
    config = ClusterConfig.for_layer(args.layer, consoles, size=args.size,
                                     head_host=args.head_host,
                                     head_port_base=args.head_port_base)
    if args.output:
        config.save(args.output)
        print(f"wrote {args.output}: {len(config.subclusters)} subclusters, "
              f"{len(config.members())} consoles")
    else:
        json.dump(config.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
