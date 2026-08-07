#!/usr/bin/env python3
"""Run one regional coordinator (head of heads) from a cluster config.

    # on the server that fronts rg-0000's subcluster heads
    python3 tools/run_region.py --config cluster.json --region rg-0000

    # a standby for the same region, on its second configured address
    python3 tools/run_region.py --config cluster.json --region rg-0000 \
        --standby 0

    # what regions does this config declare?
    python3 tools/run_region.py --config cluster.json --list

    # are my head servers up? (PING/PONG each one, then exit)
    python3 tools/run_region.py --config cluster.json --region rg-0000 \
        --check-members

The process speaks the same P3XC batch protocol on both sides: it accepts one
batch per token from the layer coordinator, splits it across the subcluster heads
it fronts, and answers with one weighted contribution per expert, tagged, so the
layer's fp32 reduction stays bit-identical to the flat dispatcher however deep
the tree is. ``--refuse-fast`` rejects the approximate partial-sum mode.

A region reached through a standby address may re-execute a batch its primary had
already started downstream (at-least-once execution); the layer still reduces
exactly one answer. ``--dedup-entries``/``--dedup-ttl`` bound the cache that lets
a retry landing on the *same* process replay the first answer instead.
Ctrl-C drains in-flight batches and closes every socket.
"""
import argparse
import json
import os
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.coordinator import CoordinatorService
from ps3_cluster.dedup import (DEFAULT_DEDUP_ENTRIES, DEFAULT_DEDUP_TTL,
                               DedupCache)
from ps3_cluster.deployment import ClusterConfig
from ps3_cluster.hierarchy import LinkRetryPolicy
from ps3_cluster.regional import RegionalCoordinator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="cluster config JSON (see ps3_cluster/deployment.py)")
    ap.add_argument("--region", help="region id to serve")
    ap.add_argument("--host", help="override the configured listen host")
    ap.add_argument("--port", type=int, help="override the configured port")
    ap.add_argument("--standby", type=int,
                    help="listen on configured standby address N instead of "
                         "the region's primary")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="ceiling on one downstream subcluster batch, seconds")
    ap.add_argument("--attempts", type=int, default=2,
                    help="tries per downstream batch; a head server that never "
                         "accepted the frame is retried, one that may have run "
                         "it is not unless --retry-ambiguous")
    ap.add_argument("--retry-ambiguous", action="store_true",
                    help="also retry a batch that timed out or lost its "
                         "connection, which is at-least-once execution "
                         "downstream (the layer still reduces one answer)")
    ap.add_argument("--dedup-entries", type=int, default=DEFAULT_DEDUP_ENTRIES,
                    help="bound on remembered request ids")
    ap.add_argument("--dedup-ttl", type=float, default=DEFAULT_DEDUP_TTL,
                    help="seconds a remembered answer stays replayable")
    ap.add_argument("--refuse-fast", action="store_true",
                    help="reject fast (partial sum) batches, which "
                         "re-associate the layer's fp32 reduction and can "
                         "change token choices")
    ap.add_argument("--list", action="store_true",
                    help="print the config's regions and exit")
    ap.add_argument("--check-members", action="store_true",
                    help="PING every subcluster head in the region and exit")
    args = ap.parse_args()

    config = ClusterConfig.load(args.config)
    if args.list:
        for spec in config.regions:
            print(f"{spec.region_id}\t{spec.endpoint[0]}:{spec.endpoint[1]}\t"
                  f"{len(spec.subclusters)} subclusters\t"
                  f"{len(config.region_members(spec.region_id))} consoles")
        return 0
    if not config.regions:
        ap.error(f"{args.config} declares no regions (two-tier config)")
    if not args.region:
        ap.error("--region is required unless --list is given")
    spec = config.region(args.region)

    coordinator = RegionalCoordinator(
        spec.region_id, config.placement(), config.plan(spec.region_id),
        config.group_endpoints(spec.region_id), timeout=args.timeout,
        retry_policy=LinkRetryPolicy(attempts=args.attempts,
                                     retry_ambiguous=args.retry_ambiguous),
        allow_fast=not args.refuse_fast,
        dedup=DedupCache(max_entries=args.dedup_entries, ttl=args.dedup_ttl))

    if args.check_members:
        try:
            print(json.dumps(coordinator.check_members(), indent=2,
                             sort_keys=True))
        finally:
            coordinator.close()
        return 0

    listen = spec.endpoint
    if args.standby is not None:
        try:
            listen = spec.standby[args.standby]
        except IndexError:
            ap.error(f"{spec.region_id} has {len(spec.standby)} standby "
                     f"addresses, no index {args.standby}")
    host_arg = args.host or listen[0]
    port_arg = listen[1] if args.port is None else args.port
    service = CoordinatorService(coordinator, host=host_arg,
                                 port=port_arg).start()
    host, port = service.address
    print(f"{spec.region_id} listening on {host}:{port} for "
          f"{len(spec.subclusters)} subclusters", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        service.stop()
        print(f"{spec.region_id} stopped after "
              f"{coordinator.batches_served} batches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
