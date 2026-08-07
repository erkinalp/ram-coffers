#!/usr/bin/env python3
"""Run one subcluster coordinator (head server) from a cluster config.

    # on the head server that fronts sc-0000's 22 consoles
    python3 tools/run_subcluster.py --config cluster.json --subcluster sc-0000

    # an independent standby process for the same subcluster, for failover
    python3 tools/run_subcluster.py --config cluster.json \
        --subcluster sc-0000 --standby 0

    # what does this config contain?
    python3 tools/run_subcluster.py --config cluster.json --list

    # are my consoles up? (PING/PONG each one, then exit)
    python3 tools/run_subcluster.py --config cluster.json \
        --subcluster sc-0000 --check-members

The process listens for P3XC batch frames from the layer coordinator, keeps
persistent pooled connections to its own consoles, and answers each batch with
one weighted contribution per expert, so the layer's reduction stays
bit-identical to the flat dispatcher. A layer may set the request's fast flag to
get a single partial sum instead; ``--refuse-fast`` rejects such requests here.
Ctrl-C drains in-flight batches and closes every socket.
"""
import argparse
import json
import os
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster.coordinator import SubclusterCoordinator, SubclusterService
from ps3_cluster.dedup import (DEFAULT_DEDUP_BYTES, DEFAULT_DEDUP_ENTRIES,
                               DEFAULT_DEDUP_TTL, DedupCache)
from ps3_cluster.deployment import ClusterConfig
from ps3_cluster.dispatch import RetryPolicy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="cluster config JSON (see ps3_cluster/deployment.py)")
    ap.add_argument("--subcluster", help="subcluster id to serve")
    ap.add_argument("--host", help="override the configured listen host")
    ap.add_argument("--port", type=int, help="override the configured port")
    ap.add_argument("--standby", type=int,
                    help="listen on configured standby address N instead of "
                         "the subcluster's primary")
    ap.add_argument("--dedup-entries", type=int,
                    default=DEFAULT_DEDUP_ENTRIES,
                    help="max logical batches tracked (in flight or cached)")
    ap.add_argument("--dedup-ttl", type=float, default=DEFAULT_DEDUP_TTL,
                    help="seconds a completed answer stays replayable")
    ap.add_argument("--dedup-bytes", type=int, default=DEFAULT_DEDUP_BYTES,
                    help="total completed-response byte budget")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="ceiling on one downstream expert call, seconds")
    ap.add_argument("--attempts", type=int, default=2,
                    help="tries per expert; a console that never accepted the "
                         "frame is retried")
    ap.add_argument("--retry-on-timeout", action="store_true",
                    help="retry a call whose deadline expired (may execute "
                         "the expert twice; first response used)")
    ap.add_argument("--retry-on-disconnect", action="store_true",
                    help="retry if the connection drops mid-request (may "
                         "execute the expert twice; first response used)")
    ap.add_argument("--retry-on-node-error", action="store_true",
                    help="retry an explicit ERR answer from a console")
    ap.add_argument("--no-replicas", action="store_true",
                    help="do not fail over to replica endpoints")
    ap.add_argument("--refuse-fast", action="store_true",
                    help="reject fast (single partial sum) batches, which "
                         "re-associate the layer's fp32 reduction and can "
                         "change token choices")
    ap.add_argument("--list", action="store_true",
                    help="print the config's subclusters and exit")
    ap.add_argument("--check-members", action="store_true",
                    help="PING every console in the subcluster and exit")
    args = ap.parse_args()

    config = ClusterConfig.load(args.config)
    if args.list:
        for spec in config.subclusters:
            print(f"{spec.group_id}\t{spec.endpoint[0]}:{spec.endpoint[1]}\t"
                  f"{len(spec.members)} consoles")
        return 0
    if not args.subcluster:
        ap.error("--subcluster is required unless --list is given")
    spec = config.subcluster(args.subcluster)

    coordinator = SubclusterCoordinator(
        spec.group_id, config.placement(spec.group_id),
        config.expert_endpoints(spec.group_id), timeout=args.timeout,
        retry_policy=RetryPolicy(
            attempts=args.attempts,
            retry_on_node_error=args.retry_on_node_error,
            retry_on_timeout=args.retry_on_timeout,
            retry_on_disconnect=args.retry_on_disconnect,
            use_replicas=not args.no_replicas),
        allow_fast=not args.refuse_fast,
        dedup=DedupCache(max_entries=args.dedup_entries, ttl=args.dedup_ttl,
                         max_bytes=args.dedup_bytes))

    if args.check_members:
        try:
            print(json.dumps(coordinator.check_members(), indent=2,
                             sort_keys=True))
        finally:
            coordinator.close()
        return 0

    listen = spec.endpoint
    if args.standby is not None:
        if not 0 <= args.standby < len(spec.standby):
            ap.error(f"{spec.group_id} has {len(spec.standby)} standby "
                     f"addresses")
        listen = spec.standby[args.standby]
    host_arg = args.host or listen[0]
    port_arg = listen[1] if args.port is None else args.port
    service = SubclusterService(coordinator, host=host_arg,
                                port=port_arg).start()
    host, port = service.address
    print(f"{spec.group_id} listening on {host}:{port} for "
          f"{len(spec.members)} consoles", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        service.stop()
        print(f"{spec.group_id} stopped after "
              f"{coordinator.batches_served} batches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
