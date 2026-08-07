#!/usr/bin/env bash
# Build the host-sim worker and run the full ps3-cluster test suite.
set -euo pipefail
cd "$(dirname "$0")"
make host
# -W error::ResourceWarning: the suite opens real sockets, subprocesses and
# reader threads, so a leaked fd is a bug and must fail rather than warn.
python3 -W error::ResourceWarning -m unittest discover -s tests -v
python3 tools/plan_k3.py
