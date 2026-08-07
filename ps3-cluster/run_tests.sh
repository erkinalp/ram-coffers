#!/usr/bin/env bash
# Build the host-sim worker and run the full ps3-cluster test suite.
set -euo pipefail
cd "$(dirname "$0")"
make host
python3 -m unittest discover -s tests -v
python3 tools/plan_k3.py
