#!/usr/bin/env python3
"""Run the reference (numpy) expert worker for one console.

The deployable console process is ``build/expert_node_host expert.exp <port>``
(the C worker, which is what a real PS3 runs). This is its Python counterpart:
same P3XC protocol and lifecycle, numpy instead of SPE/RSX kernels, so a farm
can be brought up on one host without the toolchain.

    python3 tools/pack_expert.py /tmp/L003-E0000.exp --layer 3 --expert 0
    python3 tools/run_expert.py /tmp/L003-E0000.exp --port 9400

Use ``--identity`` to serve a cheap identity expert instead of a packed file,
which is enough to exercise the hierarchy itself.
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import sys
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pack_expert import BYTES_PER_BLOCK, BLOCK, reference_forward  # noqa: E402
from ps3_cluster.node import ExpertNode, serve  # noqa: E402

_HEADER = struct.Struct("<4sIIHH")


def load_exp(path: str):
    """Read an ``.exp`` file: ``(layer, expert, hidden, inter, forward_fn)``."""
    with open(path, "rb") as handle:
        blob = handle.read()
    magic, hidden, inter, layer, expert = _HEADER.unpack_from(blob)
    if magic != b"EXP0":
        raise ValueError(f"{path}: not an .exp file")
    off = _HEADER.size
    row = (hidden // BLOCK) * BYTES_PER_BLOCK
    gate_p = blob[off:off + inter * row]
    off += inter * row
    up_p = blob[off:off + inter * row]
    off += inter * row
    down_row = (inter // BLOCK) * BYTES_PER_BLOCK
    down_p = blob[off:off + hidden * down_row]

    def forward(x: np.ndarray) -> np.ndarray:
        return reference_forward(gate_p, up_p, down_p, hidden, inter, x)

    return layer, expert, hidden, inter, forward


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("expert_file", nargs="?",
                    help=".exp file written by tools/pack_expert.py")
    ap.add_argument("--identity", action="store_true",
                    help="serve an identity expert instead of a packed file")
    ap.add_argument("--layer", type=int, default=0, help="with --identity")
    ap.add_argument("--expert", type=int, default=0, help="with --identity")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=3830)
    args = ap.parse_args(argv)

    if args.identity:
        layer, expert = args.layer, args.expert

        def forward(x: np.ndarray) -> np.ndarray:
            return x.astype(np.float32)

        detail = "identity"
    elif args.expert_file:
        layer, expert, hidden, inter, forward = load_exp(args.expert_file)
        detail = f"hidden={hidden} inter={inter}"
    else:
        ap.error("give an .exp file or --identity")

    server = serve(args.host, args.port, ExpertNode(layer, expert, forward))
    host, port = server.server_address[:2]
    print(f"layer {layer} expert {expert} ({detail}) listening on {host}:{port}",
          flush=True)
    # SIGTERM is how a console is stopped by an init system or a farm script, so
    # it has to leave the accept loop the same way Ctrl-C does.
    signal.signal(signal.SIGTERM,
                  lambda *_: threading.Thread(target=server.shutdown,
                                              daemon=True).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    print(f"layer {layer} expert {expert} stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
