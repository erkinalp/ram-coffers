"""Subcluster wire frames: one activation, many experts, one partial sum.

A layer coordinator talking to a *subcluster coordinator* has a different
message shape than one talking to an expert node. All experts a token routes to
inside one subcluster receive the **same** activation, so sending an expert-node
REQ per expert would put the activation on the wire k times: at K3 width
(hidden 7168, fp32) that is 28 KB per expert, and a top-4-within-subcluster pick
would waste 84 KB of a console farm's 100 Mbit uplink per token per layer. The
batched request instead carries the activation **once** plus a compact
``(expert, gate)`` list, and the subcluster answers with a single partial sum —
which is exactly what Condor's coordinating servers did for their 22 consoles
(Barnell et al., IEEE HPEC 2012) and how ALF's host pushes one work block
descriptor list rather than one message per accelerator (ALF Programmer's Guide,
SDK 3.0).

Frames reuse the P3XC header and array payload verbatim (``protocol.encode``
with a trailer), so the format stays fixed big-endian and length-prefixed, and
an expert worker's parser is untouched — it simply never receives these types.

``BREQ`` (layer coordinator -> subcluster coordinator), ``msg_type=6``:

    <P3XC header, expert=0xFFFF, array=activation>
    n_entries   : uint16                 <= MAX_BATCH_ENTRIES
    flags       : uint16                 reserved, must be 0
    deadline_ms : uint32                 layer's budget, 0 = coordinator default
    entries     : n_entries * {
        expert      : uint16
        replica     : uint8              replica index to prefer, 0 = primary
        reserved    : uint8              must be 0
        gate        : float32            big-endian gate weight
    }

``BRSP`` (partial sum upward), ``msg_type=7``:

    <P3XC header, expert=0xFFFF, array=partial sum>
    n_reduced   : uint16                 experts actually summed
    flags       : uint16                 reserved, must be 0

``n_reduced`` lets the layer coordinator assert that the partial covers every
expert it asked for: a subcluster must never silently return a short sum.

``deadline_ms`` propagates the layer's remaining budget downward: the subcluster
coordinator bounds its own expert calls by ``min(deadline, its own timeout)`` so
a slow console cannot hold an upstream request open past the layer's deadline.

``BERR`` (structured failure upward), ``msg_type=8``:

    <P3XC header, expert=0xFFFF, array=[0.0]>
    code        : uint16                 ERR_* below
    n_failures  : uint16                 <= MAX_BATCH_ENTRIES
    failures    : n_failures * {
        expert      : uint16
        reason      : uint16             ERR_* for this expert
        node_len    : uint16             <= MAX_STRING_BYTES
        node        : node_len bytes, UTF-8 node id
    }
    detail_len  : uint16                 <= MAX_STRING_BYTES
    detail      : detail_len bytes, UTF-8

Every decoder here rejects a frame whose type, version, flags, counts or
lengths are not what it expects, and every count has an explicit bound, so a
hostile or corrupt frame cannot make a coordinator allocate without limit.
"""

from __future__ import annotations

import struct
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .protocol import (MAX_FRAME_BYTES, MSG_BERR, MSG_BREQ, MSG_BRSP,
                       ProtocolError, decode, encode)

#: No subcluster is expected to hold more than this many experts, let alone
#: have that many selected for one token; the bound exists so a bad count
#: cannot drive an unbounded allocation.
MAX_BATCH_ENTRIES = 1024

#: Bound on any string (node id, error detail) carried in a BERR frame.
MAX_STRING_BYTES = 512

#: ``expert`` field for frames that address a subcluster rather than an expert.
NO_EXPERT = 0xFFFF

# Failure codes. Deliberately coarse and stable: the layer coordinator decides
# whether to retry from the code, and the human reads ``detail``.
ERR_OK = 0
ERR_UNKNOWN = 1
ERR_UNKNOWN_EXPERT = 2       #: expert is not placed in this subcluster
ERR_NODE_UNREACHABLE = 3     #: connect failed; the request never ran
ERR_NODE_TIMEOUT = 4         #: no response in time; the expert may still run
ERR_NODE_ERROR = 5           #: worker answered ERR
ERR_NODE_DISCONNECTED = 6    #: peer closed mid-request
ERR_BAD_REQUEST = 7          #: malformed or unsupported batch frame
ERR_SHUTTING_DOWN = 8

_ENTRY = struct.Struct("!HBBf")
_COUNT = struct.Struct("!HH")
_REQ_HEAD = struct.Struct("!HHI")
_FAILURE_HEAD = struct.Struct("!HHH")

#: Bound on ``deadline_ms`` (one hour); a nonsense deadline is rejected rather
#: than silently clamped.
MAX_DEADLINE_MS = 3_600_000


class BatchEntry(NamedTuple):
    """One expert selected inside a subcluster for a token."""

    expert: int
    gate: float
    replica: int = 0


class BatchFailure(NamedTuple):
    """Why one expert in a batch could not contribute."""

    expert: int
    reason: int
    node_id: str


# -- requests ---------------------------------------------------------------
def encode_batch_request(layer: int, token_id: int, x: np.ndarray,
                         entries: Sequence[BatchEntry],
                         deadline_ms: int = 0) -> bytes:
    """Encode a BREQ: the activation once, plus the ``(expert, gate)`` list."""
    if not entries:
        raise ProtocolError("batch request needs at least one entry")
    if len(entries) > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{len(entries)} entries exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    if not 0 <= deadline_ms <= MAX_DEADLINE_MS:
        raise ProtocolError(f"deadline_ms {deadline_ms} out of range")
    trailer = [_REQ_HEAD.pack(len(entries), 0, deadline_ms)]
    for entry in entries:
        if not 0 <= entry.expert <= 0xFFFF:
            raise ProtocolError(f"expert {entry.expert} out of range")
        if not 0 <= entry.replica <= 0xFF:
            raise ProtocolError(f"replica {entry.replica} out of range")
        trailer.append(_ENTRY.pack(entry.expert, entry.replica, 0,
                                   float(entry.gate)))
    return encode(MSG_BREQ, layer, NO_EXPERT, token_id, x, b"".join(trailer))


def decode_batch_request(body: bytes) -> dict:
    """Decode a BREQ body, rejecting anything malformed or unsupported."""
    return parse_batch_request(_decode_typed(body, MSG_BREQ))


def parse_batch_request(msg: dict) -> dict:
    """Parse the BREQ trailer of an already-decoded P3XC frame."""
    _require_type(msg, MSG_BREQ)
    trailer = msg["trailer"]
    if len(trailer) < _REQ_HEAD.size:
        raise ProtocolError("batch request is missing its entry count")
    n_entries, flags, deadline_ms = _REQ_HEAD.unpack_from(trailer, 0)
    if flags != 0:
        raise ProtocolError(f"unsupported batch flags {flags:#x}")
    if n_entries == 0:
        raise ProtocolError("batch request has no entries")
    if n_entries > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{n_entries} entries exceeds {MAX_BATCH_ENTRIES}")
    expected = _REQ_HEAD.size + n_entries * _ENTRY.size
    if len(trailer) != expected:
        raise ProtocolError(f"batch trailer is {len(trailer)} bytes, "
                            f"expected {expected} for {n_entries} entries")
    entries: List[BatchEntry] = []
    seen = set()
    off = _REQ_HEAD.size
    for _ in range(n_entries):
        expert, replica, reserved, gate = _ENTRY.unpack_from(trailer, off)
        off += _ENTRY.size
        if reserved != 0:
            raise ProtocolError("reserved entry byte must be zero")
        if expert in seen:
            raise ProtocolError(f"expert {expert} appears twice in one batch")
        seen.add(expert)
        entries.append(BatchEntry(expert=expert, gate=gate, replica=replica))
    msg["entries"] = entries
    msg["deadline_ms"] = deadline_ms
    return msg


# -- responses --------------------------------------------------------------
def encode_batch_response(layer: int, token_id: int, partial: np.ndarray,
                          n_reduced: int) -> bytes:
    """Encode a BRSP carrying one subcluster's partial sum."""
    if n_reduced < 1 or n_reduced > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"n_reduced {n_reduced} out of range")
    return encode(MSG_BRSP, layer, NO_EXPERT, token_id,
                  np.ascontiguousarray(partial, dtype=np.float32),
                  _COUNT.pack(n_reduced, 0))


def decode_batch_response(body: bytes) -> dict:
    return parse_batch_response(_decode_typed(body, MSG_BRSP))


def parse_batch_response(msg: dict) -> dict:
    _require_type(msg, MSG_BRSP)
    trailer = msg["trailer"]
    if len(trailer) != _COUNT.size:
        raise ProtocolError(f"batch response trailer is {len(trailer)} bytes, "
                            f"expected {_COUNT.size}")
    n_reduced, flags = _COUNT.unpack_from(trailer, 0)
    if flags != 0:
        raise ProtocolError(f"unsupported batch flags {flags:#x}")
    if n_reduced == 0:
        raise ProtocolError("batch response reduced no experts")
    msg["n_reduced"] = n_reduced
    return msg


# -- errors -----------------------------------------------------------------
def encode_batch_error(layer: int, token_id: int, code: int,
                       failures: Sequence[BatchFailure] = (),
                       detail: str = "") -> bytes:
    """Encode a BERR naming every expert that could not contribute."""
    if len(failures) > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{len(failures)} failures exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    parts = [_COUNT.pack(code, len(failures))]
    for failure in failures:
        node = _clip(failure.node_id)
        parts.append(_FAILURE_HEAD.pack(failure.expert, failure.reason,
                                        len(node)))
        parts.append(node)
    body = _clip(detail)
    parts.append(struct.pack("!H", len(body)))
    parts.append(body)
    return encode(MSG_BERR, layer, NO_EXPERT, token_id,
                  np.zeros(1, np.float32), b"".join(parts))


def decode_batch_error(body: bytes) -> dict:
    return parse_batch_error(_decode_typed(body, MSG_BERR))


def parse_batch_error(msg: dict) -> dict:
    _require_type(msg, MSG_BERR)
    trailer = msg["trailer"]
    if len(trailer) < _COUNT.size:
        raise ProtocolError("batch error is missing its header")
    code, n_failures = _COUNT.unpack_from(trailer, 0)
    if n_failures > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{n_failures} failures exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    off = _COUNT.size
    failures: List[BatchFailure] = []
    for _ in range(n_failures):
        if len(trailer) < off + _FAILURE_HEAD.size:
            raise ProtocolError("truncated batch failure list")
        expert, reason, node_len = _FAILURE_HEAD.unpack_from(trailer, off)
        off += _FAILURE_HEAD.size
        if node_len > MAX_STRING_BYTES:
            raise ProtocolError(f"node id of {node_len} bytes exceeds "
                                f"{MAX_STRING_BYTES}")
        if len(trailer) < off + node_len:
            raise ProtocolError("truncated node id")
        node = trailer[off:off + node_len].decode("utf-8", "replace")
        off += node_len
        failures.append(BatchFailure(expert=expert, reason=reason,
                                     node_id=node))
    if len(trailer) < off + 2:
        raise ProtocolError("truncated batch error detail")
    (detail_len,) = struct.unpack_from("!H", trailer, off)
    off += 2
    if detail_len > MAX_STRING_BYTES:
        raise ProtocolError(f"detail of {detail_len} bytes exceeds "
                            f"{MAX_STRING_BYTES}")
    if len(trailer) != off + detail_len:
        raise ProtocolError("batch error length mismatch")
    msg["code"] = code
    msg["failures"] = failures
    msg["detail"] = trailer[off:off + detail_len].decode("utf-8", "replace")
    return msg


# -- shared -----------------------------------------------------------------
def decode_batch(body: bytes) -> dict:
    """Decode any subcluster frame, dispatching on ``msg_type``."""
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"refusing a {len(body)} byte frame")
    return parse_batch(decode(body))


def parse_batch(msg: dict) -> dict:
    """Parse any already-decoded subcluster frame by ``msg_type``."""
    kind = msg["msg_type"]
    if kind == MSG_BREQ:
        return parse_batch_request(msg)
    if kind == MSG_BRSP:
        return parse_batch_response(msg)
    if kind == MSG_BERR:
        return parse_batch_error(msg)
    raise ProtocolError(f"not a subcluster frame (msg_type {kind})")


def _decode_typed(body: bytes, expected: int) -> dict:
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"refusing a {len(body)} byte frame")
    msg = decode(body)                       # validates magic and version
    _require_type(msg, expected)
    return msg


def _require_type(msg: dict, expected: int) -> None:
    if msg["msg_type"] != expected:
        raise ProtocolError(f"expected msg_type {expected}, "
                            f"got {msg['msg_type']}")


def _clip(text: str) -> bytes:
    return text.encode("utf-8", "replace")[:MAX_STRING_BYTES]


def entries_for(expert_ids: Sequence[int], gate_weights: Sequence[float],
                positions: Sequence[int],
                replicas: Optional[Dict[int, int]] = None
                ) -> Tuple[List[BatchEntry], List[int]]:
    """Build the batch entries for ``positions`` of a top-k selection.

    Returns the entries in ascending position order together with the positions
    they came from, so the caller can map the reduction back to its top-k slots.
    """
    ordered = sorted(positions)
    entries = [BatchEntry(expert=int(expert_ids[p]),
                          gate=float(gate_weights[p]),
                          replica=0 if replicas is None
                          else int(replicas.get(p, 0)))
               for p in ordered]
    return entries, ordered
