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

The same frames address a *regional* coordinator: because every exact row is
tagged with the expert it came from, and an expert id is unique within a layer, a
coordinator can merge its downstream coordinators' rows and pass them up
unchanged. The tier count is therefore invisible to the format — a regional head
speaks BREQ upward and BREQ downward, which is why ``regional.py`` is a client of
``hierarchy.py`` rather than a second protocol.

``BREQ`` (caller -> coordinator), ``msg_type=6``:

    <P3XC header, expert=0xFFFF, array=activation>
    n_entries   : uint16                 <= MAX_BATCH_ENTRIES
    flags       : uint16                 REQ_FLAG_* below; other bits must be 0
    deadline_ms : uint32                 layer's budget, 0 = coordinator default
    request_id  : uint64                 only if flags & REQ_FLAG_REQUEST_ID
    entries     : n_entries * {
        expert      : uint16
        replica     : uint8              replica index to prefer, 0 = primary
        reserved    : uint8              must be 0
        gate        : float32            big-endian gate weight
    }

``BRSP`` (results upward), ``msg_type=7``, in one of two shapes:

    exact (default; flags = RSP_FLAG_PER_EXPERT):
        <P3XC header, expert=0xFFFF, array=float32 [n_reduced, <activation>]>
        n_reduced   : uint16             rows, one weighted contribution each
        flags       : uint16             RSP_FLAG_PER_EXPERT [| RSP_FLAG_REQUEST_ID]
        experts     : n_reduced * uint16 which expert each row belongs to
        request_id  : uint64             only if flags & RSP_FLAG_REQUEST_ID

    fast (only if the request set REQ_FLAG_FAST):
        <P3XC header, expert=0xFFFF, array=partial sum>
        n_reduced   : uint16             experts folded into the sum
        flags       : uint16             0 [| RSP_FLAG_REQUEST_ID]
        request_id  : uint64             only if flags & RSP_FLAG_REQUEST_ID

The **exact** shape is the default because collapsing a subcluster's experts into
one fp32 partial re-associates the additions: a token's top-k positions
interleave across subclusters, so per-subcluster partials cannot be summed back
into the flat left-to-right order. One row per expert, tagged with the expert it
came from, lets the layer accumulate strictly in top-k order with the same fp32
operations as the flat dispatcher, so the hierarchy is bit-identical to it. The
cost is upstream bandwidth: k rows instead of one. The **fast** shape trades that
identity for the smaller reply, must be requested explicitly, and can change
logits and therefore token choices.

``n_reduced`` lets the layer coordinator assert that the reply covers every
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

import hashlib
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

#: ``BREQ`` flag: answer with one partial sum instead of per-expert rows. Opt-in
#: only - it re-associates the layer's fp32 reduction (see above).
REQ_FLAG_FAST = 0x0001

#: ``BREQ`` flag: the trailer carries a 64-bit request id naming this *logical*
#: batch. A caller reuses it when it retries the same batch (to a replica
#: endpoint of the same coordinator, say), which is what lets the coordinator
#: recognise the retry and answer from its dedup cache instead of running the
#: experts twice. Optional so a phase-2 frame still parses.
REQ_FLAG_REQUEST_ID = 0x0002
REQ_FLAG_MASK = REQ_FLAG_FAST | REQ_FLAG_REQUEST_ID

#: ``BRSP`` flag: the array holds one weighted contribution per expert, tagged
#: with its expert id, rather than a single partial sum. Set on exact replies.
RSP_FLAG_PER_EXPERT = 0x0001

#: ``BRSP`` flag: the trailer echoes the request's 64-bit id, so a caller can
#: tell which attempt a reply belongs to and drop a stale one.
RSP_FLAG_REQUEST_ID = 0x0002
RSP_FLAG_MASK = RSP_FLAG_PER_EXPERT | RSP_FLAG_REQUEST_ID

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
ERR_DEDUP_CAPACITY = 9       #: dedup cache cannot accept another in-flight batch

_ENTRY = struct.Struct("!HBBf")
_COUNT = struct.Struct("!HH")
_REQ_HEAD = struct.Struct("!HHI")
_FAILURE_HEAD = struct.Struct("!HHH")
_REQUEST_ID = struct.Struct("!Q")

#: Bound on a request id; ids are opaque, but they must fit the wire field.
MAX_REQUEST_ID = 0xFFFFFFFFFFFFFFFF

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
                         deadline_ms: int = 0, fast: bool = False,
                         request_id: Optional[int] = None) -> bytes:
    """Encode a BREQ: the activation once, plus the ``(expert, gate)`` list.

    ``fast`` asks for a single partial sum instead of per-expert contributions,
    giving up bit-identity with the flat reduction. ``request_id`` names the
    logical batch so a retry of it can be recognised downstream.
    """
    if not entries:
        raise ProtocolError("batch request needs at least one entry")
    if len(entries) > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{len(entries)} entries exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    if not 0 <= deadline_ms <= MAX_DEADLINE_MS:
        raise ProtocolError(f"deadline_ms {deadline_ms} out of range")
    flags = REQ_FLAG_FAST if fast else 0
    if request_id is not None:
        if not 0 <= request_id <= MAX_REQUEST_ID:
            raise ProtocolError(f"request_id {request_id} out of range")
        flags |= REQ_FLAG_REQUEST_ID
    trailer = [_REQ_HEAD.pack(len(entries), flags, deadline_ms)]
    if request_id is not None:
        trailer.append(_REQUEST_ID.pack(request_id))
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
    if flags & ~REQ_FLAG_MASK:
        raise ProtocolError(f"unsupported batch flags {flags:#x}")
    if n_entries == 0:
        raise ProtocolError("batch request has no entries")
    if n_entries > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{n_entries} entries exceeds {MAX_BATCH_ENTRIES}")
    has_id = bool(flags & REQ_FLAG_REQUEST_ID)
    id_size = _REQUEST_ID.size if has_id else 0
    expected = _REQ_HEAD.size + id_size + n_entries * _ENTRY.size
    if len(trailer) != expected:
        raise ProtocolError(f"batch trailer is {len(trailer)} bytes, "
                            f"expected {expected} for {n_entries} entries")
    request_id: Optional[int] = None
    if has_id:
        (request_id,) = _REQUEST_ID.unpack_from(trailer, _REQ_HEAD.size)
    entries: List[BatchEntry] = []
    seen = set()
    off = _REQ_HEAD.size + id_size
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
    msg["fast"] = bool(flags & REQ_FLAG_FAST)
    msg["request_id"] = request_id
    return msg


# -- responses --------------------------------------------------------------
def encode_batch_response(layer: int, token_id: int, partial: np.ndarray,
                          n_reduced: int,
                          request_id: Optional[int] = None) -> bytes:
    """Encode a *fast* BRSP carrying one subcluster's partial sum."""
    if n_reduced < 1 or n_reduced > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"n_reduced {n_reduced} out of range")
    flags, echo = _echo(request_id)
    return encode(MSG_BRSP, layer, NO_EXPERT, token_id,
                  np.ascontiguousarray(partial, dtype=np.float32),
                  _COUNT.pack(n_reduced, flags) + echo)


def encode_batch_contributions(layer: int, token_id: int,
                               contributions: Sequence[np.ndarray],
                               experts: Sequence[int],
                               request_id: Optional[int] = None) -> bytes:
    """Encode an *exact* BRSP: one weighted contribution per expert.

    ``contributions[i]`` is ``gate_i * expert_i(x)`` exactly as the flat
    dispatcher computes it, and ``experts[i]`` says which expert it belongs to,
    so the layer can put it back at its own top-k position.
    """
    if len(contributions) != len(experts):
        raise ProtocolError("contribution/expert length mismatch")
    if not contributions:
        raise ProtocolError("batch response covers no experts")
    if len(contributions) > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"{len(contributions)} contributions exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    rows = np.stack([np.ascontiguousarray(c, dtype=np.float32)
                     for c in contributions])
    flags, echo = _echo(request_id)
    tags = [_COUNT.pack(len(contributions), RSP_FLAG_PER_EXPERT | flags)]
    for expert in experts:
        if not 0 <= expert <= 0xFFFF:
            raise ProtocolError(f"expert {expert} out of range")
        tags.append(struct.pack("!H", expert))
    tags.append(echo)
    return encode(MSG_BRSP, layer, NO_EXPERT, token_id, rows, b"".join(tags))


def _echo(request_id: Optional[int]) -> Tuple[int, bytes]:
    """Response flag and trailer bytes echoing ``request_id``, if any."""
    if request_id is None:
        return 0, b""
    if not 0 <= request_id <= MAX_REQUEST_ID:
        raise ProtocolError(f"request_id {request_id} out of range")
    return RSP_FLAG_REQUEST_ID, _REQUEST_ID.pack(request_id)


def decode_batch_response(body: bytes) -> dict:
    return parse_batch_response(_decode_typed(body, MSG_BRSP))


def parse_batch_response(msg: dict) -> dict:
    """Parse either BRSP shape, exposing ``per_expert`` and ``experts``."""
    _require_type(msg, MSG_BRSP)
    trailer = msg["trailer"]
    if len(trailer) < _COUNT.size:
        raise ProtocolError("batch response is missing its header")
    n_reduced, flags = _COUNT.unpack_from(trailer, 0)
    if flags & ~RSP_FLAG_MASK:
        raise ProtocolError(f"unsupported batch flags {flags:#x}")
    if n_reduced == 0:
        raise ProtocolError("batch response reduced no experts")
    if n_reduced > MAX_BATCH_ENTRIES:
        raise ProtocolError(f"n_reduced {n_reduced} exceeds "
                            f"{MAX_BATCH_ENTRIES}")
    per_expert = bool(flags & RSP_FLAG_PER_EXPERT)
    id_size = _REQUEST_ID.size if flags & RSP_FLAG_REQUEST_ID else 0
    experts: List[int] = []
    expected = _COUNT.size + (2 * n_reduced if per_expert else 0) + id_size
    if len(trailer) != expected:
        raise ProtocolError(f"batch response trailer is {len(trailer)} bytes, "
                            f"expected {expected} for {n_reduced} "
                            f"contributions")
    if per_expert:
        experts = list(struct.unpack_from("!%dH" % n_reduced, trailer,
                                          _COUNT.size))
        if len(set(experts)) != len(experts):
            raise ProtocolError("an expert is tagged twice in one response")
        rows = msg["array"]
        if rows.ndim < 2 or rows.shape[0] != n_reduced:
            raise ProtocolError(f"batch response array {rows.shape} does not "
                                f"hold {n_reduced} contributions")
    request_id: Optional[int] = None
    if id_size:
        (request_id,) = _REQUEST_ID.unpack_from(trailer, len(trailer) - id_size)
    msg["n_reduced"] = n_reduced
    msg["per_expert"] = per_expert
    msg["experts"] = experts
    msg["request_id"] = request_id
    return msg


# -- errors -----------------------------------------------------------------
def encode_batch_error(layer: int, token_id: int, code: int,
                       failures: Sequence[BatchFailure] = (),
                       detail: str = "",
                       request_id: Optional[int] = None) -> bytes:
    """Encode a BERR naming every expert that could not contribute.

    ``request_id``, when given, is appended as the same big-endian uint64 a BRSP
    echoes, so a caller can tell a failure of *its* batch from a late failure of
    an abandoned attempt. A frame from a peer that does not echo ids simply ends
    after its detail string.
    """
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
    if request_id is not None:
        if not 0 <= request_id <= MAX_REQUEST_ID:
            raise ProtocolError(f"request_id {request_id} out of range")
        parts.append(_REQUEST_ID.pack(request_id))
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
    if len(trailer) < off + detail_len:
        raise ProtocolError("batch error length mismatch")
    detail = trailer[off:off + detail_len].decode("utf-8", "replace")
    off += detail_len
    rest = len(trailer) - off
    request_id: Optional[int] = None
    if rest == _REQUEST_ID.size:
        (request_id,) = _REQUEST_ID.unpack_from(trailer, off)
    elif rest:
        raise ProtocolError(f"batch error has {rest} trailing bytes, "
                            f"expected 0 or {_REQUEST_ID.size}")
    msg["code"] = code
    msg["failures"] = failures
    msg["detail"] = detail
    msg["request_id"] = request_id
    return msg


# -- shared -----------------------------------------------------------------
def batch_fingerprint(msg: dict) -> bytes:
    """Stable 256-bit fingerprint of a decoded BREQ, ignoring ``request_id``.

    The request id is a retry handle, not part of the logical batch: the same
    activation, token, layer, deadline, entries and fast flag must hash to the
    same fingerprint even if the id is reused by the caller.
    """
    h = hashlib.sha256()
    h.update(struct.pack("!IIII", msg["layer"], msg["token_id"],
                         int(msg.get("fast", False)),
                         msg.get("deadline_ms", 0)))
    arr = msg["array"]
    h.update(struct.pack("!BB", arr.dtype.itemsize, arr.ndim))
    h.update(struct.pack(f"!{arr.ndim}I", *arr.shape))
    h.update(arr.tobytes())
    entries = msg["entries"]
    h.update(struct.pack("!I", len(entries)))
    for entry in entries:
        h.update(struct.pack("!HBBf", entry.expert, entry.replica, 0,
                             float(entry.gate)))
    return h.digest()


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
