"""Wire protocol for the PS3 expert cluster.

The coordinator sends an expert an input activation and gets back the expert's
output activation. Both sides may be a big-endian PS3 (Cell/OtherOS, ppc64 BE)
or a little-endian x86 host, so the wire format is fixed to **network byte order
(big-endian)** and every frame is length-prefixed. Cell is natively big-endian,
so on a PS3 these conversions are no-ops; the cost is paid only on x86.

Frame layout (all integers big-endian / ``!``):

    magic     : 4 bytes  b"P3XC"        (PS3 eXpert Cluster)
    version   : uint8
    msg_type  : uint8    REQ=1 / RSP=2 / ERR=3 / PING=4 / PONG=5
    layer     : uint16
    expert    : uint16
    token_id  : uint32   opaque routing/debug id
    dtype     : uint8    see DTYPE_*
    ndim      : uint8
    shape     : ndim * uint32
    payload   : product(shape) * itemsize bytes, big-endian elements
    trailer   : optional, frame-type-specific bytes to the end of the body

Expert-worker frames (REQ/RSP/ERR/PING/PONG) never carry a trailer, so an
expert node's parser is unchanged. The subcluster frames in ``batch.py`` use
the trailer to carry their ``(expert, gate)`` list after the single shared
activation; anything that only needs the header and array can decode them with
this module and ignore the trailer.

This module has no third-party dependencies beyond numpy so it can run on a
stock OtherOS Python as well as on the coordinator.
"""

from __future__ import annotations

import struct
import numpy as np

MAGIC = b"P3XC"
VERSION = 1

MSG_REQ = 1
MSG_RSP = 2
MSG_ERR = 3
MSG_PING = 4
MSG_PONG = 5
#: Subcluster frames (see ``batch.py``): batched request, partial-sum response,
#: structured error. Only spoken between a layer coordinator and a subcluster
#: coordinator; expert workers never see them.
MSG_BREQ = 6
MSG_BRSP = 7
MSG_BERR = 8

#: Refuse to allocate for a frame larger than this (matches the C worker's
#: limit in ``ppu/expert_ppu.c``), so a bad length prefix cannot exhaust a
#: console's 256 MB of XDR RAM.
MAX_FRAME_BYTES = 1 << 26

# Compact dtype tags. MXFP4 payloads travel packed (uint8) and are expanded on
# the node, exactly as AirLLM #316 keeps MXFP4 packed across PCIe; here the
# "bus" is gigabit ethernet.
DTYPE_F32 = 1
DTYPE_F16 = 2
DTYPE_BF16 = 3
DTYPE_U8 = 4  # packed MXFP4 / quantised payloads

_DTYPE_TO_NP = {
    DTYPE_F32: np.dtype(">f4"),
    DTYPE_F16: np.dtype(">f2"),
    DTYPE_U8: np.dtype(">u1"),
}
_NP_KIND_TO_DTYPE = {
    ("f", 4): DTYPE_F32,
    ("f", 2): DTYPE_F16,
    ("u", 1): DTYPE_U8,
}

_HEADER = struct.Struct("!4sBBHHIBB")  # up to shape[]


class ProtocolError(Exception):
    pass


def _dtype_tag(arr: np.ndarray) -> int:
    key = (arr.dtype.kind, arr.dtype.itemsize)
    if key not in _NP_KIND_TO_DTYPE:
        raise ProtocolError(f"unsupported dtype {arr.dtype!r}")
    return _NP_KIND_TO_DTYPE[key]


def encode(msg_type: int, layer: int, expert: int, token_id: int,
           arr: np.ndarray, trailer: bytes = b"") -> bytes:
    """Serialise one frame to length-prefixed big-endian bytes.

    ``trailer`` is appended verbatim after the array payload; it is empty for
    every expert-worker frame type.
    """
    arr = np.ascontiguousarray(arr)
    dtype_tag = _dtype_tag(arr)
    # Force big-endian element order on the wire regardless of host endianness.
    be = arr.astype(arr.dtype.newbyteorder(">"), copy=False)
    shape = be.shape
    if len(shape) > 255:
        raise ProtocolError("too many dimensions")
    header = _HEADER.pack(MAGIC, VERSION, msg_type, layer, expert, token_id,
                          dtype_tag, len(shape))
    shape_bytes = struct.pack("!%dI" % len(shape), *shape)
    payload = be.tobytes(order="C")
    body = header + shape_bytes + payload + trailer
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {len(body)} bytes exceeds the "
                            f"{MAX_FRAME_BYTES} byte limit")
    return struct.pack("!I", len(body)) + body


def decode(body: bytes) -> dict:
    """Parse one frame body (without the 4-byte length prefix)."""
    if len(body) < _HEADER.size:
        raise ProtocolError("short frame")
    (magic, version, msg_type, layer, expert, token_id,
     dtype_tag, ndim) = _HEADER.unpack_from(body, 0)
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != VERSION:
        raise ProtocolError(f"version mismatch {version}")
    off = _HEADER.size
    if len(body) < off + 4 * ndim:
        raise ProtocolError("truncated shape")
    shape = struct.unpack_from("!%dI" % ndim, body, off)
    off += 4 * ndim
    if dtype_tag not in _DTYPE_TO_NP:
        raise ProtocolError(f"unsupported dtype tag {dtype_tag}")
    np_dtype = _DTYPE_TO_NP[dtype_tag]
    count = 1
    for s in shape:
        count *= s
    end = off + count * np_dtype.itemsize
    if len(body) < end:
        raise ProtocolError("truncated payload")
    arr = np.frombuffer(body[off:end], dtype=np_dtype).reshape(shape)
    # Return a native-endian, writable copy so callers can compute on it.
    arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("=")))
    return {
        "msg_type": msg_type,
        "layer": layer,
        "expert": expert,
        "token_id": token_id,
        "array": arr,
        "trailer": body[end:],
    }


def read_frame(recv) -> dict:
    """Read one framed message using a ``recv(n) -> bytes`` callable."""
    raw_len = _recv_exact(recv, 4)
    (length,) = struct.unpack("!I", raw_len)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise ProtocolError(f"refusing a {length} byte frame")
    body = _recv_exact(recv, length)
    return decode(body)


def _recv_exact(recv, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = recv(n - got)
        if not chunk:
            raise ProtocolError("connection closed mid-frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)
