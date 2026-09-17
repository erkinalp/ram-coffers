"""Block-packed weight formats, in numpy.

The planner names a dozen quantised forms (:data:`gen9_cluster.model.QUANT_SPECS`)
and this module is the runtime half of that contract: the decoders that turn
the packed bytes a checkpoint ships into the fp32 matrices the reference
runner multiplies. It exists for the same three jobs as :mod:`gen9_cluster.fp8`
— loading on a console whose compiled kernel is unavailable, checking a
compiled kernel against something independent, and serving honestly (slowly)
without one.

Two layout families are covered:

* **GGUF (ggml) superblocks.** q8_0, q2_k through q6_k, mxfp4 and the fp4
  packing used by NVFP4/``fp4-e4m3-16`` checkpoints. Each decodes a row of
  ``block_bytes`` per ``block_elems`` values exactly as ``ggml-quants.c`` (and
  the vectorised ``gguf-py`` port) does — same field order, same nibble and
  scale unpacking — so a shard repacked to GGUF bytes needs no per-console
  format code beyond this file.
* **Dense element formats.** fp16 and bf16, which carry no scales at all and
  decode to fp32 elementwise. bf16 has no numpy dtype; it is viewed as
  ``<u2`` and re-assembled at fp32 width by shifting into the exponent.

Everything here decodes to fp32 and returns real matrices. The packed bytes
stay packed in the coffer — decoding happens per use and is never cached, so
a fleet running these loaders is visibly slower than one on the compiled
kernels, by design.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

QK_K = 256

#: format name -> (elements per block, bytes per block). The names are the
#: ``QuantSpec.dtype`` values the planner emits, so a deployment config can
#: name its format with the same string the plan used to size it.
BLOCK_SIZES: Dict[str, Tuple[int, int]] = {
    "q8_0": (32, 34),
    "q2_k": (QK_K, 84),
    "q3_k": (QK_K, 110),
    "q4_k": (QK_K, 144),
    "q5_k": (QK_K, 176),
    "q6_k": (QK_K, 210),
    "mxfp4": (32, 17),
    "fp4": (64, 36),
}

#: ``fp4`` packs identically under every name the planner gives it: E2M1
#: nibbles with one E4M3 sub-scale per 16 elements, four sub-blocks per
#: 64-element block — the GGUF NVFP4 layout. ``fp4-e4m3-16`` and ``nvfp4``
#: are names for that same packing; the tensor-level scale NVFP4 also ships
#: is a single fp32 per tensor and is folded into the checkpoint, not the
#: blocks.
_FORMAT_ALIASES = {"fp4-e4m3-16": "fp4", "nvfp4": "fp4"}


def canonical_format(name: str) -> str:
    """The decoder key for a format name, folding planner aliases."""
    return _FORMAT_ALIASES.get(name, name)


def block_bytes_per_row(width: int, name: str) -> int:
    """Packed byte length of one ``width``-element row in ``name`` format."""
    fmt = canonical_format(name)
    elems, size = BLOCK_SIZES[fmt]
    if width % elems:
        raise ValueError(f"a {width}-element row cannot be cut into whole "
                         f"{fmt} blocks of {elems}")
    return width // elems * size


def dequantize_blocks(blocks: np.ndarray, name: str) -> np.ndarray:
    """Decode ``(n, block_bytes)`` packed blocks to ``(n, block_elems)`` fp32."""
    fmt = canonical_format(name)
    try:
        decoder = _DECODERS[fmt]
    except KeyError:
        raise ValueError(f"no decoder for weight format {name!r}") from None
    elems, size = BLOCK_SIZES[fmt]
    array = np.asarray(blocks, dtype=np.uint8)
    if array.ndim != 2 or array.shape[1] != size:
        raise ValueError(f"{fmt} blocks are {size} bytes each; got shape "
                         f"{array.shape}")
    return decoder(array).astype(np.float32, copy=False)


def dequantize_rows(buf: np.ndarray, name: str) -> np.ndarray:
    """Decode ``(rows, row_bytes)`` packed rows to ``(rows, width)`` fp32.

    ``width`` is derived from the block packing — a row of
    ``k * block_bytes`` decodes to ``k * block_elems``.
    """
    fmt = canonical_format(name)
    elems, size = BLOCK_SIZES[fmt]
    array = np.asarray(buf, dtype=np.uint8)
    if array.ndim != 2 or array.shape[1] % size:
        raise ValueError(f"{fmt} rows pack to a multiple of {size} bytes "
                         f"each; got shape {array.shape}")
    decoded = dequantize_blocks(
        array.reshape(array.shape[0] * (array.shape[1] // size), size), fmt)
    return decoded.reshape(array.shape[0],
                           array.shape[1] // size * elems)


def decode_dense(array: np.ndarray, dtype: str) -> np.ndarray:
    """One fp32 copy of a dense fp32/fp16/bf16 buffer.

    bf16 arrives as ``<u2`` storage (no numpy dtype); shifting each value
    left 16 bits places it exactly in the fp32 exponent field, which is what
    bf16 is: fp32 with its low mantissa bits dropped.
    """
    raw = np.asarray(array)
    if dtype == "bf16":
        u2 = raw.astype(np.uint16, copy=False).astype(np.uint32) << 16
        return u2.reshape(-1).view(np.float32).reshape(raw.shape)
    if dtype == "fp16":
        return raw.astype(np.float32)
    return raw.astype(np.float32, copy=False)


# -- GGUF superblock decoders ------------------------------------------------
#
# Each is a straight port of the matching ``dequantize_row_*`` in
# ``ggml-quants.c``, shaped like the ``gguf-py`` vectorisations. ``blocks`` is
# ``(n_blocks, block_bytes)`` uint8; the result is ``(n_blocks, block_elems)``
# float32.

def _q8_0(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d, x = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32).reshape(n, 1)
    x = x.view(np.int8).astype(np.float32)
    return x * d


def _q2_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    scales, rest = np.hsplit(blocks, [QK_K // 16])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    d, dmin = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    dl = (d * (scales & 0xF).astype(np.float32)).reshape(n, QK_K // 16, 1)
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape(n, QK_K // 16, 1)

    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qs = (qs.reshape(n, -1, 1, 32) >> shift) & np.uint8(3)
    qs = qs.reshape(n, QK_K // 16, 16).astype(np.float32)
    return (dl * qs - ml).reshape(n, QK_K)


def _q3_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    hmask, rest = np.hsplit(blocks, [QK_K // 8])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)

    # 16 scales at 6 bits: low nibbles from bytes 0-7 (scale i<8 from byte i,
    # i>=8 from byte i-8's high nibble), top two bits from bytes 8-11.
    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape(n, 1, 8) >> np.array(
        [0, 4], dtype=np.uint8).reshape(1, 2, 1)
    lscales = lscales.reshape(n, 16)
    hscales = hscales.reshape(n, 1, 4) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)
    hscales = hscales.reshape(n, 16)
    unpacked = ((lscales & np.uint8(0x0F))
                | ((hscales & np.uint8(0x03)) << np.uint8(4)))
    unpacked = (unpacked.astype(np.int8) - np.int8(32)).astype(np.float32)
    dl = (d * unpacked).reshape(n, 16, 1)

    ql = qs.reshape(n, -1, 1, 32) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = hmask.reshape(n, -1, 1, 32) >> np.array(
        list(range(8)), dtype=np.uint8).reshape(1, 1, 8, 1)
    ql = ql.reshape(n, 16, QK_K // 16) & np.uint8(3)
    # A set hmask bit means no -4 offset — inverted from what the name suggests.
    qh = (qh.reshape(n, 16, QK_K // 16) & np.uint8(1)) ^ np.uint8(1)
    q = (ql.astype(np.int8)
         - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
    return (dl * q).reshape(n, QK_K)


def _k4_scale_min(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """q4_K/q5_K's 12-byte packing: eight 6-bit scales and eight 6-bit mins."""
    n = scales.shape[0]
    scales = scales.reshape(n, 3, 4)
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F,
                         (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mins = np.concatenate([m & 0x3F,
                           (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(n, 8), mins.reshape(n, 8)


def _q4_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, m = _k4_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape(n, -1, 1)
    dm = (dmin * m.astype(np.float32)).reshape(n, -1, 1)

    qs = qs.reshape(n, -1, 1, 32) >> np.array(
        [0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs = (qs & np.uint8(0x0F)).reshape(n, -1, 32).astype(np.float32)
    return (d * qs - dm).reshape(n, QK_K)


def _q5_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [12])
    qh, qs = np.hsplit(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, m = _k4_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape(n, -1, 1)
    dm = (dmin * m.astype(np.float32)).reshape(n, -1, 1)

    ql = qs.reshape(n, -1, 1, 32) >> np.array(
        [0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qh = qh.reshape(n, -1, 1, 32) >> np.array(
        list(range(8)), dtype=np.uint8).reshape(1, 1, 8, 1)
    ql = (ql & np.uint8(0x0F)).reshape(n, -1, 32)
    qh = (qh & np.uint8(0x01)).reshape(n, -1, 32)
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape(n, QK_K)


def _q6_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    ql, rest = np.hsplit(blocks, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])
    scales = scales.view(np.int8).astype(np.float32)
    d = (d.view(np.float16).astype(np.float32) * scales).reshape(
        n, QK_K // 16, 1)

    ql = ql.reshape(n, -1, 1, 64) >> np.array(
        [0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql = (ql & np.uint8(0x0F)).reshape(n, -1, 32)
    qh = qh.reshape(n, -1, 1, 32) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = (qh & np.uint8(0x03)).reshape(n, -1, 32)
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape(n, QK_K // 16, -1).astype(np.float32)
    return (d * q).reshape(n, QK_K)


#: E2M1 magnitudes, doubled (the ggml ``kvalues`` convention — the block
#: scale decoders below return half their nominal value to compensate).
_E2M1_X2 = np.array((0, 1, 2, 3, 4, 6, 8, 12,
                     0, -1, -2, -3, -4, -6, -8, -12), dtype=np.float32)


def _e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
    """E8M0 -> half its float value, as ``ggml_e8m0_to_fp32_half`` does."""
    bits = np.where(x < 2, np.uint32(0x00200000) << x.astype(np.uint32),
                    (x.astype(np.uint32) - 1) << np.uint32(23))
    return bits.view(np.float32)


def _mxfp4(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    e, qs = np.hsplit(blocks, [1])
    d = _e8m0_to_fp32_half(e)
    qs = qs.reshape(n, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(2, 1)
    qs = (qs & np.uint8(0x0F)).astype(np.int64).reshape(n, 32)
    return d * _E2M1_X2[qs].reshape(n, 32)


def _ue4m3_to_fp32(x: np.ndarray) -> np.ndarray:
    """Unsigned E4M3 (bias 7) -> float, at half scale for the x2 kvalues."""
    exp = (x.astype(np.int32) >> 3) & 0xF
    man = (x & 0x7).astype(np.float32)
    raw = np.where(exp == 0, man * 2.0 ** -9,
                   (1.0 + man / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)))
    return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)


def _fp4(blocks: np.ndarray) -> np.ndarray:
    """64-element blocks: 4 UE4M3 sub-scales + 32 nibble bytes (NVFP4 layout)."""
    n = blocks.shape[0]
    d_bytes, qs = np.hsplit(blocks, [4])
    d = _ue4m3_to_fp32(d_bytes).reshape(n, 4, 1)
    qs = qs.reshape(n, 4, 8)
    lo = (qs & np.uint8(0x0F)).astype(np.int64)
    hi = (qs >> np.uint8(4)).astype(np.int64)
    vals = np.concatenate([lo, hi], axis=-1)
    return (d * _E2M1_X2[vals]).reshape(n, 64)


_DECODERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "q8_0": _q8_0,
    "q2_k": _q2_k,
    "q3_k": _q3_k,
    "q4_k": _q4_k,
    "q5_k": _q5_k,
    "q6_k": _q6_k,
    "mxfp4": _mxfp4,
    "fp4": _fp4,
}
