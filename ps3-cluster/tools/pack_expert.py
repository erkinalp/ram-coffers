"""Pack a SwiGLU MoE expert into the .exp MXFP4 format read by expert_ppu.c.

Also exposes the quantise/dequantise helpers so tests can build an expert file
and compute a bit-faithful numpy reference (both sides consume the *same* packed
bytes and the same E2M1 table, so the C kernel and the numpy reference must
agree to floating-point tolerance).
"""

from __future__ import annotations

import struct
import numpy as np

BLOCK = 32
BYTES_PER_BLOCK = 17

# E2M1 magnitudes; index = code & 7, sign = code >> 3. Matches mxfp4.h MXFP4_LUT.
_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_LUT = np.concatenate([_MAG, -_MAG]).astype(np.float32)  # 16 entries
_MAXMAG = 6.0


def _block_scale_exp(maxabs: float) -> int:
    """Pick E8M0 biased exponent so maxabs maps within the E2M1 range."""
    if maxabs <= 0:
        return 127
    e = int(np.ceil(np.log2(maxabs / _MAXMAG)))
    return int(np.clip(e + 127, 0, 255))


def quantize_row(row: np.ndarray) -> bytes:
    """Quantise one row (len % 32 == 0) to packed MXFP4 bytes."""
    assert row.size % BLOCK == 0
    out = bytearray()
    for b in range(row.size // BLOCK):
        blk = row[b * BLOCK:(b + 1) * BLOCK].astype(np.float32)
        e = _block_scale_exp(float(np.max(np.abs(blk))) if blk.size else 0.0)
        scale = float(2.0 ** (e - 127))
        codes = np.empty(BLOCK, dtype=np.uint8)
        normed = blk / scale
        for i in range(BLOCK):
            codes[i] = int(np.argmin(np.abs(_LUT - normed[i])))
        out.append(e)
        for i in range(BLOCK // 2):
            out.append((codes[2 * i] & 0x0F) | ((codes[2 * i + 1] & 0x0F) << 4))
    return bytes(out)


def quantize_matrix(mat: np.ndarray) -> bytes:
    return b"".join(quantize_row(mat[r]) for r in range(mat.shape[0]))


def dequant_matrix(packed: bytes, rows: int, cols: int) -> np.ndarray:
    """Inverse of quantize_matrix, matching mxfp4.h exactly."""
    row_bytes = (cols // BLOCK) * BYTES_PER_BLOCK
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        base = r * row_bytes
        for b in range(cols // BLOCK):
            off = base + b * BYTES_PER_BLOCK
            e = packed[off]
            scale = 2.0 ** (e - 127)
            for i in range(BLOCK // 2):
                byte = packed[off + 1 + i]
                out[r, b * BLOCK + 2 * i] = _LUT[byte & 0x0F] * scale
                out[r, b * BLOCK + 2 * i + 1] = _LUT[(byte >> 4) & 0x0F] * scale
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def reference_forward(gate_p, up_p, down_p, hidden, inter, x):
    """Compute the expert output from packed weights (numpy ground truth)."""
    G = dequant_matrix(gate_p, inter, hidden)
    U = dequant_matrix(up_p, inter, hidden)
    D = dequant_matrix(down_p, hidden, inter)
    act = silu(G @ x) * (U @ x)
    return (D @ act).astype(np.float32)


def pack_expert(path, gate, up, down, layer=0, expert=0):
    """Write an .exp file. gate/up: [inter,hidden]; down: [hidden,inter]."""
    inter, hidden = gate.shape
    assert up.shape == (inter, hidden)
    assert down.shape == (hidden, inter)
    gate_p = quantize_matrix(gate)
    up_p = quantize_matrix(up)
    down_p = quantize_matrix(down)
    with open(path, "wb") as f:
        f.write(b"EXP0")
        f.write(struct.pack("<II", hidden, inter))
        f.write(struct.pack("<HH", layer, expert))
        f.write(gate_p)
        f.write(up_p)
        f.write(down_p)
    return gate_p, up_p, down_p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g = rng.standard_normal((a.inter, a.hidden)).astype(np.float32) * 0.1
    u = rng.standard_normal((a.inter, a.hidden)).astype(np.float32) * 0.1
    d = rng.standard_normal((a.hidden, a.inter)).astype(np.float32) * 0.1
    pack_expert(a.out, g, u, d, a.layer, a.expert)
    print(f"wrote {a.out} hidden={a.hidden} inter={a.inter}")
