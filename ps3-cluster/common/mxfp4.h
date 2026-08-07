/*
 * mxfp4.h - MXFP4 (microscaling FP4) dequantisation, shared PPE/SPU/host.
 *
 * MXFP4 packs weights in blocks of 32 elements. Each element is 4-bit E2M1
 * (1 sign, 2 exp, 1 mantissa); the block shares one 8-bit E8M0 exponent scale.
 * Layout per 32-element block (matches compressed-tensors / K3):
 *   1 byte  : E8M0 scale (biased exponent, bias 127)
 *   16 bytes: 32 x 4-bit codes, low nibble = even index, high nibble = odd
 *
 * AirLLM #316 keeps this payload packed in memory and expands per use so the
 * resident footprint stays ~4x smaller; we do the same on each PS3 node.
 */
#ifndef MXFP4_H
#define MXFP4_H

#include <stdint.h>
#include <math.h>

#define MXFP4_BLOCK 32
#define MXFP4_BYTES_PER_BLOCK 17   /* 1 scale + 16 packed nibbles */

/* E2M1 value table for the 16 possible 4-bit codes (sign,exp,mantissa). */
static const float MXFP4_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* Decode the E8M0 block scale byte to a float multiplier (2^(e-127)). */
static inline float mxfp4_scale(uint8_t e8m0) {
    return ldexpf(1.0f, (int)e8m0 - 127);
}

/* Dequantise one 32-element block into out[0..31]. block points at the
 * 17-byte packed block. Endianness-agnostic: nibbles are byte-local. */
static inline void mxfp4_decode_block(const uint8_t *block, float *out) {
    float s = mxfp4_scale(block[0]);
    const uint8_t *q = block + 1;
    for (int i = 0; i < MXFP4_BLOCK / 2; ++i) {
        uint8_t byte = q[i];
        out[2 * i]     = MXFP4_LUT[byte & 0x0F] * s;
        out[2 * i + 1] = MXFP4_LUT[(byte >> 4) & 0x0F] * s;
    }
}

/* Dot product of an MXFP4-packed weight row (n elements, n % 32 == 0) with a
 * dense fp32 input vector. This is the inner kernel the SPU accelerates. */
static inline float mxfp4_dot(const uint8_t *row_packed, const float *x, int n) {
    float acc = 0.0f;
    float blk[MXFP4_BLOCK];
    int nblocks = n / MXFP4_BLOCK;
    for (int b = 0; b < nblocks; ++b) {
        mxfp4_decode_block(row_packed + (size_t)b * MXFP4_BYTES_PER_BLOCK, blk);
        const float *xb = x + b * MXFP4_BLOCK;
        for (int i = 0; i < MXFP4_BLOCK; ++i) acc += blk[i] * xb[i];
    }
    return acc;
}

static inline float mxfp4_silu(float v) {
    return v / (1.0f + expf(-v));
}

#endif /* MXFP4_H */
