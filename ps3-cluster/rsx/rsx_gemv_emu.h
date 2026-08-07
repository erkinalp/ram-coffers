/*
 * rsx_gemv_emu.h - CPU emulation of expert_rsx.cg's per-fragment math.
 *
 * There is no RSX toolchain or PS3 GPU in the build/test environment, so this
 * header reproduces exactly what the Cg fragment shader computes, modelling the
 * same texture fetches (R8 byte normalise/recover, LUT lookup, E8M0 exp2 scale,
 * nibble unpack). Running the worker with -DGEMV_RSX_EMU routes gemv() through
 * here, so the shader *algorithm* can be validated against the numpy reference
 * on x86 before it is ever run on hardware. It is deliberately written in
 * "shader style" (per output row, per block, per nibble) rather than delegating
 * to mxfp4.h, so a bug in the shader formulation (nibble order, LUT indexing,
 * byte-normalise rounding) would show up in tests.
 */
#ifndef RSX_GEMV_EMU_H
#define RSX_GEMV_EMU_H

#include <stdint.h>
#include <math.h>

/* E2M1 table, identical to expert_rsx.cg's luttex and mxfp4.h MXFP4_LUT. */
static const float RSX_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* Model an R8 texture fetch: byte -> normalised [0,1] -> recovered integer. */
static inline float rsx_fetch_byte(const uint8_t *row_bytes_ptr, int idx) {
    float normalised = (float)row_bytes_ptr[idx] / 255.0f;   /* R8 sampler */
    return floorf(normalised * 255.0f + 0.5f);               /* recover byte */
}

static inline float rsx_lut(float code) {
    int idx = (int)(code + 0.5f);
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;
    return RSX_LUT[idx];
}

/* One "fragment": out[r] = dot(W[r], x), emulating the shader's inner loop. */
static inline float rsx_gemv_row(const uint8_t *row, const float *x, int n) {
    int nblocks = n / 32;
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        int base = b * 17;
        float e = rsx_fetch_byte(row, base);
        float scale = exp2f(e - 127.0f);
        for (int i = 0; i < 16; ++i) {
            float byte = rsx_fetch_byte(row, base + 1 + i);
            float lo = fmodf(byte, 16.0f);
            float hi = floorf(byte / 16.0f);
            int col0 = b * 32 + i * 2;
            acc += rsx_lut(lo) * scale * x[col0];
            acc += rsx_lut(hi) * scale * x[col0 + 1];
        }
    }
    return acc;
}

/* Full GEMV: one fragment per output row. Matches the SPE/scalar gemv() ABI. */
static inline void rsx_emu_gemv(const uint8_t *W, const float *x, float *out,
                                uint32_t rows, uint32_t n) {
    size_t row_bytes = (size_t)(n / 32) * 17;
    for (uint32_t r = 0; r < rows; ++r)
        out[r] = rsx_gemv_row(W + (size_t)r * row_bytes, x, (int)n);
}

#endif /* RSX_GEMV_EMU_H */
