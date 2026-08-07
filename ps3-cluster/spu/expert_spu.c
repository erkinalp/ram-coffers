/*
 * expert_spu.c - SPE kernel: MXFP4 GEMV slice for one MoE expert.
 *
 * One invocation computes out[r] = dot(W[r], x) for r in [row_begin, row_end),
 * where W is an MXFP4-packed matrix in main memory and x is a dense fp32 vector.
 * The PPE (expert_ppu.c) splits an expert's gate/up/down matmuls across the
 * available SPEs by row range, and fuses SwiGLU (silu(gate.x) * up.x) between
 * the two GEMV passes.
 *
 * Memory model: x is DMA'd into local store once; weight rows are streamed with
 * double-buffered DMA so only two rows (~2 x 3.8 KB for hidden=7168) are ever
 * resident, well inside the 256 KB local store. Build with spu-gcc from the
 * ps3dev toolchain (see Makefile). This file is SPU-only and is not part of the
 * host-sim build.
 */
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <stdint.h>
#include <string.h>

#include "../common/mxfp4.h"

typedef struct {
    uint64_t w_ea;        /* EA of MXFP4 weight matrix (row-major, packed) */
    uint64_t x_ea;        /* EA of fp32 input vector (n elements)          */
    uint64_t out_ea;      /* EA of fp32 output vector (>= row_end)         */
    uint32_t n;           /* input dim, multiple of 32                     */
    uint32_t row_begin;
    uint32_t row_end;
    uint32_t row_bytes;   /* packed bytes per row = n/32 * 17              */
} spu_gemv_ctl_t;

#define MAX_N 8192                     /* max hidden/intermediate we support */
#define MAX_ROW_BYTES ((MAX_N / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK)
#define ALIGN16 __attribute__((aligned(16)))

static float x_ls[MAX_N] ALIGN16;
static unsigned char row_buf[2][(MAX_ROW_BYTES + 15) & ~15] ALIGN16;
static float out_ls[256] ALIGN16;      /* flushed in chunks */

int main(uint64_t spe_id, uint64_t argp, uint64_t envp) {
    (void)spe_id; (void)envp;

    /* Fetch the control block. */
    spu_gemv_ctl_t ctl ALIGN16;
    mfc_get(&ctl, argp, sizeof(ctl), 0, 0, 0);
    mfc_write_tag_mask(1 << 0);
    mfc_read_tag_status_all();

    uint32_t n = ctl.n;
    if (n > MAX_N) return 1;

    /* DMA the input vector into local store once. */
    uint32_t xbytes = (n * 4 + 15) & ~15u;
    mfc_get(x_ls, ctl.x_ea, xbytes, 1, 0, 0);
    mfc_write_tag_mask(1 << 1);
    mfc_read_tag_status_all();

    uint32_t rb = (ctl.row_bytes + 15) & ~15u;
    uint32_t r = ctl.row_begin;
    int buf = 0;

    /* Prime the first weight row. */
    if (r < ctl.row_end) {
        mfc_get(row_buf[buf], ctl.w_ea + (uint64_t)r * ctl.row_bytes, rb,
                (uint32_t)(2 + buf), 0, 0);
    }

    uint32_t out_fill = 0;
    uint64_t out_base = ctl.out_ea + (uint64_t)ctl.row_begin * 4;

    while (r < ctl.row_end) {
        int cur = buf;
        int nxt = buf ^ 1;
        uint32_t rn = r + 1;

        /* Kick off the next row's DMA before computing the current one. */
        if (rn < ctl.row_end) {
            mfc_get(row_buf[nxt], ctl.w_ea + (uint64_t)rn * ctl.row_bytes, rb,
                    (uint32_t)(2 + nxt), 0, 0);
        }

        /* Wait for the current row, then compute its dot product. */
        mfc_write_tag_mask(1 << (2 + cur));
        mfc_read_tag_status_all();

        out_ls[out_fill++] = mxfp4_dot(row_buf[cur], x_ls, (int)n);

        /* Flush output in aligned chunks. */
        if (out_fill == 256 || rn >= ctl.row_end) {
            uint32_t bytes = (out_fill * 4 + 15) & ~15u;
            mfc_put(out_ls, out_base, bytes, 6, 0, 0);
            mfc_write_tag_mask(1 << 6);
            mfc_read_tag_status_all();
            out_base += (uint64_t)out_fill * 4;
            out_fill = 0;
        }

        buf = nxt;
        r = rn;
    }

    return 0;
}
