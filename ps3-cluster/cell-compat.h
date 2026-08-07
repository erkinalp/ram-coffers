/*
 * cell-compat.h - PS3 Cell Broadband Engine / OtherOS compatibility layer
 *
 * Companion to power8-compat.h, but for the PlayStation 3's Cell BE (CBEA)
 * running Linux under OtherOS (or a GameOS exploit such as AsbestOS). Where
 * power8-compat.h carefully AVOIDS POWER9 builtins on a POWER8 VSX core, this
 * header targets a much older, weaker ISA and a fundamentally different memory
 * model, so most of RAM Coffers' VSX/POWER8 collapse kernels must fall back to
 * scalar or classic AltiVec (VMX) code.
 *
 * Hardware facts that drive every decision here:
 *   - PPE: one PowerPC 970-derived core, 64-bit, BIG-ENDIAN, in-order, 2 SMT
 *     threads. Has classic AltiVec/VMX (128-bit). NO VSX, NO MMA, NO POWER8/9
 *     builtins, NO 128-bit vec_xl partial loads.
 *   - SPEs: 6 usable under OtherOS (7 with a GameOS exploit); each has 256 KB
 *     local store and its own SIMD ISA -- reached via libspe2, not intrinsics.
 *   - 256 MB XDR main RAM total. This is the binding constraint.
 *   - Endianness: BIG-ENDIAN. GGUF/safetensors are little-endian on disk, so
 *     every multi-byte weight/scale must be byte-swapped on load.
 */
#ifndef CELL_COMPAT_H
#define CELL_COMPAT_H

/* Detect the PS3/Cell PPE target. GCC defines __PPU__ under the PS3 toolchain;
 * fall back to generic big-endian ppc64 without POWER8/POWER9 vectors. */
#if defined(__PPU__) || \
    (defined(__powerpc64__) && defined(__BIG_ENDIAN__) && \
     !defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__))
#define GGML_PS3_CELL 1
#endif

#ifdef GGML_PS3_CELL

#include <stdint.h>
#include <string.h>

/* DO NOT define __POWER8_VECTOR__ / __POWER9_VECTOR__: the PPE has neither.
 * RAM Coffers' vec_perm collapse and VSX GEMM headers must select scalar paths
 * when GGML_PS3_CELL is set. */
#define GGML_NO_VSX 1
#define GGML_NO_POWER8 1

/* ---- Endianness helpers (LE on-disk weights -> BE host) ------------------ */
static inline uint16_t cell_bswap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }
static inline uint32_t cell_bswap32(uint32_t v) { return __builtin_bswap32(v); }
static inline uint64_t cell_bswap64(uint64_t v) { return __builtin_bswap64(v); }

/* Load a little-endian value from a mmap'd checkpoint into a native (BE) reg. */
#define GGML_LE16(p) cell_bswap16(*(const uint16_t *)(p))
#define GGML_LE32(p) cell_bswap32(*(const uint32_t *)(p))
#define GGML_LE64(p) cell_bswap64(*(const uint64_t *)(p))

/* Byte-swap an fp16 payload in place (used when expanding MXFP4 scales). */
static inline void cell_bswap_f16_array(uint16_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = cell_bswap16(a[i]);
}

/* ---- AltiVec shims: the POWER8 header assumes vec_xl/vec_xst partial loads
 * that the PPE lacks. Provide aligned classic-VMX equivalents. ------------- */
#ifdef __ALTIVEC__
#include <altivec.h>
#ifndef vec_xl
#define vec_xl(offset, ptr) vec_ld((offset), (ptr))
#endif
#ifndef vec_xst
#define vec_xst(v, offset, ptr) vec_st((v), (offset), (ptr))
#endif
/* No vec_xl_len on VMX: emulate a partial load via a zeroed aligned staging
 * buffer, matching power8-compat.h's fallback shape. */
#ifndef vec_xl_len
#define vec_xl_len(ptr, len) \
    __extension__ ({ \
        union { unsigned char buf[16]; vector unsigned char v; } __u; \
        memset(__u.buf, 0, 16); \
        memcpy(__u.buf, (ptr), (len) > 16 ? 16 : (len)); \
        __u.v; \
    })
#endif
#endif /* __ALTIVEC__ */

/* ---- SPE offload budget --------------------------------------------------
 * One MoE expert must fit (input tile + weights tile + output) in a 256 KB
 * SPE local store, so the SPE matmul must stream weight tiles via DMA. This
 * cap is advisory for the libspe2 kernel; the reference numpy worker ignores
 * it. */
#define CELL_SPE_LOCAL_STORE 262144            /* 256 KB */
#define CELL_SPE_USABLE      (6)               /* OtherOS; 7 with GameOS exploit */
#define CELL_MAIN_RAM_BYTES  (256UL * 1024 * 1024)

/* RSX GDDR3: 256 MB, ~240 MB usable after the framebuffer. Under OtherOS it is
 * mappable via the hypervisor but reads at ~16 MB/s (write-fast, read-slow), so
 * it is cold storage only -- never place hot, per-token weights there. Under a
 * GameOS exploit (AsbestOS) it is a full-speed (~22.4 GB/s) hot tier. */
#define CELL_RSX_RAM_BYTES        (256UL * 1024 * 1024)
#define CELL_RSX_USABLE_BYTES     (240UL * 1024 * 1024)
#define CELL_RSX_OTHEROS_READ_BPS (16UL * 1024 * 1024)   /* advisory: read-slow */

#define GGML_CELL_COMPAT_ACTIVE 1

#endif /* GGML_PS3_CELL */
#endif /* CELL_COMPAT_H */
