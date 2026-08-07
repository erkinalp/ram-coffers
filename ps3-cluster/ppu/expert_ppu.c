/*
 * expert_ppu.c - PPE driver + TCP worker for one resident MoE expert.
 *
 * This is the program that runs on each PS3. It:
 *   1. Loads its ONE expert's MXFP4 weights into main RAM and keeps them
 *      resident for the process lifetime (the inverse of AirLLM #316's stream-
 *      from-disk; here the weights never leave RAM).
 *   2. Serves the P3XC protocol over TCP: for each request it runs the expert's
 *      SwiGLU FFN on the input activation and returns the output activation.
 *   3. On real hardware (__PPU__ + HAVE_LIBSPE2) it splits each GEMV across the
 *      Cell SPEs via libspe2 running expert_spu.c; otherwise (host-sim, used for
 *      CI and for bring-up on x86) it runs the identical math scalar on the CPU,
 *      so the coordinator cannot tell a simulated node from a real one.
 *
 * Expert file format (.exp), little-endian on disk (byte-swapped on BE load):
 *   char   magic[4] = "EXP0"
 *   uint32 hidden
 *   uint32 inter
 *   uint16 layer
 *   uint16 expert
 *   MXFP4  gate[inter][hidden]   (packed, row-major)
 *   MXFP4  up  [inter][hidden]
 *   MXFP4  down[hidden][inter]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>

#include "../common/mxfp4.h"
#include "../common/p3xc.h"
#include "../cell-compat.h"

#if defined(__PPU__) && defined(HAVE_LIBSPE2)
#include <libspe2.h>
#include <pthread.h>
#define USE_SPE 1
extern spe_program_handle_t expert_spu;   /* embedded SPU image */
#endif

/* Compute backend selection (mutually exclusive; scalar is the default):
 *   USE_SPE        - real Cell SPEs via libspe2 (set on __PPU__ + HAVE_LIBSPE2)
 *   USE_RSX        - real RSX GPU shader (GameOS-exploit path, PSGL/Cg)
 *   GEMV_RSX_EMU   - CPU model of the RSX shader (host testing only)
 * USE_RSX/GEMV_RSX_EMU take precedence over USE_SPE when explicitly requested. */
#if defined(USE_RSX)
void gemv_rsx(const uint8_t *, const float *, float *, uint32_t, uint32_t);
#elif defined(GEMV_RSX_EMU)
#include "../rsx/rsx_gemv_emu.h"
#endif

typedef struct {
    uint32_t hidden, inter;
    uint16_t layer, expert;
    const uint8_t *gate, *up, *down;   /* into the resident buffer */
    uint8_t *blob;                     /* owns the packed weights   */
    size_t blob_len;
} expert_t;

static uint32_t rd_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int expert_load(expert_t *e, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);
    if (memcmp(buf, "EXP0", 4) != 0) { free(buf); return -1; }
    e->hidden = rd_le32(buf + 4);
    e->inter  = rd_le32(buf + 8);
    e->layer  = rd_le16(buf + 12);
    e->expert = rd_le16(buf + 14);
    size_t off = 16;
    size_t row_h = (size_t)(e->hidden / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK;
    size_t row_i = (size_t)(e->inter  / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK;
    e->gate = buf + off;              off += (size_t)e->inter  * row_h;
    e->up   = buf + off;              off += (size_t)e->inter  * row_h;
    e->down = buf + off;              off += (size_t)e->hidden * row_i;
    e->blob = buf;
    e->blob_len = sz;
    return (off == (size_t)sz) ? 0 : -1;
}

#ifdef USE_SPE
/* Control block shared with expert_spu.c (must match its layout exactly). */
typedef struct {
    uint64_t w_ea, x_ea, out_ea;
    uint32_t n, row_begin, row_end, row_bytes;
} spu_gemv_ctl_t __attribute__((aligned(16)));

typedef struct {
    spe_context_ptr_t ctx;
    spu_gemv_ctl_t ctl __attribute__((aligned(16)));
} spe_job_t;

static void *spe_run(void *arg) {
    spe_job_t *j = (spe_job_t *)arg;
    unsigned int entry = SPE_DEFAULT_ENTRY;
    spe_context_run(j->ctx, &entry, 0, &j->ctl, NULL, NULL);
    return NULL;
}

/* Split rows across all usable SPEs and run the MXFP4 GEMV kernel. */
static void gemv_spe(const uint8_t *W, const float *x, float *out,
                     uint32_t rows, uint32_t n) {
    int nspe = (int)spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
    if (nspe < 1) nspe = 1;
    if (nspe > CELL_SPE_USABLE) nspe = CELL_SPE_USABLE;
    uint32_t row_bytes = (n / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK;
    uint32_t per = (rows + nspe - 1) / nspe;

    spe_job_t *jobs = (spe_job_t *)calloc(nspe, sizeof(spe_job_t));
    pthread_t *th = (pthread_t *)calloc(nspe, sizeof(pthread_t));
    int used = 0;
    for (int i = 0; i < nspe; ++i) {
        uint32_t begin = i * per;
        if (begin >= rows) break;
        uint32_t end = begin + per; if (end > rows) end = rows;
        jobs[i].ctx = spe_context_create(0, NULL);
        spe_program_load(jobs[i].ctx, &expert_spu);
        jobs[i].ctl.w_ea = (uint64_t)(uintptr_t)W;
        jobs[i].ctl.x_ea = (uint64_t)(uintptr_t)x;
        jobs[i].ctl.out_ea = (uint64_t)(uintptr_t)out;
        jobs[i].ctl.n = n;
        jobs[i].ctl.row_begin = begin;
        jobs[i].ctl.row_end = end;
        jobs[i].ctl.row_bytes = row_bytes;
        pthread_create(&th[i], NULL, spe_run, &jobs[i]);
        used++;
    }
    for (int i = 0; i < used; ++i) {
        pthread_join(th[i], NULL);
        spe_context_destroy(jobs[i].ctx);
    }
    free(jobs); free(th);
}
#endif /* USE_SPE */

/* GEMV: out[r] = dot(W[r], x), r in [0,rows). n = input dim. */
static void gemv(const uint8_t *W, const float *x, float *out,
                 uint32_t rows, uint32_t n) {
#if defined(USE_RSX)
    gemv_rsx(W, x, out, rows, n);
#elif defined(GEMV_RSX_EMU)
    rsx_emu_gemv(W, x, out, rows, n);
#elif defined(USE_SPE)
    gemv_spe(W, x, out, rows, n);
#else
    size_t row_bytes = (size_t)(n / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK;
    for (uint32_t r = 0; r < rows; ++r)
        out[r] = mxfp4_dot(W + (size_t)r * row_bytes, x, (int)n);
#endif
}

/* SwiGLU expert forward: y = down( silu(gate.x) * up.x ). */
static void expert_forward(const expert_t *e, const float *x, float *y) {
    float *g = (float *)malloc(sizeof(float) * e->inter);
    float *u = (float *)malloc(sizeof(float) * e->inter);
    gemv(e->gate, x, g, e->inter, e->hidden);
    gemv(e->up,   x, u, e->inter, e->hidden);
    for (uint32_t i = 0; i < e->inter; ++i)
        g[i] = mxfp4_silu(g[i]) * u[i];
    gemv(e->down, g, y, e->hidden, e->inter);
    free(g); free(u);
}

/* ---- socket helpers ------------------------------------------------------ */
#include <sys/socket.h>
#include <netinet/in.h>

static int recv_exact(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

static int send_all(int fd, const uint8_t *buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = send(fd, buf + sent, n - sent, 0);
        if (w <= 0) return -1;
        sent += (size_t)w;
    }
    return 0;
}

/* Handle one frame. Returns 0 to keep the connection open, -1 to close it. */
static int handle_frame(int fd, const expert_t *e) {
    uint8_t lenbuf[4];
    if (recv_exact(fd, lenbuf, 4) < 0) return -1;
    uint32_t body_len = ntohl(*(uint32_t *)lenbuf);
    if (body_len == 0 || body_len > (1u << 26)) return -1;
    uint8_t *body = (uint8_t *)malloc(body_len);
    if (!body) return -1;
    if (recv_exact(fd, body, body_len) < 0) { free(body); return -1; }

    p3xc_hdr_t hdr;
    long off = p3xc_parse(body, body_len, &hdr);
    if (off < 0) { free(body); return -1; }

    int rc = 0;
    uint8_t *resp = NULL;
    if (hdr.msg_type == P3XC_PING) {
        /* Heartbeat: echo the coordinator's token so it can correlate the
         * PONG with its probe, and report our own (layer, expert). */
        float z = 0.0f;
        resp = (uint8_t *)malloc(64);
        uint32_t total = p3xc_write_f32(resp, P3XC_PONG, e->layer, e->expert,
                                        hdr.token_id, &z, 1);
        rc = send_all(fd, resp, total);
    } else if (hdr.msg_type == P3XC_REQ &&
        hdr.layer == e->layer && hdr.expert == e->expert &&
        hdr.count == e->hidden) {
        float *x = (float *)malloc(sizeof(float) * hdr.count);
        for (uint32_t i = 0; i < hdr.count; ++i)
            x[i] = p3xc_be_to_f32(*(uint32_t *)(body + off + 4 * i));
        float *y = (float *)malloc(sizeof(float) * e->hidden);
        expert_forward(e, x, y);
        resp = (uint8_t *)malloc(64 + 4 * (size_t)e->hidden);
        uint32_t total = p3xc_write_f32(resp, P3XC_RSP, e->layer, e->expert,
                                        hdr.token_id, y, e->hidden);
        rc = send_all(fd, resp, total);
        free(x); free(y);
    } else {
        float z = 0.0f;
        resp = (uint8_t *)malloc(64);
        uint32_t total = p3xc_write_f32(resp, P3XC_ERR, e->layer, e->expert,
                                        hdr.token_id, &z, 1);
        rc = send_all(fd, resp, total);
    }
    free(resp);
    free(body);
    return rc;
}

/* Serve a persistent connection: keep answering frames until the coordinator
 * closes it (or a write fails). The pooled coordinator transport holds one
 * connection per node open for the whole run, so a connection carries many
 * requests; they are answered in arrival order on this socket, while distinct
 * connections are served by distinct forked workers. */
static void handle_conn(int fd, const expert_t *e) {
    while (handle_frame(fd, e) == 0)
        ;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s expert.exp [port]\n", argv[0]);
        return 2;
    }
    int port = (argc >= 3) ? atoi(argv[2]) : P3XC_DEFAULT_PORT;

    expert_t e;
    if (expert_load(&e, argv[1]) < 0) {
        fprintf(stderr, "failed to load expert %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "expert L%u E%u resident: hidden=%u inter=%u (%zu bytes)\n",
            e.layer, e.expert, e.hidden, e.inter, e.blob_len);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    listen(srv, 16);
    fprintf(stderr, "serving P3XC on port %d\n", port);

    /* Reap connection workers automatically; we never wait() on them. */
    signal(SIGCHLD, SIG_IGN);
    /* A dead coordinator must not kill the node on write. */
    signal(SIGPIPE, SIG_IGN);

    for (;;) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        /* One process per persistent connection: the pooled coordinator keeps
         * several connections open at once, so the accept loop must not block
         * on a single one. fork() shares the resident MXFP4 weights
         * copy-on-write, and they are only ever read, so no console RAM is
         * duplicated. */
        pid_t pid = fork();
        if (pid == 0) {
            close(srv);
            handle_conn(fd, &e);
            close(fd);
            _exit(0);
        }
        if (pid < 0)         /* out of processes: serve it inline */
            handle_conn(fd, &e);
        close(fd);
    }
    free(e.blob);
    return 0;
}
