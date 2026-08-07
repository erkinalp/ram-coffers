/*
 * expert_rsx.c - RSX host driver: run the MXFP4 GEMV on the PS3 GPU.
 *
 * Compiled ONLY on the GameOS-exploit path with a PS3 GPU stack in place
 * (PSGL + the Cg runtime, or RSXGL); guarded by USE_RSX so it never enters the
 * host-sim / SPE builds. It uploads the packed weights, the input vector and
 * the E2M1 LUT as textures, binds the compiled expert_rsx.cg fragment program,
 * renders a rows-tall strip to an fp32 render target, and reads the result
 * back. Same ABI as gemv()/gemv_spe()/rsx_emu_gemv() so expert_ppu.c can pick
 * this backend transparently.
 *
 * This code path is NOT built or run in the current environment (no RSX
 * toolchain / GPU). Its numerical behaviour is validated on x86 by the CPU
 * model in rsx_gemv_emu.h, which reproduces the shader's math exactly.
 */
#ifdef USE_RSX

#include <PSGL/psgl.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../common/mxfp4.h"

/* E2M1 table, identical to the shader's luttex and mxfp4.h. */
static const float RSX_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

typedef struct {
    PSGLdevice  *device;
    PSGLcontext *context;
    CGcontext    cg;
    CGprogram    frag;
    CGprofile    profile;
    GLuint       lut_tex;
    int          ready;
} rsx_state_t;

static rsx_state_t g_rsx = {0};

/* Bring up PSGL + Cg and load the fragment program once. */
int rsx_init(const char *frag_cg_path) {
    if (g_rsx.ready) return 0;

    PSGLinitOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.enable = PSGL_INIT_MAX_SPUS | PSGL_INIT_INITIALIZE_SPUS;
    opt.maxSPUs = 1;
    psglInit(&opt);

    PSGLdeviceParameters p;
    memset(&p, 0, sizeof(p));
    p.enable = PSGL_DEVICE_PARAMETERS_COLOR_FORMAT |
               PSGL_DEVICE_PARAMETERS_DEPTH_FORMAT;
    p.colorFormat = GL_ARGB_SCE;   /* offscreen FBO overrides this for fp32 */
    g_rsx.device = psglCreateDeviceExtended(&p);
    g_rsx.context = psglCreateContext();
    psglMakeCurrent(g_rsx.context, g_rsx.device);

    g_rsx.cg = cgCreateContext();
    g_rsx.profile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    g_rsx.frag = cgCreateProgramFromFile(g_rsx.cg, CG_SOURCE, frag_cg_path,
                                         g_rsx.profile, "main", NULL);
    cgGLLoadProgram(g_rsx.frag);

    /* Upload the 16-entry LUT as a 16x1 fp32 texture. */
    glGenTextures(1, &g_rsx.lut_tex);
    glBindTexture(GL_TEXTURE_2D, g_rsx.lut_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 16, 1, 0,
                 GL_LUMINANCE, GL_FLOAT, RSX_LUT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    g_rsx.ready = 1;
    return 0;
}

/* GEMV on the RSX. W is MXFP4-packed [rows][row_bytes]; x is n fp32. */
void gemv_rsx(const uint8_t *W, const float *x, float *out,
              uint32_t rows, uint32_t n) {
    if (!g_rsx.ready) rsx_init("expert_rsx.cg");

    uint32_t row_bytes = (n / MXFP4_BLOCK) * MXFP4_BYTES_PER_BLOCK;
    uint32_t nblocks = n / MXFP4_BLOCK;

    /* Weights -> R8 texture [row_bytes x rows]. */
    GLuint wtex, xtex, out_tex, fbo;
    glGenTextures(1, &wtex);
    glBindTexture(GL_TEXTURE_2D, wtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, row_bytes, rows, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, W);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    /* Input -> R32F texture [n x 1]. */
    glGenTextures(1, &xtex);
    glBindTexture(GL_TEXTURE_2D, xtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, n, 1, 0,
                 GL_LUMINANCE, GL_FLOAT, x);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    /* fp32 render target [1 x rows] + FBO. */
    glGenTextures(1, &out_tex);
    glBindTexture(GL_TEXTURE_2D, out_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 1, rows, 0,
                 GL_LUMINANCE, GL_FLOAT, NULL);
    glGenFramebuffersOES(1, &fbo);
    glBindFramebufferOES(GL_FRAMEBUFFER_OES, fbo);
    glFramebufferTexture2DOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES,
                              GL_TEXTURE_2D, out_tex, 0);
    glViewport(0, 0, 1, rows);

    /* Bind program + uniforms + samplers. */
    cgGLEnableProfile(g_rsx.profile);
    cgGLBindProgram(g_rsx.frag);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "rows"), (float)rows);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "n"), (float)n);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "nblocks"), (float)nblocks);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "row_bytes"), (float)row_bytes);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "inv_wtex_w"), 1.0f / row_bytes);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "inv_wtex_h"), 1.0f / rows);
    cgGLSetParameter1f(cgGetNamedParameter(g_rsx.frag, "inv_xtex_w"), 1.0f / n);
    cgGLSetTextureParameter(cgGetNamedParameter(g_rsx.frag, "wtex"), wtex);
    cgGLSetTextureParameter(cgGetNamedParameter(g_rsx.frag, "xtex"), xtex);
    cgGLSetTextureParameter(cgGetNamedParameter(g_rsx.frag, "luttex"), g_rsx.lut_tex);
    cgGLEnableTextureParameter(cgGetNamedParameter(g_rsx.frag, "wtex"));
    cgGLEnableTextureParameter(cgGetNamedParameter(g_rsx.frag, "xtex"));
    cgGLEnableTextureParameter(cgGetNamedParameter(g_rsx.frag, "luttex"));

    /* Fullscreen quad -> one fragment per output row. */
    static const float quad[] = { -1,-1, 1,-1, 1,1, -1,1 };
    static const float uv[]   = {  0, 0, 1, 0, 1,1,  0,1 };
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, quad);
    glTexCoordPointer(2, GL_FLOAT, 0, uv);
    glDrawArrays(GL_QUADS, 0, 4);

    /* Read the column of results back. */
    glReadPixels(0, 0, 1, rows, GL_LUMINANCE, GL_FLOAT, out);

    glDeleteFramebuffersOES(1, &fbo);
    glDeleteTextures(1, &out_tex);
    glDeleteTextures(1, &xtex);
    glDeleteTextures(1, &wtex);
}

#endif /* USE_RSX */
