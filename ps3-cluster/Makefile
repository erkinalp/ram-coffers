# PS3 expert-cluster kernels.
#
# Two build modes:
#   make host      -> host-sim worker (x86/any): scalar MXFP4 FFN, no libspe2.
#                     Used for CI, bring-up, and Python interop tests.
#   make ps3       -> real PS3 build: PPE driver (ppu-gcc) + embedded SPU kernel
#                     (spu-gcc), SPEs driven via libspe2. Needs the ps3dev
#                     toolchain (https://github.com/ps3dev/ps3toolchain).
#
# Default target is `host` so `make` works anywhere.

BUILD_DIR ?= build

.PHONY: all host ps3 ps3-rsx clean

all: host

# ---- host-sim (portable) --------------------------------------------------
HOST_CC    ?= cc
HOST_CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Icommon

# `host` builds both the scalar worker and the RSX-shader-emulator worker; the
# latter routes gemv() through rsx/rsx_gemv_emu.h (the CPU model of the Cg
# shader) so the RSX algorithm is testable on x86.
host: $(BUILD_DIR)/expert_node_host $(BUILD_DIR)/expert_node_rsxemu

$(BUILD_DIR)/expert_node_host: ppu/expert_ppu.c common/p3xc.h common/mxfp4.h cell-compat.h
	@mkdir -p $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ ppu/expert_ppu.c -lm

$(BUILD_DIR)/expert_node_rsxemu: ppu/expert_ppu.c rsx/rsx_gemv_emu.h common/p3xc.h common/mxfp4.h cell-compat.h
	@mkdir -p $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) -DGEMV_RSX_EMU -o $@ ppu/expert_ppu.c -lm

# ---- real PS3 (Cell/OtherOS) ---------------------------------------------
# Requires PS3DEV/PSL1GHT or the IBM Cell SDK in PATH.
PPU_CC  ?= ppu-gcc
SPU_CC  ?= spu-gcc
SPU_EMBED ?= ppu-embedspu
PPU_CFLAGS ?= -O2 -std=c11 -Wall -m64 -Icommon -DHAVE_LIBSPE2
SPU_CFLAGS ?= -O2 -std=c11 -Wall -Icommon
PPU_LDFLAGS ?= -lspe2 -lpthread -lm

ps3: $(BUILD_DIR)/expert_node_ps3

$(BUILD_DIR)/expert_spu.o: spu/expert_spu.c common/mxfp4.h
	@mkdir -p $(BUILD_DIR)
	$(SPU_CC) $(SPU_CFLAGS) -o $(BUILD_DIR)/expert_spu.elf spu/expert_spu.c -lm
	$(SPU_EMBED) -m64 expert_spu $(BUILD_DIR)/expert_spu.elf $(BUILD_DIR)/expert_spu.o

$(BUILD_DIR)/expert_node_ps3: ppu/expert_ppu.c $(BUILD_DIR)/expert_spu.o common/p3xc.h common/mxfp4.h cell-compat.h
	$(PPU_CC) $(PPU_CFLAGS) -o $@ ppu/expert_ppu.c $(BUILD_DIR)/expert_spu.o $(PPU_LDFLAGS)

# ---- real PS3 RSX GPU backend (GameOS-exploit path) -----------------------
# Uses the RSX shader instead of the SPEs. Requires a PS3 GPU stack: PSGL +
# the Cg runtime (or RSXGL). expert_rsx.cg is loaded at runtime from CWD.
RSX_LDFLAGS ?= -lPSGL -lPSGLcgc -lcgc -lgcm_cmd -lgcm_sys -lresc -lsysutil -lpthread -lm

ps3-rsx: $(BUILD_DIR)/expert_node_ps3_rsx

$(BUILD_DIR)/expert_node_ps3_rsx: ppu/expert_ppu.c rsx/expert_rsx.c common/p3xc.h common/mxfp4.h cell-compat.h
	@mkdir -p $(BUILD_DIR)
	$(PPU_CC) -O2 -std=c11 -Wall -m64 -Icommon -DUSE_RSX \
	    -o $@ ppu/expert_ppu.c rsx/expert_rsx.c $(RSX_LDFLAGS)
	@cp rsx/expert_rsx.cg $(BUILD_DIR)/ 2>/dev/null || true

clean:
	rm -rf $(BUILD_DIR)
