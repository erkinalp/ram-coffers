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

.PHONY: all host ps3 clean

all: host

# ---- host-sim (portable) --------------------------------------------------
HOST_CC    ?= cc
HOST_CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Icommon

host: $(BUILD_DIR)/expert_node_host

$(BUILD_DIR)/expert_node_host: ppu/expert_ppu.c common/p3xc.h common/mxfp4.h cell-compat.h
	@mkdir -p $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ ppu/expert_ppu.c -lm

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

clean:
	rm -rf $(BUILD_DIR)
