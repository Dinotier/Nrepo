/*
 * fiver/compat.h — CUDA or CPU emulator, and the kernel launch macro.
 *
 * FIVER_LAUNCH(kernel, grid, block, stream, args...) expands to a normal
 * <<<>>> launch under nvcc and to an emulated launch with -DFIVER_EMU.
 * For template kernels take a pointer first (auto k = &kern<A, B>;) so the
 * commas inside the template arguments do not split the macro arguments.
 */
#pragma once

#ifdef FIVER_EMU
#include "fiver/cuda_emu.h"
#define FIVER_LAUNCH(k, grid, block, stream, ...) \
    ::cw_emu::launch(#k, [&]() { (k)(__VA_ARGS__); }, dim3(grid), dim3(block))
#else
#include <cuda_runtime.h>
#define FIVER_LAUNCH(k, grid, block, stream, ...) (k)<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#endif
