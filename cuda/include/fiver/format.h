/*
 * fiver/format.h — the Fiver 5-bit weight format (single source of truth).
 *
 * Code k ∈ [0..31], three value grids, all strictly increasing in k:
 *
 *   CENTERED (0)  v(k) = −2^(−k/4) (k < 16),  v(16) = 0,  +2^(−(31−k)/4) (k > 16)
 *   ANCHORED (1)  w(k) = 1 − 2^(−k/4) (k < 16),  1 + 2^(−(31−k)/4) (k ≥ 16)
 *   MAGIC    (2)  |v| = float_bits(0x3F800000 − (m << 21)), m = k or 31 − k,
 *                 sign as CENTERED, v(16) = 0.  Multiples of 1/64: exact in
 *                 fp16, FP8 (E4M3/E5M2) and int8 (e = 64·v).
 *
 * Weight = row_scale[r] · v(k).  Packed: 6 codes per uint32, k_i at bits 5i,
 * bits 30-31 reserved (0).  Row r of the external (file) layout starts at word
 * r · ceil(cols / 6); rows are padded with the grid's zero code (16, 0, 16).
 *
 * Device layout: the same words, but each row padded to a pitch that is a
 * multiple of FIVER_ROW_ALIGN_WORDS (16 words = 96 columns = 64 bytes) with
 * the zero code.  Tiles of 96 columns therefore never cross a row end and
 * every row starts 64-byte aligned.
 *
 * Usable from C++ host code, CUDA device code and the CPU emulator.
 */
#pragma once

#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define FIVER_HD __host__ __device__ __forceinline__
#else
#define FIVER_HD inline
#endif

#define FIVER_GRID_CENTERED     0
#define FIVER_GRID_ANCHORED     1
#define FIVER_GRID_MAGIC        2
#define FIVER_NUM_GRIDS         3

#define FIVER_K_MAX             31u
#define FIVER_WEIGHTS_PER_WORD  6u
#define FIVER_ROW_ALIGN_WORDS   16u
#define FIVER_MAGIC_ONE         0x3F800000u

/* Value table literals — shared by device constants and host tables. */
#define FIVER_LUT_INIT {                                                                  \
    { -1.00000000f, -0.84089642f, -0.70710678f, -0.59460356f,                            \
      -0.50000000f, -0.42044821f, -0.35355339f, -0.29730178f,                            \
      -0.25000000f, -0.21022410f, -0.17677670f, -0.14865089f,                            \
      -0.12500000f, -0.10511205f, -0.08838835f, -0.07432544f,                            \
      +0.00000000f, +0.08838835f, +0.10511205f, +0.12500000f,                            \
      +0.14865089f, +0.17677670f, +0.21022410f, +0.25000000f,                            \
      +0.29730178f, +0.35355339f, +0.42044821f, +0.50000000f,                            \
      +0.59460356f, +0.70710678f, +0.84089642f, +1.00000000f },                          \
    { +0.00000000f, +0.15910358f, +0.29289322f, +0.40539644f,                            \
      +0.50000000f, +0.57955179f, +0.64644661f, +0.70269822f,                            \
      +0.75000000f, +0.78977590f, +0.82322330f, +0.85134911f,                            \
      +0.87500000f, +0.89488795f, +0.91161165f, +0.92567456f,                            \
      +1.07432544f, +1.08838835f, +1.10511205f, +1.12500000f,                            \
      +1.14865089f, +1.17677670f, +1.21022410f, +1.25000000f,                            \
      +1.29730178f, +1.35355339f, +1.42044821f, +1.50000000f,                            \
      +1.59460356f, +1.70710678f, +1.84089642f, +2.00000000f },                          \
    { -1.00000000f, -0.87500000f, -0.75000000f, -0.62500000f,                            \
      -0.50000000f, -0.43750000f, -0.37500000f, -0.31250000f,                            \
      -0.25000000f, -0.21875000f, -0.18750000f, -0.15625000f,                            \
      -0.12500000f, -0.10937500f, -0.09375000f, -0.07812500f,                            \
      +0.00000000f, +0.09375000f, +0.10937500f, +0.12500000f,                            \
      +0.15625000f, +0.18750000f, +0.21875000f, +0.25000000f,                            \
      +0.31250000f, +0.37500000f, +0.43750000f, +0.50000000f,                            \
      +0.62500000f, +0.75000000f, +0.87500000f, +1.00000000f } }

/* int8 codes for the INT8 path:  v(k) ≈ offset + e(k) / escale */
#define FIVER_E_INIT {                                                                    \
    { -127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11,  -9,  \
         0,   11,  13,  16,  19,  22,  27,  32,  38,  45,  53,  64,  76,  90, 107, 127 },\
    { -127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11,  -9,  \
         9,   11,  13,  16,  19,  22,  27,  32,  38,  45,  53,  64,  76,  90, 107, 127 },\
    {  -64,  -56, -48, -40, -32, -28, -24, -20, -16, -14, -12, -10,  -8,  -7,  -6,  -5,  \
         0,    6,   7,   8,  10,  12,  14,  16,  20,  24,  28,  32,  40,  48,  56,  64 } }
#define FIVER_ESCALE_INIT { 127.0f, 127.0f, 64.0f }
#define FIVER_OFFSET_INIT { 0.0f, 1.0f, 0.0f }

static const float fiver_lut_host[FIVER_NUM_GRIDS][32]   = FIVER_LUT_INIT;
static const int   fiver_e_host[FIVER_NUM_GRIDS][32]     = FIVER_E_INIT;
static const float fiver_escale_host[FIVER_NUM_GRIDS]    = FIVER_ESCALE_INIT;
static const float fiver_offset_host[FIVER_NUM_GRIDS]    = FIVER_OFFSET_INIT;

FIVER_HD uint32_t fiver_k_at(uint32_t word, int i) { return (word >> (5 * i)) & 0x1Fu; }

FIVER_HD uint32_t fiver_zero_code(int grid) { return grid == FIVER_GRID_ANCHORED ? 0u : 16u; }

FIVER_HD uint32_t fiver_pad_word(int grid) {
    const uint32_t z = fiver_zero_code(grid);
    return z | (z << 5) | (z << 10) | (z << 15) | (z << 20) | (z << 25);
}

FIVER_HD uint32_t fiver_words_per_row(uint32_t cols) {
    return (cols + FIVER_WEIGHTS_PER_WORD - 1u) / FIVER_WEIGHTS_PER_WORD;
}

FIVER_HD uint32_t fiver_row_pitch(uint32_t cols) {
    return (fiver_words_per_row(cols) + FIVER_ROW_ALIGN_WORDS - 1u)
           / FIVER_ROW_ALIGN_WORDS * FIVER_ROW_ALIGN_WORDS;
}

FIVER_HD float fiver_bits_to_float(uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
#endif
}

/* MAGIC by formula: shift + subtract for |v|, mirror for k ≥ 16, sign from
 * the MSB, code 16 = 0.  Table-free decode for FPGA / AVX2 backends. */
FIVER_HD float fiver_magic_value(uint32_t k) {
    const uint32_t dir = (k >> 4) & 1u;
    const uint32_t m   = (k ^ (0u - dir)) & 15u;
    const uint32_t b   = (FIVER_MAGIC_ONE - (m << 21)) | ((dir ^ 1u) << 31);
    return k == 16u ? 0.0f : fiver_bits_to_float(b);
}

/* Reference value by formula — independent of the tables, used by tests. */
static inline float fiver_value_ref(int grid, uint32_t k) {
    if (grid == FIVER_GRID_MAGIC) return fiver_magic_value(k);
    static const float delta[16] = {
        1.00000000f, 0.84089642f, 0.70710678f, 0.59460354f, 0.50000000f, 0.42044821f,
        0.35355339f, 0.29730178f, 0.25000000f, 0.21022410f, 0.17677670f, 0.14865089f,
        0.12500000f, 0.10511205f, 0.08838835f, 0.07432544f };
    const uint32_t dir = (k >> 4) & 1u;
    const float    d   = delta[dir ? 31u - k : k];
    if (grid == FIVER_GRID_ANCHORED) return dir ? 1.0f + d : 1.0f - d;
    if (k == 16u) return 0.0f;
    return dir ? d : -d;
}

static inline int fiver_e_ref(int grid, uint32_t k) {
    const float v = fiver_value_ref(grid, k) - fiver_offset_host[grid];
    const float x = fiver_escale_host[grid] * v;
    return (int)(x < 0 ? x - 0.5f : x + 0.5f);
}

static inline uint32_t fiver_pack_word(const uint8_t k[6]) {
    uint32_t w = 0;
    for (int i = 0; i < 6; i++) w |= ((uint32_t)k[i] & 0x1Fu) << (5 * i);
    return w;
}
