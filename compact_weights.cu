/*
 * compact_weights.cu — Fiver 5-bit weight format, CUDA implementation
 *
 * Code k ∈ [0..31], 5 bits.  Two value grids share codes, storage and kernels:
 *
 *   CENTERED (default, for zero-centred weights — every weight matrix of an LLM):
 *     v(k)  = −2^(−k/4)          k =  0..15   (−1.000 … −0.074)
 *     v(16) =  0
 *     v(k)  = +2^(−(31−k)/4)     k = 17..31   (+0.088 … +1.000)
 *     The sign is the MSB of k: no sign bits.  Dense around 0, coarse
 *     towards ±1 — the shape of the bell-shaped weight distribution.
 *
 *   ANCHORED (original Fiver, for positive values clustered near 1):
 *     w(k)  = 1 − 2^(−k/4)       k =  0..15   (0.000 … 0.926)
 *     w(k)  = 1 + 2^(−(31−k)/4)  k = 16..31   (1.074 … 2.000)
 *
 *   CENTERED = ANCHORED − 1, except that code 16 is exactly 0.
 *   Both grids are strictly increasing in k, so the STE update is k ∓ 1.
 *
 * Weight = row_scale[r] · v(k).  Storage: 6 codes per uint32 (bits 0-29,
 * bits 30-31 reserved) → 5.333 bits/weight plus one float per row.
 *
 * Toolchain: CUDA 12.x (CUDA 13 removed sm_61 / GTX 1080 Ti).
 *
 * MIT License — Dinotier 2026
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Error-checking macro: prints file/line on CUDA failure and aborts. */
#define CW_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CW ERROR] %s:%d  %s: %s\n",                    \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Non-fatal warning variant — logs but continues. */
#define CW_CUDA_WARN(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            fprintf(stderr, "[CW WARN]  %s:%d  %s: %s\n",                    \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_e));        \
    } while (0)


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                          DATA  BLOCK                                ║
 * ║  word layout · FiverMatrix · value grids · int8 tables · macros     ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

/* Improved Quake III inverse-sqrt constant — no Newton-Raphson step.
 * Raw error vs exact rsqrt: up to 3.44% (0.175% would need one NR step).
 * Used in benchA() only. */
#define CW_QUAKE_CONST       0x5F375A86U

/* sqrt(2) as IEEE 754 single precision:
 * Full bits 0x3FB504F3 = sign(0) | exp(127) | mantissa(0x3504F3) */
#define CW_SQRT2_BITS        0x3FB504F3U

/* sqrt(2) approximations — fixed known deviation, no refinement ever. */
#define CW_SQRT2_HIGH        1.5f         /* error +6.07% */
#define CW_SQRT2_LOW         1.35f        /* error -4.54% */
#define CW_SQRT2_ERROR_HIGH  0.0607f
#define CW_SQRT2_ERROR_LOW   0.0454f

#define CW_GRID_CENTERED     0
#define CW_GRID_ANCHORED     1

#define CW_K_MAX             31u
#define CW_FLAG_HAS_ACT      (1u << 1)

/* Activation codes (2 bits per weight in act_bits) */
#define CW_ACT_IDENTITY      0u
#define CW_ACT_SQRT          1u
#define CW_ACT_LOG           2u
#define CW_ACT_SOFTSIGN      3u

#define CW_FULL_MASK         0xFFFFFFFFu
#define CW_WEIGHTS_PER_WORD  6u

/*
 * Packed storage: one uint32 word = 6 consecutive codes of one row.
 *
 *   ┌────┬────┬───────┬───────┬───────┬───────┬───────┬───────┐
 *   │ r  │ r  │  k5   │  k4   │  k3   │  k2   │  k1   │  k0   │
 *   │ 31 │ 30 │ 29-25 │ 24-20 │ 19-15 │ 14-10 │  9-5  │  4-0  │
 *   └────┴────┴───────┴───────┴───────┴───────┴───────┴───────┘
 *   r = reserved (0).  Byte-identical to emoji_net/fiver_quant/fiver_gguf.py.
 *
 * Rows are padded to a multiple of 6 with the grid's zero code
 * (16 centered, 0 anchored), so padding never contributes to a product.
 * Row r starts at word r * words_per_row,  words_per_row = ceil(cols / 6).
 */
__host__ __device__ __forceinline__
uint32_t cw_k_at(uint32_t word, int i) { return (word >> (5 * i)) & 0x1Fu; }

static inline uint32_t cw_words_per_row(uint32_t cols) {
    return (cols + CW_WEIGHTS_PER_WORD - 1u) / CW_WEIGHTS_PER_WORD;
}

static inline uint32_t cw_pad_k(int grid) { return grid == CW_GRID_CENTERED ? 16u : 0u; }

static inline uint32_t cw_pack_word(const uint8_t k[6]) {
    uint32_t w = 0;
    for (int i = 0; i < 6; i++) w |= ((uint32_t)k[i] & 0x1Fu) << (5 * i);
    return w;
}

static inline uint32_t cw_pad_word(int grid) {
    uint8_t k[6];
    for (int i = 0; i < 6; i++) k[i] = (uint8_t)cw_pad_k(grid);
    return cw_pack_word(k);
}

/*
 * FiverMatrix — device-resident packed weight matrix (rows × cols).
 * Weights stay packed in VRAM; kernels decode them in registers and never
 * write them back as floats.  The STE update rewrites words in place.
 */
typedef struct {
    uint32_t*       words;      /* [rows * words_per_row]                       */
    const float*    row_scale;  /* [rows], ≥ 0:  W[r,c] = row_scale[r]·v(k)      */
    const uint32_t* act_bits;   /* optional: 2 bits/weight over padded columns  */
    uint32_t rows;
    uint32_t cols;
    uint32_t words_per_row;
    int      grid;              /* CW_GRID_CENTERED | CW_GRID_ANCHORED          */
    uint8_t  flags;             /* CW_FLAG_HAS_ACT                              */
} FiverMatrix;

/*
 * Value LUT per grid:  cw_grid_lut[grid][k] = v(k).
 *
 * Each lane of a warp loads exactly one entry (lane == k) into a register
 * once per kernel.  Decoding is then a register shuffle:
 *     v = __shfl_sync(full, lut_reg, k)
 * 32 codes == 32 lanes: no memory access in the hot loop, and differing k
 * across lanes costs nothing extra.  Changing the grid only changes these
 * 32 numbers — the kernels are grid-agnostic.
 */
__constant__ float cw_grid_lut[2][32] = {
    {   /* CENTERED */
        -1.00000000f, -0.84089642f, -0.70710678f, -0.59460356f,
        -0.50000000f, -0.42044821f, -0.35355339f, -0.29730178f,
        -0.25000000f, -0.21022410f, -0.17677670f, -0.14865089f,
        -0.12500000f, -0.10511205f, -0.08838835f, -0.07432544f,
        +0.00000000f, +0.08838835f, +0.10511205f, +0.12500000f,
        +0.14865089f, +0.17677670f, +0.21022410f, +0.25000000f,
        +0.29730178f, +0.35355339f, +0.42044821f, +0.50000000f,
        +0.59460356f, +0.70710678f, +0.84089642f, +1.00000000f
    },
    {   /* ANCHORED */
        +0.00000000f, +0.15910358f, +0.29289322f, +0.40539644f,
        +0.50000000f, +0.57955179f, +0.64644661f, +0.70269822f,
        +0.75000000f, +0.78977590f, +0.82322330f, +0.85134911f,
        +0.87500000f, +0.89488795f, +0.91161165f, +0.92567456f,
        +1.07432544f, +1.08838835f, +1.10511205f, +1.12500000f,
        +1.14865089f, +1.17677670f, +1.21022410f, +1.25000000f,
        +1.29730178f, +1.35355339f, +1.42044821f, +1.50000000f,
        +1.59460356f, +1.70710678f, +1.84089642f, +2.00000000f
    }
};

/*
 * int8 tables for the INT8 path:  v(k) ≈ offset + e(k)/127
 *   CENTERED: offset 0, e = rn(127·v)        — a plain int8 dot product
 *   ANCHORED: offset 1, e = rn(127·(w − 1))  — plus 127·Σx per input row
 * All 32 values stay distinct; rounding error ≤ 0.5/127 ≈ 0.0039.
 * The two tables differ only at k = 16 (0 vs 9).
 */
__constant__ int cw_grid_e[2][32] = {
    { -127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11,  -9,
         0,   11,  13,  16,  19,  22,  27,  32,  38,  45,  53,  64,  76,  90, 107, 127 },
    { -127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11,  -9,
         9,   11,  13,  16,  19,  22,  27,  32,  38,  45,  53,  64,  76,  90, 107, 127 }
};

/* Host-side reference: delta_lut[m] = 2^(-m/4), m = 0..15 (independent of the GPU tables) */
static const float cw_delta_lut_cpu[16] = {
    1.00000000f, 0.84089642f, 0.70710678f, 0.59460354f,
    0.50000000f, 0.42044821f, 0.35355339f, 0.29730178f,
    0.25000000f, 0.21022410f, 0.17677670f, 0.14865089f,
    0.12500000f, 0.10511205f, 0.08838835f, 0.07432544f
};

/* Saturating increment/decrement — STE weight update in host code */
#define CW_INC_K(k)      ((uint8_t)(((k) < 31u) ? ((k) + 1u) : 31u))
#define CW_DEC_K(k)      ((uint8_t)(((k) > 0u)  ? ((k) - 1u) : 0u))
/* Wrapping variants for exhaustive search / cyclic traversal */
#define CW_INC_K_WRAP(k) ((uint8_t)(((k) + 1u) & 0x1Fu))
#define CW_DEC_K_WRAP(k) ((uint8_t)(((k) - 1u) & 0x1Fu))

/*
 * Option kept open: 6-bit codes (magnitude 0..31, bit 5 = δ/16 = exponent − 4)
 * give 64 distinct values and contain all 5-bit values; 5 codes per uint32
 * (6.4 bits/weight).  Measured gain: large for ANCHORED near-1 tensors, ~3%
 * for CENTERED projection weights.  The δ/4 variant yields only 48 values.
 */

/*
 * INTERFACE DISCONNECTED — Legacy 4-bit n_data format (16 weights per uint64_t)
 * with a separate formula_bits array for the ± direction.
 *
 * Reason for removal: the ±-flag created two disjoint value branches with no
 * monotone ordering in weight space, making gradient-based k-updates undefined
 * across the boundary and preventing vector-space comparisons.
 *
 * Reconnect point: repack legacy data into FiverMatrix words (ANCHORED grid)
 * on the host and use the kernels unchanged.  Math equivalence:
 *   legacy n in [0..15], formula_bit=0 (subtract)  ==  Fiver k = n
 *   legacy n in [0..15], formula_bit=1 (add)        ==  Fiver k = (31 - n)
 *
 * typedef struct {
 *     uint64_t* n_data;        // 16 x 4-bit n values per uint64_t word
 *     uint16_t* formula_bits;  // bit i: 0=subtract, 1=add
 *     uint16_t* sign_bits;
 *     uint32_t* act_bits;
 *     uint32_t  count;
 *     uint8_t   flags;
 * } WeightArray_Legacy;
 */


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                         KERNEL  BLOCK                               ║
 * ║  rsqrt · activations · forward (float / dp4a / TC) · backward       ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

/*
 * Method A: Raw Quake inverse sqrt — one shift + one subtraction, NO Newton-Raphson.
 * Compiles to plain integer ALU instructions, no Special Function Unit (SFU).
 */
__device__ __host__ __forceinline__
float cw_quake_rsqrt(float x) {
    uint32_t i;
    memcpy(&i, &x, sizeof(i));
    i = CW_QUAKE_CONST - (i >> 1);
    float y;
    memcpy(&y, &i, sizeof(y));
    return y;
}

/*
 * Method B: Hardware SFU reciprocal sqrt (MUFU.RSQ).
 * SFUs per SM: 32 on Pascal GP10x, 16 on Turing/Ampere/Ada (4 per partition).
 */
__device__ __forceinline__
float cw_hw_rsqrt(float x) {
#ifdef __CUDA_ARCH__
    return __frsqrt_rn(x);
#else
    return 1.0f / sqrtf(x);
#endif
}

/*
 * Activation on the decoded, unscaled weight v(k).
 * All four are odd and increasing, so they never change the sign of an
 * STE gradient.  Callers should group weights by activation type.
 */
__host__ __device__ __forceinline__
float cw_apply_act(float x, uint32_t code) {
    switch (code & 3u) {
#ifdef __CUDA_ARCH__
        case CW_ACT_SQRT:     return copysignf(__fsqrt_rn(fabsf(x)), x);
#else
        case CW_ACT_SQRT:     return copysignf(sqrtf(fabsf(x)), x);
#endif
        case CW_ACT_LOG:      return copysignf(log1pf(fabsf(x)), x);
        case CW_ACT_SOFTSIGN: return x / (1.0f + fabsf(x));
        default:              return x;
    }
}

/* Host reference value (formula path, independent of cw_grid_lut). */
static inline float cw_value_cpu(int grid, uint32_t k) {
    const uint32_t dir = (k >> 4) & 1u;
    const float    d   = cw_delta_lut_cpu[dir ? 31u - k : k];
    if (grid == CW_GRID_ANCHORED) return dir ? 1.0f + d : 1.0f - d;
    if (k == 16u) return 0.0f;
    return dir ? d : -d;
}

/* Host reference int8 code (formula path, independent of cw_grid_e). */
static inline int cw_e_cpu(int grid, uint32_t k) {
    const float off = grid == CW_GRID_ANCHORED ? 1.0f : 0.0f;
    return (int)lrintf(127.0f * (cw_value_cpu(grid, k) - off));
}

/*
 * Fused Fiver GEMV:  Y[b, r] = row_scale[r] · Σ_c v(k[r, c]) · X[b, c]
 *
 * One warp per output row.  Each lane loads one uint32 word per iteration
 * (a warp reads 128 contiguous bytes = 192 weights, fully coalesced),
 * decodes its 6 codes with register shuffles and FMAs them against X.
 * VRAM traffic for W is 5.333 bits/weight; no float weights exist.
 *
 * BT batch rows share each decoded word; gridDim.y = ceil(batch / BT).
 * The loop trip count is the same for all lanes of a warp, so the
 * full-mask shuffles always run convergently; lanes past the end of the
 * row decode a dummy word and skip the FMA.
 *
 * GPU support: sm_61 (GTX 1080 Ti) and newer — plain FP32 FMA path.
 */
template <int BT>
__global__ void cw_fiver_gemv(
    const uint32_t* __restrict__ W,
    const float*    __restrict__ row_scale,
    const uint32_t* __restrict__ act_bits,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t cols, uint32_t words_per_row,
    uint32_t batch, int grid, uint8_t flags)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const float    lut  = cw_grid_lut[grid][lane];

    if (row >= rows) return;   /* uniform per warp */

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    const uint32_t* wrow   = W + (size_t)row * words_per_row;
    const size_t    w_base = (size_t)row * words_per_row * CW_WEIGHTS_PER_WORD;

    for (uint32_t base = 0; base < words_per_row; base += 32u) {
        const uint32_t wi    = base + lane;
        const bool     valid = wi < words_per_row;
        const uint32_t word  = valid ? __ldg(&wrow[wi]) : 0u;

        float w[CW_WEIGHTS_PER_WORD];
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++)
            w[i] = __shfl_sync(CW_FULL_MASK, lut, (int)cw_k_at(word, i));

        if (!valid) continue;

        if (flags & CW_FLAG_HAS_ACT) {
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                const size_t   idx  = w_base + (size_t)wi * CW_WEIGHTS_PER_WORD + i;
                const uint32_t code = (__ldg(&act_bits[idx >> 4]) >> ((idx & 15u) * 2u)) & 3u;
                w[i] = cw_apply_act(w[i], code);
            }
        }

        const uint32_t c0 = wi * CW_WEIGHTS_PER_WORD;
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const float* xr = X + (size_t)(b0 + b) * cols;
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                if (c0 + i < cols) acc[b] = fmaf(w[i], __ldg(&xr[c0 + i]), acc[b]);
            }
        }
    }

#pragma unroll
    for (int b = 0; b < BT; b++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(CW_FULL_MASK, acc[b], off);
    }

    if (lane == 0) {
        const float s = __ldg(&row_scale[row]);
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b] * s;
    }
}

#define CW_GEMV_WARPS  4
#define CW_GEMV_BT     4

static void cw_fiver_gemv_launch(const FiverMatrix* m, const float* d_x, float* d_y,
                                 uint32_t batch, cudaStream_t stream) {
    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((m->rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS,
              (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);
    cw_fiver_gemv<CW_GEMV_BT><<<grid, block, 0, stream>>>(
        m->words, m->row_scale, m->act_bits, d_x, d_y, m->rows, m->cols, m->words_per_row,
        batch, m->grid, m->flags);
}

/* FP32 reference GEMV with identical structure — baseline for the benchmark. */
template <int BT>
__global__ void cw_f32_gemv(const float* __restrict__ W, const float* __restrict__ X,
                            float* __restrict__ Y, uint32_t rows, uint32_t cols,
                            uint32_t batch)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    if (row >= rows) return;

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    const float* wrow = W + (size_t)row * cols;
    for (uint32_t c = lane; c < cols; c += 32u) {
        const float w = __ldg(&wrow[c]);
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            acc[b] = fmaf(w, __ldg(&X[(size_t)(b0 + b) * cols + c]), acc[b]);
        }
    }
#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(CW_FULL_MASK, acc[b], off);
    if (lane == 0)
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b];
}

/* Decode probe: lane k writes v(k) and e(k) via the shuffle path (tests both LUTs). */
__global__ void cw_lut_probe(float* out_v, int* out_e, int grid) {
    const int   lane = (int)(threadIdx.x & 31u);
    const float lut  = cw_grid_lut[grid][lane];
    const int   elut = cw_grid_e[grid][lane];
    out_v[lane] = __shfl_sync(CW_FULL_MASK, lut, lane);
    out_e[lane] = __shfl_sync(CW_FULL_MASK, elut, lane);
}


/*
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  INT8 PATH                                                           │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 *   v(k) ≈ offset + e(k)/127,   x ≈ sx · xq   (per input row, sx = max|x|/127)
 *
 *   y[b,r] ≈ row_scale[r] · sx[b] / 127 · ( Σ_c e[r,c] · xq[b,c]  +  offset · 127 · Σ_c xq[b,c] )
 *                                           └──── int8 dot ────┘      └ one int per input row ┘
 *
 * CENTERED: offset = 0 — a plain int8 dot product.
 * ANCHORED: offset = 1 — Σxq is computed once per input row while quantising.
 * The integer part is exact; only x and v are rounded.
 *   __dp4a:       2 instructions per word (codes packed as two triplets).
 *   Tensor Cores: one WMMA s8 MMA per 16-column step.
 *
 * int32 range: |e·xq| ≤ 127², so cols ≤ 65536 never overflows.
 * Per-weight activations (CW_FLAG_HAS_ACT) cannot be expressed this way; the
 * INT8 path refuses such matrices — use cw_fiver_gemv for those.
 */
#define CW_INT8_MAX_COLS     65536u
#define CW_TC_TILE           16u
#define CW_TC_KCHUNK_WORDS   16u                     /* 96 columns per chunk */
#define CW_TC_KCHUNK_COLS    (CW_TC_KCHUNK_WORDS * CW_WEIGHTS_PER_WORD)

typedef struct {
    int8_t*  xq;         /* [batch_pad × ld]  int8; cols padded to 96, batch to 16 */
    int32_t* xq3;        /* [batch × triplets] packed bytes (xq0, xq1, xq2, 0)    */
    float*   sx;         /* [batch_pad]        max|x| / 127                        */
    int32_t* sxq;        /* [batch_pad]        Σ_c xq  (offset term, ANCHORED)     */
    uint32_t ld;
    uint32_t batch;
    uint32_t batch_pad;
    uint32_t triplets;   /* 2 · words_per_row                                      */
} CwInt8Workspace;

static void cw_int8_workspace_alloc(CwInt8Workspace* ws, uint32_t cols, uint32_t batch) {
    uint32_t wpr  = cw_words_per_row(cols);
    ws->ld        = (wpr * CW_WEIGHTS_PER_WORD + CW_TC_KCHUNK_COLS - 1u)
                    / CW_TC_KCHUNK_COLS * CW_TC_KCHUNK_COLS;
    ws->batch     = batch;
    ws->batch_pad = (batch + CW_TC_TILE - 1u) / CW_TC_TILE * CW_TC_TILE;
    ws->triplets  = 2u * wpr;
    CW_CUDA_CHECK(cudaMalloc(&ws->xq,  (size_t)ws->batch_pad * ws->ld));
    CW_CUDA_CHECK(cudaMalloc(&ws->xq3, (size_t)batch * ws->triplets * sizeof(int32_t)));
    CW_CUDA_CHECK(cudaMalloc(&ws->sx,  (size_t)ws->batch_pad * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&ws->sxq, (size_t)ws->batch_pad * sizeof(int32_t)));
}

static void cw_int8_workspace_free(CwInt8Workspace* ws) {
    CW_CUDA_WARN(cudaFree(ws->xq));
    CW_CUDA_WARN(cudaFree(ws->xq3));
    CW_CUDA_WARN(cudaFree(ws->sx));
    CW_CUDA_WARN(cudaFree(ws->sxq));
}

/* Per-row symmetric int8 quantisation of X plus Σxq.  grid = batch_pad, block = 256.
 * Rows ≥ batch and columns ≥ cols are written as zero (tile padding). */
__global__ void cw_quantize_x(const float* __restrict__ X, int8_t* __restrict__ xq,
                              float* __restrict__ sx, int32_t* __restrict__ sxq,
                              uint32_t cols, uint32_t ld, uint32_t batch)
{
    __shared__ float red[32];
    __shared__ int   ired[32];
    const uint32_t b    = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nw   = blockDim.x >> 5;
    const float*   xr   = X + (size_t)b * cols;

    float m = 0.0f;
    if (b < batch)
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(xr[c]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(CW_FULL_MASK, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < nw ? red[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(CW_FULL_MASK, m, off));
        if (lane == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];

    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    int sum = 0;
    for (uint32_t c = threadIdx.x; c < ld; c += blockDim.x) {
        int q = 0;
        if (b < batch && c < cols) q = max(-127, min(127, __float2int_rn(xr[c] * inv)));
        xq[(size_t)b * ld + c] = (int8_t)q;
        sum += q;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(CW_FULL_MASK, sum, off);
    if (lane == 0) ired[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (uint32_t i = 0; i < nw; i++) t += ired[i];
        sxq[b] = t;
        sx[b]  = m / 127.0f;
    }
}

/* Triplet packing for the __dp4a path: xq3 = (xq0, xq1, xq2, 0). */
__global__ void cw_pack_x_triplets(const int8_t* __restrict__ xq, int32_t* __restrict__ xq3,
                                   uint32_t ld, uint32_t triplets, uint32_t batch)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * triplets) return;
    const uint32_t b = (uint32_t)(i / triplets), t = (uint32_t)(i % triplets);
    const int8_t*  p = xq + (size_t)b * ld + 3u * t;
    xq3[i] = (p[0] & 0xFF) | ((p[1] & 0xFF) << 8) | ((p[2] & 0xFF) << 16);
}

/*
 * __dp4a GEMV (sm_61+, i.e. GTX 1080 Ti).  Same warp-per-row structure as
 * cw_fiver_gemv; per word: 6 shuffles for e, 2 × __dp4a per batch row,
 * accumulated directly through the dp4a accumulator input.
 */
template <int BT>
__global__ void cw_fiver_gemv_dp4a(
    const uint32_t* __restrict__ W,
    const float*    __restrict__ row_scale,
    const int32_t*  __restrict__ xq3,
    const float*    __restrict__ sx,
    const int32_t*  __restrict__ sxq,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t words_per_row, uint32_t batch, int grid)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const int      elut = cw_grid_e[grid][lane];

    if (row >= rows) return;   /* uniform per warp */

    const uint32_t  T    = 2u * words_per_row;
    const uint32_t* wrow = W + (size_t)row * words_per_row;

    int acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0;

    for (uint32_t base = 0; base < words_per_row; base += 32u) {
        const uint32_t wi    = base + lane;
        const bool     valid = wi < words_per_row;
        const uint32_t word  = valid ? __ldg(&wrow[wi]) : 0u;

        int e[CW_WEIGHTS_PER_WORD];
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++)
            e[i] = __shfl_sync(CW_FULL_MASK, elut, (int)cw_k_at(word, i));

        if (!valid) continue;

        const int E0 = (e[0] & 0xFF) | ((e[1] & 0xFF) << 8) | ((e[2] & 0xFF) << 16);
        const int E1 = (e[3] & 0xFF) | ((e[4] & 0xFF) << 8) | ((e[5] & 0xFF) << 16);
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const int2 xv = __ldg(reinterpret_cast<const int2*>(xq3 + (size_t)(b0 + b) * T) + wi);
            acc[b] = __dp4a(E1, xv.y, __dp4a(E0, xv.x, acc[b]));
        }
    }

#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(CW_FULL_MASK, acc[b], off);

    if (lane == 0) {
        const float f   = __ldg(&row_scale[row]) * (1.0f / 127.0f);
        const float off = grid == CW_GRID_ANCHORED ? 127.0f : 0.0f;
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch)
                Y[(size_t)(b0 + b) * rows + row] =
                    ((float)acc[b] + off * (float)sxq[b0 + b]) * f * sx[b0 + b];
    }
}

/*
 * INT8 Tensor-Core GEMM (sm_75+, WMMA 16×16×16 s8 → s32).  One warp per
 * block computes a 16-row × 16-batch output tile.  Per 96-column chunk the
 * warp decodes 16 rows × 16 words of e-codes into a 1.5 KB shared tile —
 * weights in VRAM stay packed.  The x tile is transposed into shared
 * memory so every 16-column sub-tile starts 256-byte aligned, as
 * load_matrix_sync requires.
 *
 * Pays off for batch ≥ 16; at batch 1 GEMV is memory-bound and the __dp4a
 * kernel is the better choice.  Compiled empty below sm_75; the launcher
 * never selects it there.
 */
__global__ void cw_fiver_gemm_tc(
    const uint32_t* __restrict__ W,
    const float*    __restrict__ row_scale,
    const int8_t*   __restrict__ xq,
    const float*    __restrict__ sx,
    const int32_t*  __restrict__ sxq,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t words_per_row, uint32_t batch, uint32_t ld,
    int grid, uint32_t pad_word)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    __shared__ __align__(32) signed char As[CW_TC_KCHUNK_COLS * CW_TC_TILE];  /* col-major [c][r] */
    __shared__ __align__(32) signed char Bs[CW_TC_KCHUNK_COLS * CW_TC_TILE];  /* row-major [c][b] */
    __shared__ __align__(32) int         Cs[CW_TC_TILE * CW_TC_TILE];

    const uint32_t lane = threadIdx.x;   /* blockDim.x == 32 */
    const uint32_t row0 = blockIdx.x * CW_TC_TILE;
    const uint32_t b0   = blockIdx.y * CW_TC_TILE;
    const int      elut = cw_grid_e[grid][lane];

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
    wmma::fill_fragment(acc, 0);

    for (uint32_t kw = 0; kw < words_per_row; kw += CW_TC_KCHUNK_WORDS) {
        /* Decode 16 rows × 16 words.  r = idx % 16 keeps the byte stores
         * bank-conflict free.  Out-of-range words use the zero code; their
         * columns meet zero x anyway (xq padding). */
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t idx = j * 32u + lane;
            const uint32_t r   = idx & 15u, wc = idx >> 4;
            const uint32_t row = row0 + r,  wi = kw + wc;
            const uint32_t word = (row < rows && wi < words_per_row)
                                ? __ldg(&W[(size_t)row * words_per_row + wi]) : pad_word;
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                const int e = __shfl_sync(CW_FULL_MASK, elut, (int)cw_k_at(word, i));
                As[(wc * CW_WEIGHTS_PER_WORD + (uint32_t)i) * CW_TC_TILE + r] = (signed char)e;
            }
        }
        /* x tile: 16 batch rows × 96 columns = 384 int32 loads, transposed.
         * b = idx % 16 keeps the transposed byte stores conflict free; the
         * tile is shared by all row tiles, so the reads hit L1/L2. */
        for (uint32_t j = 0; j < 12u; j++) {
            const uint32_t idx = j * 32u + lane;
            const uint32_t b = idx & 15u, q = idx >> 4;
            const int v = __ldg(reinterpret_cast<const int*>(
                                xq + (size_t)(b0 + b) * ld + kw * CW_WEIGHTS_PER_WORD) + q);
#pragma unroll
            for (int t = 0; t < 4; t++)
                Bs[(q * 4u + (uint32_t)t) * CW_TC_TILE + b] = (signed char)(v >> (8 * t));
        }
        __syncwarp();

#pragma unroll
        for (int kk = 0; kk < (int)(CW_TC_KCHUNK_COLS / 16u); kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> bf;
            wmma::load_matrix_sync(a,  As + kk * 256, CW_TC_TILE);
            wmma::load_matrix_sync(bf, Bs + kk * 256, CW_TC_TILE);
            wmma::mma_sync(acc, a, bf, acc);
        }
        __syncwarp();
    }

    wmma::store_matrix_sync(Cs, acc, CW_TC_TILE, wmma::mem_row_major);   /* Cs[r][b] */
    __syncwarp();

    const float off = grid == CW_GRID_ANCHORED ? 127.0f : 0.0f;
    for (uint32_t idx = lane; idx < CW_TC_TILE * CW_TC_TILE; idx += 32u) {
        const uint32_t r = idx >> 4, b = idx & 15u;
        if (row0 + r < rows && b0 + b < batch)
            Y[(size_t)(b0 + b) * rows + row0 + r] =
                ((float)Cs[idx] + off * (float)sxq[b0 + b])
                * row_scale[row0 + r] * (1.0f / 127.0f) * sx[b0 + b];
    }
#endif
}

static int cw_device_has_int8_tc(void) {
    int dev = 0;
    cudaDeviceProp p;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess)
        return 0;
    return p.major * 10 + p.minor >= 75;
}

/*
 * y = W·x through the INT8 path.  Quantises X into ws, then runs the
 * Tensor-Core GEMM (use_tc, sm_75+) or the __dp4a GEMV.
 * Returns 0 without launching anything if the matrix is not eligible.
 */
static int cw_fiver_gemv_int8(const FiverMatrix* m, const float* d_x, float* d_y,
                              const CwInt8Workspace* ws, int use_tc, cudaStream_t stream)
{
    if ((m->flags & CW_FLAG_HAS_ACT) || m->cols > CW_INT8_MAX_COLS) return 0;
    if (use_tc && !cw_device_has_int8_tc()) return 0;

    const uint32_t batch = ws->batch;
    cw_quantize_x<<<ws->batch_pad, 256, 0, stream>>>(d_x, ws->xq, ws->sx, ws->sxq,
                                                     m->cols, ws->ld, batch);
    if (use_tc) {
        dim3 grid((m->rows + CW_TC_TILE - 1u) / CW_TC_TILE, ws->batch_pad / CW_TC_TILE);
        cw_fiver_gemm_tc<<<grid, 32, 0, stream>>>(
            m->words, m->row_scale, ws->xq, ws->sx, ws->sxq, d_y, m->rows, m->words_per_row,
            batch, ws->ld, m->grid, cw_pad_word(m->grid));
        return 1;
    }

    const size_t n3 = (size_t)batch * ws->triplets;
    cw_pack_x_triplets<<<(unsigned)((n3 + 255u) / 256u), 256, 0, stream>>>(
        ws->xq, ws->xq3, ws->ld, ws->triplets, batch);

    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((m->rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS,
              (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);
    cw_fiver_gemv_dp4a<CW_GEMV_BT><<<grid, block, 0, stream>>>(
        m->words, m->row_scale, ws->xq3, ws->sx, ws->sxq, d_y, m->rows, m->words_per_row,
        batch, m->grid);
    return 1;
}


/*
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  BACKWARD — fused, on packed words (no float weight or grad matrix) │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * For one minibatch (dY already contains the activation derivative):
 *
 *   1.  dX[b,c] = Σ_r row_scale[r] · v(k[r,c]) · dY[b,r]    cw_fiver_gemv_t
 *       uses the weights *before* the update, as fiver.py does.
 *
 *   2.  g_rc  = Σ_b dY[b,r] · X[b,c]                         cw_fiver_ste_update
 *       k_rc ← clamp(k_rc − sign(lr · g_rc), 0, 31)
 *       bias_r −= lr · Σ_b dY[b,r]
 *
 * Both grids are strictly increasing in k and row_scale ≥ 0, so
 * sign(∂L/∂k) = sign(∂L/∂W).  The activations are odd and increasing and do
 * not change that sign either.  On the CENTERED grid k simply walks through
 * zero (code 16) when a weight changes sign — no sign bits to manage.
 *
 * g_rc is formed in registers and consumed immediately: the full
 * out × in gradient matrix never exists.  This requires one update per
 * minibatch (train.py: backward → step).  Padding columns keep their code.
 */
#define CW_BWD_RT  4    /* rows per warp in the update kernel (reuses X loads) */

/*
 * dX = Wᵀ · (row_scale ∘ dY).  One thread per packed column word, looping
 * over a slice of rows; partial sums are combined with atomicAdd (dX must
 * be zeroed).  grid = (ceil(wpr / 128), row slices, ceil(batch / BT)).
 * All lanes run the same row loop, so the decode shuffles stay convergent.
 */
template <int BT>
__global__ void cw_fiver_gemv_t(
    const uint32_t* __restrict__ W,
    const float*    __restrict__ row_scale,
    const uint32_t* __restrict__ act_bits,
    const float*    __restrict__ dY,
    float*          __restrict__ dX,
    uint32_t rows, uint32_t cols, uint32_t words_per_row,
    uint32_t batch, uint32_t rows_per_slice, int grid, uint8_t flags)
{
    const uint32_t wi    = blockIdx.x * blockDim.x + threadIdx.x;
    const bool     valid = wi < words_per_row;
    const uint32_t r0    = blockIdx.y * rows_per_slice;
    const uint32_t r1    = min(rows, r0 + rows_per_slice);
    const uint32_t b0    = blockIdx.z * (uint32_t)BT;
    const float    lut   = cw_grid_lut[grid][threadIdx.x & 31u];

    float acc[BT][CW_WEIGHTS_PER_WORD];
#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) acc[b][i] = 0.0f;

    for (uint32_t r = r0; r < r1; r++) {
        const uint32_t word = valid ? __ldg(&W[(size_t)r * words_per_row + wi]) : 0u;
        float w[CW_WEIGHTS_PER_WORD];
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++)
            w[i] = __shfl_sync(CW_FULL_MASK, lut, (int)cw_k_at(word, i));
        if (!valid) continue;
        if (flags & CW_FLAG_HAS_ACT) {
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                const size_t   idx  = ((size_t)r * words_per_row + wi) * CW_WEIGHTS_PER_WORD + i;
                const uint32_t code = (__ldg(&act_bits[idx >> 4]) >> ((idx & 15u) * 2u)) & 3u;
                w[i] = cw_apply_act(w[i], code);
            }
        }
        const float s = __ldg(&row_scale[r]);                              /* warp-uniform */
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const float g = s * __ldg(&dY[(size_t)(b0 + b) * rows + r]);  /* warp-uniform */
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) acc[b][i] = fmaf(w[i], g, acc[b][i]);
        }
    }

    if (!valid) return;
#pragma unroll
    for (int b = 0; b < BT; b++) {
        if (b0 + b >= batch) break;
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
            const uint32_t c = wi * CW_WEIGHTS_PER_WORD + (uint32_t)i;
            if (c < cols) atomicAdd(&dX[(size_t)(b0 + b) * cols + c], acc[b][i]);
        }
    }
}

/*
 * Fused STE update.  One warp handles CW_BWD_RT rows; each lane owns one
 * packed word column of those rows and accumulates 6 × RT gradients in
 * registers, so every X load is reused RT times.  No shuffles: only the
 * code bits are touched, the word is rewritten in place if a code changed.
 */
__global__ void cw_fiver_ste_update(
    uint32_t*    __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ dY,
    float*       __restrict__ bias,
    uint32_t rows, uint32_t cols, uint32_t words_per_row, uint32_t batch, float lr)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * CW_BWD_RT;
    if (row0 >= rows) return;

    for (uint32_t wi = lane; wi < words_per_row; wi += 32u) {
        float g[CW_BWD_RT][CW_WEIGHTS_PER_WORD];
#pragma unroll
        for (int q = 0; q < CW_BWD_RT; q++)
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) g[q][i] = 0.0f;

        const uint32_t c0 = wi * CW_WEIGHTS_PER_WORD;
        for (uint32_t b = 0; b < batch; b++) {
            const float* xr = X + (size_t)b * cols;
            float xv[CW_WEIGHTS_PER_WORD];
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++)
                xv[i] = (c0 + i < cols) ? __ldg(&xr[c0 + i]) : 0.0f;
#pragma unroll
            for (int q = 0; q < CW_BWD_RT; q++) {
                if (row0 + q >= rows) break;
                const float gy = __ldg(&dY[(size_t)b * rows + row0 + q]);   /* warp-uniform */
#pragma unroll
                for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) g[q][i] = fmaf(gy, xv[i], g[q][i]);
            }
        }

#pragma unroll
        for (int q = 0; q < CW_BWD_RT; q++) {
            if (row0 + q >= rows) break;
            uint32_t* wp   = &W[(size_t)(row0 + q) * words_per_row + wi];
            uint32_t  word = *wp, out = word;
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                if (c0 + i >= cols) break;                     /* padding keeps its code */
                const uint32_t sh = 5u * (uint32_t)i;
                const uint32_t k  = (word >> sh) & 0x1Fu;
                const float    d  = g[q][i] * lr;
                const uint32_t kn = (d > 0.0f) ? (k > 0u ? k - 1u : 0u)
                                  : (d < 0.0f) ? (k < CW_K_MAX ? k + 1u : CW_K_MAX) : k;
                out = (out & ~(0x1Fu << sh)) | (kn << sh);
            }
            if (out != word) *wp = out;
        }
    }

    if (bias != NULL && lane < CW_BWD_RT && row0 + lane < rows) {
        float gs = 0.0f;
        for (uint32_t b = 0; b < batch; b++) gs += dY[(size_t)b * rows + row0 + lane];
        bias[row0 + lane] -= lr * gs;
    }
}

/*
 * One fused backward step for a Fiver layer:  dX (optional) with the old
 * weights, then the in-place STE update of m->words (and bias, optional).
 */
static void cw_fiver_backward_step(const FiverMatrix* m, const float* d_x, const float* d_dy,
                                   float* d_dx, float* d_bias, uint32_t batch, float lr,
                                   cudaStream_t stream)
{
    if (d_dx != NULL) {
        const uint32_t rows_per_slice = 64u;
        CW_CUDA_CHECK(cudaMemsetAsync(d_dx, 0, (size_t)batch * m->cols * sizeof(float), stream));
        dim3 grid((m->words_per_row + 127u) / 128u,
                  (m->rows + rows_per_slice - 1u) / rows_per_slice,
                  (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);
        cw_fiver_gemv_t<CW_GEMV_BT><<<grid, 128, 0, stream>>>(
            m->words, m->row_scale, m->act_bits, d_dy, d_dx, m->rows, m->cols,
            m->words_per_row, batch, rows_per_slice, m->grid, m->flags);
    }

    const uint32_t warps = 4u, rows_per_block = warps * CW_BWD_RT;
    dim3 grid((m->rows + rows_per_block - 1u) / rows_per_block);
    cw_fiver_ste_update<<<grid, 32u * warps, 0, stream>>>(
        m->words, d_x, d_dy, d_bias, m->rows, m->cols, m->words_per_row, batch, lr);
}


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                    TEST / BENCHMARK  BLOCK                          ║
 * ║  host helpers · rsqrt bench · tests · GEMV / backward benchmarks    ║
 * ║  (--bench-a | --bench-b | --bench-gemv | --bench-bwd | --test)      ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

static const char* cw_grid_name(int grid) { return grid == CW_GRID_ANCHORED ? "anch" : "cent"; }

static uint32_t cw_rng_state = 0x12345678u;
static uint32_t cw_rand(void) {               /* xorshift32, deterministic */
    cw_rng_state ^= cw_rng_state << 13;
    cw_rng_state ^= cw_rng_state >> 17;
    cw_rng_state ^= cw_rng_state << 5;
    return cw_rng_state;
}
static float cw_randf(float lo, float hi) {
    return lo + (hi - lo) * (float)(cw_rand() & 0xFFFFFFu) / (float)0x1000000;
}

/* Host decode of one weight (row-local column c), unscaled. */
static float cw_decode_cpu(int grid, const uint32_t* row_words, uint32_t c) {
    return cw_value_cpu(grid, cw_k_at(row_words[c / 6u], (int)(c % 6u)));
}

/* Random packed matrix on host: random codes, zero code in padding, random
 * positive row scales, optional act codes. */
typedef struct {
    uint32_t* words;
    uint32_t* act;
    float*    scale;
    uint32_t  wpr;
    uint32_t  padded;
} CwHostMatrix;

static CwHostMatrix make_random_matrix(int grid, uint32_t rows, uint32_t cols, int with_act) {
    CwHostMatrix h;
    h.wpr    = cw_words_per_row(cols);
    h.padded = h.wpr * CW_WEIGHTS_PER_WORD;
    h.words  = (uint32_t*)malloc((size_t)rows * h.wpr * sizeof(uint32_t));
    h.scale  = (float*)malloc(rows * sizeof(float));
    h.act    = NULL;
    for (uint32_t r = 0; r < rows; r++) {
        h.scale[r] = cw_randf(0.01f, 0.1f);
        for (uint32_t wi = 0; wi < h.wpr; wi++) {
            uint8_t k[6];
            for (int i = 0; i < 6; i++)
                k[i] = (wi * 6u + (uint32_t)i < cols) ? (uint8_t)(cw_rand() & 0x1Fu)
                                                      : (uint8_t)cw_pad_k(grid);
            h.words[(size_t)r * h.wpr + wi] = cw_pack_word(k);
        }
    }
    if (with_act) {
        size_t n_act = ((size_t)rows * h.padded + 15u) / 16u;
        h.act = (uint32_t*)malloc(n_act * sizeof(uint32_t));
        for (size_t i = 0; i < n_act; i++) h.act[i] = cw_rand();
    }
    return h;
}

static void free_host_matrix(CwHostMatrix* h) { free(h->words); free(h->act); free(h->scale); }

/* Upload a host matrix; returns a FiverMatrix with device pointers. */
static FiverMatrix upload_matrix(const CwHostMatrix* h, int grid, uint32_t rows, uint32_t cols) {
    FiverMatrix m;
    uint32_t*   d_words;
    float*      d_scale;
    uint32_t*   d_act = NULL;
    CW_CUDA_CHECK(cudaMalloc(&d_words, (size_t)rows * h->wpr * sizeof(uint32_t)));
    CW_CUDA_CHECK(cudaMalloc(&d_scale, rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h->words, (size_t)rows * h->wpr * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_scale, h->scale, rows * sizeof(float), cudaMemcpyHostToDevice));
    if (h->act) {
        size_t n_act = ((size_t)rows * h->padded + 15u) / 16u;
        CW_CUDA_CHECK(cudaMalloc(&d_act, n_act * sizeof(uint32_t)));
        CW_CUDA_CHECK(cudaMemcpy(d_act, h->act, n_act * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    m.words = d_words; m.row_scale = d_scale; m.act_bits = d_act;
    m.rows = rows; m.cols = cols; m.words_per_row = h->wpr; m.grid = grid;
    m.flags = (uint8_t)(h->act ? CW_FLAG_HAS_ACT : 0u);
    return m;
}

static void free_matrix(FiverMatrix* m) {
    CW_CUDA_WARN(cudaFree(m->words));
    CW_CUDA_WARN(cudaFree((void*)m->row_scale));
    if (m->act_bits) CW_CUDA_WARN(cudaFree((void*)m->act_bits));
}

static float* upload_random(size_t n, float lo, float hi, float** host_out) {
    float* h = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) h[i] = cw_randf(lo, hi);
    float* d;
    CW_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    *host_out = h;
    return d;
}

static float* download(const float* d, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    CW_CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

/* Effective (activated, unscaled) weight on the host. */
static float cw_weight_cpu(int grid, const CwHostMatrix* h, uint32_t r, uint32_t c) {
    float w = cw_decode_cpu(grid, h->words + (size_t)r * h->wpr, c);
    if (h->act) {
        size_t idx = (size_t)r * h->padded + c;
        w = cw_apply_act(w, (h->act[idx >> 4] >> ((idx & 15u) * 2u)) & 3u);
    }
    return w;
}

__global__
void bench_kernel_quake(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cw_quake_rsqrt(in[idx]);
}

__global__
void bench_kernel_hw(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cw_hw_rsqrt(in[idx]);
}

static void run_benchmark(const char* label, int use_quake, int iterations) {
    const uint32_t N = 1u << 20;  /* 1M elements */

    float *d_in, *d_out;
    float *h_in      = (float*)malloc(N * sizeof(float));
    float *h_out_gpu = (float*)malloc(N * sizeof(float));
    float *h_ref     = (float*)malloc(N * sizeof(float));

    /* Test values: 2^(m/4) for m = 0..15 — the Fiver delta denominators. */
    for (uint32_t i = 0; i < N; i++) {
        float m_f  = (float)(i & 15u);
        h_in[i]    = powf(2.0f, m_f * 0.25f);
        h_ref[i]   = 1.0f / sqrtf(h_in[i]);
    }

    CW_CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    if (use_quake) bench_kernel_quake<<<(N+127)/128, 128>>>(d_in, d_out, N);
    else           bench_kernel_hw   <<<(N+127)/128, 128>>>(d_in, d_out, N);
    CW_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CW_CUDA_CHECK(cudaEventCreate(&t0));
    CW_CUDA_CHECK(cudaEventCreate(&t1));
    CW_CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iterations; it++) {
        if (use_quake) bench_kernel_quake<<<(N+127)/128, 128>>>(d_in, d_out, N);
        else           bench_kernel_hw   <<<(N+127)/128, 128>>>(d_in, d_out, N);
    }
    CW_CUDA_CHECK(cudaEventRecord(t1));
    CW_CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CW_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ns_per_op = (ms * 1e6f) / ((float)iterations * (float)N);

    CW_CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0, sum_err = 0.0;
    for (uint32_t i = 0; i < N; i++) {
        double rel = fabs((double)h_out_gpu[i] - (double)h_ref[i])
                   / ((double)h_ref[i] + 1e-12);
        if (rel > max_err) max_err = rel;
        sum_err += rel;
    }

    printf("%-22s | %8.3f ns/op | max_err %7.4f%% | avg_err %7.4f%%\n",
           label, ns_per_op, max_err * 100.0, sum_err / N * 100.0);

    float exact_s2 = sqrtf(2.0f);
    printf("  [sqrt2 approx]  HIGH=%.4f err=+%.2f%%  LOW=%.4f err=-%.2f%%\n",
           CW_SQRT2_HIGH,
           fabsf(CW_SQRT2_HIGH - exact_s2) / exact_s2 * 100.0f,
           CW_SQRT2_LOW,
           fabsf(CW_SQRT2_LOW  - exact_s2) / exact_s2 * 100.0f);

    CW_CUDA_WARN(cudaFree(d_in));
    CW_CUDA_WARN(cudaFree(d_out));
    free(h_in); free(h_out_gpu); free(h_ref);
    CW_CUDA_WARN(cudaEventDestroy(t0));
    CW_CUDA_WARN(cudaEventDestroy(t1));
}

/* 1. Both grids: GPU LUTs (shuffle path) vs the independent host formula. */
static int run_lut_test(void) {
    int failures = 0;
    float* d_v; int* d_e;
    CW_CUDA_CHECK(cudaMalloc(&d_v, 32 * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_e, 32 * sizeof(int)));
    for (int grid = 0; grid <= 1; grid++) {
        float h_v[32]; int h_e[32];
        cw_lut_probe<<<1, 32>>>(d_v, d_e, grid);
        CW_CUDA_CHECK(cudaGetLastError());
        CW_CUDA_CHECK(cudaMemcpy(h_v, d_v, sizeof(h_v), cudaMemcpyDeviceToHost));
        CW_CUDA_CHECK(cudaMemcpy(h_e, d_e, sizeof(h_e), cudaMemcpyDeviceToHost));
        uint32_t ev = 0, ee = 0, mono = 0;
        for (uint32_t k = 0; k < 32; k++) {
            if (fabsf(h_v[k] - cw_value_cpu(grid, k)) > 1e-6f) ev++;
            if (h_e[k] != cw_e_cpu(grid, k)) ee++;
            if (k > 0 && !(h_v[k] > h_v[k - 1] && h_e[k] > h_e[k - 1])) mono++;
        }
        const int zero_ok = h_v[cw_pad_k(grid)] == 0.0f;
        printf("  grid=%s  value errors=%u  int8 errors=%u  non-monotone=%u  zero code ok=%s\n",
               cw_grid_name(grid), ev, ee, mono, zero_ok ? "yes" : "NO");
        failures += (ev || ee || mono || !zero_ok);
    }
    CW_CUDA_WARN(cudaFree(d_v));
    CW_CUDA_WARN(cudaFree(d_e));
    return failures == 0;
}

/* 2. Fused float GEMV vs host double-precision reference. */
static int run_gemv_case(int grid, uint32_t rows, uint32_t cols, uint32_t batch, int with_act) {
    CwHostMatrix h = make_random_matrix(grid, rows, cols, with_act);
    FiverMatrix  m = upload_matrix(&h, grid, rows, cols);
    float* h_x;
    float* d_x = upload_random((size_t)batch * cols, -1.0f, 1.0f, &h_x);
    float* d_y;
    CW_CUDA_CHECK(cudaMalloc(&d_y, (size_t)batch * rows * sizeof(float)));

    cw_fiver_gemv_launch(&m, d_x, d_y, batch, 0);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    float* h_y = download(d_y, (size_t)batch * rows);

    double worst = 0.0;
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t r = 0; r < rows; r++) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t c = 0; c < cols; c++) {
                double t = (double)cw_weight_cpu(grid, &h, r, c) * h_x[(size_t)b * cols + c];
                ref += t; mag += fabs(t);
            }
            ref *= h.scale[r]; mag *= h.scale[r];
            double err = fabs((double)h_y[(size_t)b * rows + r] - ref) / (mag + 1e-12);
            if (err > worst) worst = err;
        }
    int ok = worst < 1e-5;
    printf("  grid=%s  %4ux%-5u  batch=%-3u act=%d   max_rel_err=%.2e  %s\n",
           cw_grid_name(grid), rows, cols, batch, with_act, worst, ok ? "ok" : "FAIL");

    free_matrix(&m); free_host_matrix(&h);
    CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_y));
    free(h_x); free(h_y);
    return ok;
}

/*
 * 3. INT8 path.  The device-quantised x (xq, sx) is read back and the
 * integer sum is recomputed on the host — it must match exactly (tolerance
 * below one integer unit).  The deviation from the exact float result
 * (quantisation of x and v) is reported separately.
 */
static int run_int8_case(int grid, uint32_t rows, uint32_t cols, uint32_t batch, int use_tc) {
    CwHostMatrix h = make_random_matrix(grid, rows, cols, 0);
    FiverMatrix  m = upload_matrix(&h, grid, rows, cols);
    float* h_x = (float*)malloc((size_t)batch * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++) h_x[i] = cw_randf(-1.0f, 1.0f);
    for (uint32_t c = 0; c < cols; c++) h_x[c] = 0.0f;   /* row 0 all-zero: sx = 0 edge case */
    float *d_x, *d_y;
    CW_CUDA_CHECK(cudaMalloc(&d_x, (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));

    CwInt8Workspace ws;
    cw_int8_workspace_alloc(&ws, cols, batch);
    int ok = cw_fiver_gemv_int8(&m, d_x, d_y, &ws, use_tc, 0);
    if (ok) {
        CW_CUDA_CHECK(cudaGetLastError());
        CW_CUDA_CHECK(cudaDeviceSynchronize());
        float*  h_y  = download(d_y, (size_t)batch * rows);
        float*  h_sx = download(ws.sx, ws.batch_pad);
        int8_t* h_xq = (int8_t*)malloc((size_t)ws.batch_pad * ws.ld);
        CW_CUDA_CHECK(cudaMemcpy(h_xq, ws.xq, (size_t)ws.batch_pad * ws.ld, cudaMemcpyDeviceToHost));
        const long long off = grid == CW_GRID_ANCHORED ? 127 : 0;

        double worst_exact = 0.0, max_qerr = 0.0, max_ref = 0.0;
        for (uint32_t b = 0; b < batch; b++) {
            const int8_t* xqr = h_xq + (size_t)b * ws.ld;
            long long sxq = 0;
            for (uint32_t c = 0; c < cols; c++) sxq += xqr[c];
            for (uint32_t r = 0; r < rows; r++) {
                const uint32_t* rw = h.words + (size_t)r * h.wpr;
                long long acc = off * sxq;
                double yf = 0.0;
                for (uint32_t c = 0; c < cols; c++) {
                    const uint32_t k = cw_k_at(rw[c / 6u], (int)(c % 6u));
                    acc += (long long)cw_e_cpu(grid, k) * xqr[c];
                    yf  += (double)cw_value_cpu(grid, k) * h_x[(size_t)b * cols + c];
                }
                const double unit = (double)h.scale[r] * h_sx[b] / 127.0;
                const double yref = (double)acc * unit;
                const double y    = h_y[(size_t)b * rows + r];
                const double ex   = fabs(y - yref) / (1e-6 * fabs(yref) + 0.25 * unit + 1e-30);
                if (ex > worst_exact) worst_exact = ex;
                yf *= h.scale[r];
                if (fabs(y - yf) > max_qerr) max_qerr = fabs(y - yf);
                if (fabs(yf) > max_ref)      max_ref  = fabs(yf);
            }
        }
        ok = worst_exact <= 1.0;
        printf("  %-5s grid=%s  %4ux%-5u  batch=%-3u  int-exact=%s  quant_err=%.2e (rel to max|y|)\n",
               use_tc ? "tc" : "dp4a", cw_grid_name(grid), rows, cols, batch,
               ok ? "ok  " : "FAIL", max_qerr / (max_ref + 1e-30));
        free(h_y); free(h_sx); free(h_xq);
    } else {
        printf("  %-5s grid=%s  %4ux%-5u  batch=%-3u  skipped (not supported here)\n",
               use_tc ? "tc" : "dp4a", cw_grid_name(grid), rows, cols, batch);
        ok = 1;
    }

    cw_int8_workspace_free(&ws);
    free_matrix(&m); free_host_matrix(&h);
    CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_y));
    free(h_x);
    return ok;
}

/*
 * 4. Fused backward vs a double-precision host reference.  A code mismatch
 * only counts as an error when |g| is clearly non-zero (> 1e-5 · Σ|dY·X|);
 * below that, float summation order may legitimately flip the sign.
 * Padding codes and reserved bits must never change.
 */
static int run_backward_case(int grid, uint32_t rows, uint32_t cols, uint32_t batch,
                             int with_act, float lr) {
    CwHostMatrix h = make_random_matrix(grid, rows, cols, with_act);
    FiverMatrix  m = upload_matrix(&h, grid, rows, cols);
    const size_t n_words = (size_t)rows * h.wpr;
    float *h_x, *h_b;
    float* d_x  = upload_random((size_t)batch * cols, -1.0f, 1.0f, &h_x);
    float* h_dy = (float*)malloc((size_t)batch * rows * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * rows; i++) h_dy[i] = cw_randf(-1.0f, 1.0f);
    for (uint32_t b = 0; b < batch; b++) h_dy[(size_t)b * rows] = 0.0f;   /* row 0: g = 0 */
    float* d_dy;
    CW_CUDA_CHECK(cudaMalloc(&d_dy, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_dy, h_dy, (size_t)batch * rows * sizeof(float), cudaMemcpyHostToDevice));
    float* d_b = upload_random(rows, 0.0f, 1.0f, &h_b);
    float* d_dx;
    CW_CUDA_CHECK(cudaMalloc(&d_dx, (size_t)batch * cols * sizeof(float)));

    cw_fiver_backward_step(&m, d_x, d_dy, d_dx, d_b, batch, lr, 0);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* n_w = (uint32_t*)malloc(n_words * sizeof(uint32_t));
    CW_CUDA_CHECK(cudaMemcpy(n_w, m.words, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    float* g_dx = download(d_dx, (size_t)batch * cols);
    float* n_b  = download(d_b, rows);

    double worst_dx = 0.0;
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t c = 0; c < cols; c++) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t r = 0; r < rows; r++) {
                double t = (double)h.scale[r] * cw_weight_cpu(grid, &h, r, c) * h_dy[(size_t)b * rows + r];
                ref += t; mag += fabs(t);
            }
            double e = fabs(g_dx[(size_t)b * cols + c] - ref) / (mag + 1e-30);
            if (e > worst_dx) worst_dx = e;
        }

    uint32_t k_err = 0, k_moved = 0, k_ambig = 0, frozen_err = 0;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t wi = 0; wi < h.wpr; wi++) {
            const uint32_t ow = h.words[(size_t)r * h.wpr + wi], nw = n_w[(size_t)r * h.wpr + wi];
            if ((ow >> 30) != (nw >> 30)) frozen_err++;                 /* reserved bits */
            for (int i = 0; i < 6; i++) {
                const uint32_t c = wi * 6u + (uint32_t)i;
                const uint32_t k_old = cw_k_at(ow, i), k_new = cw_k_at(nw, i);
                if (c >= cols) { if (k_new != k_old) frozen_err++; continue; }
                double g = 0.0, mag = 0.0;
                for (uint32_t b = 0; b < batch; b++) {
                    double t = (double)h_dy[(size_t)b * rows + r] * h_x[(size_t)b * cols + c];
                    g += t; mag += fabs(t);
                }
                const double d = g * lr;
                uint32_t k_exp = k_old;
                if (d > 0.0 && k_old > 0u)  k_exp = k_old - 1u;
                if (d < 0.0 && k_old < 31u) k_exp = k_old + 1u;
                if (k_new != k_old) k_moved++;
                if (k_new != k_exp) {
                    if (fabs(g) <= 1e-5 * mag) k_ambig++;
                    else k_err++;
                }
            }
        }

    double worst_b = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        double gs = 0.0;
        for (uint32_t b = 0; b < batch; b++) gs += h_dy[(size_t)b * rows + r];
        double e = fabs(n_b[r] - (h_b[r] - (double)lr * gs));
        if (e > worst_b) worst_b = e;
    }

    const int ok = worst_dx < 1e-5 && k_err == 0 && frozen_err == 0 && worst_b < 1e-5;
    printf("  grid=%s  %4ux%-5u  batch=%-3u act=%d lr=%+.2f  dX_err=%.1e  k moved=%u/%u"
           " wrong=%u ambiguous=%u  pad/reserved changed=%u  bias_err=%.1e  %s\n",
           cw_grid_name(grid), rows, cols, batch, with_act, lr, worst_dx, k_moved, rows * cols,
           k_err, k_ambig, frozen_err, worst_b, ok ? "ok" : "FAIL");

    free_matrix(&m); free_host_matrix(&h);
    CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_dy));
    CW_CUDA_WARN(cudaFree(d_dx)); CW_CUDA_WARN(cudaFree(d_b));
    free(h_x); free(h_dy); free(h_b); free(n_w); free(g_dx); free(n_b);
    return ok;
}

static int run_fiver_test(void) {
    int failures = 0;

    printf("\n[test] value / int8 LUTs vs host formula:\n");
    failures += !run_lut_test();

    /* Odd shapes exercise row padding (cols % 6 != 0) and batch tails (% BT). */
    printf("[test] fused GEMV vs host reference:\n");
    const uint32_t shapes[][3] = { {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 4096, 9} };
    for (int grid = 0; grid <= 1; grid++)
        for (int s = 0; s < 4; s++)
            for (int act = 0; act <= 1; act++)
                failures += !run_gemv_case(grid, shapes[s][0], shapes[s][1], shapes[s][2], act);

    printf("[test] INT8 path vs exact integer host reference:\n");
    const uint32_t ishapes[][3] = { {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 4096, 9},
                                    {17, 97, 20}, {130, 4096, 33} };
    for (int grid = 0; grid <= 1; grid++)
        for (int s = 0; s < 6; s++)
            failures += !run_int8_case(grid, ishapes[s][0], ishapes[s][1], ishapes[s][2], 0);
    if (cw_device_has_int8_tc()) {
        for (int grid = 0; grid <= 1; grid++)
            for (int s = 0; s < 6; s++)
                failures += !run_int8_case(grid, ishapes[s][0], ishapes[s][1], ishapes[s][2], 1);
    } else {
        printf("  tc    skipped — this GPU has no INT8 Tensor Cores (needs sm_75+)\n");
    }
    {
        FiverMatrix m = { NULL, NULL, NULL, 1, 6, 1, CW_GRID_CENTERED, CW_FLAG_HAS_ACT };
        CwInt8Workspace ws = { NULL, NULL, NULL, NULL, 96, 1, 16, 2 };
        int refused = !cw_fiver_gemv_int8(&m, NULL, NULL, &ws, 0, 0);
        printf("  act-flag matrix refused by INT8 path: %s\n", refused ? "ok" : "FAIL");
        failures += !refused;
    }

    printf("[test] fused backward (dX, STE update on packed words, bias):\n");
    const uint32_t bshapes[][3] = { {37, 1001, 5}, {3, 7, 4}, {130, 1024, 33}, {9, 6, 1} };
    for (int grid = 0; grid <= 1; grid++)
        for (int s = 0; s < 4; s++)
            failures += !run_backward_case(grid, bshapes[s][0], bshapes[s][1], bshapes[s][2],
                                           s == 0, s == 1 ? -0.05f : 0.05f);

    printf("TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0;
}

template <typename F>
static double cw_time_ms(F launch, int iterations) {
    cudaEvent_t t0, t1;
    CW_CUDA_CHECK(cudaEventCreate(&t0));
    CW_CUDA_CHECK(cudaEventCreate(&t1));
    launch();
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    CW_CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iterations; it++) launch();
    CW_CUDA_CHECK(cudaEventRecord(t1));
    CW_CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CW_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CW_CUDA_WARN(cudaEventDestroy(t0));
    CW_CUDA_WARN(cudaEventDestroy(t1));
    return (double)ms / iterations;
}

static double cw_max_rel_diff(const float* d_a, const float* d_ref, size_t n) {
    float* a = download(d_a, n);
    float* r = download(d_ref, n);
    double md = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (fabs((double)a[i] - r[i]) > md) md = fabs((double)a[i] - r[i]);
        if (fabs((double)r[i]) > mr)        mr = fabs((double)r[i]);
    }
    free(a); free(r);
    return md / (mr + 1e-30);
}

/*
 * Benchmark all GEMV paths on one shape (CENTERED grid).  Time includes the
 * x quantisation for the INT8 paths.  "rel err" is measured against the
 * float Fiver path, which uses the exact decoded weights.
 */
static void run_gemv_benchmark(uint32_t rows, uint32_t cols, uint32_t batch, int iterations) {
    const int grid = CW_GRID_CENTERED;
    CwHostMatrix h = make_random_matrix(grid, rows, cols, 0);
    FiverMatrix  m = upload_matrix(&h, grid, rows, cols);

    float* h_wf = (float*)malloc((size_t)rows * cols * sizeof(float));
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            h_wf[(size_t)r * cols + c] = h.scale[r] * cw_decode_cpu(grid, h.words + (size_t)r * h.wpr, c);
    float* h_x;
    float* d_x = upload_random((size_t)batch * cols, -1.0f, 1.0f, &h_x);

    const size_t ny          = (size_t)batch * rows;
    const size_t bytes_fiver = (size_t)rows * h.wpr * sizeof(uint32_t) + rows * sizeof(float);
    const size_t bytes_f32   = (size_t)rows * cols * sizeof(float);
    float *d_wf, *d_yf, *d_y32, *d_yd, *d_yt;
    CW_CUDA_CHECK(cudaMalloc(&d_wf,  bytes_f32));
    CW_CUDA_CHECK(cudaMalloc(&d_yf,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y32, ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_yd,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_yt,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_wf, h_wf, bytes_f32, cudaMemcpyHostToDevice));

    CwInt8Workspace ws;
    cw_int8_workspace_alloc(&ws, cols, batch);
    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grd((rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS, (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);

    const double t32 = cw_time_ms([&] {
        cw_f32_gemv<CW_GEMV_BT><<<grd, block>>>(d_wf, d_x, d_y32, rows, cols, batch); }, iterations);
    const double tf  = cw_time_ms([&] { cw_fiver_gemv_launch(&m, d_x, d_yf, batch, 0); }, iterations);
    const double td  = cw_time_ms([&] { cw_fiver_gemv_int8(&m, d_x, d_yd, &ws, 0, 0); }, iterations);
    const int    tc  = cw_device_has_int8_tc();
    const double tt  = tc ? cw_time_ms([&] { cw_fiver_gemv_int8(&m, d_x, d_yt, &ws, 1, 0); }, iterations)
                          : -1.0;

    printf("GEMV %ux%u  batch=%u  (%d iterations, weight bytes only in GB/s)\n",
           rows, cols, batch, iterations);
    printf("  %-18s %9s %9s %9s %10s\n", "path", "ms", "GB/s", "vs FP32", "rel err");
    printf("  %-18s %9.4f %9.1f %8.2fx %10s\n", "FP32 weights", t32, bytes_f32 / (t32 * 1e6), 1.0, "-");
    printf("  %-18s %9.4f %9.1f %8.2fx %10.2e\n", "Fiver float FMA", tf, bytes_fiver / (tf * 1e6),
           t32 / tf, cw_max_rel_diff(d_y32, d_yf, ny));
    printf("  %-18s %9.4f %9.1f %8.2fx %10.2e\n", "Fiver int8 dp4a", td, bytes_fiver / (td * 1e6),
           t32 / td, cw_max_rel_diff(d_yd, d_yf, ny));
    if (tc)
        printf("  %-18s %9.4f %9.1f %8.2fx %10.2e\n", "Fiver int8 TC", tt, bytes_fiver / (tt * 1e6),
               t32 / tt, cw_max_rel_diff(d_yt, d_yf, ny));
    else
        printf("  %-18s %9s\n", "Fiver int8 TC", "n/a (needs sm_75+)");
    printf("\n");

    cw_int8_workspace_free(&ws);
    free_matrix(&m); free_host_matrix(&h);
    CW_CUDA_WARN(cudaFree(d_wf)); CW_CUDA_WARN(cudaFree(d_x));
    CW_CUDA_WARN(cudaFree(d_yf)); CW_CUDA_WARN(cudaFree(d_y32));
    CW_CUDA_WARN(cudaFree(d_yd)); CW_CUDA_WARN(cudaFree(d_yt));
    free(h_wf); free(h_x);
}

/* Fused backward step timing (dX + STE update), CENTERED grid. */
static void run_backward_benchmark(uint32_t rows, uint32_t cols, uint32_t batch, int iterations) {
    const int grid = CW_GRID_CENTERED;
    CwHostMatrix h = make_random_matrix(grid, rows, cols, 0);
    FiverMatrix  m = upload_matrix(&h, grid, rows, cols);
    float *h_x, *h_dy, *h_b;
    float* d_x  = upload_random((size_t)batch * cols, -1.0f, 1.0f, &h_x);
    float* d_dy = upload_random((size_t)batch * rows, -1.0f, 1.0f, &h_dy);
    float* d_b  = upload_random(rows, 0.0f, 0.0f, &h_b);
    float* d_dx;
    CW_CUDA_CHECK(cudaMalloc(&d_dx, (size_t)batch * cols * sizeof(float)));

    const double t_full = cw_time_ms([&] {
        cw_fiver_backward_step(&m, d_x, d_dy, d_dx, d_b, batch, 0.01f, 0); }, iterations);
    const double t_upd  = cw_time_ms([&] {
        cw_fiver_backward_step(&m, d_x, d_dy, NULL, d_b, batch, 0.01f, 0); }, iterations);
    const double fma = (double)rows * cols * batch;

    printf("Backward %ux%u  batch=%u  (%d iterations)\n", rows, cols, batch, iterations);
    printf("  dX + STE update  %9.4f ms   %7.1f GFMA/s\n", t_full, 2.0 * fma / (t_full * 1e6));
    printf("  STE update only  %9.4f ms   %7.1f GFMA/s\n", t_upd, fma / (t_upd * 1e6));
    printf("  VRAM not needed: float W %.1f MB + float grad matrix %.1f MB (packed W: %.1f MB)\n\n",
           (double)rows * cols * 4 / 1e6, (double)rows * cols * 4 / 1e6,
           ((double)rows * h.wpr * 4 + rows * 4) / 1e6);

    free_matrix(&m); free_host_matrix(&h);
    CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_dy));
    CW_CUDA_WARN(cudaFree(d_dx)); CW_CUDA_WARN(cudaFree(d_b));
    free(h_x); free(h_dy); free(h_b);
}

static void print_device_info(void) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) {
        fprintf(stderr, "[CW WARN]  no CUDA device available: %s\n", cudaGetErrorString(e));
        return;
    }
    cudaDeviceProp prop;
    CW_CUDA_WARN(cudaGetDeviceProperties(&prop, dev));
    printf("[device]  %s  (sm_%d%d)  %.0f MB global  %d SMs  %.0f GB/s peak\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);
    printf("[device]  int8: __dp4a %s   INT8 Tensor Cores %s\n\n",
           prop.major * 10 + prop.minor >= 61 ? "yes" : "no",
           prop.major * 10 + prop.minor >= 75 ? "yes" : "no (needs sm_75+)");
}

int main(int argc, char** argv) {
    printf("compact_weights — Fiver 5-bit (centered / anchored grids)\n");
    printf("==========================================================\n\n");
    print_device_info();

    int do_a = 0, do_b = 0, do_g = 0, do_w = 0, do_t = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench-a")    == 0) do_a = 1;
        if (strcmp(argv[i], "--bench-b")    == 0) do_b = 1;
        if (strcmp(argv[i], "--bench-gemv") == 0) do_g = 1;
        if (strcmp(argv[i], "--bench-bwd")  == 0) do_w = 1;
        if (strcmp(argv[i], "--test")       == 0) do_t = 1;
    }
    if (!do_a && !do_b && !do_g && !do_w && !do_t) { do_a = do_b = do_g = do_w = do_t = 1; }

    if (do_a || do_b) {
        printf("%-22s | %-16s | %-18s | %-18s\n",
               "METHOD", "TIME_NS/OP", "MAX_REL_ERROR", "AVG_REL_ERROR");
        printf("%.80s\n",
               "--------------------------------------------------------------------------------");
        if (do_a) run_benchmark("Quake-raw (A)",    1, 10);
        if (do_b) run_benchmark("__frsqrt_rn (B)",  0, 10);
        printf("\n");
    }

    if (do_g) {
        const uint32_t batches[] = { 1, 4, 16, 64 };
        for (int i = 0; i < 4; i++) run_gemv_benchmark(4096, 4096, batches[i], 50);
    }

    if (do_w) {
        run_backward_benchmark(4096, 4096, 16, 20);
        run_backward_benchmark(4096, 4096, 64, 20);
    }

    int ok = 1;
    if (do_t) ok = run_fiver_test();

    return ok ? 0 : 1;
}
