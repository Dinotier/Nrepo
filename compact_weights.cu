/*
 * compact_weights.cu — Symmetric Fiver Weight Format, CUDA Implementation
 *
 * Encoding: k in [0..31], 5-bit unsigned, MSB = direction
 *   k in [ 0..15] (bit4=0): w = 1 - 2^(-k/4)      range [0.0  .. 0.926]
 *   k in [16..31] (bit4=1): w = 1 + 2^(-(31-k)/4)  range [1.074 .. 2.0 ]
 *
 * Values are monotonically increasing with k.
 * Distribution is sigmoid-shaped around 1.0 (dense near center, sparse at extremes).
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
 * ║  packed word layouts · FiverMatrix · 32-entry LUT · helper macros   ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

/* Improved Quake III inverse-sqrt constant — no Newton-Raphson step.
 * Raw error vs exact rsqrt: up to ~0.17%.  Used in benchA() only. */
#define CW_QUAKE_CONST       0x5F375A86U

/* sqrt(2) as IEEE 754 single precision:
 * Full bits 0x3FB504F3 = sign(0) | exp(127) | mantissa(0x3504F3) */
#define CW_SQRT2_BITS        0x3FB504F3U

/* sqrt(2) approximations — fixed known deviation, no refinement ever.
 * Choice depends on whether over- or under-estimation is preferred. */
#define CW_SQRT2_HIGH        1.5f         /* error +6.07% */
#define CW_SQRT2_LOW         1.35f        /* error -4.54% */
#define CW_SQRT2_ERROR_HIGH  0.0607f
#define CW_SQRT2_ERROR_LOW   0.0454f

/* Fiver value boundaries */
#define CW_FIVER_MAX         31u
#define CW_FIVER_CENTER_LO   15u          /* w ≈ 0.926, finest step below 1.0 */
#define CW_FIVER_CENTER_HI   16u          /* w ≈ 1.074, finest step above 1.0 */

/* Optional field flags */
#define CW_FLAG_HAS_ACT      (1u << 1)

/* Activation codes (2 bits per weight in act_bits) */
#define CW_ACT_IDENTITY      0u
#define CW_ACT_SQRT          1u
#define CW_ACT_LOG           2u
#define CW_ACT_SOFTSIGN      3u

#define CW_FULL_MASK         0xFFFFFFFFu
#define CW_WEIGHTS_PER_WORD  6u

/*
 * Packed storage: one uint32 word = 6 consecutive weights of one row,
 * plus two sign bits (S_lo for k0..k2, S_hi for k3..k5).  5.333 bpw.
 *
 * Both layouts carry the same information; only the bit positions differ.
 * They are byte-identical to what emoji_net/fiver_quant/fiver_gguf.py writes.
 *
 *   CW_LAYOUT_U16  — two little-endian uint16 triplets read as one uint32
 *     ┌──────┬─────┬─────┬─────┬──────┬─────┬─────┬─────┐
 *     │S_hi  │ k5  │ k4  │ k3  │S_lo  │ k2  │ k1  │ k0  │
 *     │ 31   │30-26│25-21│20-16│ 15   │14-10│ 9-5 │ 4-0 │
 *     └──────┴─────┴─────┴─────┴──────┴─────┴─────┴─────┘
 *
 *   CW_LAYOUT_U32  — sextet with the 2-bit sign code on top
 *     ┌──────┬──────┬─────┬─────┬─────┬─────┬─────┬─────┐
 *     │S_lo  │S_hi  │ k5  │ k4  │ k3  │ k2  │ k1  │ k0  │
 *     │ 31   │ 30   │29-25│24-20│19-15│14-10│ 9-5 │ 4-0 │
 *     └──────┴──────┴─────┴─────┴─────┴─────┴─────┴─────┘
 *     (S_lo S_hi) = 00 all+, 11 all−, 10 −−−+++, 01 +++−−−
 *
 * Rows are padded to a multiple of 6 weights with k=0, sign=0.  Since
 * w(0) = 0.0 exactly, padding contributes nothing to any dot product.
 * Row r starts at word r * words_per_row,  words_per_row = ceil(cols / 6).
 */
#define CW_LAYOUT_U16  0
#define CW_LAYOUT_U32  1

__host__ __device__ __forceinline__
uint32_t cw_k_shift(int layout, int i) {
    return (uint32_t)(5 * i + ((layout == CW_LAYOUT_U16 && i >= 3) ? 1 : 0));
}

__host__ __device__ __forceinline__
uint32_t cw_sign_shift(int layout, int i) {
    if (layout == CW_LAYOUT_U16) return i < 3 ? 15u : 31u;
    return i < 3 ? 31u : 30u;
}

static inline uint32_t cw_words_per_row(uint32_t cols) {
    return (cols + CW_WEIGHTS_PER_WORD - 1u) / CW_WEIGHTS_PER_WORD;
}

/*
 * FiverMatrix — device-resident packed weight matrix (rows × cols).
 * Weights stay packed in VRAM; they are decoded in registers inside the
 * GEMV kernel and never written back as floats.
 */
typedef struct {
    uint32_t*       words;      /* [rows * words_per_row]; updated in place     */
    const uint32_t* act_bits;   /* optional: 2 bits/weight over padded columns  */
    uint32_t rows;
    uint32_t cols;
    uint32_t words_per_row;
    float    scale;             /* per-tensor: y = scale * (W · x)              */
    int      layout;            /* CW_LAYOUT_U16 | CW_LAYOUT_U32                */
    uint8_t  flags;             /* CW_FLAG_HAS_ACT                              */
} FiverMatrix;

/*
 * 32-entry value LUT:  cw_fiver_lut[k] = w(k).
 *
 * Each lane of a warp loads exactly one entry (lane == k) into a register
 * once per kernel.  Decoding is then a register shuffle:
 *     w = __shfl_sync(full, lut_reg, k)
 * 32 Fiver values == 32 lanes, so no memory access happens in the hot loop
 * and differing k across lanes costs nothing extra.  (A __constant__ lookup
 * with per-lane different addresses would serialise up to 32-way.)
 */
__constant__ float cw_fiver_lut[32] = {
    0.00000000f, 0.15910358f, 0.29289322f, 0.40539644f,
    0.50000000f, 0.57955179f, 0.64644661f, 0.70269822f,
    0.75000000f, 0.78977590f, 0.82322330f, 0.85134911f,
    0.87500000f, 0.89488795f, 0.91161165f, 0.92567456f,
    1.07432544f, 1.08838835f, 1.10511205f, 1.12500000f,
    1.14865089f, 1.17677670f, 1.21022410f, 1.25000000f,
    1.29730178f, 1.35355339f, 1.42044821f, 1.50000000f,
    1.59460356f, 1.70710678f, 1.84089642f, 2.00000000f
};

/*
 * int8 LUT for the INT8 path (w = 1 + d):  cw_e_lut[k] = rn(127 · (w(k) − 1)).
 * All 32 values stay distinct; rounding error ≤ 0.5/127 ≈ 0.0039 in w units.
 * (int8 of w itself would need scale 63.5 and lose ~1.75× as much.)
 * Antisymmetric: e(31 − k) = −e(k).  Decoded with the same lane shuffle.
 */
__constant__ int cw_e_lut[32] = {
    -127, -107, -90, -76, -64, -53, -45, -38,
     -32,  -27, -22, -19, -16, -13, -11,  -9,
       9,   11,  13,  16,  19,  22,  27,  32,
      38,   45,  53,  64,  76,  90, 107, 127
};
static const int cw_e_lut_cpu[32] = {
    -127, -107, -90, -76, -64, -53, -45, -38,
     -32,  -27, -22, -19, -16, -13, -11,  -9,
       9,   11,  13,  16,  19,  22,  27,  32,
      38,   45,  53,  64,  76,  90, 107, 127
};

/* Host-side reference: delta_lut[m] = 2^(-m/4), m = 0..15 */
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
 * TODO: 5→6 bit extension via "2→4 trick"
 *   Bit[5] acts as a second scaling stage: delta becomes delta/4.
 *   Math: 2^(-mag/4) → 2^(-mag/4 - 2).
 *   64 distinct values — would need a 64-entry LUT (two shuffles per lookup).
 */

/*
 * INTERFACE DISCONNECTED — Legacy 4-bit n_data format (16 weights per uint64_t)
 * with a separate formula_bits array for the ± direction.
 *
 * Reason for removal: the ±-flag created two disjoint value branches with no
 * monotone ordering in weight space, making gradient-based k-updates undefined
 * across the boundary and preventing vector-space comparisons.
 *
 * Reconnect point: repack legacy data into FiverMatrix words on the host and
 * use cw_fiver_gemv unchanged.  Math equivalence:
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
 * ║  cw_quake_rsqrt · cw_hw_rsqrt · cw_apply_act · cw_fiver_gemv        ║
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
 * Activation on the decoded (signed, unscaled) Fiver weight.
 * Callers should group weights by activation type to avoid warp divergence.
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

/* Host reference decode (formula path, independent of cw_fiver_lut). */
static inline float cw_fiver_to_float_cpu(uint8_t k) {
    uint32_t dir   = (uint32_t)(k >> 4u) & 1u;
    uint32_t mag   = dir ? (31u - (uint32_t)k) : (uint32_t)k;
    float    delta = cw_delta_lut_cpu[mag];
    return 1.0f + (dir ? +delta : -delta);
}

/*
 * Fused Fiver GEMV:  Y[b, r] = scale · Σ_c W[r, c] · X[b, c]
 *
 * One warp per output row.  Each lane loads one uint32 word per iteration
 * (a warp reads 128 contiguous bytes = 192 weights, fully coalesced),
 * decodes its 6 weights with register shuffles and FMAs them against X.
 * Weights are never materialised: VRAM traffic for W is 5.333 bits/weight.
 *
 * BT batch rows share each decoded word (weights stay in registers);
 * gridDim.y = ceil(batch / BT).
 *
 * The loop trip count is identical for all lanes of a warp, so the
 * full-mask __shfl_sync calls are always executed convergently.  Lanes past
 * the end of the row decode word 0 (all k=0 → w=0) and skip the FMA.
 *
 * Grid:  gridDim.x = ceil(rows / warps_per_block), blockDim.x = 32·warps
 * GPU support: sm_61 (GTX 1080 Ti) and newer — plain FP32 FMA path,
 * no Tensor Cores and no FP16 arithmetic required.
 */
template <int LAYOUT, int BT>
__global__ void cw_fiver_gemv(
    const uint32_t* __restrict__ W,
    const uint32_t* __restrict__ act_bits,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t cols, uint32_t words_per_row,
    uint32_t batch, float scale, uint8_t flags)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const float    lut  = cw_fiver_lut[lane];

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
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
            const uint32_t k = (word >> cw_k_shift(LAYOUT, i)) & 0x1Fu;
            const uint32_t s = (word >> cw_sign_shift(LAYOUT, i)) & 1u;
            const float    v = __shfl_sync(CW_FULL_MASK, lut, (int)k);
            w[i] = __int_as_float(__float_as_int(v) ^ (int)(s << 31));
        }

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
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b] * scale;
    }
}

#define CW_GEMV_WARPS  4
#define CW_GEMV_BT     4

/* Host launcher: picks the layout specialisation. */
static void cw_fiver_gemv_launch(const FiverMatrix* m, const float* d_x, float* d_y,
                                 uint32_t batch, cudaStream_t stream) {
    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((m->rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS,
              (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);
    if (m->layout == CW_LAYOUT_U32)
        cw_fiver_gemv<CW_LAYOUT_U32, CW_GEMV_BT><<<grid, block, 0, stream>>>(
            m->words, m->act_bits, d_x, d_y, m->rows, m->cols, m->words_per_row,
            batch, m->scale, m->flags);
    else
        cw_fiver_gemv<CW_LAYOUT_U16, CW_GEMV_BT><<<grid, block, 0, stream>>>(
            m->words, m->act_bits, d_x, d_y, m->rows, m->cols, m->words_per_row,
            batch, m->scale, m->flags);
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

/* Decode probe: lane k writes w(k) via the shuffle path (tests the LUT). */
__global__ void cw_lut_probe(float* out) {
    const float lut = cw_fiver_lut[threadIdx.x & 31u];
    out[threadIdx.x] = __shfl_sync(CW_FULL_MASK, lut, (int)(threadIdx.x & 31u));
}


/*
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  INT8 PATH — decomposition around the Fiver anchor 1.0               │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 *   w = σ · (1 + d),   d = w(k) − 1 ∈ [−1, 1],   σ = shared sign of a triplet
 *
 *   y_r = s · Σ_c σ_c (1 + d_c) x_c
 *       = s · [ Σ_t σ_t · X3_t      +      Σ_c σ_c · d_c · x_c ]
 *               └ triplet sums ┘           └── int8 GEMM ───┘
 *
 *   X3_t = x_3t + x_3t+1 + x_3t+2 is computed once per input row and shared
 *   by every output row.  For unsigned layers (σ ≡ +1) the first term is Σx.
 *
 * Quantisation (per input row b):  sx = max|x| / 127,  xq = rn(x / sx)
 *                                  e  = cw_e_lut[k]  = rn(127 · d)
 *
 *   y_r ≈ s · sx / 127 · Σ_t σ_t · ( 127 · X3q_t  +  Σ_{i∈t} e_i · xq_i )
 *
 * The sum is exact int32 arithmetic.
 *   __dp4a:       one instruction per triplet, the 127·X3q term rides on the
 *                 accumulator input:   v_t = __dp4a(E_t, Xq_t, 127·X3q_t),
 *                 the shared sign is a conditional negate of v_t.
 *   Tensor Cores: C = 127 · (A1 · Xq) + A2 · Xq  with A1 = σ (±1), A2 = σ·e,
 *                 both MMAs reuse the same Xq fragments.
 *
 * int32 range: |v_t| ≤ 3·127² + 127·381 = 96774  →  cols ≤ 65536 is safe.
 * Per-weight activations (CW_FLAG_HAS_ACT) break the decomposition; the
 * INT8 path refuses such matrices — use cw_fiver_gemv for those.
 */
#define CW_INT8_MAX_COLS     65536u
#define CW_TC_TILE           16u
#define CW_TC_KCHUNK_WORDS   16u                     /* 96 columns per chunk */
#define CW_TC_KCHUNK_COLS    (CW_TC_KCHUNK_WORDS * CW_WEIGHTS_PER_WORD)

typedef struct {
    int8_t*  xq;         /* [batch_pad × ld]  int8; cols padded to 96, batch to 16 */
    int32_t* xq3;        /* [batch × triplets] packed bytes (xq0, xq1, xq2, 0)    */
    int32_t* c3;         /* [batch × triplets] 127 · (xq0 + xq1 + xq2)            */
    float*   sx;         /* [batch_pad]        max|x| / 127                        */
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
    CW_CUDA_CHECK(cudaMalloc(&ws->c3,  (size_t)batch * ws->triplets * sizeof(int32_t)));
    CW_CUDA_CHECK(cudaMalloc(&ws->sx,  (size_t)ws->batch_pad * sizeof(float)));
}

static void cw_int8_workspace_free(CwInt8Workspace* ws) {
    CW_CUDA_WARN(cudaFree(ws->xq));
    CW_CUDA_WARN(cudaFree(ws->xq3));
    CW_CUDA_WARN(cudaFree(ws->c3));
    CW_CUDA_WARN(cudaFree(ws->sx));
}

/* Per-row symmetric int8 quantisation of X.  grid = batch_pad, block = 256.
 * Rows ≥ batch and columns ≥ cols are written as zero (tile padding). */
__global__ void cw_quantize_x(const float* __restrict__ X, int8_t* __restrict__ xq,
                              float* __restrict__ sx, uint32_t cols, uint32_t ld,
                              uint32_t batch)
{
    __shared__ float red[32];
    const uint32_t b    = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const float*   xr   = X + (size_t)b * cols;

    float m = 0.0f;
    if (b < batch)
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(xr[c]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(CW_FULL_MASK, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < (blockDim.x >> 5) ? red[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(CW_FULL_MASK, m, off));
        if (lane == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];

    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    if (threadIdx.x == 0) sx[b] = m / 127.0f;
    for (uint32_t c = threadIdx.x; c < ld; c += blockDim.x) {
        int q = 0;
        if (b < batch && c < cols) q = max(-127, min(127, __float2int_rn(xr[c] * inv)));
        xq[(size_t)b * ld + c] = (int8_t)q;
    }
}

/* Triplet packing for the __dp4a path: xq3 = (xq0, xq1, xq2, 0), c3 = 127·Σ. */
__global__ void cw_pack_x_triplets(const int8_t* __restrict__ xq, int32_t* __restrict__ xq3,
                                   int32_t* __restrict__ c3, uint32_t ld, uint32_t triplets,
                                   uint32_t batch)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * triplets) return;
    const uint32_t b = (uint32_t)(i / triplets), t = (uint32_t)(i % triplets);
    const int8_t*  p = xq + (size_t)b * ld + 3u * t;
    const int a0 = p[0], a1 = p[1], a2 = p[2];
    xq3[i] = (a0 & 0xFF) | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16);
    c3[i]  = 127 * (a0 + a1 + a2);
}

/*
 * __dp4a GEMV (sm_61+, i.e. GTX 1080 Ti).  Same warp-per-row structure as
 * cw_fiver_gemv; per word: 6 shuffles for e, 2 × __dp4a per batch row.
 */
template <int LAYOUT, int BT>
__global__ void cw_fiver_gemv_dp4a(
    const uint32_t* __restrict__ W,
    const int32_t*  __restrict__ xq3,
    const int32_t*  __restrict__ c3,
    const float*    __restrict__ sx,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t words_per_row, uint32_t batch, float scale)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const int      elut = cw_e_lut[lane];

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
            e[i] = __shfl_sync(CW_FULL_MASK, elut, (int)((word >> cw_k_shift(LAYOUT, i)) & 0x1Fu));

        if (!valid) continue;

        const int E0 = (e[0] & 0xFF) | ((e[1] & 0xFF) << 8) | ((e[2] & 0xFF) << 16);
        const int E1 = (e[3] & 0xFF) | ((e[4] & 0xFF) << 8) | ((e[5] & 0xFF) << 16);
        const int n0 = -(int)((word >> cw_sign_shift(LAYOUT, 0)) & 1u);   /* 0 or −1 */
        const int n1 = -(int)((word >> cw_sign_shift(LAYOUT, 3)) & 1u);

#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const int2 xv = __ldg(reinterpret_cast<const int2*>(xq3 + (size_t)(b0 + b) * T) + wi);
            const int2 cv = __ldg(reinterpret_cast<const int2*>(c3  + (size_t)(b0 + b) * T) + wi);
            const int  v0 = __dp4a(E0, xv.x, cv.x);
            const int  v1 = __dp4a(E1, xv.y, cv.y);
            acc[b] += ((v0 ^ n0) - n0) + ((v1 ^ n1) - n1);   /* (v ^ −1) + 1 = −v */
        }
    }

#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(CW_FULL_MASK, acc[b], off);

    if (lane == 0) {
        const float f = scale * (1.0f / 127.0f);
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch)
                Y[(size_t)(b0 + b) * rows + row] = (float)acc[b] * f * sx[b0 + b];
    }
}

/*
 * INT8 Tensor-Core GEMM (sm_75+, WMMA 16×16×16 s8 → s32).  One warp per
 * block computes a 16-row × 16-batch output tile.  Per 96-column chunk the
 * warp decodes 16 rows × 16 words into two tiny shared-memory tiles
 * (A1 = σ, A2 = σ·e, 1.5 KB each) — weights in VRAM stay packed.
 * The x tile is transposed into shared memory so every 16-column sub-tile
 * starts 256-byte aligned, as load_matrix_sync requires.
 *
 * Pays off for batch ≥ 16; for batch 1 GEMV is memory-bound and the __dp4a
 * kernel is the better choice.  Compiled empty below sm_75; the launcher
 * never selects it there.
 */
template <int LAYOUT>
__global__ void cw_fiver_gemm_tc(
    const uint32_t* __restrict__ W,
    const int8_t*   __restrict__ xq,
    const float*    __restrict__ sx,
    float*          __restrict__ Y,
    uint32_t rows, uint32_t words_per_row, uint32_t batch, uint32_t ld, float scale)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    __shared__ __align__(32) signed char A1[CW_TC_KCHUNK_COLS * CW_TC_TILE];  /* col-major [c][r] */
    __shared__ __align__(32) signed char A2[CW_TC_KCHUNK_COLS * CW_TC_TILE];
    __shared__ __align__(32) signed char Bs[CW_TC_KCHUNK_COLS * CW_TC_TILE];  /* row-major [c][b] */
    __shared__ __align__(32) int         Cs[CW_TC_TILE * CW_TC_TILE];

    const uint32_t lane = threadIdx.x;   /* blockDim.x == 32 */
    const uint32_t row0 = blockIdx.x * CW_TC_TILE;
    const uint32_t b0   = blockIdx.y * CW_TC_TILE;
    const int      elut = cw_e_lut[lane];

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c1, c2;
    wmma::fill_fragment(c1, 0);
    wmma::fill_fragment(c2, 0);

    for (uint32_t kw = 0; kw < words_per_row; kw += CW_TC_KCHUNK_WORDS) {
        /* Decode 16 rows × 16 words.  r = idx % 16 keeps the byte stores
         * bank-conflict free; out-of-range words decode as k=0, σ=+1, whose
         * contribution 127·1 + e(0) = 127 − 127 = 0 vanishes. */
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t idx = j * 32u + lane;
            const uint32_t r   = idx & 15u, wc = idx >> 4;
            const uint32_t row = row0 + r,  wi = kw + wc;
            const uint32_t word = (row < rows && wi < words_per_row)
                                ? __ldg(&W[(size_t)row * words_per_row + wi]) : 0u;
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                const int      e = __shfl_sync(CW_FULL_MASK, elut,
                                               (int)((word >> cw_k_shift(LAYOUT, i)) & 0x1Fu));
                const uint32_t s = (word >> cw_sign_shift(LAYOUT, i)) & 1u;
                const uint32_t c = wc * CW_WEIGHTS_PER_WORD + (uint32_t)i;
                A1[c * CW_TC_TILE + r] = (signed char)(s ? -1 : 1);
                A2[c * CW_TC_TILE + r] = (signed char)(s ? -e : e);
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::col_major> a1, a2;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> bf;
            wmma::load_matrix_sync(a1, A1 + kk * 256, CW_TC_TILE);
            wmma::load_matrix_sync(a2, A2 + kk * 256, CW_TC_TILE);
            wmma::load_matrix_sync(bf, Bs + kk * 256, CW_TC_TILE);
            wmma::mma_sync(c1, a1, bf, c1);
            wmma::mma_sync(c2, a2, bf, c2);
        }
        __syncwarp();
    }

    for (int i = 0; i < c2.num_elements; i++) c2.x[i] += 127 * c1.x[i];
    wmma::store_matrix_sync(Cs, c2, CW_TC_TILE, wmma::mem_row_major);   /* Cs[r][b] */
    __syncwarp();

    const float f = scale * (1.0f / 127.0f);
    for (uint32_t idx = lane; idx < CW_TC_TILE * CW_TC_TILE; idx += 32u) {
        const uint32_t r = idx >> 4, b = idx & 15u;
        if (row0 + r < rows && b0 + b < batch)
            Y[(size_t)(b0 + b) * rows + row0 + r] = (float)Cs[idx] * f * sx[b0 + b];
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
 * y = W·x through the INT8 decomposition.  Quantises X into ws, then runs
 * the Tensor-Core GEMM (use_tc, sm_75+) or the __dp4a GEMV.
 * Returns 0 without launching anything if the matrix is not eligible.
 */
static int cw_fiver_gemv_int8(const FiverMatrix* m, const float* d_x, float* d_y,
                              const CwInt8Workspace* ws, int use_tc, cudaStream_t stream)
{
    if ((m->flags & CW_FLAG_HAS_ACT) || m->cols > CW_INT8_MAX_COLS) return 0;
    if (use_tc && !cw_device_has_int8_tc()) return 0;

    const uint32_t batch = ws->batch;
    cw_quantize_x<<<ws->batch_pad, 256, 0, stream>>>(d_x, ws->xq, ws->sx, m->cols, ws->ld, batch);

    if (use_tc) {
        dim3 grid((m->rows + CW_TC_TILE - 1u) / CW_TC_TILE, ws->batch_pad / CW_TC_TILE);
        if (m->layout == CW_LAYOUT_U32)
            cw_fiver_gemm_tc<CW_LAYOUT_U32><<<grid, 32, 0, stream>>>(
                m->words, ws->xq, ws->sx, d_y, m->rows, m->words_per_row, batch, ws->ld, m->scale);
        else
            cw_fiver_gemm_tc<CW_LAYOUT_U16><<<grid, 32, 0, stream>>>(
                m->words, ws->xq, ws->sx, d_y, m->rows, m->words_per_row, batch, ws->ld, m->scale);
        return 1;
    }

    const size_t n3 = (size_t)batch * ws->triplets;
    cw_pack_x_triplets<<<(unsigned)((n3 + 255u) / 256u), 256, 0, stream>>>(
        ws->xq, ws->xq3, ws->c3, ws->ld, ws->triplets, batch);

    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((m->rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS,
              (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);
    if (m->layout == CW_LAYOUT_U32)
        cw_fiver_gemv_dp4a<CW_LAYOUT_U32, CW_GEMV_BT><<<grid, block, 0, stream>>>(
            m->words, ws->xq3, ws->c3, ws->sx, d_y, m->rows, m->words_per_row, batch, m->scale);
    else
        cw_fiver_gemv_dp4a<CW_LAYOUT_U16, CW_GEMV_BT><<<grid, block, 0, stream>>>(
            m->words, ws->xq3, ws->c3, ws->sx, d_y, m->rows, m->words_per_row, batch, m->scale);
    return 1;
}


/*
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  BACKWARD — fused, on packed words (no float weight or grad matrix) │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * For one minibatch (dY already contains the activation derivative):
 *
 *   1.  dX[b,c] = s · Σ_r W[r,c] · dY[b,r]        cw_fiver_gemv_t
 *       uses the weights *before* the update, as fiver.py does.
 *
 *   2.  g_rc  = Σ_b dY[b,r] · X[b,c]              cw_fiver_ste_update
 *       k_rc ← clamp(k_rc − sign(σ_rc · lr · g_rc), 0, 31)
 *       bias_r −= lr · Σ_b dY[b,r]
 *
 * Why σ·g:  w = σ · s · f(k) with f strictly increasing and s ≥ 0, so
 * sign(∂L/∂k) = σ · sign(∂L/∂w).  Per-weight activations (sqrt, log1p,
 * softsign) are odd and increasing, so they never change that sign either.
 * Shared sign bits are not trained (same as fiver.py).
 *
 * g_rc is formed in registers and consumed immediately: the full
 * out × in gradient matrix never exists.  This requires one update per
 * minibatch (train.py: backward → step), i.e. no gradient accumulation
 * across several backward calls.  Padding columns get g = 0 and keep k = 0.
 */
#define CW_BWD_RT  4    /* rows per warp in the update kernel (reuses X loads) */

/*
 * dX = s · Wᵀ · dY.  One thread per packed column word, looping over a slice
 * of rows; partial sums are combined with atomicAdd (dX must be zeroed).
 * grid = (ceil(wpr / 128), row slices, ceil(batch / BT)).
 * All lanes run the same row loop, so the decode shuffles stay convergent.
 */
template <int LAYOUT, int BT>
__global__ void cw_fiver_gemv_t(
    const uint32_t* __restrict__ W,
    const uint32_t* __restrict__ act_bits,
    const float*    __restrict__ dY,
    float*          __restrict__ dX,
    uint32_t rows, uint32_t cols, uint32_t words_per_row,
    uint32_t batch, uint32_t rows_per_slice, float scale, uint8_t flags)
{
    const uint32_t wi    = blockIdx.x * blockDim.x + threadIdx.x;
    const bool     valid = wi < words_per_row;
    const uint32_t r0    = blockIdx.y * rows_per_slice;
    const uint32_t r1    = min(rows, r0 + rows_per_slice);
    const uint32_t b0    = blockIdx.z * (uint32_t)BT;
    const float    lut   = cw_fiver_lut[threadIdx.x & 31u];

    float acc[BT][CW_WEIGHTS_PER_WORD];
#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) acc[b][i] = 0.0f;

    for (uint32_t r = r0; r < r1; r++) {
        const uint32_t word = valid ? __ldg(&W[(size_t)r * words_per_row + wi]) : 0u;
        float w[CW_WEIGHTS_PER_WORD];
#pragma unroll
        for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
            const uint32_t k = (word >> cw_k_shift(LAYOUT, i)) & 0x1Fu;
            const uint32_t s = (word >> cw_sign_shift(LAYOUT, i)) & 1u;
            const float    v = __shfl_sync(CW_FULL_MASK, lut, (int)k);
            w[i] = __int_as_float(__float_as_int(v) ^ (int)(s << 31));
        }
        if (!valid) continue;
        if (flags & CW_FLAG_HAS_ACT) {
#pragma unroll
            for (int i = 0; i < (int)CW_WEIGHTS_PER_WORD; i++) {
                const size_t   idx  = ((size_t)r * words_per_row + wi) * CW_WEIGHTS_PER_WORD + i;
                const uint32_t code = (__ldg(&act_bits[idx >> 4]) >> ((idx & 15u) * 2u)) & 3u;
                w[i] = cw_apply_act(w[i], code);
            }
        }
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const float g = __ldg(&dY[(size_t)(b0 + b) * rows + r]);   /* warp-uniform */
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
            if (c < cols) atomicAdd(&dX[(size_t)(b0 + b) * cols + c], acc[b][i] * scale);
        }
    }
}

/*
 * Fused STE update.  One warp handles CW_BWD_RT rows; each lane owns one
 * packed word column of those rows and accumulates 6 × RT gradients in
 * registers, so every X load is reused RT times.  No shuffles here: only
 * the k bits are touched, the word is rewritten in place if a k changed.
 */
template <int LAYOUT>
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
                const uint32_t sh = cw_k_shift(LAYOUT, i);
                const uint32_t k  = (word >> sh) & 0x1Fu;
                const float    d  = (((word >> cw_sign_shift(LAYOUT, i)) & 1u) ? -g[q][i] : g[q][i]) * lr;
                const uint32_t kn = (d > 0.0f) ? (k > 0u ? k - 1u : 0u)
                                  : (d < 0.0f) ? (k < 31u ? k + 1u : 31u) : k;
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
        if (m->layout == CW_LAYOUT_U32)
            cw_fiver_gemv_t<CW_LAYOUT_U32, CW_GEMV_BT><<<grid, 128, 0, stream>>>(
                m->words, m->act_bits, d_dy, d_dx, m->rows, m->cols, m->words_per_row,
                batch, rows_per_slice, m->scale, m->flags);
        else
            cw_fiver_gemv_t<CW_LAYOUT_U16, CW_GEMV_BT><<<grid, 128, 0, stream>>>(
                m->words, m->act_bits, d_dy, d_dx, m->rows, m->cols, m->words_per_row,
                batch, rows_per_slice, m->scale, m->flags);
    }

    const uint32_t warps = 4u, rows_per_block = warps * CW_BWD_RT;
    dim3 grid((m->rows + rows_per_block - 1u) / rows_per_block);
    if (m->layout == CW_LAYOUT_U32)
        cw_fiver_ste_update<CW_LAYOUT_U32><<<grid, 32u * warps, 0, stream>>>(
            m->words, d_x, d_dy, d_bias, m->rows, m->cols, m->words_per_row, batch, lr);
    else
        cw_fiver_ste_update<CW_LAYOUT_U16><<<grid, 32u * warps, 0, stream>>>(
            m->words, d_x, d_dy, d_bias, m->rows, m->cols, m->words_per_row, batch, lr);
}


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                    TEST / BENCHMARK  BLOCK                          ║
 * ║  host pack/decode · rsqrt bench · GEMV test · GEMV bench · main     ║
 * ║  (--bench-a | --bench-b | --bench-gemv | --bench-bwd | --test)      ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

static uint32_t cw_rng_state = 0x12345678u;
static uint32_t cw_rand(void) {               /* xorshift32, deterministic */
    cw_rng_state ^= cw_rng_state << 13;
    cw_rng_state ^= cw_rng_state >> 17;
    cw_rng_state ^= cw_rng_state << 5;
    return cw_rng_state;
}

/* Pack 6 k-values + two half-signs into one word of the given layout. */
static uint32_t cw_pack_word(int layout, const uint8_t k[6], uint32_t s_lo, uint32_t s_hi) {
    uint32_t w = 0;
    for (int i = 0; i < 6; i++) w |= ((uint32_t)k[i] & 0x1Fu) << cw_k_shift(layout, i);
    w |= (s_lo & 1u) << cw_sign_shift(layout, 0);
    w |= (s_hi & 1u) << cw_sign_shift(layout, 3);
    return w;
}

/* Host decode of one packed weight (row-local padded column c). */
static float cw_decode_cpu(int layout, const uint32_t* row_words, uint32_t c) {
    uint32_t word = row_words[c / 6u];
    int      i    = (int)(c % 6u);
    uint8_t  k    = (uint8_t)((word >> cw_k_shift(layout, i)) & 0x1Fu);
    uint32_t s    = (word >> cw_sign_shift(layout, i)) & 1u;
    float    v    = cw_fiver_to_float_cpu(k);
    return s ? -v : v;
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

/* Random packed matrix on host: random k, random half-signs, optional act codes. */
static void make_random_matrix(int layout, uint32_t rows, uint32_t cols, int with_act,
                               uint32_t** words_out, uint32_t** act_out) {
    uint32_t wpr    = cw_words_per_row(cols);
    uint32_t padded = wpr * CW_WEIGHTS_PER_WORD;
    uint32_t* words = (uint32_t*)malloc((size_t)rows * wpr * sizeof(uint32_t));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t wi = 0; wi < wpr; wi++) {
            uint8_t k[6];
            for (int i = 0; i < 6; i++) {
                uint32_t c = wi * 6u + (uint32_t)i;
                k[i] = (c < cols) ? (uint8_t)(cw_rand() & 0x1Fu) : 0u;
            }
            uint32_t s_lo = cw_rand() & 1u, s_hi = cw_rand() & 1u;
            if (wi * 6u + 3u >= cols) s_hi = 0u;    /* padded half carries no sign */
            words[(size_t)r * wpr + wi] = cw_pack_word(layout, k, s_lo, s_hi);
        }
    }
    *words_out = words;
    *act_out   = NULL;
    if (with_act) {
        size_t n_act = ((size_t)rows * padded + 15u) / 16u;
        uint32_t* act = (uint32_t*)malloc(n_act * sizeof(uint32_t));
        for (size_t i = 0; i < n_act; i++) act[i] = cw_rand();
        *act_out = act;
    }
}

static int run_gemv_case(int layout, uint32_t rows, uint32_t cols, uint32_t batch,
                         int with_act) {
    uint32_t wpr    = cw_words_per_row(cols);
    uint32_t padded = wpr * CW_WEIGHTS_PER_WORD;
    uint32_t *h_words, *h_act;
    make_random_matrix(layout, rows, cols, with_act, &h_words, &h_act);

    float* h_x   = (float*)malloc((size_t)batch * cols * sizeof(float));
    float* h_y   = (float*)malloc((size_t)batch * rows * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++)
        h_x[i] = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;
    const float scale = 0.0371f;

    uint32_t *d_words, *d_act = NULL;
    float *d_x, *d_y;
    size_t n_words = (size_t)rows * wpr;
    size_t n_act   = ((size_t)rows * padded + 15u) / 16u;
    CW_CUDA_CHECK(cudaMalloc(&d_words, n_words * sizeof(uint32_t)));
    CW_CUDA_CHECK(cudaMalloc(&d_x, (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));
    if (with_act) {
        CW_CUDA_CHECK(cudaMalloc(&d_act, n_act * sizeof(uint32_t)));
        CW_CUDA_CHECK(cudaMemcpy(d_act, h_act, n_act * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    FiverMatrix m = { d_words, d_act, rows, cols, wpr, scale, layout,
                      (uint8_t)(with_act ? CW_FLAG_HAS_ACT : 0u) };
    cw_fiver_gemv_launch(&m, d_x, d_y, batch, 0);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    CW_CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)batch * rows * sizeof(float), cudaMemcpyDeviceToHost));

    double worst = 0.0;
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t* rw = h_words + (size_t)r * wpr;
            double ref = 0.0, mag = 0.0;
            for (uint32_t c = 0; c < cols; c++) {
                float w = cw_decode_cpu(layout, rw, c);
                if (with_act) {
                    size_t idx = (size_t)r * padded + c;
                    w = cw_apply_act(w, (h_act[idx >> 4] >> ((idx & 15u) * 2u)) & 3u);
                }
                double xv = h_x[(size_t)b * cols + c];
                ref += (double)w * xv;
                mag += fabs((double)w * xv);
            }
            ref *= scale; mag *= scale;
            double err = fabs((double)h_y[(size_t)b * rows + r] - ref) / (mag + 1e-12);
            if (err > worst) worst = err;
        }
    }
    int ok = worst < 1e-5;
    printf("  layout=%s  %4ux%-5u  batch=%u  act=%d   max_rel_err=%.2e  %s\n",
           layout == CW_LAYOUT_U32 ? "u32" : "u16", rows, cols, batch, with_act,
           worst, ok ? "ok" : "FAIL");

    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_y));
    if (d_act) CW_CUDA_WARN(cudaFree(d_act));
    free(h_words); free(h_act); free(h_x); free(h_y);
    return ok;
}

/*
 * INT8 path check.  The device-quantised x (xq, sx) is read back and the
 * integer sum is recomputed on the host — it must match exactly (tolerance
 * below one integer unit).  The deviation from the exact float result
 * (quantisation error of x and d) is reported separately.
 */
static int run_int8_case(int layout, uint32_t rows, uint32_t cols, uint32_t batch, int use_tc) {
    uint32_t wpr = cw_words_per_row(cols);
    uint32_t *h_words, *h_act;
    make_random_matrix(layout, rows, cols, 0, &h_words, &h_act);

    float* h_x = (float*)malloc((size_t)batch * cols * sizeof(float));
    float* h_y = (float*)malloc((size_t)batch * rows * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++)
        h_x[i] = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;
    for (uint32_t c = 0; c < cols; c++) h_x[c] = 0.0f;   /* row 0 all-zero: sx = 0 edge case */
    const float scale = 0.0371f;

    uint32_t* d_words; float *d_x, *d_y;
    size_t n_words = (size_t)rows * wpr;
    CW_CUDA_CHECK(cudaMalloc(&d_words, n_words * sizeof(uint32_t)));
    CW_CUDA_CHECK(cudaMalloc(&d_x, (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));

    CwInt8Workspace ws;
    cw_int8_workspace_alloc(&ws, cols, batch);
    FiverMatrix m = { d_words, NULL, rows, cols, wpr, scale, layout, 0 };
    int ok = cw_fiver_gemv_int8(&m, d_x, d_y, &ws, use_tc, 0);
    if (ok) {
        CW_CUDA_CHECK(cudaGetLastError());
        CW_CUDA_CHECK(cudaDeviceSynchronize());
        int8_t* h_xq = (int8_t*)malloc((size_t)ws.batch_pad * ws.ld);
        float*  h_sx = (float*)malloc((size_t)ws.batch_pad * sizeof(float));
        CW_CUDA_CHECK(cudaMemcpy(h_y,  d_y,  (size_t)batch * rows * sizeof(float), cudaMemcpyDeviceToHost));
        CW_CUDA_CHECK(cudaMemcpy(h_xq, ws.xq, (size_t)ws.batch_pad * ws.ld, cudaMemcpyDeviceToHost));
        CW_CUDA_CHECK(cudaMemcpy(h_sx, ws.sx, (size_t)ws.batch_pad * sizeof(float), cudaMemcpyDeviceToHost));

        double worst_exact = 0.0, max_qerr = 0.0, max_ref = 0.0;
        for (uint32_t b = 0; b < batch; b++) {
            const int8_t* xqr = h_xq + (size_t)b * ws.ld;
            for (uint32_t r = 0; r < rows; r++) {
                const uint32_t* rw = h_words + (size_t)r * wpr;
                long long acc = 0;
                for (uint32_t t = 0; t < 2u * wpr; t++) {
                    const uint32_t word = rw[t >> 1];
                    const int      h    = (int)(t & 1u) * 3;
                    long long dot = 0, x3 = 0;
                    for (int i = 0; i < 3; i++) {
                        const uint32_t k = (word >> cw_k_shift(layout, h + i)) & 0x1Fu;
                        const int      xv = xqr[3u * t + (uint32_t)i];
                        dot += (long long)cw_e_lut_cpu[k] * xv;
                        x3  += xv;
                    }
                    const long long v = 127 * x3 + dot;
                    acc += ((word >> cw_sign_shift(layout, h)) & 1u) ? -v : v;
                }
                const double unit = (double)scale * h_sx[b] / 127.0;
                const double yref = (double)acc * unit;
                const double y    = h_y[(size_t)b * rows + r];
                const double ex   = fabs(y - yref) / (1e-6 * fabs(yref) + 0.25 * unit + 1e-30);
                if (ex > worst_exact) worst_exact = ex;

                double yf = 0.0;
                for (uint32_t c = 0; c < cols; c++)
                    yf += (double)cw_decode_cpu(layout, rw, c) * h_x[(size_t)b * cols + c];
                yf *= scale;
                if (fabs(y - yf) > max_qerr) max_qerr = fabs(y - yf);
                if (fabs(yf) > max_ref)      max_ref  = fabs(yf);
            }
        }
        ok = worst_exact <= 1.0;
        printf("  %-5s layout=%s  %4ux%-5u  batch=%-3u  int-exact=%s  quant_err=%.2e (rel to max|y|)\n",
               use_tc ? "tc" : "dp4a", layout == CW_LAYOUT_U32 ? "u32" : "u16",
               rows, cols, batch, ok ? "ok  " : "FAIL", max_qerr / (max_ref + 1e-30));
        free(h_xq); free(h_sx);
    } else {
        printf("  %-5s layout=%s  %4ux%-5u  batch=%-3u  skipped (not supported here)\n",
               use_tc ? "tc" : "dp4a", layout == CW_LAYOUT_U32 ? "u32" : "u16", rows, cols, batch);
        ok = 1;
    }

    cw_int8_workspace_free(&ws);
    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_y));
    free(h_words); free(h_x); free(h_y);
    return ok;
}

/*
 * Fused backward check against a double-precision host reference.
 * A k mismatch only counts as an error when |g| is clearly non-zero
 * (|g| > 1e-5 · Σ|dY·X|); below that, float summation order may legitimately
 * flip the sign.  Sign bits and padding k must never change.
 */
static int run_backward_case(int layout, uint32_t rows, uint32_t cols, uint32_t batch,
                             int with_act, float lr) {
    uint32_t wpr = cw_words_per_row(cols), padded = wpr * CW_WEIGHTS_PER_WORD;
    uint32_t *h_words, *h_act;
    make_random_matrix(layout, rows, cols, with_act, &h_words, &h_act);
    const size_t n_words = (size_t)rows * wpr, n_act = ((size_t)rows * padded + 15u) / 16u;

    float* h_x  = (float*)malloc((size_t)batch * cols * sizeof(float));
    float* h_dy = (float*)malloc((size_t)batch * rows * sizeof(float));
    float* h_b  = (float*)malloc(rows * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++) h_x[i]  = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;
    for (size_t i = 0; i < (size_t)batch * rows; i++) h_dy[i] = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;
    for (uint32_t r = 0; r < rows; r++) h_b[r] = ((float)(cw_rand() & 0xFFFFu) / 65536.0f);
    for (uint32_t b = 0; b < batch; b++) h_dy[(size_t)b * rows] = 0.0f;   /* row 0: g = 0 */
    const float scale = 0.0371f;

    uint32_t *d_words, *d_act = NULL;
    float *d_x, *d_dy, *d_dx, *d_b;
    CW_CUDA_CHECK(cudaMalloc(&d_words, n_words * sizeof(uint32_t)));
    CW_CUDA_CHECK(cudaMalloc(&d_x,  (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_dy, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_dx, (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_b,  rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x,  h_x,  (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_dy, h_dy, (size_t)batch * rows * sizeof(float), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_b,  h_b,  rows * sizeof(float), cudaMemcpyHostToDevice));
    if (with_act) {
        CW_CUDA_CHECK(cudaMalloc(&d_act, n_act * sizeof(uint32_t)));
        CW_CUDA_CHECK(cudaMemcpy(d_act, h_act, n_act * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    FiverMatrix m = { d_words, d_act, rows, cols, wpr, scale, layout,
                      (uint8_t)(with_act ? CW_FLAG_HAS_ACT : 0u) };
    cw_fiver_backward_step(&m, d_x, d_dy, d_dx, d_b, batch, lr, 0);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* n_w  = (uint32_t*)malloc(n_words * sizeof(uint32_t));
    float*    g_dx = (float*)malloc((size_t)batch * cols * sizeof(float));
    float*    n_b  = (float*)malloc(rows * sizeof(float));
    CW_CUDA_CHECK(cudaMemcpy(n_w,  d_words, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CW_CUDA_CHECK(cudaMemcpy(g_dx, d_dx, (size_t)batch * cols * sizeof(float), cudaMemcpyDeviceToHost));
    CW_CUDA_CHECK(cudaMemcpy(n_b,  d_b,  rows * sizeof(float), cudaMemcpyDeviceToHost));

    /* dX with the old weights */
    double worst_dx = 0.0;
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t c = 0; c < cols; c++) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t r = 0; r < rows; r++) {
                float w = cw_decode_cpu(layout, h_words + (size_t)r * wpr, c);
                if (with_act) {
                    size_t idx = (size_t)r * padded + c;
                    w = cw_apply_act(w, (h_act[idx >> 4] >> ((idx & 15u) * 2u)) & 3u);
                }
                ref += (double)w * h_dy[(size_t)b * rows + r];
                mag += fabs((double)w * h_dy[(size_t)b * rows + r]);
            }
            double e = fabs(g_dx[(size_t)b * cols + c] - scale * ref) / (scale * mag + 1e-30);
            if (e > worst_dx) worst_dx = e;
        }

    /* k update, sign bits, padding */
    uint32_t k_err = 0, k_moved = 0, k_ambig = 0, other_bits_err = 0;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t wi = 0; wi < wpr; wi++) {
            const uint32_t ow = h_words[(size_t)r * wpr + wi], nw = n_w[(size_t)r * wpr + wi];
            uint32_t kmask = 0;
            for (int i = 0; i < 6; i++) {
                const uint32_t sh = cw_k_shift(layout, i), c = wi * 6u + (uint32_t)i;
                kmask |= 0x1Fu << sh;
                const uint32_t k_old = (ow >> sh) & 0x1Fu, k_new = (nw >> sh) & 0x1Fu;
                double g = 0.0, mag = 0.0;
                if (c < cols)
                    for (uint32_t b = 0; b < batch; b++) {
                        g   += (double)h_dy[(size_t)b * rows + r] * h_x[(size_t)b * cols + c];
                        mag += fabs((double)h_dy[(size_t)b * rows + r] * h_x[(size_t)b * cols + c]);
                    }
                const double d = (((ow >> cw_sign_shift(layout, i)) & 1u) ? -g : g) * lr;
                uint32_t k_exp = k_old;
                if (d > 0.0 && k_old > 0u)  k_exp = k_old - 1u;
                if (d < 0.0 && k_old < 31u) k_exp = k_old + 1u;
                if (k_new != k_old) k_moved++;
                if (k_new != k_exp) {
                    if (c < cols && fabs(g) <= 1e-5 * mag) k_ambig++;
                    else k_err++;
                }
            }
            if ((ow & ~kmask) != (nw & ~kmask)) other_bits_err++;
        }

    double worst_b = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        double gs = 0.0;
        for (uint32_t b = 0; b < batch; b++) gs += h_dy[(size_t)b * rows + r];
        double e = fabs(n_b[r] - (h_b[r] - (double)lr * gs));
        if (e > worst_b) worst_b = e;
    }

    const int ok = worst_dx < 1e-5 && k_err == 0 && other_bits_err == 0 && worst_b < 1e-5;
    printf("  layout=%s  %4ux%-5u  batch=%-3u act=%d lr=%+.2f  dX_err=%.1e  k moved=%u/%u"
           " wrong=%u ambiguous=%u  sign/pad bits changed=%u  bias_err=%.1e  %s\n",
           layout == CW_LAYOUT_U32 ? "u32" : "u16", rows, cols, batch, with_act, lr,
           worst_dx, k_moved, rows * padded, k_err, k_ambig, other_bits_err, worst_b,
           ok ? "ok" : "FAIL");

    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_dy));
    CW_CUDA_WARN(cudaFree(d_dx));    CW_CUDA_WARN(cudaFree(d_b));
    if (d_act) CW_CUDA_WARN(cudaFree(d_act));
    free(h_words); free(h_act); free(h_x); free(h_dy); free(h_b); free(n_w); free(g_dx); free(n_b);
    return ok;
}

static int run_fiver_test(void) {
    int failures = 0;

    /* 1. LUT table and shuffle decode vs. independent formula path */
    float h_lut[32], *d_out;
    CW_CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));
    cw_lut_probe<<<1, 32>>>(d_out);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaMemcpy(h_lut, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));
    CW_CUDA_WARN(cudaFree(d_out));

    uint32_t lut_err = 0;
    for (uint32_t k = 0; k < 32; k++)
        if (fabsf(h_lut[k] - cw_fiver_to_float_cpu((uint8_t)k)) > 1e-6f) lut_err++;
    printf("\n[test] shuffle LUT decode:   %u / 32 errors\n", lut_err);
    failures += lut_err != 0;

    /* 2. Fused GEMV vs. host double-precision reference.
     *    Odd shapes exercise row padding (cols % 6 != 0) and batch tail (% BT). */
    printf("[test] fused GEMV vs host reference:\n");
    const uint32_t shapes[][3] = { {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 4096, 9} };
    for (int layout = 0; layout <= 1; layout++)
        for (int s = 0; s < 4; s++)
            for (int act = 0; act <= 1; act++)
                failures += !run_gemv_case(layout, shapes[s][0], shapes[s][1], shapes[s][2], act);

    /* 3. INT8 decomposition: __dp4a GEMV and Tensor-Core GEMM. */
    printf("[test] INT8 path (Σx-term + int8 dot) vs exact integer host reference:\n");
    const uint32_t ishapes[][3] = { {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 4096, 9},
                                    {17, 97, 20}, {130, 4096, 33} };
    for (int layout = 0; layout <= 1; layout++)
        for (int s = 0; s < 6; s++)
            failures += !run_int8_case(layout, ishapes[s][0], ishapes[s][1], ishapes[s][2], 0);
    if (cw_device_has_int8_tc()) {
        for (int layout = 0; layout <= 1; layout++)
            for (int s = 0; s < 6; s++)
                failures += !run_int8_case(layout, ishapes[s][0], ishapes[s][1], ishapes[s][2], 1);
    } else {
        printf("  tc    skipped — this GPU has no INT8 Tensor Cores (needs sm_75+)\n");
    }

    /* 4. Fused backward: dX + in-place STE update of the packed k values. */
    printf("[test] fused backward (dX, STE update on packed words, bias):\n");
    const uint32_t bshapes[][3] = { {37, 1001, 5}, {3, 7, 4}, {130, 1024, 33}, {9, 6, 1} };
    for (int layout = 0; layout <= 1; layout++)
        for (int s = 0; s < 4; s++)
            failures += !run_backward_case(layout, bshapes[s][0], bshapes[s][1], bshapes[s][2],
                                           s == 0, s == 1 ? -0.05f : 0.05f);

    /* Matrices with per-weight activations must be refused by the INT8 path. */
    {
        FiverMatrix m = { NULL, NULL, 1, 6, 1, 1.0f, CW_LAYOUT_U16, CW_FLAG_HAS_ACT };
        CwInt8Workspace ws = { NULL, NULL, NULL, NULL, 96, 1, 16, 2 };
        int refused = !cw_fiver_gemv_int8(&m, NULL, NULL, &ws, 0, 0);
        printf("  act-flag matrix refused by INT8 path: %s\n", refused ? "ok" : "FAIL");
        failures += !refused;
    }

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
    float* a = (float*)malloc(n * sizeof(float));
    float* r = (float*)malloc(n * sizeof(float));
    CW_CUDA_CHECK(cudaMemcpy(a, d_a,   n * sizeof(float), cudaMemcpyDeviceToHost));
    CW_CUDA_CHECK(cudaMemcpy(r, d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
    double md = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (fabs((double)a[i] - r[i]) > md) md = fabs((double)a[i] - r[i]);
        if (fabs((double)r[i]) > mr)        mr = fabs((double)r[i]);
    }
    free(a); free(r);
    return md / (mr + 1e-30);
}

/*
 * Benchmark all GEMV paths on one shape (u32 layout).  Time includes the
 * x quantisation for the INT8 paths.  "rel err" is measured against the
 * float Fiver path, which uses the exact decoded weights.
 */
static void run_gemv_benchmark(uint32_t rows, uint32_t cols, uint32_t batch, int iterations) {
    uint32_t wpr = cw_words_per_row(cols);
    uint32_t *h_words, *h_act;
    make_random_matrix(CW_LAYOUT_U32, rows, cols, 0, &h_words, &h_act);

    float* h_wf = (float*)malloc((size_t)rows * cols * sizeof(float));
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            h_wf[(size_t)r * cols + c] = cw_decode_cpu(CW_LAYOUT_U32, h_words + (size_t)r * wpr, c);
    float* h_x = (float*)malloc((size_t)batch * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++)
        h_x[i] = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;

    const size_t ny          = (size_t)batch * rows;
    const size_t bytes_fiver = (size_t)rows * wpr * sizeof(uint32_t);
    const size_t bytes_f32   = (size_t)rows * cols * sizeof(float);
    uint32_t* d_words; float *d_wf, *d_x, *d_yf, *d_y32, *d_yd, *d_yt;
    CW_CUDA_CHECK(cudaMalloc(&d_words, bytes_fiver));
    CW_CUDA_CHECK(cudaMalloc(&d_wf,    bytes_f32));
    CW_CUDA_CHECK(cudaMalloc(&d_x,  (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_yf,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y32, ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_yd,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_yt,  ny * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, bytes_fiver, cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_wf,    h_wf,    bytes_f32,   cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));

    FiverMatrix m = { d_words, NULL, rows, cols, wpr, 1.0f, CW_LAYOUT_U32, 0 };
    CwInt8Workspace ws;
    cw_int8_workspace_alloc(&ws, cols, batch);
    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS, (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);

    const double t32 = cw_time_ms([&] {
        cw_f32_gemv<CW_GEMV_BT><<<grid, block>>>(d_wf, d_x, d_y32, rows, cols, batch); }, iterations);
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
    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_wf)); CW_CUDA_WARN(cudaFree(d_x));
    CW_CUDA_WARN(cudaFree(d_yf));    CW_CUDA_WARN(cudaFree(d_y32));
    CW_CUDA_WARN(cudaFree(d_yd));    CW_CUDA_WARN(cudaFree(d_yt));
    free(h_words); free(h_wf); free(h_x);
}

/* Fused backward step timing (dX + STE update), u32 layout. */
static void run_backward_benchmark(uint32_t rows, uint32_t cols, uint32_t batch, int iterations) {
    uint32_t wpr = cw_words_per_row(cols);
    uint32_t *h_words, *h_act;
    make_random_matrix(CW_LAYOUT_U32, rows, cols, 0, &h_words, &h_act);
    float* h_x  = (float*)malloc((size_t)batch * cols * sizeof(float));
    float* h_dy = (float*)malloc((size_t)batch * rows * sizeof(float));
    for (size_t i = 0; i < (size_t)batch * cols; i++) h_x[i]  = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;
    for (size_t i = 0; i < (size_t)batch * rows; i++) h_dy[i] = ((float)(cw_rand() & 0xFFFFu) / 32768.0f) - 1.0f;

    const size_t bytes_fiver = (size_t)rows * wpr * sizeof(uint32_t);
    uint32_t* d_words; float *d_x, *d_dy, *d_dx, *d_b;
    CW_CUDA_CHECK(cudaMalloc(&d_words, bytes_fiver));
    CW_CUDA_CHECK(cudaMalloc(&d_x,  (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_dy, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_dx, (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_b,  rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, bytes_fiver, cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x,  h_x,  (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_dy, h_dy, (size_t)batch * rows * sizeof(float), cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemset(d_b, 0, rows * sizeof(float)));

    FiverMatrix m = { d_words, NULL, rows, cols, wpr, 1.0f, CW_LAYOUT_U32, 0 };
    const double t_full = cw_time_ms([&] {
        cw_fiver_backward_step(&m, d_x, d_dy, d_dx, d_b, batch, 0.01f, 0); }, iterations);
    const double t_upd  = cw_time_ms([&] {
        cw_fiver_backward_step(&m, d_x, d_dy, NULL, d_b, batch, 0.01f, 0); }, iterations);
    const double fma = (double)rows * cols * batch;

    printf("Backward %ux%u  batch=%u  (%d iterations)\n", rows, cols, batch, iterations);
    printf("  dX + STE update  %9.4f ms   %7.1f GFMA/s\n", t_full, 2.0 * fma / (t_full * 1e6));
    printf("  STE update only  %9.4f ms   %7.1f GFMA/s\n", t_upd, fma / (t_upd * 1e6));
    printf("  VRAM not needed: float W %.1f MB + float grad matrix %.1f MB (packed W: %.1f MB)\n\n",
           (double)rows * cols * 4 / 1e6, (double)rows * cols * 4 / 1e6, bytes_fiver / 1e6);

    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_x)); CW_CUDA_WARN(cudaFree(d_dy));
    CW_CUDA_WARN(cudaFree(d_dx));    CW_CUDA_WARN(cudaFree(d_b));
    free(h_words); free(h_x); free(h_dy);
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
    printf("compact_weights — Symmetric Fiver Encoding\n");
    printf("===========================================\n\n");
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
