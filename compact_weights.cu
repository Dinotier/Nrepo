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
    const uint32_t* words;      /* [rows * words_per_row]                       */
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
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                    TEST / BENCHMARK  BLOCK                          ║
 * ║  host pack/decode · rsqrt bench · GEMV test · GEMV bench · main     ║
 * ║  (--bench-a | --bench-b | --bench-gemv | --test)                    ║
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

    printf("TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0;
}

/*
 * GEMV benchmark: packed Fiver (u32 layout) vs. FP32 weights, same kernel shape.
 * Reports time and effective weight bandwidth (bytes of W actually read).
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

    uint32_t* d_words; float *d_wf, *d_x, *d_y1, *d_y2;
    size_t bytes_fiver = (size_t)rows * wpr * sizeof(uint32_t);
    size_t bytes_f32   = (size_t)rows * cols * sizeof(float);
    CW_CUDA_CHECK(cudaMalloc(&d_words, bytes_fiver));
    CW_CUDA_CHECK(cudaMalloc(&d_wf,    bytes_f32));
    CW_CUDA_CHECK(cudaMalloc(&d_x,  (size_t)batch * cols * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y1, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_y2, (size_t)batch * rows * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_words, h_words, bytes_fiver, cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_wf,    h_wf,    bytes_f32,   cudaMemcpyHostToDevice));
    CW_CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)batch * cols * sizeof(float), cudaMemcpyHostToDevice));

    FiverMatrix m = { d_words, NULL, rows, cols, wpr, 1.0f, CW_LAYOUT_U32, 0 };
    dim3 block(32u * CW_GEMV_WARPS);
    dim3 grid((rows + CW_GEMV_WARPS - 1u) / CW_GEMV_WARPS, (batch + CW_GEMV_BT - 1u) / CW_GEMV_BT);

    cudaEvent_t t0, t1;
    CW_CUDA_CHECK(cudaEventCreate(&t0));
    CW_CUDA_CHECK(cudaEventCreate(&t1));
    float ms_fiver = 0.0f, ms_f32 = 0.0f;

    cw_fiver_gemv_launch(&m, d_x, d_y1, batch, 0);
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    CW_CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iterations; it++) cw_fiver_gemv_launch(&m, d_x, d_y1, batch, 0);
    CW_CUDA_CHECK(cudaEventRecord(t1));
    CW_CUDA_CHECK(cudaEventSynchronize(t1));
    CW_CUDA_CHECK(cudaEventElapsedTime(&ms_fiver, t0, t1));

    cw_f32_gemv<CW_GEMV_BT><<<grid, block>>>(d_wf, d_x, d_y2, rows, cols, batch);
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    CW_CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iterations; it++)
        cw_f32_gemv<CW_GEMV_BT><<<grid, block>>>(d_wf, d_x, d_y2, rows, cols, batch);
    CW_CUDA_CHECK(cudaEventRecord(t1));
    CW_CUDA_CHECK(cudaEventSynchronize(t1));
    CW_CUDA_CHECK(cudaEventElapsedTime(&ms_f32, t0, t1));

    float* h_y1 = (float*)malloc((size_t)batch * rows * sizeof(float));
    float* h_y2 = (float*)malloc((size_t)batch * rows * sizeof(float));
    CW_CUDA_CHECK(cudaMemcpy(h_y1, d_y1, (size_t)batch * rows * sizeof(float), cudaMemcpyDeviceToHost));
    CW_CUDA_CHECK(cudaMemcpy(h_y2, d_y2, (size_t)batch * rows * sizeof(float), cudaMemcpyDeviceToHost));
    double max_diff = 0.0;
    for (size_t i = 0; i < (size_t)batch * rows; i++) {
        double d = fabs((double)h_y1[i] - (double)h_y2[i]);
        if (d > max_diff) max_diff = d;
    }

    double t_f = ms_fiver / iterations, t_r = ms_f32 / iterations;
    printf("GEMV %ux%u batch=%u  (%d iterations)\n", rows, cols, batch, iterations);
    printf("  %-14s %9.4f ms   W=%7.2f MB   %7.1f GB/s   %7.1f Gweights/s\n",
           "Fiver u32", t_f, bytes_fiver / 1e6, bytes_fiver / (t_f * 1e6),
           (double)rows * cols / (t_f * 1e6));
    printf("  %-14s %9.4f ms   W=%7.2f MB   %7.1f GB/s   %7.1f Gweights/s\n",
           "FP32", t_r, bytes_f32 / 1e6, bytes_f32 / (t_r * 1e6),
           (double)rows * cols / (t_r * 1e6));
    printf("  speedup %.2fx   max |y_fiver - y_fp32| = %.2e\n\n", t_r / t_f, max_diff);

    CW_CUDA_WARN(cudaEventDestroy(t0)); CW_CUDA_WARN(cudaEventDestroy(t1));
    CW_CUDA_WARN(cudaFree(d_words)); CW_CUDA_WARN(cudaFree(d_wf)); CW_CUDA_WARN(cudaFree(d_x));
    CW_CUDA_WARN(cudaFree(d_y1));    CW_CUDA_WARN(cudaFree(d_y2));
    free(h_words); free(h_wf); free(h_x); free(h_y1); free(h_y2);
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
    printf("[device]  tensor cores: %s — Fiver GEMV runs on the FP32 FMA path\n\n",
           prop.major >= 7 ? "yes (unused so far)" : "none on this GPU");
}

int main(int argc, char** argv) {
    printf("compact_weights — Symmetric Fiver Encoding\n");
    printf("===========================================\n\n");
    print_device_info();

    int do_a = 0, do_b = 0, do_g = 0, do_t = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench-a")    == 0) do_a = 1;
        if (strcmp(argv[i], "--bench-b")    == 0) do_b = 1;
        if (strcmp(argv[i], "--bench-gemv") == 0) do_g = 1;
        if (strcmp(argv[i], "--test")       == 0) do_t = 1;
    }
    if (!do_a && !do_b && !do_g && !do_t) { do_a = do_b = do_g = do_t = 1; }

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
        run_gemv_benchmark(4096, 4096, 1, 50);
        run_gemv_benchmark(4096, 4096, 4, 50);
    }

    int ok = 1;
    if (do_t) ok = run_fiver_test();

    return ok ? 0 : 1;
}
