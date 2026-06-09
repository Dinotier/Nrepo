/*
 * compact_weights.cu — Symmetric Fiver Weight Format, CUDA Implementation
 *
 * Encoding: k in [0..31], 5-bit unsigned, MSB = direction
 *   k in [ 0..15] (bit4=0): w = 1 - 2^(-k/4)      range [0.0  .. 0.926]
 *   k in [16..31] (bit4=1): w = 1 + 2^(-(31-k)/4)  range [1.074 .. 2.0 ]
 *
 * Values are monotonically increasing with k.
 * Distribution is sigmoid-shaped around 1.0 (dense near center, sparse at extremes).
 * No separate formula_bits array — direction is implicit in the MSB of k.
 *
 * MIT License — Dinotier 2026
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
 * ║  FiverArray struct · constants · 16-entry LUT · helper macros       ║
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
#define CW_FLAG_HAS_SIGN     (1u << 0)
#define CW_FLAG_HAS_ACT      (1u << 1)

/* Activation codes (2 bits per weight in act_bits) */
#define CW_ACT_IDENTITY      0u
#define CW_ACT_SQRT          1u
#define CW_ACT_LOG           2u
#define CW_ACT_SOFTSIGN      3u

/*
 * FiverArray — compact weight storage
 *
 * One byte per weight (upper 3 bits unused).
 * Bits [4:0] of each byte hold k in [0..31].
 *
 * CUDA optimization note:
 *   fiver_data loaded via __ldg() (read-only L1 texture cache).
 *   At 1 byte/weight, 128 weights fit in two 64-byte cache lines.
 */
typedef struct {
    uint8_t*  fiver_data;   /* [count]      — k values, one byte each         */
    uint8_t*  sign_bits;    /* [ceil(count/8)] — optional, byte-packed         */
    uint32_t* act_bits;     /* [ceil(count/16)]— optional, 2 bits per weight   */
    uint32_t  count;        /* total number of weights                         */
    uint8_t   flags;        /* CW_FLAG_HAS_SIGN | CW_FLAG_HAS_ACT             */
} FiverArray;

static inline uint32_t cw_num_sign_bytes(const FiverArray* fa) {
    return (fa->count + 7u) / 8u;
}
static inline uint32_t cw_num_act_words(const FiverArray* fa) {
    return (fa->count + 15u) / 16u;
}

/*
 * 16-entry constant LUT:  delta_lut[m] = 2^(-m/4),  m = 0..15
 *
 * Sigmoid-like weight distribution visualized:
 *   k=  0: delta=1.000  w=0.000   k=16: delta=LUT[15]≈0.074  w≈1.074
 *   k=  4: delta=0.500  w=0.500   k=20: delta=LUT[11]≈0.149  w≈1.149
 *   k=  8: delta=0.250  w=0.750   k=23: delta=LUT[ 8]=0.250  w= 1.250
 *   k= 12: delta=0.125  w=0.875   k=27: delta=LUT[ 4]=0.500  w= 1.500
 *   k= 15: delta≈0.074  w≈0.926   k=31: delta=LUT[ 0]=1.000  w= 2.000
 *
 * CUDA optimization note:
 *   Stored in __constant__ memory — broadcast read, ~1 cycle for full warp.
 *   Fits in one 64-byte cache line.
 */
__constant__ float cw_delta_lut[16] = {
    1.00000000f,  /* m= 0: 2^( 0/4) */
    0.84089642f,  /* m= 1: 2^(-1/4) */
    0.70710678f,  /* m= 2: 2^(-2/4) = 1/sqrt(2) */
    0.59460354f,  /* m= 3: 2^(-3/4) */
    0.50000000f,  /* m= 4: 2^(-4/4) = 1/2 */
    0.42044821f,  /* m= 5 */
    0.35355339f,  /* m= 6: 2^(-6/4) = 1/(2*sqrt(2)) */
    0.29730178f,  /* m= 7 */
    0.25000000f,  /* m= 8: 2^(-8/4) = 1/4 */
    0.21022410f,  /* m= 9 */
    0.17677670f,  /* m=10 */
    0.14865089f,  /* m=11 */
    0.12500000f,  /* m=12: 2^(-12/4) = 1/8 */
    0.10511205f,  /* m=13 */
    0.08838835f,  /* m=14 */
    0.07432544f   /* m=15 */
};

/* Host-side copy (same values, plain array for CPU reference code) */
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
 *   64 distinct values, same outer struct, no format change.
 */

/*
 * INTERFACE DISCONNECTED — Legacy 4-bit n_data format (16 weights per uint64_t)
 * with a separate formula_bits array for the ± direction.
 *
 * Reason for removal: the ±-flag created two disjoint value branches with no
 * monotone ordering in weight space, making gradient-based k-updates undefined
 * across the boundary and preventing vector-space comparisons.
 *
 * Reconnect point: implement cw_materialize_legacy_f32() and call it when
 * BenchParams.use_legacy == true.  Math equivalence:
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
 * __global__ void cw_materialize_legacy_f32(const WeightArray_Legacy*, float*, uint32_t);
 */


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                         KERNEL  BLOCK                               ║
 * ║  cw_quake_rsqrt · cw_hw_rsqrt · cw_fiver_to_float_dev              ║
 * ║  cw_apply_act · cw_materialize_f32 · cw_materialize_f16            ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

/*
 * Method A: Raw Quake inverse sqrt — one shift + one subtraction, NO Newton-Raphson.
 *
 * CUDA note: compiles to MOV + SHR + SUB + MOV (4 integer ALU instructions).
 * No Special Function Unit (SFU) used.  Useful as a baseline to measure
 * the overhead of __frsqrt_rn's SFU path in benchB().
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
 * Method B: Hardware SFU reciprocal sqrt (device only).
 *
 * CUDA note: maps to single MUFU.RSQ instruction on all supported architectures.
 * Correctly rounded to nearest, ~1 clock on Ampere/Hopper SFU.
 * 4 SFUs per SM on Ampere → throughput 1 op / 4 clocks per warp (8 threads/clock).
 * Falls back to 1/sqrtf() on host for correctness testing.
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
 * Apply activation function by 2-bit code.
 *
 * CUDA note: with --use_fast_math:
 *   CW_ACT_SQRT     → SFU.SQRT (hardware square root)
 *   CW_ACT_LOG      → SFU.LG2 + correction (log base 2 → natural log)
 *   CW_ACT_SOFTSIGN → FADD + FABS + FRCP, fully branchless
 * To avoid warp divergence: caller should batch weights by activation type.
 */
__device__ __forceinline__
float cw_apply_act(float x, uint32_t code) {
    switch (code & 3u) {
        case CW_ACT_SQRT:     return copysignf(__fsqrt_rn(fabsf(x)), x);
        case CW_ACT_LOG:      return copysignf(log1pf(fabsf(x)), x);
        case CW_ACT_SOFTSIGN: return x / (1.0f + fabsf(x));
        default:              return x;  /* CW_ACT_IDENTITY */
    }
}

/*
 * Materialize one Fiver k-value → float weight (device).
 *
 * dir = k >> 4            (0 = below 1.0,  1 = above 1.0)
 * mag = dir ? (31-k) : k  (magnitude index 0..15)
 * w   = 1.0 + (dir ? +delta : -delta)
 *
 * CUDA note:
 *   __constant__ LUT fetch broadcasts across the warp in ~1 cycle.
 *   Sign application uses XOR on float bit-31 — single LOP3.LUT instruction.
 *   No branch instructions generated.
 */
__device__ __forceinline__
float cw_fiver_to_float_dev(uint8_t k) {
    uint32_t dir = (uint32_t)(k >> 4u) & 1u;
    uint32_t mag = dir ? (31u - (uint32_t)k) : (uint32_t)k;
    float    delta = cw_delta_lut[mag];
    uint32_t d_bits;
    __builtin_memcpy(&d_bits, &delta, 4);
    d_bits ^= ((dir ^ 1u) << 31u);     /* flip sign bit when dir==0 (subtract) */
    float signed_delta;
    __builtin_memcpy(&signed_delta, &d_bits, 4);
    return 1.0f + signed_delta;
}

/* Host version using the CPU LUT copy */
static inline float cw_fiver_to_float_cpu(uint8_t k) {
    uint32_t dir   = (uint32_t)(k >> 4u) & 1u;
    uint32_t mag   = dir ? (31u - (uint32_t)k) : (uint32_t)k;
    float    delta = cw_delta_lut_cpu[mag];
    return 1.0f + (dir ? +delta : -delta);
}

/*
 * Materialize FiverArray → float32 output buffer.
 *
 * Grid:  gridDim.x  = ceil(count / 128)
 * Block: blockDim.x = 128 threads, each handles one weight.
 *
 * CUDA optimization notes:
 *   - fiver_data/sign_bits/act_bits loaded via __ldg() (read-only texture cache)
 *   - Sign flip: XOR on float bit-31, zero branch instructions
 *   - Activation: switch compiles to predicated FADD/FABS/etc., no real branches
 *     for IDENTITY (default) path; warp divergence only if act codes differ
 *   - For f16 variant: __float2half() at write → 2× store bandwidth
 *   - __ballot_sync() could check uniform activation across a warp sub-group
 *     to skip the switch entirely on homogeneous layers (not implemented here
 *     to keep code readable; add as CW_FAST_UNIFORM_ACT compile flag if needed)
 */
__global__
void cw_materialize_f32(
    const uint8_t*  __restrict__ fiver_data,
    const uint8_t*  __restrict__ sign_bits,
    const uint32_t* __restrict__ act_bits,
    float*          __restrict__ out,
    uint32_t        count,
    uint8_t         flags)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint8_t k = __ldg(&fiver_data[idx]);
    float   w = cw_fiver_to_float_dev(k);

    if (flags & CW_FLAG_HAS_SIGN) {
        uint8_t  sbyte = __ldg(&sign_bits[idx >> 3u]);
        uint32_t sbit  = (uint32_t)(sbyte >> (idx & 7u)) & 1u;
        uint32_t w_bits;
        __builtin_memcpy(&w_bits, &w, 4);
        w_bits ^= (sbit << 31u);
        __builtin_memcpy(&w, &w_bits, 4);
    }

    if (flags & CW_FLAG_HAS_ACT) {
        uint32_t aword    = __ldg(&act_bits[idx >> 4u]);
        uint32_t act_code = (aword >> ((idx & 15u) * 2u)) & 3u;
        w = cw_apply_act(w, act_code);
    }

    out[idx] = w;
}

__global__
void cw_materialize_f16(
    const uint8_t*  __restrict__ fiver_data,
    const uint8_t*  __restrict__ sign_bits,
    const uint32_t* __restrict__ act_bits,
    __half*         __restrict__ out,
    uint32_t        count,
    uint8_t         flags)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint8_t k = __ldg(&fiver_data[idx]);
    float   w = cw_fiver_to_float_dev(k);

    if (flags & CW_FLAG_HAS_SIGN) {
        uint8_t  sbyte = __ldg(&sign_bits[idx >> 3u]);
        uint32_t sbit  = (uint32_t)(sbyte >> (idx & 7u)) & 1u;
        uint32_t w_bits;
        __builtin_memcpy(&w_bits, &w, 4);
        w_bits ^= (sbit << 31u);
        __builtin_memcpy(&w, &w_bits, 4);
    }

    if (flags & CW_FLAG_HAS_ACT) {
        uint32_t aword    = __ldg(&act_bits[idx >> 4u]);
        uint32_t act_code = (aword >> ((idx & 15u) * 2u)) & 3u;
        w = cw_apply_act(w, act_code);
    }

    out[idx] = __float2half(w);
}


/*
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║                    TEST / BENCHMARK  BLOCK                          ║
 * ║  bench_kernel_quake · bench_kernel_hw · run_benchmark               ║
 * ║  run_fiver_test · main (--bench-a | --bench-b | --test)             ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

__global__
void bench_kernel_quake(const float* __restrict__ in,
                        float*       __restrict__ out,
                        uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cw_quake_rsqrt(in[idx]);
}

__global__
void bench_kernel_hw(const float* __restrict__ in,
                     float*       __restrict__ out,
                     uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cw_hw_rsqrt(in[idx]);
}

static void run_benchmark(const char* label, int use_quake, int iterations) {
    const uint32_t N = 1u << 20;  /* 1M elements */

    float *d_in, *d_out;
    float *h_in      = (float*)malloc(N * sizeof(float));
    float *h_out_gpu = (float*)malloc(N * sizeof(float));
    float *h_ref     = (float*)malloc(N * sizeof(float));

    /* Test values: 2^(m/4) for m = 0..15, cycling — the exact denominators
     * used in cw_compute_delta, so error is directly meaningful. */
    for (uint32_t i = 0; i < N; i++) {
        float m_f  = (float)(i & 15u);
        h_in[i]    = powf(2.0f, m_f * 0.25f);
        h_ref[i]   = 1.0f / sqrtf(h_in[i]);
    }

    CW_CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Warm-up pass */
    if (use_quake) bench_kernel_quake<<<(N+127)/128, 128>>>(d_in, d_out, N);
    else           bench_kernel_hw   <<<(N+127)/128, 128>>>(d_in, d_out, N);
    CW_CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed passes */
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

static void run_fiver_test(void) {
    const uint32_t N = 32u;
    uint8_t h_fiver[32];
    float   h_out[32], h_ref[32];
    uint8_t* d_fiver; float* d_out;

    for (uint32_t k = 0; k < 32; k++) h_fiver[k] = (uint8_t)k;
    for (uint32_t k = 0; k < 32; k++) h_ref[k]   = cw_fiver_to_float_cpu((uint8_t)k);

    CW_CUDA_CHECK(cudaMalloc(&d_fiver, 32));
    CW_CUDA_CHECK(cudaMalloc(&d_out,   32 * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_fiver, h_fiver, 32, cudaMemcpyHostToDevice));

    cw_materialize_f32<<<1, 128>>>(d_fiver, NULL, NULL, d_out, N, 0);
    CW_CUDA_CHECK(cudaGetLastError());
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    CW_CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n%-4s  %-10s  %-10s  %-12s\n", "k", "expected", "got", "abs_err");
    printf("%.50s\n", "--------------------------------------------------");
    uint32_t errors = 0;
    for (uint32_t k = 0; k < 32; k++) {
        float err = fabsf(h_out[k] - h_ref[k]);
        if (err > 1e-5f) errors++;
        printf("%-4u  %-10.6f  %-10.6f  %-12.2e%s\n",
               k, h_ref[k], h_out[k], err, err > 1e-5f ? "  FAIL" : "");
    }
    printf("\nFiver test: %u / 32 errors\n", errors);
    printf("TEST %s\n", errors == 0 ? "PASSED" : "FAILED");

    CW_CUDA_WARN(cudaFree(d_fiver));
    CW_CUDA_WARN(cudaFree(d_out));
}

static void print_device_info(void) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) {
        fprintf(stderr, "[CW WARN]  no CUDA device available: %s\n",
                cudaGetErrorString(e));
        return;
    }
    cudaDeviceProp prop;
    CW_CUDA_WARN(cudaGetDeviceProperties(&prop, dev));
    printf("[device]  %s  (sm_%d%d)  %.0f MB global  %d SMs\n\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.multiProcessorCount);
}

int main(int argc, char** argv) {
    printf("compact_weights — Symmetric Fiver Encoding\n");
    printf("===========================================\n\n");
    print_device_info();

    int do_a = 0, do_b = 0, do_t = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench-a") == 0) do_a = 1;
        if (strcmp(argv[i], "--bench-b") == 0) do_b = 1;
        if (strcmp(argv[i], "--test")    == 0) do_t = 1;
    }
    if (!do_a && !do_b && !do_t) { do_a = do_b = do_t = 1; }

    if (do_a || do_b) {
        printf("%-22s | %-16s | %-18s | %-18s\n",
               "METHOD", "TIME_NS/OP", "MAX_REL_ERROR", "AVG_REL_ERROR");
        printf("%.80s\n",
               "--------------------------------------------------------------------------------");
        if (do_a) run_benchmark("Quake-raw (A)",    1, 10);
        if (do_b) run_benchmark("__frsqrt_rn (B)",  0, 10);
        printf("\n");
    }

    if (do_t) run_fiver_test();

    return 0;
}
