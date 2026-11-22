/*
 * cpu_fiver_decode.c — CPU decode experiment for the Fiver MAGIC grid
 * (groundwork for the pyTorch_i86 backend).
 *
 * One GEMV row = dot(packed row, x), whole SIMD blocks:
 *   AVX2     24 weights = 4 words → 3 × 8 lanes
 *   AVX-512  48 weights = 8 words → 3 × 16 lanes
 * Codes are brought into natural order with a word permute + variable shift,
 * so x is read contiguously.  Decoders: LUT via permute (AVX2: 4 × vpermps +
 * blends; AVX-512: one vpermt2ps), LUT via gather, MAGIC bit formula.
 * Packed layout as in compact_weights.cu: 6 codes per uint32, k_i at bits 5i.
 *
 *   clang -O3 -o cpu_fiver_decode bench/cpu_fiver_decode.c -lm && ./cpu_fiver_decode
 *   clang -O3 -S -DFIVER_MCA -o decode.s bench/cpu_fiver_decode.c   # regions for llvm-mca
 * Needs an AVX-512 CPU to run (the AVX2 kernels alone run anywhere with AVX2).
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef FIVER_MCA   /* region markers for llvm-mca; compile with -S -DFIVER_MCA */
#define MCA_BEGIN(n) __asm__ volatile("# LLVM-MCA-BEGIN " n)
#define MCA_END()    __asm__ volatile("# LLVM-MCA-END")
#else
#define MCA_BEGIN(n)
#define MCA_END()
#endif

static float MAGIC[32] __attribute__((aligned(64)));

static float magic_scalar(uint32_t k) {
    uint32_t dir = (k >> 4) & 1u, m = (k ^ (0u - dir)) & 15u;
    uint32_t b = (0x3F800000u - (m << 21)) | ((dir ^ 1u) << 31);
    float v; memcpy(&v, &b, 4);
    return k == 16u ? 0.0f : v;
}

static double dot_scalar(const uint32_t* w, const float* x, int cols) {
    double acc = 0.0;
    for (int c = 0; c < cols; c++) acc += (double)MAGIC[(w[c / 6] >> (5 * (c % 6))) & 31u] * x[c];
    return acc;
}

/* ───────────────────────── AVX2: 24 weights = 4 words = 3 × 8 lanes ───────────────────────── */
#define T2 __attribute__((target("avx2,fma"), noinline))

/* LUT: 4 × vpermps over 8-entry tables, selected by bit 3 and bit 4 (moved to the sign bit) */
static inline __attribute__((target("avx2,fma"), always_inline))
__m256 avx2_lut(__m256i k, const __m256 t[4]) {
    const __m256 b3 = _mm256_castsi256_ps(_mm256_slli_epi32(k, 28));
    const __m256 b4 = _mm256_castsi256_ps(_mm256_slli_epi32(k, 27));
    const __m256 lo = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t[0], k), _mm256_permutevar8x32_ps(t[1], k), b3);
    const __m256 hi = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t[2], k), _mm256_permutevar8x32_ps(t[3], k), b3);
    return _mm256_blendv_ps(lo, hi, b4);
}

static inline __attribute__((target("avx2,fma"), always_inline))
__m256 avx2_gather(__m256i k) {
    return _mm256_i32gather_ps(MAGIC, _mm256_and_si256(k, _mm256_set1_epi32(31)), 4);
}

/* MAGIC by formula: |v| = bits(0x3F800000 − (m << 21)), mirror, sign, zero code */
static inline __attribute__((target("avx2,fma"), always_inline))
__m256 avx2_magic(__m256i k) {
    const __m256i t    = _mm256_slli_epi32(k, 27);                /* bit 4 → bit 31, garbage gone */
    const __m256i dirm = _mm256_srai_epi32(t, 31);                /* all ones where k ≥ 16        */
    const __m256i m    = _mm256_and_si256(_mm256_xor_si256(k, dirm), _mm256_set1_epi32(15));
    __m256i b          = _mm256_sub_epi32(_mm256_set1_epi32(0x3F800000), _mm256_slli_epi32(m, 21));
    b = _mm256_or_si256(b, _mm256_andnot_si256(dirm, _mm256_set1_epi32((int)0x80000000)));
    const __m256i zero = _mm256_cmpeq_epi32(t, _mm256_set1_epi32((int)0x80000000));   /* k == 16 */
    return _mm256_castsi256_ps(_mm256_andnot_si256(zero, b));
}

#define AVX2_DOT(NAME, DECODE)                                                              \
static T2 double NAME(const uint32_t* w, const float* x, int cols) {                         \
    __m256 t[4];                                                                             \
    for (int j = 0; j < 4; j++) t[j] = _mm256_load_ps(MAGIC + 8 * j);                        \
    (void)t;                                                                                 \
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0;                                       \
    for (int blk = 0; blk < cols / 24; blk++) {                                              \
        MCA_BEGIN(#NAME);                                                                    \
        const __m256i w4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(w + 4 * blk))); \
        const __m256i k0 = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(w4, _mm256_setr_epi32(0,0,0,0,0,0,1,1)), _mm256_setr_epi32(0,5,10,15,20,25,0,5));   \
        const __m256i k1 = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(w4, _mm256_setr_epi32(1,1,1,1,2,2,2,2)), _mm256_setr_epi32(10,15,20,25,0,5,10,15)); \
        const __m256i k2 = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(w4, _mm256_setr_epi32(2,2,3,3,3,3,3,3)), _mm256_setr_epi32(20,25,0,5,10,15,20,25)); \
        const float* xb = x + 24 * blk;                                                      \
        a0 = _mm256_fmadd_ps(DECODE(k0), _mm256_loadu_ps(xb),      a0);                      \
        a1 = _mm256_fmadd_ps(DECODE(k1), _mm256_loadu_ps(xb + 8),  a1);                      \
        a2 = _mm256_fmadd_ps(DECODE(k2), _mm256_loadu_ps(xb + 16), a2);                      \
        MCA_END();                                                                           \
    }                                                                                        \
    __m256 a = _mm256_add_ps(_mm256_add_ps(a0, a1), a2);                                     \
    float r[8]; _mm256_storeu_ps(r, a);                                                      \
    return (double)r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];                     \
}
#define DEC_LUT(k)    avx2_lut(k, t)
#define DEC_GATHER(k) avx2_gather(k)
#define DEC_MAGIC(k)  avx2_magic(k)
AVX2_DOT(avx2_dot_lut,    DEC_LUT)
AVX2_DOT(avx2_dot_gather, DEC_GATHER)
AVX2_DOT(avx2_dot_magic,  DEC_MAGIC)

/* ─────────────────────── AVX-512: 48 weights = 8 words = 3 × 16 lanes ─────────────────────── */
#define T5 __attribute__((target("avx512f,avx512vl,avx512bw,avx2,fma"), noinline))
#define I5 __attribute__((target("avx512f,avx512vl,avx512bw,avx2,fma"), always_inline)) inline

static int32_t IDX5[3][16] __attribute__((aligned(64))), SH5[3][16] __attribute__((aligned(64)));

/* LUT: one vpermt2ps over two 16-entry tables; it reads index bits 4:0 only,
 * so the shifted codes need no masking at all */
static I5 __m512 avx512_lut(__m512i k, __m512 lo, __m512 hi) { return _mm512_permutex2var_ps(lo, k, hi); }

static I5 __m512 avx512_gather(__m512i k) {
    return _mm512_i32gather_ps(_mm512_and_si512(k, _mm512_set1_epi32(31)), MAGIC, 4);
}

static I5 __m512 avx512_magic(__m512i k) {
    const __mmask16 dir  = _mm512_test_epi32_mask(k, _mm512_set1_epi32(16));
    __m512i m            = _mm512_and_si512(k, _mm512_set1_epi32(15));
    m                    = _mm512_mask_xor_epi32(m, dir, m, _mm512_set1_epi32(15));   /* 31 − k */
    __m512i b            = _mm512_sub_epi32(_mm512_set1_epi32(0x3F800000), _mm512_slli_epi32(m, 21));
    b                    = _mm512_mask_or_epi32(b, (__mmask16)~dir, b, _mm512_set1_epi32((int)0x80000000));
    const __mmask16 zero = _mm512_mask_cmpeq_epi32_mask(dir, m, _mm512_set1_epi32(15));  /* k == 16 */
    return _mm512_maskz_mov_ps((__mmask16)~zero, _mm512_castsi512_ps(b));
}

#define AVX512_DOT(NAME, DECODE)                                                            \
static T5 double NAME(const uint32_t* w, const float* x, int cols) {                         \
    const __m512 lo = _mm512_load_ps(MAGIC), hi = _mm512_load_ps(MAGIC + 16);                \
    (void)lo; (void)hi;                                                                      \
    const __m512i i0 = _mm512_load_si512(IDX5[0]), i1 = _mm512_load_si512(IDX5[1]), i2 = _mm512_load_si512(IDX5[2]); \
    const __m512i s0 = _mm512_load_si512(SH5[0]),  s1 = _mm512_load_si512(SH5[1]),  s2 = _mm512_load_si512(SH5[2]);  \
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0;                                       \
    for (int blk = 0; blk < cols / 48; blk++) {                                              \
        MCA_BEGIN(#NAME);                                                                    \
        const __m512i w8 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(w + 8 * blk))); \
        const __m512i k0 = _mm512_srlv_epi32(_mm512_permutexvar_epi32(i0, w8), s0);          \
        const __m512i k1 = _mm512_srlv_epi32(_mm512_permutexvar_epi32(i1, w8), s1);          \
        const __m512i k2 = _mm512_srlv_epi32(_mm512_permutexvar_epi32(i2, w8), s2);          \
        const float* xb = x + 48 * blk;                                                      \
        a0 = _mm512_fmadd_ps(DECODE(k0), _mm512_loadu_ps(xb),      a0);                      \
        a1 = _mm512_fmadd_ps(DECODE(k1), _mm512_loadu_ps(xb + 16), a1);                      \
        a2 = _mm512_fmadd_ps(DECODE(k2), _mm512_loadu_ps(xb + 32), a2);                      \
        MCA_END();                                                                           \
    }                                                                                        \
    return (double)_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), a2));           \
}
#define DEC5_LUT(k)    avx512_lut(k, lo, hi)
#define DEC5_GATHER(k) avx512_gather(k)
#define DEC5_MAGIC(k)  avx512_magic(k)
AVX512_DOT(avx512_dot_lut,    DEC5_LUT)
AVX512_DOT(avx512_dot_gather, DEC5_GATHER)
AVX512_DOT(avx512_dot_magic,  DEC5_MAGIC)

/* ───────────────────────────────────────── driver ───────────────────────────────────────── */
typedef double (*dot_fn)(const uint32_t*, const float*, int);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void bench(const char* name, dot_fn f, const uint32_t* W, const float* x, int rows, int cols,
                  const double* ref, int reps) {
    const int wpr = cols / 6;
    volatile double sink = 0;
    double worst = 0;
    for (int r = 0; r < rows; r++) {
        double y = f(W + (size_t)r * wpr, x, cols), e = fabs(y - ref[r]) / (fabs(ref[r]) + 1e-3);
        if (e > worst) worst = e;
    }
    double best = 1e30;
    for (int t = 0; t < 5; t++) {
        double t0 = now();
        for (int it = 0; it < reps; it++)
            for (int r = 0; r < rows; r++) sink += f(W + (size_t)r * wpr, x, cols);
        double dt = (now() - t0) / reps;
        if (dt < best) best = dt;
    }
    const double n = (double)rows * cols;
    printf("  %-18s %7.3f ns/weight  %6.2f Gweights/s  (%5.1f MB/s packed)  max rel err %.1e\n",
           name, best / n * 1e9, n / best / 1e9, n * 32.0 / 6.0 / 8.0 / best / 1e6, worst);
}

int main(int argc, char** argv) {
    for (int k = 0; k < 32; k++) MAGIC[k] = magic_scalar((uint32_t)k);
    for (int v = 0; v < 3; v++)
        for (int l = 0; l < 16; l++) { int c = 16 * v + l; IDX5[v][l] = c / 6; SH5[v][l] = 5 * (c % 6); }

    const int cols = 4608;                       /* multiple of 6, 24 and 48 */
    const int sizes[2] = { 256, 8192 };          /* ~0.8 MB (L2-resident) and ~25 MB (beyond L2) */
    for (int si = 0; si < 2; si++) {
        const int rows = sizes[si], wpr = cols / 6;
        uint32_t* W = aligned_alloc(64, (size_t)rows * wpr * 4);
        float*    x = aligned_alloc(64, (size_t)cols * 4);
        double*   ref = malloc(rows * sizeof(double));
        srand(1);
        for (size_t i = 0; i < (size_t)rows * wpr; i++) {
            uint32_t w = 0;
            for (int j = 0; j < 6; j++) w |= (uint32_t)(rand() & 31) << (5 * j);
            W[i] = w;
        }
        for (int c = 0; c < cols; c++) x[c] = (float)((double)rand() / RAND_MAX) - 0.5f;
        for (int r = 0; r < rows; r++) ref[r] = dot_scalar(W + (size_t)r * wpr, x, cols);
        const int reps = si == 0 ? 400 : 12;
        printf("%d x %d, packed %.1f MB\n", rows, cols, (double)rows * wpr * 4 / 1e6);
        bench("AVX2   LUT permute", avx2_dot_lut,      W, x, rows, cols, ref, reps);
        bench("AVX2   LUT gather",  avx2_dot_gather,   W, x, rows, cols, ref, reps);
        bench("AVX2   magic bits",  avx2_dot_magic,    W, x, rows, cols, ref, reps);
        bench("AVX512 LUT vpermt2", avx512_dot_lut,    W, x, rows, cols, ref, reps);
        bench("AVX512 LUT gather",  avx512_dot_gather, W, x, rows, cols, ref, reps);
        bench("AVX512 magic bits",  avx512_dot_magic,  W, x, rows, cols, ref, reps);
        free(W); free(x); free(ref);
    }
    (void)argc; (void)argv;
    return 0;
}
