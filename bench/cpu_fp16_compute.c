/*
 * cpu_fp16_compute.c — compute directly in fp16 / bf16 after the table lookup.
 *
 * One GEMV row = Σ v(k_c) · x_c over 4608 codes (5-bit MAGIC grid, exact in
 * fp16 and bf16), codes as uint16 lanes, 64 rows (L2-resident):
 *   fp32         vpermt2ps + vfmadd231ps         16 MACs / FMA, fp32 sum   (reference)
 *   fp16         vpermw    + vfmadd231ph         32 MACs / FMA, fp16 sum
 *   fp16 flush   as fp16, every 8 FMAs the fp16 partials go into an fp32 sum
 *   bf16         vpermw    + vdpbf16ps           32 MACs / instr, fp32 sum
 * x is converted once to the matching format (fp16 / bf16), as it would be
 * once per layer.  Error: max over rows of |y − y_double| / Σ|w·x|.
 *
 *   clang -O3 -o cpu_fp16_compute bench/cpu_fp16_compute.c -lm && taskset -c 1 ./cpu_fp16_compute
 * Needs AVX512-FP16 and AVX512-BF16 (Sapphire Rapids and newer).
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TGT "avx512f,avx512bw,avx512vl,avx512fp16,avx512bf16"
#define TF __attribute__((target(TGT), noinline))

enum { ROWS = 64, COLS = 4608 };   /* 590 KB of codes: L2-resident, so the loop is compute-bound */
static float    T32[32] __attribute__((aligned(64)));
static uint16_t T16[32] __attribute__((aligned(64)));
static uint16_t TBF[32] __attribute__((aligned(64)));

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static float magic(uint32_t k) {
    uint32_t dir = (k >> 4) & 1u, m = (k ^ (0u - dir)) & 15u;
    uint32_t b = (0x3F800000u - (m << 21)) | ((dir ^ 1u) << 31);
    float v; memcpy(&v, &b, 4);
    return k == 16u ? 0.0f : v;
}

static TF float dot_f32(const uint16_t* c, const float* x) {
    const __m512 lo = _mm512_load_ps(T32), hi = _mm512_load_ps(T32 + 16);
    __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    for (int i = 0; i < COLS; i += 64)
        for (int u = 0; u < 4; u++) {
            const __m512i k = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*)(c + i + 16 * u)));
            a[u] = _mm512_fmadd_ps(_mm512_permutex2var_ps(lo, k, hi), _mm512_load_ps(x + i + 16 * u), a[u]);
        }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
}

static TF float dot_f16(const uint16_t* c, const uint16_t* xh, int flush) {
    const __m512i t = _mm512_load_si512(T16);
    __m512h a[4] = { _mm512_setzero_ph(), _mm512_setzero_ph(), _mm512_setzero_ph(), _mm512_setzero_ph() };
    __m512 s = _mm512_setzero_ps();
    int steps = 0;
    for (int i = 0; i < COLS; i += 128) {
        for (int u = 0; u < 4; u++) {
            const __m512i k = _mm512_load_si512(c + i + 32 * u);
            a[u] = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_permutexvar_epi16(k, t)),
                                   _mm512_castsi512_ph(_mm512_load_si512(xh + i + 32 * u)), a[u]);
        }
        if (flush && ++steps == 8) {
            for (int u = 0; u < 4; u++) {
                const __m512i h = _mm512_castph_si512(a[u]);
                s = _mm512_add_ps(s, _mm512_add_ps(_mm512_cvtph_ps(_mm512_castsi512_si256(h)),
                                                   _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 1))));
                a[u] = _mm512_setzero_ph();
            }
            steps = 0;
        }
    }
    for (int u = 0; u < 4; u++) {
        const __m512i h = _mm512_castph_si512(a[u]);
        s = _mm512_add_ps(s, _mm512_add_ps(_mm512_cvtph_ps(_mm512_castsi512_si256(h)),
                                           _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 1))));
    }
    return _mm512_reduce_add_ps(s);
}

static TF float dot_bf16(const uint16_t* c, const uint16_t* xb) {
    const __m512i t = _mm512_load_si512(TBF);
    __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    for (int i = 0; i < COLS; i += 128)
        for (int u = 0; u < 4; u++) {
            const __m512i k = _mm512_load_si512(c + i + 32 * u);
            a[u] = _mm512_dpbf16_ps(a[u], (__m512bh)_mm512_permutexvar_epi16(k, t),
                                    (__m512bh)_mm512_load_si512(xb + i + 32 * u));
        }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
}

static TF void convert_tables_and_x(const float* x, uint16_t* xh, uint16_t* xb) {
    for (int i = 0; i < 32; i += 16) {
        _mm256_store_si256((__m256i*)(T16 + i), _mm512_cvtps_ph(_mm512_load_ps(T32 + i), _MM_FROUND_TO_NEAREST_INT));
        _mm256_store_si256((__m256i*)(TBF + i), (__m256i)_mm512_cvtneps_pbh(_mm512_load_ps(T32 + i)));
    }
    for (int i = 0; i < COLS; i += 16) {
        _mm256_store_si256((__m256i*)(xh + i), _mm512_cvtps_ph(_mm512_load_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
        _mm256_store_si256((__m256i*)(xb + i), (__m256i)_mm512_cvtneps_pbh(_mm512_load_ps(x + i)));
    }
}

int main(void) {
    for (uint32_t k = 0; k < 32; k++) T32[k] = magic(k);
    uint16_t* c  = aligned_alloc(64, ROWS * COLS * 2);
    float*    x  = aligned_alloc(64, COLS * 4);
    uint16_t* xh = aligned_alloc(64, COLS * 2);
    uint16_t* xb = aligned_alloc(64, COLS * 2);
    srand(11);
    for (int i = 0; i < ROWS * COLS; i++) c[i] = (uint16_t)(rand() & 31);
    for (int i = 0; i < COLS; i++) x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    convert_tables_and_x(x, xh, xb);

    double ref[ROWS], mag[ROWS];
    for (int r = 0; r < ROWS; r++) {
        ref[r] = mag[r] = 0;
        for (int i = 0; i < COLS; i++) {
            const double t = (double)T32[c[r * COLS + i]] * x[i];
            ref[r] += t; mag[r] += fabs(t);
        }
    }
    const char* names[4] = { "fp32  vpermt2ps + vfmadd231ps (16/FMA)", "fp16  vpermw + vfmadd231ph (32/FMA)  ",
                             "fp16  + fp32 flush every 8 FMAs      ", "bf16  vpermw + vdpbf16ps (32/instr)  " };
    for (int v = 0; v < 4; v++) {
        double worst = 0;
        for (int r = 0; r < ROWS; r++) {
            const uint16_t* cr = c + (size_t)r * COLS;
            const float y = v == 0 ? dot_f32(cr, x) : v == 3 ? dot_bf16(cr, xb) : dot_f16(cr, xh, v == 2);
            worst = fmax(worst, fabs(y - ref[r]) / mag[r]);
        }
        volatile float sink = 0;
        double best = 1e9;
        for (int rep = 0; rep < 7; rep++) {
            const double t0 = now();
            for (int it = 0; it < 160; it++)
                for (int r = 0; r < ROWS; r++) {
                    const uint16_t* cr = c + (size_t)r * COLS;
                    sink += v == 0 ? dot_f32(cr, x) : v == 3 ? dot_bf16(cr, xb) : dot_f16(cr, xh, v == 2);
                }
            const double dt = (now() - t0) / (160.0 * ROWS * COLS);
            if (dt < best) best = dt;
        }
        printf("  %s  %6.4f ns/weight  %6.1f Gweights/s   max rel err %.1e\n", names[v], best * 1e9, 1e-9 / best, worst);
    }
    return 0;
}
