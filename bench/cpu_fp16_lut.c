/*
 * cpu_fp16_lut.c — fp16 tables for the float path on AVX-512.
 *
 * Codes are 16-bit lanes (32 per register); the table holds fp16 values,
 * vcvtph2ps widens 16 of them to float32 per instruction.
 *   5 bit fp32  vpermt2ps                      (16 codes / lookup)   reference
 *   6 bit fp32  2 × vpermt2ps + blend          (16 codes / lookup)   reference
 *   5 bit fp16  vpermw   + 2 × vcvtph2ps       (32 codes / lookup)
 *   6 bit fp16  vpermi2w + 2 × vcvtph2ps       (64-entry table from two registers)
 *   7 bit fp16  2 × vpermi2w + blend + 2 × vcvtph2ps
 * vpermw / vpermi2w need AVX512BW only (no VBMI).
 *
 * Checks: every variant against a scalar lookup; MAGIC grid must arrive
 * bit-exact through fp16, CENTERED is reported as relative rounding error.
 *
 *   clang -O3 -o cpu_fp16_lut bench/cpu_fp16_lut.c && taskset -c 1 ./cpu_fp16_lut
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TB __attribute__((target("avx512f,avx512bw,avx512vl,f16c"), noinline))
#define IB __attribute__((target("avx512f,avx512bw,avx512vl,f16c"), always_inline)) inline

static float    F32[128] __attribute__((aligned(64)));
static uint16_t F16[128] __attribute__((aligned(64)));

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static float magic(uint32_t k) {
    uint32_t dir = (k >> 4) & 1u, m = (k ^ (0u - dir)) & 15u;
    uint32_t b = (0x3F800000u - (m << 21)) | ((dir ^ 1u) << 31);
    float v; memcpy(&v, &b, 4);
    return k == 16u ? 0.0f : v;
}
static float centered(uint32_t k) {
    if (k == 16u) return 0.0f;
    return k < 16u ? -powf(2.0f, -(float)k / 4.0f) : powf(2.0f, -(float)(31u - k) / 4.0f);
}

/* each decoder: 32 codes (uint16 lanes) → two __m512 of floats */
typedef struct { __m512 lo, hi; } f32x32;

static IB f32x32 cvt(__m512i h) {
    f32x32 r = { _mm512_cvtph_ps(_mm512_castsi512_si256(h)), _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 1)) };
    return r;
}
static IB f32x32 d5_f16(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    (void)t1; (void)t2; (void)t3; return cvt(_mm512_permutexvar_epi16(k, t0));
}
static IB f32x32 d6_f16(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    (void)t2; (void)t3; return cvt(_mm512_permutex2var_epi16(t0, k, t1));
}
static IB f32x32 d7_f16(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    const __mmask32 hi = _mm512_test_epi16_mask(k, _mm512_set1_epi16(64));
    return cvt(_mm512_mask_blend_epi16(hi, _mm512_permutex2var_epi16(t0, k, t1), _mm512_permutex2var_epi16(t2, k, t3)));
}
static IB f32x32 d5_f32(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    (void)t1; (void)t2; (void)t3;
    const __m512 a = _mm512_castsi512_ps(t0), b = _mm512_castsi512_ps(t1);
    f32x32 r = { _mm512_permutex2var_ps(a, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(k)), b),
                 _mm512_permutex2var_ps(a, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(k, 1)), b) };
    return r;
}
static IB __m512 l6(__m512i k32, __m512 a, __m512 b, __m512 c, __m512 d) {
    const __mmask16 hi = _mm512_test_epi32_mask(k32, _mm512_set1_epi32(32));
    return _mm512_mask_blend_ps(hi, _mm512_permutex2var_ps(a, k32, b), _mm512_permutex2var_ps(c, k32, d));
}
static IB f32x32 d6_f32(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    const __m512 a = _mm512_castsi512_ps(t0), b = _mm512_castsi512_ps(t1), c = _mm512_castsi512_ps(t2), d = _mm512_castsi512_ps(t3);
    f32x32 r = { l6(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(k)), a, b, c, d),
                 l6(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(k, 1)), a, b, c, d) };
    return r;
}

#define VARIANT(NAME, DEC, TAB)                                                            \
static TB void NAME##_store(const uint16_t* c, float* out, int n) {                         \
    const __m512i t0 = _mm512_load_si512(TAB), t1 = _mm512_load_si512((const char*)(TAB) + 64), \
                  t2 = _mm512_load_si512((const char*)(TAB) + 128), t3 = _mm512_load_si512((const char*)(TAB) + 192); \
    for (int i = 0; i < n; i += 32) {                                                       \
        f32x32 v = DEC(_mm512_load_si512(c + i), t0, t1, t2, t3);                           \
        _mm512_store_ps(out + i, v.lo); _mm512_store_ps(out + i + 16, v.hi);                \
    }                                                                                       \
}                                                                                           \
static TB float NAME##_sum(const uint16_t* c, int n) {                                      \
    const __m512i t0 = _mm512_load_si512(TAB), t1 = _mm512_load_si512((const char*)(TAB) + 64), \
                  t2 = _mm512_load_si512((const char*)(TAB) + 128), t3 = _mm512_load_si512((const char*)(TAB) + 192); \
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;                             \
    for (int i = 0; i < n; i += 64) {                                                       \
        f32x32 v = DEC(_mm512_load_si512(c + i), t0, t1, t2, t3);                           \
        f32x32 w = DEC(_mm512_load_si512(c + i + 32), t0, t1, t2, t3);                      \
        a0 = _mm512_add_ps(a0, v.lo); a1 = _mm512_add_ps(a1, v.hi);                         \
        a2 = _mm512_add_ps(a2, w.lo); a3 = _mm512_add_ps(a3, w.hi);                         \
    }                                                                                       \
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3))); \
}
VARIANT(v5_f32, d5_f32, F32)
VARIANT(v6_f32, d6_f32, F32)
VARIANT(v5_f16, d5_f16, F16)
VARIANT(v6_f16, d6_f16, F16)
VARIANT(v7_f16, d7_f16, F16)

typedef struct { const char* name; int bits; int fp16; void (*st)(const uint16_t*, float*, int); float (*sum)(const uint16_t*, int); } V;

static TB void fill_tables(const float* vals, int n) {
    for (int i = 0; i < 128; i++) F32[i] = vals[i % n];
    for (int i = 0; i < 128; i += 16)
        _mm256_store_si256((__m256i*)(F16 + i), _mm512_cvtps_ph(_mm512_loadu_ps(F32 + i), _MM_FROUND_TO_NEAREST_INT));
}

int main(void) {
    const int n = 1 << 17;                                     /* 256 KB of uint16 codes: L2-resident */
    uint16_t* c = aligned_alloc(64, n * 2);
    float* out = aligned_alloc(64, n * 4);
    float vals[128];
    srand(5);
    for (int i = 0; i < 128; i++) vals[i] = (float)(rand() % 2001 - 1000) / 1024.0f;  /* fp16-exact */
    fill_tables(vals, 128);
    V v[] = {
        { "5 bit fp32  vpermt2ps                ", 5, 0, v5_f32_store, v5_f32_sum },
        { "6 bit fp32  2x vpermt2ps + blend     ", 6, 0, v6_f32_store, v6_f32_sum },
        { "5 bit fp16  vpermw   + 2x vcvtph2ps  ", 5, 1, v5_f16_store, v5_f16_sum },
        { "6 bit fp16  vpermi2w + 2x vcvtph2ps  ", 6, 1, v6_f16_store, v6_f16_sum },
        { "7 bit fp16  2x vpermi2w + blend + cvt", 7, 1, v7_f16_store, v7_f16_sum } };
    int fails = 0;
    printf("lookup cost (best of 7, ns per code; tables with fp16-exact values)\n");
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < n; i++) c[i] = (uint16_t)(rand() & ((1 << v[j].bits) - 1));
        v[j].st(c, out, n);
        int bad = 0;
        for (int i = 0; i < n; i++) bad += out[i] != F32[c[i]];
        fails += bad != 0;
        volatile float sink = 0; double best = 1e9;
        for (int r = 0; r < 7; r++) {
            double t0 = now();
            for (int it = 0; it < 400; it++) sink += v[j].sum(c, n);
            double dt = (now() - t0) / (400.0 * n);
            if (dt < best) best = dt;
        }
        printf("  %s  %6.4f ns/code  %6.1f Gcodes/s   check: %s\n", v[j].name, best * 1e9, 1e-9 / best, bad ? "FAIL" : "ok");
    }

    printf("\nFiver grids through an fp16 table (5 bit, vpermw)\n");
    for (int g = 0; g < 2; g++) {
        float gv[32];
        for (uint32_t k = 0; k < 32; k++) gv[k] = g ? centered(k) : magic(k);
        fill_tables(gv, 32);
        for (int i = 0; i < n; i++) c[i] = (uint16_t)(rand() & 31);
        v5_f16_store(c, out, n);
        double worst = 0;
        int exact = 1;
        for (int i = 0; i < n; i++) {
            const double ref = gv[c[i]], d = fabs(out[i] - ref);
            if (out[i] != gv[c[i]]) exact = 0;
            if (ref != 0 && d / fabs(ref) > worst) worst = d / fabs(ref);
        }
        printf("  %-8s bit-exact: %-3s  max relative error %.2e\n", g ? "centered" : "magic", exact ? "yes" : "no", worst);
        if (!g && !exact) fails++;
    }
    printf("\n%s\n", fails ? "CHECK FAILED" : "all checks ok");
    return fails != 0;
}
