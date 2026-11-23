/*
 * cpu_lut_width.c — decode cost versus code width on AVX-512.
 *
 * Decodes 64K codes (L2-resident) to float through a table of 16, 32, 64 or
 * 256 entries and sums them; plus a uniform int grid without any table.
 *   4 bit: one vpermps (16 floats)       5 bit: one vpermt2ps (2 × 16 floats)
 *   6 bit: 2 × vpermt2ps + test + blend   8 bit: gather
 *
 *   clang -O3 -o cpu_lut_width bench/cpu_lut_width.c && taskset -c 1 ./cpu_lut_width
 * Granite Rapids: 4 and 5 bit 0.027 ns/code, 6 bit 0.053, 8 bit 0.151,
 * uniform int5 0.035 (see docs/ARCHITEKTUR_CUDA.md, 8.3).
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define T __attribute__((target("avx512f,avx512vl,avx512bw"), noinline))
static float TAB[256] __attribute__((aligned(64)));
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* each kernel: decode n codes (int32, < table size) to float and sum; 4 independent accumulators */
#define LOOP(BODY) \
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0; \
    for (int i = 0; i < n; i += 64) { \
        __m512i k0 = _mm512_load_si512(c + i), k1 = _mm512_load_si512(c + i + 16), \
                k2 = _mm512_load_si512(c + i + 32), k3 = _mm512_load_si512(c + i + 48); \
        a0 = _mm512_add_ps(a0, BODY(k0)); a1 = _mm512_add_ps(a1, BODY(k1)); \
        a2 = _mm512_add_ps(a2, BODY(k2)); a3 = _mm512_add_ps(a3, BODY(k3)); } \
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
T float int_uniform(const int* c, int n) {                         /* int5 grid: k − 16 → float */
    const __m512i o = _mm512_set1_epi32(16);
#define B(k) _mm512_cvtepi32_ps(_mm512_sub_epi32(k, o))
    LOOP(B)
#undef B
}
T float lut16(const int* c, int n) {                               /* 4 bit: one vpermps */
    const __m512 t0 = _mm512_load_ps(TAB);
#define B(k) _mm512_permutexvar_ps(k, t0)
    LOOP(B)
#undef B
}
T float lut32(const int* c, int n) {                               /* 5 bit: one vpermt2ps */
    const __m512 t0 = _mm512_load_ps(TAB), t1 = _mm512_load_ps(TAB + 16);
#define B(k) _mm512_permutex2var_ps(t0, k, t1)
    LOOP(B)
#undef B
}
static inline __attribute__((target("avx512f,avx512vl,avx512bw"), always_inline))
__m512 dec64(__m512i k, __m512 t0, __m512 t1, __m512 t2, __m512 t3) {   /* 6 bit: 2 × vpermt2ps + test + blend */
    const __mmask16 hi = _mm512_test_epi32_mask(k, _mm512_set1_epi32(32));
    return _mm512_mask_blend_ps(hi, _mm512_permutex2var_ps(t0, k, t1), _mm512_permutex2var_ps(t2, k, t3));
}
T float lut64(const int* c, int n) {
    const __m512 t0 = _mm512_load_ps(TAB), t1 = _mm512_load_ps(TAB + 16), t2 = _mm512_load_ps(TAB + 32), t3 = _mm512_load_ps(TAB + 48);
#define B(k) dec64(k, t0, t1, t2, t3)
    LOOP(B)
#undef B
}
T float lut256(const int* c, int n) {                              /* 8 bit: gather */
#define B(k) _mm512_i32gather_ps(k, TAB, 4)
    LOOP(B)
#undef B
}
typedef float (*fn)(const int*, int);
int main(void) {
    for (int i = 0; i < 256; i++) TAB[i] = (float)i * 0.01f - 1.0f;
    const int n = 1 << 16;                                         /* 256 KB of codes: L2-resident */
    int* c = aligned_alloc(64, n * 4);
    struct { const char* name; fn f; int range; } v[] = {
        {"int5 uniform (sub+cvt)", int_uniform, 32}, {"4 bit LUT16  vpermps", lut16, 16},
        {"5 bit LUT32  vpermt2ps", lut32, 32}, {"6 bit LUT64  2x vpermt2ps+blend", lut64, 64},
        {"8 bit LUT256 gather", lut256, 256} };
    for (int j = 0; j < 5; j++) {
        srand(1);
        for (int i = 0; i < n; i++) c[i] = rand() % v[j].range;
        volatile float sink = 0;
        double best = 1e9;
        for (int r = 0; r < 7; r++) {
            double t0 = now();
            for (int it = 0; it < 400; it++) sink += v[j].f(c, n);
            double dt = (now() - t0) / (400.0 * n);
            if (dt < best) best = dt;
        }
        printf("  %-34s %6.3f ns/code  %6.2f Gcodes/s\n", v[j].name, best * 1e9, 1e-9 / best);
    }
    return 0;
}
